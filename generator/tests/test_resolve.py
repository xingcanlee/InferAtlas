"""Offline resolver checks using small local Git repositories."""

import hashlib
import json
import subprocess

import pytest

import inferatlas.resolve as resolver
from inferatlas.resolve import (
    checkout_commit,
    get_model,
    git,
    load_catalog,
    resolve_version_pair,
    write_config,
)
from inferatlas.schema.models import VersionPair


def init_repo(path):
    path.mkdir(parents=True)
    git(path, "init", "-b", "main")
    git(path, "config", "user.email", "test@example.invalid")
    git(path, "config", "user.name", "Resolver test")
    return path


def commit(repo, content):
    for name, text in content.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    git(repo, "add", ".")
    git(repo, "commit", "-m", "test revision")
    return git(repo, "rev-parse", "HEAD")


def test_catalog_is_valid_sorted_and_pinned(tmp_path):
    catalog = load_catalog()
    assert [m.id for m in catalog.models] == sorted(m.id for m in catalog.models)
    for model in catalog.models:
        path = write_config(model, tmp_path / model.id)
        assert json.loads(path.read_text()) == model.config
        before = path.read_bytes()
        write_config(model, path.parent)
        assert path.read_bytes() == before
        digest = hashlib.sha256(
            json.dumps(model.config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert model.config_rev == "inline:sha256:" + digest
        assert get_model(model.hf_id) == model
    with pytest.raises(ValueError, match="not pinned"):
        get_model("not/a-catalog-model")


def test_resolve_writes_snapshot_and_offline_config(tmp_path, monkeypatch):
    from inferatlas.schema.models import Revision

    revisions = (Revision(tag="main", commit="a" * 40), Revision(tag="main", commit="b" * 40))
    monkeypatch.setattr(resolver, "resolve_version_pair", lambda *args, **kwargs: revisions)
    output = tmp_path / "snapshot.json"
    snapshot = resolver.resolve("qwen3-8b", out_path=output)
    assert json.loads(output.read_text()) == snapshot.model_dump(mode="json")
    assert json.loads((tmp_path / "config.json").read_text()) == get_model("qwen3-8b").config
    assert snapshot.model.hf_id == "Qwen/Qwen3-8B"
    before = output.read_bytes()
    resolver.resolve("Qwen/Qwen3-8B", out_path=output)
    assert output.read_bytes() == before


def test_config_rejects_stale_hash_before_writing(tmp_path):
    model = get_model("qwen3-8b").model_copy(deep=True)
    model.config["hidden_size"] = 1
    with pytest.raises(ValueError, match="hash mismatch"):
        write_config(model, tmp_path)
    assert not (tmp_path / "config.json").exists()


def test_pairing_reads_selected_revision_and_not_working_tree(tmp_path):
    vllm = init_repo(tmp_path / "vllm")
    release = commit(vllm, {"model.py": "release"})
    git(vllm, "tag", "v1.0")
    main = commit(vllm, {"model.py": "main"})
    ascend = init_repo(tmp_path / "vllm-ascend")
    release_ascend = commit(ascend, {".github/vllm-release-tag.commit": "v1.0\n"})
    git(ascend, "tag", "v1.0")
    main_ascend = commit(
        ascend,
        {
            ".github/vllm-main-verified.commit": main + "\n",
            ".github/vllm-release-tag.commit": "nonexistent-release\n",
        },
    )
    for ref, filename, expected in [
        ("v1.0", "release-tag", (release, release_ascend)),
        ("main", "main-verified", (main, main_ascend)),
    ]:
        pair = VersionPair(
            id=ref,
            ascend=ref,
            vllm="ignored-stale-value",
            pairing_file=f".github/vllm-{filename}.commit",
        )
        actual = resolve_version_pair(pair, src=tmp_path)
        assert tuple(revision.commit for revision in actual) == expected


def test_dispatch_pins_do_not_resolve_moving_refs(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERATLAS_VLLM_COMMIT", "a" * 40)
    monkeypatch.setenv("INFERATLAS_ASCEND_COMMIT", "b" * 40)
    monkeypatch.setattr(resolver, "_revision", lambda *args: pytest.fail("must not resolve main"))
    actual = resolve_version_pair(src=tmp_path)
    assert [(r.tag, r.commit) for r in actual] == [("main", "a" * 40), ("main", "b" * 40)]


@pytest.mark.parametrize("vllm,ascend", [("a" * 40, None), (None, "b" * 40), ("main", "b" * 40)])
def test_dispatch_rejects_partial_or_mutable_pins(monkeypatch, vllm, ascend):
    for name, pin in [("VLLM", vllm), ("ASCEND", ascend)]:
        monkeypatch.delenv(f"INFERATLAS_{name}_COMMIT", raising=False)
        if pin is not None:
            monkeypatch.setenv(f"INFERATLAS_{name}_COMMIT", pin)
    with pytest.raises(ValueError, match="both full lowercase"):
        resolve_version_pair()


def test_checkout_fetches_one_commit_then_works_without_remote(tmp_path):
    remote = init_repo(tmp_path / "remote")
    commit(remote, {"first": "old"})
    sha = commit(remote, {"model.py": "pinned"})
    target = tmp_path / "cache" / sha
    assert checkout_commit(remote, sha, target) == target
    assert git(target, "rev-parse", "HEAD") == sha
    assert git(target, "rev-list", "--count", "HEAD") == "1"
    assert checkout_commit(tmp_path / "missing-remote", sha, target) == target
    (target / "model.py").write_text("local edit")
    with pytest.raises(ValueError, match="local changes"):
        checkout_commit(remote, sha, target)
    assert (target / "model.py").read_text() == "local edit"


def test_checkout_rejects_non_sha_and_recovers_after_failed_fetch(tmp_path):
    with pytest.raises(ValueError, match="full immutable"):
        checkout_commit("unused", "main", tmp_path / "bad")
    target = tmp_path / "cache"
    with pytest.raises(subprocess.CalledProcessError):
        checkout_commit(tmp_path / "missing", "a" * 40, target)
    remote = init_repo(tmp_path / "remote")
    sha = commit(remote, {"file": "recovered"})
    checkout_commit(remote, sha, target)
    assert (target / "file").read_text() == "recovered"


def test_missing_release_tags_resolve_in_cache_without_mutating_clones(tmp_path, monkeypatch):
    vllm = init_repo(tmp_path / "remotes/vllm")
    sha = commit(vllm, {"file": "release"})
    git(vllm, "tag", "-a", "v1.0", "-m", "annotated release")
    ascend = init_repo(tmp_path / "remotes/vllm-ascend")
    ascend_sha = commit(ascend, {".github/vllm-release-tag.commit": "v1.0\n"})
    git(ascend, "tag", "v1.0")
    monkeypatch.setattr(resolver, "UPSTREAM", {"vllm": str(vllm), "vllm-ascend": str(ascend)})
    pair = VersionPair(
        id="release", ascend="v1.0", vllm="v1.0", pairing_file=".github/vllm-release-tag.commit"
    )
    actual = resolve_version_pair(pair, src=tmp_path / "no-clones", cache=tmp_path / "cache")
    assert tuple(revision.commit for revision in actual) == (sha, ascend_sha)
    assert git(tmp_path / "cache/vllm" / sha, "rev-parse", "HEAD") == sha
    assert not (tmp_path / "no-clones").exists()
