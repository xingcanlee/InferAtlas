"""Resolve immutable upstream pairs and pinned, offline model configurations.

Public entry point for the CLI: ``resolve(model, version, device, out_path)``.
``INFERATLAS_SRC`` contains the read-only ``vllm`` and ``vllm-ascend`` clones;
missing revisions are checked out under ``.cache/src``, never in those clones.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

from ruamel.yaml import YAML

from inferatlas import __version__
from inferatlas.schema import Catalog, Snapshot
from inferatlas.schema.models import CatalogModel, Device, Revision, VersionPair

ROOT = Path(__file__).resolve().parents[3]
_SHA = re.compile(r"[0-9a-f]{40}")
UPSTREAM = {
    "vllm": "https://github.com/vllm-project/vllm.git",
    "vllm-ascend": "https://github.com/vllm-project/vllm-ascend.git",
}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def experiment_dir() -> Path:
    """Worktrees share the main checkout's heavyweight CPU environment."""
    local = ROOT / "experiments/a2_qwen3moe"
    if (local / ".venv/bin/python").exists():
        return local
    common = Path(git(ROOT, "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent / "experiments/a2_qwen3moe"


def source_dir() -> Path:
    return Path(os.environ.get("INFERATLAS_SRC") or experiment_dir() / "src").resolve()


def load_catalog(path: str | Path = ROOT / "catalog.yaml") -> Catalog:
    with Path(path).open() as stream:
        catalog = Catalog.model_validate(YAML(typ="safe").load(stream))
    for name in ("models", "version_pairs", "devices", "hardware_presets"):
        ids = [item.id for item in getattr(catalog, name)]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate {name} id")
        getattr(catalog, name).sort(key=lambda item: item.id)
    return catalog


def get_model(model: str | CatalogModel, catalog: Catalog | None = None) -> CatalogModel:
    if isinstance(model, CatalogModel):
        return model
    for entry in (catalog or load_catalog()).models:
        if model in (entry.id, entry.hf_id):
            return entry
    raise ValueError(f"model is not pinned in catalog: {model}")


def write_json(path: str | Path, value: object) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    return path


def write_config(model: str | CatalogModel, directory: str | Path) -> Path:
    entry = get_model(model)
    digest = hashlib.sha256(
        json.dumps(entry.config, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if entry.config_rev.startswith("inline:") and entry.config_rev != f"inline:sha256:{digest}":
        raise ValueError(f"catalog config hash mismatch: {entry.id}")
    return write_json(Path(directory) / "config.json", entry.config)


def _revision(repo: str, ref: str, src: Path, cache: Path) -> tuple[str, Path]:
    local = src / repo
    if (local / ".git").exists():
        try:
            return git(local, "rev-parse", "--verify", f"{ref}^{{commit}}"), local
        except subprocess.CalledProcessError:
            pass
    remote = UPSTREAM[repo]
    if _SHA.fullmatch(ref):
        commit = ref
    else:
        # Annotated tags resolve to the peeled commit; lightweight tags have one entry.
        names = (
            ["refs/heads/main"] if ref == "main" else [f"refs/tags/{ref}", f"refs/tags/{ref}^{{}}"]
        )
        output = subprocess.run(
            ["git", "ls-remote", "--exit-code", "--", str(remote), *names],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        refs = dict(line.split()[::-1] for line in output.splitlines())
        commit = refs[names[-1]] if names[-1] in refs else refs[names[0]]
    return commit, checkout_commit(remote, commit, cache / repo / commit)


def resolve_version_pair(
    version: str | VersionPair = "main-verified",
    *,
    catalog: Catalog | None = None,
    src: Path | None = None,
    cache: Path | None = None,
) -> tuple[Revision, Revision]:
    """Read the pairing declaration at the selected Ascend revision, not at HEAD."""
    catalog = catalog or load_catalog()
    if isinstance(version, str):
        pair = next((p for p in catalog.version_pairs if p.id == version), None)
        if pair is None:
            raise ValueError(f"unknown version pair: {version}")
    else:
        pair = version
    src = src or source_dir()
    pairing_file = (
        ".github/vllm-main-verified.commit"
        if pair.ascend == "main"
        else ".github/vllm-release-tag.commit"
    )
    if pair.pairing_file != pairing_file:
        raise ValueError(f"unexpected pairing file for {pair.ascend}: {pair.pairing_file}")
    cache = cache or ROOT / ".cache/src"
    ascend_commit, ascend_root = _revision("vllm-ascend", pair.ascend, src, cache)
    vllm_ref = git(ascend_root, "show", f"{ascend_commit}:{pairing_file}")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", vllm_ref):
        raise ValueError(f"invalid upstream pairing declaration: {vllm_ref!r}")
    vllm_commit, _ = _revision("vllm", vllm_ref, src, cache)
    return (
        Revision(tag="main" if pair.ascend == "main" else vllm_ref, commit=vllm_commit),
        Revision(tag=pair.ascend, commit=ascend_commit),
    )


def checkout_commit(remote: str | Path, commit: str, directory: str | Path) -> Path:
    """Fetch just one commit into a dedicated cache; reuse an intact checkout."""
    if not _SHA.fullmatch(commit):
        raise ValueError("checkout requires a full immutable commit SHA")
    directory = Path(directory).resolve()
    if (directory / ".git").exists():
        if git(directory, "status", "--porcelain", "--untracked-files=all"):
            raise ValueError(f"cached checkout has local changes: {directory}")
        try:
            if git(directory, "rev-parse", "HEAD") == commit:
                return directory
        except subprocess.CalledProcessError:
            pass  # An interrupted fetch left an initialized, empty repository.
    directory.mkdir(parents=True, exist_ok=True)
    git(directory, "init", "--quiet")
    git(directory, "fetch", "--depth", "1", "--", str(remote), commit)
    git(directory, "checkout", "--detach", "--quiet", commit)
    return directory


def checkout_sources(snapshot: Snapshot, cache: Path | None = None) -> dict[str, Path]:
    cache = cache or ROOT / ".cache/src"
    sources = {}
    for repo, revision in (("vllm", snapshot.vllm), ("vllm-ascend", snapshot.ascend)):
        local = source_dir() / repo
        if (local / ".git").exists() and git(local, "rev-parse", "HEAD") == revision.commit:
            if git(local, "status", "--porcelain", "--untracked-files=no"):
                raise ValueError(f"upstream checkout has tracked changes: {local}")
            sources[repo] = local
        else:
            sources[repo] = checkout_commit(
                UPSTREAM[repo], revision.commit, cache / repo / revision.commit
            )
    return sources


def resolve(
    model: str,
    version: str = "main-verified",
    device: Device = "A2",
    out_path: str | Path | None = None,
    *,
    catalog_path: str | Path = ROOT / "catalog.yaml",
) -> Snapshot:
    catalog = load_catalog(catalog_path)
    entry = get_model(model, catalog)
    vllm, ascend = resolve_version_pair(version, catalog=catalog)
    snapshot = Snapshot(
        vllm=vllm,
        ascend=ascend,
        model=entry.model_dump(include={"hf_id", "config_rev"}),
        device=device,
        generator_version=__version__,
    )
    directory = Path(out_path).parent if out_path else ROOT / ".cache/config" / entry.id
    write_config(entry, directory)
    if out_path:
        write_json(out_path, snapshot.model_dump(mode="json"))
    return snapshot


__all__ = [
    "checkout_commit",
    "checkout_sources",
    "get_model",
    "load_catalog",
    "resolve",
    "resolve_version_pair",
    "write_config",
]
