"""Exercise publication gates without network access, model weights or hardware."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ACTIONS = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ACTIONS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


pipeline = module("pipeline")
cpu_env = module("cpu_env")
draft = module("draft")

PAIR = {"ascend": "a" * 40, "vllm": "b" * 40}
SNAPSHOT = pipeline.snapshot_id(PAIR)


def catalog():
    return {
        "models": [
            {"id": "moe", "hf_id": "Qwen/Moe", "config_rev": "inline:sha256:x"},
            {"id": "dense", "hf_id": "Qwen/Dense", "config_rev": "inline:sha256:y"},
        ],
        "devices": [{"id": "A3"}, {"id": "A2"}],
    }


def artifact(root, item, status):
    path = root / f"cell-{item['id']}-{item['device']}"
    pipeline.write_json(path / "result.json", {**item, "status": status, "snapshot": SNAPSHOT})
    for name in ("ir", "view", "report", "triage"):
        pipeline.write_json(path / f"{name}.json", {"fixture": name})


def test_catalog_order_identity_and_no_architecture_assumptions():
    data = catalog()
    matrix = pipeline.catalog_matrix(data)
    assert [(item["id"], item["device"]) for item in matrix] == [
        ("dense", "A2"),
        ("dense", "A3"),
        ("moe", "A2"),
        ("moe", "A3"),
    ]
    data["models"].reverse()
    data["devices"].reverse()
    assert pipeline.catalog_matrix(data) == matrix
    assert pipeline.snapshot_id({**PAIR, "vllm": "d" * 40}) != SNAPSHOT
    assert pipeline.pair_key(dict(reversed(list(PAIR.items())))) == pipeline.pair_key(PAIR)


@pytest.mark.parametrize("bad", ["../escape", "name/with/slashes", "bad\nname"])
def test_matrix_rejects_unsafe_artifact_paths(bad):
    data = catalog()
    data["models"][0]["id"] = bad
    with pytest.raises(ValueError):
        pipeline.catalog_matrix(data)


def test_one_triaged_device_blocks_entire_model_and_retains_other_models(tmp_path):
    expected = pipeline.catalog_matrix(catalog())
    for item in expected:
        artifact(
            tmp_path / "cells",
            item,
            "triage" if item["id"] == "moe" and item["device"] == "A3" else "ok",
        )
    count, triage = pipeline.collect_cells(tmp_path / "cells", tmp_path / "bundle", expected)
    assert count == 2 and triage
    assert len(list((tmp_path / "bundle/data").rglob("ir.json"))) == 2
    assert not list((tmp_path / "bundle/data").rglob("moe"))
    assert (tmp_path / "bundle/triage/moe/A3/triage.json").exists()
    summary = json.loads((tmp_path / "bundle/summary.json").read_text())
    assert summary == {"blocked_models": ["moe"], "clean": 2, "errors": []}


def test_missing_cell_and_hard_failure_are_not_treated_as_triage(tmp_path):
    expected = pipeline.catalog_matrix(catalog())
    for item in expected[:-1]:
        artifact(tmp_path / "cells", item, "error" if item["id"] == "dense" else "ok")
    count, triage = pipeline.collect_cells(tmp_path / "cells", tmp_path / "bundle", expected)
    assert count == 0 and not triage
    summary = json.loads((tmp_path / "bundle/summary.json").read_text())
    assert len(summary["errors"]) == 3


def test_fallback_snapshot_and_skipped_coverage_cannot_pass():
    item = pipeline.catalog_matrix(catalog())[0]
    snapshot = {
        "ascend": {"commit": PAIR["ascend"]},
        "vllm": {"commit": PAIR["vllm"]},
        "device": item["device"],
        "model": {"hf_id": item["model"], "config_rev": item["config_rev"]},
    }
    pipeline.check_identity(snapshot, item, PAIR)
    snapshot["vllm"]["commit"] = "f" * 40
    with pytest.raises(ValueError, match="refusing fallback"):
        pipeline.check_identity(snapshot, item, PAIR)
    report = {
        "checks": [
            {"id": name, "status": "pass"} for name in ("schema", "coverage", "source_links")
        ],
        "golden_diff": [],
        "source_link_failures": [],
    }
    assert pipeline.report_passes(report)
    report["checks"][1]["status"] = "skip"
    assert not pipeline.report_passes(report)
    report["checks"][1]["status"] = "pass"
    report["checks"].insert(0, {"id": "source_links", "status": "fail"})
    assert not pipeline.report_passes(report)


def test_cpu_constraints_exclude_npu_deps_and_pin_all_torch_wheels():
    pins, optional = cpu_env.requirements(
        "# comment\ntorch==2.10.0\ntorchvision==0.25.0\ntorchaudio==2.10.0\n"
        "torch_npu==2.10.0\ntriton-ascend==3.2.2\nmemfabric_hybrid==1.0\n"
        "numpy>=2\ntransformers==5.0.0\n"
    )
    assert pins == {"torch": "2.10.0", "torchvision": "0.25.0", "torchaudio": "2.10.0"}
    assert optional == ["numpy>=2", "transformers==5.0.0"]
    with pytest.raises(ValueError, match="exact release pin"):
        cpu_env.requirements("torch>=2")


@pytest.mark.parametrize(
    "path",
    [
        "generator/inferatlas/cli.py",
        ".github/workflows/ci.yml",
        "golden/moe.yaml",
        "rules/../catalog.yaml",
        "/rules/fake.yaml",
        "rules/fake.py",
        "rules/README.md",
    ],
)
def test_agent_boundary_rejects_non_rule_changes(path):
    assert not draft.allowed(path)


def test_agent_boundary_accepts_only_rule_and_annotation_yaml():
    assert draft.allowed("rules/v1/linear.yaml")
    assert draft.allowed("annotations/moe.yaml")


def test_published_release_is_verified_without_clobber(tmp_path, monkeypatch):
    singles = tmp_path / "singles"
    singles.mkdir()
    (singles / "moe--A2.html").write_text("<!doctype html>snapshot")
    monkeypatch.setattr(pipeline, "CACHE", tmp_path)
    calls = []

    def gh(*args, **kwargs):
        calls.append(args)
        if args[:2] == ("release", "view"):
            return subprocess.CompletedProcess(args, 0, '{"isDraft":false}', "")
        assert args[:2] == ("release", "download")
        target = Path(args[args.index("--dir") + 1])
        for path in singles.iterdir():
            (target / path.name).write_bytes(path.read_bytes())
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(pipeline, "gh", gh)
    pipeline.release(SNAPSHOT, singles)
    assert len(calls) == 2
    assert all("upload" not in call and "edit" not in call for call in calls)


@pytest.mark.parametrize("golden_status", [0, 1])
def test_real_contract_cell_turns_golden_drift_into_actionable_triage(
    tmp_path, monkeypatch, golden_status
):
    import shutil

    fixture = pipeline.ROOT / "fixtures/qwen3-30b-a3b"
    triage = json.loads((fixture / "triage.json").read_text())
    snapshot = triage["snapshot"]
    current = {
        "id": "qwen3-30b-a3b",
        "model": snapshot["model"]["hf_id"],
        "config_rev": snapshot["model"]["config_rev"],
        "device": snapshot["device"],
    }
    monkeypatch.setattr(pipeline, "CACHE", tmp_path)
    monkeypatch.setenv("CELL", json.dumps(current))
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    for name in ("vllm", "ascend"):
        monkeypatch.setenv(f"INFERATLAS_{name.upper()}_COMMIT", snapshot[name]["commit"])

    def run(arguments, log):
        dest = tmp_path / "cell"
        log.write_text("golden drift\n" if arguments[0] == "golden" else "ok\n")
        if arguments[0] == "build":
            for name in ("ir", "view", "report"):
                shutil.copyfile(fixture / f"{name}.json", dest / f"{name}.json")
            pipeline.write_json(dest / "triage.json", {**triage, "items": []})
        elif arguments[0] == "verify":
            report = json.loads((dest / "report.json").read_text())
            report["checks"].extend(
                {"id": name, "status": "pass", "message": "Checked"}
                for name in ("coverage", "source_links")
            )
            pipeline.write_json(dest / "report.json", report)
        return golden_status if arguments[0] == "golden" else 0

    monkeypatch.setattr(pipeline, "run_cli", run)
    pipeline.cell()
    result = json.loads((tmp_path / "cell/result.json").read_text())
    assert result["status"] == ("triage" if golden_status else "ok")
    items = json.loads((tmp_path / "cell/triage.json").read_text())["items"]
    assert len(items) == golden_status
    if items:
        assert items[0] == {
            "kind": "golden_diff",
            "model": current["model"],
            "diff": "golden drift",
        }


@pytest.mark.parametrize("bootstrap_failure", [False, True])
def test_state_roundtrip_preserves_history_and_rejects_snapshot_overwrites(
    tmp_path, monkeypatch, bootstrap_failure
):
    remote = tmp_path / "remote.git"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    monkeypatch.setenv("GITHUB_REPOSITORY", "test/atlas")
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    monkeypatch.setenv("INFERATLAS_ASCEND_COMMIT", PAIR["ascend"])
    monkeypatch.setenv("INFERATLAS_VLLM_COMMIT", PAIR["vllm"])
    monkeypatch.setenv("SNAPSHOT_ID", SNAPSHOT)
    original_git = pipeline.git

    def local_git(*arguments, **kwargs):
        if arguments[:2] == ("remote", "add"):
            arguments = (*arguments[:3], str(remote))
        return original_git(*arguments, **kwargs)

    monkeypatch.setattr(pipeline, "git", local_git)
    if bootstrap_failure:
        cache = tmp_path / "bootstrap-failure"
        monkeypatch.setattr(pipeline, "CACHE", cache)
        pipeline.write_json(cache / "bundle/data/partial/ir.json", {"partial": True})
        pipeline.record_failure()
        state = json.loads((cache / "state/state.json").read_text())
        assert state == {"pairs": {}, "failures": {pipeline.pair_key(PAIR): {"c" * 40: True}}}
        assert not (cache / "state/data").exists()
    for iteration in ("old", "new"):
        cache = tmp_path / iteration
        monkeypatch.setattr(pipeline, "CACHE", cache)
        pipeline.write_json(cache / f"bundle/data/{iteration}/ir.json", {"version": iteration})
        pipeline.write_json(
            cache / "bundle/summary.json",
            {
                "errors": [],
                "clean": 1,
                "blocked_models": [],
            },
        )
        pipeline.restore()
        pipeline.publish()
    retained = tmp_path / "new/state/data"
    assert json.loads((retained / "old/ir.json").read_text()) == {"version": "old"}
    assert json.loads((retained / "new/ir.json").read_text()) == {"version": "new"}
    state = json.loads((tmp_path / "new/state/state.json").read_text())
    assert state["pairs"][pipeline.pair_key(PAIR)]["snapshot"] == SNAPSHOT
    # A hard failure must retain successes and historical JSON without publishing data.
    cache = tmp_path / "failed"
    monkeypatch.setattr(pipeline, "CACHE", cache)
    pipeline.record_failure()
    state = json.loads((cache / "state/state.json").read_text())
    assert state["failures"] == {pipeline.pair_key(PAIR): {"c" * 40: True}}
    assert state["pairs"][pipeline.pair_key(PAIR)]["snapshot"] == SNAPSHOT
    assert (cache / "state/data/old/ir.json").exists()
    # A different generator revision gets its own failure record.
    cache = tmp_path / "failed-again"
    monkeypatch.setattr(pipeline, "CACHE", cache)
    monkeypatch.setenv("GITHUB_SHA", "d" * 40)
    pipeline.record_failure()
    state = json.loads((cache / "state/state.json").read_text())
    assert state["failures"][pipeline.pair_key(PAIR)] == {"c" * 40: True, "d" * 40: True}
    cache = tmp_path / "conflict"
    monkeypatch.setattr(pipeline, "CACHE", cache)
    pipeline.write_json(cache / "bundle/data/old/ir.json", {"overwritten": True})
    with pytest.raises(ValueError, match="Immutable snapshot changed"):
        pipeline.restore()


def test_snapshot_hashes_relevant_git_trees_and_ignores_docs(tmp_path, monkeypatch):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.com")
    inputs = (
        "generator/code.py",
        "rules/rule.yaml",
        "annotations/note.yaml",
        "catalog.yaml",
        "packages/layout/src/index.ts",
        "web/src/App.tsx",
        "schemas/ir.schema.json",
        "pnpm-lock.yaml",
        "generator/uv.lock",
        ".github/actions/pipeline.py",
    )
    for path in inputs:
        dest = tmp_path / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("original\n")
    git("add", ".")
    git("commit", "-m", "initial")
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    initial = pipeline.snapshot_id(PAIR)
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs/ci.md").write_text("docs only\n")
    git("add", ".")
    git("commit", "-m", "docs")
    assert pipeline.snapshot_id(PAIR) == initial
    for path in inputs:
        before = pipeline.snapshot_id(PAIR)
        (tmp_path / path).write_text("changed\n")
        git("add", ".")
        git("commit", "-m", f"change {path}")
        assert pipeline.snapshot_id(PAIR) != before
    before = pipeline.snapshot_id(PAIR)
    git("rm", "annotations/note.yaml")
    git("commit", "-m", "remove annotation tree")
    assert pipeline.snapshot_id(PAIR) != before


def test_plan_expands_the_real_catalog_without_running_the_generator(tmp_path, monkeypatch):
    destination = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(destination))
    monkeypatch.setenv("GITHUB_SHA", "c" * 40)
    monkeypatch.setenv("INFERATLAS_ASCEND_COMMIT", PAIR["ascend"])
    monkeypatch.setenv("INFERATLAS_VLLM_COMMIT", PAIR["vllm"])
    pipeline.plan()
    values = dict(line.split("=", 1) for line in destination.read_text().splitlines())
    assert values["snapshot"] == SNAPSHOT
    cells = json.loads(values["matrix"])["include"]
    assert cells and len(cells) == len({item["model"] for item in cells}) * len(
        {item["device"] for item in cells}
    )


def test_triage_preparation_requires_one_schema_valid_pair(tmp_path, monkeypatch):
    import shutil

    monkeypatch.chdir(tmp_path)
    (tmp_path / "schemas").mkdir()
    shutil.copyfile(pipeline.ROOT / "schemas/triage.schema.json", "schemas/triage.schema.json")
    triage = json.loads((pipeline.ROOT / "fixtures/qwen3-30b-a3b/triage.json").read_text())
    pipeline.write_json(Path(".cache/incoming/triage/moe/A2/triage.json"), triage)
    destination = tmp_path / "outputs"
    monkeypatch.setenv("GITHUB_OUTPUT", str(destination))
    draft.prepare()
    values = dict(line.split("=", 1) for line in destination.read_text().splitlines())
    assert values == {name: triage["snapshot"][name]["commit"] for name in ("vllm", "ascend")}
    triage["snapshot"]["ascend"]["commit"] = "f" * 40
    pipeline.write_json(Path(".cache/incoming/triage/moe/A3/triage.json"), triage)
    with pytest.raises(ValueError, match="exactly one upstream pair"):
        draft.prepare()


def test_agent_scope_guard_rejects_symlinks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "rules").mkdir()
    (tmp_path / "rules/escape.yaml").symlink_to(tmp_path / "outside.yaml")
    monkeypatch.setattr(draft, "changed_paths", lambda: ["rules/escape.yaml"])
    with pytest.raises(ValueError, match="forbidden path"):
        draft.check()


def test_only_unpublished_drafts_can_replace_release_assets(tmp_path, monkeypatch):
    singles = tmp_path / "singles"
    singles.mkdir()
    (singles / "moe--A2.html").write_text("<!doctype html>snapshot")
    calls = []

    def gh(*args, **kwargs):
        calls.append(args)
        return subprocess.CompletedProcess(args, 0, '{"isDraft":true}', "")

    monkeypatch.setattr(pipeline, "gh", gh)
    pipeline.release(SNAPSHOT, singles)
    assert calls[1][:3] == ("release", "upload", SNAPSHOT)
    assert "--clobber" in calls[1]
    assert calls[2] == ("release", "edit", SNAPSHOT, "--draft=false", "--latest=false")
