"""CLI adapters use the owning stages' public signatures and fail closed."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

import inferatlas.annotations
import inferatlas.cli as cli
import inferatlas.extract
import inferatlas.lower
import inferatlas.resolve
import inferatlas.verify as verification
from inferatlas.schema import IR, Extract, Index, Report, Snapshot, Triage
from inferatlas.schema.models import Check, StaleAnnotation
from inferatlas.verify.golden import Evaluator
from inferatlas.verify.io import ROOT, read_json, write_json

FIXTURE = ROOT / "fixtures/qwen3-30b-a3b"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setenv("INFERATLAS_ALLOW_FIXTURES", "1")
    monkeypatch.setattr(cli, "source_root", lambda root: tmp_path / "sources")
    monkeypatch.setattr(verification, "check_sources", lambda *args: [])


def invoke(*args, code=0):
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    assert result.exit_code == code, result.output or repr(result.exception)
    return result


def test_command_help_and_version():
    for name in (
        "resolve",
        "extract",
        "lower",
        "annotate",
        "verify",
        "layout",
        "build",
        "site",
        "single",
        "golden",
    ):
        assert name in invoke("--help").output
    assert invoke("version").output.strip() == "0.1.0"


def test_no_implicit_fixture_fallback(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES")
    output = tmp_path / "snapshot.json"
    assert "--allow-fixtures" in invoke("resolve", "--output", output, code=3).output
    assert not output.exists()
    invoke("resolve", "--allow-fixtures", "--output", output)
    assert read_json(output.with_suffix(".provenance.json")) == {"fixtures": ["resolve"]}


def test_standalone_provenance_survives_commands(tmp_path):
    snapshot, extract, graph, issues, report = [
        tmp_path / name
        for name in ("snapshot.json", "extract.json", "ir.json", "triage.json", "report.json")
    ]
    invoke("resolve", "--output", snapshot)
    invoke("extract", "--snapshot", snapshot, "--output", extract)
    invoke("lower", "--extract", extract, "--output", graph, "--triage-output", issues, code=2)
    invoke(
        "annotate",
        "--ir",
        graph,
        "--triage",
        issues,
        "--output",
        graph,
        "--triage-output",
        issues,
        code=2,
    )
    invoke("verify", "--ir", graph, "--triage", issues, "--output", report, code=3)
    provenance = next(c for c in read_json(report)["checks"] if c["id"] == "provenance")
    assert provenance["status"] == "fail"
    assert all(
        stage in provenance["message"] for stage in ("resolve", "extract", "lower", "annotations")
    )


def test_build_complete_deterministic_and_removes_stale_outputs(tmp_path, monkeypatch):
    invoke("build", "--output", tmp_path, code=3)
    expected = {
        "snapshot.json",
        "extract.json",
        "extract-tp4.json",
        "ir.json",
        "triage.json",
        "report.json",
        "view.json",
        "provenance.json",
    }
    assert {p.name for p in tmp_path.iterdir()} == expected
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    invoke("build", "--output", tmp_path, "--tp", 4, "--tp", 1, code=3)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    invoke("build", "--output", tmp_path, "--tp", 1, code=3)
    assert not (tmp_path / "extract-tp4.json").exists()

    def fail(*args):
        raise ValueError("layout unavailable")

    monkeypatch.setattr(cli, "layout_stage", fail)
    invoke("build", "--output", tmp_path, code=3)
    assert not (tmp_path / "view.json").exists()
    assert any(
        c["id"] == "layout" and c["status"] == "fail"
        for c in read_json(tmp_path / "report.json")["checks"]
    )


def test_failed_early_build_preserves_previous_generation(tmp_path):
    invoke("build", "--output", tmp_path, code=3)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    invoke("build", "--output", tmp_path, "--tp", 8, code=3)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_resolve_and_extract_exact_apis(monkeypatch):
    snapshot = Snapshot.model_validate(read_json(FIXTURE / "snapshot.json"))
    calls = []

    def resolve(model, version="main-verified", device="A2", out_path=None, *, catalog_path):
        calls.append((model, version, device, out_path, catalog_path))
        return snapshot

    def extract(value, model, tp=1, ep=False, device=None, out_path=None):
        calls.append((value, model, tp, ep, device, out_path))
        result = Extract.model_validate(read_json(FIXTURE / "extract-tp4.json"))
        result.parallel.ep = ep
        return result

    monkeypatch.setattr(inferatlas.resolve, "resolve", resolve, raising=False)
    monkeypatch.setattr(inferatlas.extract, "extract", extract, raising=False)
    assert cli.resolve_stage(cli.DEFAULT_MODEL, "A2", "main-verified") == snapshot
    assert cli.extract_stage(snapshot, 4, True).parallel.ep
    assert calls[0] == (cli.DEFAULT_MODEL, "main-verified", "A2", None, ROOT / "catalog.yaml")
    assert calls[1] == (snapshot, cli.DEFAULT_MODEL, 4, True, "A2", None)

    def broken(*args, **kwargs):
        raise ValueError("real extraction failed")

    monkeypatch.setattr(inferatlas.extract, "extract", broken)
    with pytest.raises(ValueError, match="real extraction failed"):
        cli.extract_stage(snapshot, 4)


def test_lower_exact_single_extract_api(monkeypatch):
    extracts = [
        Extract.model_validate(read_json(FIXTURE / name))
        for name in ("extract-tp4.json", "extract.json")
    ]
    calls = []

    @dataclass
    class LowerResult:
        ir: IR
        triage: Triage
        parameter_shapes: dict

    def lower(extract, rules=ROOT / "rules", *, vllm_version, ascend_version, mro):
        calls.append((extract.parallel.tp, rules, vllm_version, ascend_version, mro))
        return LowerResult(
            IR.model_validate(read_json(FIXTURE / "ir.json")),
            Triage.model_validate(read_json(FIXTURE / "triage.json")),
            {},
        )

    monkeypatch.setattr(inferatlas.lower, "lower", lower, raising=False)
    ir, triage = cli.lower_stage(extracts)
    assert ir.snapshot == triage.snapshot
    assert calls == [(tp, ROOT / "rules", None, None, None) for tp in (1, 4)]
    with pytest.raises(ValueError, match="tp=1"):
        cli.lower_stage(extracts[:1])


def test_annotations_exact_apis_bind_observed_swaps_and_keep_stale(monkeypatch):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    extracts = [Extract.model_validate(read_json(FIXTURE / "extract.json"))]
    node = next(node for node in ir.nodes if node.id == "attn.qkv")
    fact = next(m for m in extracts[0].modules if m.class_name == node.class_name)
    replacement = fact.ascend.class_name
    ann = SimpleNamespace(
        id="valid", target=replacement, resolves_to=replacement, applies_when="tp == 1 && is_a2"
    )
    # anchors live on the loaded object and are passed intact to T5.
    loaded = SimpleNamespace(anchors=["helper"])
    stale = StaleAnnotation(
        kind="stale_annotation",
        id="stale",
        anchor_hash_old="sha256:" + "a" * 64,
        anchor_hash_new="sha256:" + "b" * 64,
    )
    calls = []

    def load_annotations(path):
        assert path == ROOT / "annotations"
        return [loaded]

    def check_annotations(anns, sources):
        assert anns == [loaded]
        assert set(sources) == {"vllm", "vllm_ascend"}
        assert sources["vllm"].commit == ir.snapshot.vllm.commit
        return [SimpleNamespace(annotation=ann, src=[])], [stale]

    def apply_annotations(graph, anns, *, bindings, active_ids):
        calls.append((bindings, active_ids))
        return graph

    module = inferatlas.annotations
    for name, value in {
        "load_annotations": load_annotations,
        "check_annotations": check_annotations,
        "apply_annotations": apply_annotations,
        "SourceTree": lambda root, repo, commit: SimpleNamespace(
            root=root, repo=repo, commit=commit
        ),
    }.items():
        monkeypatch.setattr(module, name, value, raising=False)
    _, triage = cli.annotate_stage(ir, extracts, Triage(snapshot=ir.snapshot, items=[]))
    assert calls == [({"attn.qkv": "valid"}, {"valid"})]
    assert triage.items == [stale]
    ann.applies_when = "unknown_runtime_flag"
    _, triage = cli.annotate_stage(ir, extracts, Triage(snapshot=ir.snapshot, items=[]))
    assert calls[-1] == ({}, set())
    assert any(item.kind == "conditional_override" for item in triage.items)


def test_layout_real_command_and_build_if_missing(tmp_path, monkeypatch):
    executable = tmp_path / "packages/layout/dist/cli.js"
    source = tmp_path / "packages/layout/src/cli.ts"
    source.parent.mkdir(parents=True)
    source.touch()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    graph, output = tmp_path / "ir.json", tmp_path / "view.json"
    write_json(graph, read_json(FIXTURE / "ir.json"))
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        if command[0] == "pnpm":
            executable.parent.mkdir(parents=True)
            executable.touch()
        else:
            write_json(output, read_json(FIXTURE / "view.json"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("layout", "--ir", graph, "--output", output)
    assert commands == [
        ["pnpm", "--filter", "@inferatlas/layout", "build"],
        ["node", str(executable), str(graph), "-o", str(output)],
    ]


def site_build(tmp_path, status):
    directory = tmp_path / "build/pinned/qwen3-30b-a3b"
    for name in ("ir", "view", "triage"):
        write_json(directory / f"{name}.json", read_json(FIXTURE / f"{name}.json"))
    ir = IR.model_validate(read_json(directory / "ir.json"))
    checks = [Check(id="provenance", status="pass", message="real")]
    if status == 3:
        checks.append(Check(id="golden", status="fail", message="drift"))
    if status == 0:
        write_json(directory / "triage.json", Triage(snapshot=ir.snapshot, items=[]))
    write_json(
        directory / "report.json",
        Report(snapshot=ir.snapshot, checks=checks, golden_diff=[], source_link_failures=[]),
    )
    return directory


@pytest.mark.parametrize(
    "status,include,expected", [(0, False, 0), (2, False, 2), (2, True, 2), (3, True, 3)]
)
def test_site_gates_before_web_or_data_write(tmp_path, monkeypatch, status, include, expected):
    site_build(tmp_path, status)
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    args = ["site", "--build-root", tmp_path / "build", "--data-dir", tmp_path / "data"]
    invoke(*args, *(["--include-blocked"] if include else []), code=expected)
    allowed = status == 0 or (status == 2 and include)
    assert bool(calls) == allowed
    assert (tmp_path / "data/index.json").exists() == allowed
    if allowed:
        index = Index.model_validate(read_json(tmp_path / "data/index.json"))
        assert any(c.status == ("triage" if status == 2 else "ok") for c in index.cells)


def test_site_always_refuses_fixture_provenance(tmp_path, monkeypatch):
    directory = site_build(tmp_path, 0)
    report = read_json(directory / "report.json")
    report["checks"][0]["status"] = "fail"
    write_json(directory / "report.json", report)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("web ran"))
    invoke(
        "site",
        "--build-root",
        tmp_path / "build",
        "--data-dir",
        tmp_path / "data",
        "--include-blocked",
        code=3,
    )
    assert not (tmp_path / "data").exists()


def test_single_blocked_gate_and_handoff(tmp_path, monkeypatch):
    directory, output = tmp_path / "model-build", tmp_path / "atlas.html"
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 2))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        Path(kwargs["env"]["INFERATLAS_SINGLE_OUTPUT"]).write_text("<!doctype html><p>Atlas</p>")

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("single", "--output", output, code=2)
    assert not calls and not output.exists()
    invoke("single", "--output", output, "--include-blocked", code=2)
    assert output.read_text().startswith("<!doctype html>")
    assert calls == [["pnpm", "--filter", "web", "build:single"]]
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 3))
    invoke("single", "--output", output, "--include-blocked", code=3)
    assert len(calls) == 1


def test_golden_missing_runtime_is_skipped_and_update_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES")
    monkeypatch.setattr(cli, "Evaluator", lambda *args: Evaluator(tmp_path, cli.warning))
    args = ["golden", "check", "--ir", FIXTURE / "ir.json"]
    assert "SKIPPED" in invoke(*args, code=3).output
    assert "SKIPPED" in invoke(*args, "--allow-fixtures").output
    assert (
        "refuses fixture"
        in invoke(
            "golden",
            "update",
            "--ir",
            FIXTURE / "ir.json",
            "--directory",
            tmp_path / "golden",
            code=3,
        ).output
    )
    assert not (tmp_path / "golden").exists()


def test_verify_calibration_env_command(tmp_path):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    evaluator = Evaluator()
    env = evaluator.bind(ir, {"tp": 4, "gpu_memory_utilization": 0.8})
    weights, tokens = evaluator.many([ir.memory.weights, ir.memory.kv_tokens], env)
    write_json(tmp_path / "calibration.json", {"weight_bytes": weights, "kv_tokens": tokens})
    write_json(
        tmp_path / "env.json", {"tp": 4, "gpu_memory_utilization": 0.8, "max_model_len": 4096}
    )
    invoke(
        "verify",
        "--ir",
        FIXTURE / "ir.json",
        "--calibration",
        tmp_path / "calibration.json",
        "--calibration-env",
        tmp_path / "env.json",
        "--output",
        tmp_path / "report.json",
        code=3,
    )
    checks = read_json(tmp_path / "report.json")["checks"]
    assert all(c["status"] == "pass" for c in checks if c["id"].startswith("calibration."))


def test_build_provenance_survives_standalone_reverify(tmp_path):
    invoke("build", "--output", tmp_path, code=3)
    invoke(
        "verify", "--ir", tmp_path / "ir.json", "--output", tmp_path / "fresh-report.json", code=3
    )
    assert (
        next(
            c
            for c in read_json(tmp_path / "fresh-report.json")["checks"]
            if c["id"] == "provenance"
        )["status"]
        == "fail"
    )


@pytest.mark.parametrize(
    "arguments",
    [
        ["extract", "--snapshot", FIXTURE / "snapshot.json"],
        ["lower", "--extract", FIXTURE / "extract.json"],
        ["annotate", "--ir", FIXTURE / "ir.json", "--triage", FIXTURE / "triage.json"],
        ["layout", "--ir", FIXTURE / "ir.json"],
    ],
)
def test_all_missing_stages_require_opt_in(arguments, tmp_path, monkeypatch):
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES")
    monkeypatch.chdir(tmp_path)
    assert "--allow-fixtures" in invoke(*arguments, code=3).output
    assert not list(tmp_path.glob("*.json"))


def test_site_failed_cell_takes_precedence_over_blocked_cell(tmp_path, monkeypatch):
    import shutil

    directory = site_build(tmp_path, 2)
    failed = tmp_path / "build/z-failed/qwen3-30b-a3b"
    shutil.copytree(directory, failed)
    report = read_json(failed / "report.json")
    report["checks"].append({"id": "golden", "status": "fail", "message": "drift", "node": None})
    write_json(failed / "report.json", report)
    # Distinct snapshot/device cell, so the failure cannot be hidden by the first triage cell.
    for name in ("ir", "report", "triage"):
        value = read_json(failed / f"{name}.json")
        value["snapshot"]["device"] = "A3"
        write_json(failed / f"{name}.json", value)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("web ran"))
    invoke("site", "--build-root", tmp_path / "build", "--data-dir", tmp_path / "data", code=3)
    assert not (tmp_path / "data").exists()
