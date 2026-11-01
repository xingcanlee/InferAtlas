"""CLI adapters use the owning stages' public signatures and fail closed."""

import subprocess
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
from t6_runtime import node_backend as node_backend
from t6_runtime import runtime as runtime
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
    for module, name in (
        (inferatlas.resolve, "resolve"),
        (inferatlas.extract, "extract"),
        (inferatlas.lower, "lower"),
        (inferatlas.annotations, "load_annotations"),
    ):
        monkeypatch.setattr(module, name, None)

    is_file, is_dir = Path.is_file, Path.is_dir
    monkeypatch.setattr(
        Path,
        "is_file",
        lambda path: path != ROOT / "packages/layout/dist/cli.js" and is_file(path),
    )
    monkeypatch.setattr(
        Path,
        "is_dir",
        lambda path: path != ROOT / "packages/layout/src" and is_dir(path),
    )
    missing = Evaluator(tmp_path, lambda message: None)
    monkeypatch.setattr(missing, "backend", "fixture")
    monkeypatch.setattr(cli, "Evaluator", lambda *args: missing)


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
        "hub",
        "release",
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


def test_missing_fixture_preserves_previous_numeric_generation(tmp_path):
    invoke("build", "--output", tmp_path, code=3)
    previous = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert "no fixture for" in invoke("build", "--output", tmp_path, "--tp", 8, code=3).output
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == previous


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


def test_annotations_exact_apis_bind_observed_swaps_and_keep_stale(monkeypatch, runtime):
    monkeypatch.setattr(cli, "Evaluator", lambda *args: runtime)
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
    assert stale in triage.items
    assert any(getattr(item, "reason", None) == "unannotated_override" for item in triage.items)
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
        if status == 2:
            assert all(c.data_path is None for c in index.cells)
            assert not list((tmp_path / "data").rglob("ir.json"))


@pytest.mark.parametrize("empty", [False, True])
def test_site_triage_only_cell_never_publishes_numeric_artifacts(tmp_path, monkeypatch, empty):
    directory = site_build(tmp_path, 0 if empty else 2)
    for name in ("ir", "view", "report"):
        (directory / f"{name}.json").unlink()
    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: calls.append(command))
    result = invoke(
        "site",
        "--build-root",
        tmp_path / "build",
        "--data-dir",
        tmp_path / "data",
        "--include-blocked",
        code=3 if empty else 2,
    )
    assert bool(calls) == (not empty)
    if empty:
        assert "must have blocking items" in result.output
    else:
        index = Index.model_validate(read_json(tmp_path / "data/index.json"))
        cell = next(c for c in index.cells if c.status == "triage")
        assert cell.data_path is None
        assert cell.triage_count == len(read_json(directory / "triage.json")["items"])
        assert sorted(p.name for p in (tmp_path / "data").iterdir()) == [
            "hardware.json",
            "index.json",
        ]


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
    for name in ("ir", "view", "report", "triage"):
        write_json(directory / f"{name}.json", read_json(FIXTURE / f"{name}.json"))
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 2))
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        data = Path(kwargs["env"]["INFERATLAS_DATA_DIR"])
        assert (data / "index.json").is_file()
        assert (data / "hardware.json").is_file()
        assert kwargs["env"]["INFERATLAS_SINGLE_MODEL"] == "qwen3-30b-a3b"
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


@pytest.mark.parametrize("device", ["A2", "H100"])
def test_single_default_filename_includes_device(tmp_path, monkeypatch, device):
    directory = tmp_path / "model-build"
    for name in ("ir", "view", "report", "triage"):
        write_json(directory / f"{name}.json", read_json(FIXTURE / f"{name}.json"))
    write_json(directory / "triage.json", {"items": []})
    catalog = cli.catalog()
    monkeypatch.setattr(cli, "catalog", lambda: catalog)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 0))

    def run(command, **kwargs):
        Path(kwargs["env"]["INFERATLAS_SINGLE_OUTPUT"]).write_text("<!doctype html>")

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = invoke("single", "--device", device)
    output = tmp_path / "dist" / f"inferatlas-qwen3-30b-a3b-{device.lower()}.html"
    assert result.output.strip() == str(output)
    assert sorted(p.name for p in output.parent.iterdir()) == [".inferatlas", output.name]
    assert read_json(output.parent / ".inferatlas" / f"{output.name}.json")["status"] == "ok"


def test_golden_missing_runtime_is_skipped_and_update_refuses(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES")
    monkeypatch.setattr(cli, "Evaluator", lambda *args: Evaluator(tmp_path, cli.warning))
    directory = tmp_path / "baseline"
    for name in ("ir", "eval"):
        write_json(directory / f"{name}.json", read_json(FIXTURE / f"{name}.json"))
    args = ["golden", "check", "--ir", FIXTURE / "ir.json", "--directory", directory]
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


def test_verify_calibration_env_command(tmp_path, monkeypatch, runtime):
    monkeypatch.setattr(cli, "Evaluator", lambda *args: runtime)
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    evaluator = runtime
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


def test_non_ascend_skips_annotations_entirely(monkeypatch):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    ir.snapshot.device = "H100"
    triage = Triage(snapshot=ir.snapshot, items=[])
    monkeypatch.setattr(cli, "optional", lambda *args: pytest.fail("annotation module loaded"))
    monkeypatch.setattr(cli, "Evaluator", lambda *args: pytest.fail("evaluator loaded"))
    assert cli.annotate_stage(ir, [], triage) == (ir, triage)


def test_ascend_annotations_require_extraction_facts(monkeypatch):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    monkeypatch.setattr(inferatlas.annotations, "load_annotations", lambda path: [])
    with pytest.raises(ValueError, match="supply --extract"):
        cli.annotate_stage(ir, [], Triage(snapshot=ir.snapshot, items=[]))


def test_annotation_context_uses_facts_and_node_scoped_selectors():
    primary = Extract.model_validate(read_json(FIXTURE / "extract.json"))
    primary.ascend.profile.update(
        weight_layout_policy="WeightLayoutPolicy.CONFIGURABLE",
        finegrained_tp_config={"oproj_tensor_parallel_size": 2, "mlp_tensor_parallel_size": 2},
    )
    row = next(m for m in primary.modules if m.path.endswith("o_proj"))
    qkv = next(m for m in primary.modules if m.path.endswith("qkv_proj"))
    assert cli.annotation_context(primary, row)["custom_parallel"]
    assert not cli.annotation_context(primary, qkv)["custom_parallel"]
    column = qkv.model_copy(update={"path": "model.layers.0.mlp.gate_up_proj"})
    assert cli.annotation_context(primary, column)["custom_parallel"]
    context = cli.annotation_context(primary, row)
    assert context["unquantized"] and not context["linear_nz"]
    assert not context["fused_mc2"]  # Hardware capability alone cannot enable it.
    primary.ascend.profile["weight_nz_mode"] = 2
    assert cli.annotation_context(primary, row)["linear_nz"]
    row.attrs["skip_weight_nz_conversion"] = True
    assert not cli.annotation_context(primary, row)["linear_nz"]
    row.attrs.pop("skip_weight_nz_conversion")
    primary.ascend.profile["weight_layout_policy"] = "WeightLayoutPolicy.FORCE_NZ"
    primary.ascend.profile["weight_nz_mode"] = 0
    assert cli.annotation_context(primary, row)["linear_nz"]
    primary.config["quantization_config"] = {"quant_method": "awq"}
    assert not cli.annotation_context(primary, row)["unquantized"]
    primary.ascend.profile["weight_layout_policy"] = "UNKNOWN"
    assert "linear_nz" not in cli.annotation_context(primary, row)
    primary.snapshot.ascend.commit = "f" * 40
    context = cli.annotation_context(primary, row)
    assert "capture_routing" not in context and "custom_parallel" not in context


def test_annotations_select_per_node_and_keep_collisions(monkeypatch, runtime):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    primary = Extract.model_validate(read_json(FIXTURE / "extract.json"))
    primary.ascend.profile.update(
        weight_layout_policy="CONFIGURABLE",
        finegrained_tp_config={"oproj_tensor_parallel_size": 2},
    )
    row = next(m for m in primary.modules if m.path.endswith("o_proj"))
    primary.modules.append(row.model_copy(update={"path": "model.layers.0.mlp.down_proj"}))
    node = next(n for n in ir.nodes if n.id == "attn.o")
    ir.nodes.append(node.model_copy(update={"id": "mlp.down", "label": "mlp.down"}))
    replacement = row.ascend.class_name
    resolved = [
        SimpleNamespace(
            annotation=SimpleNamespace(
                id=name,
                target=replacement,
                resolves_to=replacement,
                applies_when=condition,
            ),
            src=[],
        )
        for name, condition in (("parallel", "custom_parallel"), ("kernel", "!custom_parallel"))
    ]
    calls = []
    monkeypatch.setattr(cli, "Evaluator", lambda *args: runtime)
    monkeypatch.setattr(inferatlas.annotations, "load_annotations", lambda path: [])
    monkeypatch.setattr(inferatlas.annotations, "check_annotations", lambda *args: (resolved, []))
    monkeypatch.setattr(
        inferatlas.annotations,
        "apply_annotations",
        lambda graph, anns, **kw: calls.append(kw) or graph,
    )
    _, triage = cli.annotate_stage(ir, [primary], Triage(snapshot=ir.snapshot, items=[]))
    expected = {
        ("fused_moe.routed_experts.RoutedExperts", "fused_moe.routed_experts.AscendRoutedExperts"),
        ("fused_moe.router.gate_linear.GateLinear", "fused_moe.gate_linear.AscendGateLinear"),
        ("fused_moe.runner.moe_runner.MoERunner", "fused_moe.fused_moe.AscendMoERunner"),
        (
            "fused_moe.unquantized_fused_moe_method.UnquantizedFusedMoEMethod",
            "fused_moe.routed_experts.AscendUnquantizedFusedMoEMethod",
        ),
        ("layernorm.RMSNorm", "layernorm.AscendRMSNorm"),
        ("linear.QKVParallelLinear", "linear.AscendQKVParallelLinear"),
        ("logits_processor.LogitsProcessor", "vocab_parallel_embedding.AscendLogitsProcessor"),
        ("rotary_embedding.base.RotaryEmbedding", "rotary_embedding.AscendRotaryEmbedding"),
        ("rotary_embedding.common.ApplyRotaryEmb", "rotary_embedding.AscendApplyRotaryEmb"),
        (
            "vocab_parallel_embedding.ParallelLMHead",
            "vocab_parallel_embedding.AscendParallelLMHead",
        ),
        (
            "vocab_parallel_embedding.VocabParallelEmbedding",
            "vocab_parallel_embedding.AscendVocabParallelEmbedding",
        ),
    }
    assert {
        (item.symbol, tuple(item.candidates), item.reason, tuple(item.conditions))
        for item in triage.items
    } == {
        (
            f"vllm.model_executor.layers.{symbol}",
            (f"vllm_ascend.ops.{replacement}",),
            "unannotated_override",
            (),
        )
        for symbol, replacement in expected
    }
    assert calls[-1]["bindings"]["attn.o"] == "parallel"
    assert calls[-1]["bindings"]["mlp.down"] == "kernel"
    resolved.append(
        SimpleNamespace(
            annotation=SimpleNamespace(
                id="collision",
                target=replacement,
                resolves_to=replacement,
                applies_when="custom_parallel",
            ),
            src=[],
        )
    )
    _, triage = cli.annotate_stage(ir, [primary], Triage(snapshot=ir.snapshot, items=[]))
    assert "attn.o" not in calls[-1]["bindings"]
    assert any(len(item.candidates) == 2 for item in triage.items)


@pytest.mark.parametrize("reverse", [False, True])
def test_annotation_context_must_agree_across_extracts(monkeypatch, runtime, reverse):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    primary = Extract.model_validate(read_json(FIXTURE / "extract.json"))
    secondary = primary.model_copy(deep=True)
    secondary.parallel.tp = 4
    # This fact is outside the predicate; equal outcomes must not hide context drift.
    secondary.ascend.profile["enable_fused_mc2"] = True
    row = next(m for m in primary.modules if m.path.endswith("o_proj"))
    annotation = SimpleNamespace(id="row", resolves_to=row.ascend.class_name, applies_when="is_a2")
    calls = []
    monkeypatch.setattr(cli, "Evaluator", lambda *args: runtime)
    monkeypatch.setattr(inferatlas.annotations, "load_annotations", lambda path: [])
    monkeypatch.setattr(
        inferatlas.annotations,
        "check_annotations",
        lambda *args: ([SimpleNamespace(annotation=annotation, src=[])], []),
    )
    monkeypatch.setattr(
        inferatlas.annotations,
        "apply_annotations",
        lambda graph, anns, **kw: calls.append(kw) or graph,
    )
    extracts = [secondary, primary] if reverse else [primary, secondary]
    _, triage = cli.annotate_stage(ir, extracts, Triage(snapshot=ir.snapshot, items=[]))
    assert calls[-1] == {"bindings": {}, "active_ids": set()}
    assert any(
        item.symbol == row.class_name and item.conditions == ["is_a2"] for item in triage.items
    )
    assert (
        cli.exit_code(
            Report(snapshot=ir.snapshot, checks=[], golden_diff=[], source_link_failures=[]), triage
        )
        == 2
    )


def test_site_hands_off_fresh_data_and_replaces_explicit_export(tmp_path, monkeypatch):
    site_build(tmp_path, 0)
    export = tmp_path / "data"
    write_json(export / "stale.json", {})
    seen = []

    def run(command, **kwargs):
        data = Path(kwargs["env"]["INFERATLAS_DATA_DIR"])
        assert data != export and data != ROOT / "web/public/data"
        assert not (data / "stale.json").exists()
        assert (data / "index.json").is_file()
        seen.append(data)

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("site", "--build-root", tmp_path / "build", "--data-dir", export)
    assert not (export / "stale.json").exists()
    assert (export / "index.json").is_file()
    invoke("site", "--build-root", tmp_path / "build")
    assert len(seen) == 2 and all(not path.exists() for path in seen)


def test_optional_does_not_hide_a_broken_installed_stage(monkeypatch):
    def broken_import(name):
        raise ModuleNotFoundError("missing stage dependency", name="missing_dependency")

    monkeypatch.setattr(cli.importlib, "import_module", broken_import)
    with pytest.raises(ModuleNotFoundError, match="stage dependency"):
        cli.optional("extract", "extract")


def test_site_reports_the_reason_for_refusal(tmp_path, monkeypatch):
    directory = site_build(tmp_path, 3)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("web ran"))
    args = ["site", "--build-root", tmp_path / "build", "--data-dir", tmp_path / "data"]
    (directory / "view.json").unlink()
    assert "failed verification" in invoke(*args, code=3).output
    report = read_json(directory / "report.json")
    report["checks"][0]["status"] = "fail"
    write_json(directory / "report.json", report)
    assert "fixture or unverified provenance" in invoke(*args, code=3).output
    site_build(tmp_path, 2)
    assert "blocked by triage" in invoke(*args, code=2).output


@pytest.mark.parametrize("inactive", [False, True])
def test_observed_swaps_without_active_annotations_block_publication(
    monkeypatch, runtime, inactive
):
    ir = IR.model_validate(read_json(FIXTURE / "ir.json"))
    primary = Extract.model_validate(read_json(FIXTURE / "extract.json"))
    secondary = primary.model_copy(deep=True)
    secondary.parallel.tp = 4
    next(m for m in secondary.modules if m.ascend).ascend.class_name += "TP4"
    unmatched = next(m for m in secondary.modules if m.ascend).model_copy(deep=True)
    unmatched.class_name = "new.UnrepresentedOp"
    unmatched.ascend.class_name = "new.AscendUnrepresentedOp"
    secondary.modules.append(unmatched)
    monkeypatch.setattr(cli, "Evaluator", lambda *args: runtime)
    monkeypatch.setattr(inferatlas.annotations, "load_annotations", lambda path: [])
    resolved = (
        [
            SimpleNamespace(
                annotation=SimpleNamespace(
                    id="inactive", resolves_to=m.ascend.class_name, applies_when="false"
                ),
                src=[],
            )
            for m in primary.modules
            if m.ascend
        ]
        if inactive
        else []
    )
    # Repeated modules share the same implementation; keep annotation ids unique.
    resolved = list({item.annotation.resolves_to: item for item in resolved}.values())
    for i, item in enumerate(resolved):
        item.annotation.id = f"inactive-{i}"
    monkeypatch.setattr(inferatlas.annotations, "check_annotations", lambda *args: (resolved, []))
    result, triage = cli.annotate_stage(
        ir, [primary, secondary], Triage(snapshot=ir.snapshot, items=[])
    )
    replacements = {
        m.ascend.class_name for extract in (primary, secondary) for m in extract.modules if m.ascend
    }
    blocked = {candidate for item in triage.items for candidate in item.candidates}
    assert replacements <= blocked
    assert all(item.reason == "unannotated_override" for item in triage.items)
    report = Report(snapshot=result.snapshot, checks=[], golden_diff=[], source_link_failures=[])
    assert cli.exit_code(report, triage) == 2
