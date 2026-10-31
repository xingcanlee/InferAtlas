# ruff: noqa: UP031  # Preserve upstream logger format strings verbatim.
"""Verification boundaries: immutable Git evidence, numeric adapters, golden drift."""

import json
import shutil
import subprocess

import pytest

from inferatlas.schema import IR, Eval, Report, Snapshot, Triage
from inferatlas.schema.models import Check
from inferatlas.verify import exit_code, verify_ir
from inferatlas.verify.calibration import compare_calibration, parse_vllm_logs
from inferatlas.verify.golden import (
    Evaluator,
    check_golden,
    differences,
    evaluate_grid,
    format_diff,
    pointer,
    projection,
    update_golden,
)
from inferatlas.verify.io import ROOT, read_json, write_json
from inferatlas.verify.source import check_sources, source_refs

FIXTURE = ROOT / "fixtures/qwen3-30b-a3b"


@pytest.fixture
def ir():
    return IR.model_validate(read_json(FIXTURE / "ir.json"))


@pytest.fixture
def runtime():
    evaluator = Evaluator(ROOT)
    assert evaluator.backend == "node", "build @inferatlas/contracts and @inferatlas/expr first"
    return evaluator


def git(path, *args):
    return subprocess.run(
        ["git", "-C", str(path), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def test_sources_read_commit_not_working_tree(tmp_path, ir):
    git(tmp_path, "init", "-q")
    source = tmp_path / "module.py"
    source.write_text("class Layer:\n    pass\n\nasync def forward():\n    pass\n")
    git(tmp_path, "add", "module.py")
    git(
        tmp_path,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "fixture",
    )
    commit = git(tmp_path, "rev-parse", "HEAD")
    snapshot = ir.snapshot.model_copy(
        update={"vllm": ir.snapshot.vllm.model_copy(update={"commit": commit})}
    )
    ref = dict(repo="vllm", file="module.py", line=1, commit=commit, symbol="package.Layer")
    forward = {**ref, "line": 4, "symbol": "package.forward"}
    document = {"first": ref, "nested": [ref, forward]}
    assert len(source_refs(document)) == 2
    source.write_text("changed working tree\n")
    assert not check_sources([document], snapshot, {"vllm": tmp_path})
    cases = [
        ({**ref, "line": 2}, "does not declare"),
        ({**ref, "line": 100}, "outside file"),
        ({**ref, "symbol": "Other"}, "does not declare"),
        ({**ref, "file": "missing.py"}, "unavailable via git show"),
        ({**ref, "file": "../module.py"}, "repository-relative"),
        ({**ref, "commit": "f" * 40}, "disagrees with snapshot"),
    ]
    for changed, reason in cases:
        failures = check_sources([changed], snapshot, {"vllm": tmp_path})
        assert len(failures) == 1
        assert reason in failures[0].reason
    assert "repository unavailable" in check_sources([ref], snapshot, {})[0].reason


def test_fixture_golden_and_changed_field(tmp_path, ir, runtime):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    update_golden(tmp_path, ir, grid, runtime)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    update_golden(tmp_path, ir, grid, runtime)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert check_golden(tmp_path, ir, runtime) == []
    changed = ir.model_copy(deep=True)
    changed.nodes[0].label = "Changed label"
    diffs = check_golden(tmp_path, changed, runtime)
    assert [d.path for d in diffs] == [f"/ir/nodes/{ir.nodes[0].id}/label"]
    assert "Changed label" in format_diff(diffs)
    changed.memory.weights = "0"
    assert "/ir/memory/weights" in format_diff(check_golden(tmp_path, changed, runtime))
    update_golden(tmp_path, changed, grid, runtime)
    assert first != {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_numeric_drift_uses_independent_node_runtime(tmp_path, ir, runtime):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    update_golden(tmp_path, ir, grid, runtime)
    data = read_json(tmp_path / "eval.json")
    case = next(c for c in data["cases"] if "/memory/weights" in c["expected"])
    case["expected"]["/memory/weights"] = 123
    write_json(tmp_path / "eval.json", data)
    diffs = check_golden(tmp_path, ir, runtime)
    assert len(diffs) == 1
    assert diffs[0].path.endswith("/expected/~1memory~1weights")
    assert diffs[0].expected == 123
    assert diffs[0].actual == 61064245248


def test_node_evaluates_new_points(ir, runtime):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    grid.cases[0].env["tp"] = 8
    assert evaluate_grid(ir, grid, runtime).cases
    grid.model = "other/model"
    with pytest.raises(ValueError, match="model disagrees"):
        evaluate_grid(ir, grid, runtime)


def test_json_pointer_escapes_and_node_ids(ir):
    doc = projection(ir)
    assert pointer(doc, "/nodes/attn.o.reduce/comm/bus_bytes/prefill")
    assert pointer({"a/b": {"x~y": [3]}}, "/a~1b/x~0y/0") == 3
    diff = differences({"a/b": {"x~y": 3}}, {"a/b": {"x~y": 4}})
    assert diff[0].path == "/a~1b/x~0y"


@pytest.mark.skipif(not shutil.which("node"), reason="Node is required")
def test_node_evaluator_protocol_and_failures(tmp_path):
    module = tmp_path / "packages/expr/dist/index.js"
    module.parent.mkdir(parents=True)
    module.with_name("package.json").write_text('{"type":"module"}')
    module.write_text("export function evaluate(expr, env) {return expr === 'x' ? env.x : env.x+1}")
    evaluator = Evaluator(tmp_path)
    assert evaluator.backend == "node"
    assert evaluator.many(["x", "x+1"], {"x": 7}) == [7, 8]
    module.write_text("throw new Error('broken evaluator')")
    with pytest.raises(ValueError, match="broken evaluator"):
        evaluator.many(["x"], {})
    module.write_text("export function evaluate() {return NaN}")
    with pytest.raises(ValueError, match="non-finite/non-numeric"):
        evaluator.many(["x"], {})


def test_node_parameter_binding(ir, runtime):
    evaluator = runtime
    env = evaluator.bind(ir, {"tp": 4})
    assert env["tp"] == 4 and env["H"] == 2048 and env["dtype_w"] == "bf16"
    assert evaluator.many(["H * 2"], env) == [4096]
    for overrides in (
        {"H": 2},
        {"tp": 3},
        {"tp": True},
        {"tp": 1.5},
        {"dtype_w": "invalid"},
        {"gpu_memory_utilization": 2},
    ):
        with pytest.raises(ValueError):
            evaluator.bind(ir, overrides)
    graph = ir.model_copy(deep=True)
    graph.symbols.update(A="B", B="A")
    with pytest.raises(ValueError, match="cyclic"):
        evaluator.bind(graph, {})
    graph.symbols = {**ir.symbols, "illegal": "tp"}
    with pytest.raises(ValueError, match="unknown environment"):
        evaluator.bind(graph, {"tp": 4})


def test_grid_batches_shapes_and_phase_expressions(ir, runtime):
    evaluator = runtime
    node = next(node for node in ir.nodes if node.id == "attn.qkv")
    node.out[0].shape = ["T", "H"]
    comm = next(node for node in ir.nodes if node.id == "attn.o.reduce").comm
    comm.when = "tp > 1"
    comm.logical_bytes.prefill = "T"
    comm.logical_bytes.decode = "S"
    grid = Eval.model_validate(
        {
            "model": ir.snapshot.model.hf_id,
            "cases": [
                {
                    "id": "custom",
                    "env": {"tp": 4, "T": 8, "S": 3},
                    "expected": {
                        "/nodes/attn.qkv/out/0/shape": [0, 0],
                        "/nodes/attn.o.reduce/comm/when": 0,
                        "/nodes/attn.o.reduce/comm/logical_bytes/prefill": 0,
                        "/nodes/attn.o.reduce/comm/logical_bytes/decode": 0,
                    },
                }
            ],
        }
    )
    values = evaluate_grid(ir, grid, evaluator).cases[0].expected
    assert values["/nodes/attn.qkv/out/0/shape"] == [8, 2048]
    assert values["/nodes/attn.o.reduce/comm/when"] == 1
    assert values["/nodes/attn.o.reduce/comm/logical_bytes/prefill"] == 8
    assert values["/nodes/attn.o.reduce/comm/logical_bytes/decode"] == 3


# Format strings copied verbatim from the pinned upstream sources:
# vllm/v1/worker/gpu_model_runner.py, vllm/v1/core/kv_cache_utils.py,
# vllm/v1/worker/gpu_worker.py, vllm_ascend/worker/v1/model_runner.py.
@pytest.mark.parametrize("device", ["GPU", "NPU", "GPU + host"])
def test_calibration_upstream_formats(device):
    metrics = parse_vllm_logs(
        "\n".join(
            [
                "Model loading took %s GiB memory and %.6f seconds" % (2.5, 0.1),
                "Loading model weights took %.4f GB" % 3.25,
                "%s KV cache size: %s tokens, "
                "Maximum concurrency for %s tokens per request: %.2fx"
                % (device, "131,072", "4,096", 32),
                "Available KV cache memory: %s GiB" % -0.25,
                "Available KV cache memory: %.2f GiB" % -0.5,
                "# GPU blocks: 1,024, # CPU blocks: 0",
            ]
        )
    )
    assert metrics == {
        "weight_bytes": 3.25 * 2**30,
        "kv_tokens": 131072,
        "available_kv_bytes": -0.5 * 2**30,
        "maximum_concurrency": 32,
    }
    assert parse_vllm_logs("KV cache size: 10 tokens") == {}


def test_calibration_tolerance():
    for estimate, status in [(90, "pass"), (110, "pass"), (110.01, "fail")]:
        assert compare_calibration({"kv_tokens": estimate}, {"kv_tokens": 100})[0].status == status
    assert compare_calibration({"kv_tokens": 0}, {"kv_tokens": 0})[0].status == "pass"
    assert compare_calibration({"kv_tokens": 1}, {"kv_tokens": 0})[0].status == "fail"
    assert (
        compare_calibration({"available_kv_bytes": -99}, {"available_kv_bytes": -100})[0].status
        == "pass"
    )
    assert compare_calibration({}, {})[0].status == "skip"
    with pytest.raises(ValueError, match="invalid calibration"):
        compare_calibration({}, {"kv_tokens": float("nan")})


def test_report_contract_and_exit_precedence(ir, runtime, monkeypatch):
    import inferatlas.verify as verify

    monkeypatch.setattr(verify, "check_sources", lambda *args: [])
    triage = Triage.model_validate(read_json(FIXTURE / "triage.json"))
    report = verify_ir(ir, triage=triage, repos={}, evaluator=runtime)
    assert Report.model_validate_json(report.model_dump_json()) == report
    assert exit_code(report, triage) == 2
    assert next(c for c in report.checks if c.id == "evaluation").status == "pass"
    report.checks.append(Check(id="golden", status="fail", message="drift"))
    assert exit_code(report, triage) == 3
    assert (
        exit_code(Report(snapshot=ir.snapshot, checks=[], golden_diff=[], source_link_failures=[]))
        == 0
    )
    wrong = triage.model_copy(
        update={
            "snapshot": Snapshot.model_validate(
                {**json.loads(ir.snapshot.model_dump_json()), "device": "A3"}
            )
        }
    )
    with pytest.raises(ValueError, match="triage snapshot"):
        verify_ir(ir, triage=wrong, repos={}, evaluator=runtime)


def test_duplicate_ids_are_a_difference(ir, runtime, tmp_path):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    update_golden(tmp_path, ir, grid, runtime)
    changed = ir.model_copy(deep=True)
    changed.nodes.append(changed.nodes[0].model_copy(deep=True))
    assert check_golden(tmp_path, changed, runtime)
    duplicate = [{"id": "same"}, {"id": "same"}]
    assert differences(duplicate, duplicate)[0].path == "/duplicate_ids"
    assert differences(duplicate, [{"id": "same"}])
    data = read_json(tmp_path / "eval.json")
    data["cases"].append(data["cases"][0])
    write_json(tmp_path / "eval.json", data)
    assert check_golden(tmp_path, ir, runtime)[0].path == "/eval/cases/duplicate_ids"


def test_missing_runtime_report_and_fixture_provenance(ir, tmp_path, monkeypatch):
    import inferatlas.verify as verify

    monkeypatch.setattr(verify, "check_sources", lambda *args: [])
    missing = Evaluator(tmp_path, lambda message: None)
    report = verify_ir(ir, repos={}, evaluator=missing)
    assert exit_code(report) == 3
    assert next(c for c in report.checks if c.id == "evaluation").status == "fail"
    report = verify_ir(ir, repos={}, evaluator=missing, allow_fixtures=True)
    assert next(c for c in report.checks if c.id == "evaluation").status == "skip"
    assert next(c for c in report.checks if c.id == "provenance").status == "fail"
    assert exit_code(report) == 3


def test_calibration_binds_environment_and_available_memory(ir, runtime, monkeypatch):
    import inferatlas.verify as verify

    monkeypatch.setattr(verify, "check_sources", lambda *args: [])
    overrides = {"tp": 4, "gpu_memory_utilization": 0.8}
    env = runtime.bind(ir, overrides)
    weights, tokens, activation, non_torch, reserved = runtime.many(
        [
            ir.memory.weights,
            ir.memory.kv_tokens,
            ir.memory.activation_peak,
            ir.memory.non_torch,
            ir.memory.reserved,
        ],
        env,
    )
    available = env["hbm_bytes"] * 0.8 - weights - activation - non_torch - reserved
    observed = {
        "weight_bytes": weights,
        "kv_tokens": tokens,
        "available_kv_bytes": available,
        "maximum_concurrency": tokens / 4096,
    }
    report = verify_ir(
        ir,
        repos={},
        evaluator=runtime,
        calibration=observed,
        calibration_env={**overrides, "max_model_len": 4096},
    )
    checks = [c for c in report.checks if c.id.startswith("calibration.")]
    assert len(checks) == 4 and all(c.status == "pass" for c in checks)
    report = verify_ir(ir, repos={}, evaluator=runtime, calibration=observed)
    assert any(c.status == "fail" for c in report.checks if c.id.startswith("calibration."))
