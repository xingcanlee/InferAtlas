"""Verification boundaries: immutable Git evidence, numeric adapters, golden drift."""

import json
import shutil
import subprocess

import pytest

import inferatlas.expr
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
def oracle(monkeypatch):
    monkeypatch.delattr(inferatlas.expr, "evaluate", raising=False)
    result = Evaluator(ROOT, lambda message: None)
    result.backend = "fixture"
    return result


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


def test_fixture_golden_and_changed_field(tmp_path, ir, oracle):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    update_golden(tmp_path, ir, grid, oracle)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    update_golden(tmp_path, ir, grid, oracle)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert check_golden(tmp_path, ir, oracle) == []
    changed = ir.model_copy(deep=True)
    changed.nodes[0].label = "Changed label"
    diffs = check_golden(tmp_path, changed, oracle)
    assert [d.path for d in diffs] == [f"/ir/nodes/{ir.nodes[0].id}/label"]
    assert "Changed label" in format_diff(diffs)
    changed.memory.weights = "0"
    assert "/ir/memory/weights" in format_diff(check_golden(tmp_path, changed, oracle))
    with pytest.raises(ValueError, match="changed expression"):
        update_golden(tmp_path, changed, grid, oracle)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_numeric_drift_uses_independent_fixture_oracle(tmp_path, ir, oracle):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    update_golden(tmp_path, ir, grid, oracle)
    data = read_json(tmp_path / "eval.json")
    case = next(c for c in data["cases"] if "/memory/weights" in c["expected"])
    case["expected"]["/memory/weights"] = 123
    write_json(tmp_path / "eval.json", data)
    diffs = check_golden(tmp_path, ir, oracle)
    assert len(diffs) == 1
    assert diffs[0].path.endswith("/expected/~1memory~1weights")
    assert diffs[0].expected == 123
    assert diffs[0].actual == 61064245248


def test_oracle_rejects_new_inputs(ir, oracle):
    grid = Eval.model_validate(read_json(FIXTURE / "eval.json"))
    grid.cases[0].env["tp"] = 8
    with pytest.raises(ValueError, match="no parameter point"):
        evaluate_grid(ir, grid, oracle)
    grid.model = "other/model"
    with pytest.raises(ValueError, match="model disagrees"):
        evaluate_grid(ir, grid, oracle)


def test_json_pointer_escapes_and_node_ids(ir):
    doc = projection(ir)
    assert pointer(doc, "/nodes/attn.o.reduce/comm/bus_bytes/prefill")
    assert pointer({"a/b": {"x~y": [3]}}, "/a~1b/x~0y/0") == 3
    diff = differences({"a/b": {"x~y": 3}}, {"a/b": {"x~y": 4}})
    assert diff[0].path == "/a~1b/x~0y"


@pytest.mark.skipif(not shutil.which("node"), reason="Node is required for the CLI adapter")
def test_node_evaluator_protocol_and_failures(tmp_path):
    cli = tmp_path / "packages/expr/dist/cli.js"
    cli.parent.mkdir(parents=True)
    cli.write_text(
        "let s=''; process.stdin.on('data', x=>s+=x); process.stdin.on('end',()=>{"
        "const q=JSON.parse(s); process.stdout.write(JSON.stringify({values:"
        "q.expressions.map((_,i)=>q.env.x+i)}));});"
    )
    evaluator = Evaluator(tmp_path)
    assert evaluator.backend == "node"
    assert evaluator.many(["x", "x+1"], {"x": 7}) == [7, 8]
    cli.write_text("process.stderr.write('broken evaluator'); process.exit(1)")
    with pytest.raises(ValueError, match="broken evaluator"):
        evaluator.many(["x"], {})
    cli.write_text("process.stdout.write('{\"values\":[null]}')")
    with pytest.raises(ValueError, match="non-finite/non-numeric"):
        evaluator.many(["x"], {})


def test_python_reference_fallback_and_parameter_binding(tmp_path, ir, monkeypatch):
    messages = []
    monkeypatch.setattr(inferatlas.expr, "evaluate", lambda expr, env: env["H"] * 2, raising=False)
    evaluator = Evaluator(tmp_path, messages.append)
    assert evaluator.backend == "python" and messages
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


def test_grid_batches_shapes_and_phase_expressions(tmp_path, ir, monkeypatch):
    def reference(expression, env):
        if expression == "tp > 1":
            return int(env["tp"] > 1)
        return env[expression]

    monkeypatch.setattr(inferatlas.expr, "evaluate", reference, raising=False)
    evaluator = Evaluator(tmp_path, lambda message: None)
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


def test_calibration_parser_and_tolerance():
    metrics = parse_vllm_logs("""
    INFO Model loading took 2.5 GiB and 0.1 seconds
    INFO # GPU blocks: 1,024, # CPU blocks: 0
    INFO GPU KV cache size: 131,072 tokens
    INFO Available KV cache memory: 8.25 GiB
    INFO Model loading took 3 GiB
    """)
    assert metrics == {
        "weight_bytes": 3 * 2**30,
        "gpu_blocks": 1024,
        "kv_tokens": 131072,
        "available_kv_bytes": 8.25 * 2**30,
    }
    assert parse_vllm_logs("KV cache size: 10 tokens")["kv_tokens"] == 10
    for estimate, status in [(90, "pass"), (110, "pass"), (110.01, "fail")]:
        assert compare_calibration({"kv_tokens": estimate}, {"kv_tokens": 100})[0].status == status
    assert compare_calibration({"kv_tokens": 0}, {"kv_tokens": 0})[0].status == "pass"
    assert compare_calibration({"kv_tokens": 1}, {"kv_tokens": 0})[0].status == "fail"
    assert compare_calibration({}, {"gpu_blocks": 10})[0].status == "skip"
    assert compare_calibration({}, {})[0].status == "skip"
    with pytest.raises(ValueError, match="invalid calibration"):
        compare_calibration({}, {"kv_tokens": float("nan")})


def test_report_contract_and_exit_precedence(ir, oracle, monkeypatch):
    import inferatlas.verify as verify

    monkeypatch.setattr(verify, "check_sources", lambda *args: [])
    triage = Triage.model_validate(read_json(FIXTURE / "triage.json"))
    report = verify_ir(ir, triage=triage, repos={}, evaluator=oracle)
    assert Report.model_validate_json(report.model_dump_json()) == report
    assert exit_code(report, triage) == 2
    assert next(c for c in report.checks if c.id == "evaluation").status == "skip"
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
        verify_ir(ir, triage=wrong, repos={}, evaluator=oracle)
