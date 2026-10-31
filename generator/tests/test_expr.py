"""Cross-language DSL vectors; only this test module imports the reference evaluator."""

import json
from pathlib import Path

import pytest

from inferatlas.expr import ExprSyntaxError, validate
from inferatlas.expr._reference import evaluate_reference

ROOT = Path(__file__).resolve().parents[2]
VECTORS = json.loads((ROOT / "packages/expr/test-vectors.json").read_text())


@pytest.mark.parametrize("vector", VECTORS, ids=lambda v: v["expr"])
def test_expr_vectors(vector):
    expr, env = vector["expr"], vector["env"]
    if "error" in vector:
        with pytest.raises(ExprSyntaxError) as caught:
            evaluate_reference(expr, env)
        assert caught.value.code == vector["error"]
        assert caught.value.offset >= 0
        if vector["error"] in {"syntax", "arity"} or "1e309" in expr:
            with pytest.raises(ExprSyntaxError):
                validate(expr)
        else:
            # Runtime type/domain/unknown-symbol errors are valid symbolic expressions.
            validate(expr)
    else:
        assert evaluate_reference(expr, env) == vector["value"]
        free = validate(expr)
        assert free == sorted(set(free))
        if "free" in vector:
            assert free == vector["free"]


def test_expr_free_symbols_and_positions():
    assert len(VECTORS) >= 150
    assert validate("flag ? bytes(dtype_w) : max(z, a, z)") == ["a", "dtype_w", "flag", "z"]
    assert validate("false && missing") == ["missing"]
    assert validate("bytes(bf16) + bytes(int64)") == []
    with pytest.raises(ExprSyntaxError) as caught:
        validate("1 +\n @")
    assert caught.value.position == {"offset": 5, "line": 2, "column": 2}
    with pytest.raises(ExprSyntaxError) as caught:
        validate("1 +\n")
    assert caught.value.position == {"offset": 4, "line": 2, "column": 1}


def test_expr_all_fixture_formulas():
    """Cover all formula slots, not only metrics with a pinned numeric oracle."""
    ir = json.loads((ROOT / "fixtures/qwen3-30b-a3b/ir.json").read_text())
    symbols = {name: float(expr) for name, expr in ir["symbols"].items()}
    env = dict(symbols)
    for name, param in ir["params"].items():
        env[name] = (
            param["default"]
            if param["kind"] in {"dtype", "enum"}
            else evaluate_reference(param["default"], env)
        )
    expressions = list(ir["memory"].values())
    expressions = [expr for expr in expressions if isinstance(expr, str)]
    for node in ir["nodes"]:
        expressions.append(node["weight_bytes"])
        for key in ("flops", "bytes_moved"):
            expressions.extend(node[key].values())
        if node.get("comm"):
            expressions.append(node["comm"]["when"])
            for key in ("logical_bytes", "bus_bytes"):
                expressions.extend(node["comm"][key].values())
        for slot in node["in"] + node["out"]:
            shapes = slot["shape"]
            expressions.extend(
                shapes if isinstance(shapes, list) else shapes["prefill"] + shapes["decode"]
            )
    for tp in (1, 4):
        for expr in expressions:
            assert set(validate(expr)) <= env.keys()
            evaluate_reference(expr, {**env, "tp": tp})
