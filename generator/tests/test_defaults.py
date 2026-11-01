"""Release defaults use the real device ledger and Node expression semantics."""

import subprocess

import pytest
from t6_runtime import node_backend as node_backend
from t6_runtime import runtime as runtime

from inferatlas.schema import IR
from inferatlas.verify.defaults import TP_DEFAULT_RULE, select_default_tp
from inferatlas.verify.golden import Evaluator, ParameterDomainError
from inferatlas.verify.io import ROOT, read_json


@pytest.fixture
def ir():
    return IR.model_validate(read_json(ROOT / "golden/qwen3-30b-a3b/A2/ir.json"))


@pytest.mark.parametrize(
    "model,device,expected",
    [
        ("qwen3-30b-a3b", "A2", 2),
        ("qwen3-30b-a3b", "H100", 1),
        ("qwen3-8b", "A2", 1),
        ("qwen3-8b", "H100", 1),
    ],
)
def test_device_default_is_smallest_positive_capacity(model, device, expected, runtime):
    ir = IR.model_validate(read_json(ROOT / f"golden/{model}/{device}/ir.json"))
    before = ir.model_copy(deep=True)
    ir.params["tp"].choices.reverse()
    check = select_default_tp(ir, runtime)
    assert ir.params["tp"].default == str(expected)
    assert runtime.many([ir.memory.kv_tokens], runtime.bind(ir, {}))[0] > 0
    for tp in range(1, expected):
        assert runtime.many([ir.memory.kv_tokens], runtime.bind(ir, {"tp": tp}))[0] == 0
    assert {k: v for k, v in ir.params.items() if k != "tp"} == {
        k: v for k, v in before.params.items() if k != "tp"
    }
    assert ir.memory == before.memory
    assert TP_DEFAULT_RULE in ir.params["tp"].description
    assert f"{device}: tp={expected}" in check.message
    assert check.status == "pass"
    assert select_default_tp(ir, runtime) == check
    assert ir.params["tp"].description.count(TP_DEFAULT_RULE) == 1


def test_no_capacity_keeps_honest_zero_warning(ir, runtime):
    ir.params["hbm_bytes"].default = "1"
    ir.params["tp"].default = "4"
    check = select_default_tp(ir, runtime)
    assert ir.params["tp"].default == "1"
    assert "No TP choice has positive KV capacity" in check.message
    assert runtime.many([ir.memory.kv_tokens], runtime.bind(ir, {})) == [0]


def test_other_defaults_and_constraints_are_respected(ir, runtime):
    ir.params["dp"].default = "2"
    ir.params["ep"].default = "8"
    select_default_tp(ir, runtime)
    assert ir.params["tp"].default == "4"
    assert ir.params["dp"].default == "2"
    assert ir.params["ep"].default == "8"


def test_no_valid_candidate_fails(ir, runtime):
    ir.params["tp"].constraint = "tp>8"
    with pytest.raises(ValueError, match="no valid TP choice"):
        select_default_tp(ir, runtime)


@pytest.mark.parametrize(
    "field,value,expected",
    [("min", "2", "4"), ("max", "2", "1"), ("step", "2", "4"), ("constraint", "tp > 1", "4")],
)
def test_domain_errors_skip_only_invalid_candidates(ir, runtime, field, value, expected):
    setattr(ir.params["tp"], field, value)
    ir.memory.kv_tokens = "tp == 4 ? 1 : 0"
    select_default_tp(ir, runtime)
    assert ir.params["tp"].default == expected


def test_choice_domain_errors_are_distinct(ir, runtime):
    ir.params["dp"].choices = ["2"]
    with pytest.raises(ParameterDomainError, match="outside parameter choices"):
        runtime.bind(ir, {"tp": 1})
    with pytest.raises(ValueError, match="no valid TP choice"):
        select_default_tp(ir, runtime)


def test_tp_choices_use_shared_expression_semantics(ir, runtime):
    ir.params["tp"].choices = ["2 ** 2", "1 + 1", "1"]
    select_default_tp(ir, runtime)
    assert ir.params["tp"].default == "2"


@pytest.mark.parametrize("symbol", ["missing", "bad"])
def test_symbol_binding_failures_propagate(ir, runtime, symbol):
    ir.symbols["bad"] = symbol
    with pytest.raises(ValueError, match="cyclic or unknown"):
        select_default_tp(ir, runtime)


def test_nonfinite_binding_failure_propagates(ir, runtime):
    ir.params["hbm_bytes"].default = "1e999"
    with pytest.raises(ValueError, match="[Nn]on-finite"):
        select_default_tp(ir, runtime)


@pytest.mark.parametrize(
    "error",
    [
        ValueError("expression CLI failed: broken evaluator"),
        ValueError("expression evaluator returned a non-finite/non-numeric value"),
        subprocess.TimeoutExpired("node", 60),
    ],
)
def test_evaluator_failures_propagate(ir, runtime, monkeypatch, error):
    def fail(*args):
        raise error

    monkeypatch.setattr(runtime, "many", fail)
    with pytest.raises(type(error), match=str(error)):
        select_default_tp(ir, runtime)


def test_missing_node_keeps_extracted_defaults(ir, tmp_path, monkeypatch):
    evaluator = Evaluator(tmp_path)
    monkeypatch.setattr(evaluator, "bind", lambda *args: pytest.fail("binding attempted"))
    before = ir.model_copy(deep=True)
    check = select_default_tp(ir, evaluator)
    assert (check.id, check.status) == ("default_parameters", "skip")
    assert "kept the extracted default" in check.message
    assert ir == before
