"""Release defaults use the real device ledger and Node expression semantics."""

import pytest
from t6_runtime import node_backend as node_backend
from t6_runtime import runtime as runtime

from inferatlas.schema import IR
from inferatlas.verify.defaults import TP_DEFAULT_RULE, select_default_tp
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
