"""Joint control domains and version diagnostics across the public wire contract."""

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from ruamel.yaml import YAML
from test_lower_helpers import environment

from inferatlas.expr import validate
from inferatlas.expr._reference import evaluate_reference
from inferatlas.lower import coverage_gate, lower
from inferatlas.lower.pipeline import runtime_params
from inferatlas.schema import IR, Extract, Triage
from inferatlas.schema.models import ParamSpec

ROOT = Path(__file__).resolve().parents[2]
MOE = ROOT / "fixtures/qwen3-30b-a3b"


def test_optional_constraint_roundtrip_and_types():
    legacy = {"kind": "integer", "default": "1", "description": "Expert degree."}
    assert ParamSpec.model_validate(legacy).constraint is None
    schema = json.loads((ROOT / "schemas/ir.schema.json").read_text())["$defs"]["ParamSpec"]
    validator = Draft202012Validator(schema)
    for value in [None, "ep==1 || (ep==tp*dp && E%ep==0)"]:
        wire = legacy | {"constraint": value}
        validator.validate(wire)
        model = ParamSpec.model_validate(wire)
        assert ParamSpec.model_validate_json(model.model_dump_json()) == model
    for invalid in ["", 1, True, ["ep==1"]]:
        wire = legacy | {"constraint": invalid}
        assert not validator.is_valid(wire)
        with pytest.raises(ValidationError):
            ParamSpec.model_validate(wire)


@pytest.mark.parametrize("model", ["qwen3-30b-a3b", "qwen3-8b"])
def test_lowering_and_fixtures_share_constraints_and_valid_defaults(model):
    ir = IR.model_validate_json((ROOT / f"fixtures/{model}/ir.json").read_text())
    extract = Extract.model_validate_json((MOE / "extract.json").read_text())
    catalog = YAML(typ="safe").load(ROOT / "catalog.yaml")
    config = next(m["config"] for m in catalog["models"] if m["id"] == model)
    extract = extract.model_copy(
        update={"config": config, "layer_types": ["full_attention"] * config["num_hidden_layers"]}
    )
    actual = runtime_params(extract)
    assert {k: p.constraint for k, p in actual.items()} == {
        k: p.constraint for k, p in ir.params.items()
    }
    env = environment(ir)
    for spec in actual.values():
        if spec.constraint is not None:
            assert set(validate(spec.constraint)) <= env.keys()
            assert evaluate_reference(spec.constraint, env) == 1
    # Mixed layer bands cannot represent equal PP partitions across templates.
    extract.layer_types[-1] = "other"
    assert runtime_params(extract)["pp"].constraint == "pp==1"


@pytest.mark.parametrize(
    "model,overrides,invalid",
    [
        ("qwen3-30b-a3b", {"tp": 4, "dp": 2, "ep": 1}, set()),
        ("qwen3-30b-a3b", {"tp": 4, "dp": 2, "ep": 8}, set()),
        ("qwen3-30b-a3b", {"tp": 8, "dp": 1, "ep": 1}, set()),
        ("qwen3-30b-a3b", {"tp": 4, "dp": 2, "ep": 4}, {"ep"}),
        ("qwen3-30b-a3b", {"tp": 4, "dp": 3, "ep": 12}, {"ep"}),
        ("qwen3-30b-a3b", {"tp": 4, "dp": 5, "ep": 1}, {"dp"}),
        ("qwen3-30b-a3b", {"tp": 4, "dp": 2, "ep": 8, "I_moe": 769}, set()),
        ("qwen3-30b-a3b", {"tp": 8, "Hkv": 3}, {"tp"}),
        ("qwen3-30b-a3b", {"tp": 8, "Hq": 30}, {"tp"}),
        ("qwen3-30b-a3b", {"tp": 128, "Hq": 128, "Hkv": 4, "V": 65}, set()),
        ("qwen3-30b-a3b", {"tp": 128, "Hq": 128, "Hkv": 4, "V": 64}, {"tp"}),
        ("qwen3-30b-a3b", {"pp": 5}, {"pp"}),
        ("qwen3-30b-a3b", {"T": 4096, "batch": 3}, {"T"}),
        ("qwen3-30b-a3b", {"T": 4096, "batch": 2}, set()),
        ("qwen3-8b", {"tp": 4, "dp": 5, "ep": 1, "pp": 9}, set()),
        ("qwen3-8b", {"tp": 4, "dp": 2, "ep": 8}, {"ep"}),
        ("qwen3-8b", {"tp": 4, "I": 12289}, {"tp"}),
    ],
)
def test_joint_predicates_accept_and_reject_candidate_environments(model, overrides, invalid):
    # Deliberate config overrides isolate divisibility, beyond the ordinary control domain.
    ir = IR.model_validate_json((ROOT / f"fixtures/{model}/ir.json").read_text())
    env = environment(ir, **overrides)
    rejected = {
        key
        for key, spec in ir.params.items()
        if spec.constraint is not None and evaluate_reference(spec.constraint, env) == 0
    }
    assert rejected == invalid


@pytest.mark.parametrize("version", ["1.0", "not-a-version", "0.32.dev0"])
def test_version_mismatch_is_one_blocking_schema_valid_diagnostic(version, tmp_path):
    extract = Extract.model_validate_json((MOE / "extract.json").read_text())
    result = lower(extract, vllm_version=version)
    assert not coverage_gate(result.triage)
    assert not result.ir.nodes and not result.parameter_shapes
    assert len(result.triage.items) == 1
    item = result.triage.items[0]
    assert item.kind == "version_mismatch"
    assert item.vllm_version == version
    assert item.expected_ranges == [">=0.32,<0.33"]
    assert item.message
    wire = result.triage.model_dump(mode="json", by_alias=True)
    Draft202012Validator(json.loads((ROOT / "schemas/triage.schema.json").read_text())).validate(
        wire
    )
    assert Triage.model_validate(wire) == result.triage
    result.write(tmp_path)
    before = (tmp_path / "triage.json").read_bytes()
    lower(extract, vllm_version=version).write(tmp_path)
    assert (tmp_path / "triage.json").read_bytes() == before
