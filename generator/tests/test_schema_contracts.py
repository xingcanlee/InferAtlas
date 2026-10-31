"""Contract boundaries, deterministic exports and facts used by the seven dependent tasks."""

import copy
import hashlib
import json
import math
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from ruamel.yaml import YAML

from inferatlas.schema import IR, Catalog, Extract, View
from inferatlas.schema.export import MODELS, export, schema_text

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures"
MOE = FIXTURES / "qwen3-30b-a3b"


def read(path):
    return json.loads(path.read_text())


def fixture_schema(path):
    if path.stem.startswith("extract"):
        return "extract"
    return path.stem


@pytest.mark.parametrize(
    "path", sorted(FIXTURES.rglob("*.json")), ids=lambda p: str(p.relative_to(FIXTURES))
)
def test_schema_fixture(path):
    name = fixture_schema(path)
    schema = read(ROOT / "schemas" / f"{name}.schema.json")
    Draft202012Validator.check_schema(schema)
    value = read(path)
    Draft202012Validator(schema).validate(value)
    instance = MODELS[name].model_validate(value)
    assert MODELS[name].model_validate(instance.model_dump(mode="json", by_alias=True)) == instance


def test_schema_catalog():
    data = YAML(typ="safe").load(ROOT / "catalog.yaml")
    Draft202012Validator(read(ROOT / "schemas/catalog.schema.json")).validate(data)
    catalog = Catalog.model_validate(data)
    assert len(catalog.models) == 3
    assert {d.id for d in catalog.devices} == {"A2", "A3", "A5", "310P"}
    assert {p.device for p in catalog.hardware_presets} == {"A2", "A3", "A5", "310P", "H100"}
    for model in catalog.models:
        digest = hashlib.sha256(
            json.dumps(model.config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        assert model.config_rev == f"inline:sha256:{digest}"
    presets = {p.id for p in catalog.hardware_presets}
    assert all(d.hardware_preset in presets for d in catalog.devices)
    for preset in catalog.hardware_presets:
        assert set(preset.peak_tflops) == {"fp32", "bf16", "fp16", "fp8", "int8", "int4"}
        if any(v is None for v in preset.peak_tflops.values()):
            assert any("TODO" in note for note in preset.notes)


def test_schema_exports_are_current_and_deterministic(tmp_path):
    export(tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    export(tmp_path)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    assert sorted(p.name for p in (ROOT / "schemas").glob("*.schema.json")) == sorted(first)
    for name in MODELS:
        assert (ROOT / "schemas" / f"{name}.schema.json").read_text() == schema_text(name)
        assert (tmp_path / f"{name}.schema.json").read_text() == schema_text(name)


def test_schema_property_documentation():
    for name in MODELS:
        schema = json.loads(schema_text(name))
        for definition in [schema, *schema.get("$defs", {}).values()]:
            for field in definition.get("properties", {}).values():
                assert field.get("description"), f"{name}: undocumented field {field}"


def test_schema_aliases_and_unknown_fields():
    module = Extract.model_validate(read(MOE / "extract.json")).modules[0]
    assert "class" in module.model_dump(by_alias=True)
    assert "class_name" not in module.model_dump(by_alias=True)
    broken = read(MOE / "ir.json")
    broken["nodes"][0]["weight_bytes"] = 123
    with pytest.raises(ValidationError):
        IR.model_validate(broken)
    broken = read(MOE / "extract.json")
    broken["surprise"] = "silently ignored bugs are not a contract"
    with pytest.raises(ValidationError):
        Extract.model_validate(broken)


def test_schema_ir_requires_valid_references_and_acyclic_parents():
    original = read(MOE / "ir.json")
    for mutation in ["duplicate", "endpoint", "cycle", "template"]:
        broken = copy.deepcopy(original)
        if mutation == "duplicate":
            broken["nodes"].append(broken["nodes"][0])
        elif mutation == "endpoint":
            broken["edges"][0]["to"] = "missing"
        elif mutation == "cycle":
            broken["nodes"][0]["parent"] = broken["nodes"][0]["id"]
        else:
            broken["groups"][0]["template"] = "missing"
        with pytest.raises(ValidationError):
            IR.model_validate(broken)


def test_schema_layout_routes_and_ir_references():
    ir = IR.model_validate(read(MOE / "ir.json"))
    ids = {n.id for n in ir.nodes}
    view = View.model_validate(read(MOE / "view.json"))
    assert {v.id for v in view.views} == {"L0", "L1", "L2:attn", "L2:moe"}
    for graph in view.views:
        assert all(n.ir_node in ids for n in graph.nodes)
        assert set(graph.collapsed_of) <= ids
        assert graph.nodes == sorted(graph.nodes, key=lambda n: n.id)
    for mutation in ["diagonal", "off_grid", "missing", "outside"]:
        broken = read(MOE / "view.json")
        graph = broken["views"][0]
        if mutation == "diagonal":
            graph["edges"][0]["points"][1][1] += 4
        elif mutation == "off_grid":
            graph["nodes"][0]["x"] += 1
        elif mutation == "missing":
            graph["edges"][0]["to"] = "missing"
        else:
            graph["nodes"][0]["w"] = 100000
        with pytest.raises(ValidationError):
            View.model_validate(broken)


def test_schema_fixture_extraction_matches_experiment_facts():
    for tp in [1, 4]:
        filename = "extract.json" if tp == 1 else "extract-tp4.json"
        extract = Extract.model_validate(read(MOE / filename))
        measured = read(ROOT / f"experiments/a2_qwen3moe/out/exp1_vllm_meta_tp{tp}.json")
        modules = {m.path: m for m in extract.modules}
        observed = {
            f"{m.path}.{p.name}": p.shape
            for m in extract.modules
            for p in m.params
            if m.path.startswith("model.layers.0.")
        }
        assert observed == {
            name: p["shape"] for name, p in measured["build"]["layer0_params"].items()
        }
        assert extract.fixture is True
        assert len(extract.layer_types) == 48
        assert len(extract.ascend.custom_ops) == 31
        assert len(extract.ascend.patched) == 130
        assert (
            extract.ascend.patched["vllm.config.vllm.VllmConfig._validate_v2_model_runner"]
            == "vllm_ascend/mrv2_utils.py"
        )
        assert (
            sum(
                m.ascend is not None
                for m in extract.modules
                if m.path.startswith("model.layers.0.")
            )
            == 12
        )
        assert list(modules) == sorted(modules)
        assert all(m.params == sorted(m.params, key=lambda p: p.name) for m in extract.modules)
        per_layer = sum(math.prod(shape) for shape in observed.values())
        endpoints = sum(
            math.prod(p.shape)
            for key in ["model.embed_tokens", "model.norm", "lm_head"]
            for p in modules[key].params
        )
        assert 2 * (48 * per_layer + endpoints) == measured["build"]["param_bytes"]


def test_schema_oracle_preserves_verified_totals_and_replication():
    cases = read(MOE / "eval.json")["cases"]
    assert [c["expected"]["memory.weights"] for c in cases] == [61064245248, 15285252096]
    assert cases[1]["expected"]["memory.weights"] != cases[0]["expected"]["memory.weights"] / 4
    for case in cases:
        assert case["env"]["T"] == 4096
        assert case["expected"]["nodes.moe.gate.weight_bytes"] == 524288
    ir = IR.model_validate(read(MOE / "ir.json"))
    for node in ir.nodes:
        assert isinstance(node.weight_bytes, str)
        for slot in [*node.inputs, *node.out]:
            assert all(isinstance(dim, str) for dim in slot.shape)
    assert sum(n.ascend is not None for n in ir.nodes) == 12
    assert ir.nodes == sorted(ir.nodes, key=lambda n: n.id)
    assert ir.edges == sorted(ir.edges, key=lambda e: e.id)


def test_schema_matrix_is_complete_and_statuses_consistent():
    index = read(FIXTURES / "index.json")
    expected = {(v["id"], m["id"], "A2") for v in index["versions"] for m in index["models"]}
    assert {(c["version"], c["model"], c["device"]) for c in index["cells"]} == expected
    assert {c["status"] for c in index["cells"]} == {"ok", "triage", "missing"}
    for cell in index["cells"]:
        assert (cell["data_path"] is None) == (cell["status"] == "missing")
        assert (cell["triage_count"] > 0) == (cell["status"] == "triage")


def test_schema_cli_stub_entry_point():
    from typer.testing import CliRunner

    from inferatlas.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout == "0.1.0\n"
    assert CliRunner().invoke(app, ["--help"]).exit_code == 0
