"""Structure survives missing rules, heterogeneity and numeric review notes."""

import runpy

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from t6_runtime import node_backend as node_backend
from t6_runtime import runtime as runtime

from inferatlas.arch import emit_arch, validate_arch_identity
from inferatlas.lower import lower
from inferatlas.schema import Arch, Extract, Triage
from inferatlas.schema.arch import ArchStackStage
from inferatlas.schema.models import ExtractModule, ExtractParam, UnknownLayerClass
from inferatlas.verify.golden import check_arch_golden
from inferatlas.verify.io import ROOT, read_json, wire, write_json

EXAMPLES = ROOT / "generator/tests/fixtures_arch"


@pytest.fixture
def facts():
    return runpy.run_path(ROOT / "generator/scripts/build_arch_fixtures.py")["synthetic_facts"]()


def architecture(facts):
    result = lower(facts)
    return emit_arch(
        facts, result.ir, result.triage, model_id="synthetic-heterogeneous", family="hybrid"
    )


def bands(arch):
    return next(s.bands for s in arch.stages if isinstance(s, ArchStackStage))


@pytest.mark.parametrize("slug", ["qwen3-8b", "qwen3-30b-a3b", "synthetic-heterogeneous"])
def test_frontend_examples_validate_and_are_canonical(slug):
    path = EXAMPLES / f"{slug}.json"
    data = read_json(path)
    Draft202012Validator(read_json(ROOT / "schemas/arch.schema.json")).validate(data)
    arch = Arch.model_validate(data)
    assert wire(arch) == data

    def sorted_keys(value):
        if isinstance(value, dict):
            assert list(value) == sorted(value)
            for child in value.values():
                sorted_keys(child)
        elif isinstance(value, list):
            for child in value:
                sorted_keys(child)

    sorted_keys(data)
    assert arch.input.shape == ["T"] and arch.output.shape == ["T", "V"]


@pytest.mark.parametrize("slug", ["qwen3-8b", "qwen3-30b-a3b"])
@pytest.mark.parametrize("device", ["A2", "H100"])
def test_qwen_goldens_have_full_stack_and_semantics(slug, device):
    arch = Arch.model_validate(read_json(ROOT / "golden" / slug / device / "arch.json"))
    assert arch.device == device
    (band,) = bands(arch)
    assert band.layers == (0, 47 if "30b" in slug else 35)
    assert band.count == (48 if "30b" in slug else 36)
    assert band.block.residual == "pre_norm"
    assert [s.role for s in band.block.sublayers] == [
        "norm",
        "mixer",
        "residual_add",
        "norm",
        "ffn",
        "residual_add",
    ]
    attention = band.block.sublayers[1].component
    assert attention.kind == "attention" and attention.variant == "gqa"
    assert attention.qk_norm and attention.rope.theta == 1_000_000
    ffn = band.block.sublayers[4].component
    assert ffn.kind == ("moe" if "30b" in slug else "mlp")
    if ffn.kind == "moe":
        assert (ffn.experts, ffn.top_k, ffn.shared_experts) == (128, 8, 0)
    if device == "A2":
        assert any(s.module and s.module.status == "ascend_override" for s in band.block.sublayers)
        assert wire(arch) == read_json(EXAMPLES / f"{slug}.json")


def test_dense_prefix_and_moe_bands_with_mla(facts):
    arch = architecture(facts)
    assert wire(arch) == read_json(EXAMPLES / "synthetic-heterogeneous.json")
    dense, moe = bands(arch)
    assert (dense.layers, dense.count, moe.layers, moe.count) == ((0, 1), 2, (2, 4), 3)
    attention = dense.block.sublayers[1].component
    assert attention.variant == "mla" and attention.extras["kv_lora_rank"] == 4
    assert dense.block.sublayers[4].component.kind == "mlp"
    ffn = moe.block.sublayers[4].component
    assert ffn.kind == "moe" and ffn.shared_experts == 1 and ffn.shared.intermediate == 12
    assert ffn.router.scoring == "sigmoid"


def test_generic_parameter_io_children_and_targeted_notes(facts):
    facts.modules.extend(
        [
            ExtractModule(
                path="model.layers.0.adapter",
                class_name="vendor.Unrecognized",
                params=[ExtractParam(name="weight", shape=[7, 16], dtype="bfloat16")],
            ),
            ExtractModule(
                path="model.layers.0.adapter.child",
                class_name="vendor.Child",
                params=[ExtractParam(name="bias", shape=[7], dtype="bfloat16")],
            ),
        ]
    )
    arch = architecture(facts)
    assert [b.layers for b in bands(arch)] == [(0, 0), (1, 1), (2, 4)]
    generic = next(s for s in bands(arch)[0].block.sublayers if s.id.endswith("adapter"))
    assert generic.component.kind == "generic" and generic.module.status == "generic"
    assert generic.component.children[0].class_name == "vendor.Child"
    assert {p.name: p.shape for p in generic.module.params} == {
        "weight": ["7", "16"],
        "child.bias": ["7"],
    }
    assert generic.io.inputs[0].shape == ["T", "16"]
    assert generic.io.outputs[0].shape == ["T", "7"]
    assert any(
        n.target == generic.id and n.triage_kind == "unknown_layer_class" for n in arch.notes
    )


def test_unknown_named_attention_remains_generic_and_unknown_io_is_empty(facts):
    module = next(m for m in facts.modules if m.path == "model.layers.0.self_attn")
    module.class_name = "vendor.Attention"
    arch = architecture(facts)
    sub = bands(arch)[0].block.sublayers[1]
    assert sub.component.kind == "generic" and sub.io.inputs == sub.io.outputs == []


def test_linear_sliding_and_mqa_variants_split_actual_bands(facts):
    facts.layer_types[0] = "linear_attention"
    facts.layer_types[1] = "sliding_attention"
    arch = architecture(facts)
    assert [b.block.sublayers[1].component.variant for b in bands(arch)] == [
        "linear",
        "sliding_window",
        "mla",
    ]
    facts.config["num_key_value_heads"] = 1
    for m in facts.modules:
        if m.path.endswith("self_attn"):
            m.class_name = "vllm.model_executor.models.qwen3.Qwen3Attention"
    arch = architecture(facts)
    assert bands(arch)[-1].block.sublayers[1].component.variant == "mqa"


def test_generic_sequential_boundaries_and_subtree_parameters(facts):
    facts.modules.extend(
        [
            ExtractModule(path="adapter", class_name="torch.nn.modules.container.Sequential"),
            ExtractModule(
                path="adapter.0",
                class_name="vendor.First",
                params=[ExtractParam(name="weight", shape=[5, 16], dtype="bfloat16")],
            ),
            ExtractModule(
                path="adapter.1",
                class_name="vendor.Last",
                params=[ExtractParam(name="weight", shape=[16, 5], dtype="bfloat16")],
            ),
        ]
    )
    stage = next(s for s in architecture(facts).stages if s.id == "adapter")
    assert stage.io.inputs[0].shape == ["T", "16"]
    assert stage.io.outputs[0].shape == ["T", "16"]
    assert [p.name for p in stage.module.params] == ["0.weight", "1.weight"]


def test_shapes_use_shared_node_evaluator(facts, runtime):
    arch = architecture(facts)
    tensors = [arch.input, arch.output]
    for band in bands(arch):
        for sub in band.block.sublayers:
            tensors.extend([*sub.io.inputs, *sub.io.outputs])
    env = {k: s.value for k, s in arch.symbols.items()} | {"T": 23}
    assert runtime.many(arch.output.shape, env) == [23, 32]
    for tensor in tensors:
        dimensions = runtime.many(tensor.shape, env)
        assert all(d >= 0 and int(d) == d for d in dimensions)


def test_module_and_parameter_order_does_not_change_output(facts):
    before = architecture(facts).model_dump_json()
    facts.modules.reverse()
    for module in facts.modules:
        module.params.reverse()
    assert architecture(facts).model_dump_json() == before


def test_architecture_rejects_identity_duplicate_targets_and_unbound_shapes(facts):
    result = lower(facts)
    result.ir.snapshot = result.ir.snapshot.model_copy(update={"device": "A3"})
    with pytest.raises(ValueError, match="snapshot identity"):
        emit_arch(facts, result.ir, result.triage, model_id="synthetic", family="hybrid")
    data = wire(architecture(facts))
    data["input"]["shape"] = ["undefined_symbol"]
    with pytest.raises(ValidationError, match="unbound"):
        Arch.model_validate(data)
    data = wire(architecture(facts))
    data["stages"].append(data["stages"][0])
    with pytest.raises(ValidationError, match="duplicate"):
        Arch.model_validate(data)
    facts.modules.append(facts.modules[0])
    with pytest.raises(ValueError, match="duplicate"):
        architecture(facts)


def test_architecture_golden_diff_is_field_specific(facts, tmp_path):
    arch = architecture(facts)
    write_json(tmp_path / "arch.json", arch)
    assert check_arch_golden(tmp_path, arch) == []
    bands(arch)[0].block.sublayers[1].component.head_dim += 1
    differences = check_arch_golden(tmp_path, arch)
    assert len(differences) == 1 and differences[0].path.endswith("/component/head_dim")


def test_unknown_root_tree_needs_no_model_specific_config(facts):
    facts.config = {}
    facts.modules = [
        ExtractModule(path="", class_name="vendor.UnknownModel"),
        ExtractModule(
            path="projection",
            class_name="vendor.UnknownLinear",
            params=[ExtractParam(name="weight", shape=[9, 4], dtype="float32")],
        ),
    ]
    arch = architecture(facts)
    assert arch.stages[0].component.kind == "generic"
    assert arch.stages[0].io.outputs[0].shape == ["T", "9"]


def test_triage_symbol_maps_to_visible_component(facts):
    result = lower(facts)
    triage = Triage(
        snapshot=facts.snapshot,
        items=[
            UnknownLayerClass(
                kind="unknown_layer_class",
                symbol="vllm.model_executor.models.deepseek_v2.DeepseekV2MoE",
                used_by=[facts.snapshot.model.hf_id],
            )
        ],
    )
    arch = emit_arch(facts, result.ir, triage, model_id="synthetic", family="hybrid")
    assert any(
        n.target == "model.layers.2.mlp" and n.triage_kind == "unknown_layer_class"
        for n in arch.notes
    )


@pytest.mark.parametrize("partial_extraction", [False, True])
def test_unknown_numeric_family_build_and_site_keep_structure(
    facts, tmp_path, monkeypatch, partial_extraction
):
    from typer.testing import CliRunner

    import inferatlas.cli as cli
    from inferatlas.schema import Index, View

    # Inject this synthetic architecture under a catalog cell to exercise publication.
    facts.snapshot.model = Extract.model_validate(
        read_json(ROOT / "fixtures/qwen3-30b-a3b/extract.json")
    ).snapshot.model
    facts.fixture = False
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES", raising=False)
    monkeypatch.setattr(cli, "resolve_stage", lambda *args: facts.snapshot)

    def extract(snapshot, degree, ep=False):
        if partial_extraction and degree == 4:
            from inferatlas.extract import ExtractionError

            raise ExtractionError(
                "Optional TP construction failed", "NotImplementedError", structured=True
            )
        result = facts.model_copy(deep=True)
        result.parallel.tp = degree
        return result

    monkeypatch.setattr(cli, "extract_stage", extract)
    monkeypatch.setattr(cli, "annotate_stage", lambda ir, extracts, triage: (ir, triage))
    monkeypatch.setattr(cli, "layout_stage", lambda ir, out: write_json(out, View(views=[])))
    directory = tmp_path / "build/pinned/qwen3-30b-a3b"
    result = CliRunner().invoke(
        cli.app, ["build", "--tp", "1", "--tp", "4", "--output", str(directory)]
    )
    assert result.exit_code == 0, result.output or repr(result.exception)
    assert {"arch.json", "ir.json", "view.json", "report.json"} <= {
        p.name for p in directory.iterdir()
    }
    assert not read_json(directory / "report.json")["numeric_verified"]
    assert read_json(directory / "arch.json")["notes"]
    if partial_extraction:
        assert any("tp=4" in n["text"] for n in read_json(directory / "arch.json")["notes"])
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    destination = tmp_path / "data"
    result = CliRunner().invoke(
        cli.app, ["site", "--build-root", str(tmp_path / "build"), "--data-dir", str(destination)]
    )
    assert result.exit_code == 0, result.output or repr(result.exception)
    (cell,) = [
        c for c in Index.model_validate(read_json(destination / "index.json")).cells if c.data_path
    ]
    assert cell.status == "triage" and cell.triage_count > 0
    assert (destination / cell.data_path.removeprefix("data/") / "arch.json").is_file()


def test_published_architecture_source_commit_must_match_snapshot():
    facts = Extract.model_validate(read_json(ROOT / "fixtures/qwen3-30b-a3b/extract.json"))
    arch = Arch.model_validate(read_json(EXAMPLES / "qwen3-30b-a3b.json"))
    validate_arch_identity(arch, facts.snapshot, arch.model.id)
    source = arch.stages[0].module.source
    source.url = source.url.replace(facts.snapshot.ascend.commit, "f" * 40)
    with pytest.raises(ValueError, match="source URL"):
        validate_arch_identity(arch, facts.snapshot, arch.model.id)


def test_nullable_config_and_named_sequential_keep_unknown_structure(facts):
    facts.config.update(text_config=None, head_dim=None, torch_dtype=None)
    facts.modules.append(
        ExtractModule(path="adapter", class_name="torch.nn.modules.container.Sequential")
    )
    facts.modules.append(ExtractModule(path="adapter.named", class_name="vendor.Unknown"))
    arch = architecture(facts)
    stage = next(s for s in arch.stages if s.id == "adapter")
    assert stage.component.kind == "generic" and stage.io.inputs == stage.io.outputs == []
    assert bands(arch)[0].block.sublayers[1].component.head_dim == 8
