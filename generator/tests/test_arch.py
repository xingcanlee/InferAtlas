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
        n.target == bands(arch)[0].id + "/" + generic.id and n.triage_kind == "unknown_layer_class"
        for n in arch.notes
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
            tensors.extend([*sub.io.inputs, *sub.io.outputs, *sub.io.internals])
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


def test_ruled_component_suppresses_obsolete_unknown_class_triage(facts):
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
    assert not any(n.triage_kind == "unknown_layer_class" for n in arch.notes)


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


@pytest.mark.parametrize("slug", ["llama-4-scout", "minimax-m3", "deepseek-v4-flash"])
def test_nested_and_missing_numeric_config_never_blocks_structure(facts, slug):
    from inferatlas.resolve import get_model

    facts.config = get_model(slug).config
    result = lower(facts)
    arch = emit_arch(facts, result.ir, result.triage, model_id=slug, family="hybrid")
    assert bands(arch)
    assert "T" in arch.symbols and "tp" in arch.symbols


def test_v11_targets_residuals_and_informative_io(facts):
    arch = architecture(facts)
    for band in bands(arch):
        for sub in band.block.sublayers:
            assert not sub.id.startswith("model.")
            if sub.role == "residual_add":
                assert sub.component is sub.module is None
    attention = bands(arch)[0].block.sublayers[1]
    assert next(t for t in attention.io.internals if t.name == "v").shape == ["T", "4*4/tp"]
    assert {t.name for t in attention.io.internals} >= {"q", "k", "v", "o", "kv_lora_latent"}
    moe = bands(arch)[-1].block.sublayers[4]
    tensors = {t.name: t for t in moe.io.internals}
    assert tensors["router_logits"].shape == ["T", "E"]
    assert tensors["topk_ids"].shape == ["T", "K"]
    assert tensors["topk_ids"].dtype == "int64"
    assert not any(n.text.startswith("{") for n in arch.notes)


def test_sliding_none_and_window_layer_direction(facts):
    facts.config.update(sliding_window=128, use_sliding_window=True, max_window_layers=1)
    module = next(m for m in facts.modules if m.path == "model.layers.0.self_attn")
    module.attrs["sliding_window"] = None
    arch = architecture(facts)
    assert bands(arch)[0].block.sublayers[1].component.variant == "mla"
    assert bands(arch)[1].block.sublayers[1].component.variant == "sliding_window"


def test_tp_parameter_shapes_match_observed_rank_and_recompute(runtime):
    facts = Extract.model_validate(read_json(ROOT / "fixtures/qwen3-30b-a3b/extract-tp4.json"))
    arch = architecture(facts)
    env = {name: symbol.value for name, symbol in arch.symbols.items()}
    modules = {m.path: m for m in facts.modules}
    for stage in arch.stages:
        refs = (
            [s.module for b in stage.bands for s in b.block.sublayers]
            if stage.kind == "stack"
            else [stage.module]
        )
        for ref in filter(None, refs):
            for param in ref.params:
                path, _, name = (ref.path + "." + param.name).rpartition(".")
                observed = next(p for p in modules[path].params if p.name == name)
                assert runtime.many(param.shape, env) == observed.shape


@pytest.mark.parametrize("style", ["post_norm", "sandwich"])
def test_residual_styles_emit_null_junctions_in_dataflow_order(facts, monkeypatch, style):
    from inferatlas.arch import structure_rules

    rules = structure_rules()
    rule = rules["vllm.model_executor.models.deepseek_v2.DeepseekV2DecoderLayer"]
    rule.residual = style
    if style == "post_norm":
        rule.order = ["self_attn", "input_layernorm", "mlp", "post_attention_layernorm"]
    else:
        rule.order = [
            "input_layernorm",
            "self_attn",
            "post_attention_layernorm",
            "pre_mlp_norm",
            "mlp",
            "post_mlp_norm",
        ]
        for name in ("pre_mlp_norm", "post_mlp_norm"):
            facts.modules.append(
                ExtractModule(
                    path="model.layers.0." + name,
                    class_name="vllm.model_executor.layers.layernorm.RMSNorm",
                )
            )
    monkeypatch.setattr("inferatlas.arch.structure_rules", lambda: rules)
    subs = bands(architecture(facts))[0].block.sublayers
    junctions = [i for i, sub in enumerate(subs) if sub.role == "residual_add"]
    assert len(junctions) == 2
    for i in junctions:
        assert subs[i].component is subs[i].module is None
        assert subs[i - 1].role == (
            "norm" if style == "sandwich" else "mixer" if i == junctions[0] else "ffn"
        )


def test_ep_shapes_declare_and_use_parallel_symbols(facts):
    facts.parallel.ep = True
    facts.parallel.tp = 4
    arch = architecture(facts)
    assert {"tp", "dp", "ep"} <= arch.symbols.keys()
    moe = bands(arch)[-1].block.sublayers[4]
    assert next(t for t in moe.io.internals if t.name == "routed_expert_input").shape == [
        "T*dp*K/ep",
        "H",
    ]


def test_shared_parameter_free_leaf_does_not_split_first_band(facts):
    facts.modules.append(
        ExtractModule(path="model.layers.0.mlp.act", class_name="vendor.SharedActivation")
    )
    assert [b.layers for b in bands(architecture(facts))] == [(0, 1), (2, 4)]


def test_v12_boundaries_generic_slots_and_note_deduplication(facts):
    for m in facts.modules:
        if m.path.endswith("self_attn"):
            m.class_name = "vendor.UnknownAttention"
    arch = architecture(facts)
    for band in bands(arch):
        assert [s.role for s in band.block.sublayers].count("residual_add") == 2
        assert band.block.sublayers[1].role == "mixer"
        ffn = band.block.sublayers[4]
        assert [t.name for t in ffn.io.outputs] == ["hidden_states"]
        assert {t.name for t in ffn.io.internals} >= {"gate", "up", "down"}
    generic = [n for n in arch.notes if n.text.startswith("Generic component")]
    assert len(generic) == 1
    unknown = [n for n in arch.notes if n.triage_kind == "unknown_layer_class"]
    assert len(unknown) == 1 and "vendor.UnknownAttention" in unknown[0].text


def test_stage_order_ignores_head_first_registration_and_sorts_stack_norms(facts):
    root = next(m for m in facts.modules if m.path == "")
    root.attrs["child_order"] = ["lm_head", "model"]
    parent = next(m for m in facts.modules if m.path == "model")
    parent.attrs["child_order"] = ["norm", "layers", "embed_tokens"]
    facts.modules.append(ExtractModule(path="model.extra", class_name="vendor.Extra"))
    assert [s.kind for s in architecture(facts).stages] == [
        "embedding",
        "stack",
        "norm",
        "lm_head",
        "other",
    ]


def test_vision_mlp_uses_fc_width_and_patch_tokens(facts):
    facts.config["vision_config"] = {"hidden_size": 1408, "intermediate_size": 5632}
    facts.modules.extend(
        [
            ExtractModule(path="vision", class_name="vendor.Vision"),
            ExtractModule(path="vision.layers", class_name="torch.nn.modules.container.ModuleList"),
            ExtractModule(
                path="vision.layers.0",
                class_name="vllm.model_executor.models.mllama4.Llama4VisionEncoderLayer",
            ),
            ExtractModule(
                path="vision.layers.0.mlp",
                class_name="vllm.model_executor.models.mllama4.Llama4VisionMLP",
            ),
            ExtractModule(
                path="vision.layers.0.mlp.fc1",
                class_name="vendor.Linear",
                attrs={"input_size": 1408, "output_size": 5632},
            ),
            ExtractModule(
                path="vision.layers.0.mlp.fc2",
                class_name="vendor.Linear",
                attrs={"input_size": 5632, "output_size": 1408},
            ),
        ]
    )
    arch = architecture(facts)
    stage = next(s for s in arch.stages if s.id == "vision.layers")
    mlp = stage.bands[0].block.sublayers[0]
    assert mlp.io.inputs[0].shape == mlp.io.outputs[0].shape == ["P", "1408"]
    assert arch.symbols["P"].description.startswith("Vision")


def test_deepseek_v4_hyper_connection_io_and_compression(facts):
    facts.config["hc_mult"] = 4
    for m in facts.modules:
        if m.path == "model.layers.0":
            m.class_name = "vllm.models.deepseek_v4.cpu.model.DeepseekV4DecoderLayer"
        if m.path == "model.layers.0.self_attn":
            m.path = "model.layers.0.attn"
            m.class_name = "vllm.models.deepseek_v4.cpu.cpu_sparse.DeepseekV4CPUAttention"
            m.attrs["compress_ratio"] = 4
        if m.path == "model.layers.0.mlp":
            m.path = "model.layers.0.ffn"
    facts.modules.append(
        ExtractModule(path="model.layers.0.attn.indexer_rotary_emb", class_name="vendor.Rope")
    )
    for name in ("mhc_pre", "mhc_post", "mhc_fused_post_pre"):
        facts.modules.append(
            ExtractModule(path="model.layers.0." + name, class_name="vllm.mhc." + name)
        )
    arch = architecture(facts)
    block = bands(arch)[0].block
    assert block.residual == "hyper_connection"
    names = [s.id for s in block.sublayers]
    assert names.index("mhc_pre_attn") < names.index("attn") < names.index("mhc_post_attn")
    assert names.index("mhc_pre_ffn") < names.index("ffn") < names.index("mhc_post_ffn")
    assert all(s.io.inputs and s.io.outputs for s in block.sublayers if s.id.startswith("mhc_"))
    attn = next(s for s in block.sublayers if s.id == "attn")
    assert attn.component.extras["compress_ratio"] == 4
    assert attn.component.extras["indexer"] is False
    assert [t.name for t in attn.io.outputs] == ["hidden_states"]


def test_ruled_component_hides_building_block_warnings(facts):
    facts.modules.append(
        ExtractModule(
            path="model.layers.0.self_attn.linear",
            class_name="vllm.model_executor.layers.linear.ColumnParallelLinear",
        )
    )
    arch = architecture(facts)
    assert not any(n.severity == "warn" for n in arch.notes)


def test_extractor_cache_key_tracks_worker_source(tmp_path, monkeypatch):
    namespace = runpy.run_path(ROOT / "generator/scripts/build_catalog_arch.py")
    key = namespace["extractor_hash"]
    monkeypatch.setitem(key.__globals__, "ROOT", tmp_path)
    source = tmp_path / "generator/inferatlas/extract/worker.py"
    source.parent.mkdir(parents=True)
    source.write_text("first")
    before = key()
    assert key() == before
    source.write_text("second")
    assert key() != before


def test_expert_bias_uses_same_tp_axis_and_expert_symbol_as_weight(facts):
    facts.modules.append(
        ExtractModule(
            path="model.layers.2.mlp.experts",
            class_name="vllm.model_executor.models.gpt_oss.GptOssRoutedExperts",
            params=[
                ExtractParam(name="w13_weight", shape=[8, 24, 16], dtype="bfloat16"),
                ExtractParam(name="w13_bias", shape=[8, 24], dtype="bfloat16"),
                ExtractParam(name="w2_bias", shape=[8, 16], dtype="bfloat16"),
            ],
        )
    )
    band = next(b for b in bands(architecture(facts)) if b.layers[0] == 2)
    ref = next(s.module for s in band.block.sublayers if s.id == "mlp")
    params = {p.name: p.shape for p in ref.params}
    assert params["experts.w13_bias"] == params["experts.w13_weight"][:2] == ["E", "24/tp"]
    assert params["experts.w2_bias"] == ["E", "16"]
