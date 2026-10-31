"""Resolve every committed annotation and cover all experiment-observed swaps."""

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from ruamel.yaml import YAML

from inferatlas.annotations import (
    SourceTree,
    apply_annotations,
    check_annotations,
    load_annotations,
)
from inferatlas.schema import IR, Annotation, Annotations

ROOT = Path(__file__).resolve().parents[2]


def test_committed_yaml_matches_shared_exported_schemas():
    path = ROOT / "annotations/a2/qwen3-moe.yaml"
    data = YAML(typ="safe").load(path.read_text())
    assert any(item.get("anchors") for item in data)
    list_schema = json.loads((ROOT / "schemas/annotations.schema.json").read_text())
    Draft202012Validator(list_schema).validate(data)
    item_schema = json.loads((ROOT / "schemas/annotation.schema.json").read_text())
    validator = Draft202012Validator(item_schema)
    for item in data:
        validator.validate(item)
    shared = Annotations.model_validate(data).root
    loaded = load_annotations(path)
    assert loaded == sorted(shared, key=lambda annotation: annotation.id)
    assert all(type(annotation) is Annotation for annotation in loaded)


@pytest.fixture(scope="module")
def source_trees():
    default = ROOT / "experiments/a2_qwen3moe/src"
    if not default.exists():
        default = ROOT.parents[1] / "experiments/a2_qwen3moe/src"
    source_root = Path(os.environ.get("INFERATLAS_SRC", default))
    if not (source_root / "vllm-ascend").exists():
        pytest.skip("Set INFERATLAS_SRC to the directory holding the pinned upstream clones")
    snapshot = json.loads((ROOT / "fixtures/qwen3-30b-a3b/ir.json").read_text())["snapshot"]
    return {
        "vllm": SourceTree(source_root / "vllm", "vllm", snapshot["vllm"]["commit"]),
        "vllm_ascend": SourceTree(
            source_root / "vllm-ascend", "vllm-ascend", snapshot["ascend"]["commit"]
        ),
    }


@pytest.fixture(scope="module")
def checked_annotations(source_trees):
    annotations = load_annotations(ROOT / "annotations")
    valid, stale = check_annotations(annotations, source_trees)
    assert stale == []
    assert len(valid) == len(annotations)
    return valid


def test_all_committed_annotations_resolve_and_validate(checked_annotations):
    schema = json.loads((ROOT / "schemas/annotation.schema.json").read_text())
    validator = Draft202012Validator(schema)
    for item in checked_annotations:
        validator.validate(item.annotation.model_dump(mode="json"))
        assert item.annotation.reviewed_by == "pending"
        assert item.annotation.note
        assert all(src.commit and src.line > 0 for src in item.src)
    assert {item.annotation.level for item in checked_annotations} == {
        "KERNEL",
        "LAYOUT",
        "SEMANTIC",
    }


def test_every_experiment_swap_is_covered(checked_annotations):
    experiment = json.loads(
        (ROOT / "experiments/a2_qwen3moe/out/exp3_ascend_meta_A2.json").read_text()
    )
    swaps = experiment["compare"]["layer0_class_swaps"]
    assert len(swaps) == 12
    targets = {item.annotation.target for item in checked_annotations}
    assert {swap["ascend"] for swap in swaps.values()} <= targets
    assert set(experiment["build"]["top_level_classes"].values()) & targets == {
        cls
        for cls in experiment["build"]["top_level_classes"].values()
        if cls.startswith("vllm_ascend.")
    }


def test_all_twelve_fixture_swap_nodes_can_receive_real_annotations(checked_annotations):
    ir = IR.model_validate_json((ROOT / "fixtures/qwen3-30b-a3b/ir.json").read_text())
    # Select the ordinary non-NZ/non-custom-parallel lane without a Python DSL evaluator.
    suffixes = {
        "apply_rotary",
        "gate",
        "rms_norm",
        "rotary",
        "moe_runner",
        "routed_experts",
        "unquantized_experts",
        "linear.qkv.kernel",
        "linear.row.kernel",
    }
    selected = {
        item.annotation.target: item
        for item in checked_annotations
        if item.annotation.id.removeprefix("a2.qwen3_moe.") in suffixes
    }
    bindings = {}
    for node in ir.nodes:
        if node.ascend is not None:
            bindings[node.id] = selected[node.ascend.src[0].symbol].annotation.id
    assert len(bindings) == 12
    applied = apply_annotations(
        ir, checked_annotations, bindings=bindings, active_ids=set(bindings.values())
    )
    for node in applied.nodes:
        if node.id in bindings:
            assert node.ascend.annotation_id == bindings[node.id]
            assert node.ascend.src[0].repo == "vllm-ascend"
            assert node.ascend.src[0].commit == ir.snapshot.ascend.commit


def test_patch_annotations_are_runtime_detected_origins(checked_annotations):
    experiment = json.loads(
        (ROOT / "experiments/a2_qwen3moe/out/exp2_ascend_overlay_A2.json").read_text()
    )
    origins = experiment["diff"]["by_origin"]
    patches = [item for item in checked_annotations if "PATCH" in item.annotation.tag]
    assert patches
    for item in patches:
        assert item.src[0].file.removeprefix("vllm_ascend/") in origins
    attention = next(item for item in patches if item.annotation.id.endswith("patch.attention"))
    assert (
        "vllm.model_executor.models.qwen3_moe.Qwen3MoeAttention.forward"
        in origins[attention.src[0].file.removeprefix("vllm_ascend/")]
    )
    assert attention.annotation.level == "KERNEL"
    gate = next(item for item in checked_annotations if item.annotation.id.endswith(".gate"))
    assert gate.annotation.level == "SEMANTIC"


@pytest.mark.parametrize(
    "suffix,required",
    [
        (
            suffix,
            {
                "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.process_weights_after_loading",
                "vllm_ascend.utils.maybe_trans_nz",
                "vllm_ascend.utils._should_trans_nz",
            },
        )
        for suffix in [
            "linear.qkv.kernel",
            "linear.qkv.layout",
            "linear.row.kernel",
            "linear.row.layout",
        ]
    ]
    + [
        ("rotary", {"vllm_ascend.ops.rotary_embedding._record_cos_and_sin_cache_interleaved"}),
        (
            "linear.row.parallel",
            {
                "vllm_ascend.ops.linear_op._get_row_parallel_op",
                "vllm_ascend.ops.linear_op.get_parallel_op",
                "vllm_ascend.ops.linear_op.OProjRowParallelOp",
            },
        ),
        ("unquantized_experts", {"vllm_ascend.utils.maybe_trans_nz"}),
    ],
)
def test_notes_hash_the_implementations_they_describe(checked_annotations, suffix, required):
    item = next(
        item for item in checked_annotations if item.annotation.id == "a2.qwen3_moe." + suffix
    )
    assert required <= {anchor.target for anchor in item.annotation.anchors}
    assert required | {item.annotation.target} <= {ref.symbol for ref in item.src}


def test_qkv_has_no_custom_parallel_lane(checked_annotations):
    notes = {
        item.annotation.id.removeprefix("a2.qwen3_moe."): item.annotation
        for item in checked_annotations
    }
    assert "linear.qkv.parallel" not in notes
    for lane in ["kernel", "layout"]:
        assert "custom_parallel" not in notes["linear.qkv." + lane].applies_when
    assert "FlashComm" not in notes["linear.row.parallel"].note
    expert = notes["unquantized_experts"].note
    assert "enable_fused_mc2" in expert and "MegaMoE" in expert and "maybe_trans_nz" in expert


@pytest.mark.parametrize("enabled", [False, True])
def test_pinned_parallel_selector_never_customizes_qkv(source_trees, enabled):
    # Execute only the pure selectors, with all custom TP switches enabled/disabled.
    namespace = {
        "oproj_tp_enable": lambda: enabled,
        "mlp_tp_enable": lambda: enabled,
        "is_moe_layer": lambda prefix: False,
        "DSV4OProjColumnParallelOp": lambda layer: "wo_a",
        "MLPColumnParallelOp": lambda layer: "gate_up_proj",
        "OProjRowParallelOp": lambda layer: "o_proj",
    }
    tree = source_trees["vllm_ascend"]
    for name in ["_get_column_parallel_op", "_get_row_parallel_op"]:
        source = tree.resolve("vllm_ascend.ops.linear_op." + name).source
        exec("from __future__ import annotations\n" + source, namespace)
    column = namespace["_get_column_parallel_op"]
    row = namespace["_get_row_parallel_op"]
    assert column("model.layers.0.self_attn.qkv_proj", None) is None
    assert column("model.layers.0.self_attn.wo_a", None) == ("wo_a" if enabled else None)
    assert column("model.layers.0.mlp.gate_up_proj", None) == ("gate_up_proj" if enabled else None)
    assert row("model.layers.0.self_attn.o_proj", None) == ("o_proj" if enabled else None)
