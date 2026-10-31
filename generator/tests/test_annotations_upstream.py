"""Resolve every committed annotation and cover all experiment-observed swaps."""

import json
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from inferatlas.annotations import (
    SourceTree,
    apply_annotations,
    check_annotations,
    load_annotations,
)
from inferatlas.schema import IR

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def checked_annotations():
    default = ROOT / "experiments/a2_qwen3moe/src"
    if not default.exists():
        default = ROOT.parents[1] / "experiments/a2_qwen3moe/src"
    source_root = Path(os.environ.get("INFERATLAS_SRC", default))
    if not (source_root / "vllm-ascend").exists():
        pytest.skip("Set INFERATLAS_SRC to the directory holding the pinned upstream clones")
    snapshot = json.loads((ROOT / "fixtures/qwen3-30b-a3b/ir.json").read_text())["snapshot"]
    sources = {
        "vllm": SourceTree(source_root / "vllm", "vllm", snapshot["vllm"]["commit"]),
        "vllm_ascend": SourceTree(
            source_root / "vllm-ascend", "vllm-ascend", snapshot["ascend"]["commit"]
        ),
    }
    annotations = load_annotations(ROOT / "annotations")
    valid, stale = check_annotations(annotations, sources)
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
