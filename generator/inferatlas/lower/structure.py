"""Compatibility IR for extracted architectures without a numeric lowering implementation."""

from inferatlas.schema import IR, Extract, Triage
from inferatlas.schema.models import IRNode, Memory, PhaseExpr, UnknownLayerClass, ViewFamily


def structure_ir(extract: Extract):
    from .pipeline import LowerResult, hardware_profile_triage

    zero = PhaseExpr(prefill="0", decode="0")
    modules = {m.path: m for m in extract.modules}
    if len(modules) != len(extract.modules):
        raise ValueError("duplicate extracted module path")
    if not modules:
        raise ValueError("structure requires an extracted module tree")
    nodes = [
        IRNode(
            id=m.path or "root",
            kind="group",
            class_name=m.class_name,
            src=m.src,
            rule="generic",
            label=m.path or m.class_name.rsplit(".", 1)[-1],
            inputs=[],
            out=[],
            weight_bytes="0",
            flops=zero,
            bytes_moved=zero,
        )
        for m in sorted(extract.modules, key=lambda m: m.path)
        if m.src is not None
    ]
    ir = IR(
        snapshot=extract.snapshot,
        symbols={},
        params={},
        nodes=nodes,
        edges=[],
        groups=[],
        memory=Memory(
            weights="0",
            kv_per_token="0",
            kv_tokens="0",
            activation_peak="0",
            non_torch="0",
            reserved="0",
            unmodelled=["Numeric lowering unavailable; use arch.json for structure."],
        ),
        views=ViewFamily(family="hybrid"),
        notes=["Compatibility IR; numeric fields are unavailable placeholders, not estimates."],
    )
    from inferatlas.arch import structure_rules

    ruled = structure_rules()
    items = [
        UnknownLayerClass(
            kind="unknown_layer_class", symbol=symbol, used_by=[extract.snapshot.model.hf_id]
        )
        for symbol in sorted({m.class_name for m in extract.modules} - ruled.keys())
    ]
    from inferatlas.schema.models import ConfigFieldsUnavailable

    config = {**extract.config, **(extract.config.get("text_config") or {})}
    missing = sorted(
        k
        for k in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "intermediate_size",
            "vocab_size",
        )
        if config.get(k) is None
    )
    if missing:
        items.append(
            ConfigFieldsUnavailable(
                kind="missing_config",
                model=extract.snapshot.model.hf_id,
                fields=missing,
                message="Numeric config fields unavailable: "
                + ", ".join(missing)
                + "; structure uses observed module facts.",
            )
        )
    if issue := hardware_profile_triage(extract):
        items.append(issue)
    return LowerResult(ir=ir, triage=Triage(snapshot=ir.snapshot, items=items), parameter_shapes={})
