"""Bind GLOBAL module sizes; observed partition sizes never become runtime constants."""

import re
from collections.abc import Mapping

from inferatlas.schema.models import Extract, ExtractModule

from .rules import RuleSpec

DTYPES = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}


def substitute(expr: str, bindings: Mapping[str, str]) -> str:
    return re.sub(r"\b[A-Za-z_][A-Za-z_0-9]*\b", lambda m: bindings.get(m[0], m[0]), expr)


def bind_context(module: ExtractModule, extract: Extract, rule: RuleSpec) -> dict[str, str]:
    """Config/global constructor attrs are authoritative; partition attrs cross-check them."""
    a, c, observed_tp = module.attrs, extract.config, extract.parallel.tp
    cls = rule.match.class_name.removeprefix("Ascend")
    # Inherited custom subclasses retain base semantics, independent of their name.
    identity = rule.inherits or rule.id
    row = cls == "RowParallelLinear" or identity == "linear.row"
    column = cls in {"ColumnParallelLinear", "MergedColumnParallelLinear"} or identity in {
        "linear.column",
        "linear.merged",
    }
    qkv = cls == "QKVParallelLinear" or identity == "linear.qkv"
    gate = cls == "GateLinear" or identity == "moe.gate"
    weight = next((p for p in module.params if p.name == "weight"), None)
    shape = weight.shape if weight else []
    values: dict[str, str] = {
        "tied": str(bool(c.get("tie_word_embeddings", False))).lower(),
        "num_heads": "Hq",
        "num_kv_heads": "Hkv",
        "head_size": "D",
        "num_experts": "E",
        "top_k": "topk",
        "intermediate_size": "I_moe" if "moe_intermediate_size" in c else "I",
        "reduce_results": str(a.get("reduce_results", True)).lower(),
        "gather_output": str(a.get("gather_output", False)).lower(),
        "has_bias": str(any(p.name == "bias" for p in module.params)).lower(),
    }
    h = int(c["hidden_size"])
    d = int(c.get("head_dim", h // int(c["num_attention_heads"])))
    norm = d if module.path.endswith(("q_norm", "k_norm")) else h
    values["norm_size"] = "D" if norm == d and module.path.endswith(("q_norm", "k_norm")) else "H"
    values["norm_copies"] = (
        "Hq/tp"
        if module.path.endswith("q_norm")
        else "max(1,Hkv/tp)"
        if module.path.endswith("k_norm")
        else "1"
    )
    if "norm_size" in rule.context and shape and shape != [norm]:
        raise ValueError(f"{module.path}: norm shape disagrees with config")
    if "norm_size" in rule.context and "hidden_size" in a and a["hidden_size"] != norm:
        raise ValueError(f"{module.path}: norm size attribute disagrees with config")
    in_size = a.get("input_size", a.get("in_features"))
    out_size = a.get("output_size", a.get("out_features"))
    if out_size is None and column and "output_sizes" in a:
        sizes = a["output_sizes"]
        if not isinstance(sizes, list) or any(type(v) is not int or v <= 0 for v in sizes):
            raise ValueError(f"{module.path}: output_sizes must be positive integers")
        out_size = sum(sizes)
    if in_size is None and len(shape) == 2:
        in_size = shape[1] * (observed_tp if row else 1)
    if out_size is None and len(shape) == 2:
        out_size = shape[0] * (observed_tp if column else 1)
    if gate:
        in_size, out_size = h, int(c["num_experts"])
        if shape and shape != [out_size, in_size]:
            raise ValueError(f"{module.path}: router shape disagrees with config")
    for key in ("reduce_results", "gather_output"):
        if key in rule.context and key in a and not isinstance(a[key], bool):
            raise ValueError(f"{module.path}: {key} must be a boolean")
    for key, value in (("in_features", in_size), ("out_features", out_size)):
        if key in rule.context and value is not None and (type(value) is not int or value <= 0):
            raise ValueError(f"{module.path}: {key} must be a positive integer")
    if qkv:
        in_size = in_size if in_size is not None else h
    if in_size is not None:
        values["in_features"] = "H" if in_size == h else str(in_size)
    if out_size is not None:
        values["out_features"] = "H" if out_size == h else str(out_size)
    if row or column:
        if in_size is None or out_size is None:
            raise ValueError(f"{module.path}: missing global linear dimensions")
        local_in = int(in_size) // observed_tp if row else int(in_size)
        local_out = int(out_size) // observed_tp if column else int(out_size)
        for key, expected in {
            "input_size_per_partition": local_in,
            "output_size_per_partition": local_out,
        }.items():
            if key in a and a[key] != expected:
                raise ValueError(f"{module.path}: {key} disagrees with global sizes")
        if shape and shape != [local_out, local_in]:
            raise ValueError(f"{module.path}: linear shape disagrees with global sizes")
        expected = None
        if module.path.endswith("self_attn.o_proj"):
            expected = (int(c["num_attention_heads"]) * d, h)
        elif module.path.endswith("mlp.down_proj"):
            expected = (int(c["intermediate_size"]), h)
        elif module.path.endswith("mlp.gate_up_proj"):
            expected = (h, 2 * int(c["intermediate_size"]))
        if expected and (in_size, out_size) != expected:
            raise ValueError(f"{module.path}: constructor sizes disagree with config")
    if qkv:
        hq, hkv = int(c["num_attention_heads"]), int(c["num_key_value_heads"])
        if hq % observed_tp or (hkv % observed_tp and observed_tp % hkv):
            raise ValueError(f"{module.path}: incompatible TP/head counts")
        expected = [(hq // observed_tp + 2 * max(1, hkv // observed_tp)) * d, int(in_size)]
        if shape and shape != expected:
            raise ValueError(f"{module.path}: QKV shape disagrees with config")
        if int(in_size) != h:
            raise ValueError(f"{module.path}: QKV input disagrees with config")
        for key, expected in {
            "total_num_heads": hq,
            "total_num_kv_heads": hkv,
            "num_heads": hq // observed_tp,
            "num_kv_heads": max(1, hkv // observed_tp),
            "head_size": d,
            "head_dim": d,
        }.items():
            if key in a and a[key] != expected:
                raise ValueError(f"{module.path}: {key} disagrees with global config")
    if not (row or column or qkv or gate) and "in_features" in rule.context:
        if shape and shape != [out_size, in_size]:
            raise ValueError(f"{module.path}: replicated shape disagrees with global sizes")
    bias = next((p for p in module.params if p.name == "bias"), None)
    if bias and "bias" in rule.parameter_shapes and (qkv or out_size is not None):
        output = (
            (
                int(c["num_attention_heads"]) // observed_tp
                + 2 * max(1, int(c["num_key_value_heads"]) // observed_tp)
            )
            * d
            if qkv
            else int(out_size) // (observed_tp if column else 1)
        )
        if bias.shape != [output]:
            raise ValueError(f"{module.path}: bias shape disagrees with output partition")
    if "num_experts" in rule.context:
        e = int(c["num_experts"])
        degree = observed_tp * extract.parallel.dp
        ep = degree if extract.parallel.ep else 1
        expert_tp = 1 if ep > 1 else degree
        inter = int(c["moe_intermediate_size"])
        for key, expected in {
            "global_num_experts": e,
            "intermediate_size_per_partition": inter // expert_tp,
            "local_num_experts": e // ep,
            "num_local_experts": e // ep,
            "top_k": int(c["num_experts_per_tok"]),
        }.items():
            if key in a and a[key] != expected:
                raise ValueError(f"{module.path}: {key} disagrees with config")
        # num_experts is global on RoutedExperts, local on some older fused implementations.
        if "num_experts" in a and a["num_experts"] not in {e, e // ep}:
            raise ValueError(f"{module.path}: expert count disagrees with config")
        for p in module.params:
            expected = {
                "w13_weight": [e // ep, 2 * inter // expert_tp, h],
                "w2_weight": [e // ep, h, inter // expert_tp],
            }.get(p.name)
            if expected is not None and p.shape != expected:
                raise ValueError(f"{module.path}: expert shape disagrees with config")
    if module.path.endswith(("self_attn", "self_attn.attn")):
        for key, expected in {
            "num_heads": int(c["num_attention_heads"]) // observed_tp,
            "num_kv_heads": max(1, int(c["num_key_value_heads"]) // observed_tp),
            "head_size": d,
            "head_dim": d,
        }.items():
            if key in a and a[key] != expected:
                raise ValueError(f"{module.path}: observed attention {key} disagrees with config")
    if cls in {"VocabParallelEmbedding", "ParallelLMHead"}:
        vocab = int(c["vocab_size"])
        padded = (vocab + 63) // 64 * 64
        if shape and shape != [padded // observed_tp, h]:
            raise ValueError(f"{module.path}: vocabulary shape disagrees with config")
        if padded % observed_tp:
            raise ValueError(f"{module.path}: padded vocabulary is not divisible by TP")
        if "padding_size" in a and a["padding_size"] != 64:
            raise ValueError(f"{module.path}: nondefault vocabulary padding needs a rule")
    missing = set(rule.context) - values.keys()
    if missing:
        raise ValueError(f"{module.path}: missing context {sorted(missing)}")
    return {key: values[key] for key in rule.context}
