"""Config-only Qwen3 projection with AST-resolved provenance, never model weights.

This is explicitly a curated fixture, not a substitute for CPU/meta extraction. Its
shapes are unsharded constructor dimensions; production facts come from T1.
"""

import ast
import os
from pathlib import Path

from inferatlas.schema.models import (
    CatalogModel,
    Extract,
    ExtractModule,
    ExtractParallel,
    ExtractParam,
    ModelRef,
    Snapshot,
    SourceRef,
)

from .pipeline import LowerResult, lower

DEFAULT_SOURCE_ROOT = Path("/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/src")


def catalog_extract(
    model: CatalogModel, snapshot: Snapshot, source_root: str | Path | None = None
) -> Extract:
    if model.config.get("model_type") not in {"qwen3", "qwen3_moe"}:
        raise ValueError("Config projection currently supports Qwen3 dense/MoE only")
    source_root = Path(source_root or os.environ.get("INFERATLAS_SRC", DEFAULT_SOURCE_ROOT))
    c = model.config
    moe = c["model_type"] == "qwen3_moe"
    family = "qwen3_moe" if moe else "qwen3"
    prefix = "Qwen3Moe" if moe else "Qwen3"
    h, hq, hkv, d, vocab = (
        int(c[k])
        for k in (
            "hidden_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
        )
    )
    inter = int(c["intermediate_size"])
    expert_inter, experts = int(c.get("moe_intermediate_size", inter)), int(c.get("num_experts", 0))
    dtype = str(c.get("torch_dtype", "bfloat16"))
    snapshot = snapshot.model_copy(
        update={"model": ModelRef(hf_id=model.hf_id, config_rev=model.config_rev)}
    )
    modules: list[ExtractModule] = []
    parsed: dict[Path, ast.Module] = {}

    def module(path, namespace, cls, shapes=None, attrs=None):
        file = source_root / "vllm" / (namespace.replace(".", "/") + ".py")
        if file not in parsed:
            parsed[file] = ast.parse(file.read_text())
        declarations = [
            n for n in parsed[file].body if isinstance(n, ast.ClassDef) and n.name == cls
        ]
        if len(declarations) != 1:
            raise ValueError(f"Cannot resolve declaration for {namespace}.{cls}")
        symbol = namespace + "." + cls
        modules.append(
            ExtractModule(
                path=path,
                class_name=symbol,
                attrs=attrs or {},
                params=[
                    ExtractParam(name=name, shape=shape, dtype=dtype)
                    for name, shape in sorted((shapes or {}).items())
                ],
                src=SourceRef(
                    repo="vllm",
                    file=namespace.replace(".", "/") + ".py",
                    line=declarations[0].lineno,
                    commit=snapshot.vllm.commit,
                    symbol=symbol,
                ),
            )
        )

    models = "vllm.model_executor.models." + family
    layers = "vllm.model_executor.layers."
    module("", models, prefix + "ForCausalLM")
    module("model", models, prefix + "Model")
    module("model.layers.0", models, prefix + "DecoderLayer")
    module(
        "model.embed_tokens",
        layers + "vocab_parallel_embedding",
        "VocabParallelEmbedding",
        {"weight": [(vocab + 63) // 64 * 64, h]},
    )
    module(
        "lm_head",
        layers + "vocab_parallel_embedding",
        "ParallelLMHead",
        {"weight": [(vocab + 63) // 64 * 64, h]},
    )
    module("model.norm", layers + "layernorm", "RMSNorm", {"weight": [h]})
    module("logits_processor", layers + "logits_processor", "LogitsProcessor")
    # Sampler is an engine operation and need not be registered on the model itself.
    module("sampler", "vllm.v1.sample.sampler", "Sampler")
    base = "model.layers.0."
    for name in ("input_layernorm", "post_attention_layernorm"):
        module(base + name, layers + "layernorm", "RMSNorm", {"weight": [h]})
    module(base + "self_attn", models, prefix + "Attention")
    module(
        base + "self_attn.qkv_proj",
        layers + "linear",
        "QKVParallelLinear",
        {"weight": [(hq + 2 * hkv) * d, h]},
        {"input_size": h, "output_size": (hq + 2 * hkv) * d},
    )
    module(
        base + "self_attn.o_proj",
        layers + "linear",
        "RowParallelLinear",
        {"weight": [h, hq * d]},
        {"input_size": hq * d, "output_size": h, "reduce_results": True},
    )
    for name in ("q_norm", "k_norm"):
        module(base + "self_attn." + name, layers + "layernorm", "RMSNorm", {"weight": [d]})
    module(base + "self_attn.rotary_emb", layers + "rotary_embedding.base", "RotaryEmbedding")
    module(
        base + "self_attn.rotary_emb.apply_rotary_emb",
        layers + "rotary_embedding.common",
        "ApplyRotaryEmb",
    )
    module(base + "self_attn.attn", layers + "attention.attention", "Attention")
    if moe:
        module(base + "mlp", models, "Qwen3MoeSparseMoeBlock")
        module(
            base + "mlp.gate",
            layers + "fused_moe.router.gate_linear",
            "GateLinear",
            {"weight": [experts, h]},
        )
        module(base + "mlp.experts", layers + "fused_moe.runner.moe_runner", "MoERunner")
        module(
            base + "mlp.experts.routed_experts",
            layers + "fused_moe.routed_experts",
            "RoutedExperts",
            {"w13_weight": [experts, 2 * expert_inter, h], "w2_weight": [experts, h, expert_inter]},
        )
        module(
            base + "mlp.experts.routed_experts.quant_method",
            layers + "fused_moe.unquantized_fused_moe_method",
            "UnquantizedFusedMoEMethod",
        )
    else:
        module(base + "mlp", "vllm.model_executor.models.qwen2", "Qwen2MLP")
        module(
            base + "mlp.gate_up_proj",
            layers + "linear",
            "MergedColumnParallelLinear",
            {"weight": [2 * inter, h]},
            {"input_size": h, "output_size": 2 * inter},
        )
        module(
            base + "mlp.down_proj",
            layers + "linear",
            "RowParallelLinear",
            {"weight": [h, inter]},
            {"input_size": inter, "output_size": h, "reduce_results": True},
        )
        module(base + "mlp.act_fn", layers + "activation", "SiluAndMul")
    return Extract(
        snapshot=snapshot,
        fixture=True,
        config=c,
        parallel=ExtractParallel(),
        modules=sorted(modules, key=lambda m: m.path),
        layer_types=["full_attention"] * int(c["num_hidden_layers"]),
        env={},
        notes=[
            "Config-only curated Qwen3 projection with AST source declarations; not CPU/meta extraction."
        ],
    )


def lower_catalog(
    model: CatalogModel, snapshot: Snapshot, source_root: str | Path | None = None
) -> LowerResult:
    return lower(catalog_extract(model, snapshot, source_root))
