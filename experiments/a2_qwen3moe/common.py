"""Shared helpers for the A2 / Qwen3-MoE verification experiments.

Everything runs on CPU. No NPU, no model weights, no network access to HF:
the Qwen3-30B-A3B config is written out locally from the values below.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"
OUT.mkdir(exist_ok=True)
SRC = Path(os.environ.get("INFERATLAS_SRC", ROOT / "src"))
VLLM_ASCEND = SRC / "vllm-ascend"

# Qwen/Qwen3-30B-A3B config.json (public values).
QWEN3_MOE_CONFIG = {
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "num_hidden_layers": 48,
    "num_attention_heads": 32,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "intermediate_size": 6144,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "decoder_sparse_step": 1,
    "mlp_only_layers": [],
    "norm_topk_prob": True,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-06,
    "hidden_act": "silu",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "bos_token_id": 151643,
    "eos_token_id": 151645,
    "router_aux_loss_coef": 0.001,
    "output_router_logits": False,
    "sliding_window": None,
    "use_sliding_window": False,
    "max_window_layers": 48,
    "initializer_range": 0.02,
    "use_cache": True,
}


def write_model_dir() -> str:
    d = Path(tempfile.mkdtemp(prefix="qwen3moe_cfg_"))
    (d / "config.json").write_text(json.dumps(QWEN3_MOE_CONFIG, indent=2))
    return str(d)


def expected_params() -> dict:
    """Closed-form parameter count for Qwen3-MoE, used to check the meta build."""
    c = QWEN3_MOE_CONFIG
    H, L, V = c["hidden_size"], c["num_hidden_layers"], c["vocab_size"]
    D, Hq, Hkv = c["head_dim"], c["num_attention_heads"], c["num_key_value_heads"]
    E, I = c["num_experts"], c["moe_intermediate_size"]
    attn = H * (Hq + 2 * Hkv) * D + Hq * D * H + 2 * D  # qkv + o + q_norm/k_norm
    moe = H * E + E * 3 * H * I  # router + experts(gate,up,down)
    norms = 2 * H
    per_layer = attn + moe + norms
    total = V * H * 2 + L * per_layer + H  # embed + lm_head + layers + final norm
    return {"per_layer": per_layer, "total": total}


def dump(name: str, payload: dict) -> None:
    p = OUT / f"{name}.json"
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
    print(f"[written] {p}")


def run_stage(result: dict, stage: str, fn):
    """Run one stage, record ok / error with traceback, never raise."""
    try:
        value = fn()
        result[stage] = {"ok": True, **(value or {})}
    except BaseException as e:  # noqa: BLE001 - we want SystemExit too
        result[stage] = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc().splitlines()[-25:],
        }
        print(f"[{stage}] FAILED: {type(e).__name__}: {e}", file=sys.stderr)
    return result[stage]


def summarize_model(model) -> dict:
    """Parameter shapes, counts, bytes, and module classes of a meta-built model."""
    import torch

    params, total, total_bytes, bad = {}, 0, 0, []
    for name, p in model.named_parameters():
        if not isinstance(p, torch.Tensor) or not all(isinstance(s, int) for s in p.shape):
            bad.append(name)
            continue
        params[name] = {"shape": list(p.shape), "dtype": str(p.dtype), "device": str(p.device)}
        total += p.numel()
        total_bytes += p.numel() * p.element_size()
    classes = {}
    for name, m in model.named_modules():
        cls = type(m)
        classes[name] = f"{cls.__module__}.{cls.__qualname__}"
    # Layer 0 is enough to see per-layer structure; all 48 layers are identical.
    layer0 = {k: v for k, v in classes.items() if ".layers.0." in k or k.endswith(".layers.0")}
    layer0_params = {k: v for k, v in params.items() if ".layers.0." in k}
    return {
        "num_params": total,
        "param_bytes": total_bytes,
        "bad_params": bad,
        "top_level_classes": {k: v for k, v in classes.items() if k.count(".") <= 1},
        "layer0_classes": layer0,
        "layer0_params": layer0_params,
        "devices": sorted({v["device"] for v in params.values()}),
    }
