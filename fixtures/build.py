"""Regenerate curated T0 examples from experiment facts, without upstream imports/weights.

Run from repo root: uv run --project generator python fixtures/build.py
Only source declaration locations are discovered; IR formulas/layouts are hand curated here.
INFERATLAS_SRC may select an alternate directory containing the pinned upstream clones.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import os
from pathlib import Path

from ruamel.yaml import YAML

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "fixtures"
EXP = ROOT / "experiments/a2_qwen3moe"
DEFAULT_SRC = ROOT / "experiments/a2_qwen3moe/src"
if not DEFAULT_SRC.exists():
    DEFAULT_SRC = ROOT.parents[1] / "experiments/a2_qwen3moe/src"
SRC = Path(os.environ.get("INFERATLAS_SRC", DEFAULT_SRC))
VLLM = "544f182d9f0c9bfe8bd7645525951d5e3692a542"
ASCEND = "24a76e84394ae3427195e3715fc08e90769d398c"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def load(name):
    return json.loads((EXP / "out" / name).read_text())


def source(symbol):
    parts = symbol.split(".")
    repo, commit = ("vllm-ascend", ASCEND) if parts[0] == "vllm_ascend" else ("vllm", VLLM)
    for length in range(len(parts) - 1, 0, -1):
        file = Path(*parts[:length]).with_suffix(".py")
        path = SRC / repo / file
        if path.exists():
            target = parts[length:]
            body = ast.parse(path.read_text()).body
            for name in target:
                node = next(
                    n
                    for n in body
                    if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name == name
                )
                body = node.body
            return {
                "repo": repo,
                "file": file.as_posix(),
                "line": node.lineno,
                "commit": commit,
                "symbol": symbol,
            }
    raise ValueError(f"Source not found: {symbol}")


common = ast.parse((EXP / "common.py").read_text())
config = next(
    ast.literal_eval(n.value)
    for n in common.body
    if isinstance(n, ast.Assign)
    and any(isinstance(t, ast.Name) and t.id == "QWEN3_MOE_CONFIG" for t in n.targets)
)
dense_config = {
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "hidden_size": 4096,
    "num_hidden_layers": 36,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "intermediate_size": 12288,
    "vocab_size": 151936,
    "max_position_embeddings": 40960,
    "rope_theta": 1000000.0,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
}
deepseek_config = {
    "architectures": ["DeepseekV3ForCausalLM"],
    "model_type": "deepseek_v3",
    "hidden_size": 7168,
    "num_hidden_layers": 61,
    "num_attention_heads": 128,
    "num_key_value_heads": 128,
    "intermediate_size": 18432,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "first_k_dense_replace": 3,
    "moe_layer_freq": 1,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "vocab_size": 129280,
    "max_position_embeddings": 163840,
    "rms_norm_eps": 1e-6,
    "hidden_act": "silu",
    "attention_bias": False,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_cache": True,
    "scoring_func": "sigmoid",
    "topk_method": "noaux_tc",
    "n_group": 8,
    "topk_group": 4,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
    "rope_theta": 10000,
    "rope_scaling": {
        "type": "yarn",
        "factor": 40,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1.0,
        "mscale_all_dim": 1.0,
    },
}


def model_ref(hf_id, cfg):
    digest = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {"hf_id": hf_id, "config_rev": "inline:sha256:" + digest}


snapshot = {
    "vllm": {"tag": "main-verified", "commit": VLLM},
    "ascend": {"tag": "main", "commit": ASCEND},
    "model": model_ref("Qwen/Qwen3-30B-A3B", config),
    "device": "A2",
    "generator_version": "0.1.0",
}
base = load("exp1_vllm_meta_tp1.json")
base4 = load("exp1_vllm_meta_tp4.json")
a2 = load("exp3_ascend_meta_A2.json")
overlay = load("exp2_ascend_overlay_A2.json")
classes = base["build"]["top_level_classes"] | base["build"]["layer0_classes"]
ascend_classes = a2["build"]["top_level_classes"] | a2["build"]["layer0_classes"]


def make_extract(tp, experiment):
    params = dict(experiment["build"]["layer0_params"])
    for key, shape in {
        "model.embed_tokens.weight": [151936 // tp, 2048],
        "model.norm.weight": [2048],
        "lm_head.weight": [151936 // tp, 2048],
    }.items():
        params[key] = {"shape": shape, "dtype": "torch.bfloat16"}
    modules = []
    for path, cls in sorted(classes.items()):
        local = [
            {
                "name": name.rsplit(".", 1)[1],
                "shape": p["shape"],
                "dtype": p["dtype"].removeprefix("torch."),
            }
            for name, p in sorted(params.items())
            if name.rsplit(".", 1)[0] == path
        ]
        attrs = {}
        if path.endswith("qkv_proj"):
            attrs = {"input_size": 2048, "output_size": 5120, "tp_size": tp}
        if path.endswith("o_proj"):
            attrs = {
                "input_size": 4096,
                "input_size_per_partition": 4096 // tp,
                "output_size": 2048,
                "output_size_per_partition": 2048,
                "tp_size": tp,
                "reduce_results": True,
            }
        if path.endswith("self_attn"):
            attrs = {
                "num_heads": 32 // tp,
                "num_kv_heads": max(1, 4 // tp),
                "head_dim": 128,
            }
        if path.endswith("routed_experts"):
            attrs = {
                "num_experts": 128,
                "top_k": 8,
                "intermediate_size_per_partition": 768 // tp,
            }
        module = {
            "path": path,
            "class": cls,
            "src": source(cls) if not cls.startswith("torch.") else None,
            "attrs": attrs,
            "params": local,
        }
        ac = ascend_classes.get(path, cls)
        if ac != cls:
            module["ascend"] = {"class": ac, "src": source(ac)}
        modules.append(module)
    return {
        "snapshot": snapshot,
        "fixture": True,
        "config": config,
        "parallel": {"tp": tp, "pp": 1, "dp": 1, "ep": False, "rank": 0},
        "modules": modules,
        "layer_types": ["full_attention"] * 48,
        "ascend": {
            "device": "A2",
            "profile": {k: v for k, v in overlay["profile"].items() if k != "ok"},
            "custom_ops": overlay["customop"]["registered"],
            "patched": {
                symbol: origin
                if origin.startswith("vllm_ascend/")
                else "vllm_ascend/" + origin.replace("vllm_ascend.", "").replace(".", "/") + ".py"
                if origin.startswith("vllm_ascend.")
                else "vllm_ascend/" + origin
                for origin, symbols in overlay["diff"]["by_origin"].items()
                for symbol in symbols
            },
        },
        "env": {"torch": base["env"]["torch"], "python": "3.12"},
        "notes": [
            "Curated partial module tree: root, embedding, complete layer 0, final norm, lm_head, logits processor. Other decoder layers repeat layer 0.",
            "Shapes copied from exp1 tp1/tp4; Ascend classes/profile/patches from exp2/exp3 A2. Ascend shape equivalence was measured only at tp=1; tp4 overlay is illustrative.",
            "Declaration lines are resolved by AST against the pinned upstream clones, not copied from experiment environment version strings.",
        ],
    }


extract = make_extract(1, base)
write(OUT / "qwen3-30b-a3b/extract.json", extract)
write(OUT / "qwen3-30b-a3b/extract-tp4.json", make_extract(4, base4))
write(OUT / "qwen3-30b-a3b/snapshot.json", snapshot)
modules = {m["path"]: m for m in extract["modules"]}


def param(kind, default, description, **domain):
    return {"kind": kind, "default": str(default), "description": description, **domain}


params = {
    "dtype_ids": param(
        "dtype",
        "int64",
        "Token index dtype; bytes(dtype_ids) is supported; token index storage is excluded from the metric ledger.",
        choices=["int32", "int64"],
    ),
    "tp": param(
        "integer",
        1,
        "Attention tensor parallel degree. Must divide Hq and be compatible with Hkv.",
        choices=["1", "2", "4", "8"],
    ),
    "pp": param(
        "integer",
        1,
        "Pipeline stages. Fixture supports equal partitions only; custom/uneven PP needs explicit rank ledgers.",
        choices=["1", "2", "3", "4", "6", "8", "12", "16", "24", "48"],
    ),
    "dp": param(
        "integer",
        1,
        "Independent attention data parallel replicas; T and S are local to one replica.",
        min="1",
        max="8",
        step="1",
    ),
    "ep": param(
        "integer",
        1,
        "Expert parallel degree: 1 disables EP, otherwise must equal tp*dp (vLLM flattened expert group).",
        choices=["1", "2", "4", "8", "16", "32", "64", "128"],
    ),
    "T": param(
        "integer",
        4096,
        "Local attention tokens in one prefill chunk, independent of cache length.",
        min="1",
        max="40960",
        step="1",
    ),
    "S": param(
        "integer",
        1,
        "Local decode sequences; one generated token per sequence per step.",
        min="1",
        max="256",
        step="1",
    ),
    "context": param(
        "integer",
        4096,
        "Existing cache length per decode sequence in tokens; uniform length approximation.",
        min="0",
        max="40960",
        step="1",
    ),
    "batch": param(
        "integer",
        1,
        "Equal-length sequences within a prefill chunk. T must be divisible by batch.",
        min="1",
        max="256",
        step="1",
    ),
    "dtype_w": param(
        "dtype",
        "bf16",
        "Stored weight dtype; fixture excludes quantization scales and packing overhead.",
        choices=["bf16", "fp16", "fp32"],
    ),
    "dtype_kv": param(
        "dtype",
        "bf16",
        "Stored KV dtype; quantized KV scales are not modelled.",
        choices=["bf16", "fp16", "fp32"],
    ),
    "dtype_a": param(
        "dtype",
        "bf16",
        "Activation dtype; router/logits may internally accumulate in fp32.",
        choices=["bf16", "fp16", "fp32"],
    ),
    "hbm_bytes": param(
        "number",
        64000000000,
        "Device memory budget in bytes (decimal 64 GB default, user configurable).",
        min="1",
    ),
    "gpu_memory_utilization": param(
        "number",
        0.9,
        "Fraction of device memory available to vLLM.",
        min="0.01",
        max="1",
        step="0.01",
    ),
    "max_num_batched_tokens": param(
        "integer",
        4096,
        "Token count used for chunked-prefill memory profiling.",
        min="1",
        max="40960",
        step="1",
    ),
    "max_num_seqs": param(
        "integer",
        256,
        "Sampler warmup concurrent sequence bound.",
        min="1",
        max="256",
        step="1",
    ),
    "block_size": param(
        "integer",
        128,
        "KV tokens per cache block; device/backend restrictions apply.",
        choices=["16", "32", "64", "128"],
    ),
    "non_torch_bytes": param(
        "number",
        0,
        "Configured non-torch byte deduction; default excludes unknown HCCL workspace.",
        min="0",
    ),
    "reserved_bytes": param(
        "number",
        0,
        "Additional reserved byte deduction; default excludes graph capture and fragmentation.",
        min="0",
    ),
}
symbols = {
    k: str(v)
    for k, v in {
        "H": 2048,
        "L": 48,
        "E": 128,
        "topk": 8,
        "Hq": 32,
        "Hkv": 4,
        "D": 128,
        "V": 151936,
        "I_moe": 768,
    }.items()
}
qw = "H*(Hq/tp+2*max(1,Hkv/tp))*D*bytes(dtype_w)"
ow = "H*Hq*D/tp*bytes(dtype_w)"
expert_divisor = "(ep>1 ? ep : tp*dp)"
ew = f"3*E*H*I_moe/{expert_divisor}*bytes(dtype_w)"
aw = f"({qw})+({ow})+2*D*bytes(dtype_w)"
mw = f"H*E*bytes(dtype_w)+({ew})"
lw = f"({aw})+({mw})+2*H*bytes(dtype_w)"
vw = "round_up(V,64)/tp*H*bytes(dtype_w)"
weights = f"({lw})*(L/pp)+(pp==1 ? 2*({vw})+H*bytes(dtype_w) : ({vw})+H*bytes(dtype_w))"
kv = "2*(L/pp)*max(1,Hkv/tp)*D*bytes(dtype_kv)"
activation = "max(max_num_batched_tokens*(4*H+(Hq/tp+2*max(1,Hkv/tp))*D+dp*topk/(ep>1 ? ep : 1)*(2*H+3*I_moe/(ep>1 ? 1 : tp*dp))+dp*(H+E))*bytes(dtype_a),max_num_seqs*V*(bytes(dtype_a)+3*bytes(fp32)+bytes(int64)))"
capacity = f"floor(max(0,hbm_bytes*gpu_memory_utilization-({weights})-({activation})-non_torch_bytes-reserved_bytes)/(({kv})*block_size))*block_size"


def tensor(name="hidden", shape=None, dtype="dtype_a"):
    return {"name": name, "shape": shape or ["T", "H"], "dtype": dtype}


def phase(p="0", d="0"):
    return {"prefill": p, "decode": d}


def comm(op, group, logical, bus, when):
    return {
        "op": op,
        "group": group,
        "logical_bytes": phase(logical, logical.replace("T*", "S*")),
        "bus_bytes": phase(bus, bus.replace("T*", "S*")),
        "when": when,
    }


nodes = []


def node(
    id,
    path,
    label,
    *,
    parent=None,
    kind="compute",
    weight="0",
    flops=None,
    moved=None,
    inputs=None,
    outputs=None,
    communication=None,
    parallel=None,
    notes=None,
):
    module = modules[path]
    rule_id = {
        "attn.o.reduce": "attn.o",
        "embed.reduce": "embed",
        "logits.gather": "logits",
        "output": "logits",
    }.get(id, id)
    if id in {"moe.experts.dispatch", "moe.experts.combine", "moe.experts.reduce"}:
        rule_id = "moe.experts"
    n = {
        "id": id,
        "kind": kind,
        "class": module["class"],
        "src": module["src"],
        "rule": "fixture.linear.row" if rule_id == "attn.o" else "fixture." + rule_id,
        "label": label,
        "parent": parent,
        "in": inputs or [tensor()],
        "out": outputs or [tensor()],
        "weight_bytes": weight,
        "flops": flops or phase(),
        "bytes_moved": moved or phase(),
        "comm": communication,
        "parallel": parallel or {},
        "notes": notes or [],
    }
    if id in {
        "norm.input",
        "norm.post",
        "attn.qkv",
        "attn.o",
        "attn.q_norm",
        "attn.k_norm",
        "attn.rope",
        "attn.rope.apply",
        "moe.gate",
        "moe.experts",
        "moe.experts.weights",
        "moe.experts.method",
    } and module.get("ascend"):
        n["ascend"] = {
            "tag": ["OOT"],
            "level": "KERNEL",
            "note": "A2 CPU/mock CustomOp replacement preserves the observed bf16 parameter shapes. Hardware-only kernel behavior is uncalibrated.",
            "src": [module["ascend"]["src"]],
            "annotation_id": "fixture.ascend." + id,
        }
    nodes.append(n)
    return n


layer = "model.layers.0"
attn = layer + ".self_attn"
moe = layer + ".mlp"
experts = moe + ".experts"
node(
    "input",
    "model",
    "Token IDs",
    kind="io",
    inputs=[tensor("tokens", ["T"], "dtype_ids")],
    outputs=[tensor("tokens", ["T"], "dtype_ids")],
    notes=[
        "Token IDs use dtype_ids (int64 by default); their storage is excluded from the metric ledger."
    ],
)
node(
    "embed",
    "model.embed_tokens",
    "Embedding",
    weight=vw,
    inputs=[tensor("tokens", ["T"], "dtype_ids")],
    moved=phase("T*H*bytes(dtype_w)+T*H*bytes(dtype_a)", "S*H*bytes(dtype_w)+S*H*bytes(dtype_a)"),
    parallel={"tp": {"dim": "out"}},
)
node(
    "decoder",
    layer,
    "MoE decoder × L",
    kind="group",
    weight=f"({lw})*(L/pp)",
    notes=[
        "Summary of L/pp resident layers; children describe one representative layer. Never sum this summary with its descendants."
    ],
)
node(
    "norm.input",
    layer + ".input_layernorm",
    "Input RMSNorm",
    parent="decoder",
    weight="H*bytes(dtype_w)",
    flops=phase("5*T*H", "5*S*H"),
    moved=phase("H*bytes(dtype_w)+2*T*H*bytes(dtype_a)", "H*bytes(dtype_w)+2*S*H*bytes(dtype_a)"),
)
node("attn", attn, "GQA attention", parent="decoder", kind="group", weight=aw)
node(
    "attn.qkv",
    attn + ".qkv_proj",
    "QKV projection",
    parent="attn",
    weight=qw,
    flops=phase("2*T*H*(Hq/tp+2*max(1,Hkv/tp))*D", "2*S*H*(Hq/tp+2*max(1,Hkv/tp))*D"),
    moved=phase(
        f"({qw})+T*(H+(Hq/tp+2*max(1,Hkv/tp))*D)*bytes(dtype_a)",
        f"({qw})+S*(H+(Hq/tp+2*max(1,Hkv/tp))*D)*bytes(dtype_a)",
    ),
    outputs=[
        tensor("q", ["T", "Hq/tp", "D"]),
        tensor("k", ["T", "max(1,Hkv/tp)", "D"]),
        tensor("v", ["T", "max(1,Hkv/tp)", "D"]),
    ],
    parallel={"tp": {"dim": "out"}},
)
node(
    "attn.qk_norm",
    attn,
    "Q/K head norms",
    parent="attn",
    kind="group",
    weight="2*D*bytes(dtype_w)",
)
for name, heads in [("q", "Hq/tp"), ("k", "max(1,Hkv/tp)")]:
    node(
        f"attn.{name}_norm",
        f"{attn}.{name}_norm",
        f"{name.upper()} RMSNorm",
        parent="attn.qk_norm",
        weight="D*bytes(dtype_w)",
        inputs=[tensor(name, ["T", heads, "D"])],
        outputs=[tensor(name, ["T", heads, "D"])],
        flops=phase(f"5*T*({heads})*D", f"5*S*({heads})*D"),
        moved=phase(
            f"D*bytes(dtype_w)+2*T*({heads})*D*bytes(dtype_a)",
            f"D*bytes(dtype_w)+2*S*({heads})*D*bytes(dtype_a)",
        ),
    )
node(
    "attn.rope",
    attn + ".rotary_emb",
    "RoPE",
    parent="attn",
    kind="group",
    inputs=[tensor("q", ["T", "Hq/tp", "D"]), tensor("k", ["T", "max(1,Hkv/tp)", "D"])],
    outputs=[
        tensor("q", ["T", "Hq/tp", "D"]),
        tensor("k", ["T", "max(1,Hkv/tp)", "D"]),
    ],
)
node(
    "attn.rope.apply",
    attn + ".rotary_emb.apply_rotary_emb",
    "Apply rotary",
    parent="attn.rope",
    flops=phase("3*T*(Hq/tp+max(1,Hkv/tp))*D", "3*S*(Hq/tp+max(1,Hkv/tp))*D"),
    moved=phase(
        "2*T*(Hq/tp+max(1,Hkv/tp))*D*bytes(dtype_a)",
        "2*S*(Hq/tp+max(1,Hkv/tp))*D*bytes(dtype_a)",
    ),
)
node(
    "attn.core",
    attn + ".attn",
    "Causal attention + KV",
    parent="attn",
    flops=phase("2*T*(T/batch+1)*Hq/tp*D", "4*S*(context+1)*Hq/tp*D"),
    moved=phase(
        "T*(2*Hq/tp+4*max(1,Hkv/tp))*D*bytes(dtype_a)",
        "2*S*(context+1)*max(1,Hkv/tp)*D*bytes(dtype_kv)+2*S*Hq/tp*D*bytes(dtype_a)",
    ),
    inputs=[
        tensor("q", ["T", "Hq/tp", "D"]),
        tensor("k", ["T", "max(1,Hkv/tp)", "D"]),
        tensor("v", ["T", "max(1,Hkv/tp)", "D"]),
    ],
    outputs=[tensor("context", ["T", "Hq/tp*D"])],
    notes=[
        "Prefill assumes equal-length fresh causal sequences, no cached prefix; QK/AV GEMM FLOPs only. Flash attention HBM traffic is an optimistic fused-kernel estimate; softmax costs and repeated tiles excluded.",
        "Decode reads context+1 keys/values and appends one KV token per sequence.",
    ],
)
node(
    "attn.o",
    attn + ".o_proj",
    "Output projection",
    parent="attn",
    weight=ow,
    flops=phase("2*T*Hq*D*H/tp", "2*S*Hq*D*H/tp"),
    moved=phase(f"({ow})+T*(H+Hq*D/tp)*bytes(dtype_a)", f"({ow})+S*(H+Hq*D/tp)*bytes(dtype_a)"),
    inputs=[tensor("context", ["T", "Hq/tp*D"])],
    parallel={"tp": {"dim": "in"}},
)
node(
    "attn.o.reduce",
    attn + ".o_proj",
    "TP all-reduce",
    parent="attn",
    kind="comm",
    communication=comm(
        "all_reduce",
        "tp",
        "T*H*bytes(dtype_a)",
        "tp>1 ? 2*(tp-1)/tp*T*H*bytes(dtype_a) : 0",
        "tp>1",
    ),
)
node(
    "norm.post",
    layer + ".post_attention_layernorm",
    "Residual + RMSNorm",
    parent="decoder",
    weight="H*bytes(dtype_w)",
    flops=phase("6*T*H", "6*S*H"),
    moved=phase("H*bytes(dtype_w)+3*T*H*bytes(dtype_a)", "H*bytes(dtype_w)+3*S*H*bytes(dtype_a)"),
)
node("moe", moe, "Sparse MoE · topk / E", parent="decoder", kind="group", weight=mw)
node(
    "moe.gate",
    moe + ".gate",
    "Router",
    parent="moe",
    weight="H*E*bytes(dtype_w)",
    flops=phase("2*T*H*E", "2*S*H*E"),
    moved=phase(
        "H*E*bytes(dtype_w)+T*(H+E)*bytes(dtype_a)",
        "H*E*bytes(dtype_w)+S*(H+E)*bytes(dtype_a)",
    ),
    outputs=[tensor("scores", ["T", "E"]), tensor("hidden")],
    parallel={"tp": {"dim": "replicated"}},
    notes=[
        "Top-k routing metadata and softmax/sigmoid costs excluded; router weights are replicated across TP/EP."
    ],
)
node("moe.experts", experts, "MoE runner", parent="moe", kind="group", weight=ew)
node(
    "moe.experts.weights",
    experts + ".routed_experts",
    "Routed experts",
    parent="moe.experts",
    kind="group",
    weight=ew,
    parallel={"tp": {"dim": "intermediate"}, "ep": {"dim": "experts"}},
    notes=[
        "Without EP, expert intermediate dimension is partitioned across tp*dp. With EP=tp*dp>1, experts are partitioned by EP and intermediate dimensions stay whole."
    ],
)
node(
    "moe.experts.method",
    experts + ".routed_experts.quant_method",
    "Unquantized MoE method",
    parent="moe.experts.weights",
    notes=[
        "Execution strategy metadata node; no additional tensors/weights. Not included in data-flow views."
    ],
)
logical = "T*H*bytes(dtype_a)"
for name, op in [("dispatch", "all_gather"), ("combine", "reduce_scatter")]:
    node(
        "moe.experts." + name,
        experts,
        "EP " + name,
        parent="moe",
        kind="comm",
        communication=comm(
            op,
            "ep",
            logical,
            f"dp>1 ? (tp*dp-1)/(tp*dp)*({logical}) : 0",
            "dp>1",
        ),
        notes=[
            "Default allgather_reducescatter backend, flattened tp*dp group even with ep=1. Contract logical tensor is T*H; router metadata omitted. Expert compute uses T*dp tokens."
        ],
    )
for name, factor in [("w13", 4), ("w2", 2)]:
    weight = f"{2 if name == 'w13' else 1}*E*H*I_moe/{expert_divisor}*bytes(dtype_w)"
    token_factor = f"{factor}*T*dp*topk*H*I_moe/{expert_divisor}"
    shape = [
        "T*dp*topk/(ep>1 ? ep : 1)",
        f"{2 if name == 'w13' else 1}*I_moe/(ep>1 ? 1 : tp*dp)" if name == "w13" else "H",
    ]
    node(
        "moe." + name,
        experts + ".routed_experts",
        "Gate/up GEMM" if name == "w13" else "Down GEMM",
        parent="moe.experts.weights",
        weight=weight,
        flops=phase(token_factor, token_factor.replace("T*", "S*")),
        moved=phase(
            f"({weight})+T*dp*topk*(H+2*I_moe)/{expert_divisor}*bytes(dtype_a)",
            f"({weight})+S*dp*topk*(H+2*I_moe)/{expert_divisor}*bytes(dtype_a)",
        ),
        outputs=[tensor("expert_activations", shape)],
        notes=[
            "Weight traffic reads min(E_local, local routed token count) experts; unquantized uniform-routing estimate."
        ],
    )
node(
    "moe.act",
    experts + ".routed_experts",
    "SiLU × up",
    parent="moe.experts.weights",
    flops=phase(f"4*T*dp*topk*I_moe/{expert_divisor}", f"4*S*dp*topk*I_moe/{expert_divisor}"),
    moved=phase(
        f"3*T*dp*topk*I_moe/{expert_divisor}*bytes(dtype_a)",
        f"3*S*dp*topk*I_moe/{expert_divisor}*bytes(dtype_a)",
    ),
    notes=[
        "Approximate scalar activation operation count; transcendental implementation cost is hardware-dependent."
    ],
)
node(
    "moe.experts.reduce",
    experts,
    "MoE TP reduce",
    parent="moe",
    kind="comm",
    communication=comm(
        "all_reduce",
        "tp",
        "T*H*bytes(dtype_a)",
        "tp>1 || ep>1 ? 2*(tp-1)/tp*T*H*bytes(dtype_a) : 0",
        "tp>1 || ep>1",
    ),
    notes=[
        "moe_runner.py output reduction uses the attention TP group, including when EP is enabled; TP=1 has zero bus traffic. DP dispatch/combine is modelled separately."
    ],
)
node(
    "final_norm",
    "model.norm",
    "Final RMSNorm",
    weight="H*bytes(dtype_w)",
    flops=phase("5*T*H", "5*S*H"),
    moved=phase("H*bytes(dtype_w)+2*T*H*bytes(dtype_a)", "H*bytes(dtype_w)+2*S*H*bytes(dtype_a)"),
)
selected_shape = {"prefill": ["batch", "H"], "decode": ["S", "H"]}
local_logits = {"prefill": ["batch", "round_up(V,64)/tp"], "decode": ["S", "round_up(V,64)/tp"]}
full_logits = {"prefill": ["batch", "round_up(V,64)"], "decode": ["S", "round_up(V,64)"]}
node(
    "lm_head",
    "lm_head",
    "LM head",
    weight=vw,
    flops=phase("2*batch*H*round_up(V,64)/tp", "2*S*H*round_up(V,64)/tp"),
    moved=phase(
        f"({vw})+batch*(H+round_up(V,64)/tp)*bytes(dtype_a)",
        f"({vw})+S*(H+round_up(V,64)/tp)*bytes(dtype_a)",
    ),
    inputs=[tensor("selected_hidden", selected_shape)],
    outputs=[tensor("logits", local_logits)],
    parallel={"tp": {"dim": "out"}},
    notes=[
        "Last-token logits: batch rows in prefill, S in decode. Default head_dtype is model dtype (dtype_a); fp32 sampler workspace is accounted separately."
    ],
)
node(
    "logits.gather",
    "logits_processor",
    "TP logits gather",
    kind="comm",
    inputs=[tensor("local_logits", local_logits)],
    outputs=[tensor("logits", full_logits)],
    communication={
        "op": "gather",
        "group": "tp",
        "when": "tp>1",
        "logical_bytes": phase(
            "batch*round_up(V,64)/tp*bytes(dtype_a)", "S*round_up(V,64)/tp*bytes(dtype_a)"
        ),
        "bus_bytes": phase(
            "tp>1 ? (tp-1)/tp*batch*round_up(V,64)/tp*bytes(dtype_a) : 0",
            "tp>1 ? (tp-1)/tp*S*round_up(V,64)/tp*bytes(dtype_a) : 0",
        ),
        "notes": [
            "Gather to TP rank 0: logical bytes are one local shard; bus bytes are rank-average outgoing bytes (rank 0 sends zero, others send one shard). Platforms selecting use_all_gather lower op=all_gather and bus=(tp-1)*local logical bytes instead. Truncate padding to V after collection."
        ],
    },
)
node(
    "output",
    "logits_processor",
    "Logits / sampling",
    kind="io",
    inputs=[tensor("logits", full_logits)],
    outputs=[tensor("tokens", {"prefill": ["batch"], "decode": ["S"]}, "dtype_ids")],
    notes=[
        "Truncate collected logits to V. Sampler internals excluded from FLOPs; max_num_seqs*V*fp32 included in memory peak."
    ],
)
node(
    "embed.reduce",
    "model.embed_tokens",
    "Embedding TP all-reduce",
    kind="comm",
    communication=comm(
        "all_reduce",
        "tp",
        "T*H*bytes(dtype_a)",
        "tp>1 ? 2*(tp-1)/tp*T*H*bytes(dtype_a) : 0",
        "tp>1",
    ),
)

# Spell out routed tensor slots and HBM traffic (activation bytes are not TP-sharded on input).
routed_tokens = "T*dp*topk/(ep>1 ? ep : 1)"
moe_tp = "(ep>1 ? 1 : tp*dp)"
node_map = {n["id"]: n for n in nodes}
node_map["moe.experts.dispatch"]["in"] = [tensor(), tensor("scores", ["T", "E"])]
node_map["moe.experts.dispatch"]["out"] = [tensor("routed", [routed_tokens, "H"])]
for key, in_dim, out_dim in [
    ("moe.w13", "H", f"2*I_moe/{moe_tp}"),
    ("moe.act", f"2*I_moe/{moe_tp}", f"I_moe/{moe_tp}"),
    ("moe.w2", f"I_moe/{moe_tp}", "H"),
]:
    n = node_map[key]
    n["in"] = [tensor("routed", [routed_tokens, in_dim])]
    n["out"] = [tensor("routed", [routed_tokens, out_dim])]
    touched = "min(E/(ep>1 ? ep : 1),T*dp*topk/(ep>1 ? ep : 1))"
    traffic_weight = (
        f"{2 if key == 'moe.w13' else 1}*({touched})*H*I_moe/{moe_tp}*bytes(dtype_w)"
        if key != "moe.act"
        else "0"
    )
    n["bytes_moved"] = phase(
        f"({traffic_weight})+({routed_tokens})*(({in_dim})+({out_dim}))*bytes(dtype_a)",
        f"({traffic_weight.replace('T*', 'S*')})+({routed_tokens.replace('T*', 'S*')})*(({in_dim})+({out_dim}))*bytes(dtype_a)",
    )
node_map["moe.experts.combine"]["in"] = [tensor("routed", [routed_tokens, "H"])]
node_map["attn.rope.apply"]["in"] = node_map["attn.rope"]["in"]
node_map["attn.rope.apply"]["out"] = node_map["attn.rope"]["out"]

node_map["attn.qk_norm"]["in"] = node_map["attn.rope"]["in"]
node_map["attn.qk_norm"]["out"] = node_map["attn.rope"]["out"]

# Group work/traffic summarize descendants; decoder metrics represent L/pp resident layers.
by_id = {n["id"]: n for n in nodes}


def summarize(id):
    n = by_id[id]
    children = [c for c in nodes if c["parent"] == id]
    for child in children:
        if child["kind"] == "group":
            summarize(child["id"])
    if children:
        for field in ["flops", "bytes_moved"]:
            n[field] = {
                p: "+".join(f"({c[field][p]})" for c in children) for p in ["prefill", "decode"]
            }
        if id == "decoder":
            for field in ["flops", "bytes_moved"]:
                n[field] = {p: f"({expr})*(L/pp)" for p, expr in n[field].items()}


summarize("decoder")


def edge(a, b, kind="data", shape=None):
    return {
        "id": a + "->" + b + ":" + kind,
        "from": a,
        "to": b,
        "kind": kind,
        "tensor": tensor(shape=shape),
    }


edges = [
    edge(a, b)
    for a, b in [
        ("input", "embed"),
        ("embed", "embed.reduce"),
        ("embed.reduce", "decoder"),
        ("decoder", "final_norm"),
        ("final_norm", "lm_head"),
        ("lm_head", "logits.gather"),
        ("logits.gather", "output"),
        ("norm.input", "attn"),
        ("attn", "norm.post"),
        ("norm.post", "moe"),
        ("attn.qkv", "attn.qk_norm"),
        ("attn.qk_norm", "attn.rope"),
        ("attn.rope", "attn.core"),
        ("attn.core", "attn.o"),
        ("attn.o", "attn.o.reduce"),
        ("moe.gate", "moe.experts.dispatch"),
        ("moe.experts.dispatch", "moe.w13"),
        ("moe.w13", "moe.act"),
        ("moe.act", "moe.w2"),
        ("moe.w2", "moe.experts.combine"),
        ("moe.experts.combine", "moe.experts.reduce"),
    ]
]
edges += [
    edge("norm.input", "norm.post", "residual"),
    edge("attn.core", "attn.core", "kv", ["context", "2", "max(1,Hkv/tp)", "D"]),
]


# Make raw semantic edges carry actual slots, independently of collapsed view edges.
def replace_edge(a, b, slot):
    matches = [e for e in edges if e["from"] == a and e["to"] == b and e["kind"] == "data"]
    assert len(matches) == 1
    matches[0]["tensor"] = slot


replace_edge("input", "embed", tensor("tokens", ["T"], "dtype_ids"))
replace_edge("final_norm", "lm_head", tensor("selected_hidden", selected_shape))
replace_edge("lm_head", "logits.gather", node_map["lm_head"]["out"][0])
replace_edge("logits.gather", "output", node_map["logits.gather"]["out"][0])
for a, b, slot in [
    ("attn.qkv", "attn.qk_norm", node_map["attn.qkv"]["out"][0]),
    ("attn.qk_norm", "attn.rope", node_map["attn.qk_norm"]["out"][0]),
    ("attn.rope", "attn.core", node_map["attn.rope"]["out"][0]),
    ("attn.core", "attn.o", node_map["attn.core"]["out"][0]),
    ("moe.gate", "moe.experts.dispatch", node_map["moe.gate"]["out"][0]),
    ("moe.experts.dispatch", "moe.w13", node_map["moe.experts.dispatch"]["out"][0]),
    ("moe.w13", "moe.act", node_map["moe.w13"]["out"][0]),
    ("moe.act", "moe.w2", node_map["moe.act"]["out"][0]),
    ("moe.w2", "moe.experts.combine", node_map["moe.w2"]["out"][0]),
]:
    replace_edge(a, b, slot)
for a, b, slot in [
    ("attn.qkv", "attn.qk_norm", node_map["attn.qkv"]["out"][1]),
    ("attn.qkv", "attn.core", node_map["attn.qkv"]["out"][2]),
    ("attn.qk_norm", "attn.rope", node_map["attn.qk_norm"]["out"][1]),
    ("attn.rope", "attn.core", node_map["attn.rope"]["out"][1]),
    ("moe.gate", "moe.experts.dispatch", tensor()),
]:
    e = edge(a, b)
    e["id"] += ":" + slot["name"]
    e["tensor"] = slot
    edges.append(e)
for e in edges:
    if e["kind"] == "kv":
        e["tensor"]["dtype"] = "dtype_kv"

ir = {
    "snapshot": snapshot,
    "symbols": symbols,
    "params": params,
    "nodes": sorted(nodes, key=lambda n: n["id"]),
    "edges": sorted(edges, key=lambda e: e["id"]),
    "groups": [
        {
            "id": "decoder",
            "layers": ["0", "L-1"],
            "template": "decoder",
            "representative_layer": "0",
        }
    ],
    "memory": {
        "weights": weights,
        "kv_per_token": kv,
        "kv_tokens": capacity,
        "activation_peak": activation,
        "non_torch": "non_torch_bytes",
        "reserved": "reserved_bytes",
        "unmodelled": [
            "allocator fragmentation",
            "graph capture workspace",
            "HCCL/NCCL workspace beyond configured non_torch_bytes",
            "RoPE tables and non-parameter buffers",
            "quantization scales/zeros and NZ padding",
            "backend-specific MoE transport buffers beyond gathered activations",
            "backend-specific attention scratch space",
            "input/output token index storage",
        ],
    },
    "views": {"family": "moe"},
    "notes": [
        "Curated fixture, not production extraction/lowering. Per-rank bytes; decimal GB=10^9, GiB=2^30. TP1/TP4 weight totals and shapes match experiment exactly.",
        "All symbols/metric dimensions are expression strings; source line numbers are provenance integers. T/S are local to one attention DP replica; context is separate cache length.",
        "Group metrics summarize descendants; decoder summarizes L/pp layers. For totals use memory.weights, never sum all nodes.",
        "PP choices deliberately divide L; first/last embeddings live on different stages for pp>1. Memory ledger chooses the final stage with the extra final norm as the heaviest equal-size rank.",
        "EP=1 disables EP; otherwise ep=tp*dp and E must be divisible by ep. EP and dp>1 formulas are uniform-routing estimates, not CPU/mock verified facts.",
        "TP node weights describe a representative local decoder stage. L0 endpoint weights are whole endpoint tensors; stage residency is applied in memory.weights.",
        "No latency predictions; bytes_moved, activation peaks and uncalibrated overheads are transparent approximations.",
    ],
}
write(OUT / "qwen3-30b-a3b/ir.json", ir)

# Independent arithmetic/oracle values, not a second DSL evaluator.
cases = []
for tp in [1, 4]:
    env = {
        "tp": tp,
        "pp": 1,
        "dp": 1,
        "ep": 1,
        "T": 4096,
        "S": 3,
        "batch": 2,
        "dtype_w": "bf16",
        "dtype_kv": "bf16",
        "dtype_a": "bf16",
    }
    expected = {
        "/memory/weights": base["build"]["param_bytes"]
        if tp == 1
        else base4["build"]["param_bytes"],
        "/memory/kv_per_token": 2 * 48 * max(1, 4 // tp) * 128 * 2,
        "/nodes/attn.qkv/weight_bytes": 2048 * (32 // tp + 2 * max(1, 4 // tp)) * 128 * 2,
        "/nodes/attn.qkv/flops/prefill": 2 * 4096 * 2048 * (32 // tp + 2 * max(1, 4 // tp)) * 128,
        "/nodes/attn.qkv/out/0/shape": [4096, 32 // tp, 128],
        "/nodes/moe.w13/weight_bytes": 2 * 128 * 2048 * 768 // tp * 2,
        "/nodes/moe.w2/weight_bytes": 128 * 2048 * 768 // tp * 2,
        "/nodes/moe.gate/weight_bytes": 2048 * 128 * 2,
        "/nodes/attn.o.reduce/comm/bus_bytes/prefill": 0
        if tp == 1
        else 2 * (tp - 1) / tp * 4096 * 2048 * 2,
    }
    expected.update(
        {
            "/nodes/lm_head/flops/prefill": 2 * 2 * 2048 * 151936 / tp,
            "/nodes/lm_head/flops/decode": 2 * 3 * 2048 * 151936 / tp,
            "/nodes/embed.reduce/comm/logical_bytes/prefill": 4096 * 2048 * 2,
            "/nodes/embed.reduce/comm/logical_bytes/decode": 3 * 2048 * 2,
            "/nodes/logits.gather/comm/logical_bytes/prefill": 2 * 151936 / tp * 2,
            "/nodes/logits.gather/comm/logical_bytes/decode": 3 * 151936 / tp * 2,
        }
    )
    cases.append({"id": f"tp{tp}-bf16-T4096", "env": env, "expected": expected})
# Unverified transport estimates, independent arithmetic for the EP/DP boundary cases.
for tp, dp, ep in [(4, 1, 4), (2, 2, 4), (1, 4, 4), (1, 4, 1)]:
    active_dispatch = dp > 1
    active_reduce = tp > 1 or ep > 1
    expected = {}
    for phase_name, tokens in [("prefill", 4096), ("decode", 3)]:
        for suffix in ["dispatch", "combine"]:
            expected[f"/nodes/moe.experts.{suffix}/comm/bus_bytes/{phase_name}"] = (
                (tp * dp - 1) / (tp * dp) * tokens * 2048 * 2 if active_dispatch else 0
            )
        expected[f"/nodes/moe.experts.reduce/comm/logical_bytes/{phase_name}"] = tokens * 2048 * 2
        expected[f"/nodes/moe.experts.reduce/comm/bus_bytes/{phase_name}"] = (
            2 * (tp - 1) / tp * tokens * 2048 * 2 if active_reduce else 0
        )
    expected["/nodes/moe.experts.dispatch/comm/when"] = int(active_dispatch)
    expected["/nodes/moe.experts.reduce/comm/when"] = int(active_reduce)
    cases.append(
        {
            "id": f"tp{tp}-dp{dp}-ep{ep}-comm-estimate",
            "env": env | {"tp": tp, "dp": dp, "ep": ep},
            "expected": expected,
        }
    )
cases.append(
    {
        "id": "tp4-decode-expert-traffic",
        "env": env | {"tp": 4, "dp": 1, "ep": 1, "S": 1},
        "expected": {
            "/nodes/moe.w13/bytes_moved/decode": 2 * 8 * 2048 * 192 * 2 + 8 * (2048 + 2 * 192) * 2,
            "/nodes/moe.act/bytes_moved/decode": 8 * 3 * 192 * 2,
            "/nodes/moe.w2/bytes_moved/decode": 8 * 2048 * 192 * 2 + 8 * (192 + 2048) * 2,
            "/memory/activation_peak": 256 * 151936 * 22,
        },
    }
)
cases.sort(key=lambda c: c["id"])
write(
    OUT / "qwen3-30b-a3b/eval.json",
    {"model": snapshot["model"]["hf_id"], "cases": cases},
)


# Views use curated semantic selections. Each row has orthogonal edge centers on 4px grid.
def graph(id, selections, collapsed):
    ns = []
    for i, key in enumerate(selections):
        ns.append(
            {
                "id": key,
                "ir_node": key,
                "x": 48 + i * 208,
                "y": 64,
                "w": 160,
                "h": 64,
                "label": by_id[key]["label"],
            }
        )

    def represented(a, b):
        via = {("embed", "decoder"): "embed.reduce", ("lm_head", "output"): "logits.gather"}.get(
            (a, b)
        )
        pairs = [(a, via), (via, b)] if via else [(a, b)]
        return sorted(
            e["id"] for e in edges if (e["from"], e["to"]) in pairs and e["kind"] == "data"
        )

    es = [
        {
            "id": f"e{i}",
            "ir_edges": represented(a["id"], b["id"]),
            "kind": "data",
            "label": next(
                e["tensor"]["shape"] for e in edges if e["id"] == represented(a["id"], b["id"])[0]
            ),
            "from": a["id"],
            "to": b["id"],
            "points": [[a["x"] + 160, 96], [b["x"], 96]],
            "label_pos": [a["x"] + 184, 80],
        }
        for i, (a, b) in enumerate(zip(ns, ns[1:], strict=False))
    ]
    return {
        "id": id,
        "nodes": sorted(ns, key=lambda n: n["id"]),
        "edges": es,
        "width": len(ns) * 208 + 48,
        "height": 192,
        "collapsed_of": sorted(collapsed),
    }


views = [
    graph(
        "L0",
        ["input", "embed", "decoder", "final_norm", "lm_head", "output"],
        ["decoder"],
    ),
    graph("L1:decoder", ["norm.input", "attn", "norm.post", "moe"], ["attn", "moe"]),
    graph(
        "L2:attn",
        ["attn.qkv", "attn.qk_norm", "attn.rope", "attn.core", "attn.o", "attn.o.reduce"],
        ["attn.q_norm", "attn.k_norm", "attn.rope.apply"],
    ),
    graph(
        "L2:moe",
        [
            "moe.gate",
            "moe.experts.dispatch",
            "moe.w13",
            "moe.act",
            "moe.w2",
            "moe.experts.combine",
            "moe.experts.reduce",
        ],
        ["moe.experts", "moe.experts.weights", "moe.experts.method"],
    ),
]
# Residual route around the layer row, avoiding node boxes.
views[1]["edges"].append(
    {
        "id": "residual",
        "ir_edges": ["norm.input->norm.post:residual"],
        "kind": "residual",
        "label": ["T", "H"],
        "from": "norm.input",
        "to": "norm.post",
        "points": [[128, 128], [128, 160], [544, 160], [544, 128]],
        "label_pos": [336, 160],
    }
)
write(OUT / "qwen3-30b-a3b/view.json", {"views": views})
write(
    OUT / "qwen3-30b-a3b/triage.json",
    {
        "snapshot": snapshot,
        "items": [
            {
                "kind": "conditional_override",
                "symbol": "vllm_ascend.platform.NPUPlatform.get_attn_backend_cls",
                "candidates": [
                    "vllm_ascend.attention.attention_v1.AscendAttentionBackend",
                    "vllm_ascend.attention.mla_v1.AscendMLABackend",
                ],
                "conditions": ["use_mla", "!use_mla"],
            },
            {
                "kind": "unknown_layer_class",
                "symbol": "fixture.FutureLinear",
                "used_by": ["fixture.UnverifiedArchitecture"],
            },
        ],
    },
)
write(
    OUT / "qwen3-30b-a3b/report.json",
    {
        "snapshot": snapshot,
        "checks": [
            {
                "id": "calibration",
                "status": "skip",
                "message": "Fixture has no NPU calibration.",
            },
            {
                "id": "schema",
                "status": "pass",
                "message": "T0 contract examples validate.",
            },
        ],
        "golden_diff": [],
        "source_link_failures": [],
    },
)
# Example annotation hash is script-derived from the exact target declaration span.
target = source(ascend_classes[attn + ".qkv_proj"])
source_lines = (SRC / target["repo"] / target["file"]).read_text().splitlines(keepends=True)
target_ast = next(
    n
    for n in ast.walk(ast.parse("".join(source_lines)))
    if isinstance(n, ast.ClassDef) and n.lineno == target["line"]
)
anchor_start = min([target_ast.lineno] + [d.lineno for d in target_ast.decorator_list])
anchor = "".join(source_lines[anchor_start - 1 : target_ast.end_lineno]).rstrip("\n") + "\n"
annotation = {
    "id": "fixture.ascend.attn.qkv",
    "target": target["symbol"],
    "anchor_hash": "sha256:" + hashlib.sha256(anchor.encode()).hexdigest(),
    "applies_when": "true",
    "resolves_to": target["symbol"],
    "level": "KERNEL",
    "tag": ["OOT"],
    "note": "Fixture-only annotation: bf16 parameter dimensions are unchanged in the A2 CPU/mock comparison.",
    "reviewed_by": "fixture (not a maintainer approval)",
}
write(OUT / "annotation.json", annotation)
write(OUT / "annotations.json", [annotation])
rule = {
    "id": "fixture.linear.row",
    "match": {
        "class": "vllm.model_executor.layers.linear.RowParallelLinear",
        "vllm": ">=0.32,<0.33",
        "ascend": ">=0.32,<0.33",
        "when": "true",
    },
    "context": ["in_features", "out_features", "reduce_results"],
    "shard": {"tp": "in"},
    "out": [tensor(shape=["T", "out_features"])],
    "weight": "in_features/tp*out_features*bytes(dtype_w)",
    "flops": phase("2*T*in_features/tp*out_features", "2*S*in_features/tp*out_features"),
    "bytes_moved": phase(
        "in_features/tp*out_features*bytes(dtype_w)+T*(in_features/tp+out_features)*bytes(dtype_a)",
        "in_features/tp*out_features*bytes(dtype_w)+S*(in_features/tp+out_features)*bytes(dtype_a)",
    ),
    "emits": [
        {
            "suffix": "reduce",
            "kind": "comm",
            "comm": comm(
                "all_reduce",
                "tp",
                "T*out_features*bytes(dtype_a)",
                "reduce_results && tp>1 ? 2*(tp-1)/tp*T*out_features*bytes(dtype_a) : 0",
                "reduce_results && tp>1",
            ),
        }
    ],
    "notes": [
        "GLOBAL context; rule divides by tp once. o_proj: in_features=Hq*D=4096, out_features=H=4096; down_proj: in_features=I=12288, out_features=H=4096 (Qwen3-8B).",
        "TP4 bf16 o_proj: local [4096,1024], weight=8388608 bytes, prefill FLOPs=8388608*T; down_proj: local [4096,3072], weight=25165824 bytes, prefill FLOPs=25165824*T. Both emit T*4096*2 logical bytes and 1.5 times that in ring traffic.",
    ],
}
write(OUT / "rule.json", rule)
write(OUT / "rules.json", [rule])

# Small dense IR, complete top-level accounting with one collapsed decoder template.
dense_snapshot = snapshot | {"model": model_ref("Qwen/Qwen3-8B", dense_config)}
dense_layer = "(H*(Hq/tp+2*max(1,Hkv/tp))*D+H*Hq*D/tp+2*D+3*H*I/tp+2*H)*bytes(dtype_w)"
dense_nodes = []
for key, cls, weight in [
    (
        "embed",
        "vllm.model_executor.layers.vocab_parallel_embedding.VocabParallelEmbedding",
        vw,
    ),
    (
        "decoder",
        "vllm.model_executor.models.qwen3.Qwen3DecoderLayer",
        f"({dense_layer})*(L/pp)",
    ),
    ("final_norm", "vllm.model_executor.layers.layernorm.RMSNorm", "H*bytes(dtype_w)"),
    (
        "lm_head",
        "vllm.model_executor.layers.vocab_parallel_embedding.ParallelLMHead",
        vw,
    ),
]:
    dense_nodes.append(
        {
            "id": key,
            "kind": "group" if key == "decoder" else "compute",
            "class": cls,
            "src": source(cls),
            "rule": "fixture.dense." + key,
            "label": key.replace("_", " "),
            "in": [tensor()],
            "out": [tensor()],
            "weight_bytes": weight,
            "flops": phase(),
            "bytes_moved": phase(),
            "parallel": {},
            "notes": [
                "Dense minimal fixture: detailed compute/activation traffic deferred to T3; zero is a placeholder, not a measured value."
            ],
        }
    )
# Endpoint communication/selection semantics are identical for Dense and MoE.
dense_nodes = [n for n in dense_nodes if n["id"] != "lm_head"]
for key in ["embed.reduce", "lm_head", "logits.gather", "output"]:
    dense_nodes.append(copy.deepcopy(node_map[key]))
next(n for n in dense_nodes if n["id"] == "embed")["in"] = [tensor("tokens", ["T"], "dtype_ids")]
dense_params = params | {
    "pp": param(
        "integer",
        1,
        "Equal pipeline partitions only.",
        choices=["1", "2", "3", "4", "6", "9", "12", "18", "36"],
    ),
    "ep": param("integer", 1, "Dense model has no expert parallelism.", choices=["1"]),
}
dw = f"({dense_layer})*(L/pp)+(pp==1 ? 2*({vw})+H*bytes(dtype_w) : ({vw})+H*bytes(dtype_w))"
da = "max(max_num_batched_tokens*(4*H+(Hq/tp+2*max(1,Hkv/tp))*D+3*I/tp)*bytes(dtype_a),max_num_seqs*V*(bytes(dtype_a)+3*bytes(fp32)+bytes(int64)))"
dense_ir = {
    "snapshot": dense_snapshot,
    "symbols": {
        k: str(v)
        for k, v in {
            "H": 4096,
            "L": dense_config["num_hidden_layers"],
            "Hq": 32,
            "Hkv": 8,
            "D": 128,
            "V": 151936,
            "I": dense_config["intermediate_size"],
        }.items()
    },
    "params": dense_params,
    "nodes": sorted(dense_nodes, key=lambda n: n["id"]),
    "edges": sorted(
        [
            edge("embed", "embed.reduce"),
            edge("embed.reduce", "decoder"),
            edge("decoder", "final_norm"),
            edge("final_norm", "lm_head", shape=selected_shape),
            edge("lm_head", "logits.gather", shape=local_logits),
            edge("logits.gather", "output", shape=full_logits),
        ],
        key=lambda e: e["id"],
    ),
    "groups": [
        {
            "id": "decoder",
            "layers": ["0", "L-1"],
            "template": "decoder",
            "representative_layer": "0",
        }
    ],
    "memory": {
        "weights": dw,
        "kv_per_token": kv,
        "kv_tokens": f"floor(max(0,hbm_bytes*gpu_memory_utilization-({dw})-({da})-non_torch_bytes-reserved_bytes)/(({kv})*block_size))*block_size",
        "activation_peak": da,
        "non_torch": "non_torch_bytes",
        "reserved": "reserved_bytes",
        "unmodelled": [
            "graph capture",
            "allocator fragmentation",
            "communication workspace",
            "detailed dense FLOP/traffic fixture",
        ],
    },
    "views": {"family": "dense"},
    "notes": [
        "Minimal dense contract fixture, not a calibrated performance model. Symbolic weight ledger includes Q/K norms and replicated decoder norms."
    ],
}
write(OUT / "qwen3-8b/ir.json", dense_ir)

versions = [
    {
        "id": "main-verified",
        "label": "main · verified 544f182",
        "vllm": VLLM,
        "ascend": "main",
    },
    {"id": "v0.32.0", "label": "v0.32.0 pair", "vllm": "v0.32.0", "ascend": "v0.32.0"},
]
models = [
    {
        "id": "qwen3-30b-a3b",
        "hf_id": "Qwen/Qwen3-30B-A3B",
        "label": "Qwen3 30B A3B",
        "family": "moe",
    },
    {
        "id": "qwen3-8b",
        "hf_id": "Qwen/Qwen3-8B",
        "label": "Qwen3 8B",
        "family": "dense",
    },
]
cells = []
for version, model, status, count in [
    ("main-verified", "qwen3-30b-a3b", "ok", 0),
    ("main-verified", "qwen3-8b", "triage", 2),
    ("v0.32.0", "qwen3-30b-a3b", "missing", 0),
    ("v0.32.0", "qwen3-8b", "missing", 0),
]:
    cells.append(
        {
            "version": version,
            "model": model,
            "device": "A2",
            "status": status,
            "triage_count": count,
            "data_path": None if status == "missing" else f"data/{version}-A2/{model}",
        }
    )
write(OUT / "index.json", {"versions": versions, "models": models, "cells": cells})

# YAML catalog emitted with ruamel.yaml by the declared generator dependency.


def hardware(id, device, label, peak, hbm, bandwidth, interconnect, url, notes):
    return {
        "id": id,
        "device": device,
        "label": label,
        "peak_tflops": {
            k: peak.get(k) for k in sorted(["fp32", "bf16", "fp16", "fp8", "int8", "int4"])
        },
        "hbm_gb": hbm,
        "bandwidth_gbps": bandwidth,
        "interconnect_gbps": interconnect,
        "source": "vendor spec",
        "source_url": url,
        "notes": notes,
    }


catalog = {
    "models": sorted(
        [
            {
                "id": "qwen3-30b-a3b",
                "family": "moe",
                **model_ref("Qwen/Qwen3-30B-A3B", config),
                "config": config,
            },
            {
                "id": "qwen3-8b",
                "family": "dense",
                **model_ref("Qwen/Qwen3-8B", dense_config),
                "config": dense_config,
            },
            {
                "id": "deepseek-v3",
                "family": "mla",
                **model_ref("deepseek-ai/DeepSeek-V3", deepseek_config),
                "config": deepseek_config,
            },
        ],
        key=lambda m: m["id"],
    ),
    "version_pairs": [
        {
            "id": v["id"],
            "vllm": v["vllm"],
            "ascend": v["ascend"],
            "pairing_file": ".github/vllm-main-verified.commit"
            if v["id"] == "main-verified"
            else ".github/vllm-release-tag.commit",
        }
        for v in versions
    ],
    "devices": [
        {"id": d, "hardware_preset": key}
        for d, key in [
            ("310P", "ascend-310p-pro"),
            ("A2", "atlas-800i-a2-64gb"),
            ("A3", "atlas-800i-a3-128gb"),
            ("A5", "atlas-a5-unconfirmed"),
        ]
    ],
    "hardware_presets": sorted(
        [
            hardware(
                "ascend-310p-pro",
                "310P",
                "Atlas 300I Pro (LPDDR4X, not HBM)",
                {"fp16": 70, "int8": 140},
                24,
                204.8,
                None,
                "https://e.huawei.com/cn/products/computing/ascend/atlas-300-ai",
                [
                    "INT8 is dense TOPS, stored under the common 10^12 operations/s field.",
                    "TODO: verify exact 310P SKU, bf16/fp32/int4 support and usable per-device PCIe interconnect; null fields are unknown, not zero.",
                ],
            ),
            hardware(
                "atlas-800i-a2-64gb",
                "A2",
                "Atlas 800I A2, 64 GB HCCS variant",
                {"fp16": 280, "bf16": 280, "fp32": 75, "int8": 560},
                64,
                1600,
                None,
                "https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a2",
                [
                    "TODO: verify per-device 280/75/560 peaks, bf16 equivalence and 1600 GB/s against the exact 64 GB board whitepaper; provisional vendor-spec figures, not measurements.",
                    "Vendor product page gives 392 GB/s at server interconnect scope; per-device/duplex convention is uncertain, so interconnect remains null.",
                    "TODO: fp8/int4 unknown for this SKU; do not derive throughput by dtype byte width.",
                ],
            ),
            hardware(
                "atlas-800i-a3-128gb",
                "A3",
                "Atlas 800I A3, 8 x 128 GB server normalized per accelerator",
                {"fp16": 560, "int8": 1120},
                128,
                3200,
                None,
                "https://e.huawei.com/cn/products/computing/ascend/atlas-800i-a3",
                [
                    "FP16 4.48 PFLOPS and INT8 8.96 POPS server totals divided by eight. INT8 uses TOPS.",
                    "TODO: confirm bf16/fp32/fp8/int4 peaks and logical-device vs package scope before enabling those rooflines.",
                    "Vendor advertises 784 GB/s interconnect without a clear per-device/duplex scope; null pending confirmation.",
                ],
            ),
            hardware(
                "atlas-a5-unconfirmed",
                "A5",
                "A5 / Atlas 950 profile, board SKU unconfirmed",
                {},
                None,
                None,
                None,
                "https://support.huaweicloud.com/resmgmt-modelarts/resmgmt-a5_0001.html",
                [
                    "TODO: pin a board SKU and confirm every dtype peak, memory capacity, bandwidth and interconnect from its vendor sheet. No invented values; all uncertain quantities are null.",
                    "Upstream profile identifier stays A5; vendor may label the product Atlas 950. Hide roofline until a numeric preset is selected.",
                ],
            ),
            hardware(
                "h100-sxm-80gb",
                "H100",
                "NVIDIA H100 SXM 80 GB (dense)",
                {"fp32": 67, "bf16": 989.5, "fp16": 989.5, "fp8": 1979, "int8": 1979},
                80,
                3350,
                900,
                "https://www.nvidia.com/en-us/data-center/h100/",
                [
                    "Tensor peaks are dense: vendor sparsity-enabled BF16/FP16 1979 and FP8/INT8 3958 divided by two. FP32 is CUDA-core FP32, not TF32.",
                    "NVLink is aggregate bidirectional 900 GB/s; INT8 is TOPS in the common operations/s field.",
                    "TODO: int4 throughput not published in this source; null.",
                ],
            ),
        ],
        key=lambda p: p["id"],
    ),
}
yaml = YAML()
yaml.width = 4096
yaml.indent(mapping=2, sequence=4, offset=2)


# Object keys sorted recursively for deterministic serialization, semantic lists preserved.
def sort_keys(value):
    if isinstance(value, dict):
        return {k: sort_keys(v) for k, v in sorted(value.items())}
    if isinstance(value, list):
        return [sort_keys(v) for v in value]
    return value


catalog_text = io.StringIO()
yaml.dump(sort_keys(catalog), catalog_text)
if (ROOT / "catalog.yaml").read_text() != catalog_text.getvalue():
    raise RuntimeError("Catalog differs; regeneration requires catalog owner coordination")
