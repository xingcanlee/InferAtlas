"""Architecture from CPU/meta facts; no forward execution or numeric verification gate."""

import json
import re
from collections import Counter, defaultdict
from itertools import groupby
from pathlib import Path

from pydantic import Field, TypeAdapter
from ruamel.yaml import YAML

from inferatlas.schema import IR, Arch, Extract, Snapshot, Triage
from inferatlas.schema.arch import (
    ArchAttention,
    ArchBand,
    ArchBlock,
    ArchChild,
    ArchComponent,
    ArchEmbedding,
    ArchExpert,
    ArchGeneric,
    ArchIO,
    ArchMLP,
    ArchModel,
    ArchModule,
    ArchModuleStage,
    ArchMoE,
    ArchNorm,
    ArchNote,
    ArchReplacement,
    ArchRope,
    ArchRouter,
    ArchSource,
    ArchStackStage,
    ArchSublayer,
    ArchSymbol,
    ArchTensor,
)
from inferatlas.schema.models import Contract

RULES = Path(__file__).resolve().parents[2] / "rules/architecture/components.yaml"
LAYER = re.compile(r"^(.*(?:layers|blocks|\.h))\.(\d+)$")


class StructureRule(Contract):
    kind: str
    variant: str = "other"
    order: list[str] = Field(default_factory=list)
    residual: str = "unknown"


def structure_rules() -> dict[str, StructureRule]:
    return TypeAdapter(dict[str, StructureRule]).validate_python(YAML(typ="safe").load(RULES))


def emit_arch(
    extract: Extract,
    ir: IR,
    triage: Triage,
    *,
    model_id: str,
    family: str,
) -> Arch:
    """Use actual layer structures, including unknown children, rather than layer_types labels."""
    if extract.snapshot != ir.snapshot or triage.snapshot != ir.snapshot:
        raise ValueError("architecture inputs disagree on snapshot identity")
    if not extract.modules:
        raise ValueError("architecture requires an extracted module tree")
    modules = {m.path: m for m in extract.modules}
    if len(modules) != len(extract.modules):
        raise ValueError("duplicate extracted module path")
    rules = structure_rules()
    config = {**extract.config, **(extract.config.get("text_config") or {})}
    children = defaultdict(list)
    for module in sorted(extract.modules, key=lambda m: m.path):
        if module.path:
            children[module.path.rpartition(".")[0]].append(module)
    observed_dtype = next(
        (
            p.dtype
            for m in sorted(extract.modules, key=lambda m: m.path)
            for p in sorted(m.params, key=lambda p: p.name)
        ),
        "bfloat16",
    )
    dtype = str(config.get("torch_dtype") or config.get("dtype") or observed_dtype)
    hidden = int(config.get("hidden_size") or 0)
    heads = int(config.get("num_attention_heads") or 0)
    kv_heads = int(config.get("num_key_value_heads") or heads)
    symbols = {
        "T": ArchSymbol(value=4096, description="tokens in the batch"),
        "H": ArchSymbol(value=hidden, description="hidden size"),
        "V": ArchSymbol(value=int(config.get("vocab_size") or 0), description="vocabulary size"),
    }
    if "T" in ir.params:
        symbols["T"].value = int(ir.params["T"].default)
    for key, name, description in (
        ("num_attention_heads", "Hq", "query heads"),
        ("num_key_value_heads", "Hkv", "KV heads"),
        ("head_dim", "D", "head width"),
        ("intermediate_size", "I", "dense intermediate width"),
        ("moe_intermediate_size", "I_moe", "expert intermediate width"),
    ):
        if config.get(key) is not None:
            symbols[name] = ArchSymbol(value=int(config[key]), description=description)
    symbols["tp"] = ArchSymbol(value=extract.parallel.tp, description="Tensor parallel degree")
    symbols.setdefault("Hkv", ArchSymbol(value=kv_heads, description="KV heads"))
    symbols.setdefault(
        "D",
        ArchSymbol(
            value=int(config.get("qk_nope_head_dim", 0) + config.get("qk_rope_head_dim", 0))
            or (hidden // heads if heads else 0),
            description="Head width",
        ),
    )
    if extract.parallel.ep or extract.parallel.dp > 1:
        symbols["dp"] = ArchSymbol(
            value=extract.parallel.dp, description="Attention data parallel degree"
        )
        symbols["ep"] = ArchSymbol(
            value=extract.parallel.tp * extract.parallel.dp if extract.parallel.ep else 1,
            description="Expert parallel degree",
        )
    notes = []
    targets = defaultdict(set)
    visible = {}
    generic_seen = set()
    residual_seen = set()

    def descendants(module):
        return [
            m
            for p, m in sorted(modules.items())
            if not module.path or p == module.path or p.startswith(module.path + ".")
        ]

    def tensor(name="hidden_states", shape=None):
        return ArchTensor(name=name, shape=shape if shape is not None else ["T", "H"], dtype=dtype)

    def rule_for(module):
        return rules.get(module.class_name)

    def component(module) -> ArchComponent:
        vision = any(k in module.path for k in ("vision", "visual"))
        cfg = (extract.config.get("vision_config") or {}) if vision else config
        hidden = int(cfg.get("hidden_size") or module.attrs.get("hidden_size") or 0)
        heads = int(
            cfg.get("num_attention_heads")
            or module.attrs.get("total_num_heads")
            or module.attrs.get("num_heads")
            or 0
        )
        kv_heads = int(
            cfg.get("num_key_value_heads") or module.attrs.get("total_num_kv_heads") or heads
        )
        rule = rule_for(module)
        subtree = descendants(module)
        if rule is None or rule.kind in {"decoder", "container"}:
            return ArchGeneric(
                class_name=module.class_name,
                children=[
                    ArchChild(name=m.path.rsplit(".", 1)[-1], class_name=m.class_name)
                    for m in children[module.path]
                ],
            )
        attrs = module.attrs
        if rule.kind == "norm":
            return ArchNorm(
                variant=rule.variant,
                eps=attrs.get(
                    "variance_epsilon",
                    attrs.get("eps", cfg.get("rms_norm_eps", cfg.get("layer_norm_eps"))),
                ),
            )
        if rule.kind in {"embedding", "lm_head"}:
            return ArchEmbedding(
                kind=rule.kind,
                vocab=int(cfg.get("vocab_size") or 0),
                hidden=hidden,
                tied=bool(cfg.get("tie_word_embeddings", False)),
            )
        activation = str(cfg.get("hidden_act", "silu"))
        variant = {"silu": "swiglu", "swiglu": "swiglu", "gelu": "gelu", "geglu": "geglu"}.get(
            activation, "other"
        )
        if rule.kind == "mlp":
            down = next((m for m in subtree if m.path.endswith((".down_proj", ".fc2"))), None)
            own_intermediate = down.attrs.get("input_size") if down else None
            return ArchMLP(
                variant=variant,
                intermediate=int(
                    attrs.get(
                        "intermediate_size", own_intermediate or cfg.get("intermediate_size", 0)
                    )
                ),
            )
        if rule.kind == "moe":
            experts = int(
                attrs.get(
                    "num_experts",
                    cfg.get(
                        "num_experts",
                        cfg.get("n_routed_experts", cfg.get("num_local_experts", 0)),
                    ),
                )
            )
            topk = int(attrs.get("top_k", cfg.get("num_experts_per_tok", 0)))
            shared_count = int(cfg.get("n_shared_experts", cfg.get("num_shared_experts", 0)) or 0)
            shared_module = next(
                (
                    m
                    for m in subtree
                    if m.path.rsplit(".", 1)[-1] in {"shared_expert", "shared_experts"}
                ),
                None,
            )
            if shared_module and shared_count == 0:
                shared_count = 1
            intermediate = int(cfg.get("moe_intermediate_size", cfg.get("intermediate_size", 0)))
            shared_intermediate = int(
                cfg.get("shared_expert_intermediate_size", intermediate * shared_count)
            )
            return ArchMoE(
                experts=experts,
                top_k=topk,
                shared_experts=shared_count,
                router=ArchRouter(
                    scoring="hash"
                    if any(p.name == "tid2eid" for m in subtree for p in m.params)
                    else str(attrs.get("scoring_func", cfg.get("scoring_func", "softmax"))),
                    norm_topk=attrs.get("norm_topk_prob", cfg.get("norm_topk_prob")),
                    groups={"n_group": cfg["n_group"], "topk_group": cfg.get("topk_group", 1)}
                    if cfg.get("n_group")
                    else None,
                ),
                expert=ArchExpert(variant=variant, intermediate=intermediate),
                shared=ArchExpert(variant=variant, intermediate=shared_intermediate)
                if shared_count
                else None,
            )
        if rule.kind == "attention":
            variant = rule.variant
            if variant == "other":
                variant = (
                    "mqa" if kv_heads == 1 and heads > 1 else "gqa" if kv_heads < heads else "mha"
                )
            index = next((int(p) for p in module.path.split(".") if p.isdigit()), 0)
            vision = any(k in module.path for k in ("vision", "visual"))
            layer_type = (
                extract.layer_types[index]
                if not vision and index < len(extract.layer_types)
                else ""
            )
            inner = next((m for m in subtree if m.path == module.path + ".attn"), module)
            sliding = inner.attrs.get("sliding_window", attrs.get("per_layer_sliding_window"))
            explicit_sliding = (
                "sliding_window" in inner.attrs or "per_layer_sliding_window" in attrs
            )
            if not explicit_sliding and cfg.get("use_sliding_window"):
                sliding = (
                    cfg.get("sliding_window")
                    if index >= int(cfg.get("max_window_layers", 0))
                    else None
                )
            if layer_type == "linear_attention":
                variant = "linear"
            if not explicit_sliding and layer_type == "sliding_attention":
                sliding = cfg.get("sliding_window")
            extras = {
                k: cfg[k]
                for k in (
                    "q_lora_rank",
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                    "index_topk",
                    "o_lora_rank",
                    "o_groups",
                )
                if k in cfg
            }
            if "compress_ratio" in attrs:
                extras["compress_ratio"] = attrs["compress_ratio"]
                extras["indexer"] = any(
                    m.path == module.path + ".indexer"
                    or m.path.startswith(module.path + ".indexer.")
                    for m in subtree
                )
            if "compress_ratio" in attrs:
                # V4 shares a single KV vector across query heads, not an MLA latent cache.
                variant = "mqa"
                kv_heads = 1
                sliding = cfg.get("sliding_window", sliding)
            if variant == "linear":
                for key, source in (
                    ("num_k_heads", "linear_num_key_heads"),
                    ("num_v_heads", "linear_num_value_heads"),
                    ("head_k_dim", "linear_key_head_dim"),
                    ("head_v_dim", "linear_value_head_dim"),
                    ("conv_kernel", "linear_conv_kernel_dim"),
                ):
                    extras[key] = attrs.get(key, cfg.get(source, 0))
                extras.update(kind="gated_deltanet", output_gate=attrs.get("output_gate", True))
            if "ChunkedLocalAttention" in inner.class_name:
                extras["chunk_size"] = inner.attrs.get(
                    "attention_chunk_size", cfg.get("attention_chunk_size")
                )
                extras["kind"] = "chunked_local"
            if sliding:
                extras["sliding_window"] = sliding
            rope = None
            if (
                variant != "linear"
                and not attrs.get("nope")
                and (cfg.get("rope_theta") is not None or any("rotary" in m.path for m in subtree))
            ):
                scaling = cfg.get("rope_scaling")
                rope = ArchRope(
                    kind=str(
                        (scaling or {}).get("rope_type", (scaling or {}).get("type", "default"))
                    ),
                    theta=cfg.get("rope_theta"),
                    scaling=scaling,
                    partial_rotary_factor=cfg.get("partial_rotary_factor"),
                )
            return ArchAttention(
                variant=variant,
                heads_q=int(
                    extras.get(
                        "num_v_heads", attrs.get("total_num_heads", attrs.get("num_heads", heads))
                    )
                ),
                heads_kv=int(
                    extras.get(
                        "num_k_heads",
                        1
                        if "compress_ratio" in attrs
                        else attrs.get("total_num_kv_heads", attrs.get("num_kv_heads", kv_heads)),
                    )
                ),
                head_dim=int(
                    extras.get("head_k_dim")
                    or attrs.get("head_dim")
                    or cfg.get("head_dim")
                    or ((cfg.get("qk_nope_head_dim") or 0) + (cfg.get("qk_rope_head_dim") or 0))
                    or (hidden // heads if heads else 0)
                ),
                rope=rope,
                qk_norm=bool(attrs.get("use_qk_norm"))
                or any(
                    m.path.rsplit(".", 1)[-1]
                    in {"q_norm", "k_norm", "q_layernorm", "k_layernorm", "qk_norm"}
                    for m in subtree
                ),
                extras=extras,
            )
        raise ValueError(f"unsupported architecture rule kind: {rule.kind}")

    def boundary(module, comp):
        if module.path.rsplit(".", 1)[-1] == "hc_head_op":
            symbols["HC"] = ArchSymbol(
                value=int(config.get("hc_mult", 1)), description="Hyper-connection streams"
            )
            return ArchIO(inputs=[tensor("residual_streams", ["T", "HC", "H"])], outputs=[tensor()])
        if module.path.rsplit(".", 1)[-1].startswith("mhc_"):
            symbols["HC"] = ArchSymbol(
                value=int(config.get("hc_mult", 1)), description="Hyper-connection streams"
            )
            stream = tensor("residual_streams", ["T", "HC", "H"])
            post = tensor("post_mix", ["T", "HC"])
            mix = tensor("res_mix", ["T", "HC", "HC"])
            name = module.path.rsplit(".", 1)[-1]
            if name == "mhc_pre":
                return ArchIO(inputs=[stream], outputs=[post, mix, tensor()])
            inputs = [tensor(), stream, post, mix]
            outputs = [stream] if name == "mhc_post" else [stream, post, mix, tensor()]
            return ArchIO(inputs=inputs, outputs=outputs)
        if comp.kind == "embedding":
            return ArchIO(
                inputs=[ArchTensor(name="input_ids", shape=["T"], dtype="int64")],
                outputs=[tensor()],
            )
        if comp.kind == "lm_head":
            return ArchIO(inputs=[tensor()], outputs=[tensor("logits", ["T", "V"])])
        if comp.kind in {"norm", "attention", "mlp", "moe"}:
            subtree = descendants(module)
            width = module.attrs.get("hidden_size") or module.attrs.get("input_size")
            if width is None:
                norm = next(
                    (
                        p
                        for m in subtree
                        for p in m.params
                        if len(p.shape) == 1 and p.name == "weight"
                    ),
                    None,
                )
                projection = next(
                    (
                        m
                        for m in subtree
                        if m.path.endswith(
                            ("qkv_proj", "q_proj", "gate_up_proj", "gate_proj", "fc1")
                        )
                    ),
                    None,
                )
                width = projection.attrs.get("input_size") if projection else None
                vision = any(k in module.path for k in ("vision", "visual"))
                cfg = (extract.config.get("vision_config") or {}) if vision else config
                down = next((m for m in subtree if m.path.endswith(".fc2")), None)
                width = width or (down.attrs.get("output_size") if down else None)
                width = width or (norm.shape[0] if norm else cfg.get("hidden_size", 0))
            h = "H" if width == hidden else str(width)
            vision = any(k in module.path for k in ("vision", "visual"))
            token = "P" if vision else "T"
            if vision:
                symbols["P"] = ArchSymbol(
                    value=1024, description="Vision patch tokens in the batch"
                )
            inputs, outputs = [tensor(shape=[token, h])], [tensor(shape=[token, h])]
            internals = []

            def out(name, dim, scalar_dtype=None):
                t = tensor(name, [token, dim])
                if scalar_dtype:
                    t.dtype = scalar_dtype
                internals.append(t)

            if comp.kind == "attention":
                if comp.variant == "linear":
                    kd = comp.heads_kv * comp.head_dim
                    vd = comp.heads_q * int(comp.extras.get("head_v_dim", comp.head_dim))
                    out("qkvz", f"{2 * kd + 2 * vd}/tp")
                    out("ba", f"{2 * comp.heads_q}/tp")
                    out("conv_out", f"{2 * kd + vd}/tp")
                    symbols["B"] = ArchSymbol(value=1, description="Sequences in the batch")
                    internals.append(
                        tensor(
                            "recurrent_state",
                            [
                                "B",
                                f"{comp.heads_q}/tp",
                                str(comp.head_dim),
                                str(comp.extras.get("head_v_dim", comp.head_dim)),
                            ],
                        )
                    )
                elif h == "H" and comp.heads_q == heads and comp.heads_kv == kv_heads:
                    symbols.setdefault("Hq", ArchSymbol(value=heads, description="Query heads"))
                    out("q", "Hq*D/tp")
                    out("k", "max(1,Hkv/tp)*D")
                    out("v", "max(1,Hkv/tp)*D")
                else:
                    out("q", f"{comp.heads_q}*{comp.head_dim}/tp")
                    out("k", f"max(1,{comp.heads_kv}/tp)*{comp.head_dim}")
                    out("v", f"max(1,{comp.heads_kv}/tp)*{comp.head_dim}")
                if comp.extras.get("v_head_dim") and comp.variant == "mla":
                    v = next(t for t in internals if t.name == "v")
                    v.shape = ["T", f"{comp.heads_q}*{comp.extras['v_head_dim']}/tp"]
                if comp.extras.get("qk_rope_head_dim") and comp.variant == "mla":
                    out("k_rope", str(comp.extras["qk_rope_head_dim"]))
                for key in ("q_lora_rank", "kv_lora_rank"):
                    if comp.extras.get(key):
                        out(key.removesuffix("_lora_rank") + "_latent", str(comp.extras[key]))
                ratio = int(comp.extras.get("compress_ratio") or 1)
                if ratio > 1:
                    internals.append(
                        tensor("kv_compressed", [f"cdiv(T,{ratio})", str(comp.head_dim)])
                    )
                if comp.extras.get("indexer") or (
                    comp.variant == "mla" and comp.extras.get("index_topk")
                ):
                    internals.append(tensor("index_scores", ["T", f"cdiv(T,{ratio})"]))
                if comp.variant != "linear":
                    out("o", h)
            elif comp.kind in {"mlp", "moe"}:
                inter = comp.intermediate if comp.kind == "mlp" else comp.expert.intermediate
                if comp.kind == "moe":
                    # Distinct stacks can have distinct expert counts.
                    suffix = (
                        ""
                        if "E" not in symbols or symbols["E"].value == comp.experts
                        else "_" + str(comp.experts)
                    )
                    e, k = "E" + suffix, "K" + suffix
                    symbols[e] = ArchSymbol(value=comp.experts, description="Routed experts")
                    symbols[k] = ArchSymbol(value=comp.top_k, description="Experts per token")
                    out("router_logits", e)
                    out("topk_ids", k, "int64")
                    out("topk_weights", k)
                    routed = f"T*dp*{k}/ep" if "dp" in symbols else f"T*{k}"
                    internals.append(tensor("expert_input", [routed, h]))
                    packed = (
                        f"2*{inter}" if comp.expert.variant in {"swiglu", "geglu"} else str(inter)
                    )
                    width = packed if extract.parallel.ep else f"{packed}/tp"
                    if not extract.parallel.ep and extract.parallel.dp > 1:
                        width = f"{packed}/(tp*dp)"
                    internals.append(tensor("expert_gate_up", [routed, width]))
                    internals.append(tensor("expert_output", [routed, h]))
                    if comp.shared:
                        out("shared_output", h)
                else:
                    packed = f"2*{inter}" if comp.variant in {"swiglu", "geglu"} else str(inter)
                    out("gate_up", f"{packed}/tp")
                    out("act", f"{inter}/tp")
            return ArchIO(inputs=inputs, outputs=outputs, internals=internals)
        # Weight orientation is PyTorch Linear's [out, in]; never run unknown forwards.
        weight = next((p for p in module.params if p.name == "weight" and len(p.shape) == 2), None)
        if weight:
            return ArchIO(
                inputs=[tensor(shape=["T", str(weight.shape[1])])],
                outputs=[tensor("output", ["T", str(weight.shape[0])])],
            )
        if (
            module.attrs.get("input_size") is not None
            and module.attrs.get("output_size") is not None
        ):
            return ArchIO(
                inputs=[tensor(shape=["T", str(module.attrs["input_size"])])],
                outputs=[tensor("output", ["T", str(module.attrs["output_size"])])],
            )
        # Only sequential containers establish first/last execution order.
        if module.class_name == "torch.nn.modules.container.Sequential":
            direct = children[module.path]
            names = {m.path.rsplit(".", 1)[-1] for m in direct}
            if names != {str(i) for i in range(len(direct))}:
                return ArchIO(inputs=[], outputs=[])
            ordered = sorted(direct, key=lambda m: int(m.path.rsplit(".", 1)[-1]))
            if ordered:
                first, last = ordered[0], ordered[-1]
                return ArchIO(
                    inputs=boundary(first, component(first)).inputs,
                    outputs=boundary(last, component(last)).outputs,
                )
        return ArchIO(inputs=[], outputs=[])

    def param_shape(m, p):
        shape = [str(d) for d in p.shape]
        cls = m.class_name.rsplit(".", 1)[-1]
        tp = extract.parallel.tp
        local = m.path.rsplit(".", 1)[-1]
        text = not any(k in m.path for k in ("vision", "visual"))
        if text and p.name in {"weight", "bias"} and shape:
            if local == "qkv_proj" and "Hq" in symbols and "Hkv" in symbols:
                shape[0] = "(Hq/tp+2*max(1,Hkv/tp))*D"
                if len(shape) > 1:
                    shape[1] = "H"
                return shape
            if local == "o_proj" and len(shape) > 1 and "Hq" in symbols:
                return ["H", "Hq*D/tp"]
        if not m.attrs.get("disable_tp") and shape:
            axis = (
                0
                if any(
                    k in cls
                    for k in ("ColumnParallel", "QKVParallel", "VocabParallel", "ParallelLMHead")
                )
                else 1
                if "RowParallel" in cls and len(shape) > 1
                else None
            )
            if axis is not None:
                shape[axis] = f"{p.shape[axis] * tp}/tp"
            if p.name in {"w13_weight", "w2_weight"} and len(shape) == 3:
                if extract.parallel.ep:
                    shape[0] = f"{p.shape[0] * tp * extract.parallel.dp}/ep"
                else:
                    axis = 1 if p.name == "w13_weight" else 2
                    degree = tp * extract.parallel.dp
                    shape[axis] = (
                        f"{p.shape[axis] * degree}/(tp*dp)"
                        if extract.parallel.dp > 1
                        else f"{p.shape[axis] * tp}/tp"
                    )
        if p.name == "w13_bias" and len(shape) == 2 and not extract.parallel.ep:
            degree = tp * extract.parallel.dp
            shape[1] = (
                f"{p.shape[1] * degree}/(tp*dp)"
                if extract.parallel.dp > 1
                else f"{p.shape[1] * tp}/tp"
            )
        if (
            shape
            and "E" in symbols
            and str(symbols["E"].value) == shape[0]
            and ("expert" in m.path or local in {"gate", "router"})
        ):
            shape[0] = "E"
        return shape

    def module_ref(module, comp, target):
        targets[module.class_name].add(target)
        subtree = descendants(module)
        for child in subtree:
            targets[child.class_name].add(target)
            if child.ascend:
                targets[child.ascend.class_name].add(target)
        ref = module.ascend.src if module.ascend else module.src
        source = None
        if ref:
            repo = {
                "vllm": "vllm-project/vllm",
                "vllm-ascend": "vllm-project/vllm-ascend",
                "pytorch": "pytorch/pytorch",
            }[ref.repo]
            source = ArchSource(
                file=ref.file,
                line=ref.line,
                url=f"https://github.com/{repo}/blob/{ref.commit}/{ref.file}#L{ref.line}",
            )
        status = "generic" if comp.kind == "generic" else "ruled"
        replaced = None
        if module.ascend:
            replaced = ArchReplacement(
                baseline_class=module.class_name, ascend_class=module.ascend.class_name
            )
            reviewed = any(
                n.class_name == module.class_name and n.ascend and n.ascend.annotation_id
                for n in ir.nodes
            )
            unresolved = any(
                item.kind == "conditional_override"
                and (
                    item.symbol == module.class_name or module.ascend.class_name in item.candidates
                )
                for item in triage.items
            )
            status = "ascend_override" if reviewed and not unresolved else "unannotated_override"
        params = [
            ArchTensor(
                name=(
                    (m.path[len(module.path) :].lstrip(".") + ".") if m.path != module.path else ""
                )
                + p.name,
                shape=param_shape(m, p),
                dtype=p.dtype,
            )
            for m in subtree
            for p in sorted(m.params, key=lambda p: p.name)
        ]
        visible[target] = (module, comp, status)
        if comp.kind == "generic" and module.class_name not in generic_seen:
            generic_seen.add(module.class_name)
            notes.append(
                ArchNote(
                    severity="info",
                    target=target,
                    text=f"Generic component: {module.class_name}; I/O is derived only from parameter/constructor or sequential-child boundaries.",
                    triage_kind=None,
                )
            )
        return ArchModule(
            flags=module.flags,
            path=module.path,
            class_name=module.class_name,
            source=source,
            status=status,
            replaced=replaced,
            params=sorted(params, key=lambda p: p.name),
        )

    def sublayer(module, band_id, local_id=None):
        local_id = local_id or module.path.rsplit(".", 1)[-1]
        comp = component(module)
        role = {"norm": "norm", "attention": "mixer", "mlp": "ffn", "moe": "ffn"}.get(
            comp.kind,
            "mixer"
            if module.path.rsplit(".", 1)[-1] in {"self_attn", "attn", "linear_attn"}
            else "ffn"
            if module.path.rsplit(".", 1)[-1]
            in {"mlp", "moe", "ffn", "feed_forward", "block_sparse_moe"}
            else "other",
        )
        return ArchSublayer(
            id=local_id,
            role=role,
            component=comp,
            io=boundary(module, comp),
            module=module_ref(module, comp, band_id + "/" + local_id),
        )

    def block(module, band_id):
        rule = rule_for(module)
        direct = children[module.path]
        order = (
            rule.order
            if rule and rule.kind == "decoder"
            else [
                "input_layernorm",
                "layer_norm1",
                "attn_norm",
                "self_attn",
                "attn",
                "linear_attn",
                "post_attention_layernorm",
                "layer_norm2",
                "ffn_norm",
                "mlp",
                "ffn",
                "feed_forward",
                "block_sparse_moe",
            ]
        )
        ranked = {name: i for i, name in enumerate(order)}
        ordered = sorted(
            direct, key=lambda m: (ranked.get(m.path.rsplit(".", 1)[-1], len(order)), m.path)
        )
        residual = rule.residual if rule and rule.kind == "decoder" else "unknown"
        if residual == "hyper_connection":
            # Emit the equivalent unfused forward, reusing each pre/post module twice.
            by_name = {m.path.rsplit(".", 1)[-1]: m for m in direct}
            operations = (
                ("mhc_pre", "mhc_pre_attn"),
                ("attn_norm", "attn_norm"),
                ("attn", "attn"),
                ("mhc_post", "mhc_post_attn"),
                ("mhc_pre", "mhc_pre_ffn"),
                ("ffn_norm", "ffn_norm"),
                ("ffn", "ffn"),
                ("mhc_post", "mhc_post_ffn"),
            )
            return ArchBlock(
                residual=residual,
                sublayers=[
                    sublayer(by_name[name], band_id, local_id)
                    for name, local_id in operations
                    if name in by_name
                ],
            )
        subs = []
        for child in ordered:
            sub = sublayer(child, band_id)
            subs.append(sub)
            if (residual in {"pre_norm", "post_norm"} and sub.role in {"mixer", "ffn"}) or (
                residual == "sandwich"
                and sub.role == "norm"
                and len(subs) > 1
                and subs[-2].role in {"mixer", "ffn"}
            ):
                subs.append(
                    ArchSublayer(
                        id=f"residual_{sum(s.role == 'residual_add' for s in subs) + 1}",
                        role="residual_add",
                        component=None,
                        io=ArchIO(
                            inputs=[
                                tensor(
                                    "branch", sub.io.outputs[0].shape if sub.io.outputs else None
                                ),
                                tensor(
                                    "residual", sub.io.outputs[0].shape if sub.io.outputs else None
                                ),
                            ],
                            outputs=[
                                tensor(shape=sub.io.outputs[0].shape if sub.io.outputs else None)
                            ],
                        ),
                        module=None,
                    )
                )
        if not subs:
            subs = [sublayer(module, band_id)]
        if residual == "unknown" and module.class_name not in residual_seen:
            residual_seen.add(module.class_name)
            notes.append(
                ArchNote(
                    severity="warn",
                    target=band_id + "/" + subs[0].id,
                    text=f"Residual topology for {module.class_name} is unknown; children follow the decoder-slot heuristic.",
                    triage_kind=None,
                )
            )
        return ArchBlock(residual=residual, sublayers=subs)

    layer_modules = [
        (m, LAYER.fullmatch(m.path)) for m in extract.modules if LAYER.fullmatch(m.path)
    ]
    stacks = defaultdict(list)
    for module, match in layer_modules:
        stacks[match[1]].append((int(match[2]), module))
    stages = []
    stack_stages = {}
    consumed = set()
    for stack_path, layers in sorted(stacks.items()):
        occurrence = Counter(
            (m.path[len(layer.path) :], m.class_name)
            for _, layer in layers
            for m in descendants(layer)
            if not m.params and not children[m.path]
        )
        shared_once = {key for key, count in occurrence.items() if count == 1 and len(layers) > 1}
        bands = []
        last_signature = None
        for index, module in sorted(layers):
            subtree = descendants(module)
            # Ignore shared parameter-free RoPE modules deduplicated by named_modules.
            signature = json.dumps(
                [
                    (
                        m.path[len(module.path) :],
                        m.class_name,
                        {
                            k: v
                            for k, v in m.attrs.items()
                            if k not in {"prefix", "layer_idx", "layer_id"}
                        },
                        [
                            (p.name, p.shape, p.dtype)
                            for p in sorted(m.params, key=lambda p: p.name)
                        ],
                        m.ascend.class_name if m.ascend else None,
                    )
                    for m in subtree
                    if "rope" not in m.path
                    and "rotary" not in m.path
                    and (m.path[len(module.path) :], m.class_name) not in shared_once
                ],
                sort_keys=True,
            )
            # Include derived semantics (e.g. per-layer sliding-window configuration).
            semantics = [
                (component(m).model_dump(), boundary(m, component(m)).model_dump())
                for m in children[module.path]
                if "rotary" not in m.path
                and "rope" not in m.path
                and (m.path[len(module.path) :], m.class_name) not in shared_once
            ]
            signature += json.dumps(semantics, sort_keys=True)
            if bands and signature == last_signature and index == bands[-1].layers[1] + 1:
                bands[-1].layers = (bands[-1].layers[0], index)
                bands[-1].count += 1
            else:
                bands.append(
                    ArchBand(
                        id=f"{stack_path}:band:{index}",
                        layers=(index, index),
                        count=1,
                        pattern=None,
                        block=block(module, f"{stack_path}:band:{index}"),
                    )
                )
            last_signature = signature
            consumed.update(m.path for m in subtree)
        sequence, labels = [], []
        for band in bands:
            attention = next(
                (
                    s.component
                    for s in band.block.sublayers
                    if s.component and s.component.kind == "attention"
                ),
                None,
            )
            label = "other"
            if attention:
                label = "linear" if attention.variant == "linear" else attention.variant.upper()
                extras = attention.extras
                if extras.get("sliding_window"):
                    label += " SWA"
                if extras.get("compress_ratio", 1) > 1:
                    label = attention.variant.upper() + f" ÷{extras['compress_ratio']}"
                if extras.get("indexer"):
                    label += " + indexer"
                if extras.get("chunk_size"):
                    label += f" chunk {extras['chunk_size']}"
            # Include FFN, RoPE and mixer semantics, not layer paths or registration order.
            signature = json.dumps(
                [
                    (s.role, s.component.model_dump() if s.component else None)
                    for s in band.block.sublayers
                ],
                sort_keys=True,
            )
            sequence.extend([signature] * band.count)
            labels.extend([label] * band.count)
        patterns = [None] * len(sequence)
        start = 0
        while start < len(sequence):
            for period in range(2, (len(sequence) - start) // 2 + 1):
                unit = sequence[start : start + period]
                if len(set(unit)) < 2 or sequence[start + period : start + 2 * period] != unit:
                    continue
                end = start + 2 * period
                while end < len(sequence) and sequence[end] == unit[(end - start) % period]:
                    end += 1
                pattern = " + ".join(
                    f"{len(list(items))}× {label}"
                    for label, items in groupby(labels[start : start + period])
                )
                patterns[start:end] = [pattern] * (end - start)
                start = end
                break
            else:
                start += 1
        offset = 0
        for band in bands:
            band.pattern = patterns[offset]
            offset += band.count
        stack_stages[stack_path] = ArchStackStage(
            id=stack_path,
            label="Vision layers"
            if any(k in stack_path for k in ("vision", "visual"))
            else "Decoder layers",
            bands=bands,
        )
        consumed.add(stack_path)

    def top(module):
        if module.path in stack_stages:
            stages.append(stack_stages[module.path])
            return
        if module.path in consumed:
            return
        if "logits_processor" in module.path or module.class_name.endswith("LogitsProcessor"):
            return
        if children[module.path] and (
            not module.path or any(p.startswith(module.path + ".") for p in stacks)
        ):
            registration = {name: i for i, name in enumerate(module.attrs.get("child_order", []))}

            def stage_order(child):
                name = child.path.rsplit(".", 1)[-1]
                rule = rule_for(child)
                kind = (
                    "container"
                    if any(p == child.path or p.startswith(child.path + ".") for p in stacks)
                    else rule.kind
                    if rule
                    else "other"
                )
                if any(k in name for k in ("vision", "visual")) and kind == "container":
                    rank = 0.5
                elif name == "multi_modal_projector":
                    rank = 0.75
                elif name in {"patch_embedding", "embeddings"}:
                    rank = 0
                elif name == "hc_head_op":
                    rank = 1.5
                elif "norm" in name and any(k in name for k in ("pre", "before")):
                    rank = 0
                else:
                    rank = {
                        "embedding": 0,
                        "stack": 1,
                        "container": 1,
                        "norm": 2,
                        "lm_head": 3,
                    }.get(kind, 4)
                return (
                    rank,
                    registration.get(name, len(registration)),
                    child.path,
                )

            for child in sorted(children[module.path], key=stage_order):
                top(child)
            return
        comp = component(module)
        kind = comp.kind if comp.kind in {"embedding", "norm", "lm_head"} else "other"
        if any(k in module.path for k in ("vision", "visual")) and module.path.rsplit(".", 1)[
            -1
        ] in {"patch_embedding", "embeddings"}:
            kind = "embedding"
        label = {"embedding": "Token embedding", "norm": "Final norm", "lm_head": "LM head"}.get(
            kind, module.path or module.class_name.rsplit(".", 1)[-1]
        )
        stages.append(
            ArchModuleStage(
                id=module.path or "root",
                kind=kind,
                placement="aside"
                if any(k in module.path.lower() for k in ("mtp", "draft"))
                else "spine",
                label=label,
                module=module_ref(module, comp, module.path or "root"),
                io=boundary(module, comp),
                component=comp,
            )
        )
        consumed.update(m.path for m in descendants(module))

    roots = [
        m
        for m in sorted(extract.modules, key=lambda m: m.path)
        if not m.path or m.path.rpartition(".")[0] not in modules
    ]
    for root in roots:
        top(root)
    # Aside branches follow the complete spine regardless of registration order.
    stages.sort(key=lambda stage: getattr(stage, "placement", "spine") == "aside")
    seen = set()
    for item in triage.items:
        symbol = getattr(item, "symbol", getattr(item, "target", None))
        key = (symbol, item.kind)
        if key in seen:
            continue
        seen.add(key)
        affected = sorted(targets.get(symbol, ()))
        direct = [t for t in affected if t in visible and visible[t][0].class_name == symbol]
        if item.kind in {"unknown_layer_class", "conditional_override"}:
            direct = [
                t
                for t in direct
                if visible[t][1].kind == "generic" or visible[t][2] == "unannotated_override"
            ]
            if not direct:
                continue
        target = direct[0] if direct else None
        data = item.model_dump(mode="json")
        if item.kind == "conditional_override":
            candidates = ", ".join(data.get("candidates", [])) or "an unresolved implementation"
            text = f"Ascend replaces {symbol} with {candidates}; not yet annotated."
        elif item.kind == "unknown_layer_class":
            text = f"No architecture rule describes {symbol}; shown as a generic component."
        else:
            detail = data.get("message") or data.get("diff")
            detail = detail if isinstance(detail, str) else None
            text = (
                item.kind.replace("_", " ").capitalize()
                + (f" for {symbol}" if symbol else "")
                + (f": {detail}" if detail else ".")
            )
        changes_figure = target is not None and visible[target][1].kind == "generic"
        notes.append(
            ArchNote(
                severity="warn" if changes_figure else "info",
                target=target,
                text=text,
                triage_kind=item.kind,
            )
        )
    from inferatlas.expr import validate

    used = set()
    for stage in stages:
        refs = (
            [s for b in stage.bands for s in b.block.sublayers]
            if isinstance(stage, ArchStackStage)
            else [stage]
        )
        for ref in refs:
            for t in [
                *ref.io.inputs,
                *ref.io.outputs,
                *ref.io.internals,
                *(ref.module.params if ref.module else []),
            ]:
                for d in t.shape:
                    used.update(validate(d))
    used.update({"T", "V"})
    symbols = {k: v for k, v in symbols.items() if k in used}
    architectures = config.get("architectures", [])
    return Arch(
        flags=extract.flags,
        model=ArchModel(
            id=model_id,
            hf_id=extract.snapshot.model.hf_id,
            architecture=architectures[0]
            if architectures
            else modules.get("", modules[min(modules)]).class_name.rsplit(".", 1)[-1],
            family=family,
        ),
        device=extract.snapshot.device,
        symbols=dict(sorted(symbols.items())),
        input=ArchTensor(name="input_ids", shape=["T"], dtype="int64"),
        output=tensor("logits", ["T", "V"]),
        stages=stages,
        notes=sorted(
            {n.model_dump_json(): n for n in notes}.values(),
            key=lambda n: (n.target or "", n.triage_kind or "", n.text),
        ),
    )


def validate_arch_identity(arch: Arch, snapshot: Snapshot, model_id: str) -> None:
    """Validate published architecture identity and its immutable source URL commits."""
    if (arch.model.id, arch.model.hf_id, arch.device) != (
        model_id,
        snapshot.model.hf_id,
        snapshot.device,
    ):
        raise ValueError("architecture disagrees with build identity")
    for stage in arch.stages:
        refs = (
            [s.module for b in stage.bands for s in b.block.sublayers]
            if isinstance(stage, ArchStackStage)
            else [stage.module]
        )
        for module in refs:
            if module is None or module.source is None:
                continue
            source = module.source
            urls = [
                f"https://github.com/{repo}/blob/{revision.commit}/{source.file}#L{source.line}"
                for repo, revision in (
                    ("vllm-project/vllm", snapshot.vllm),
                    ("vllm-project/vllm-ascend", snapshot.ascend),
                )
            ]
            if source.url not in urls and not re.fullmatch(
                r"https://github.com/pytorch/pytorch/blob/[0-9a-f]{40}/"
                + re.escape(source.file)
                + r"#L"
                + str(source.line),
                source.url,
            ):
                raise ValueError("architecture source URL disagrees with snapshot identity")
