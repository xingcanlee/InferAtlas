"""Architecture from CPU/meta facts; no forward execution or numeric verification gate."""

import json
import re
from collections import defaultdict
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
    notes = []
    targets = defaultdict(set)

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
                    attrs.get("eps", config.get("rms_norm_eps", config.get("layer_norm_eps"))),
                ),
            )
        if rule.kind in {"embedding", "lm_head"}:
            return ArchEmbedding(
                kind=rule.kind,
                vocab=int(config.get("vocab_size") or 0),
                hidden=hidden,
                tied=bool(config.get("tie_word_embeddings", False)),
            )
        activation = str(config.get("hidden_act", "silu"))
        variant = {"silu": "swiglu", "swiglu": "swiglu", "gelu": "gelu", "geglu": "geglu"}.get(
            activation, "other"
        )
        if rule.kind == "mlp":
            return ArchMLP(
                variant=variant,
                intermediate=int(
                    attrs.get("intermediate_size", config.get("intermediate_size", 0))
                ),
            )
        if rule.kind == "moe":
            experts = int(
                attrs.get(
                    "num_experts",
                    config.get(
                        "num_experts",
                        config.get("n_routed_experts", config.get("num_local_experts", 0)),
                    ),
                )
            )
            topk = int(attrs.get("top_k", config.get("num_experts_per_tok", 0)))
            shared_count = int(
                config.get("n_shared_experts", config.get("num_shared_experts", 0)) or 0
            )
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
            intermediate = int(
                config.get("moe_intermediate_size", config.get("intermediate_size", 0))
            )
            shared_intermediate = int(
                config.get("shared_expert_intermediate_size", intermediate * shared_count)
            )
            return ArchMoE(
                experts=experts,
                top_k=topk,
                shared_experts=shared_count,
                router=ArchRouter(
                    scoring=str(attrs.get("scoring_func", config.get("scoring_func", "softmax"))),
                    norm_topk=attrs.get("norm_topk_prob", config.get("norm_topk_prob")),
                    groups=config.get("n_group"),
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
            layer_type = extract.layer_types[index] if index < len(extract.layer_types) else ""
            sliding = attrs.get("sliding_window", attrs.get("per_layer_sliding_window"))
            if (
                sliding is None
                and config.get("sliding_window")
                and config.get("use_sliding_window", True)
            ):
                sliding = (
                    config["sliding_window"]
                    if index
                    < int(config.get("max_window_layers", config.get("num_hidden_layers", 0)))
                    else None
                )
            if layer_type == "linear_attention":
                variant = "linear"
            elif layer_type == "sliding_attention" or sliding:
                variant = "sliding_window"
            extras = {
                k: config[k]
                for k in (
                    "q_lora_rank",
                    "kv_lora_rank",
                    "qk_nope_head_dim",
                    "qk_rope_head_dim",
                    "v_head_dim",
                    "index_topk",
                )
                if k in config
            }
            if sliding:
                extras["sliding_window"] = sliding
            rope = None
            if variant != "linear" and (
                config.get("rope_theta") is not None or any("rotary" in m.path for m in subtree)
            ):
                scaling = config.get("rope_scaling")
                rope = ArchRope(
                    kind=str(
                        (scaling or {}).get("rope_type", (scaling or {}).get("type", "default"))
                    ),
                    theta=config.get("rope_theta"),
                    scaling=scaling,
                )
            return ArchAttention(
                variant=variant,
                heads_q=heads,
                heads_kv=kv_heads,
                head_dim=int(
                    config.get("head_dim")
                    or (
                        (config.get("qk_nope_head_dim") or 0)
                        + (config.get("qk_rope_head_dim") or 0)
                    )
                    or (hidden // heads if heads else 0)
                ),
                rope=rope,
                qk_norm=any(
                    m.path.rsplit(".", 1)[-1] in {"q_norm", "k_norm", "q_layernorm", "k_layernorm"}
                    for m in subtree
                ),
                extras=extras,
            )
        raise ValueError(f"unsupported architecture rule kind: {rule.kind}")

    def boundary(module, comp):
        if comp.kind == "embedding":
            return ArchIO(
                inputs=[ArchTensor(name="input_ids", shape=["T"], dtype="int64")],
                outputs=[tensor()],
            )
        if comp.kind == "lm_head":
            return ArchIO(inputs=[tensor()], outputs=[tensor("logits", ["T", "V"])])
        if comp.kind in {"norm", "attention", "mlp", "moe"}:
            return ArchIO(inputs=[tensor()], outputs=[tensor()])
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
                shape=[str(d) for d in p.shape],
                dtype=p.dtype,
            )
            for m in subtree
            for p in sorted(m.params, key=lambda p: p.name)
        ]
        if comp.kind == "generic":
            notes.append(
                ArchNote(
                    severity="info",
                    target=target,
                    text=f"Generic component: {module.class_name}; I/O is derived only from parameter/constructor or sequential-child boundaries.",
                    triage_kind=None,
                )
            )
        return ArchModule(
            path=module.path,
            class_name=module.class_name,
            source=source,
            status=status,
            replaced=replaced,
            params=sorted(params, key=lambda p: p.name),
        )

    def sublayer(module):
        comp = component(module)
        role = {"norm": "norm", "attention": "mixer", "mlp": "ffn", "moe": "ffn"}.get(
            comp.kind, "other"
        )
        return ArchSublayer(
            id=module.path,
            role=role,
            component=comp,
            io=boundary(module, comp),
            module=module_ref(module, comp, module.path),
        )

    def block(module):
        rule = rule_for(module)
        direct = children[module.path]
        order = rule.order if rule and rule.kind == "decoder" else []
        ranked = {name: i for i, name in enumerate(order)}
        ordered = sorted(
            direct, key=lambda m: (ranked.get(m.path.rsplit(".", 1)[-1], len(order)), m.path)
        )
        residual = rule.residual if rule and rule.kind == "decoder" else "unknown"
        subs = []
        for child in ordered:
            sub = sublayer(child)
            subs.append(sub)
            if residual == "pre_norm" and sub.role in {"mixer", "ffn"}:
                subs.append(
                    ArchSublayer(
                        id=child.path + ":residual",
                        role="residual_add",
                        component=ArchGeneric(class_name="ResidualAdd", children=[]),
                        io=ArchIO(
                            inputs=[tensor("branch"), tensor("residual")], outputs=[tensor()]
                        ),
                        module=None,
                    )
                )
        if not subs:
            subs = [sublayer(module)]
        if residual == "unknown":
            notes.append(
                ArchNote(
                    severity="warn",
                    target=module.path if not direct else direct[0].path,
                    text="Execution order and residual topology are unknown; children are listed by module path.",
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
    consumed = set()
    for stack_path, layers in sorted(stacks.items()):
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
                        [
                            (p.name, p.shape, p.dtype)
                            for p in sorted(m.params, key=lambda p: p.name)
                        ],
                        m.ascend.class_name if m.ascend else None,
                    )
                    for m in subtree
                    if "rotary_emb" not in m.path
                ],
                sort_keys=True,
            )
            # Include derived semantics (e.g. per-layer sliding-window configuration).
            semantics = [
                (component(m).model_dump(), boundary(m, component(m)).model_dump())
                for m in children[module.path]
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
                        block=block(module),
                    )
                )
            last_signature = signature
            consumed.update(m.path for m in subtree)
        stages.append(ArchStackStage(id=stack_path, label="Decoder layers", bands=bands))
        consumed.add(stack_path)

    def top(module):
        if module.path in consumed:
            return
        if (
            module.path in {"", "model", "transformer", "language_model", "language_model.model"}
            and children[module.path]
        ):
            for child in children[module.path]:
                top(child)
            rule = rule_for(module)
            if rule and rule.kind == "container" and not module.params:
                return
        comp = component(module)
        kind = comp.kind if comp.kind in {"embedding", "norm", "lm_head"} else "other"
        label = {"embedding": "Token embedding", "norm": "Final norm", "lm_head": "LM head"}.get(
            kind, module.path or module.class_name.rsplit(".", 1)[-1]
        )
        stages.append(
            ArchModuleStage(
                id=module.path or "root",
                kind=kind,
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
    rank = {"embedding": 0, "stack": 1, "norm": 2, "lm_head": 3, "other": 4}
    stages.sort(key=lambda s: (rank[s.kind], s.id))
    for item in triage.items:
        symbol = getattr(item, "symbol", getattr(item, "target", None))
        affected = sorted(targets.get(symbol, ())) or [None]
        data = item.model_dump(mode="json")
        text = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        for target in affected:
            notes.append(ArchNote(severity="warn", target=target, text=text, triage_kind=item.kind))
    architectures = config.get("architectures", [])
    return Arch(
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
