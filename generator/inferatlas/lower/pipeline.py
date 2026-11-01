"""Extract facts to symbolic IR and numeric review notes, without a runtime DSL evaluator."""

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version

from inferatlas.schema.models import (
    IR,
    Comm,
    ConditionalOverride,
    Extract,
    GoldenDiffItem,
    HardwareProfileIncomplete,
    IREdge,
    IRNode,
    LayerGroup,
    Memory,
    ParallelAxis,
    ParamSpec,
    PhaseExpr,
    TensorSpec,
    Triage,
    UnknownLayerClass,
    VersionMismatch,
    ViewFamily,
)

from .context import DTYPES, bind_context, substitute
from .rules import DEFAULT_RULES, RuleMatchError, RuleSpec, load_rules, match_rule

# Development snapshots have no PEP 440 tag. Bind only the explicitly verified revision.
VERIFIED_VERSIONS = {"544f182d9f0c9bfe8bd7645525951d5e3692a542": "0.32.0"}
LAYER = re.compile(r"^model\.layers\.(\d+)(?:\.(.*))?$")
ZERO = PhaseExpr(prefill="0", decode="0")


def add(*terms: str) -> str:
    terms = tuple(t for t in terms if t != "0")
    return "+".join(f"({t})" for t in terms) or "0"


def phase_map(value: PhaseExpr, bindings: Mapping[str, str]) -> PhaseExpr:
    return PhaseExpr(**{k: substitute(v, bindings) for k, v in value.model_dump().items()})


def tensor_map(value: TensorSpec, bindings: Mapping[str, str]) -> TensorSpec:
    data = value.model_dump()
    shape = data["shape"]
    data["shape"] = (
        [substitute(x, bindings) for x in shape]
        if isinstance(shape, list)
        else {k: [substitute(x, bindings) for x in v] for k, v in shape.items()}
    )
    data["dtype"] = substitute(data["dtype"], bindings)
    return TensorSpec.model_validate(data)


def comm_map(value: Comm, bindings: Mapping[str, str]) -> Comm:
    return value.model_copy(
        update={
            "when": substitute(value.when, bindings),
            "logical_bytes": phase_map(value.logical_bytes, bindings),
            "bus_bytes": phase_map(value.bus_bytes, bindings),
        }
    )


def hardware_preset(device_id: str):
    from inferatlas.resolve import load_catalog

    data = load_catalog()
    device = next((d for d in data.devices if d.id == device_id), None)
    return next(
        (
            p
            for p in data.hardware_presets
            if p.id == (device.hardware_preset if device else "")
            or (device is None and p.device == device_id)
        ),
        None,
    )


def hardware_profile_triage(extract: Extract) -> HardwareProfileIncomplete | None:
    device = extract.snapshot.device
    preset = hardware_preset(device)
    if preset is not None and preset.hbm_gb is not None:
        return None
    return HardwareProfileIncomplete(
        kind="hardware_profile_incomplete",
        device=device,
        preset=preset.id if preset else None,
        message=f"{device} preset {preset.id if preset else '<missing>'} has no confirmed "
        "device memory capacity; pin a board SKU/capacity before numeric publication.",
    )


def runtime_params(extract: Extract) -> dict[str, ParamSpec]:
    c = extract.config
    moe = bool(c.get("num_experts", 0))
    hq, hkv, layers = (
        int(c["num_attention_heads"]),
        int(c["num_key_value_heads"]),
        int(c["num_hidden_layers"]),
    )
    dtype = DTYPES.get(str(c.get("torch_dtype", "bfloat16")), str(c.get("torch_dtype")))
    if dtype not in {"bf16", "fp16", "fp32"}:
        raise ValueError(f"unsupported unquantized model dtype: {dtype}")
    values: dict[str, ParamSpec] = {}

    def param(name, kind, default, description, **kw):
        values[name] = ParamSpec(kind=kind, default=str(default), description=description, **kw)

    tp_choices = [str(n) for n in (1, 2, 4, 8) if hq % n == 0 and (hkv % n == 0 or n % hkv == 0)]
    param(
        "tp",
        "integer",
        extract.parallel.tp,
        "Attention tensor parallel degree.",
        choices=tp_choices,
        constraint="Hq%tp==0 && (Hkv%tp==0 || tp%Hkv==0) && round_up(V,64)%tp==0"
        + ("" if moe else " && I%tp==0"),
    )
    pp_choices = [str(n) for n in range(1, layers + 1) if layers % n == 0]
    if len(set(extract.layer_types)) > 1:
        pp_choices = ["1"]
    param(
        "pp",
        "integer",
        extract.parallel.pp,
        "Equal layer partitions; heaviest stage ledger.",
        choices=pp_choices,
        constraint="pp==1" if len(set(extract.layer_types)) > 1 else "L%pp==0",
    )
    param(
        "dp",
        "integer",
        extract.parallel.dp,
        "Attention data replicas; expert TP flattens tp*dp.",
        min="1",
        step="1",
        constraint="ep>1 || I_moe%(tp*dp)==0" if moe else None,
    )
    param(
        "ep",
        "integer",
        extract.parallel.tp * extract.parallel.dp if extract.parallel.ep else 1,
        "Expert degree: 1 disables EP; otherwise ep must equal tp*dp and divide E."
        if moe
        else "Dense model has no expert parallelism.",
        min="1",
        step="1",
        constraint="ep==1 || (ep==tp*dp && E%ep==0)" if moe else "ep==1",
        choices=[
            str(n) for n in range(1, int(c["num_experts"]) + 1) if int(c["num_experts"]) % n == 0
        ]
        if moe
        else ["1"],
    )
    for key, default, desc in [
        ("T", 4096, "Local prefill chunk tokens; divisible by batch."),
        ("S", 1, "Local concurrent decode sequences."),
        ("batch", 1, "Equal-length prefill sequences, distinct from S."),
        ("context", 4096, "Existing cached tokens per decode sequence."),
        ("max_num_batched_tokens", 4096, "Profiling prefill token count, independent of context."),
        ("max_num_seqs", 256, "Sampler warmup concurrent sequences."),
    ]:
        param(key, "integer", default, desc, min="0" if key == "context" else "1", step="1")
    values["T"].constraint = "T%batch==0"
    for key in ("dtype_w", "dtype_a", "dtype_kv"):
        param(
            key,
            "dtype",
            dtype,
            "Unquantized scalar dtype; scale/packing overhead excluded.",
            choices=["bf16", "fp16", "fp32"],
        )
    param("dtype_ids", "dtype", "int64", "Token index scalar dtype.", choices=["int32", "int64"])
    param(
        "block_size",
        "integer",
        128,
        "Cache capacity rounds down to whole blocks.",
        choices=["16", "32", "64", "128"],
    )
    preset = hardware_preset(extract.snapshot.device)
    capacity = int(preset.hbm_gb * 1e9) if preset and preset.hbm_gb is not None else None
    param(
        "hbm_bytes",
        "number",
        capacity if capacity is not None else 0,
        "Per-device memory bytes; override with catalog hardware preset."
        if capacity is not None
        else "Unknown capacity; zero is a sentinel and hardware triage blocks numeric use.",
        min="1" if capacity is not None else "0",
    )
    param(
        "gpu_memory_utilization",
        "number",
        0.9,
        "Fraction of device memory budgeted to vLLM.",
        min="0.01",
        max="1",
    )
    for key in ("non_torch_bytes", "reserved_bytes"):
        param(
            key, "number", 0, "Explicit byte deduction; zero excludes unmeasured overhead.", min="0"
        )
    # Bind catalog defaults here so direct lower and CLI builds share the same ledger.
    if preset:
        for key in ("non_torch_bytes", "reserved_bytes"):
            values[key].default = str(getattr(preset, key))
            values[key].description = (
                "Catalog estimate for communication/runtime buffers."
                if key == "non_torch_bytes"
                else "Catalog estimate for graph capture/allocator reserve."
            )
    return dict(sorted(values.items()))


@dataclass
class LowerResult:
    ir: IR
    triage: Triage
    parameter_shapes: dict[str, dict[str, list[str]]]

    def write(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, model in (("ir", self.ir), ("triage", self.triage)):
            (directory / f"{name}.json").write_text(
                json.dumps(model.model_dump(by_alias=True, mode="json"), sort_keys=True, indent=2)
                + "\n"
            )


def coverage_gate(triage: Triage | dict) -> bool:
    """True permits numeric verification; structure emission never uses this gate."""
    return not Triage.model_validate(triage).items


def semantic_id(path: str, moe: bool) -> str:
    suffix = LAYER.fullmatch(path)
    name = suffix[2] if suffix else path
    aliases = {
        "": "model",
        "model": "model",
        "model.layers": "layers",
        "model.embed_tokens": "embed",
        "model.norm": "final_norm",
        "lm_head": "lm_head",
        "logits_processor": "logits",
        "sampler": "output",
        "input_layernorm": "norm.input",
        "post_attention_layernorm": "norm.post",
        "self_attn": "attn",
        "self_attn.qkv_proj": "attn.qkv",
        "self_attn.o_proj": "attn.o",
        "self_attn.q_norm": "attn.q_norm",
        "self_attn.k_norm": "attn.k_norm",
        "self_attn.rotary_emb": "attn.rope",
        "self_attn.rotary_emb.apply_rotary_emb": "attn.rope.apply",
        "self_attn.attn": "attn.core",
        "mlp": "moe" if moe else "mlp",
        "mlp.gate": "moe.gate",
        "mlp.experts": "moe.experts",
        "mlp.experts.routed_experts": "moe.experts.weights",
        "mlp.experts.routed_experts.quant_method": "moe.experts.method",
        "mlp.gate_up_proj": "mlp.gate_up",
        "mlp.down_proj": "mlp.down",
        "mlp.act_fn": "mlp.act",
    }
    if suffix and name is None:
        return "decoder"
    if name not in aliases:
        raise ValueError(f"unmapped module path: {path}")
    return aliases[name]


def _version(extract: Extract, explicit: str | None) -> str:
    if explicit:
        return str(Version(explicit))
    # The CPU experiment's installed dist metadata predates its .pth source checkout.
    # Immutable source identity therefore wins over that known-stale package string.
    if extract.snapshot.vllm.commit in VERIFIED_VERSIONS:
        return VERIFIED_VERSIONS[extract.snapshot.vllm.commit]
    try:
        version = Version(extract.snapshot.vllm.tag.removeprefix("v"))
        if not version.is_devrelease:
            return str(version)
    except InvalidVersion:
        pass
    raise ValueError("Unverified development snapshot needs an explicit vllm_version")


def lower(
    extract: Extract | dict | str | Path,
    rules: list[RuleSpec] | str | Path = DEFAULT_RULES,
    *,
    vllm_version: str | None = None,
    ascend_version: str | None = None,
    mro: Mapping[str, tuple[str, ...]] | None = None,
    moe_a2a: bool = False,
) -> LowerResult:
    """Lower all reachable modules; unknown/ambiguous/invalid facts block coverage.

    Numeric evaluation belongs to T2. Parameter formulas are exposed as metadata and
    weight tensor inputs for independent shape/bytes verification against extraction.
    """
    if isinstance(extract, str | Path):
        extract = Extract.model_validate_json(Path(extract).read_text())
    else:
        extract = Extract.model_validate(extract)
    if extract.config.get("model_type") not in {"qwen3", "qwen3_moe"}:
        from .structure import structure_ir

        return structure_ir(extract)
    rules = load_rules(rules) if isinstance(rules, str | Path) else rules
    version_error = None
    version = vllm_version or extract.snapshot.vllm.tag
    try:
        version = _version(extract, vllm_version)
        if not any(SpecifierSet(r.match.vllm).contains(version, prereleases=True) for r in rules):
            version_error = f"No rules support vLLM {version}"
    except (ValueError, InvalidVersion) as error:
        version_error = str(error)
    c = extract.config
    moe = bool(c.get("num_experts", 0))
    symbols = {
        "H": str(c["hidden_size"]),
        "L": str(c["num_hidden_layers"]),
        "Hq": str(c["num_attention_heads"]),
        "Hkv": str(c["num_key_value_heads"]),
        "D": str(c.get("head_dim", int(c["hidden_size"]) // int(c["num_attention_heads"]))),
        "V": str(c["vocab_size"]),
        "E": str(c.get("num_experts", 0)),
        "topk": str(c.get("num_experts_per_tok", 1)),
        "I": str(c["intermediate_size"]),
        "I_moe": str(c.get("moe_intermediate_size", c["intermediate_size"])),
    }
    if version_error:
        return LowerResult(
            ir=IR(
                snapshot=extract.snapshot,
                symbols=dict(sorted(symbols.items())),
                params=runtime_params(extract),
                nodes=[],
                edges=[],
                groups=[],
                memory=Memory(
                    weights="0",
                    kv_per_token="0",
                    kv_tokens="0",
                    activation_peak="0",
                    non_torch="0",
                    reserved="0",
                    unmodelled=["Unsupported vLLM version; metrics unavailable"],
                ),
                views=ViewFamily(family="moe" if moe else "dense"),
            ),
            triage=Triage(
                snapshot=extract.snapshot,
                items=sorted(
                    [
                        VersionMismatch(
                            kind="version_mismatch",
                            vllm_version=version,
                            expected_ranges=sorted({r.match.vllm for r in rules}),
                            message=version_error,
                        ),
                        *([issue] if (issue := hardware_profile_triage(extract)) else []),
                    ],
                    key=lambda item: item.kind,
                ),
            ),
            parameter_shapes={},
        )
    items, nodes, shapes, module_weights, kv_terms = [], {}, {}, {}, {}
    modules = {m.path: m for m in extract.modules}
    groups: list[LayerGroup] = []
    layer_types = extract.layer_types
    bands: list[tuple[int, int, str]] = []
    for i, kind in enumerate(layer_types):
        if not bands or kind != bands[-1][2]:
            bands.append((i, i, kind))
        else:
            bands[-1] = (bands[-1][0], i, kind)
    multi = len(bands) > 1
    selected: dict[int, str] = {}
    for start, end, _kind in bands:
        band = f"band_{start}_{end}" if multi else "decoder"
        selected[start] = band
        groups.append(
            LayerGroup(
                id=band,
                layers=(str(start), str(end)),
                template=f"{band}.decoder" if multi else "decoder",
                representative_layer=str(start),
            )
        )

    def problem(message: str):
        items.append(
            GoldenDiffItem(kind="golden_diff", model=extract.snapshot.model.hf_id, diff=message)
        )

    if not layer_types or len(layer_types) != int(c["num_hidden_layers"]):
        problem("layer_types must contain one entry per configured decoder layer")
    if unsupported := sorted(set(layer_types) - {"full_attention"}):
        problem(
            f"Unsupported layer_types: {unsupported}; only full_attention KV residency is modelled"
        )
    if c.get("sliding_window") and c.get("use_sliding_window", True):
        problem("Active sliding_window needs a window-aware attention and KV rule")
    if extract.parallel.pp != 1:
        problem(
            "Observed PP extraction needs a rank-specific partition ledger; lower currently requires PP1 facts"
        )
    if c.get("quantization_config"):
        problem("Quantized weights require explicit packing/scale rules")
    if len(modules) != len(extract.modules):
        problem("Duplicate module paths in extract")
    for required in ("model.embed_tokens", "model.norm", "lm_head", "logits_processor"):
        if required not in modules:
            problem(f"Missing required Qwen3 endpoint: {required}")
    for module in sorted(extract.modules, key=lambda m: m.path):
        try:
            rule = match_rule(
                module.class_name,
                rules,
                vllm_version=version,
                ascend_version=ascend_version,
                mro=(mro or {}).get(module.class_name, ()),
            )
            if module.ascend:
                replacement_rule = match_rule(
                    module.ascend.class_name,
                    rules,
                    vllm_version=version,
                    ascend_version=ascend_version,
                )
                if any(
                    getattr(rule, key) != getattr(replacement_rule, key)
                    for key in (
                        "weight",
                        "parameter_shapes",
                        "flops",
                        "emits",
                    )
                ):
                    raise RuleMatchError(module.ascend.class_name, [rule, replacement_rule])
        except RuleMatchError as error:
            if error.candidates:
                items.append(
                    ConditionalOverride(
                        kind="conditional_override",
                        symbol=error.symbol,
                        candidates=sorted(r.id for r in error.candidates),
                        conditions=sorted({r.match.when for r in error.candidates}),
                    )
                )
            else:
                items.append(
                    UnknownLayerClass(
                        kind="unknown_layer_class",
                        symbol=error.symbol,
                        used_by=[extract.snapshot.model.hf_id],
                    )
                )
            continue
        try:
            bindings = bind_context(module, extract, rule)
            param_shapes = {
                name: [substitute(x, bindings) for x in shape]
                for name, shape in rule.parameter_shapes.items()
            }
            required = set(param_shapes) - {"bias"}
            if module.path == "lm_head" and c.get("tie_word_embeddings", False):
                required.discard("weight")
            if missing := required - {p.name for p in module.params}:
                raise ValueError(f"{module.path}: missing parameters {sorted(missing)}")
            for p in module.params:
                if p.name not in param_shapes:
                    raise ValueError(f"{module.path}: unmodelled parameter {p.name}")
                if DTYPES.get(p.dtype, p.dtype) not in {"bf16", "fp16", "fp32"}:
                    raise ValueError(f"{module.path}: unsupported weight dtype {p.dtype}")
                if p.dtype != c.get("torch_dtype", "bfloat16"):
                    raise ValueError(
                        f"{module.path}: mixed parameter dtypes need dtype-specific rules"
                    )
            if any(
                module.attrs.get(key, False)
                for key in (
                    "skip_final_all_reduce",
                    "is_sequence_parallel",
                    "output_is_reduced",
                )
            ):
                raise ValueError(f"{module.path}: fused/SP reduction requires a specialized rule")
            shapes[module.path] = {p.name: param_shapes[p.name] for p in module.params}
            nid = semantic_id(module.path, moe)
        except KeyError as error:
            problem(f"{module.path}: missing key {error.args[0]}")
            continue
        except ValueError as error:
            problem(str(error))
            continue
        layer = LAYER.fullmatch(module.path)
        if layer and int(layer[1]) not in selected:
            continue
        prefix = f"{selected[int(layer[1])]}." if layer and multi else ""
        nid = prefix + nid
        if nid in {"model", "layers"}:
            continue
        if module.src is None:
            problem(f"{module.path}: semantic module has no source declaration")
            continue
        group = rule.id.startswith("container.") or nid.endswith(
            ("moe.experts", "moe.experts.weights", "attn.rope")
        )
        weight = substitute(rule.weight, bindings)
        module_weights[module.path] = weight
        if rule.kv_per_token:
            kv_terms[module.path] = substitute(rule.kv_per_token, bindings)
        output = [tensor_map(t, bindings) for t in rule.out]
        input_shape = ["T", "H"]
        if "in_features" in bindings:
            input_shape = [
                "T",
                bindings["in_features"] + ("/tp" if rule.shard.get("tp") == "in" else ""),
            ]
        if nid.endswith("lm_head"):
            input_tensor = TensorSpec(
                name="input", shape={"prefill": ["batch", "H"], "decode": ["S", "H"]}
            )
        else:
            input_tensor = TensorSpec(name="input", shape=input_shape)
        activation_inputs = [input_tensor]
        if nid == "embed":
            activation_inputs = [TensorSpec(name="tokens", shape=["T"], dtype="dtype_ids")]
        elif nid == "logits":
            activation_inputs = [
                TensorSpec(
                    name="logits",
                    shape={
                        "prefill": ["batch", "round_up(V,64)"],
                        "decode": ["S", "round_up(V,64)"],
                    },
                )
            ]
        elif nid == "output":
            activation_inputs = [
                TensorSpec(
                    name="logits",
                    shape={
                        "prefill": ["batch", "V"],
                        "decode": ["S", "V"],
                    },
                )
            ]
        elif nid.endswith(("attn.core", "attn.rope", "attn.rope.apply")):
            activation_inputs = [
                TensorSpec(name=name, shape=["T", heads, "D"])
                for name, heads in (("q", "Hq/tp"), ("k", "max(1,Hkv/tp)"), ("v", "max(1,Hkv/tp)"))
                if name != "v" or nid.endswith("attn.core")
            ]
        elif "norm_size" in bindings:
            activation_inputs = [
                TensorSpec(
                    name="input",
                    shape=["T", f"{bindings['norm_size']}*({bindings['norm_copies']})"],
                )
            ]
        elif nid.endswith("mlp.act"):
            activation_inputs = [TensorSpec(name="input", shape=["T", "2*I/tp"])]
        inputs = activation_inputs + [
            TensorSpec(name=p.name, shape=param_shapes[p.name], dtype="dtype_w")
            for p in module.params
        ]
        parent = prefix + "decoder" if layer and layer[2] else None
        for stem in ("moe.experts.weights", "attn.rope", "attn", "moe.experts", "moe", "mlp"):
            if nid.startswith(prefix + stem + "."):
                parent = prefix + stem
                break
        node = IRNode(
            id=nid,
            kind="group" if group else "compute",
            class_name=module.class_name,
            src=module.src,
            rule=rule.id,
            label=nid.removeprefix(prefix),
            parent=parent,
            inputs=inputs,
            out=output,
            weight_bytes=weight,
            flops=phase_map(rule.flops, bindings),
            bytes_moved=phase_map(rule.bytes_moved, bindings),
            comm=comm_map(rule.comm, bindings) if rule.comm else None,
            parallel={k: ParallelAxis(dim=v) for k, v in rule.shard.items()},
            notes=rule.notes,
        )
        nodes[nid] = node
        if module.ascend:
            node.notes = [
                *node.notes,
                "Observed Ascend replacement: "
                + module.ascend.class_name
                + "; reviewed significance is applied by the annotations stage.",
            ]
        for emission in rule.emits:
            if emission.suffix == "gather" and bindings.get("gather_output") == "false":
                continue
            cid = nid + "." + emission.suffix
            collective = comm_map(emission.comm, bindings)
            if moe_a2a and emission.suffix in {"dispatch", "combine"}:
                # Comm.op is static in IR; backend selection happens at lowering time.
                logical = PhaseExpr(
                    prefill="T*topk*H*bytes(dtype_a)", decode="S*topk*H*bytes(dtype_a)"
                )
                collective = collective.model_copy(
                    update={
                        "op": "all_to_all",
                        "group": "ep",
                        "when": "ep>1 && dp>1",
                        "logical_bytes": logical,
                        "bus_bytes": PhaseExpr(
                            **{
                                phase: f"ep>1 && dp>1 ? (ep-1)/ep*({value}) : 0"
                                for phase, value in logical.model_dump().items()
                            }
                        ),
                        "notes": [
                            "All-to-all kernel backend selected at lowering time; uniform routing, metadata/padding excluded."
                        ],
                    }
                )
            comm_in, comm_out = output, output
            if not moe_a2a and emission.suffix == "combine":
                comm_in = [TensorSpec(name="gathered", shape=["dp*T", "H"])]
            if nid == "logits":
                comm_in = [
                    TensorSpec(
                        name="logits_shard",
                        shape={
                            "prefill": ["batch", "round_up(V,64)/tp"],
                            "decode": ["S", "round_up(V,64)/tp"],
                        },
                    )
                ]
                comm_out = activation_inputs
            if nid == "logits" and module.attrs.get("use_all_gather", False):
                collective = collective.model_copy(
                    update={
                        "op": "all_gather",
                        "bus_bytes": PhaseExpr(
                            **{
                                k: f"tp>1 ? (tp-1)*({v}) : 0"
                                for k, v in collective.logical_bytes.model_dump().items()
                            }
                        ),
                        "notes": [
                            "Full collection on every rank; logical bytes are one local shard."
                        ],
                    }
                )
            nodes[cid] = IRNode(
                id=cid,
                kind="comm",
                class_name=module.class_name,
                src=module.src,
                rule=rule.id,
                label=collective.op,
                parent=nid if group else parent,
                inputs=comm_in,
                out=comm_out,
                weight_bytes="0",
                flops=ZERO,
                bytes_moved=ZERO,
                comm=collective,
            )
    # A partial fixture may repeat its representative layer. Production extracts may not.
    actual_layers = {int(m[1]) for p in modules if (m := LAYER.fullmatch(p))}
    if not extract.fixture and actual_layers != set(range(int(c["num_hidden_layers"]))):
        problem("Production extract is missing configured decoder layers")
    if not extract.fixture:

        def signature_layer(index):
            # vLLM caches a shared RoPE instance; named_modules emits it only once.
            # It has no parameter storage, but its representative arithmetic repeats.
            return {
                found[2]: (
                    m.class_name,
                    [(p.name, p.shape, p.dtype) for p in m.params],
                    m.attrs,
                    m.ascend.class_name if m.ascend else None,
                )
                for m in extract.modules
                if (found := LAYER.fullmatch(m.path))
                and int(found[1]) == index
                and found[2]
                and found[2]
                not in {"self_attn.rotary_emb", "self_attn.rotary_emb.apply_rotary_emb"}
            }

        for start, end, _ in bands:
            reference = signature_layer(start)
            for index in range(start + 1, end + 1):
                signature = signature_layer(index)
                if reference != signature:
                    problem(
                        f"Layer {index} differs from representative {start}; split layer_types bands"
                    )
    for g in groups:
        if g.template not in nodes:
            problem(f"Missing representative decoder module for {g.id}")

    # Engine sampling is outside named_modules on vLLM models. Preserve the observed
    # logits provenance without claiming that a sampler class was actually extracted.
    if "logits" in nodes and "output" not in nodes:
        source = nodes["logits"]
        nodes["output"] = source.model_copy(
            update={
                "id": "output",
                "kind": "io",
                "label": "Logits / sampling",
                "weight_bytes": "0",
                "inputs": source.out,
                "out": [
                    TensorSpec(
                        name="tokens",
                        dtype="dtype_ids",
                        shape={"prefill": ["batch"], "decode": ["S"]},
                    )
                ],
                "flops": ZERO,
                "bytes_moved": ZERO,
                "notes": [
                    "Sampling internals are outside the extracted model; fp32 workspace is included in activation_peak."
                ],
            }
        )

    for g in groups:
        prefix = g.id + "." if multi else ""
        core = nodes.get(prefix + "attn.core")
        if core:
            nodes[prefix + "kv_cache"] = core.model_copy(
                update={
                    "id": prefix + "kv_cache",
                    "kind": "io",
                    "label": "KV cache",
                    "parent": prefix + "attn",
                    "weight_bytes": "0",
                    "inputs": [],
                    "flops": ZERO,
                    "bytes_moved": ZERO,
                    "out": [
                        TensorSpec(
                            name="kv",
                            shape=["2", "S", "context", "max(1,Hkv/tp)", "D"],
                            dtype="dtype_kv",
                        )
                    ],
                    "notes": ["KV state input; capacity is reported by memory.kv_tokens."],
                }
            )
        weights_node = nodes.get(prefix + "moe.experts.weights")
        if weights_node is None:
            fused = nodes.get(prefix + "moe.experts")
            if fused and any(t.name == "w13_weight" for t in fused.inputs):
                weights_node = fused
        if weights_node:
            node = weights_node
            degree = "(ep>1 ? ep : tp*dp)"
            work = f"T*dp*topk*H*I_moe/{degree}"
            routed = "T*dp*topk/(ep>1 ? ep : 1)"
            inter = "I_moe/(ep>1 ? 1 : tp*dp)"
            touched = "min(E/(ep>1 ? ep : 1),T*dp*topk/(ep>1 ? ep : 1))"
            for suffix, multiplier, param in (
                ("w13", 4, "w13_weight"),
                ("act", 0, None),
                ("w2", 2, "w2_weight"),
            ):
                tensor = next((t for t in node.inputs if t.name == param), None)
                weight = (
                    "*".join(f"({d})" for d in tensor.shape) + "*bytes(dtype_w)" if tensor else "0"
                )
                flops = f"{multiplier}*{work}" if multiplier else f"4*T*dp*topk*I_moe/{degree}"
                in_dim = inter if suffix == "w2" else f"2*({inter})" if suffix == "act" else "H"
                out_dim = "H" if suffix == "w2" else f"2*({inter})" if suffix == "w13" else inter
                moved = f"({routed})*(({in_dim})+({out_dim}))*bytes(dtype_a)"
                if param:
                    share = 2 if suffix == "w13" else 1
                    moved = f"{share}*{touched}*H*({inter})*bytes(dtype_w)+({moved})"
                nodes[prefix + "moe." + suffix] = node.model_copy(
                    update={
                        "id": prefix + "moe." + suffix,
                        "parent": node.id,
                        "kind": "compute",
                        "label": suffix,
                        "weight_bytes": weight,
                        "flops": PhaseExpr(prefill=flops, decode=flops.replace("T", "S")),
                        "bytes_moved": PhaseExpr(prefill=moved, decode=moved.replace("T", "S")),
                        "out": [
                            TensorSpec(
                                name="output",
                                shape=[
                                    "T*dp*topk/(ep>1 ? ep : 1)",
                                    "H"
                                    if suffix == "w2"
                                    else ("2*" if suffix == "w13" else "")
                                    + "I_moe/(ep>1 ? 1 : tp*dp)",
                                ],
                            )
                        ],
                        "inputs": [
                            TensorSpec(name="routed", shape=[routed, in_dim]),
                            *[t for t in node.inputs if t.name == param],
                        ],
                        "notes": [
                            "Curated expert operation; provenance and rule inherited from RoutedExperts."
                        ],
                    }
                )
            # The group summarizes synthetic GEMMs; its own direct parameter ledger
            # remains in module_weights and is never counted twice in memory.weights.
            node.weight_bytes = "0"
            node.flops = ZERO
            node.bytes_moved = ZERO

    edges: dict[str, IREdge] = {}

    def edge(a: str, b: str, kind="data"):
        if a not in nodes or b not in nodes or a == b:
            return
        key = f"{a}->{b}:{kind}"
        edges[key] = IREdge(id=key, from_node=a, to=b, tensor=nodes[a].out[0], kind=kind)

    def chain(ids: list[str]):
        present = [i for i in ids if i in nodes]
        for a, b in zip(present, present[1:], strict=False):
            edge(a, b)

    for g in groups:
        prefix = g.id + "." if multi else ""

        def p(x, prefix=prefix):
            return prefix + x

        chain(
            [
                p(x)
                for x in [
                    "norm.input",
                    "attn.qkv",
                    "attn.rope",
                    "attn.core",
                    "attn.o",
                    "attn.o.reduce",
                    "norm.post",
                ]
            ]
        )
        edge(p("attn.qkv"), p("attn.q_norm"))
        edge(p("attn.qkv"), p("attn.k_norm"))
        edge(p("attn.q_norm"), p("attn.rope"))
        edge(p("attn.k_norm"), p("attn.rope"))
        chain(
            [
                p(x)
                for x in (
                    [
                        "norm.post",
                        "moe.gate",
                        "moe.experts.dispatch",
                        "moe.w13",
                        "moe.act",
                        "moe.w2",
                        "moe.experts.combine",
                        "moe.experts.reduce",
                    ]
                    if moe
                    else [
                        "norm.post",
                        "mlp.gate_up",
                        "mlp.gate_up.gather",
                        "mlp.act",
                        "mlp.down",
                        "mlp.down.reduce",
                    ]
                )
            ]
        )
        edge(p("norm.input"), p("norm.post"), "residual")
        tail = p("moe.experts.reduce" if moe else "mlp.down.reduce")
        edge(p("norm.post"), tail, "residual")
        edge(p("kv_cache"), p("attn.core"), "kv")
        # Parameter-bearing groups retain metrics but are not independently executed twice.
    chain(
        [
            "embed",
            "embed.reduce",
            *[g.template for g in groups],
            "final_norm",
            "lm_head",
            "logits.gather",
            "logits",
            "output",
        ]
    )
    # Containers may be absent in minimal inputs: preserve a valid hierarchy and block missing bands.
    for n in nodes.values():
        if n.parent not in nodes:
            n.parent = None

    # Aggregate existing descendants from the deepest containers outward.
    def aggregate(n):
        children = [ch for ch in nodes.values() if ch.parent == n.id]
        for child in children:
            if child.kind == "group":
                aggregate(child)
        if not children:
            return
        n.weight_bytes = add(n.weight_bytes, *(ch.weight_bytes for ch in children))
        n.flops = PhaseExpr(
            **{
                phase: add(getattr(n.flops, phase), *(getattr(ch.flops, phase) for ch in children))
                for phase in ("prefill", "decode")
            }
        )
        n.bytes_moved = PhaseExpr(
            **{
                phase: add(
                    getattr(n.bytes_moved, phase),
                    *(getattr(ch.bytes_moved, phase) for ch in children),
                )
                for phase in ("prefill", "decode")
            }
        )

    for n in nodes.values():
        if n.parent is None and n.kind == "group":
            aggregate(n)
    layer_weights, layer_kv = [], []
    for start, end, _ in bands:
        terms = [
            w
            for path, w in module_weights.items()
            if (m := LAYER.fullmatch(path)) and int(m[1]) == start
        ]
        kv = [
            w for path, w in kv_terms.items() if (m := LAYER.fullmatch(path)) and int(m[1]) == start
        ]
        layer_weights.append(f"({add(*terms)})*({end - start + 1}/pp)")
        layer_kv.append(f"({add(*kv)})*({end - start + 1}/pp)")
    embed = module_weights.get("model.embed_tokens", "0")
    head = module_weights.get("lm_head", "0")
    final = module_weights.get("model.norm", "0")
    # Tied storage is shared within a stage, but each endpoint PP stage needs a copy.
    pp_head = embed if c.get("tie_word_embeddings", False) else head
    endpoints = f"pp==1 ? ({add(embed, head, final)}) : max(({embed}),({add(pp_head, final)}))"
    weights = add(*layer_weights, f"({endpoints})")
    kv_per_token = add(*layer_kv)
    if not kv_terms:
        problem("No attention KV rule matched; cache capacity cannot be trusted")
        kv_per_token = "0"
    activation = (
        "max(max_num_batched_tokens*(4*H+(Hq/tp+2*max(1,Hkv/tp))*D+"
        + ("dp*topk/(ep>1 ? ep : 1)*(2*H+3*I_moe/(ep>1 ? 1 : tp*dp))+dp*(H+E)" if moe else "3*I/tp")
        + ")*bytes(dtype_a),max_num_seqs*V*(bytes(dtype_a)+3*bytes(fp32)+bytes(int64)))"
    )
    kv_tokens = f"({kv_per_token})>0 ? floor(max(0,hbm_bytes*gpu_memory_utilization-({weights})-({activation})-non_torch_bytes-reserved_bytes)/(({kv_per_token})*block_size))*block_size : 0"
    # A decoder group represents its entire resident band; children remain one-layer templates.
    for g in groups:
        if g.template in nodes:
            node = nodes[g.template]
            count = f"({int(g.layers[1]) - int(g.layers[0]) + 1}/pp)"
            node.weight_bytes = f"({node.weight_bytes})*{count}"
            node.flops = PhaseExpr(
                **{k: f"({getattr(node.flops, k)})*{count}" for k in ("prefill", "decode")}
            )
            node.bytes_moved = PhaseExpr(
                **{k: f"({getattr(node.bytes_moved, k)})*{count}" for k in ("prefill", "decode")}
            )
    ir = IR(
        snapshot=extract.snapshot,
        symbols=dict(sorted(symbols.items())),
        params=runtime_params(extract),
        nodes=sorted(nodes.values(), key=lambda n: n.id),
        edges=sorted(edges.values(), key=lambda e: e.id),
        groups=sorted((g for g in groups if g.template in nodes), key=lambda g: g.id),
        memory=Memory(
            weights=weights,
            kv_per_token=kv_per_token,
            kv_tokens=kv_tokens,
            activation_peak=activation,
            non_torch="non_torch_bytes",
            reserved="reserved_bytes",
            unmodelled=[
                "allocator fragmentation",
                "graph capture workspace",
                "HCCL/NCCL workspace beyond configured non_torch_bytes",
                "RoPE tables and non-parameter buffers",
                "quantization scales/zeros and NZ padding",
                "backend-specific MoE transport buffers beyond gathered activations",
                "backend-specific attention scratch space",
                "input/output token index storage",
            ],
        ),
        views=ViewFamily(family="moe" if moe else "dense"),
        notes=[
            "Rank-local metrics; group summaries include children, never sum both.",
            "Activation peak is a profiling-token liveness estimate including fp32 sampler warmup; uncalibrated.",
            "Equal PP partitions only; mixed bands restrict PP to one. Runtime EP is a degree, not a boolean.",
        ],
    )
    if issue := hardware_profile_triage(extract):
        items.append(issue)
    unique = {json.dumps(item.model_dump(), sort_keys=True): item for item in items}
    triage = Triage(
        snapshot=extract.snapshot,
        items=sorted(
            unique.values(),
            key=lambda item: (
                item.kind,
                getattr(item, "symbol", getattr(item, "model", "")),
                item.model_dump_json(),
            ),
        ),
    )
    return LowerResult(ir=ir, triage=triage, parameter_shapes=dict(sorted(shapes.items())))


lower_extract = lower
