"""Wire contracts. Export with aliases; reject undeclared fields at every boundary.

IR quantities are expression strings; source line numbers remain provenance integers.
Expressions are syntax/symbol checked by T2, never evaluated by this package.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, RootModel, model_validator

Expr = Annotated[
    str, Field(min_length=1, description="Expression in docs/expr-dsl.md; never a JSON number.")
]
Device = Literal["A2", "A3", "A5", "310P", "H100"]
DType = Literal["fp32", "bf16", "fp16", "fp8", "int8", "int4", "int32", "int64"]
Commit = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]
AnchorHash = Annotated[str, Field(pattern=r"^sha256:[0-9a-f]{64}$")]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True, allow_inf_nan=False)


class Revision(Contract):
    tag: str = Field(description="Release tag or main; display only, identity is commit.")
    commit: Commit = Field(description="Full immutable Git commit SHA, never an abbreviated hash.")


class ModelRef(Contract):
    hf_id: str = Field(description="Hugging Face repository identifier, e.g. Qwen/Qwen3-30B-A3B.")
    config_rev: str = Field(
        description="HF commit SHA or inline:sha256:<canonical-config-hash>; no weights revision."
    )


class Snapshot(Contract):
    vllm: Revision = Field(description="Paired vLLM revision resolved before extraction.")
    ascend: Revision = Field(
        description="Paired vllm-ascend revision, including for the baseline graph."
    )
    model: ModelRef = Field(description="Model identity plus immutable configuration identity.")
    device: Device = Field(description="Selected hardware profile; H100 is a reference profile.")
    generator_version: str = Field(
        description="Generator package version; no generation timestamp."
    )


class CatalogModel(ModelRef):
    id: str = Field(description="Stable filesystem-safe slug, e.g. qwen3-30b-a3b.")
    family: Literal["dense", "moe", "mla", "hybrid", "vlm"] = Field(
        description="Layout/template family."
    )
    config: dict[str, JsonValue] = Field(
        description="Pinned HF config contents; tests never fetch HF or weights."
    )


class VersionPair(Contract):
    id: str = Field(
        description="Stable matrix key; versions are paired, never a Cartesian product."
    )
    vllm: str = Field(description="vLLM tag or full commit. Resolve tags to SHA in Snapshot.")
    ascend: str = Field(description="vllm-ascend tag or main. Resolve main to SHA in Snapshot.")
    pairing_file: str = Field(
        description="Upstream .github/vllm-main-verified.commit or vllm-release-tag.commit."
    )


class HardwarePreset(Contract):
    id: str = Field(
        description="Stable preset key; each preset is per accelerator, not per server."
    )
    device: Device = Field(description="Device family or H100 reference.")
    label: str = Field(description="Specific board/form factor; do not conflate variants.")
    peak_tflops: dict[DType, float | None] = Field(
        description="Dense peak per dtype in 10^12 operations/s; null means unknown/unsupported, never zero."
    )
    hbm_gb: float | None = Field(
        description="Device memory in decimal GB (10^9 bytes), including non-HBM memory on 310P; null if uncertain."
    )
    bandwidth_gbps: float | None = Field(
        description="Memory bandwidth in decimal GB/s; null if uncertain."
    )
    interconnect_gbps: float | None = Field(
        description="Aggregate interconnect bandwidth in decimal GB/s; record duplex convention in notes."
    )
    source: Literal["vendor spec"] = Field(
        description="Provenance class; these are specs, not measured throughput."
    )
    source_url: str = Field(description="Vendor specification/documentation URL.")
    notes: list[str] = Field(
        default_factory=list,
        description="Variant, sparsity/duplex conventions and TODOs for uncertain specifications.",
    )
    non_torch_bytes: int = Field(
        default=0,
        ge=0,
        description="Estimated communication/runtime workspace deduction; not a measurement.",
    )
    reserved_bytes: int = Field(
        default=0,
        ge=0,
        description="Estimated graph capture/allocator deduction; not a measurement.",
    )


class CatalogDevice(Contract):
    id: Device = Field(description="Supported extraction device family.")
    hardware_preset: str = Field(
        title="HardwarePresetId", description="Default hardware_presets[].id for this device."
    )


class Catalog(Contract):
    models: list[CatalogModel] = Field(description="Curated model configurations, sorted by id.")
    version_pairs: list[VersionPair] = Field(
        description="Explicit compatible upstream pairs, sorted by id."
    )
    devices: list[CatalogDevice] = Field(
        description="Extractable Ascend families; H100 can be reference-only."
    )
    hardware_presets: list[HardwarePreset] = Field(
        description="Roofline/memory presets including H100 reference, sorted by id."
    )


class SourceRef(Contract):
    repo: Literal["vllm", "vllm-ascend", "pytorch"] = Field(
        description="Source repository, used to construct immutable links."
    )
    file: str = Field(description="Repository-relative POSIX path, without a URL or line fragment.")
    line: int = Field(
        ge=1,
        description="One-based class/def declaration line found by AST/inspect, never agent-authored.",
    )
    commit: Commit = Field(description="Full source commit; vllm/ascend must agree with Snapshot.")
    symbol: str = Field(
        description="Qualified class/function symbol resolved by the extraction script."
    )


class ExtractParallel(Contract):
    tp: int = Field(
        default=1, ge=1, description="Meta construction TP world size, not a runtime expression."
    )
    pp: int = Field(default=1, ge=1, description="Meta construction PP world size.")
    dp: int = Field(default=1, ge=1, description="Meta construction DP world size.")
    ep: bool = Field(
        default=False,
        description="vLLM enable_expert_parallel flag. IR ep is an integer degree; see DSL spec.",
    )
    rank: int = Field(
        default=0, ge=0, description="Global construction rank; fixtures use rank zero."
    )


class ExtractParam(Contract):
    name: str = Field(
        description="Local parameter name, e.g. weight or w13_weight, not a full module path."
    )
    shape: list[Annotated[int, Field(ge=0)]] = Field(
        description="Observed local-rank meta parameter dimensions, ordinary nonnegative integers."
    )
    dtype: str = Field(
        description="Observed torch dtype normalized without torch. prefix, e.g. bfloat16."
    )


class ExtractOverride(Contract):
    class_name: str = Field(
        alias="class", description="Actual Ascend class after CustomOp registration."
    )
    src: SourceRef = Field(description="Actual replacement declaration from immutable source.")
    attrs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Optional whitelisted replacement attributes from the CPU/meta instance.",
    )


class ExtractModule(Contract):
    path: str = Field(
        description="named_modules path, empty string for root; modules sorted lexically by path."
    )
    class_name: str = Field(
        alias="class",
        description="Baseline fully qualified class; keep even when Ascend replacement is present.",
    )
    src: SourceRef | None = Field(
        default=None,
        description="Baseline declaration; null only for external containers such as torch ModuleList.",
    )
    attrs: dict[str, JsonValue] = Field(
        default_factory=dict,
        description="Whitelisted observed attributes; partition/head attrs are per-rank facts, constructor input_size/output_size can be global. Lower normalizes GLOBAL rule context, never divides local attrs twice.",
    )
    params: list[ExtractParam] = Field(
        default_factory=list,
        description="Direct parameters only, sorted by local name; do not recursively double count.",
    )
    ascend: ExtractOverride | None = Field(
        default=None,
        description="Observed replacement, absent/null if class identity did not change.",
    )


class AscendExtract(Contract):
    device: Device = Field(description="Profile selected before upstream mocks are imported.")
    profile: dict[str, JsonValue] = Field(
        description="Declarative hardware profile and sorted capability names."
    )
    custom_ops: dict[str, str] = Field(
        description="Baseline short class name to qualified replacement class; all registered ops, even unused."
    )
    patched: dict[str, str] = Field(
        description="Qualified patched symbol to repo-relative origin file; runtime identity diff is authoritative."
    )


class Extract(Contract):
    snapshot: Snapshot = Field(description="Immutable extraction identity.")
    fixture: bool = Field(
        default=False,
        description="True for curated partial examples; false for production extraction.",
    )
    config: dict[str, JsonValue] = Field(
        description="Full pinned HF config contents, no model weights."
    )
    parallel: ExtractParallel = Field(description="Observed meta construction settings.")
    modules: list[ExtractModule] = Field(
        description="Flat module tree in lexical path order; fixtures may include only layer zero."
    )
    layer_types: list[str] = Field(
        description="One layer type per decoder layer in original execution order."
    )
    ascend: AscendExtract | None = Field(
        default=None, description="CPU/mock-derived overlay facts, not measured NPU behavior."
    )
    env: dict[str, str] = Field(
        description="Version strings only, e.g. python/torch; no host paths or timestamps."
    )
    notes: list[str] = Field(
        default_factory=list, description="Extraction limitations and provenance."
    )


class ParamSpec(Contract):
    kind: Literal["integer", "number", "boolean", "dtype", "enum"] = Field(
        description="UI/control and environment value type."
    )
    default: str = Field(
        description="Numeric/boolean expression literal, or raw dtype/enum string. Bind into env before evaluating expressions."
    )
    min: Expr | None = Field(default=None, description="Inclusive numeric lower bound expression.")
    max: Expr | None = Field(default=None, description="Inclusive numeric upper bound expression.")
    step: Expr | None = Field(default=None, description="Numeric control increment expression.")
    choices: list[str] | None = Field(
        default=None, description="Allowed literals for dtype/enum or discrete numeric parameters."
    )
    constraint: Expr | None = Field(
        default=None,
        description="DSL boolean predicate over all bound IR symbols/params; all non-null constraints must hold after any control change.",
    )
    description: str = Field(
        description="Meaning and units; integer ep is a degree, not enable_expert_parallel."
    )


class PhaseShape(Contract):
    prefill: list[Expr] = Field(description="Local-rank prefill dimensions.")
    decode: list[Expr] = Field(
        description="Local-rank decode dimensions; no T substitution needed."
    )


class TensorSpec(Contract):
    name: str = Field(
        description="Local input/output slot name; multiple inputs/outputs stay separate."
    )
    shape: list[Expr] | PhaseShape = Field(
        description="Local-rank dimensions: array uses T=S in decode; phase object selects explicit dimensions."
    )
    dtype: Expr = Field(
        default="dtype_a", description="Dtype symbol or builtin dtype name, consumed by bytes()."
    )


class PhaseExpr(Contract):
    prefill: Expr = Field(description="Work per prefill chunk of T local-attention tokens.")
    decode: Expr = Field(description="Work for one decode step of S local-attention sequences.")


class Comm(Contract):
    op: Literal[
        "all_reduce", "gather", "all_gather", "reduce_scatter", "all_to_all", "send_recv", "none"
    ] = Field(
        description="Logical collective; fused kernels retain semantic communication metadata."
    )
    group: str = Field(description="Parallel group name (tp/ep/pp/dp) or a rule-declared group.")
    logical_bytes: PhaseExpr = Field(
        description="Local-rank logical tensor bytes explicitly for each phase."
    )
    bus_bytes: PhaseExpr = Field(
        description="Local-rank estimated link traffic, includes collective algorithm factor; zero when inactive."
    )
    when: Expr = Field(
        default="true", description="Activation condition; inactive communication contributes zero."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Ring/dispatch/routing assumptions, especially EP uniform routing.",
    )


class ParallelAxis(Contract):
    dim: str = Field(
        description="Shard dimension/strategy: in, out, intermediate, experts, layers, replicated, tokens, or a dimension expression."
    )


class AscendTag(Contract):
    tag: list[Literal["OOT", "PATCH", "QUANT", "FUSED"]] = Field(
        description="Sorted overlay badges; independent of significance level."
    )
    level: Literal["KERNEL", "LAYOUT", "SEMANTIC"] = Field(
        description="KERNEL preserves semantics; LAYOUT changes storage; SEMANTIC changes graph/communication."
    )
    note: str = Field(description="Reviewed explanation; KERNEL badges hidden by default.")
    src: list[SourceRef] = Field(
        description="Script-resolved replacement/patch declarations, can include multiple relevant sources."
    )
    annotation_id: str = Field(description="Reviewed annotation identifier, not an inferred note.")
    applies_when: Expr | None = Field(
        default=None,
        description="Selected annotation predicate, re-evaluated after runtime parameter changes.",
    )
    context: dict[str, bool] = Field(
        default_factory=dict,
        description="Fixed extraction facts for applies_when; runtime params remain in the IR environment.",
    )


class IRNode(Contract):
    id: str = Field(
        description="Stable semantic node key; does not encode layout coordinates or mutable labels."
    )
    kind: Literal["compute", "comm", "io", "group"] = Field(
        description="Semantic node category; group metrics summarize descendants."
    )
    class_name: str = Field(
        alias="class", description="Qualified baseline class or semantic operation symbol."
    )
    src: SourceRef = Field(description="Baseline class/function declaration resolved by script.")
    rule: str = Field(description="Matched Rule.id; coverage gate fails if no rule matches.")
    label: str = Field(description="Human-readable node label, independent of identity.")
    parent: str | None = Field(
        default=None,
        description="Containing group node id; null for top-level nodes. Acyclic hierarchy.",
    )
    inputs: list[TensorSpec] = Field(
        alias="in", description="Named input tensor slots, each with symbolic local shape."
    )
    out: list[TensorSpec] = Field(
        description="Named output tensor slots, each with symbolic local shape."
    )
    weight_bytes: Expr = Field(
        description="Resident weight bytes on this rank; group totals include descendants, never sum groups and children together."
    )
    flops: PhaseExpr = Field(
        description="Local-rank floating-point work; one multiply-add is two FLOPs."
    )
    bytes_moved: PhaseExpr = Field(
        description="Local-rank estimated HBM bytes read/written per phase, excluding communication bus bytes."
    )
    comm: Comm | None = Field(
        default=None, description="Collective metadata or null; traffic is separate from HBM bytes."
    )
    parallel: dict[str, ParallelAxis] = Field(
        default_factory=dict,
        description="Extensible ParallelSpec axes, including tp/pp/dp/ep/sp/dcp/pcp where supported.",
    )
    ascend: AscendTag | None = Field(
        default=None, description="Reviewed overlay; absence means no claim of Ascend differences."
    )
    notes: list[str] = Field(
        default_factory=list, description="Formula assumptions/limitations, shown in details."
    )


class IREdge(Contract):
    id: str = Field(description="Stable unique semantic edge key.")
    from_node: str = Field(alias="from", description="Producer IRNode.id.")
    to: str = Field(description="Consumer IRNode.id.")
    tensor: TensorSpec = Field(description="Transported tensor and symbolic shape.")
    kind: Literal["data", "kv", "residual"] = Field(description="Semantic edge style.")


class LayerGroup(Contract):
    id: str = Field(description="Stable layer-band key, used by L1:<band>.")
    layers: tuple[Expr, Expr] = Field(
        description="Inclusive start/end layer indices as expression strings."
    )
    template: str = Field(description="Decoder/group IR node id instantiated for this layer band.")
    representative_layer: Expr = Field(
        description="Representative zero-based layer index as an expression string."
    )


class Memory(Contract):
    weights: Expr = Field(
        description="Resident bytes on the heaviest PP rank, accounting for tied weights, padding, and replicated parameters."
    )
    kv_per_token: Expr = Field(
        description="Local-rank KV bytes for one cached token across resident layers, including KV head replication."
    )
    kv_tokens: Expr = Field(
        description="Local-rank cache token capacity, floored to whole block_size blocks after all budget deductions."
    )
    activation_peak: Expr = Field(
        description="Peak live HBM bytes during max_num_batched_tokens profiling; include sampler/logits peak."
    )
    non_torch: Expr = Field(
        description="Explicit configured non-torch bytes; zero is an excluded estimate, not proof of no overhead."
    )
    reserved: Expr = Field(
        description="Explicit additional reserved bytes deducted from KV budget; no double counting."
    )
    unmodelled: list[str] = Field(
        description="Named exclusions/uncertainties (graph capture, allocator fragmentation, HCCL workspace); prose, not numeric expressions."
    )


class ViewFamily(Contract):
    family: Literal["dense", "moe", "mla", "hybrid", "vlm"] = Field(
        description="Curated layout family; no raw module tree rendering."
    )


class IR(Contract):
    snapshot: Snapshot = Field(description="Immutable graph identity.")
    symbols: dict[str, Expr] = Field(
        description="Config constants as expression literals, e.g. H='2048'; bind evaluated constants into env."
    )
    params: dict[str, ParamSpec] = Field(
        description="Runtime controls keyed by DSL identifier, sorted by key; source of defaults and domains."
    )
    nodes: list[IRNode] = Field(
        description="Semantic nodes in stable id order; use parent for hierarchy and avoid double-counting summaries."
    )
    edges: list[IREdge] = Field(
        description="Semantic edges in stable id order; endpoints refer to nodes."
    )
    groups: list[LayerGroup] = Field(
        description="Decoder layer bands in stable id order; representative template is not all layers expanded."
    )
    memory: Memory = Field(
        description="Authoritative rank-local memory ledger; independent of diagram summaries."
    )
    views: ViewFamily = Field(description="Layout family selector.")
    notes: list[str] = Field(
        default_factory=list,
        description="Accounting conventions and known approximation boundaries.",
    )

    @model_validator(mode="after")
    def check_references(self) -> "IR":
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes):
            raise ValueError("duplicate IR node id")
        if len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("duplicate IR edge id")
        if len({g.id for g in self.groups}) != len(self.groups):
            raise ValueError("duplicate layer group id")
        parents = {n.id: n.parent for n in self.nodes}
        for node in self.nodes:
            seen = {node.id}
            parent = node.parent
            while parent is not None:
                if parent not in ids or parent in seen:
                    raise ValueError("IR parent hierarchy must reference nodes and be acyclic")
                seen.add(parent)
                parent = parents[parent]
        if any(e.from_node not in ids or e.to not in ids for e in self.edges):
            raise ValueError("IR edge endpoint missing")
        if any(g.template not in ids for g in self.groups):
            raise ValueError("layer group template missing")
        return self


class ViewNode(Contract):
    id: str = Field(description="View-local unique node id.")
    ir_node: str = Field(
        description="Referenced IRNode.id; multiple view instances may reference one template."
    )
    x: int = Field(ge=0, multiple_of=4, description="Left coordinate in pixels on a 4px grid.")
    y: int = Field(ge=0, multiple_of=4, description="Top coordinate in pixels on a 4px grid.")
    w: int = Field(gt=0, multiple_of=4, description="Width in pixels on a 4px grid.")
    h: int = Field(gt=0, multiple_of=4, description="Height in pixels on a 4px grid.")
    label: str = Field(description="Visible label; may specialize a representative layer.")


GridCoordinate = Annotated[int, Field(ge=0, multiple_of=4)]
Point = tuple[GridCoordinate, GridCoordinate]


class ViewEdge(Contract):
    ir_edges: list[str] = Field(
        min_length=1,
        description="IR edge ids represented by this route, sorted; may span collapsed nodes.",
    )
    kind: Literal["data", "kv", "residual"] = Field(
        description="Semantic kind shared by the represented IR edges."
    )
    label: list[Expr] | PhaseShape | None = Field(
        default=None,
        description="Optional symbolic shape copied from a represented IR edge tensor.shape.",
    )
    id: str = Field(description="View-local unique edge id.")
    from_node: str = Field(alias="from", description="Producer ViewNode.id, not an IR node id.")
    to: str = Field(description="Consumer ViewNode.id.")
    points: list[Point] = Field(
        min_length=2,
        description="Ordered source-to-target orthogonal polyline, including endpoints; all coordinates on 4px grid.",
    )
    label_pos: Point | None = Field(
        default=None,
        description="Optional precomputed edge-label center in pixels, on the 4px grid.",
    )

    @model_validator(mode="after")
    def orthogonal(self) -> "ViewEdge":
        if any(
            a[0] != b[0] and a[1] != b[1]
            for a, b in zip(self.points, self.points[1:], strict=False)
        ):
            raise ValueError("view edges must be orthogonal")
        return self


class GraphView(Contract):
    id: str = Field(
        pattern=r"^(L0|L1:[^\s:]+|L2:[^\s:]+)$",
        description="L0 overview, L1:<band>, or L2:<node>; bare L1 is invalid.",
    )
    nodes: list[ViewNode] = Field(
        max_length=9, description="Curated visible nodes, stable id order; complexity budget <=9."
    )
    edges: list[ViewEdge] = Field(
        max_length=12, description="Orthogonal edges, stable id order; complexity budget <=12."
    )
    width: int = Field(gt=0, multiple_of=4, description="View canvas width in pixels.")
    height: int = Field(gt=0, multiple_of=4, description="View canvas height in pixels.")
    collapsed_of: list[str] = Field(
        default_factory=list, description="IR node ids represented by this collapsed view, sorted."
    )

    @model_validator(mode="after")
    def check_geometry(self) -> "GraphView":
        ids = {n.id for n in self.nodes}
        if len(ids) != len(self.nodes) or len({e.id for e in self.edges}) != len(self.edges):
            raise ValueError("duplicate view node/edge id")
        if any(e.from_node not in ids or e.to not in ids for e in self.edges):
            raise ValueError("view edge endpoint missing")
        if any(n.x + n.w > self.width or n.y + n.h > self.height for n in self.nodes):
            raise ValueError("view node outside canvas")
        if any(x > self.width or y > self.height for e in self.edges for x, y in e.points):
            raise ValueError("view edge outside canvas")
        return self


class View(Contract):
    views: list[GraphView] = Field(
        description="Precomputed drill-down views in stable id order; no browser-time layout."
    )


class UnknownLayerClass(Contract):
    kind: Literal["unknown_layer_class"] = Field(
        description="No rule matched a reachable class; architecture uses a generic component."
    )
    symbol: str = Field(description="Unmatched qualified class symbol.")
    used_by: list[str] = Field(description="Affected architectures/models, sorted.")


class ConditionalOverride(Contract):
    kind: Literal["conditional_override"] = Field(
        description="Runtime branch needs a reviewed applies_when annotation."
    )
    symbol: str = Field(description="Qualified conditional implementation symbol.")
    candidates: list[str] = Field(description="Candidate target symbols, sorted.")
    conditions: list[Expr] = Field(
        description="Candidate predicate expressions; undeclared symbols go to triage."
    )
    reason: Literal["conditional_override", "unannotated_override"] = Field(
        default="conditional_override",
        description="Unannotated observed replacements need review; structure remains available.",
    )


class StaleAnnotation(Contract):
    kind: Literal["stale_annotation"] = Field(
        description="Source anchor changed; old annotation must not be applied."
    )
    id: str = Field(description="Invalidated Annotation.id.")
    target: str | None = Field(
        default=None,
        description="Qualified symbol of the changed primary or additional source anchor, if known.",
    )
    anchor_hash_old: AnchorHash = Field(description="Previously reviewed source hash.")
    anchor_hash_new: AnchorHash = Field(description="New script-computed source hash.")


class GoldenDiffItem(Contract):
    kind: Literal["golden_diff"] = Field(
        description="Deterministic golden comparison needs review."
    )
    model: str = Field(description="Affected HF model id.")
    diff: str = Field(
        description="Stable human-readable diff, no timestamps or absolute local paths."
    )


class VersionMismatch(Contract):
    kind: Literal["version_mismatch"] = Field(
        description="No rules support the resolved vLLM version, or its version cannot be resolved."
    )
    vllm_version: str = Field(
        description="Resolved vLLM base version; raw explicit version or snapshot tag when resolution fails."
    )
    expected_ranges: list[str] = Field(
        description="Supported vLLM PEP 440 rule ranges, sorted and unique; an empty string means any version."
    )
    message: str = Field(
        description="Stable explanation of the unsupported numeric lowering version."
    )


class HardwareProfileIncomplete(Contract):
    kind: Literal["hardware_profile_incomplete"] = Field(
        description="Device capacity is unconfirmed; extracted facts remain valid but numeric publication is blocked."
    )
    device: Device = Field(description="Affected device.")
    preset: str | None = Field(description="Catalog hardware preset id, or null if absent.")
    message: str = Field(description="Stable explanation of the missing hardware prerequisite.")


class ExtractionFailure(Contract):
    kind: Literal["extraction_failure"] = Field(
        description="Resolve or CPU/meta extraction failed; blocks publication."
    )
    stage: Literal["resolve", "extract"] = Field(description="Failed pipeline stage.")
    exception_type: str = Field(
        min_length=1, description="Exception class name, without traceback."
    )
    message: str = Field(
        max_length=1000,
        description="Deterministic exception message, normalized and truncated to 1000 characters.",
    )
    model: str = Field(description="Affected HF model id.")
    device: Device = Field(description="Requested extraction device.")
    version: str = Field(description="Requested catalog version-pair id, even if unresolved.")


TriageItem = Annotated[
    UnknownLayerClass
    | ConditionalOverride
    | StaleAnnotation
    | GoldenDiffItem
    | VersionMismatch
    | ExtractionFailure
    | HardwareProfileIncomplete,
    Field(discriminator="kind"),
]


class Triage(Contract):
    snapshot: Snapshot | None = Field(
        description="Same structured identity as extract/IR; null only when resolution failed before a snapshot existed."
    )
    items: list[TriageItem] = Field(
        description="Unresolved items sorted by kind then symbol/id/model. Items become architecture notes; source integrity and provenance remain hard errors."
    )

    @model_validator(mode="after")
    def unresolved_identity(self) -> "Triage":
        if self.snapshot is None and (
            not self.items
            or any(
                not isinstance(item, ExtractionFailure) or item.stage != "resolve"
                for item in self.items
            )
        ):
            raise ValueError("unresolved triage requires resolve extraction_failure items")
        return self


class AnnotationAnchor(Contract):
    target: str = Field(
        description="Qualified implementation dependency symbol resolved by scripts, without source paths/line numbers."
    )
    anchor_hash: AnchorHash = Field(
        description="sha256 of this dependency's source declaration computed by scripts; a change invalidates the whole annotation."
    )


class Annotation(Contract):
    id: str = Field(description="Stable reviewed annotation id.")
    target: str = Field(
        description="Qualified symbol only; agent must not author source paths/line numbers."
    )
    anchor_hash: AnchorHash = Field(
        description="sha256 of source anchor computed by scripts; changed hash invalidates annotation."
    )
    anchors: list[AnnotationAnchor] = Field(
        default_factory=list,
        description="Additional independently hashed implementation dependencies, sorted by target and distinct from each other and the primary target; every hash must match.",
    )
    applies_when: Expr = Field(
        description="DSL predicate over declared config/runtime/profile symbols."
    )
    resolves_to: str = Field(
        description="Qualified replacement/patch target symbol resolved by scripts."
    )
    level: Literal["KERNEL", "LAYOUT", "SEMANTIC"] = Field(
        description="Reviewed significance, independent of badge tag."
    )
    tag: list[Literal["OOT", "PATCH", "QUANT", "FUSED"]] = Field(
        default_factory=list,
        description="Sorted badge tags; scripts may derive OOT/PATCH from facts.",
    )
    note: str = Field(
        description="Reviewed explanation; no authored measurements or source line numbers."
    )
    reviewed_by: str = Field(
        min_length=1,
        description="Maintainer review identity; fixtures must explicitly say fixture.",
    )


class Annotations(RootModel[list[Annotation]]):
    """YAML annotation files are lists, not wrapper objects; sort entries by id."""


class RuleMatch(Contract):
    class_name: str = Field(
        alias="class",
        description="Exact qualified class or exact short class name; qualified match wins, no glob/regex matching.",
    )
    vllm: str = Field(
        default="",
        description="PEP 440 SpecifierSet; empty means any. For development commits use base package version plus snapshot identity.",
    )
    ascend: str = Field(
        default="", description="PEP 440 SpecifierSet for vllm-ascend; empty means any."
    )
    when: Expr = Field(
        default="true",
        description="Additional DSL applicability predicate over declared symbols/attrs/params.",
    )


class RuleEmission(Contract):
    suffix: str = Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Sub-node id is <module-node-id>.<suffix>; unique within a rule.",
    )
    kind: Literal["comm"] = Field(
        description="Emitted communication sub-node; compute remains on the module node."
    )
    comm: Comm = Field(description="Collective attached only to this emitted node.")


class Rule(Contract):
    id: str = Field(
        description="Stable rule key; multiple matching rules are an error, never pick by file order."
    )
    match: RuleMatch = Field(description="Class, upstream version ranges and optional predicate.")
    context: list[str] = Field(
        default_factory=list,
        description="Module identifiers required by this rule; lower binds GLOBAL unsharded sizes per docs/expr-dsl.md, never raw local attrs.",
    )
    emits: list[RuleEmission] = Field(
        default_factory=list,
        description="Additional semantic nodes, sorted by suffix; no duplicate collective on the compute node.",
    )
    shard: dict[str, str] = Field(
        default_factory=dict,
        description="Parallel axis to shard dimension/strategy, converted into IR parallel axes.",
    )
    out: list[TensorSpec] = Field(
        description="Output tensor slots with expression dimensions, same representation as IR."
    )
    weight: Expr = Field(
        description="Resident local-rank weight bytes expression, including scale/padding overhead when modelled."
    )
    flops: PhaseExpr = Field(description="Phase-specific FLOP expressions, not a scalar shorthand.")
    bytes_moved: PhaseExpr = Field(description="Phase-specific estimated HBM byte expressions.")
    comm: Comm | None = Field(
        default=None, description="Collective metadata or null; same representation as IR."
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Assumptions and explanation of non-obvious formula terms.",
    )

    @model_validator(mode="after")
    def check_emissions(self) -> "Rule":
        if len({e.suffix for e in self.emits}) != len(self.emits):
            raise ValueError("duplicate rule emission suffix")
        if self.comm is not None and self.emits:
            raise ValueError("use either module comm or emitted comm, not both")
        return self


class Rules(RootModel[list[Rule]]):
    """Rule YAML files contain lists sorted by id; no implicit version-range directory semantics."""


class Check(Contract):
    id: str = Field(
        description="Stable verification check key, e.g. schema, coverage, source_links, golden."
    )
    status: Literal["pass", "fail", "skip"] = Field(
        description="skip is explicit absence of evidence, not a pass."
    )
    message: str = Field(
        description="Human-readable result without host-specific paths/timestamps."
    )
    node: str | None = Field(default=None, description="Affected IR node id, if scoped.")


class GoldenDifference(Contract):
    path: str = Field(description="JSON pointer into canonical golden output.")
    expected: JsonValue = Field(description="Committed value; null when absent.")
    actual: JsonValue = Field(description="Observed value; null when absent.")


class SourceLinkFailure(Contract):
    src: SourceRef = Field(description="Failing source reference.")
    reason: str = Field(
        description="Missing file, invalid declaration line, symbol or commit mismatch."
    )


class Report(Contract):
    numeric_verified: bool = Field(
        default=False, description="Independent numeric ledger verification; never gates structure."
    )
    snapshot: Snapshot = Field(description="Verified immutable graph identity.")
    checks: list[Check] = Field(description="Verification checks sorted by id then node.")
    golden_diff: list[GoldenDifference] = Field(
        description="Structured differences sorted by JSON pointer; empty means no differences."
    )
    source_link_failures: list[SourceLinkFailure] = Field(
        description="Invalid immutable source references sorted by repo/file/line/symbol."
    )


class EvalCase(Contract):
    id: str = Field(description="Stable named evaluation case.")
    env: dict[str, float | str | bool] = Field(
        description="Overrides of IR params; bind IR symbols and defaults first. Raw values, not expressions."
    )
    expected: dict[
        Annotated[str, Field(pattern=r"^/(?:[^~/]|~[01])*(?:/(?:[^~/]|~[01])*)*$")],
        float | list[float],
    ] = Field(
        description="JSON pointers into evaluated IR with nodes indexed by id (not array position), e.g. /nodes/attn.qkv/out/0/shape. Escape ~ and / as ~0 and ~1.",
        json_schema_extra={"additionalProperties": False},
    )


class Eval(Contract):
    model: str = Field(description="Fixture IR snapshot.model.hf_id.")
    cases: list[EvalCase] = Field(
        description="Golden numeric examples in stable case id order; no evaluator implementation in Python."
    )


class IndexVersion(Contract):
    id: str = Field(description="Catalog version-pair key.")
    label: str = Field(description="Display label for the paired upstream versions.")
    vllm: str = Field(description="vLLM tag or commit label.")
    ascend: str = Field(description="vllm-ascend tag or commit label.")


class IndexModel(Contract):
    id: str = Field(description="Filesystem-safe CatalogModel.id.")
    hf_id: str = Field(description="HF repository id.")
    label: str = Field(description="Short model display label.")
    family: Literal["dense", "moe", "mla", "hybrid", "vlm"] = Field(
        description="Architecture family."
    )


class IndexCell(Contract):
    version: str = Field(description="IndexVersion.id.")
    model: str = Field(description="IndexModel.id.")
    device: Device = Field(description="Extraction device used for this matrix cell.")
    status: Literal["ok", "triage", "missing"] = Field(
        description="ok has structure without notes; triage has review notes or extraction failure; missing has no artifacts."
    )
    data_path: str | None = Field(
        default=None,
        description="Relative directory containing arch.json/ir.json/view.json/report.json; null only when no structure was extracted.",
    )
    triage_count: int = Field(
        default=0,
        ge=0,
        description="Unresolved item count; only triage cells may have a nonzero count.",
    )


class Index(Contract):
    versions: list[IndexVersion] = Field(description="Version matrix rows, sorted by id.")
    models: list[IndexModel] = Field(description="Version matrix columns, sorted by id.")
    cells: list[IndexCell] = Field(
        description="One cell per version/model/device combination, sorted by version/model/device."
    )
