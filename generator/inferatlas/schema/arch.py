"""Structure contract from docs/arch-contract.md, independent of numeric ledgers."""

from typing import Annotated, Literal

from pydantic import ConfigDict, Field, JsonValue, model_validator

from .models import Contract as BaseContract
from .models import Device, Expr


class Contract(BaseContract):
    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class ArchSymbol(Contract):
    value: int | float = Field(description="Default numeric value; expressions reference this key.")
    description: str = Field(description="Meaning of the symbolic dimension.")


class ArchModel(Contract):
    id: str = Field(description="Catalog slug.")
    hf_id: str = Field(description="Hugging Face repository identifier.")
    architecture: str = Field(description="Extracted architecture class.")
    family: Literal["dense", "moe", "mla", "hybrid", "vlm"] = Field(description="Model family.")


class ArchTensor(Contract):
    name: str = Field(description="Tensor or local parameter name.")
    shape: list[Expr] = Field(description="Dimensions in the existing expression DSL.")
    dtype: str = Field(description="Observed scalar dtype, e.g. bfloat16 or int64.")


class ArchIO(Contract):
    inputs: list[ArchTensor] = Field(description="Dataflow inputs; empty when unknown.")
    outputs: list[ArchTensor] = Field(description="Dataflow outputs; empty when unknown.")
    internals: list[ArchTensor] = Field(
        default_factory=list, description="Tensors local to the component."
    )


class ArchSource(Contract):
    file: str = Field(description="Repository-relative source file.")
    line: int = Field(ge=1, description="Extracted declaration line.")
    url: str = Field(description="Immutable source URL including commit and declaration line.")


class ArchReplacement(Contract):
    baseline_class: str = Field(description="Baseline qualified class.")
    ascend_class: str = Field(description="Observed replacement qualified class.")


class ArchModule(Contract):
    flags: list[str] = Field(default_factory=list, description="Observed structural changes.")
    path: str = Field(description="Extracted named_modules path.")
    class_name: str = Field(description="Qualified baseline class name.")
    source: ArchSource | None = Field(
        description="Declaration source; null for external containers."
    )
    status: Literal["ruled", "generic", "ascend_override", "unannotated_override"] = Field(
        description="Semantic rule and reviewed replacement status."
    )
    replaced: ArchReplacement | None = Field(description="Observed replacement, if any.")
    params: list[ArchTensor] = Field(
        description="Component subtree parameters in stable relative-name order."
    )


class ArchChild(Contract):
    name: str = Field(description="Relative direct child name.")
    class_name: str = Field(description="Qualified child class.")


class ArchGeneric(Contract):
    kind: Literal["generic"] = Field(default="generic", description="Unrecognized component.")
    class_name: str = Field(description="Qualified class, or ResidualAdd for a residual junction.")
    children: list[ArchChild] = Field(description="Direct children in stable name order.")


class ArchNorm(Contract):
    kind: Literal["norm"] = Field(default="norm", description="Normalization component.")
    variant: Literal["rmsnorm", "layernorm", "other"] = Field(
        description="Normalization algorithm."
    )
    eps: float | None = Field(default=None, ge=0, description="Observed/configured epsilon.")


class ArchRope(Contract):
    partial_rotary_factor: float | None = Field(default=None, description="Rotated head fraction.")
    kind: str = Field(description="RoPE type, e.g. default, yarn or linear.")
    theta: float | None = Field(default=None, description="RoPE base frequency.")
    scaling: dict[str, JsonValue] | None = Field(
        default=None, description="Config scaling metadata."
    )


class ArchAttention(Contract):
    kind: Literal["attention"] = Field(default="attention", description="Token mixer.")
    variant: Literal["mha", "gqa", "mqa", "mla", "linear", "sliding_window", "sparse", "other"] = (
        Field(description="Attention algorithm.")
    )
    heads_q: int = Field(ge=0, description="Global query heads; zero when unavailable.")
    heads_kv: int = Field(ge=0, description="Global KV heads; zero when unavailable.")
    head_dim: int = Field(ge=0, description="Head width; zero when unavailable.")
    rope: ArchRope | None = Field(description="RoPE metadata or null.")
    qk_norm: bool = Field(description="Whether Q/K normalization modules are present.")
    extras: dict[str, JsonValue] = Field(description="Variant metadata, including MLA ranks.")


class ArchMLP(Contract):
    kind: Literal["mlp"] = Field(default="mlp", description="Dense feed-forward component.")
    variant: Literal["swiglu", "geglu", "gelu", "other"] = Field(description="Activation variant.")
    intermediate: int = Field(ge=0, description="Global intermediate width; zero when unavailable.")


class ArchRouterGroups(Contract):
    n_group: int = Field(ge=1, description="Routing group count.")
    topk_group: int = Field(ge=1, description="Selected routing groups.")


class ArchRouter(Contract):
    scoring: str = Field(description="Router scoring function, or unknown.")
    norm_topk: bool | None = Field(default=None, description="Whether selected weights normalize.")
    groups: ArchRouterGroups | None = Field(
        default=None, description="Expert routing groups, if specified."
    )


class ArchExpert(Contract):
    variant: Literal["swiglu", "geglu", "gelu", "other"] = Field(description="Expert activation.")
    intermediate: int = Field(ge=0, description="Global expert intermediate width.")


class ArchMoE(Contract):
    kind: Literal["moe"] = Field(default="moe", description="Routed feed-forward component.")
    experts: int = Field(ge=0, description="Number of routed experts.")
    top_k: int = Field(ge=0, description="Routed experts selected per token.")
    shared_experts: int = Field(ge=0, description="Number of shared experts.")
    router: ArchRouter = Field(description="Router semantics.")
    expert: ArchExpert = Field(description="One routed expert.")
    shared: ArchExpert | None = Field(description="Combined shared branch, if present.")


class ArchEmbedding(Contract):
    kind: Literal["embedding", "lm_head"] = Field(description="Vocabulary projection direction.")
    vocab: int = Field(ge=0, description="Unpadded vocabulary size.")
    hidden: int = Field(ge=0, description="Hidden width.")
    tied: bool = Field(description="Whether embedding and head share weights.")


ArchComponent = Annotated[
    ArchNorm | ArchAttention | ArchMLP | ArchMoE | ArchGeneric | ArchEmbedding,
    Field(discriminator="kind"),
]


class ArchSublayer(Contract):
    id: str = Field(
        description="Stable target id; representative module path for observed modules."
    )
    role: Literal["norm", "mixer", "ffn", "residual_add", "other"] = Field(
        description="Dataflow role."
    )
    component: ArchComponent | None = Field(description="Semantic component or generic fallback.")
    io: ArchIO = Field(description="Symbolic dataflow boundary.")
    module: ArchModule | None = Field(description="Observed module; null for residual junctions.")

    @model_validator(mode="after")
    def residual_and_id(self) -> "ArchSublayer":
        if "/" in self.id or "." in self.id:
            raise ValueError("sublayer id must be band-relative")
        if self.role == "residual_add":
            if self.component is not None or self.module is not None:
                raise ValueError("residual junction must have null component and module")
        elif self.component is None:
            raise ValueError("observed sublayer requires a component")
        return self


class ArchBlock(Contract):
    kind: Literal["decoder"] = Field(default="decoder", description="Decoder block.")
    residual: Literal["pre_norm", "post_norm", "sandwich", "hyper_connection", "unknown"] = Field(
        description="Residual topology; unknown when class has no reviewed block rule."
    )
    sublayers: list[ArchSublayer] = Field(description="Sublayers in dataflow order.")


class ArchBand(Contract):
    id: str = Field(description="Stable band id based on inclusive layer interval.")
    layers: tuple[Annotated[int, Field(ge=0)], Annotated[int, Field(ge=0)]] = Field(
        description="Inclusive start and end decoder indices."
    )
    count: int = Field(ge=1, description="Number of consecutive structurally identical layers.")
    pattern: str | None = Field(description="Optional repeating-pattern label.")
    block: ArchBlock = Field(description="First layer's representative block.")

    @model_validator(mode="after")
    def interval(self) -> "ArchBand":
        if self.layers[1] - self.layers[0] + 1 != self.count:
            raise ValueError("band count must match inclusive layer interval")
        return self


class ArchStackStage(Contract):
    id: str = Field(description="Stable stack target id.")
    kind: Literal["stack"] = Field(default="stack", description="Decoder stack stage.")
    label: str = Field(description="Human-readable stack label.")
    bands: list[ArchBand] = Field(description="Consecutive layer bands in dataflow order.")


class ArchModuleStage(Contract):
    placement: Literal["spine", "aside"] = Field(default="spine", description="Dataflow placement.")
    id: str = Field(description="Stable stage target id.")
    kind: Literal["embedding", "norm", "lm_head", "other"] = Field(description="Stage role.")
    label: str = Field(description="Human-readable stage label.")
    module: ArchModule = Field(description="Extracted module provenance and parameters.")
    io: ArchIO = Field(description="Symbolic dataflow boundary.")
    component: ArchComponent | None = Field(
        default=None, description="Semantic or generic component."
    )


ArchStage = Annotated[ArchStackStage | ArchModuleStage, Field(discriminator="kind")]


class ArchNote(Contract):
    severity: Literal["info", "warn"] = Field(description="Informational or review-needed badge.")
    target: str | None = Field(
        description="Stage, band or sublayer id; null means whole model. For unknown_layer_class, unannotated_override, conditional_override and generic-component notes, target names the first instance and the note applies to every instance of that class. Other notes apply only to their target."
    )
    text: str = Field(description="Stable explanation without timestamps or host paths.")
    triage_kind: str | None = Field(
        description="Original triage kind, or null for structure notes."
    )


class Arch(Contract):
    # Keep old snapshots valid while requiring other serialized architecture defaults.
    model_config = ConfigDict(
        json_schema_extra=lambda schema: schema.update(
            required=[name for name in schema.get("required", []) if name != "inputs"]
        )
    )

    flags: list[str] = Field(
        default_factory=list, description="Source representation and extraction limitations."
    )
    schema_version: Literal[1] = Field(default=1, description="Structure contract version.")
    model: ArchModel = Field(description="Catalog identity and extracted architecture.")
    device: Device = Field(description="Extraction hardware profile.")
    symbols: dict[str, ArchSymbol] = Field(description="Symbol defaults used by tensor dimensions.")
    inputs: list[ArchTensor] = Field(
        default_factory=list,
        description="All model inputs, including pixel_values for multimodal models; input remains the primary text input.",
    )
    input: ArchTensor = Field(description="Whole-model token input.")
    output: ArchTensor = Field(description="Whole-model logits output.")
    stages: list[ArchStage] = Field(description="Top-level stages in dataflow order.")
    notes: list[ArchNote] = Field(description="Stable review notes; never suppress structure.")

    @model_validator(mode="after")
    def references_and_expressions(self) -> "Arch":
        from inferatlas.expr import validate

        ids, tensors = [], [self.input, *self.inputs, self.output]
        for stage in self.stages:
            ids.append(stage.id)
            if isinstance(stage, ArchStackStage):
                last = -1
                for band in stage.bands:
                    if band.layers[0] <= last:
                        raise ValueError("bands must be ordered and nonoverlapping")
                    last = band.layers[1]
                    ids.append(band.id)
                    for sublayer in band.block.sublayers:
                        ids.append(f"{band.id}/{sublayer.id}")
                        tensors.extend(
                            [*sublayer.io.inputs, *sublayer.io.outputs, *sublayer.io.internals]
                        )
                        if sublayer.module:
                            tensors.extend(sublayer.module.params)
            else:
                tensors.extend(
                    [*stage.io.inputs, *stage.io.outputs, *stage.io.internals, *stage.module.params]
                )
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate architecture target id")
        if any(note.target is not None and note.target not in ids for note in self.notes):
            raise ValueError("architecture note target missing")
        used = set()
        for tensor in tensors:
            for dim in tensor.shape:
                used.update(validate(dim))
                if set(validate(dim)) - self.symbols.keys():
                    raise ValueError("unbound architecture shape symbol")
        if self.symbols.keys() - used:
            raise ValueError("unused architecture shape symbol")
        return self
