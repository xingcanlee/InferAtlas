/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Extraction hardware profile.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Source representation and extraction limitations.
 */
export type Flags = string[];
/**
 * Observed scalar dtype, e.g. bfloat16 or int64.
 */
export type Dtype = string;
/**
 * Tensor or local parameter name.
 */
export type Name = string;
/**
 * Dimensions in the existing expression DSL.
 */
export type Shape = string[];
/**
 * All model inputs, including pixel_values for multimodal models; input remains the primary text input.
 */
export type Inputs = ArchTensor1[];
/**
 * Extracted architecture class.
 */
export type Architecture = string;
/**
 * Model family.
 */
export type Family = "dense" | "moe" | "mla" | "hybrid" | "vlm";
/**
 * Hugging Face repository identifier.
 */
export type HfId = string;
/**
 * Catalog slug.
 */
export type Id = string;
/**
 * Informational or review-needed badge.
 */
export type Severity = "info" | "warn";
/**
 * Stage, band or sublayer id; null means whole model. For unknown_layer_class, unannotated_override, conditional_override and generic-component notes, target names the first instance and the note applies to every instance of that class. Other notes apply only to their target.
 */
export type Target = string | null;
/**
 * Stable explanation without timestamps or host paths.
 */
export type Text = string;
/**
 * Original triage kind, or null for structure notes.
 */
export type TriageKind = string | null;
/**
 * Stable review notes; never suppress structure.
 */
export type Notes = ArchNote[];
/**
 * Structure contract version.
 */
export type SchemaVersion = 1;
/**
 * Decoder block.
 */
export type Kind = "decoder";
/**
 * Residual topology; unknown when class has no reviewed block rule.
 */
export type Residual = "pre_norm" | "post_norm" | "sandwich" | "hyper_connection" | "unknown";
/**
 * Semantic component or generic fallback.
 */
export type Component =
  (ArchNorm | ArchAttention | ArchMLP | ArchMoE | ArchGeneric | ArchEmbedding) | null;
/**
 * Observed/configured epsilon.
 */
export type Eps = number | null;
/**
 * Normalization component.
 */
export type Kind1 = "norm";
/**
 * Normalization algorithm.
 */
export type Variant = "rmsnorm" | "layernorm" | "other";
export type JsonValue = unknown;
/**
 * Head width; zero when unavailable.
 */
export type HeadDim = number;
/**
 * Global KV heads; zero when unavailable.
 */
export type HeadsKv = number;
/**
 * Global query heads; zero when unavailable.
 */
export type HeadsQ = number;
/**
 * Token mixer.
 */
export type Kind2 = "attention";
/**
 * Whether Q/K normalization modules are present.
 */
export type QkNorm = boolean;
/**
 * RoPE type, e.g. default, yarn or linear.
 */
export type Kind3 = string;
/**
 * Rotated head fraction.
 */
export type PartialRotaryFactor = number | null;
/**
 * Config scaling metadata.
 */
export type Scaling = {
  [k: string]: JsonValue;
} | null;
/**
 * RoPE base frequency.
 */
export type Theta = number | null;
/**
 * Attention algorithm.
 */
export type Variant1 =
  "mha" | "gqa" | "mqa" | "mla" | "linear" | "sliding_window" | "sparse" | "other";
/**
 * Global intermediate width; zero when unavailable.
 */
export type Intermediate = number;
/**
 * Dense feed-forward component.
 */
export type Kind4 = "mlp";
/**
 * Activation variant.
 */
export type Variant2 = "swiglu" | "geglu" | "gelu" | "other";
/**
 * Global expert intermediate width.
 */
export type Intermediate1 = number;
/**
 * Expert activation.
 */
export type Variant3 = "swiglu" | "geglu" | "gelu" | "other";
/**
 * Number of routed experts.
 */
export type Experts = number;
/**
 * Routed feed-forward component.
 */
export type Kind5 = "moe";
/**
 * Routing group count.
 */
export type NGroup = number;
/**
 * Selected routing groups.
 */
export type TopkGroup = number;
/**
 * Whether selected weights normalize.
 */
export type NormTopk = boolean | null;
/**
 * Router scoring function, or unknown.
 */
export type Scoring = string;
/**
 * Number of shared experts.
 */
export type SharedExperts = number;
/**
 * Routed experts selected per token.
 */
export type TopK = number;
/**
 * Qualified child class.
 */
export type ClassName = string;
/**
 * Relative direct child name.
 */
export type Name1 = string;
/**
 * Direct children in stable name order.
 */
export type Children = ArchChild[];
/**
 * Qualified class, or ResidualAdd for a residual junction.
 */
export type ClassName1 = string;
/**
 * Unrecognized component.
 */
export type Kind6 = "generic";
/**
 * Hidden width.
 */
export type Hidden = number;
/**
 * Vocabulary projection direction.
 */
export type Kind7 = "embedding" | "lm_head";
/**
 * Whether embedding and head share weights.
 */
export type Tied = boolean;
/**
 * Unpadded vocabulary size.
 */
export type Vocab = number;
/**
 * Stable target id; representative module path for observed modules.
 */
export type Id1 = string;
/**
 * Dataflow inputs; empty when unknown.
 */
export type Inputs1 = ArchTensor1[];
/**
 * Tensors local to the component.
 */
export type Internals = ArchTensor1[];
/**
 * Dataflow outputs; empty when unknown.
 */
export type Outputs = ArchTensor1[];
/**
 * Qualified baseline class name.
 */
export type ClassName2 = string;
/**
 * Observed structural changes.
 */
export type Flags1 = string[];
/**
 * Component subtree parameters in stable relative-name order.
 */
export type Params = ArchTensor1[];
/**
 * Extracted named_modules path.
 */
export type Path = string;
/**
 * Observed replacement qualified class.
 */
export type AscendClass = string;
/**
 * Baseline qualified class.
 */
export type BaselineClass = string;
/**
 * Repository-relative source file.
 */
export type File = string;
/**
 * Extracted declaration line.
 */
export type Line = number;
/**
 * Immutable source URL including commit and declaration line.
 */
export type Url = string;
/**
 * Semantic rule and reviewed replacement status.
 */
export type Status = "ruled" | "generic" | "ascend_override" | "unannotated_override";
/**
 * Dataflow role.
 */
export type Role = "norm" | "mixer" | "ffn" | "residual_add" | "other";
/**
 * Sublayers in dataflow order.
 */
export type Sublayers = ArchSublayer[];
/**
 * Number of consecutive structurally identical layers.
 */
export type Count = number;
/**
 * Stable band id based on inclusive layer interval.
 */
export type Id2 = string;
/**
 * Inclusive start and end decoder indices.
 *
 * @minItems 2
 * @maxItems 2
 */
export type Layers = [number, number];
/**
 * Optional repeating-pattern label.
 */
export type Pattern = string | null;
/**
 * Consecutive layer bands in dataflow order.
 */
export type Bands = ArchBand[];
/**
 * Stable stack target id.
 */
export type Id3 = string;
/**
 * Decoder stack stage.
 */
export type Kind8 = "stack";
/**
 * Human-readable stack label.
 */
export type Label = string;
/**
 * Semantic or generic component.
 */
export type Component1 =
  (ArchNorm | ArchAttention | ArchMLP | ArchMoE | ArchGeneric | ArchEmbedding) | null;
/**
 * Stable stage target id.
 */
export type Id4 = string;
/**
 * Stage role.
 */
export type Kind9 = "embedding" | "norm" | "lm_head" | "other";
/**
 * Human-readable stage label.
 */
export type Label1 = string;
/**
 * Dataflow placement.
 */
export type Placement = "spine" | "aside";
/**
 * Top-level stages in dataflow order.
 */
export type Stages = (ArchStackStage | ArchModuleStage)[];
/**
 * Meaning of the symbolic dimension.
 */
export type Description = string;
/**
 * Default numeric value; expressions reference this key.
 */
export type Value = number;

export interface Arch {
  device: Device;
  flags: Flags;
  input: ArchTensor;
  inputs?: Inputs;
  model: ArchModel;
  notes: Notes;
  output: ArchTensor2;
  schema_version: SchemaVersion;
  stages: Stages;
  symbols: Symbols;
}
/**
 * Whole-model token input.
 */
export interface ArchTensor {
  dtype: Dtype;
  name: Name;
  shape: Shape;
}
export interface ArchTensor1 {
  dtype: Dtype;
  name: Name;
  shape: Shape;
}
/**
 * Catalog identity and extracted architecture.
 */
export interface ArchModel {
  architecture: Architecture;
  family: Family;
  hf_id: HfId;
  id: Id;
}
export interface ArchNote {
  severity: Severity;
  target: Target;
  text: Text;
  triage_kind: TriageKind;
}
/**
 * Whole-model logits output.
 */
export interface ArchTensor2 {
  dtype: Dtype;
  name: Name;
  shape: Shape;
}
export interface ArchStackStage {
  bands: Bands;
  id: Id3;
  kind: Kind8;
  label: Label;
}
export interface ArchBand {
  block: ArchBlock;
  count: Count;
  id: Id2;
  layers: Layers;
  pattern: Pattern;
}
/**
 * First layer's representative block.
 */
export interface ArchBlock {
  kind: Kind;
  residual: Residual;
  sublayers: Sublayers;
}
export interface ArchSublayer {
  component: Component;
  id: Id1;
  io: ArchIO;
  /**
   * Observed module; null for residual junctions.
   */
  module: ArchModule | null;
  role: Role;
}
export interface ArchNorm {
  eps: Eps;
  kind: Kind1;
  variant: Variant;
}
export interface ArchAttention {
  extras: Extras;
  head_dim: HeadDim;
  heads_kv: HeadsKv;
  heads_q: HeadsQ;
  kind: Kind2;
  qk_norm: QkNorm;
  /**
   * RoPE metadata or null.
   */
  rope: ArchRope | null;
  variant: Variant1;
}
/**
 * Variant metadata, including MLA ranks.
 */
export interface Extras {
  [k: string]: JsonValue;
}
export interface ArchRope {
  kind: Kind3;
  partial_rotary_factor: PartialRotaryFactor;
  scaling: Scaling;
  theta: Theta;
}
export interface ArchMLP {
  intermediate: Intermediate;
  kind: Kind4;
  variant: Variant2;
}
export interface ArchMoE {
  expert: ArchExpert;
  experts: Experts;
  kind: Kind5;
  router: ArchRouter;
  /**
   * Combined shared branch, if present.
   */
  shared: ArchExpert1 | null;
  shared_experts: SharedExperts;
  top_k: TopK;
}
/**
 * One routed expert.
 */
export interface ArchExpert {
  intermediate: Intermediate1;
  variant: Variant3;
}
/**
 * Router semantics.
 */
export interface ArchRouter {
  /**
   * Expert routing groups, if specified.
   */
  groups: ArchRouterGroups | null;
  norm_topk: NormTopk;
  scoring: Scoring;
}
export interface ArchRouterGroups {
  n_group: NGroup;
  topk_group: TopkGroup;
}
export interface ArchExpert1 {
  intermediate: Intermediate1;
  variant: Variant3;
}
export interface ArchGeneric {
  children: Children;
  class_name: ClassName1;
  kind: Kind6;
}
export interface ArchChild {
  class_name: ClassName;
  name: Name1;
}
export interface ArchEmbedding {
  hidden: Hidden;
  kind: Kind7;
  tied: Tied;
  vocab: Vocab;
}
/**
 * Symbolic dataflow boundary.
 */
export interface ArchIO {
  inputs: Inputs1;
  internals: Internals;
  outputs: Outputs;
}
export interface ArchModule {
  class_name: ClassName2;
  flags: Flags1;
  params: Params;
  path: Path;
  /**
   * Observed replacement, if any.
   */
  replaced: ArchReplacement | null;
  /**
   * Declaration source; null for external containers.
   */
  source: ArchSource | null;
  status: Status;
}
export interface ArchReplacement {
  ascend_class: AscendClass;
  baseline_class: BaselineClass;
}
export interface ArchSource {
  file: File;
  line: Line;
  url: Url;
}
export interface ArchModuleStage {
  component: Component1;
  id: Id4;
  io: ArchIO1;
  kind: Kind9;
  label: Label1;
  module: ArchModule1;
  placement: Placement;
}
/**
 * Symbolic dataflow boundary.
 */
export interface ArchIO1 {
  inputs: Inputs1;
  internals: Internals;
  outputs: Outputs;
}
/**
 * Extracted module provenance and parameters.
 */
export interface ArchModule1 {
  class_name: ClassName2;
  flags: Flags1;
  params: Params;
  path: Path;
  /**
   * Observed replacement, if any.
   */
  replaced: ArchReplacement | null;
  /**
   * Declaration source; null for external containers.
   */
  source: ArchSource | null;
  status: Status;
}
/**
 * Symbol defaults used by tensor dimensions.
 */
export interface Symbols {
  [k: string]: ArchSymbol;
}
export interface ArchSymbol {
  description: Description;
  value: Value;
}
