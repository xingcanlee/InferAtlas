/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Producer IRNode.id.
 */
export type From = string;
/**
 * Stable unique semantic edge key.
 */
export type Id = string;
/**
 * Semantic edge style.
 */
export type Kind = "data" | "kv" | "residual";
/**
 * Dtype symbol or builtin dtype name, consumed by bytes().
 */
export type Dtype = string;
/**
 * Local input/output slot name; multiple inputs/outputs stay separate.
 */
export type Name = string;
/**
 * Local-rank dimensions as DSL expressions, in dimension order.
 */
export type Shape = string[];
/**
 * Consumer IRNode.id.
 */
export type To = string;
/**
 * Semantic edges in stable id order; endpoints refer to nodes.
 */
export type Edges = IREdge[];
/**
 * Stable layer-band key, used by L1:<band>.
 */
export type Id1 = string;
/**
 * Inclusive start/end layer indices as expression strings.
 *
 * @minItems 2
 * @maxItems 2
 */
export type Layers = [unknown, unknown];
/**
 * Representative zero-based layer index as an expression string.
 */
export type RepresentativeLayer = string;
/**
 * Decoder/group IR node id instantiated for this layer band.
 */
export type Template = string;
/**
 * Decoder layer bands in stable id order; representative template is not all layers expanded.
 */
export type Groups = LayerGroup[];
/**
 * Peak live HBM bytes during max_num_batched_tokens profiling; include sampler/logits peak.
 */
export type ActivationPeak = string;
/**
 * Local-rank KV bytes for one cached token across resident layers, including KV head replication.
 */
export type KvPerToken = string;
/**
 * Local-rank cache token capacity, floored to whole block_size blocks after all budget deductions.
 */
export type KvTokens = string;
/**
 * Explicit configured non-torch bytes; zero is an excluded estimate, not proof of no overhead.
 */
export type NonTorch = string;
/**
 * Explicit additional reserved bytes deducted from KV budget; no double counting.
 */
export type Reserved = string;
/**
 * Named exclusions/uncertainties (graph capture, allocator fragmentation, HCCL workspace); prose, not numeric expressions.
 */
export type Unmodelled = string[];
/**
 * Resident bytes on the heaviest PP rank, accounting for tied weights, padding, and replicated parameters.
 */
export type Weights = string;
/**
 * Reviewed annotation identifier, not an inferred note.
 */
export type AnnotationId = string;
/**
 * KERNEL preserves semantics; LAYOUT changes storage; SEMANTIC changes graph/communication.
 */
export type Level = "KERNEL" | "LAYOUT" | "SEMANTIC";
/**
 * Reviewed explanation; KERNEL badges hidden by default.
 */
export type Note = string;
/**
 * Full source commit; vllm/ascend must agree with Snapshot.
 */
export type Commit = string;
/**
 * Repository-relative POSIX path, without a URL or line fragment.
 */
export type File = string;
/**
 * One-based class/def declaration line found by AST/inspect, never agent-authored.
 */
export type Line = number;
/**
 * Source repository, used to construct immutable links.
 */
export type Repo = "vllm" | "vllm-ascend" | "pytorch";
/**
 * Qualified class/function symbol resolved by the extraction script.
 */
export type Symbol = string;
/**
 * Script-resolved replacement/patch declarations, can include multiple relevant sources.
 */
export type Src = SourceRef[];
/**
 * Sorted overlay badges; independent of significance level.
 */
export type Tag = ("OOT" | "PATCH" | "QUANT" | "FUSED")[];
/**
 * Work for one decode step of S local-attention sequences.
 */
export type Decode = string;
/**
 * Work per prefill chunk of T local-attention tokens.
 */
export type Prefill = string;
/**
 * Qualified baseline class or semantic operation symbol.
 */
export type Class = string;
/**
 * Local-rank estimated link traffic, includes collective algorithm factor; zero when inactive.
 */
export type BusBytes = string;
/**
 * Parallel group name (tp/ep/pp/dp) or a rule-declared group.
 */
export type Group = string;
/**
 * Local-rank logical tensor bytes per prefill invocation; use T=S for decode.
 */
export type LogicalBytes = string;
/**
 * Ring/dispatch/routing assumptions, especially EP uniform routing.
 */
export type Notes = string[];
/**
 * Logical collective; fused kernels retain semantic communication metadata.
 */
export type Op =
  "all_reduce" | "all_gather" | "reduce_scatter" | "all_to_all" | "send_recv" | "none";
/**
 * Activation condition; inactive communication contributes zero.
 */
export type When = string;
/**
 * Stable semantic node key; does not encode layout coordinates or mutable labels.
 */
export type Id2 = string;
/**
 * Named input tensor slots, each with symbolic local shape.
 */
export type In = TensorSpec1[];
/**
 * Semantic node category; group metrics summarize descendants.
 */
export type Kind1 = "compute" | "comm" | "io" | "group";
/**
 * Human-readable node label, independent of identity.
 */
export type Label = string;
/**
 * Formula assumptions/limitations, shown in details.
 */
export type Notes1 = string[];
/**
 * Named output tensor slots, each with symbolic local shape.
 */
export type Out = TensorSpec1[];
/**
 * Shard dimension/strategy: in, out, intermediate, experts, layers, replicated, tokens, or a dimension expression.
 */
export type Dim = string;
/**
 * Containing group node id; null for top-level nodes. Acyclic hierarchy.
 */
export type Parent = string | null;
/**
 * Matched Rule.id; coverage gate fails if no rule matches.
 */
export type Rule = string;
/**
 * Resident weight bytes on this rank; group totals include descendants, never sum groups and children together.
 */
export type WeightBytes = string;
/**
 * Semantic nodes in stable id order; use parent for hierarchy and avoid double-counting summaries.
 */
export type Nodes = IRNode[];
/**
 * Accounting conventions and known approximation boundaries.
 */
export type Notes2 = string[];
/**
 * Allowed literals for dtype/enum or discrete numeric parameters.
 */
export type Choices = string[] | null;
/**
 * Numeric/boolean expression literal, or raw dtype/enum string. Bind into env before evaluating expressions.
 */
export type Default = string;
/**
 * Meaning and units; integer ep is a degree, not enable_expert_parallel.
 */
export type Description = string;
/**
 * UI/control and environment value type.
 */
export type Kind2 = "integer" | "number" | "boolean" | "dtype" | "enum";
/**
 * Inclusive numeric upper bound expression.
 */
export type Max = string | null;
/**
 * Inclusive numeric lower bound expression.
 */
export type Min = string | null;
/**
 * Numeric control increment expression.
 */
export type Step = string | null;
/**
 * Full immutable Git commit SHA, never an abbreviated hash.
 */
export type Commit1 = string;
/**
 * Release tag or main; display only, identity is commit.
 */
export type Tag1 = string;
/**
 * Selected hardware profile; H100 is a reference profile.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Generator package version; no generation timestamp.
 */
export type GeneratorVersion = string;
/**
 * HF commit SHA or inline:sha256:<canonical-config-hash>; no weights revision.
 */
export type ConfigRev = string;
/**
 * Hugging Face repository identifier, e.g. Qwen/Qwen3-30B-A3B.
 */
export type HfId = string;
/**
 * Curated layout family; no raw module tree rendering.
 */
export type Family = "dense" | "moe" | "mla" | "hybrid" | "vlm";

export interface IR {
  edges: Edges;
  groups: Groups;
  memory: Memory;
  nodes: Nodes;
  notes?: Notes2;
  params: Params;
  snapshot: Snapshot;
  symbols: Symbols;
  views: ViewFamily;
}
export interface IREdge {
  from: From;
  id: Id;
  kind: Kind;
  tensor: TensorSpec;
  to: To;
}
/**
 * Transported tensor and symbolic shape.
 */
export interface TensorSpec {
  dtype?: Dtype;
  name: Name;
  shape: Shape;
}
export interface LayerGroup {
  id: Id1;
  layers: Layers;
  representative_layer: RepresentativeLayer;
  template: Template;
}
/**
 * Authoritative rank-local memory ledger; independent of diagram summaries.
 */
export interface Memory {
  activation_peak: ActivationPeak;
  kv_per_token: KvPerToken;
  kv_tokens: KvTokens;
  non_torch: NonTorch;
  reserved: Reserved;
  unmodelled: Unmodelled;
  weights: Weights;
}
export interface IRNode {
  /**
   * Reviewed overlay; absence means no claim of Ascend differences.
   */
  ascend?: AscendTag | null;
  bytes_moved: PhaseExpr;
  class: Class;
  /**
   * Collective metadata or null; traffic is separate from HBM bytes.
   */
  comm?: Comm | null;
  flops: PhaseExpr1;
  id: Id2;
  in: In;
  kind: Kind1;
  label: Label;
  notes?: Notes1;
  out: Out;
  parallel?: Parallel;
  parent?: Parent;
  rule: Rule;
  src: SourceRef1;
  weight_bytes: WeightBytes;
}
export interface AscendTag {
  annotation_id: AnnotationId;
  level: Level;
  note: Note;
  src: Src;
  tag: Tag;
}
export interface SourceRef {
  commit: Commit;
  file: File;
  line: Line;
  repo: Repo;
  symbol: Symbol;
}
/**
 * Local-rank estimated HBM bytes read/written per phase, excluding communication bus bytes.
 */
export interface PhaseExpr {
  decode: Decode;
  prefill: Prefill;
}
export interface Comm {
  bus_bytes: BusBytes;
  group: Group;
  logical_bytes: LogicalBytes;
  notes?: Notes;
  op: Op;
  when?: When;
}
/**
 * Local-rank floating-point work; one multiply-add is two FLOPs.
 */
export interface PhaseExpr1 {
  decode: Decode;
  prefill: Prefill;
}
export interface TensorSpec1 {
  dtype?: Dtype;
  name: Name;
  shape: Shape;
}
/**
 * Extensible ParallelSpec axes, including tp/pp/dp/ep/sp/dcp/pcp where supported.
 */
export interface Parallel {
  [k: string]: ParallelAxis;
}
export interface ParallelAxis {
  dim: Dim;
}
/**
 * Baseline class/function declaration resolved by script.
 */
export interface SourceRef1 {
  commit: Commit;
  file: File;
  line: Line;
  repo: Repo;
  symbol: Symbol;
}
/**
 * Runtime controls keyed by DSL identifier, sorted by key; source of defaults and domains.
 */
export interface Params {
  [k: string]: ParamSpec;
}
export interface ParamSpec {
  choices?: Choices;
  default: Default;
  description: Description;
  kind: Kind2;
  max?: Max;
  min?: Min;
  step?: Step;
}
/**
 * Immutable graph identity.
 */
export interface Snapshot {
  ascend: Revision;
  device: Device;
  generator_version: GeneratorVersion;
  model: ModelRef;
  vllm: Revision1;
}
/**
 * Paired vllm-ascend revision, including for the baseline graph.
 */
export interface Revision {
  commit: Commit1;
  tag: Tag1;
}
/**
 * Model identity plus immutable configuration identity.
 */
export interface ModelRef {
  config_rev: ConfigRev;
  hf_id: HfId;
}
/**
 * Paired vLLM revision resolved before extraction.
 */
export interface Revision1 {
  commit: Commit1;
  tag: Tag1;
}
/**
 * Config constants as expression literals, e.g. H='2048'; bind evaluated constants into env.
 */
export interface Symbols {
  /**
   * Expression in docs/expr-dsl.md; never a JSON number.
   */
  [k: string]: string;
}
/**
 * Layout family selector.
 */
export interface ViewFamily {
  family: Family;
}
