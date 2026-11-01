/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Profile selected before upstream mocks are imported.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
export type JsonValue = unknown;
/**
 * True for curated partial examples; false for production extraction.
 */
export type Fixture = boolean;
/**
 * Machine-readable extraction limitations and source representation.
 */
export type Flags = string[];
/**
 * One layer type per decoder layer in original execution order.
 */
export type LayerTypes = string[];
/**
 * Actual Ascend class after CustomOp registration.
 */
export type Class = string;
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
 * Baseline fully qualified class; keep even when Ascend replacement is present.
 */
export type Class1 = string;
/**
 * Observed Ascend additions or parameter changes.
 */
export type Flags1 = string[];
/**
 * Observed torch dtype normalized without torch. prefix, e.g. bfloat16.
 */
export type Dtype = string;
/**
 * Local parameter name, e.g. weight or w13_weight, not a full module path.
 */
export type Name = string;
/**
 * Observed local-rank meta parameter dimensions, ordinary nonnegative integers.
 */
export type Shape = number[];
/**
 * Direct parameters only, sorted by local name; do not recursively double count.
 */
export type Params = ExtractParam[];
/**
 * named_modules path, empty string for root; modules sorted lexically by path.
 */
export type Path = string;
/**
 * Flat module tree in lexical path order; fixtures may include only layer zero.
 */
export type Modules = ExtractModule[];
/**
 * Extraction limitations and provenance.
 */
export type Notes = string[];
/**
 * Meta construction DP world size.
 */
export type Dp = number;
/**
 * vLLM enable_expert_parallel flag. IR ep is an integer degree; see DSL spec.
 */
export type Ep = boolean;
/**
 * Meta construction PP world size.
 */
export type Pp = number;
/**
 * Global construction rank; fixtures use rank zero.
 */
export type Rank = number;
/**
 * Meta construction TP world size, not a runtime expression.
 */
export type Tp = number;
/**
 * Full immutable Git commit SHA, never an abbreviated hash.
 */
export type Commit1 = string;
/**
 * Release tag or main; display only, identity is commit.
 */
export type Tag = string;
/**
 * Selected hardware profile; H100 is a reference profile.
 */
export type Device1 = "A2" | "A3" | "A5" | "310P" | "H100";
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

export interface Extract {
  /**
   * CPU/mock-derived overlay facts, not measured NPU behavior.
   */
  ascend?: AscendExtract | null;
  config: Config;
  env: Env;
  fixture?: Fixture;
  flags?: Flags;
  layer_types: LayerTypes;
  modules: Modules;
  notes?: Notes;
  parallel: ExtractParallel;
  snapshot: Snapshot;
}
export interface AscendExtract {
  custom_ops: CustomOps;
  device: Device;
  patched: Patched;
  profile: Profile;
}
/**
 * Baseline short class name to qualified replacement class; all registered ops, even unused.
 */
export interface CustomOps {
  [k: string]: string;
}
/**
 * Qualified patched symbol to repo-relative origin file; runtime identity diff is authoritative.
 */
export interface Patched {
  [k: string]: string;
}
/**
 * Declarative hardware profile and sorted capability names.
 */
export interface Profile {
  [k: string]: JsonValue;
}
/**
 * Full pinned HF config contents, no model weights.
 */
export interface Config {
  [k: string]: JsonValue;
}
/**
 * Version strings only, e.g. python/torch; no host paths or timestamps.
 */
export interface Env {
  [k: string]: string;
}
export interface ExtractModule {
  /**
   * Observed replacement, absent/null if class identity did not change.
   */
  ascend?: ExtractOverride | null;
  attrs?: Attrs1;
  class: Class1;
  flags?: Flags1;
  params?: Params;
  path: Path;
  /**
   * Baseline declaration; null only for external containers such as torch ModuleList.
   */
  src?: SourceRef1 | null;
}
export interface ExtractOverride {
  attrs?: Attrs;
  class: Class;
  src: SourceRef;
}
/**
 * Optional whitelisted replacement attributes from the CPU/meta instance.
 */
export interface Attrs {
  [k: string]: JsonValue;
}
/**
 * Actual replacement declaration from immutable source.
 */
export interface SourceRef {
  commit: Commit;
  file: File;
  line: Line;
  repo: Repo;
  symbol: Symbol;
}
/**
 * Whitelisted observed attributes; partition/head attrs are per-rank facts, constructor input_size/output_size can be global. Lower normalizes GLOBAL rule context, never divides local attrs twice.
 */
export interface Attrs1 {
  [k: string]: JsonValue;
}
export interface ExtractParam {
  dtype: Dtype;
  name: Name;
  shape: Shape;
}
export interface SourceRef1 {
  commit: Commit;
  file: File;
  line: Line;
  repo: Repo;
  symbol: Symbol;
}
/**
 * Observed meta construction settings.
 */
export interface ExtractParallel {
  dp?: Dp;
  ep?: Ep;
  pp?: Pp;
  rank?: Rank;
  tp?: Tp;
}
/**
 * Immutable extraction identity.
 */
export interface Snapshot {
  ascend: Revision;
  device: Device1;
  generator_version: GeneratorVersion;
  model: ModelRef;
  vllm: Revision1;
}
/**
 * Paired vllm-ascend revision, including for the baseline graph.
 */
export interface Revision {
  commit: Commit1;
  tag: Tag;
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
  tag: Tag;
}
