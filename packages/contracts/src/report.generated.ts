/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Stable verification check key, e.g. schema, coverage, source_links, golden.
 */
export type Id = string;
/**
 * Human-readable result without host-specific paths/timestamps.
 */
export type Message = string;
/**
 * Affected IR node id, if scoped.
 */
export type Node = string | null;
/**
 * skip is explicit absence of evidence, not a pass.
 */
export type Status = "pass" | "fail" | "skip";
/**
 * Verification checks sorted by id then node.
 */
export type Checks = Check[];
/**
 * JSON pointer into canonical golden output.
 */
export type Path = string;
/**
 * Structured differences sorted by JSON pointer; empty means no differences.
 */
export type GoldenDiff = GoldenDifference[];
/**
 * Independent numeric ledger verification; never gates structure.
 */
export type NumericVerified = boolean;
/**
 * Full immutable Git commit SHA, never an abbreviated hash.
 */
export type Commit = string;
/**
 * Release tag or main; display only, identity is commit.
 */
export type Tag = string;
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
 * Missing file, invalid declaration line, symbol or commit mismatch.
 */
export type Reason = string;
/**
 * Full source commit; vllm/ascend must agree with Snapshot.
 */
export type Commit1 = string;
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
 * Invalid immutable source references sorted by repo/file/line/symbol.
 */
export type SourceLinkFailures = SourceLinkFailure[];

export interface Report {
  checks: Checks;
  golden_diff: GoldenDiff;
  numeric_verified?: NumericVerified;
  snapshot: Snapshot;
  source_link_failures: SourceLinkFailures;
}
export interface Check {
  id: Id;
  message: Message;
  node?: Node;
  status: Status;
}
export interface GoldenDifference {
  /**
   * Observed value; null when absent.
   */
  actual: {
    [k: string]: unknown;
  };
  /**
   * Committed value; null when absent.
   */
  expected: {
    [k: string]: unknown;
  };
  path: Path;
}
/**
 * Verified immutable graph identity.
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
  commit: Commit;
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
  commit: Commit;
  tag: Tag;
}
export interface SourceLinkFailure {
  reason: Reason;
  src: SourceRef;
}
/**
 * Failing source reference.
 */
export interface SourceRef {
  commit: Commit1;
  file: File;
  line: Line;
  repo: Repo;
  symbol: Symbol;
}
