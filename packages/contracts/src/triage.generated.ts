/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * No rule matched a reachable class; architecture uses a generic component.
 */
export type Kind = "unknown_layer_class";
/**
 * Unmatched qualified class symbol.
 */
export type Symbol = string;
/**
 * Affected architectures/models, sorted.
 */
export type UsedBy = string[];
/**
 * Candidate target symbols, sorted.
 */
export type Candidates = string[];
/**
 * Candidate predicate expressions; undeclared symbols go to triage.
 */
export type Conditions = string[];
/**
 * Runtime branch needs a reviewed applies_when annotation.
 */
export type Kind1 = "conditional_override";
/**
 * Unannotated observed replacements need review; structure remains available.
 */
export type Reason = "conditional_override" | "unannotated_override";
/**
 * Qualified conditional implementation symbol.
 */
export type Symbol1 = string;
/**
 * New script-computed source hash.
 */
export type AnchorHashNew = string;
/**
 * Previously reviewed source hash.
 */
export type AnchorHashOld = string;
/**
 * Invalidated Annotation.id.
 */
export type Id = string;
/**
 * Source anchor changed; old annotation must not be applied.
 */
export type Kind2 = "stale_annotation";
/**
 * Qualified symbol of the changed primary or additional source anchor, if known.
 */
export type Target = string | null;
/**
 * Stable human-readable diff, no timestamps or absolute local paths.
 */
export type Diff = string;
/**
 * Deterministic golden comparison needs review.
 */
export type Kind3 = "golden_diff";
/**
 * Affected HF model id.
 */
export type Model = string;
/**
 * Supported vLLM PEP 440 rule ranges, sorted and unique; an empty string means any version.
 */
export type ExpectedRanges = string[];
/**
 * No rules support the resolved vLLM version, or its version cannot be resolved.
 */
export type Kind4 = "version_mismatch";
/**
 * Stable explanation of the unsupported numeric lowering version.
 */
export type Message = string;
/**
 * Resolved vLLM base version; raw explicit version or snapshot tag when resolution fails.
 */
export type VllmVersion = string;
/**
 * Requested extraction device.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Exception class name, without traceback.
 */
export type ExceptionType = string;
/**
 * Resolve or CPU/meta extraction failed; blocks publication.
 */
export type Kind5 = "extraction_failure";
/**
 * Deterministic exception message, normalized and truncated to 1000 characters.
 */
export type Message1 = string;
/**
 * Affected HF model id.
 */
export type Model1 = string;
/**
 * Failed pipeline stage.
 */
export type Stage = "resolve" | "extract";
/**
 * Requested catalog version-pair id, even if unresolved.
 */
export type Version = string;
/**
 * Affected device.
 */
export type Device1 = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Device capacity is unconfirmed; extracted facts remain valid but numeric publication is blocked.
 */
export type Kind6 = "hardware_profile_incomplete";
/**
 * Stable explanation of the missing hardware prerequisite.
 */
export type Message2 = string;
/**
 * Catalog hardware preset id, or null if absent.
 */
export type Preset = string | null;
/**
 * Unresolved items sorted by kind then symbol/id/model. Items become architecture notes; source integrity and provenance remain hard errors.
 */
export type Items = (
  | UnknownLayerClass
  | ConditionalOverride
  | StaleAnnotation
  | GoldenDiffItem
  | VersionMismatch
  | ExtractionFailure
  | HardwareProfileIncomplete
)[];
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
export type Device2 = "A2" | "A3" | "A5" | "310P" | "H100";
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

export interface Triage {
  items: Items;
  /**
   * Same structured identity as extract/IR; null only when resolution failed before a snapshot existed.
   */
  snapshot: Snapshot | null;
}
export interface UnknownLayerClass {
  kind: Kind;
  symbol: Symbol;
  used_by: UsedBy;
}
export interface ConditionalOverride {
  candidates: Candidates;
  conditions: Conditions;
  kind: Kind1;
  reason?: Reason;
  symbol: Symbol1;
}
export interface StaleAnnotation {
  anchor_hash_new: AnchorHashNew;
  anchor_hash_old: AnchorHashOld;
  id: Id;
  kind: Kind2;
  target?: Target;
}
export interface GoldenDiffItem {
  diff: Diff;
  kind: Kind3;
  model: Model;
}
export interface VersionMismatch {
  expected_ranges: ExpectedRanges;
  kind: Kind4;
  message: Message;
  vllm_version: VllmVersion;
}
export interface ExtractionFailure {
  device: Device;
  exception_type: ExceptionType;
  kind: Kind5;
  message: Message1;
  model: Model1;
  stage: Stage;
  version: Version;
}
export interface HardwareProfileIncomplete {
  device: Device1;
  kind: Kind6;
  message: Message2;
  preset: Preset;
}
export interface Snapshot {
  ascend: Revision;
  device: Device2;
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
