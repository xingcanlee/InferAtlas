/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * No rule matched a reachable class; blocks this model's publication.
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
 * Unresolved items sorted by kind then symbol/id/model. Empty list permits publication subject to verify.
 */
export type Items = (UnknownLayerClass | ConditionalOverride | StaleAnnotation | GoldenDiffItem)[];
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

export interface Triage {
  items: Items;
  snapshot: Snapshot;
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
  symbol: Symbol1;
}
export interface StaleAnnotation {
  anchor_hash_new: AnchorHashNew;
  anchor_hash_old: AnchorHashOld;
  id: Id;
  kind: Kind2;
}
export interface GoldenDiffItem {
  diff: Diff;
  kind: Kind3;
  model: Model;
}
/**
 * Same structured identity as extract/IR; not a concatenated display string.
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
