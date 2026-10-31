/* Generated from schemas; edit generator/inferatlas/schema instead. */

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
