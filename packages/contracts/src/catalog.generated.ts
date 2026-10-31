/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Default hardware_presets[].id for this device.
 */
export type HardwarePreset = string;
/**
 * Supported extraction device family.
 */
export type Id = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Extractable Ascend families; H100 can be reference-only.
 */
export type Devices = CatalogDevice[];
/**
 * Memory bandwidth in decimal GB/s; null if uncertain.
 */
export type BandwidthGbps = number | null;
/**
 * Device family or H100 reference.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * Device memory in decimal GB (10^9 bytes), including non-HBM memory on 310P; null if uncertain.
 */
export type HbmGb = number | null;
/**
 * Stable preset key; each preset is per accelerator, not per server.
 */
export type Id1 = string;
/**
 * Aggregate interconnect bandwidth in decimal GB/s; record duplex convention in notes.
 */
export type InterconnectGbps = number | null;
/**
 * Specific board/form factor; do not conflate variants.
 */
export type Label = string;
/**
 * Variant, sparsity/duplex conventions and TODOs for uncertain specifications.
 */
export type Notes = string[];
/**
 * Provenance class; these are specs, not measured throughput.
 */
export type Source = "vendor spec";
/**
 * Vendor specification/documentation URL.
 */
export type SourceUrl = string;
/**
 * Roofline/memory presets including H100 reference, sorted by id.
 */
export type HardwarePresets = HardwarePreset1[];
export type JsonValue = unknown;
/**
 * HF commit SHA or inline:sha256:<canonical-config-hash>; no weights revision.
 */
export type ConfigRev = string;
/**
 * Layout/template family.
 */
export type Family = "dense" | "moe" | "mla" | "hybrid" | "vlm";
/**
 * Hugging Face repository identifier, e.g. Qwen/Qwen3-30B-A3B.
 */
export type HfId = string;
/**
 * Stable filesystem-safe slug, e.g. qwen3-30b-a3b.
 */
export type Id2 = string;
/**
 * Curated model configurations, sorted by id.
 */
export type Models = CatalogModel[];
/**
 * vllm-ascend tag or main. Resolve main to SHA in Snapshot.
 */
export type Ascend = string;
/**
 * Stable matrix key; versions are paired, never a Cartesian product.
 */
export type Id3 = string;
/**
 * Upstream .github/vllm-main-verified.commit or vllm-release-tag.commit.
 */
export type PairingFile = string;
/**
 * vLLM tag or full commit. Resolve tags to SHA in Snapshot.
 */
export type Vllm = string;
/**
 * Explicit compatible upstream pairs, sorted by id.
 */
export type VersionPairs = VersionPair[];

export interface Catalog {
  devices: Devices;
  hardware_presets: HardwarePresets;
  models: Models;
  version_pairs: VersionPairs;
}
export interface CatalogDevice {
  hardware_preset: HardwarePreset;
  id: Id;
}
export interface HardwarePreset1 {
  bandwidth_gbps: BandwidthGbps;
  device: Device;
  hbm_gb: HbmGb;
  id: Id1;
  interconnect_gbps: InterconnectGbps;
  label: Label;
  notes?: Notes;
  peak_tflops: PeakTflops;
  source: Source;
  source_url: SourceUrl;
}
/**
 * Dense peak per dtype in 10^12 operations/s; null means unknown/unsupported, never zero.
 */
export interface PeakTflops {
  [k: string]: number | null;
}
export interface CatalogModel {
  config: Config;
  config_rev: ConfigRev;
  family: Family;
  hf_id: HfId;
  id: Id2;
}
/**
 * Pinned HF config contents; tests never fetch HF or weights.
 */
export interface Config {
  [k: string]: JsonValue;
}
export interface VersionPair {
  ascend: Ascend;
  id: Id3;
  pairing_file: PairingFile;
  vllm: Vllm;
}
