/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Whether architecture data is available.
 */
export type Arch = boolean;
/**
 * Architecture informational notes.
 */
export type NotesInfo = number;
/**
 * Architecture warning notes.
 */
export type NotesWarn = number;
/**
 * Representative generic sublayers.
 */
export type SublayersGeneric = number;
/**
 * Representative semantic sublayers.
 */
export type SublayersRuled = number;
/**
 * Relative directory containing arch.json/ir.json/view.json/report.json; null only when no structure was extracted.
 */
export type DataPath = string | null;
/**
 * Extraction device used for this matrix cell.
 */
export type Device = "A2" | "A3" | "A5" | "310P" | "H100";
/**
 * IndexModel.id.
 */
export type Model = string;
/**
 * ok has structure without notes; triage has review notes or extraction failure; missing has no artifacts.
 */
export type Status = "ok" | "triage" | "missing";
/**
 * Unresolved item count; only triage cells may have a nonzero count.
 */
export type TriageCount = number;
/**
 * IndexVersion.id.
 */
export type Version = string;
/**
 * One cell per version/model/device combination, sorted by version/model/device.
 */
export type Cells = IndexCell[];
/**
 * Architecture family.
 */
export type Family = "dense" | "moe" | "mla" | "hybrid" | "vlm";
/**
 * HF repository id.
 */
export type HfId = string;
/**
 * Filesystem-safe CatalogModel.id.
 */
export type Id = string;
/**
 * Short model display label.
 */
export type Label = string;
/**
 * Version matrix columns, sorted by id.
 */
export type Models = IndexModel[];
/**
 * vllm-ascend tag or commit label.
 */
export type Ascend = string;
/**
 * Catalog version-pair key.
 */
export type Id1 = string;
/**
 * Display label for the paired upstream versions.
 */
export type Label1 = string;
/**
 * vLLM tag or commit label.
 */
export type Vllm = string;
/**
 * Version matrix rows, sorted by id.
 */
export type Versions = IndexVersion[];

export interface Index {
  cells: Cells;
  models: Models;
  versions: Versions;
}
export interface IndexCell {
  /**
   * Structure coverage from arch.json, independent of numeric triage.
   */
  coverage?: IndexCoverage | null;
  data_path?: DataPath;
  device: Device;
  model: Model;
  status: Status;
  triage_count?: TriageCount;
  version: Version;
}
export interface IndexCoverage {
  arch: Arch;
  notes_info: NotesInfo;
  notes_warn: NotesWarn;
  sublayers_generic: SublayersGeneric;
  sublayers_ruled: SublayersRuled;
}
export interface IndexModel {
  family: Family;
  hf_id: HfId;
  id: Id;
  label: Label;
}
export interface IndexVersion {
  ascend: Ascend;
  id: Id1;
  label: Label1;
  vllm: Vllm;
}
