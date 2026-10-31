/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * sha256 of source anchor computed by scripts; changed hash invalidates annotation.
 */
export type AnchorHash = string;
/**
 * DSL predicate over declared config/runtime/profile symbols.
 */
export type AppliesWhen = string;
/**
 * Stable reviewed annotation id.
 */
export type Id = string;
/**
 * Reviewed significance, independent of badge tag.
 */
export type Level = "KERNEL" | "LAYOUT" | "SEMANTIC";
/**
 * Reviewed explanation; no authored measurements or source line numbers.
 */
export type Note = string;
/**
 * Qualified replacement/patch target symbol resolved by scripts.
 */
export type ResolvesTo = string;
/**
 * Maintainer review identity; fixtures must explicitly say fixture.
 */
export type ReviewedBy = string;
/**
 * Sorted badge tags; scripts may derive OOT/PATCH from facts.
 */
export type Tag = ("OOT" | "PATCH" | "QUANT" | "FUSED")[];
/**
 * Qualified symbol only; agent must not author source paths/line numbers.
 */
export type Target = string;
/**
 * YAML annotation files are lists, not wrapper objects; sort entries by id.
 */
export type Annotations = Annotation[];

export interface Annotation {
  anchor_hash: AnchorHash;
  applies_when: AppliesWhen;
  id: Id;
  level: Level;
  note: Note;
  resolves_to: ResolvesTo;
  reviewed_by: ReviewedBy;
  tag?: Tag;
  target: Target;
}
