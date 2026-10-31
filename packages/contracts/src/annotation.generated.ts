/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * sha256 of source anchor computed by scripts; changed hash invalidates annotation.
 */
export type AnchorHash = string;
/**
 * sha256 of this dependency's source declaration computed by scripts; a change invalidates the whole annotation.
 */
export type AnchorHash1 = string;
/**
 * Qualified implementation dependency symbol resolved by scripts, without source paths/line numbers.
 */
export type Target = string;
/**
 * Additional independently hashed implementation dependencies, sorted by target and distinct from each other and the primary target; every hash must match.
 */
export type Anchors = AnnotationAnchor[];
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
export type Target1 = string;

export interface Annotation {
  anchor_hash: AnchorHash;
  anchors?: Anchors;
  applies_when: AppliesWhen;
  id: Id;
  level: Level;
  note: Note;
  resolves_to: ResolvesTo;
  reviewed_by: ReviewedBy;
  tag?: Tag;
  target: Target1;
}
export interface AnnotationAnchor {
  anchor_hash: AnchorHash1;
  target: Target;
}
