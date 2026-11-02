/**
 * Structure coverage of a cell (contract v1.4 §1). The home page leads with structure — how much
 * of the model the architecture rules describe — and keeps numeric / annotation verification
 * (`status`, `triage_count`) as a secondary indicator.
 */
import type { Arch, IndexCell } from "@inferatlas/contracts";
import { allBands, uniqueNotes } from "./arch";

export interface CellCoverage {
  sublayers_ruled: number;
  sublayers_generic: number;
  notes_warn: number;
  notes_info: number;
  /** The cell has an `arch.json`. */
  arch: boolean;
}

const isCount = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v) && v >= 0;

/**
 * `IndexCell.coverage` when the index carries a well-formed one (v1.4), else `undefined`. Read
 * defensively: older indexes do not have the field and the generated types may not either.
 */
export function cellCoverage(cell: IndexCell): CellCoverage | undefined {
  const c = (cell as IndexCell & { coverage?: unknown }).coverage;
  if (!c || typeof c !== "object") return undefined;
  const o = c as Record<string, unknown>;
  if (
    !isCount(o.sublayers_ruled) ||
    !isCount(o.sublayers_generic) ||
    !isCount(o.notes_warn) ||
    !isCount(o.notes_info) ||
    typeof o.arch !== "boolean"
  )
    return undefined;
  return {
    sublayers_ruled: o.sublayers_ruled,
    sublayers_generic: o.sublayers_generic,
    notes_warn: o.notes_warn,
    notes_info: o.notes_info,
    arch: o.arch,
  };
}

/**
 * The same numbers derived from an `arch.json` (model page, and a reference for the index): every
 * band sublayer with a module counts once per band; `generic` status is generic, every other
 * status (ruled, Ascend override, unannotated override) has a rule.
 */
export function archCoverage(arch: Arch): CellCoverage {
  let ruled = 0;
  let generic = 0;
  for (const b of allBands(arch))
    for (const s of b.block.sublayers) {
      if (!s.module) continue;
      if (s.module.status === "generic") generic++;
      else ruled++;
    }
  const notes = uniqueNotes(arch.notes);
  return {
    sublayers_ruled: ruled,
    sublayers_generic: generic,
    notes_warn: notes.filter((n) => n.severity === "warn").length,
    notes_info: notes.filter((n) => n.severity !== "warn").length,
    arch: true,
  };
}

export type CoverageTone = "full" | "partial" | "none" | "unknown";

export interface CoverageText {
  /** Short badge text ("structure ✓ · 4/4 ruled"). */
  label: string;
  /** Sentence for titles / screen readers. */
  detail: string;
  tone: CoverageTone;
}

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

/** Badge wording for a coverage record; `undefined` = index without coverage (neutral badge). */
export function coverageText(cov: CellCoverage | undefined): CoverageText {
  if (!cov)
    return {
      label: "structure",
      detail: "Structure extracted (coverage not reported by this index).",
      tone: "unknown",
    };
  if (!cov.arch)
    return { label: "no structure", detail: "No arch.json for this cell.", tone: "none" };
  const total = cov.sublayers_ruled + cov.sublayers_generic;
  const warn = cov.notes_warn ? `; ${plural(cov.notes_warn, "warning")}` : "";
  if (total === 0)
    return {
      label: "structure",
      detail: `Structure extracted, no sublayers${warn}.`,
      tone: "unknown",
    };
  if (cov.sublayers_generic === 0)
    return {
      label: `structure ✓ · ${cov.sublayers_ruled}/${total} ruled`,
      detail: `Every sublayer (${total}) is described by an architecture rule${warn}.`,
      tone: "full",
    };
  return {
    label: `${cov.sublayers_ruled}/${total} ruled · ${cov.sublayers_generic} generic`,
    detail: `${cov.sublayers_ruled} of ${total} sublayers are described by a rule; ${plural(cov.sublayers_generic, "generic component")}${warn}.`,
    tone: "partial",
  };
}

/** Secondary, numeric/annotation verification wording (never hides structure). */
export function verificationText(cell: Pick<IndexCell, "status" | "triage_count">): {
  label: string;
  detail: string;
} {
  if (cell.status === "ok")
    return { label: "verified", detail: "Numeric memory/roofline data passed verification." };
  if (cell.status === "triage") {
    const n = cell.triage_count ?? 0;
    return {
      label: n ? `unverified · ${n}` : "unverified",
      detail: `Numeric/annotation verification: ${n ? plural(n, "open triage item") : "open triage items"}. Structure is unaffected.`,
    };
  }
  return { label: "missing", detail: "No artifacts were generated for this combination." };
}
