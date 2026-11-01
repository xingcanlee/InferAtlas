import type { Index, IndexCell } from "@inferatlas/contracts";

/** Pure matrix/status helpers shared by the site, the single-file export and the hub. */

export function devicesOf(index: Index): string[] {
  return [...new Set(index.cells.map((c) => c.device))].sort();
}

export type CellRef = Pick<IndexCell, "version" | "model" | "device">;

export const cellLabel = (c: CellRef) => `${c.model} @ ${c.version} on ${c.device}`;

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

/**
 * Honest, status-derived wording for a cell (backlog #3): a triage cell is never called
 * "missing", and only a cell whose status is `missing` claims there are no artifacts.
 */
export function cellStatusText(cell: Pick<IndexCell, "status" | "triage_count" | "data_path">): {
  title: string;
  detail: string;
} {
  const n = cell.triage_count ?? 0;
  switch (cell.status) {
    case "ok":
      return cell.data_path
        ? { title: "Verified", detail: "Passed verification and is publishable." }
        : { title: "Verified", detail: "Passed verification, but no data bundle was published." };
    case "triage": {
      const items = n > 0 ? plural(n, "unresolved triage item") : "unresolved triage items";
      return cell.data_path
        ? { title: "Needs triage", detail: `Published with ${items}.` }
        : {
            title: "Needs triage",
            detail: `Blocked by ${items}; no numeric data was published.`,
          };
    }
    default:
      return { title: "Missing", detail: "No artifacts were generated for this combination." };
  }
}

/** Message for a deep link whose cell has no numeric bundle to open. */
export function cellUnavailableMessage(cell: IndexCell): string {
  return `${cellLabel(cell)}: ${cellStatusText(cell).detail}`;
}
