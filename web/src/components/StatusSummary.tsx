import type { IndexCell } from "@inferatlas/contracts";

/** "N publishable · N need triage · N missing" row above a matrix. */
export function StatusCounts({ cells }: { cells: Pick<IndexCell, "status">[] }) {
  const counts = { ok: 0, triage: 0, missing: 0 };
  for (const c of cells) counts[c.status]++;
  const item = (n: number, text: string) => (
    <span>
      <span className="font-mono text-foreground tabular">{n}</span> {text}
    </span>
  );
  return (
    <div className="flex gap-4 text-[12px] text-muted-foreground" data-testid="status-counts">
      {item(counts.ok, "publishable")}
      {item(counts.triage, "need triage")}
      {item(counts.missing, "missing")}
    </div>
  );
}

export function StatusLegend({ className }: { className?: string }) {
  return (
    <p className={className ?? "text-[11px] text-muted-foreground"}>
      <span className="font-medium">ok</span> passed verification and is publishable ·{" "}
      <span className="font-medium">triage</span> has unresolved items (count) ·{" "}
      <span className="font-medium">missing</span> has no artifacts for this pair.
    </p>
  );
}
