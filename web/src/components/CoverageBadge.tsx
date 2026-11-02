import type { IndexCell } from "@inferatlas/contracts";
import { CheckCircle2, CircleDashed, CircleSlash, Shapes, TriangleAlert } from "lucide-react";
import { type CellCoverage, coverageText, verificationText } from "@/lib/coverage";
import { cn } from "@/lib/utils";

const TONE = {
  full: { icon: CheckCircle2, cls: "text-ok border-ok/30" },
  partial: { icon: Shapes, cls: "text-foreground border-border" },
  unknown: { icon: Shapes, cls: "text-foreground border-border" },
  none: { icon: CircleSlash, cls: "text-muted-foreground border-border" },
} as const;

/** Structure coverage pill (v1.4 §1); neutral "structure" when the index has no coverage. */
export function CoverageBadge({
  coverage,
  className,
}: {
  coverage: CellCoverage | undefined;
  className?: string;
}) {
  const t = coverageText(coverage);
  const m = TONE[t.tone];
  const Icon = m.icon;
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center gap-1 rounded-full border px-1.5 text-[11px] font-medium whitespace-nowrap",
        m.cls,
        className,
      )}
      title={t.detail}
      data-coverage={t.tone}
    >
      <Icon className="size-3 shrink-0" aria-hidden="true" />
      {t.label}
    </span>
  );
}

/** Warning-note count next to the coverage pill (glyph + number, never colour only). */
export function WarnCount({ count }: { count: number }) {
  if (!count) return null;
  return (
    <span
      className="inline-flex items-center gap-0.5 font-mono text-[11px] text-warn tabular"
      title={`${count} warning note${count === 1 ? "" : "s"}`}
      data-testid="warn-count"
    >
      <TriangleAlert className="size-3" aria-hidden="true" />
      <span className="sr-only">warnings: </span>
      {count}
    </span>
  );
}

/** Secondary numeric/annotation verification status: subtle, text + glyph. */
export function VerificationMark({
  cell,
  className,
}: {
  cell: Pick<IndexCell, "status" | "triage_count">;
  className?: string;
}) {
  const v = verificationText(cell);
  const Icon = cell.status === "ok" ? CheckCircle2 : CircleDashed;
  return (
    <span
      className={cn(
        "inline-flex items-center gap-0.5 text-[11px] whitespace-nowrap text-muted-foreground",
        className,
      )}
      title={v.detail}
      data-status={cell.status}
      data-testid="verification"
    >
      <Icon className="size-3" aria-hidden="true" />
      {v.label}
    </span>
  );
}
