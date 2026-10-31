import { AlertTriangle, CheckCircle2, CircleDashed, CircleSlash, MinusCircle } from "lucide-react";
import { cn } from "@/lib/utils";

export type Status = "ok" | "triage" | "missing" | "pass" | "fail" | "skip";

const META: Record<Status, { icon: typeof CheckCircle2; cls: string; text: string }> = {
  ok: { icon: CheckCircle2, cls: "text-ok border-ok/30", text: "ok" },
  pass: { icon: CheckCircle2, cls: "text-ok border-ok/30", text: "pass" },
  triage: { icon: AlertTriangle, cls: "text-warn border-warn/35", text: "triage" },
  fail: { icon: CircleSlash, cls: "text-bad border-bad/35", text: "fail" },
  missing: { icon: CircleDashed, cls: "text-muted-foreground border-border", text: "missing" },
  skip: { icon: MinusCircle, cls: "text-muted-foreground border-border", text: "skip" },
};

/** Status is never color-only: icon + label always accompany the hue. */
export function StatusBadge({
  status,
  count,
  className,
}: {
  status: Status;
  count?: number | undefined;
  className?: string;
}) {
  const m = META[status];
  const Icon = m.icon;
  return (
    <span
      className={cn(
        "inline-flex h-5 items-center gap-1 rounded-full border px-1.5 text-[11px] font-medium",
        m.cls,
        className,
      )}
      data-status={status}
    >
      <Icon className="size-3" />
      {m.text}
      {count ? <span className="font-mono tabular">· {count}</span> : null}
    </span>
  );
}
