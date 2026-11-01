import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";
import { cn } from "../cn";

export type ModuleStatus = "ruled" | "generic" | "ascend_override" | "unannotated_override";

const STATUS: Record<ModuleStatus, { label: string; short: string; glyph: string; cls: string }> = {
  ruled: { label: "ruled", short: "R", glyph: "✓", cls: "text-ok border-ok/35" },
  generic: {
    label: "generic",
    short: "G",
    glyph: "◌",
    cls: "text-muted-foreground border-dashed border-arch-generic-border",
  },
  ascend_override: {
    label: "Ascend override",
    short: "A",
    glyph: "A",
    cls: "text-overlay border-overlay/40",
  },
  unannotated_override: {
    label: "unannotated override",
    short: "!",
    glyph: "!",
    cls: "text-warn border-warn/45",
  },
};

export const statusLabel = (s: ModuleStatus): string => STATUS[s].label;

const badge = cva(
  "inline-flex shrink-0 items-center gap-1 rounded-full border bg-background font-medium whitespace-nowrap",
  {
    variants: {
      size: {
        /** Icon-only marker for figure nodes (label in title/aria-label). */
        dot: "size-4 justify-center p-0 text-[10px] leading-none",
        sm: "h-[18px] px-1.5 text-[10px]",
        md: "h-5 px-2 text-[11px]",
      },
    },
    defaultVariants: { size: "sm" },
  },
);

export interface StatusBadgeProps
  extends Omit<React.ComponentProps<"span">, "children">,
    VariantProps<typeof badge> {
  status: ModuleStatus;
}

/** Module lowering status. Never colour-only: glyph + label (or title on the dot size). */
export function StatusBadge({ status, size, className, ...props }: StatusBadgeProps) {
  const m = STATUS[status];
  const dot = size === "dot";
  return (
    <span
      data-slot="status-badge"
      data-status={status}
      title={m.label}
      {...(dot ? { role: "img", "aria-label": `status: ${m.label}` } : {})}
      className={cn(badge({ size }), m.cls, className)}
      {...props}
    >
      <span aria-hidden="true" className="font-mono">
        {dot ? m.short : m.glyph}
      </span>
      {dot ? null : m.label}
    </span>
  );
}

export type NoteSeverity = "info" | "warn";

export interface NoteBadgeProps
  extends Omit<React.ComponentProps<"span">, "children">,
    VariantProps<typeof badge> {
  severity: NoteSeverity;
  count?: number;
}

/** Count of notes attached to a component; warn notes take the warn hue. */
export function NoteBadge({ severity, count = 1, size, className, ...props }: NoteBadgeProps) {
  const text = `${count} ${severity === "warn" ? "warning" : "note"}${count === 1 ? "" : "s"}`;
  return (
    <span
      data-slot="note-badge"
      data-severity={severity}
      title={text}
      role="img"
      aria-label={text}
      className={cn(
        badge({ size }),
        severity === "warn" ? "border-warn/45 text-warn" : "border-border text-muted-foreground",
        className,
      )}
      {...props}
    >
      <span aria-hidden="true" className="font-mono">
        {severity === "warn" ? "⚠" : "i"}
      </span>
      {size === "dot" ? null : count}
    </span>
  );
}

const variantBadge = cva(
  "inline-flex h-[17px] shrink-0 items-center rounded-sm px-1 font-mono text-[10px] font-semibold tracking-wide uppercase",
  {
    variants: {
      tone: {
        attention: "bg-arch-attn-ink/12 text-arch-attn-ink",
        ffn: "bg-arch-ffn-ink/12 text-arch-ffn-ink",
        norm: "bg-arch-norm-ink/12 text-arch-norm-ink",
        router: "bg-arch-router-ink/12 text-arch-router-ink",
        neutral: "bg-muted text-muted-foreground",
      },
    },
    defaultVariants: { tone: "neutral" },
  },
);

export interface VariantBadgeProps
  extends React.ComponentProps<"span">,
    VariantProps<typeof variantBadge> {}

/** Small uppercase variant tag (`GQA`, `MLA`, `SwiGLU`, `RMS`). */
export function VariantBadge({ tone, className, ...props }: VariantBadgeProps) {
  return (
    <span data-slot="variant-badge" className={cn(variantBadge({ tone }), className)} {...props} />
  );
}
