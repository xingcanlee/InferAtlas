import { cva, type VariantProps } from "class-variance-authority";
import type * as React from "react";
import { cn } from "../cn";
import { type ModuleStatus, NoteBadge, type NoteSeverity, StatusBadge } from "./badges";

export const figureNode = cva(
  "relative flex w-full flex-col items-center rounded-md border text-center leading-tight text-foreground outline-none transition-[border-color,box-shadow]",
  {
    variants: {
      kind: {
        norm: "border-arch-norm-border bg-arch-norm",
        attention: "border-arch-attn-border bg-arch-attn",
        mlp: "border-arch-ffn-border bg-arch-ffn",
        moe: "border-arch-ffn-border bg-arch-ffn",
        router: "border-arch-router-border bg-arch-router",
        shared: "border-arch-shared-border bg-arch-shared",
        embedding: "border-arch-embed-border bg-arch-embed",
        head: "border-arch-head-border bg-arch-head",
        generic: "border-dashed border-arch-generic-border bg-arch-generic",
        plain: "border-border bg-card",
      },
      size: {
        sm: "min-h-6 gap-0.5 px-2 py-1 text-[11px]",
        md: "min-h-8 gap-0.5 px-2.5 py-1 text-[12px]",
      },
      interactive: {
        true: "cursor-pointer hover:border-foreground/45 focus-visible:ring-2 focus-visible:ring-ring/70 focus-visible:ring-offset-1 focus-visible:ring-offset-background",
        false: "",
      },
      selected: {
        true: "ring-2 ring-focus ring-offset-1 ring-offset-background",
        false: "",
      },
    },
    defaultVariants: { kind: "plain", size: "md", interactive: false, selected: false },
  },
);

const titleInk: Record<NonNullable<VariantProps<typeof figureNode>["kind"]>, string> = {
  norm: "text-arch-norm-ink",
  attention: "text-arch-attn-ink",
  mlp: "text-arch-ffn-ink",
  moe: "text-arch-ffn-ink",
  router: "text-arch-router-ink",
  shared: "text-arch-shared-ink",
  embedding: "text-arch-embed-ink",
  head: "text-arch-head-ink",
  generic: "text-arch-generic-ink",
  plain: "text-foreground",
};

export type FigureNodeKind = NonNullable<VariantProps<typeof figureNode>["kind"]>;

/** Status/notes markers shared by every figure node. */
export interface NodeMarkers {
  /** Module lowering status; `ruled` draws no marker in the figure. */
  status?: ModuleStatus | undefined;
  /** Notes attached to the component, by severity. */
  notes?: { severity: NoteSeverity; count: number } | undefined;
}

export interface FigureNodeProps
  extends Omit<React.HTMLAttributes<HTMLElement>, "title">,
    Omit<VariantProps<typeof figureNode>, "interactive">,
    NodeMarkers {
  title?: React.ReactNode;
  /** Small tag next to the title (e.g. a VariantBadge). */
  badge?: React.ReactNode;
  subtitle?: React.ReactNode;
  /** Makes the node a button. */
  onSelect?: (() => void) | undefined;
  /** Accessible name when the visible title is not enough. */
  label?: string | undefined;
}

/**
 * The box every figure component is built from. Interactive nodes are real buttons
 * (Tab / Enter / Space) with `aria-pressed` reflecting selection.
 */
export function FigureNode({
  kind,
  size,
  selected,
  title,
  badge,
  subtitle,
  status,
  notes,
  onSelect,
  label,
  className,
  children,
  ...props
}: FigureNodeProps) {
  const k = kind ?? "plain";
  const content = (
    <>
      {title || badge ? (
        <span className="flex max-w-full items-center justify-center gap-1.5">
          {title ? (
            <span className={cn("truncate font-semibold", titleInk[k])}>{title}</span>
          ) : null}
          {badge}
        </span>
      ) : null}
      {subtitle ? (
        <span className="max-w-full truncate text-[10.5px] text-muted-foreground">{subtitle}</span>
      ) : null}
      {children}
      {(status && status !== "ruled") || notes ? (
        <span className="absolute -top-2 -right-2 flex gap-0.5">
          {status && status !== "ruled" ? <StatusBadge status={status} size="dot" /> : null}
          {notes ? <NoteBadge severity={notes.severity} count={notes.count} size="dot" /> : null}
        </span>
      ) : null}
    </>
  );
  const cls = cn(
    figureNode({ kind: k, size, selected, interactive: !!onSelect }),
    status === "generic" && "border-dashed",
    className,
  );
  if (onSelect) {
    return (
      <button
        type="button"
        data-slot="figure-node"
        data-kind={k}
        aria-pressed={!!selected}
        aria-label={label}
        className={cls}
        onClick={(e) => {
          e.stopPropagation();
          onSelect();
        }}
        {...(props as React.ButtonHTMLAttributes<HTMLButtonElement>)}
      >
        {content}
      </button>
    );
  }
  return (
    <div data-slot="figure-node" data-kind={k} title={label} className={cls} {...props}>
      {content}
    </div>
  );
}

/** Shared props for the semantic components below. */
export interface ComponentNodeProps
  extends Omit<FigureNodeProps, "kind" | "title" | "badge" | "subtitle">,
    NodeMarkers {}

const chipTone: Record<FigureNodeKind | "accent", string> = {
  norm: "border-arch-norm-border bg-arch-norm",
  attention: "border-arch-attn-border bg-arch-attn",
  mlp: "border-arch-ffn-border bg-arch-ffn",
  moe: "border-arch-ffn-border bg-arch-ffn",
  router: "border-arch-router-border bg-arch-router",
  shared: "border-arch-shared-border bg-arch-shared",
  embedding: "border-arch-embed-border bg-arch-embed",
  head: "border-arch-head-border bg-arch-head",
  generic: "border-dashed border-arch-generic-border bg-arch-generic",
  plain: "border-border bg-background",
  accent: "border-focus/70 bg-focus-soft",
};

/** A small inline chip used inside composite nodes (projections, expert boxes…). */
export function Chip({
  className,
  tone = "plain",
  ...props
}: React.ComponentProps<"span"> & { tone?: FigureNodeKind | "accent" }) {
  return (
    <span
      data-slot="chip"
      data-tone={tone}
      className={cn(
        "inline-flex h-[19px] shrink-0 items-center justify-center rounded-[3px] border px-1 font-mono text-[10px] leading-none whitespace-nowrap text-foreground",
        chipTone[tone],
        className,
      )}
      {...props}
    />
  );
}
