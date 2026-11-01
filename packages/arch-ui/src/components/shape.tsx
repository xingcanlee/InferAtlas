import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "../cn";
import { useArchFigure } from "../context";

/** A symbolic shape: one expression string per dimension (e.g. `["T", "Hq*D/tp"]`). */
export type Shape = readonly string[];

/** `[T, H]` — expressions shown as written, with light spacing around top-level operators. */
export function formatSymbolic(shape: Shape): string {
  return `[${shape.join(", ")}]`;
}

export function formatNumber(n: number): string {
  if (Number.isInteger(n)) return String(n);
  return Number(n.toPrecision(4)).toString();
}

/**
 * Evaluate every dimension; a dimension that cannot be evaluated is rendered as `?`.
 * Returns `undefined` when no evaluator is available at all.
 */
export function evaluateShape(
  shape: Shape,
  evaluate: ((expr: string) => number | undefined) | undefined,
): (number | undefined)[] | undefined {
  if (!evaluate) return undefined;
  return shape.map((d) => evaluate(d));
}

export function formatNumeric(values: readonly (number | undefined)[]): string {
  return `[${values.map((v) => (v === undefined ? "?" : formatNumber(v))).join(", ")}]`;
}

/** Number of elements, when every dimension evaluates. */
export function numel(values: readonly (number | undefined)[]): number | undefined {
  let n = 1;
  for (const v of values) {
    if (v === undefined) return undefined;
    n *= v;
  }
  return n;
}

const shapeLabel = cva(
  "group/shape relative inline-flex cursor-help items-baseline gap-1 whitespace-nowrap rounded-sm bg-transparent p-0 font-mono leading-none outline-none focus-visible:ring-2 focus-visible:ring-ring/60",
  {
    variants: {
      size: {
        xs: "text-[10px]",
        sm: "text-[11px]",
      },
      tone: {
        muted: "text-muted-foreground",
        default: "text-foreground",
      },
    },
    defaultVariants: { size: "xs", tone: "muted" },
  },
);

export interface ShapeLabelProps
  extends Omit<React.ComponentProps<"button">, "children">,
    VariantProps<typeof shapeLabel> {
  shape: Shape;
  /** Optional tensor name drawn before the shape (`hidden [T, H]`). */
  name?: string | undefined;
  dtype?: string | undefined;
  /**
   * `symbolic` (default) shows the expressions and reveals the evaluated shape on hover/focus;
   * `numeric` shows the evaluated shape and reveals the expressions.
   */
  mode?: "symbolic" | "numeric";
}

/**
 * A tensor shape label. A real button: the alternate form (numeric ↔ symbolic) appears in a
 * small tooltip on hover or keyboard focus, and clicking pins it (`aria-pressed`). Both forms are
 * exposed through `title`/`aria-label`.
 */
export function ShapeLabel({
  shape,
  name,
  dtype,
  mode = "symbolic",
  size,
  tone,
  className,
  ...props
}: ShapeLabelProps) {
  const { evaluate } = useArchFigure();
  const [flipped, setFlipped] = React.useState(false);
  const values = evaluateShape(shape, evaluate);
  const sym = formatSymbolic(shape);
  const num = values ? formatNumeric(values) : undefined;
  const numericFirst = (mode === "numeric") !== flipped;
  const shown = numericFirst && num ? num : sym;
  const alt = numericFirst ? sym : num;
  const label = [name, sym, num && num !== sym ? `= ${num}` : undefined, dtype]
    .filter(Boolean)
    .join(" ");
  return (
    <button
      type="button"
      aria-pressed={flipped}
      onClick={(e) => {
        e.stopPropagation();
        setFlipped((f) => !f);
      }}
      title={label}
      aria-label={label}
      data-slot="shape-label"
      data-symbolic={sym}
      data-numeric={num}
      className={cn(shapeLabel({ size, tone }), className)}
      {...props}
    >
      {name ? <span className="font-sans text-foreground/80">{name}</span> : null}
      <span data-testid="shape-text">{shown}</span>
      {dtype ? <span className="opacity-70">{dtype}</span> : null}
      {alt && alt !== shown ? (
        <span
          role="tooltip"
          className="pointer-events-none absolute top-full left-0 z-20 mt-1 hidden rounded border bg-popover px-1.5 py-1 text-[10px] text-popover-foreground group-hover/shape:block group-focus-visible/shape:block"
        >
          {alt}
        </span>
      ) : null}
    </button>
  );
}

export interface TensorChipProps extends Omit<React.ComponentProps<"span">, "children"> {
  name: string;
  shape: Shape;
}

/**
 * A tiny non-interactive tensor tag for use inside figure nodes (which may themselves be
 * buttons): `q [T, Hq*D/tp]`, evaluated shape in the title.
 */
export function TensorChip({ name, shape, className, ...props }: TensorChipProps) {
  const { evaluate } = useArchFigure();
  const values = evaluateShape(shape, evaluate);
  const sym = formatSymbolic(shape);
  return (
    <span
      data-slot="tensor-chip"
      data-numeric={values ? formatNumeric(values) : undefined}
      title={`${name} ${sym}${values ? ` = ${formatNumeric(values)}` : ""}`}
      className={cn(
        "inline-flex items-baseline gap-1 rounded-sm bg-background/70 px-1 font-mono text-[10px] leading-4 whitespace-nowrap text-muted-foreground",
        className,
      )}
      {...props}
    >
      <span className="text-foreground/80">{name}</span>
      {sym}
    </span>
  );
}
