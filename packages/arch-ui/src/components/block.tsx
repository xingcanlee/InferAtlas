import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "../cn";
import { Flow, FlowArrow } from "./flow";
import { Norm, ResidualAdd } from "./layers";

// -------------------------------------------------------------------------- ResidualBranch

export interface ResidualBranchProps extends React.ComponentProps<"div"> {
  /** Props for the closing ⊕ (selection, test ids…). */
  add?: React.ComponentProps<typeof ResidualAdd>;
  /** Hide the skip line (e.g. residual style unknown) but keep the vertical flow. */
  noSkip?: boolean;
}

/**
 * One residual segment: the stream splits at the top, the children run down the spine and the
 * skip line rejoins at ⊕. Geometry is pure CSS (the rail is a bordered box whose right edge is
 * the spine), so it scales with any content height.
 */
export function ResidualBranch({
  add,
  noSkip,
  className,
  children,
  ...props
}: ResidualBranchProps) {
  return (
    <div
      data-slot="residual-branch"
      className={cn("relative flex flex-col items-center px-[var(--arch-rail)]", className)}
      {...props}
    >
      {noSkip ? null : (
        <>
          {/* skip line: from the branch point (top) to the ⊕ centre (bottom) */}
          <span
            aria-hidden="true"
            className="pointer-events-none absolute top-1 right-1/2 bottom-2.5 left-[calc(var(--arch-rail)*0.4)] rounded-l-md border-y border-l border-arch-wire"
          />
          <span
            aria-hidden="true"
            className="pointer-events-none absolute top-[3px] left-1/2 -ml-[3px] size-[5px] rounded-full bg-arch-wire"
          />
          <svg
            aria-hidden="true"
            viewBox="0 0 7 9"
            className="pointer-events-none absolute right-[calc(50%+10px)] bottom-[6px] h-[9px] w-[7px] text-arch-wire"
          >
            <path d="M0 0 L7 4.5 L0 9 Z" fill="currentColor" />
          </svg>
        </>
      )}
      <div className="flex w-[var(--arch-col)] flex-col items-center">
        <FlowArrow />
        <Flow>{children}</Flow>
        <FlowArrow />
        {noSkip ? null : <ResidualAdd {...add} />}
      </div>
    </div>
  );
}

/** A node on the main stream inside a block (outside any residual branch), with its arrow. */
export function BlockStep({ className, children, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="block-step"
      className={cn("flex w-[var(--arch-col)] flex-col items-center", className)}
      {...props}
    >
      <FlowArrow />
      {children}
    </div>
  );
}

// ---------------------------------------------------------------------------- DecoderBlock

export type ResidualStyle = "pre_norm" | "post_norm" | "sandwich" | "unknown";

export interface DecoderBlockProps extends React.ComponentProps<"div"> {
  residual?: ResidualStyle;
  /** Token mixer (attention / linear attention / generic). */
  mixer?: React.ReactNode;
  /** Channel mixer (MLP / MoE / generic). */
  ffn?: React.ReactNode;
  /** Norm before the mixer (pre/sandwich) or after its ⊕ (post). Defaults to RMSNorm. */
  mixerNorm?: React.ReactNode;
  ffnNorm?: React.ReactNode;
  /** Sandwich only: norms applied to the branch outputs before ⊕. */
  mixerPostNorm?: React.ReactNode;
  ffnPostNorm?: React.ReactNode;
}

/**
 * A transformer block drawn as in the papers. Either pass `mixer`/`ffn` (+ optional norms) for
 * the canonical layout of the given residual style, or compose `ResidualBranch` / nodes as
 * children for anything else (children win).
 */
export function DecoderBlock({
  residual = "pre_norm",
  mixer,
  ffn,
  mixerNorm = <Norm />,
  ffnNorm = <Norm />,
  mixerPostNorm = <Norm />,
  ffnPostNorm = <Norm />,
  className,
  children,
  ...props
}: DecoderBlockProps) {
  let body: React.ReactNode = children;
  if (!body) {
    switch (residual) {
      case "post_norm":
        body = (
          <>
            <ResidualBranch>{mixer}</ResidualBranch>
            <BlockStep>{mixerNorm}</BlockStep>
            <ResidualBranch>{ffn}</ResidualBranch>
            <BlockStep>{ffnNorm}</BlockStep>
          </>
        );
        break;
      case "sandwich":
        body = (
          <>
            <ResidualBranch>
              {mixerNorm}
              {mixer}
              {mixerPostNorm}
            </ResidualBranch>
            <ResidualBranch>
              {ffnNorm}
              {ffn}
              {ffnPostNorm}
            </ResidualBranch>
          </>
        );
        break;
      case "unknown":
        body = (
          <ResidualBranch noSkip>
            {mixerNorm}
            {mixer}
            {ffnNorm}
            {ffn}
          </ResidualBranch>
        );
        break;
      default:
        body = (
          <>
            <ResidualBranch>
              {mixerNorm}
              {mixer}
            </ResidualBranch>
            <ResidualBranch>
              {ffnNorm}
              {ffn}
            </ResidualBranch>
          </>
        );
    }
  }
  return (
    <div
      data-slot="decoder-block"
      data-residual={residual}
      className={cn("flex flex-col items-center", className)}
      {...props}
    >
      {body}
      <div className="flex w-[var(--arch-col)] flex-col items-center">
        <FlowArrow />
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------------ Band

/** `58×` as drawn next to a repeated block. */
export function RepeatCount({
  count,
  className,
  ...props
}: React.ComponentProps<"span"> & { count: number }) {
  return (
    <span
      data-slot="repeat-count"
      className={cn(
        "shrink-0 font-semibold text-[15px] text-foreground/85 tabular-nums tracking-tight",
        className,
      )}
      {...props}
    >
      <span aria-hidden="true">{count}×</span>
      <span className="sr-only">repeated {count} times</span>
    </span>
  );
}

const bandFrame = cva(
  "rounded-xl border bg-arch-frame px-1 pt-1 transition-[border-color,box-shadow]",
  {
    variants: {
      selected: {
        true: "border-focus/70 ring-2 ring-focus/25",
        false: "border-arch-frame-border",
      },
    },
    defaultVariants: { selected: false },
  },
);

export interface BandProps
  extends Omit<React.ComponentProps<"div">, "title">,
    VariantProps<typeof bandFrame> {
  /** Header text, e.g. `layers 3–60`. */
  title: React.ReactNode;
  /** Secondary header text, e.g. `MoE`. */
  subtitle?: React.ReactNode;
  /** Layers in the band; drawn as `N×` beside the block. */
  count: number;
  /** Header becomes a disclosure button (`aria-expanded` = selected). */
  onSelect?: (() => void) | undefined;
  /** id of the element the header controls (detail panel). */
  controls?: string | undefined;
}

/** A run of identical layers: header + `N×` + the block drawn once. */
export function Band({
  title,
  subtitle,
  count,
  selected,
  onSelect,
  controls,
  className,
  children,
  ...props
}: BandProps) {
  const header = (
    <>
      <span className="font-semibold text-foreground">{title}</span>
      {subtitle ? <span className="text-muted-foreground">{subtitle}</span> : null}
    </>
  );
  return (
    <div
      data-slot="band"
      data-selected={selected || undefined}
      className={cn("flex flex-col items-center gap-1", className)}
      {...props}
    >
      {onSelect ? (
        <button
          type="button"
          data-slot="band-header"
          aria-expanded={!!selected}
          aria-controls={controls}
          onClick={onSelect}
          className={cn(
            "inline-flex h-6 items-center gap-1.5 rounded-md px-2 text-[11.5px] outline-none transition-colors hover:bg-accent focus-visible:ring-2 focus-visible:ring-ring/70",
            selected && "bg-focus-soft",
          )}
        >
          {header}
          <svg
            viewBox="0 0 10 10"
            aria-hidden="true"
            className={cn(
              "size-2.5 text-muted-foreground transition-transform",
              selected && "rotate-90",
            )}
          >
            <path d="M3.5 1.5 L7 5 L3.5 8.5" fill="none" stroke="currentColor" strokeWidth="1.5" />
          </svg>
        </button>
      ) : (
        <div className="inline-flex h-6 items-center gap-1.5 px-2 text-[11.5px]">{header}</div>
      )}
      <div className="flex items-center gap-2">
        <RepeatCount count={count} />
        <div className={bandFrame({ selected })}>{children}</div>
      </div>
    </div>
  );
}

// ------------------------------------------------------------------------------ BlockStack

/** Horizontal "then" connector between consecutive bands. */
function Then() {
  return (
    <span
      aria-hidden="true"
      data-slot="band-then"
      className="flex shrink-0 items-center self-center text-arch-wire"
    >
      <span className="h-px w-4 bg-current" />
      <svg viewBox="0 0 7 9" className="h-[9px] w-[7px]" aria-hidden="true">
        <path d="M0 0 L7 4.5 L0 9 Z" fill="currentColor" />
      </svg>
    </span>
  );
}

function withThen(children: React.ReactNode): React.ReactNode[] {
  const items = React.Children.toArray(children).filter(Boolean);
  return items.flatMap((c, i) =>
    // biome-ignore lint/suspicious/noArrayIndexKey: connectors are positional
    i === 0 ? [c] : [<Then key={`then-${i}`} />, c],
  );
}

export interface BlockStackProps extends Omit<React.ComponentProps<"section">, "title"> {
  title?: React.ReactNode;
  /** Total layers (shown in the header). */
  layers?: number | undefined;
}

/**
 * The decoder stack: a dashed frame containing its bands left → right in execution order.
 * A single band reads exactly like the classic "N×" block figure.
 */
export function BlockStack({
  title = "Decoder layers",
  layers,
  className,
  children,
  ...props
}: BlockStackProps) {
  return (
    <section
      data-slot="block-stack"
      className={cn(
        "relative flex flex-col items-center rounded-2xl border border-dashed border-arch-frame-border px-4 pt-2 pb-3",
        className,
      )}
      {...props}
    >
      <header className="mb-1 flex items-baseline gap-1.5 text-[11px] text-muted-foreground">
        <span className="font-medium tracking-wide uppercase">{title}</span>
        {layers !== undefined ? <span className="font-mono tabular-nums">· {layers}</span> : null}
      </header>
      <div className="flex items-start justify-center gap-3">{withThen(children)}</div>
    </section>
  );
}

export interface PatternGroupProps extends Omit<React.ComponentProps<"div">, "title"> {
  title: React.ReactNode;
  subtitle?: React.ReactNode;
  /** How many times the period repeats. */
  repeats: number;
}

/** A periodic run of bands (e.g. 3 × linear attention + 1 × full attention, repeated 12×). */
export function PatternGroup({
  title,
  subtitle,
  repeats,
  className,
  children,
  ...props
}: PatternGroupProps) {
  return (
    <div
      data-slot="pattern-group"
      className={cn("flex flex-col items-center gap-1", className)}
      {...props}
    >
      <div className="inline-flex h-6 items-center gap-1.5 px-2 text-[11.5px]">
        <span className="font-semibold">{title}</span>
        {subtitle ? <span className="text-muted-foreground">{subtitle}</span> : null}
      </div>
      <div className="flex items-stretch gap-1.5">
        <div className="flex items-start gap-3 rounded-2xl border border-arch-frame-border px-3 pt-1 pb-3">
          {withThen(children)}
        </div>
        {/* period bracket on the right so it never reads as part of a band's own N× */}
        <span
          aria-hidden="true"
          className="my-3 w-1.5 rounded-r-sm border-y border-r border-arch-wire"
        />
        <RepeatCount count={repeats} className="self-center" />
      </div>
    </div>
  );
}
