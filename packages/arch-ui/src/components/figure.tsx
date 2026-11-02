import type * as React from "react";
import { cn } from "../cn";
import { type ArchFigureContextValue, ArchFigureProvider } from "../context";
import { Flow } from "./flow";
import { type Shape, ShapeLabel } from "./shape";

export interface ModelFigureProps
  extends Omit<React.ComponentProps<"figure">, "title">,
    ArchFigureContextValue {
  /** Caption under the figure. */
  caption?: React.ReactNode;
  /** Column width of spine nodes (CSS length); default `--arch-col` from the theme. */
  columnWidth?: string | undefined;
}

/**
 * Whole-model vertical dataflow (Input → Embedding → Stack → Norm → LM head → Output). Children
 * are stacked on one spine with arrows inserted between them (use TensorEdge to annotate a
 * connector with its tensor shape, FigureBranch for a side head attached to a spine node).
 */
export function ModelFigure({
  caption,
  evaluate,
  columnWidth,
  className,
  style,
  children,
  ...props
}: ModelFigureProps) {
  return (
    <ArchFigureProvider evaluate={evaluate}>
      <figure
        data-slot="model-figure"
        className={cn("m-0 inline-flex flex-col items-center gap-3", className)}
        style={{ ...(columnWidth ? { "--arch-col": columnWidth } : {}), ...style }}
        {...props}
      >
        <Flow className="*:data-[slot=figure-node]:w-[var(--arch-col)]">{children}</Flow>
        {caption ? (
          <figcaption className="max-w-xl text-center text-[11.5px] text-muted-foreground">
            {caption}
          </figcaption>
        ) : null}
      </figure>
    </ArchFigureProvider>
  );
}

export interface FigureTerminalProps extends React.ComponentProps<"div"> {
  name: string;
  /** Omit when the tensor's shape is unknown (the pill then shows only its name). */
  shape?: Shape | undefined;
  dtype?: string | undefined;
  /**
   * Placeholder terminal: the tensor is not in the data (e.g. an image input the arch.json does
   * not list). Drawn muted, with `hint` explaining why.
   */
  muted?: boolean;
  hint?: React.ReactNode;
}

/** Model input / output tensor drawn as a pill at either end of the spine. */
export function FigureTerminal({
  name,
  shape,
  dtype,
  muted,
  hint,
  className,
  ...props
}: FigureTerminalProps) {
  return (
    <div
      data-slot="figure-terminal"
      data-muted={muted || undefined}
      className={cn(
        "inline-flex h-6 items-center gap-2 rounded-full border border-dashed border-arch-frame-border bg-background px-3 text-[11px]",
        muted && "text-muted-foreground",
        className,
      )}
      {...props}
    >
      <span className={cn("font-medium", muted && "font-normal italic")}>{name}</span>
      {shape ? <ShapeLabel shape={shape} dtype={dtype} /> : null}
      {hint ? <span className="text-[10px] text-muted-foreground">{hint}</span> : null}
    </div>
  );
}

export interface FigureBranchProps extends Omit<React.ComponentProps<"div">, "children"> {
  /** The spine node the branch leaves from (stays centred on the spine). */
  node: React.ReactNode;
  /** The side head(s), drawn to the right. */
  branch: React.ReactNode;
  /** Text on the dashed connector. */
  label?: React.ReactNode;
}

/**
 * A side branch (MTP / draft head) attached to a spine node: a dashed connector leaves the spine
 * at a junction dot on the node's edge and points at the branch. The row reserves the same width
 * on the left so the spine stays centred.
 */
export function FigureBranch({ node, branch, label, className, ...props }: FigureBranchProps) {
  return (
    <div
      data-slot="figure-branch"
      className={cn(
        "grid grid-cols-[calc(var(--arch-col)+6rem)_var(--arch-col)_calc(var(--arch-col)+6rem)] items-center",
        className,
      )}
      {...props}
    >
      <div aria-hidden="true" />
      <div className="relative flex w-[var(--arch-col)] flex-col items-center">
        {node}
        <span
          aria-hidden="true"
          className="absolute top-1/2 -right-[3px] z-[2] size-[6px] -translate-y-1/2 rounded-full bg-arch-wire"
        />
      </div>
      {/* Out of flow vertically: the branch never changes the spine row's height. */}
      <div className="relative h-full">
        <div className="absolute top-1/2 left-0 flex -translate-y-1/2 items-center">
          <div className="flex w-24 shrink-0 flex-col items-stretch gap-0.5 text-arch-wire">
            {label ? (
              <span className="px-1 text-center text-[10px] leading-tight text-muted-foreground">
                {label}
              </span>
            ) : null}
            <span className="flex items-center">
              <span className="h-0 flex-1 border-t border-dashed border-current" />
              <svg viewBox="0 0 7 9" aria-hidden="true" className="h-[9px] w-[7px]">
                <path d="M0 0 L7 4.5 L0 9 Z" fill="currentColor" />
              </svg>
            </span>
          </div>
          <div
            data-slot="figure-branch-content"
            className="flex w-[var(--arch-col)] flex-col gap-2"
          >
            {branch}
          </div>
        </div>
      </div>
    </div>
  );
}

export interface FigureMergeProps extends Omit<React.ComponentProps<"div">, "children"> {
  /** The side branch (e.g. a vision encoder), drawn left of the spine; may be any height. */
  side: React.ReactNode;
  /** Spine content above the merge node (e.g. the `input_ids` terminal). */
  top?: React.ReactNode;
  /** The spine node the branch merges into (e.g. the token embedding). */
  node: React.ReactNode;
  /** Text on the merge connector (what the branch contributes). */
  label?: React.ReactNode;
}

/**
 * A second input branch that merges into the spine below `node` (paper-style VLM figure: the
 * vision encoder's features join the token embeddings before the decoder). The side column sits
 * left of the spine; an elbow wire leaves its bottom and points right into a junction dot on the
 * spine. Columns left/right of the spine are equal (`1fr`), so the spine stays centred whatever
 * the branch's width.
 */
export function FigureMerge({ side, top, node, label, className, ...props }: FigureMergeProps) {
  return (
    <div
      data-slot="figure-merge"
      className={cn("inline-grid grid-cols-[1fr_auto_1fr] text-arch-wire", className)}
      {...props}
    >
      <div data-slot="figure-merge-side" className="flex flex-col items-center px-4">
        {side}
        <span aria-hidden="true" className="min-h-3 w-px flex-1 bg-current" />
      </div>
      <div className="flex w-[var(--arch-col)] flex-col items-center">
        {top}
        {top ? (
          <span aria-hidden="true" className="relative min-h-4 w-px flex-1 bg-current">
            <svg
              viewBox="0 0 9 7"
              aria-hidden="true"
              className="absolute -bottom-px left-[-4px] h-[7px] w-[9px]"
            >
              <path d="M0 0 L4.5 7 L9 0 Z" fill="currentColor" />
            </svg>
          </span>
        ) : null}
        <div className="flex w-full flex-col items-center *:data-[slot=figure-node]:w-full">
          {node}
        </div>
      </div>
      <div aria-hidden="true" />
      {/* Elbow: down from the side column's centre, then right into the spine. */}
      <div className="relative h-6">
        <span
          aria-hidden="true"
          className="absolute top-0 left-1/2 h-1/2 border-l border-current"
        />
        <span
          aria-hidden="true"
          className="absolute top-1/2 right-0 left-1/2 border-t border-current"
        />
        {label ? (
          <span
            data-slot="figure-merge-label"
            className="absolute right-1 bottom-1/2 left-[calc(50%+6px)] mb-0.5 truncate text-right text-[10px] leading-tight text-muted-foreground"
          >
            {label}
          </span>
        ) : null}
      </div>
      <div className="relative h-6">
        <span aria-hidden="true" className="absolute inset-y-0 left-1/2 border-l border-current" />
        <span
          aria-hidden="true"
          className="absolute top-1/2 right-[calc(50%+4px)] left-0 border-t border-current"
        />
        <svg
          viewBox="0 0 7 9"
          aria-hidden="true"
          className="absolute top-[calc(50%-4.5px)] right-[calc(50%+3px)] h-[9px] w-[7px]"
        >
          <path d="M0 0 L7 4.5 L0 9 Z" fill="currentColor" />
        </svg>
        <span
          aria-hidden="true"
          className="absolute top-1/2 left-1/2 z-[2] size-[7px] -translate-x-1/2 -translate-y-1/2 rounded-full bg-current"
        />
      </div>
      <div aria-hidden="true" />
    </div>
  );
}
