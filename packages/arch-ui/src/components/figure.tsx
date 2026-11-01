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
  shape: Shape;
  dtype?: string | undefined;
}

/** Model input / output tensor drawn as a pill at either end of the spine. */
export function FigureTerminal({ name, shape, dtype, className, ...props }: FigureTerminalProps) {
  return (
    <div
      data-slot="figure-terminal"
      className={cn(
        "inline-flex h-6 items-center gap-2 rounded-full border border-dashed border-arch-frame-border bg-background px-3 text-[11px]",
        className,
      )}
      {...props}
    >
      <span className="font-medium">{name}</span>
      <ShapeLabel shape={shape} dtype={dtype} />
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
