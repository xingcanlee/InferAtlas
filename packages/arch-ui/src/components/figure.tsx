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
  /** Side branch drawn to the right of the lower spine (e.g. an MTP head). */
  aside?: React.ReactNode;
  /** Column width of spine nodes (CSS length); default `--arch-col` from the theme. */
  columnWidth?: string | undefined;
}

/**
 * Whole-model vertical dataflow (Input → Embedding → Stack → Norm → LM head → Output). Children
 * are stacked on one spine with arrows inserted between them (use TensorEdge to annotate a
 * connector with its tensor shape).
 */
export function ModelFigure({
  caption,
  aside,
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
        {/* The aside only takes space on its own side, so a wide stack is not shrunk twice. */}
        <div className="flex items-end">
          <Flow className="*:data-[slot=figure-node]:w-[var(--arch-col)]">{children}</Flow>
          {aside ? <div className="shrink-0">{aside}</div> : null}
        </div>
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

export interface FigureAsideProps extends React.ComponentProps<"div"> {
  /** Text on the dashed connector from the spine. */
  label?: React.ReactNode;
}

/** A side branch: dashed connector from the spine + its content. */
export function FigureAside({ label, className, children, ...props }: FigureAsideProps) {
  return (
    <div data-slot="figure-aside" className={cn("flex items-center pb-1", className)} {...props}>
      <div className="flex w-20 flex-col items-center gap-0.5 text-arch-wire">
        {label ? (
          <span className="text-center text-[10px] leading-tight text-muted-foreground">
            {label}
          </span>
        ) : null}
        <span className="flex w-full items-center">
          <span className="h-0 flex-1 border-t border-dashed border-current" />
          <svg viewBox="0 0 7 9" aria-hidden="true" className="h-[9px] w-[7px]">
            <path d="M0 0 L7 4.5 L0 9 Z" fill="currentColor" />
          </svg>
        </span>
      </div>
      <div className="w-[var(--arch-col)]">{children}</div>
    </div>
  );
}
