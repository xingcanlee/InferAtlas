import { cva, type VariantProps } from "class-variance-authority";
import * as React from "react";
import { cn } from "../cn";
import { type Shape, ShapeLabel } from "./shape";

const arrow = cva("relative flex w-full shrink-0 justify-center text-arch-wire", {
  variants: {
    length: {
      sm: "h-3",
      md: "h-4",
      lg: "h-5",
      xl: "h-8",
    },
  },
  defaultVariants: { length: "md" },
});

export interface FlowArrowProps extends React.ComponentProps<"div">, VariantProps<typeof arrow> {
  /** Drawn to the right of the wire, vertically centred (e.g. a ShapeLabel). */
  label?: React.ReactNode;
  /** Plain wire without arrow head (e.g. a continuation segment). */
  plain?: boolean;
  dashed?: boolean;
}

/** Vertical wire with an arrow head pointing down the dataflow. */
export function FlowArrow({ label, plain, dashed, length, className, ...props }: FlowArrowProps) {
  return (
    <div
      aria-hidden={label ? undefined : true}
      data-slot="flow-arrow"
      className={cn(arrow({ length }), className)}
      {...props}
    >
      <span
        className={cn(
          "absolute top-0 left-1/2 -ml-px w-px",
          plain ? "bottom-0" : "bottom-1",
          dashed
            ? "bg-[repeating-linear-gradient(to_bottom,currentColor_0_3px,transparent_3px_5px)]"
            : "bg-current",
        )}
      />
      {plain ? null : (
        <svg
          className="absolute bottom-0 left-[calc(50%-5px)] h-[7px] w-[9px]"
          viewBox="0 0 9 7"
          aria-hidden="true"
        >
          <path d="M0 0 L4.5 7 L9 0 Z" fill="currentColor" />
        </svg>
      )}
      {label ? (
        // On the canvas colour so wires never cross the text; node corner badges (later in
        // the DOM) still paint above a long label.
        <div className="absolute top-1/2 left-[calc(50%+10px)] -translate-y-1/2 rounded-sm bg-arch-surface px-0.5 leading-none">
          {label}
        </div>
      ) : null}
    </div>
  );
}

export interface TensorEdgeProps extends Omit<FlowArrowProps, "label"> {
  shape: Shape;
  name?: string | undefined;
  dtype?: string | undefined;
}

/** A dataflow arrow annotated with the tensor it carries. */
export function TensorEdge({ shape, name, dtype, length = "lg", ...props }: TensorEdgeProps) {
  return (
    <FlowArrow
      length={length}
      data-slot="tensor-edge"
      label={<ShapeLabel shape={shape} name={name} dtype={dtype} />}
      {...props}
    />
  );
}

const isEdge = (c: React.ReactNode) =>
  React.isValidElement(c) && (c.type === FlowArrow || c.type === TensorEdge);

export interface FlowProps extends React.ComponentProps<"div"> {
  /** Arrow length used for the auto-inserted connectors. */
  gap?: VariantProps<typeof arrow>["length"];
}

/**
 * A vertical spine: renders its children top to bottom and inserts a FlowArrow between two
 * consecutive nodes unless an explicit FlowArrow/TensorEdge is already there.
 */
export function Flow({ gap = "md", className, children, ...props }: FlowProps) {
  const items = React.Children.toArray(children).filter(Boolean);
  const out: React.ReactNode[] = [];
  items.forEach((child, i) => {
    const prev = items[i - 1];
    if (i > 0 && !isEdge(child) && !isEdge(prev)) {
      // biome-ignore lint/suspicious/noArrayIndexKey: connectors are positional
      out.push(<FlowArrow key={`arrow-${i}`} length={gap} />);
    }
    out.push(child);
  });
  return (
    <div data-slot="flow" className={cn("flex w-full flex-col items-center", className)} {...props}>
      {out}
    </div>
  );
}
