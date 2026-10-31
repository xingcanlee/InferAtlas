import { BaseEdge, EdgeLabelRenderer, type EdgeProps } from "@xyflow/react";
import { cn } from "@/lib/utils";
import type { GraphEdge } from "./types";

/** Polyline through the precomputed orthogonal bend points, with 6px rounded corners. */
export function polylinePath(points: [number, number][], radius = 6): string {
  if (points.length === 0) return "";
  const [x0, y0] = points[0] as [number, number];
  let d = `M ${x0} ${y0}`;
  for (let i = 1; i < points.length; i++) {
    const [x, y] = points[i] as [number, number];
    const next = points[i + 1];
    if (!next) {
      d += ` L ${x} ${y}`;
      break;
    }
    const [px, py] = points[i - 1] as [number, number];
    const [nx, ny] = next;
    const lin = Math.hypot(x - px, y - py);
    const lout = Math.hypot(nx - x, ny - y);
    const r = Math.min(radius, lin / 2, lout / 2);
    if (r <= 0) {
      d += ` L ${x} ${y}`;
      continue;
    }
    const ax = x - ((x - px) / lin) * r;
    const ay = y - ((y - py) / lin) * r;
    const bx = x + ((nx - x) / lout) * r;
    const by = y + ((ny - y) / lout) * r;
    d += ` L ${ax} ${ay} Q ${x} ${y} ${bx} ${by}`;
  }
  return d;
}

export function OrthEdge({ id, data }: EdgeProps<GraphEdge>) {
  if (!data) return null;
  const { points, kind, label, labelPos, focus, changed } = data;
  const hot = focus || changed;
  const stroke = hot ? "var(--focus)" : kind === "residual" ? "var(--edge-soft)" : "var(--edge)";
  const dash = kind === "kv" ? "5 4" : kind === "residual" ? "2 3" : undefined;
  const mid = points[Math.floor(points.length / 2)] as [number, number];
  const [lx, ly] = labelPos ?? mid;
  return (
    <>
      <BaseEdge
        id={id}
        path={polylinePath(points)}
        markerEnd={hot ? "url(#ia-arrow-focus)" : "url(#ia-arrow)"}
        style={{ stroke, strokeWidth: hot ? 1.75 : 1.25, strokeDasharray: dash }}
        interactionWidth={12}
      />
      {label ? (
        <EdgeLabelRenderer>
          <div
            className={cn(
              "nodrag nopan pointer-events-none absolute rounded-[3px] border bg-canvas px-1 py-px font-mono text-[10px] leading-[14px] whitespace-nowrap tabular",
              hot ? "border-focus/50 text-focus" : "border-border text-muted-foreground",
            )}
            style={{ transform: `translate(-50%, -50%) translate(${lx}px, ${ly}px)` }}
            data-testid="edge-label"
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      ) : null}
    </>
  );
}

export function ArrowDefs() {
  return (
    <svg width="0" height="0" className="absolute" aria-hidden="true">
      <defs>
        <marker
          id="ia-arrow"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M0,1 L9,5 L0,9 z" fill="var(--edge)" />
        </marker>
        <marker
          id="ia-arrow-focus"
          viewBox="0 0 10 10"
          refX="9"
          refY="5"
          markerWidth="7"
          markerHeight="7"
          orient="auto-start-reverse"
        >
          <path d="M0,1 L9,5 L0,9 z" fill="var(--focus)" />
        </marker>
      </defs>
    </svg>
  );
}
