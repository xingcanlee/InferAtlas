import {
  Background,
  BackgroundVariant,
  Controls,
  getViewportForBounds,
  ReactFlow,
  ReactFlowProvider,
  type Rect,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { useEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";
import { type BuildOptions, buildGraph, contentBounds } from "./build";
import { nodeTypes } from "./nodes";
import { ArrowDefs, OrthEdge } from "./OrthEdge";
import type { GraphNode } from "./types";

const edgeTypes = { orth: OrthEdge };

export interface GraphCanvasProps extends BuildOptions {
  onSelect: (irNode: string | undefined) => void;
  onDrill: (irNode: string) => void;
  className?: string;
  /**
   * Readable floor for views with more than 12 nodes: they pan instead of shrinking text below
   * this zoom. Smaller views fit whole down to 0.6 (or this value, if lower).
   */
  minFitZoom?: number;
  /** Distinguishes two canvases on the diff page for tests. */
  testId?: string;
}

/** Views this small are fitted whole down to `SMALL_VIEW_MIN_ZOOM` (12.5px titles → ~7.5px). */
const SMALL_VIEW_NODES = 12;
/** 12.5px node titles stay at ~10px or larger. */
const READABLE_ZOOM = 0.8;
const SMALL_VIEW_MIN_ZOOM = 0.6;
/** Never magnify a fitted view beyond this. */
const MAX_FIT_ZOOM = 1.1;
const FIT_PADDING = "12px";
/** Resizes (window, analysis panel) settle before the camera re-fits. */
const RESIZE_DEBOUNCE_MS = 120;

/**
 * Place the precomputed layout. The whole view (nodes, edges and labels) is fitted whenever
 * that keeps zoom at or above the readable floor; otherwise the view is fitted to the canvas
 * width at no less than the floor and panned to the anchor (the selected node, else the first
 * drillable group). Re-runs on view change and, debounced, on canvas resize (window or panel).
 */
function FitOnChange({
  viewKey,
  bounds: { x: bx, y: by, width: bw, height: bh },
  minZoom,
  focus,
}: {
  viewKey: string;
  bounds: Rect;
  minZoom: number;
  /** Center (graph px) to keep in view when the graph does not fit. */
  focus: { x: number; y: number } | undefined;
}) {
  const rf = useReactFlow();
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  // Focus only steers placement; clicking must not move the camera.
  const focusRef = useRef(focus);
  focusRef.current = focus;
  const placedKey = useRef<string | undefined>(undefined);
  useEffect(() => {
    if (!width || !height) return;
    const place = () => {
      const bounds = { x: bx, y: by, width: bw, height: bh };
      const fit = getViewportForBounds(bounds, width, height, 0, MAX_FIT_ZOOM, FIT_PADDING);
      if (fit.zoom >= minZoom) {
        rf.setViewport(fit);
        return;
      }
      // Too big to fit readably: fit the width (never below the floor) and pan to the anchor.
      const f = focusRef.current;
      const z = Math.max(minZoom, Math.min((width - 24) / bw, MAX_FIT_ZOOM));
      const axis = (avail: number, start: number, size: number, at: number | undefined) => {
        if (size * z <= avail) return (avail - size * z) / 2 - start * z;
        const want = at === undefined ? 8 - start * z : avail / 2 - at * z;
        return Math.min(8 - start * z, Math.max(avail - (start + size) * z - 8, want));
      };
      rf.setViewport({ x: axis(width, bx, bw, f?.x), y: axis(height, by, bh, f?.y), zoom: z });
    };
    // A new view is placed immediately (no flash); resizes are debounced.
    if (placedKey.current !== viewKey) {
      placedKey.current = viewKey;
      place();
      return;
    }
    const t = window.setTimeout(place, RESIZE_DEBOUNCE_MS);
    return () => window.clearTimeout(t);
  }, [rf, viewKey, width, height, bx, by, bw, bh, minZoom]);
  return null;
}

function Canvas(p: GraphCanvasProps) {
  const {
    ir,
    view,
    graph,
    metrics,
    env,
    phase,
    shapes,
    heat,
    overlay,
    selected,
    nodeDiff,
    changedEdges,
  } = p;
  const { nodes, edges } = useMemo(
    () =>
      buildGraph({
        ir,
        view,
        graph,
        metrics,
        env,
        phase,
        shapes,
        heat,
        overlay,
        selected,
        nodeDiff,
        changedEdges,
      }),
    [ir, view, graph, metrics, env, phase, shapes, heat, overlay, selected, nodeDiff, changedEdges],
  );
  const bounds = useMemo(() => contentBounds(nodes, edges), [nodes, edges]);
  // Anchor for views too big to fit: the selected node, else the first changed node, else the
  // first drillable group (e.g. `decoder`, the node a first-time viewer opens).
  const focusNode =
    nodes.find((n) => n.data.selected) ??
    nodes.find((n) => n.data.diff && n.data.diff !== "removed") ??
    nodes.find((n) => n.data.drillable);
  const focus = focusNode
    ? {
        x: focusNode.position.x + (focusNode.width ?? 0) / 2,
        y: focusNode.position.y + (focusNode.height ?? 0) / 2,
      }
    : undefined;
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "Enter" && e.key !== " ") return;
    const el = (e.target as HTMLElement).closest<HTMLElement>(".react-flow__node");
    const n = el ? nodes.find((x) => x.id === el.dataset.id) : undefined;
    if (!n) return;
    e.preventDefault();
    e.stopPropagation();
    const id = n.data.vn.ir_node;
    // Enter opens an already-selected drillable node; otherwise both keys select.
    if (e.key === "Enter" && n.data.selected && n.data.drillable) p.onDrill(id);
    else p.onSelect(e.key === " " && n.data.selected ? undefined : id);
  };
  return (
    // biome-ignore lint/a11y/noStaticElementInteractions: delegates keys from focusable React Flow nodes
    <div
      className={cn("relative h-full w-full", p.className)}
      data-testid={p.testId ?? "graph"}
      onKeyDown={onKeyDown}
    >
      <ArrowDefs />
      <ReactFlow<GraphNode>
        nodes={nodes}
        edges={edges}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        panOnScroll
        fitViewOptions={{ padding: FIT_PADDING, maxZoom: MAX_FIT_ZOOM }}
        minZoom={0.25}
        maxZoom={2.5}
        nodesDraggable={false}
        nodesConnectable={false}
        elementsSelectable={false}
        zoomOnDoubleClick={false}
        proOptions={{ hideAttribution: true }}
        onNodeClick={(_, n) =>
          p.onSelect(n.data.vn.ir_node === p.selected ? undefined : n.data.vn.ir_node)
        }
        onNodeDoubleClick={(_, n) => {
          if (n.data.drillable) p.onDrill(n.data.vn.ir_node);
        }}
        onPaneClick={() => p.onSelect(undefined)}
      >
        <Background variant={BackgroundVariant.Dots} gap={16} size={1} />
        <Controls showInteractive={false} position="bottom-right" />
        <FitOnChange
          viewKey={p.graph.id}
          bounds={bounds}
          minZoom={
            nodes.length <= SMALL_VIEW_NODES
              ? Math.min(SMALL_VIEW_MIN_ZOOM, p.minFitZoom ?? READABLE_ZOOM)
              : (p.minFitZoom ?? READABLE_ZOOM)
          }
          focus={focus}
        />
      </ReactFlow>
    </div>
  );
}

export function GraphCanvas(p: GraphCanvasProps) {
  return (
    <ReactFlowProvider>
      <Canvas {...p} />
    </ReactFlowProvider>
  );
}
