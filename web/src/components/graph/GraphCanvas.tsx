import {
  Background,
  BackgroundVariant,
  Controls,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
  useStore,
} from "@xyflow/react";
import { useEffect, useMemo, useRef } from "react";
import { cn } from "@/lib/utils";
import { type BuildOptions, buildGraph } from "./build";
import { nodeTypes } from "./nodes";
import { ArrowDefs, OrthEdge } from "./OrthEdge";
import type { GraphNode } from "./types";

const edgeTypes = { orth: OrthEdge };

export interface GraphCanvasProps extends BuildOptions {
  onSelect: (irNode: string | undefined) => void;
  onDrill: (irNode: string) => void;
  className?: string;
  /** Readable floor: views pan instead of shrinking text below this zoom. */
  minFitZoom?: number;
  /** Distinguishes two canvases on the diff page for tests. */
  testId?: string;
}

/**
 * Place the precomputed layout: fit when it is readable, otherwise keep a readable zoom and
 * pan (scroll) — wide/tall views never shrink text below `minZoom`.
 */
function FitOnChange({
  viewKey,
  gw,
  gh,
  minZoom,
  focus,
}: {
  viewKey: string;
  gw: number;
  gh: number;
  minZoom: number;
  /** Center (graph px) to keep in view when the graph does not fit. */
  focus: { x: number; y: number } | undefined;
}) {
  const rf = useReactFlow();
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  // Focus only steers the initial placement of a view; clicking must not move the camera.
  const focusRef = useRef(focus);
  focusRef.current = focus;
  // biome-ignore lint/correctness/useExhaustiveDependencies: viewKey intentionally re-triggers placement
  useEffect(() => {
    if (!width || !height) return;
    const f = focusRef.current;
    const z = Math.max(minZoom, Math.min((width - 24) / gw, (height - 24) / gh, 1.1));
    const place = (avail: number, size: number, at: number | undefined) => {
      if (size * z <= avail) return (avail - size * z) / 2;
      const want = at === undefined ? 8 : avail / 2 - at * z;
      return Math.min(8, Math.max(avail - size * z - 8, want));
    };
    rf.setViewport({ x: place(width, gw, f?.x), y: place(height, gh, f?.y), zoom: z });
  }, [rf, viewKey, width, height, gw, gh, minZoom]);
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
  // Keep the selected (or first changed) node in view on wide graphs.
  const focusNode =
    nodes.find((n) => n.data.selected) ??
    nodes.find((n) => n.data.diff && n.data.diff !== "removed");
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
          gw={p.graph.width}
          gh={p.graph.height}
          minZoom={p.minFitZoom ?? 0.9}
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
