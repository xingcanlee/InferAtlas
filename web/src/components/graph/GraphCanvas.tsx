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
  /** Readable floor: wide views pan instead of shrinking text below this zoom. */
  minFitZoom?: number;
  /** Distinguishes two canvases on the diff page for tests. */
  testId?: string;
}

/** Readable floor: wide views pan horizontally instead of shrinking text below this. */
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
  /** Horizontal center (graph px) to keep in view when the graph is wider than the canvas. */
  focus: number | undefined;
}) {
  const rf = useReactFlow();
  const width = useStore((s) => s.width);
  const height = useStore((s) => s.height);
  // Focus only steers the initial placement of a view; clicking must not move the camera.
  const focusRef = useRef(focus);
  focusRef.current = focus;
  // biome-ignore lint/correctness/useExhaustiveDependencies: viewKey intentionally re-triggers placement
  useEffect(() => {
    const focus = focusRef.current;
    if (!width || !height) return;
    // Re-fit whenever the drill level or canvas size changes; layout itself is precomputed.
    const zoom = Math.min((width - 24) / gw, (height - 16) / gh, 1.1);
    if (zoom >= minZoom) {
      rf.setViewport({ x: (width - gw * zoom) / 2, y: (height - gh * zoom) / 2, zoom });
    } else {
      const z = minZoom;
      const centered = focus === undefined ? 8 : width / 2 - focus * z;
      const x = Math.min(8, Math.max(width - gw * z - 8, centered));
      rf.setViewport({ x, y: (height - gh * z) / 2, zoom: z });
    }
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
  const focusX = focusNode ? focusNode.position.x + (focusNode.width ?? 0) / 2 : undefined;
  return (
    <div className={cn("relative h-full w-full", p.className)} data-testid={p.testId ?? "graph"}>
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
          focus={focusX}
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
