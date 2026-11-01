import { createRequire } from "node:module";
import type { GraphView, IR, View, ViewEdge } from "@inferatlas/contracts";
import type { ELK as ElkInstance, ElkNode } from "elkjs/lib/elk-api.js";
import { assertBudget } from "./budget.js";
import { compare, type Template, templates } from "./templates.js";
import { ceilGrid, nodeSize, shapeText, snap, textWidth } from "./text.js";
import { validateIR, validateView } from "./validation.js";

export { assertBudget, BUDGET, checkBudget } from "./budget.js";
export { layerBandTargets } from "./templates.js";
export { nodeSize, shapeText, textWidth } from "./text.js";
export { validateIR, validateView } from "./validation.js";

// ELK's CommonJS bundle has incompatible NodeNext default-export declarations.
const ELK = createRequire(import.meta.url)("elkjs/lib/elk.bundled.js");

export type Point = [number, number];
export interface LayoutOptions {
  onWarning?: (message: string) => void;
}

export const ELK_OPTIONS = Object.freeze({
  "elk.algorithm": "layered",
  "elk.edgeRouting": "ORTHOGONAL",
  "elk.randomSeed": "1",
  "elk.padding": "[top=24,left=24,bottom=24,right=24]",
  "elk.spacing.nodeNode": "48",
  "elk.spacing.edgeNode": "24",
  "elk.spacing.edgeEdge": "16",
  "elk.layered.spacing.nodeNodeBetweenLayers": "48",
  "elk.layered.spacing.edgeNodeBetweenLayers": "24",
  "elk.layered.considerModelOrder.strategy": "NODES_AND_EDGES",
  "elk.layered.nodePlacement.strategy": "BRANDES_KOEPF",
  "elk.layered.nodePlacement.favorStraightEdges": "true",
  "elk.layered.nodePlacement.bk.fixedAlignment": "BALANCED",
  "elk.layered.wrapping.strategy": "MULTI_EDGE",
  "elk.aspectRatio": "1.5",
});

async function place(template: Template, wrapping = true): Promise<GraphView> {
  const elk: ElkInstance = new ELK();
  const placed = await elk.layout<ElkNode>({
    id: template.id,
    layoutOptions: {
      ...ELK_OPTIONS,
      "elk.direction": template.direction,
      "elk.layered.wrapping.strategy": wrapping ? "MULTI_EDGE" : "OFF",
    },
    children: template.nodes.map(({ node, label }) => {
      const { w, h } = nodeSize(label);
      return { id: node.id, width: w, height: h };
    }),
    edges: template.edges.map((edge) => {
      const text = edge.label ? shapeText(edge.label) : "";
      return {
        id: edge.id,
        sources: [edge.from],
        targets: [edge.to],
        labels: text
          ? [
              {
                text,
                layoutOptions: { "elk.edgeLabels.inline": "true" },
                width: ceilGrid(textWidth(text, "mono") + 16),
                height: text.split("\n").length * 20,
              },
            ]
          : [],
      };
    }),
  });
  const positions = new Map(placed.children?.map((node) => [node.id, node]));
  const routes = new Map(placed.edges?.map((edge) => [edge.id, edge]));
  const nodes = template.nodes.map(({ node, label }) => {
    const position = positions.get(node.id);
    if (position?.x == null || position.y == null) throw new Error(`ELK omitted node ${node.id}`);
    return {
      id: node.id,
      ir_node: node.id,
      label,
      x: snap(position.x),
      y: snap(position.y),
      ...nodeSize(label),
    };
  });
  const edges = template.edges.map((edge): ViewEdge => {
    const route = routes.get(edge.id);
    const section = route?.sections?.[0];
    if (!section || route?.sections?.length !== 1)
      throw new Error(`ELK omitted or split route ${edge.id}`);
    const points = [section.startPoint, ...(section.bendPoints ?? []), section.endPoint].map(
      ({ x, y }): Point => [snap(x), snap(y)],
    );
    const label = route.labels?.[0];
    // The snapped ELK label center reserves space for the opaque mono label in the renderer.
    const label_pos: Point =
      label?.x != null && label.y != null
        ? [snap(label.x + (label.width ?? 0) / 2), snap(label.y + (label.height ?? 0) / 2)]
        : (points[Math.floor(points.length / 2)] as Point);
    return { ...edge, points: points as [Point, Point, ...Point[]], label_pos };
  });
  const graph: GraphView = {
    id: template.id,
    nodes: nodes as GraphView["nodes"],
    edges: edges as GraphView["edges"],
    collapsed_of: template.collapsed_of,
    width: Math.max(48, ceilGrid(placed.width ?? 0)),
    height: Math.max(48, ceilGrid(placed.height ?? 0)),
  };
  try {
    validateView({ views: [graph] });
  } catch (error) {
    // ELK wrapping can leave tiny/diagonal routes or endpoints detached from their nodes.
    if (wrapping) return place(template, false);
    throw error;
  }
  return graph;
}

/** Validates input, curates family views, then runs deterministic build-time ELK layout. */
export async function layoutIR(input: unknown, options: LayoutOptions = {}): Promise<View> {
  validateIR(input);
  const warn = options.onWarning ?? ((message: string) => console.warn(message));
  const views: GraphView[] = [];
  for (const template of templates(input, warn)) views.push(await place(template));
  for (const view of views) assertBudget(view);
  const result = { views };
  validateView(result);
  return result;
}

/** Stable recursive key ordering without changing semantically ordered arrays. */
export function serializeView(view: View): string {
  return `${JSON.stringify(
    view,
    (_key, value: unknown) => {
      if (value !== null && typeof value === "object" && !Array.isArray(value)) {
        return Object.fromEntries(Object.entries(value).sort(([a], [b]) => compare(a, b)));
      }
      return value;
    },
    2,
  )}\n`;
}

/** Local rank-boundary lane segments, centered on each collective (no physical rank claims). */
export function rankBoundaryLanes(
  view: GraphView,
  ir: IR,
  direction = templates(ir, () => {}).find((template) => template.id === view.id)?.direction,
): { ir_node: string; points: [Point, Point] }[] {
  const comm = new Set(ir.nodes.filter((node) => node.kind === "comm").map((node) => node.id));
  return view.nodes
    .filter((node) => comm.has(node.ir_node))
    .map((node) => ({
      ir_node: node.ir_node,
      points:
        direction === "RIGHT"
          ? [
              [snap(node.x + node.w / 2), node.y - 16],
              [snap(node.x + node.w / 2), node.y + node.h + 16],
            ]
          : [
              [node.x - 16, snap(node.y + node.h / 2)],
              [node.x + node.w + 16, snap(node.y + node.h / 2)],
            ],
    }));
}
