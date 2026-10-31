import type { IRNode, ViewNode } from "@inferatlas/contracts";
import type { Edge, Node } from "@xyflow/react";
import type { NodeMetrics } from "@/lib/model";

export type DiffMark = "added" | "removed" | "changed";

export interface GraphNodeData extends Record<string, unknown> {
  vn: ViewNode;
  ir: IRNode | undefined;
  metrics: NodeMetrics | undefined;
  /** Share of the view's total for the heat metric, 0..1; undefined hides the bar. */
  share: number | undefined;
  heatText: string;
  selected: boolean;
  related: boolean;
  overlay: boolean;
  /** Ascend-tagged descendants hidden inside a collapsed node. */
  hiddenTagged: number;
  drillable: boolean;
  /** "×48" style repetition for a layer-band template. */
  repeat: string | undefined;
  /** Symbolic or numeric KV tensor feeding this node (KV flow stub). */
  kv: string | undefined;
  /** Rank-group degree label for comm nodes ("tp×4"). */
  lane: string | undefined;
  diff: DiffMark | undefined;
}

export type GraphNode = Node<GraphNodeData>;

export interface GraphEdgeData extends Record<string, unknown> {
  points: [number, number][];
  kind: "data" | "kv" | "residual";
  label: string | undefined;
  labelPos: [number, number] | undefined;
  focus: boolean;
  changed: boolean;
}

export type GraphEdge = Edge<GraphEdgeData>;
