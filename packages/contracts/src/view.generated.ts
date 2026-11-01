/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * IR node ids represented by this collapsed view, sorted.
 */
export type CollapsedOf = string[];
/**
 * Orthogonal edges, stable id order; complexity budget <=12.
 *
 * @maxItems 12
 */
export type Edges =
  | []
  | [ViewEdge]
  | [ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge, ViewEdge]
  | [
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge
    ]
  | [
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge
    ]
  | [
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge,
      ViewEdge
    ];
/**
 * Producer ViewNode.id, not an IR node id.
 */
export type From = string;
/**
 * View-local unique edge id.
 */
export type Id = string;
/**
 * IR edge ids represented by this route, sorted; may span collapsed nodes.
 *
 * @minItems 1
 */
export type IrEdges = [string, ...string[]];
/**
 * Semantic kind shared by the represented IR edges.
 */
export type Kind = "data" | "kv" | "residual";
/**
 * Optional symbolic shape copied from a represented IR edge tensor.shape.
 */
export type Label = string[] | PhaseShape | null;
/**
 * Local-rank decode dimensions; no T substitution needed.
 */
export type Decode = string[];
/**
 * Local-rank prefill dimensions.
 */
export type Prefill = string[];
/**
 * Optional precomputed edge-label center in pixels, on the 4px grid.
 */
export type LabelPos = [number, number] | null;
/**
 * Ordered source-to-target orthogonal polyline, including endpoints; all coordinates on 4px grid.
 *
 * @minItems 2
 */
export type Points = [[number, number], [number, number], ...[number, number][]];
/**
 * Consumer ViewNode.id.
 */
export type To = string;
/**
 * View canvas height in pixels.
 */
export type Height = number;
/**
 * L0 overview, L1:<band>, or L2:<node>; bare L1 is invalid.
 */
export type Id1 = string;
/**
 * Curated visible nodes, stable id order; complexity budget <=9.
 *
 * @maxItems 9
 */
export type Nodes =
  | []
  | [ViewNode]
  | [ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode]
  | [ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode, ViewNode];
/**
 * Height in pixels on a 4px grid.
 */
export type H = number;
/**
 * View-local unique node id.
 */
export type Id2 = string;
/**
 * Referenced IRNode.id; multiple view instances may reference one template.
 */
export type IrNode = string;
/**
 * Visible label; may specialize a representative layer.
 */
export type Label1 = string;
/**
 * Width in pixels on a 4px grid.
 */
export type W = number;
/**
 * Left coordinate in pixels on a 4px grid.
 */
export type X = number;
/**
 * Top coordinate in pixels on a 4px grid.
 */
export type Y = number;
/**
 * View canvas width in pixels.
 */
export type Width = number;
/**
 * Precomputed drill-down views in stable id order; no browser-time layout.
 */
export type Views = GraphView[];

export interface View {
  views: Views;
}
export interface GraphView {
  collapsed_of?: CollapsedOf;
  edges: Edges;
  height: Height;
  id: Id1;
  nodes: Nodes;
  width: Width;
}
export interface ViewEdge {
  from: From;
  id: Id;
  ir_edges: IrEdges;
  kind: Kind;
  label?: Label;
  label_pos?: LabelPos;
  points: Points;
  to: To;
}
export interface PhaseShape {
  decode: Decode;
  prefill: Prefill;
}
export interface ViewNode {
  h: H;
  id: Id2;
  ir_node: IrNode;
  label: Label1;
  w: W;
  x: X;
  y: Y;
}
