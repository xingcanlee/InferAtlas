import type { ExprEnv, GraphView, IR, View } from "@inferatlas/contracts";
import { tryEvaluate } from "@/lib/expr";
import { fmtBytes, fmtDim, fmtFlops, pct } from "@/lib/format";
import { evalShape, type NodeMetrics, type Phase, type Shape, shapeDims } from "@/lib/model";
import type { Heat } from "@/lib/search";
import { bandForTemplate, descendants, drillTarget } from "@/lib/views";
import type { DiffMark, GraphEdge, GraphNode } from "./types";

export interface BuildOptions {
  ir: IR;
  view: View | undefined;
  graph: GraphView;
  metrics: Map<string, NodeMetrics>;
  env: ExprEnv;
  phase: Phase;
  shapes: "sym" | "num";
  heat: Heat;
  overlay: boolean;
  selected: string | undefined;
  nodeDiff?: Map<string, DiffMark>;
  changedEdges?: Set<string>;
}

export function shapeText(shape: Shape, env: ExprEnv, phase: Phase, mode: "sym" | "num"): string {
  if (mode === "sym") return `[${shapeDims(shape, phase).join(", ")}]`;
  return `[${evalShape(shape, env, phase)
    .map((d) => (d.v === undefined ? "?" : fmtDim(d.v)))
    .join(", ")}]`;
}

function heatValue(m: NodeMetrics | undefined, heat: Heat): number | undefined {
  if (!m) return undefined;
  const q = heat === "flops" ? m.flops : heat === "weight" ? m.weight : m.bytes;
  return q.v;
}

export function heatLabel(heat: Heat, v: number | undefined): string {
  if (v === undefined) return "—";
  return heat === "flops" ? fmtFlops(v) : fmtBytes(v);
}

export function buildGraph(o: BuildOptions): { nodes: GraphNode[]; edges: GraphEdge[] } {
  const { ir, graph, metrics, env, phase, heat } = o;
  const byId = new Map(ir.nodes.map((n) => [n.id, n]));

  let total = 0;
  for (const vn of graph.nodes) {
    const n = byId.get(vn.ir_node);
    if (!n || n.kind === "comm") continue;
    total += heatValue(metrics.get(vn.ir_node), heat) ?? 0;
  }

  // Edges incident to the selected node form the highlighted data path.
  const related = new Set<string>();
  const focusEdges = new Set<string>();
  for (const e of graph.edges) {
    const from = graph.nodes.find((n) => n.id === e.from)?.ir_node;
    const to = graph.nodes.find((n) => n.id === e.to)?.ir_node;
    if (o.selected && (from === o.selected || to === o.selected)) {
      focusEdges.add(e.id);
      if (from) related.add(from);
      if (to) related.add(to);
    }
  }

  // Node-level KV feeds (IR kv edges) shown as a dashed stub on the visible consumer.
  const kvFor = new Map<string, string>();
  for (const e of ir.edges) {
    if (e.kind !== "kv") continue;
    const label = shapeText(e.tensor.shape, env, phase, o.shapes);
    kvFor.set(e.to, label);
  }

  const nodes: GraphNode[] = graph.nodes.map((vn) => {
    const n = byId.get(vn.ir_node);
    const m = metrics.get(vn.ir_node);
    const value = n?.kind === "comm" ? undefined : heatValue(m, heat);
    const desc = n ? descendants(ir, n.id) : [];
    const hiddenTagged = desc.filter((d) => (d.ascend?.tag?.length ?? 0) > 0).length;
    const band = n ? bandForTemplate(ir, n.id) : undefined;
    let repeat: string | undefined;
    if (band) {
      const [a, b] = band.layers as [string, string];
      const lo = tryEvaluate(String(a), env);
      const hi = tryEvaluate(String(b), env);
      if (lo.ok && hi.ok) repeat = `×${hi.value - lo.value + 1}`;
    }
    let kv = kvFor.get(vn.ir_node);
    if (!kv && n) {
      for (const d of desc) {
        if (kvFor.has(d.id)) {
          kv = kvFor.get(d.id);
          break;
        }
      }
    }
    const group = n?.comm?.group;
    const degree = group && typeof env[group] === "number" ? (env[group] as number) : undefined;
    const type = n?.kind === "comm" ? "comm" : n?.kind === "io" ? "io" : "compute";
    return {
      id: vn.id,
      type,
      position: { x: vn.x, y: vn.y },
      width: vn.w,
      height: vn.h,
      draggable: false,
      connectable: false,
      data: {
        vn,
        ir: n,
        metrics: m,
        share: value !== undefined && total > 0 && n?.kind !== "io" ? value / total : undefined,
        heatText: value !== undefined && total > 0 ? pct(value / total) : "",
        selected: o.selected === vn.ir_node,
        related: related.has(vn.ir_node) && o.selected !== vn.ir_node,
        overlay: o.overlay,
        hiddenTagged,
        drillable: n ? drillTarget(ir, o.view, n.id) !== undefined : false,
        repeat,
        kv,
        lane: group ? `${group}×${degree ?? "?"}` : undefined,
        diff: o.nodeDiff?.get(vn.ir_node),
      },
    } satisfies GraphNode;
  });

  const edges: GraphEdge[] = graph.edges.map((e) => {
    const label = e.label ? shapeText(e.label, env, phase, o.shapes) : undefined;
    return {
      id: e.id,
      source: e.from,
      target: e.to,
      type: "orth",
      selectable: false,
      focusable: false,
      data: {
        points: e.points as [number, number][],
        kind: e.kind,
        label,
        labelPos: placeLabel(
          e.points as [number, number][],
          e.label_pos as [number, number] | null,
          label,
        ),
        focus: focusEdges.has(e.id),
        changed: o.changedEdges?.has(e.id) ?? false,
      },
    };
  });

  return { nodes, edges };
}

/** Approximate mono label width at 10px (JetBrains Mono advance ≈ 0.6em) plus padding. */
const labelWidth = (text: string) => text.length * 6 + 10;

/**
 * Use the precomputed label position when the label fits the segment it sits on; otherwise
 * lift it above the node row so it never covers node text (layout gaps are 48px).
 */
export function placeLabel(
  points: [number, number][],
  pos: [number, number] | null | undefined,
  label: string | undefined,
): [number, number] | undefined {
  if (!label) return undefined;
  let best: { len: number; a: [number, number]; b: [number, number] } | undefined;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1] as [number, number];
    const b = points[i] as [number, number];
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    if (!best || len > best.len) best = { len, a, b };
  }
  if (!best) return pos ?? undefined;
  const horizontal = best.a[1] === best.b[1];
  const [cx, cy] = pos ?? [(best.a[0] + best.b[0]) / 2, (best.a[1] + best.b[1]) / 2];
  if (!horizontal || labelWidth(label) <= best.len - 8) return [cx, cy];
  return [cx, best.a[1] - 50];
}
