import type { GraphView, IR } from "@inferatlas/contracts";

export type Mark = "added" | "removed" | "changed";

export interface FieldChange {
  path: string;
  a: string | undefined;
  b: string | undefined;
}

export interface ItemChange {
  id: string;
  mark: Mark;
  fields: FieldChange[];
}

export interface IrDiff {
  nodes: Map<string, ItemChange>;
  edges: Map<string, ItemChange>;
  /** symbols / memory / params changes as pseudo-items. */
  model: FieldChange[];
}

/** Provenance that moves with every commit; differences there are not semantic changes. */
const IGNORED = /(^|\/)(commit|line)$/;

export function flatten(
  value: unknown,
  prefix = "",
  out: Record<string, string> = {},
): Record<string, string> {
  if (value === null || value === undefined) return out;
  if (Array.isArray(value)) {
    if (value.every((v) => typeof v !== "object" || v === null)) {
      out[prefix] = `[${value.map(String).join(", ")}]`;
      return out;
    }
    value.forEach((v, i) => {
      // name-keyed arrays (tensor slots) diff by name, not position
      const key =
        v && typeof v === "object" && "name" in v
          ? String((v as { name: unknown }).name)
          : String(i);
      flatten(v, prefix ? `${prefix}/${key}` : key, out);
    });
    return out;
  }
  if (typeof value === "object") {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      flatten(v, prefix ? `${prefix}/${k}` : k, out);
    }
    return out;
  }
  if (!IGNORED.test(prefix)) out[prefix] = String(value);
  return out;
}

export function diffFields(a: unknown, b: unknown): FieldChange[] {
  const fa = flatten(a);
  const fb = flatten(b);
  const keys = [...new Set([...Object.keys(fa), ...Object.keys(fb)])].sort();
  const out: FieldChange[] = [];
  for (const k of keys) if (fa[k] !== fb[k]) out.push({ path: k, a: fa[k], b: fb[k] });
  return out;
}

function diffById<T extends { id: string }>(a: T[], b: T[]): Map<string, ItemChange> {
  const out = new Map<string, ItemChange>();
  const am = new Map(a.map((x) => [x.id, x]));
  const bm = new Map(b.map((x) => [x.id, x]));
  for (const [id, x] of am) {
    const y = bm.get(id);
    if (!y) out.set(id, { id, mark: "removed", fields: diffFields(x, undefined) });
    else {
      const fields = diffFields(x, y);
      if (fields.length) out.set(id, { id, mark: "changed", fields });
    }
  }
  for (const [id, y] of bm)
    if (!am.has(id)) out.set(id, { id, mark: "added", fields: diffFields(undefined, y) });
  return out;
}

export function diffIR(a: IR, b: IR): IrDiff {
  return {
    nodes: diffById(a.nodes, b.nodes),
    edges: diffById(a.edges, b.edges),
    model: [
      ...diffFields({ symbols: a.symbols }, { symbols: b.symbols }),
      ...diffFields({ memory: a.memory }, { memory: b.memory }),
      ...diffFields({ params: a.params }, { params: b.params }),
    ],
  };
}

/** Marks to draw on one side: the old side shows removals, the new side additions. */
export function sideMarks(
  diff: IrDiff,
  graph: GraphView | undefined,
  side: "a" | "b",
): { nodes: Map<string, Mark>; edges: Set<string> } {
  const nodes = new Map<string, Mark>();
  const edges = new Set<string>();
  if (!graph) return { nodes, edges };
  const hide: Mark = side === "a" ? "added" : "removed";
  for (const vn of graph.nodes) {
    const c = diff.nodes.get(vn.ir_node);
    if (c && c.mark !== hide) nodes.set(vn.ir_node, c.mark);
  }
  for (const e of graph.edges) {
    if (
      e.ir_edges.some((id) => {
        const c = diff.edges.get(id);
        return c && c.mark !== hide;
      })
    ) {
      edges.add(e.id);
    }
  }
  return { nodes, edges };
}

export function viewChangeCount(
  diff: IrDiff,
  ga: GraphView | undefined,
  gb: GraphView | undefined,
): number {
  const ids = new Set<string>();
  for (const g of [ga, gb])
    for (const vn of g?.nodes ?? []) if (diff.nodes.has(vn.ir_node)) ids.add(vn.ir_node);
  return ids.size;
}
