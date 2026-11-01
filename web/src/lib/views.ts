import type { GraphView, IR, IRNode, LayerGroup, View } from "@inferatlas/contracts";

export const L0 = "L0";

export function getView(view: View | undefined, id: string): GraphView | undefined {
  return view?.views.find((v) => v.id === id);
}

export function irNode(ir: IR, id: string): IRNode | undefined {
  return ir.nodes.find((n) => n.id === id);
}

export function bandForTemplate(ir: IR, nodeId: string): LayerGroup | undefined {
  return ir.groups.find((g) => g.template === nodeId);
}

/** View reached by drilling into an IR node, if any. */
export function drillTarget(ir: IR, view: View | undefined, nodeId: string): string | undefined {
  const band = bandForTemplate(ir, nodeId);
  if (band && getView(view, `L1:${band.id}`)) return `L1:${band.id}`;
  if (getView(view, `L2:${nodeId}`)) return `L2:${nodeId}`;
  return undefined;
}

function ancestors(ir: IR, id: string): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  let cur = irNode(ir, id)?.parent ?? null;
  while (cur && !seen.has(cur)) {
    seen.add(cur);
    out.push(cur);
    cur = irNode(ir, cur)?.parent ?? null;
  }
  return out;
}

/** One level up: L2 → its band's L1 (or an enclosing L2) → L0. */
export function parentView(ir: IR, view: View | undefined, viewId: string): string | undefined {
  if (viewId === L0) return undefined;
  if (viewId.startsWith("L1:")) return L0;
  if (viewId.startsWith("L2:")) {
    const id = viewId.slice(3);
    for (const a of ancestors(ir, id)) {
      const band = bandForTemplate(ir, a);
      if (band && getView(view, `L1:${band.id}`)) return `L1:${band.id}`;
      if (getView(view, `L2:${a}`)) return `L2:${a}`;
    }
    return L0;
  }
  return L0;
}

export function breadcrumb(ir: IR, view: View | undefined, viewId: string): string[] {
  const chain = [viewId];
  let cur: string | undefined = viewId;
  for (let i = 0; i < 8; i++) {
    cur = parentView(ir, view, cur);
    if (!cur) break;
    chain.unshift(cur);
  }
  return chain;
}

export function viewTitle(ir: IR, viewId: string): string {
  if (viewId === L0) return "Model";
  if (viewId.startsWith("L1:")) {
    const band = ir.groups.find((g) => g.id === viewId.slice(3));
    const tpl = band ? irNode(ir, band.template) : undefined;
    return tpl ? tpl.label.replace(/\s*×\s*\w+$/, "") : viewId.slice(3);
  }
  if (viewId.startsWith("L2:")) return irNode(ir, viewId.slice(3))?.label ?? viewId.slice(3);
  return viewId;
}

/**
 * The view that shows an IR node: a view containing it directly (deepest first), otherwise
 * the view containing its nearest visible ancestor.
 */
export function viewForNode(
  ir: IR,
  view: View | undefined,
  nodeId: string,
): { viewId: string; visibleNode: string } | undefined {
  if (!view) return undefined;
  const rank = (id: string) => (id.startsWith("L2:") ? 0 : id.startsWith("L1:") ? 1 : 2);
  const sorted = [...view.views].sort((a, b) => rank(a.id) - rank(b.id));
  for (const target of [nodeId, ...ancestors(ir, nodeId)]) {
    for (const v of sorted) {
      if (v.nodes.some((n) => n.ir_node === target)) return { viewId: v.id, visibleNode: target };
    }
  }
  return undefined;
}

/** All IR descendants of a node (for overlay roll-ups on collapsed groups). */
export function descendants(ir: IR, id: string): IRNode[] {
  const children = new Map<string, IRNode[]>();
  for (const n of ir.nodes) {
    if (!n.parent) continue;
    const list = children.get(n.parent) ?? [];
    list.push(n);
    children.set(n.parent, list);
  }
  const out: IRNode[] = [];
  const stack = [...(children.get(id) ?? [])];
  while (stack.length) {
    const n = stack.pop() as IRNode;
    out.push(n);
    stack.push(...(children.get(n.id) ?? []));
  }
  return out;
}

/** Parallel spec of the node or its nearest ancestor that declares one. */
export function inheritedParallel(
  ir: IR,
  n: IRNode,
): { from: string; spec: NonNullable<IRNode["parallel"]> } | undefined {
  let cur: IRNode | undefined = n;
  for (let i = 0; i < 16 && cur; i++) {
    if (cur.parallel && Object.keys(cur.parallel).length)
      return { from: cur.id, spec: cur.parallel };
    cur = cur.parent ? irNode(ir, cur.parent) : undefined;
  }
  return undefined;
}

/** Layer bands ordered by their first layer (expression-valued, evaluated by the caller). */
export function sortedBands<T extends { g: LayerGroup; lo: number | undefined }>(bands: T[]): T[] {
  return [...bands].sort(
    (a, b) =>
      (a.lo ?? Number.POSITIVE_INFINITY) - (b.lo ?? Number.POSITIVE_INFINITY) ||
      a.g.id.localeCompare(b.g.id),
  );
}

/**
 * The L0 node that stands for a band. Layout folds consecutive bands into one decoder-stack
 * anchor, so a band's own template may not be on L0; fall back to any band template there.
 */
export function bandAnchor(ir: IR, view: View | undefined, bandId: string): string | undefined {
  const l0 = getView(view, L0);
  if (!l0) return undefined;
  const visible = new Set(l0.nodes.map((n) => n.ir_node));
  const own = ir.groups.find((g) => g.id === bandId)?.template;
  if (own && visible.has(own)) return own;
  return ir.groups.map((g) => g.template).find((t) => visible.has(t));
}
