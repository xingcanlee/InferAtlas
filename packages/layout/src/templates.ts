import type { IR, IRNode, ViewEdge } from "@inferatlas/contracts";
import { assertBudget } from "./budget.js";

export type TemplateEdge = Omit<ViewEdge, "points" | "label_pos">;
export interface Template {
  id: string;
  direction: "DOWN" | "RIGHT";
  nodes: { node: IRNode; label: string }[];
  edges: TemplateEdge[];
  collapsed_of: string[];
}

export const compare = (a: string, b: string): number => (a < b ? -1 : a > b ? 1 : 0);
const byId = <T extends { id: string }>(items: readonly T[]): T[] =>
  [...items].sort((a, b) => compare(a.id, b.id));

export function templates(ir: IR, warn: (message: string) => void): Template[] {
  const nodes = byId(ir.nodes);
  const lookup = new Map(nodes.map((node) => [node.id, node]));
  const edges = byId(ir.edges);
  const children = (id: string) => nodes.filter((node) => node.parent === id);
  const descendants = (id: string): IRNode[] =>
    children(id).flatMap((node) => [node, ...descendants(node.id)]);

  function project(
    id: string,
    scope: IRNode[],
    visible: IRNode[],
    labels = new Map<string, string>(),
  ): Template {
    const selected = new Set(visible.map((node) => node.id));
    const membership = new Set(scope.map((node) => node.id));
    const representative = new Map<string, string>();
    for (const node of scope) {
      let ancestor: IRNode | undefined = node;
      while (ancestor && membership.has(ancestor.id)) {
        if (selected.has(ancestor.id)) {
          representative.set(node.id, ancestor.id);
          break;
        }
        ancestor = ancestor.parent ? lookup.get(ancestor.parent) : undefined;
      }
    }
    const relevant = edges.filter((edge) => membership.has(edge.from) && membership.has(edge.to));
    const routes = new Map<string, TemplateEdge>();
    for (const edge of relevant) {
      const from = representative.get(edge.from);
      const to = representative.get(edge.to);
      if (!from || !to || (from === to && !(edge.kind === "kv" && selected.has(edge.from))))
        continue;
      const key = JSON.stringify([from, to, edge.kind, edge.tensor.shape]);
      let route = routes.get(key);
      if (!route) {
        route = {
          id: edge.id,
          from,
          to,
          kind: edge.kind,
          label: edge.tensor.shape,
          ir_edges: [edge.id],
        };
        routes.set(key, route);
      }
      const represented = new Set(route.ir_edges);
      represented.add(edge.id);
      // Walk only same-kind internal routes; a collapsed collective stays in the provenance.
      function collect(
        node: string,
        owner: string,
        backwards: boolean,
        visited = new Set<string>(),
      ) {
        if (visited.has(node)) return;
        visited.add(node);
        for (const internal of relevant) {
          if (
            internal.kind !== edge.kind ||
            representative.get(internal.from) !== owner ||
            representative.get(internal.to) !== owner
          )
            continue;
          if ((backwards ? internal.to : internal.from) !== node) continue;
          represented.add(internal.id);
          collect(backwards ? internal.from : internal.to, owner, backwards, visited);
        }
      }
      collect(edge.from, from, true);
      collect(edge.to, to, false);
      route.ir_edges = [...represented].sort(compare) as [string, ...string[]];
    }
    const result: Template = {
      id,
      direction: id === "L0" ? "DOWN" : "RIGHT",
      nodes: byId(visible).map((node) => ({
        node,
        label: labels.get(node.id) ?? node.label,
      })),
      edges: byId([...routes.values()]),
      collapsed_of: scope.map((node) => node.id).sort(compare),
    };
    assertBudget(result);
    return result;
  }

  const supported = ir.views.family === "dense" || ir.views.family === "moe";
  if (!supported)
    warn(
      `Family ${ir.views.family}: generic template fallback; family-specific merges are not implemented.`,
    );
  const labels = new Map<string, string>();
  for (const band of ir.groups) {
    labels.set(
      band.template,
      `${lookup.get(band.template)?.label}\nLayers ${band.layers[0]}…${band.layers[1]}`,
    );
  }
  const roots = nodes.filter((node) => node.parent == null);
  const result = [project("L0", nodes, roots, labels)];
  for (const band of byId(ir.groups)) {
    const scope = descendants(band.template);
    if (scope.length === 0) {
      warn(
        `Band ${band.id}: no decoder internals in IR; attention and MLP/MoE drill-downs are unavailable.`,
      );
      result.push(
        project(
          `L1:${band.id}`,
          [lookup.get(band.template) as IRNode],
          [lookup.get(band.template) as IRNode],
          labels,
        ),
      );
      continue;
    }
    result.push(project(`L1:${band.id}`, scope, children(band.template)));
    if (!supported) continue;
    for (const group of children(band.template)) {
      const role = group.id.split(".").at(-1);
      const isAttention = role === "attn" || /Attention$/.test(group.class);
      const isMlp = role === "mlp" || role === "moe" || /(?:MLP|MoeBlock|MoE)$/.test(group.class);
      if (!isAttention && !isMlp) continue;
      const internals = descendants(group.id);
      if (!internals.length) {
        warn(`Node ${group.id}: no internals in IR; drill-down is unavailable.`);
        continue;
      }
      const candidates = isAttention
        ? children(group.id)
        : internals.filter(
            (node) =>
              node.kind === "comm" ||
              /\.(?:gate|topk|gate_up|act|down|w13|w2|shared_expert)$/.test(node.id) ||
              /(?:GateLinear|MergedColumnParallelLinear|RowParallelLinear|SiluAndMul)$/.test(
                node.class,
              ),
          );
      const summaries = new Set(
        candidates.filter((node) => node.kind === "group").map((node) => node.id),
      );
      const visible = candidates.filter((node) => {
        let parent = node.parent;
        while (parent && parent !== group.id) {
          if (summaries.has(parent)) return false;
          parent = lookup.get(parent)?.parent;
        }
        return true;
      });
      if (!visible.length) {
        warn(`Node ${group.id}: no curated MLP/MoE operations matched; using generic children.`);
      }
      result.push(
        project(`L2:${group.id}`, internals, visible.length ? visible : children(group.id)),
      );
    }
  }
  return byId(result);
}
