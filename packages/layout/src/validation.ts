import { readFileSync } from "node:fs";
import type { IR, View } from "@inferatlas/contracts";
import { Ajv2020 } from "ajv/dist/2020.js";

const ajv = new Ajv2020({ allErrors: true, strictTuples: false });
const validators = Object.fromEntries(
  ["ir", "view"].map((name) => [
    name,
    ajv.compile(
      JSON.parse(
        readFileSync(new URL(`../../../schemas/${name}.schema.json`, import.meta.url), "utf8"),
      ),
    ),
  ]),
);

export function validateIR(value: unknown): asserts value is IR {
  const validate = validators.ir;
  if (!validate || !validate(value)) {
    throw new Error(`Invalid IR: ${ajv.errorsText(validate?.errors)}`);
  }
  const ir = value as IR;
  const nodes = new Map(ir.nodes.map((node) => [node.id, node]));
  if (nodes.size !== ir.nodes.length) throw new Error("Duplicate IR node id");
  if (new Set(ir.edges.map((edge) => edge.id)).size !== ir.edges.length) {
    throw new Error("Duplicate IR edge id");
  }
  if (new Set(ir.groups.map((group) => group.id)).size !== ir.groups.length) {
    throw new Error("Duplicate layer band id");
  }
  for (const node of ir.nodes) {
    const seen = new Set([node.id]);
    let parent = node.parent;
    while (parent != null) {
      const ancestor = nodes.get(parent);
      if (!ancestor || seen.has(parent)) throw new Error(`Invalid IR parent hierarchy: ${node.id}`);
      seen.add(parent);
      parent = ancestor.parent;
    }
    if (node.kind === "comm" && !node.comm) throw new Error(`Missing comm metadata: ${node.id}`);
  }
  for (const edge of ir.edges) {
    if (!nodes.has(edge.from) || !nodes.has(edge.to))
      throw new Error(`Missing IR edge endpoint: ${edge.id}`);
  }
  for (const group of ir.groups) {
    if (!nodes.has(group.template)) throw new Error(`Missing band template: ${group.id}`);
    if (!/^[^\s:]+$/.test(group.id)) throw new Error(`Invalid band id: ${group.id}`);
  }
}

export function validateView(value: unknown): asserts value is View {
  const validate = validators.view;
  if (!validate || !validate(value)) {
    throw new Error(`Invalid view: ${ajv.errorsText(validate?.errors)}`);
  }
  const view = value as View;
  if (new Set(view.views.map((graph) => graph.id)).size !== view.views.length) {
    throw new Error("Duplicate view id");
  }
  for (const graph of view.views) {
    const ids = new Set(graph.nodes.map((node) => node.id));
    if (
      ids.size !== graph.nodes.length ||
      new Set(graph.edges.map((edge) => edge.id)).size !== graph.edges.length
    ) {
      throw new Error(`Duplicate view node/edge id: ${graph.id}`);
    }
    for (const node of graph.nodes) {
      if (node.x + node.w > graph.width || node.y + node.h > graph.height) {
        throw new Error(`Node outside canvas: ${node.id}`);
      }
    }
    for (const edge of graph.edges) {
      if (!ids.has(edge.from) || !ids.has(edge.to))
        throw new Error(`Missing view endpoint: ${edge.id}`);
      for (let i = 0; i < edge.points.length; i++) {
        const [x, y] = edge.points[i] as [number, number];
        const previous = edge.points[i - 1] as [number, number] | undefined;
        if (x > graph.width || y > graph.height) throw new Error(`Edge outside canvas: ${edge.id}`);
        if (previous && previous[0] !== x && previous[1] !== y)
          throw new Error(`Diagonal edge: ${edge.id}`);
      }
    }
  }
}
