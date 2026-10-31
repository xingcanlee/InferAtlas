import { execFileSync, spawnSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { IR, IREdge, IRNode } from "@inferatlas/contracts";
import { describe, expect, it, vi } from "vitest";
import {
  assertBudget,
  checkBudget,
  layoutIR,
  nodeSize,
  rankBoundaryLanes,
  serializeView,
  textWidth,
  validateIR,
  validateView,
} from "../src/index.js";

const root = fileURLToPath(new URL("../../../", import.meta.url));
const fixture = (name: string): IR =>
  JSON.parse(readFileSync(join(root, `fixtures/${name}/ir.json`), "utf8"));
const quiet = { onWarning: () => {} };
const sorted = (items: string[]) => [...items].sort();

describe.each(["qwen3-30b-a3b", "qwen3-8b"])("%s fixture", (name) => {
  it("produces schema-valid, identical bytes with stable ids, provenance, and 4px orthogonal routes", async () => {
    const ir = fixture(name);
    const view = await layoutIR(ir, quiet);
    expect(() => validateView(view)).not.toThrow();
    const bytes = serializeView(view);
    expect(serializeView(await layoutIR(ir, quiet))).toBe(bytes);
    expect(
      serializeView(
        await layoutIR(
          {
            ...ir,
            nodes: [...ir.nodes].reverse(),
            edges: [...ir.edges].reverse(),
            groups: [...ir.groups].reverse(),
          },
          quiet,
        ),
      ),
    ).toBe(bytes);
    await expect(bytes).toMatchFileSnapshot(`__snapshots__/${name}.view.json.snap`);
    const irNodes = new Map(ir.nodes.map((node) => [node.id, node]));
    const irEdges = new Map(ir.edges.map((edge) => [edge.id, edge]));
    for (const graph of view.views) {
      expect(checkBudget(graph)).toEqual([]);
      expect(graph.nodes.map((node) => node.id)).toEqual(
        sorted(graph.nodes.map((node) => node.id)),
      );
      expect(graph.edges.map((edge) => edge.id)).toEqual(
        sorted(graph.edges.map((edge) => edge.id)),
      );
      expect(graph.collapsed_of).toEqual(sorted(graph.collapsed_of ?? []));
      for (const node of graph.nodes) {
        expect(irNodes.has(node.ir_node)).toBe(true);
        for (const quantity of [node.x, node.y, node.w, node.h]) expect(quantity % 4).toBe(0);
      }
      for (const edge of graph.edges) {
        expect(edge.ir_edges).toEqual(sorted(edge.ir_edges));
        const represented = edge.ir_edges.map((id) => irEdges.get(id));
        expect(represented.every((item) => item?.kind === edge.kind)).toBe(true);
        expect(
          represented.some(
            (item) => JSON.stringify(item?.tensor.shape) === JSON.stringify(edge.label),
          ),
        ).toBe(true);
        for (let i = 0; i < edge.points.length; i++) {
          const [x, y] = edge.points[i] as [number, number];
          expect(x % 4).toBe(0);
          expect(y % 4).toBe(0);
          const before = edge.points[i - 1] as [number, number] | undefined;
          if (before) expect(before[0] === x || before[1] === y).toBe(true);
        }
        for (const [id, point] of [
          [edge.from, edge.points[0]],
          [edge.to, edge.points.at(-1)],
        ] as const) {
          const node = graph.nodes.find((node) => node.id === id) as (typeof graph.nodes)[number];
          const [x, y] = point as [number, number];
          expect(x >= node.x && x <= node.x + node.w && y >= node.y && y <= node.y + node.h).toBe(
            true,
          );
          expect(
            x === node.x || x === node.x + node.w || y === node.y || y === node.y + node.h,
          ).toBe(true);
        }
      }
      for (const lane of rankBoundaryLanes(graph, ir)) {
        const node = graph.nodes.find(
          (node) => node.ir_node === lane.ir_node,
        ) as (typeof graph.nodes)[number];
        const [start, end] = lane.points;
        expect(
          start.every((value) => value % 4 === 0) && end.every((value) => value % 4 === 0),
        ).toBe(true);
        if (graph.id === "L0") {
          expect(start[0]).toBeLessThan(node.x);
          expect(end[0]).toBeGreaterThan(node.x + node.w);
        } else {
          expect(start[1]).toBeLessThan(node.y);
          expect(end[1]).toBeGreaterThan(node.y + node.h);
        }
      }
    }
    const overview = view.views[0];
    expect(overview?.nodes.find((node) => node.ir_node === "decoder")?.label).toContain(
      "Layers 0…L-1",
    );
    expect(overview?.collapsed_of).toEqual(sorted(ir.nodes.map((node) => node.id)));
  });
});

function denseIR(): IR {
  const ir = fixture("qwen3-30b-a3b");
  ir.views.family = "dense";
  const mapping: Record<string, string> = {
    moe: "mlp",
    "moe.w13": "mlp.gate_up",
    "moe.act": "mlp.act",
    "moe.w2": "mlp.down",
    "moe.experts.reduce": "mlp.down.reduce",
  };
  ir.nodes = ir.nodes
    .filter((node) => !node.id.startsWith("moe") || mapping[node.id])
    .map((node) => ({
      ...node,
      id: mapping[node.id] ?? node.id,
      parent: mapping[node.id] && node.id !== "moe" ? "mlp" : node.parent,
    }));
  ir.edges = ir.edges
    .filter((edge) => [edge.from, edge.to].every((id) => !id.startsWith("moe") || mapping[id]))
    .map((edge) => ({
      ...edge,
      from: mapping[edge.from] ?? edge.from,
      to: mapping[edge.to] ?? edge.to,
    }));
  const down = ir.edges.find((edge) => edge.to === "mlp.down") as IREdge;
  ir.edges.push({
    ...down,
    id: "mlp.down->mlp.down.reduce:data",
    from: "mlp.down",
    to: "mlp.down.reduce",
  });
  return ir;
}

it("curates full dense attention and MLP internals without fabricating missing fixture nodes", async () => {
  const ir = denseIR();
  const view = await layoutIR(ir, quiet);
  expect(view.views.map((graph) => graph.id)).toEqual(["L0", "L1:decoder", "L2:attn", "L2:mlp"]);
  expect(
    view.views.find((graph) => graph.id === "L2:mlp")?.nodes.map((node) => node.ir_node),
  ).toEqual(["mlp.act", "mlp.down", "mlp.down.reduce", "mlp.gate_up"]);
  const warn = vi.fn();
  await layoutIR(fixture("qwen3-8b"), { onWarning: warn });
  expect(warn).toHaveBeenCalledWith(expect.stringContaining("no decoder internals"));
});

it("keeps KV and residual semantics and records merged Q/K norms, rotary and expert wrappers", async () => {
  const view = await layoutIR(fixture("qwen3-30b-a3b"), quiet);
  const attention = view.views.find((graph) => graph.id === "L2:attn");
  expect(
    attention?.edges.some(
      (edge) => edge.kind === "kv" && edge.from === "attn.core" && edge.to === "attn.core",
    ),
  ).toBe(true);
  expect(attention?.collapsed_of).toEqual(
    expect.arrayContaining(["attn.q_norm", "attn.k_norm", "attn.rope.apply"]),
  );
  expect(attention?.nodes.map((node) => node.ir_node)).not.toContain("attn.q_norm");
  expect(
    view.views
      .find((graph) => graph.id === "L1:decoder")
      ?.edges.some((edge) => edge.kind === "residual"),
  ).toBe(true);
  const moe = view.views.find((graph) => graph.id === "L2:moe");
  expect(moe?.collapsed_of).toContain("moe.experts.method");
  expect(moe?.nodes.map((node) => node.ir_node)).toEqual(
    expect.arrayContaining(["moe.experts.dispatch", "moe.experts.combine", "moe.experts.reduce"]),
  );
});

it("retains intervening communication edges when a module subtree is collapsed", async () => {
  const ir = fixture("qwen3-30b-a3b");
  const reduction = ir.nodes.find((node) => node.id === "embed.reduce") as IRNode;
  reduction.parent = "embed";
  const overview = (await layoutIR(ir, quiet)).views[0];
  expect(overview?.nodes.some((node) => node.ir_node === "embed.reduce")).toBe(false);
  expect(
    overview?.edges.find((edge) => edge.from === "embed" && edge.to === "decoder")?.ir_edges,
  ).toEqual(["embed->embed.reduce:data", "embed.reduce->decoder:data"]);
});

it("collapses a shared expert's implementation into one semantic node", async () => {
  const ir = fixture("qwen3-30b-a3b");
  const expert = ir.nodes.find((node) => node.id === "moe.experts") as IRNode;
  const gemm = ir.nodes.find((node) => node.id === "moe.w13") as IRNode;
  ir.nodes.push(
    { ...expert, id: "moe.shared_expert", parent: "moe" },
    { ...gemm, id: "moe.shared_expert.gate_up", parent: "moe.shared_expert" },
  );
  const moe = (await layoutIR(ir, quiet)).views.find((graph) => graph.id === "L2:moe");
  expect(moe?.nodes).toHaveLength(8);
  expect(moe?.nodes.map((node) => node.ir_node)).toContain("moe.shared_expert");
  expect(moe?.nodes.map((node) => node.ir_node)).not.toContain("moe.shared_expert.gate_up");
  expect(moe?.collapsed_of).toContain("moe.shared_expert.gate_up");
});

it("creates separate, stable views for multiple prefixed layer bands", async () => {
  const ir = denseIR();
  const decoder = ir.nodes.find((node) => node.id === "decoder") as IRNode;
  const inside = ir.nodes.filter((node) => node.parent != null);
  const ids = new Set([decoder.id, ...inside.map((node) => node.id)]);
  ir.nodes.push(
    ...[decoder, ...inside].map((node) => ({
      ...node,
      id: `band2.${node.id}`,
      parent: node.parent ? `band2.${node.parent}` : null,
    })),
  );
  ir.edges.push(
    ...ir.edges
      .filter((edge) => ids.has(edge.from) && ids.has(edge.to))
      .map((edge) => ({
        ...edge,
        id: `band2.${edge.id}`,
        from: `band2.${edge.from}`,
        to: `band2.${edge.to}`,
      })),
  );
  ir.groups.push({
    id: "band2",
    layers: ["3", "L-1"],
    representative_layer: "3",
    template: "band2.decoder",
  });
  const view = await layoutIR(ir, quiet);
  expect(view.views.map((graph) => graph.id)).toEqual([
    "L0",
    "L1:band2",
    "L1:decoder",
    "L2:attn",
    "L2:band2.attn",
    "L2:band2.mlp",
    "L2:mlp",
  ]);
});

it.each(["mla", "hybrid", "vlm"] as const)(
  "uses a warned generic fallback for %s",
  async (family) => {
    const ir = fixture("qwen3-30b-a3b");
    ir.views.family = family;
    const warn = vi.fn();
    const view = await layoutIR(ir, { onWarning: warn });
    expect(warn).toHaveBeenCalledWith(
      expect.stringContaining(`Family ${family}: generic template fallback`),
    );
    expect(view.views.map((graph) => graph.id)).toEqual(["L0", "L1:decoder"]);
  },
);

it("fails excessive budgets instead of silently dropping nodes or edges", async () => {
  expect(checkBudget({ id: "bad", nodes: Array(10), edges: Array(13) }, 3)).toHaveLength(3);
  expect(() => assertBudget({ id: "bad", nodes: [], edges: [] }, 3)).toThrow("accents 3");
  const ir = fixture("qwen3-8b");
  const node = ir.nodes[0] as IRNode;
  ir.nodes.push(...[1, 2, 3].map((i) => ({ ...node, id: `extra${i}` })));
  await expect(layoutIR(ir, quiet)).rejects.toThrow("nodes 10 exceeds budget 9");
});

it("rejects invalid schema, dangling references, duplicates, cycles and diagonal output", async () => {
  await expect(layoutIR({ nodes: [] }, quiet)).rejects.toThrow("Invalid IR");
  for (const mutate of [
    (ir: IR) => ir.nodes.push(ir.nodes[0] as IRNode),
    (ir: IR) => {
      (ir.edges[0] as IREdge).to = "missing";
    },
    (ir: IR) => {
      (ir.nodes[0] as IRNode).parent = ir.nodes[0]?.id;
    },
    (ir: IR) => {
      (ir.groups[0] as IR["groups"][number]).template = "missing";
    },
  ]) {
    const ir = fixture("qwen3-8b");
    mutate(ir);
    expect(() => validateIR(ir)).toThrow();
  }
  const view = await layoutIR(fixture("qwen3-8b"), quiet);
  const edge = view.views[0]?.edges[0];
  if (!edge) throw new Error("Missing test edge");
  edge.points = [
    [0, 0],
    [4, 4],
  ];
  expect(() => validateView(view)).toThrow("Diagonal edge");
});

it("measures mono shapes and sans names with fixed glyph widths, including multiline and Unicode", () => {
  expect(textWidth("iii")).toBeLessThan(textWidth("WWW"));
  expect(textWidth("iii", "mono")).toBe(textWidth("WWW", "mono"));
  expect(textWidth("[T, H]", "mono")).toBe(48);
  expect(textWidth("模型")).toBe(28);
  expect(nodeSize("Attention\nLayers 0…L-1").h).toBeGreaterThan(nodeSize("Attention").h);
});

it("runs the built Node CLI twice with identical bytes and preserves output on invalid input", () => {
  execFileSync("pnpm", ["--filter", "@inferatlas/contracts", "build"], {
    cwd: root,
    stdio: "pipe",
  });
  execFileSync("pnpm", ["--filter", "@inferatlas/layout", "build"], {
    cwd: root,
    stdio: "pipe",
  });
  const directory = mkdtempSync(join(tmpdir(), "inferatlas-layout-"));
  const cli = join(root, "packages/layout/dist/cli.js");
  try {
    for (const name of ["qwen3-30b-a3b", "qwen3-8b"]) {
      const outputs = [join(directory, "a.json"), join(directory, "b.json")];
      for (const output of outputs)
        execFileSync(
          process.execPath,
          [cli, join(root, `fixtures/${name}/ir.json`), "-o", output],
          { stdio: "pipe" },
        );
      expect(readFileSync(outputs[0] as string)).toEqual(readFileSync(outputs[1] as string));
    }
    const invalid = join(directory, "invalid.json");
    const output = join(directory, "keep.json");
    writeFileSync(invalid, "{}");
    writeFileSync(output, "keep me");
    expect(spawnSync(process.execPath, [cli, invalid, "-o", output]).status).toBe(1);
    expect(readFileSync(output, "utf8")).toBe("keep me");
    expect(spawnSync(process.execPath, [cli, invalid, "-o", invalid]).status).toBe(1);
    expect(readFileSync(invalid, "utf8")).toBe("{}");
    expect(spawnSync(process.execPath, [cli]).status).toBe(1);
    expect(spawnSync(process.execPath, [cli, "--help"]).status).toBe(0);
  } finally {
    rmSync(directory, { recursive: true, force: true });
  }
}, 30000);
