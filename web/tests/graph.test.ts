import type { IR, View } from "@inferatlas/contracts";
import { describe, expect, it } from "vitest";
import { buildGraph, placeLabel } from "@/components/graph/build";
import { diffIR, sideMarks } from "@/lib/diff";
import { buildEnv, computeAll } from "@/lib/model";
import { breadcrumb, drillTarget, parentView, viewForNode } from "@/lib/views";
import irJson from "../../fixtures/qwen3-30b-a3b/ir.json";
import viewJson from "../../fixtures/qwen3-30b-a3b/view.json";

const ir = irJson as unknown as IR;
const view = viewJson as unknown as View;

describe("graph build", () => {
  const { env } = buildEnv(ir, { tp: 4 });
  const metrics = computeAll(ir, env, "prefill");
  const graph = view.views.find((v) => v.id === "L2:attn");
  it("maps view nodes/edges with numeric labels and comm lanes", () => {
    if (!graph) throw new Error("fixture view missing");
    const g = buildGraph({
      ir,
      view,
      graph,
      metrics,
      env,
      phase: "prefill",
      shapes: "num",
      heat: "flops",
      overlay: true,
      selected: "attn.o",
    });
    expect(g.nodes).toHaveLength(graph.nodes.length);
    const reduce = g.nodes.find((n) => n.id === "attn.o.reduce");
    expect(reduce?.type).toBe("comm");
    expect(reduce?.data.lane).toBe("tp×4");
    expect(g.edges.find((e) => e.id === "e0")?.data?.label).toBe("[4096, 8, 128]");
    expect(
      g.edges
        .filter((e) => e.data?.focus)
        .map((e) => e.id)
        .sort(),
    ).toEqual(["e3", "e4"]);
    const shares = g.nodes.map((n) => n.data.share ?? 0);
    expect(shares.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 6);
  });
  it("lifts labels that do not fit their segment", () => {
    expect(
      placeLabel(
        [
          [0, 96],
          [48, 96],
        ],
        [24, 80],
        "[4096, 2048]",
      ),
    ).toEqual([24, 46]);
    expect(
      placeLabel(
        [
          [0, 160],
          [400, 160],
        ],
        [200, 160],
        "[T, H]",
      ),
    ).toEqual([200, 160]);
  });
});

describe("navigation", () => {
  it("drills and goes back up", () => {
    expect(drillTarget(ir, view, "decoder")).toBe("L1:decoder");
    expect(drillTarget(ir, view, "attn")).toBe("L2:attn");
    expect(parentView(ir, view, "L2:moe")).toBe("L1:decoder");
    expect(breadcrumb(ir, view, "L2:attn")).toEqual(["L0", "L1:decoder", "L2:attn"]);
    expect(viewForNode(ir, view, "attn.q_norm")).toEqual({
      viewId: "L2:attn",
      visibleNode: "attn.qk_norm",
    });
  });
});

describe("diff", () => {
  it("ignores provenance and reports semantic changes", () => {
    const b = structuredClone(ir);
    for (const n of b.nodes) n.src = { ...n.src, line: n.src.line + 3 };
    const gate = b.nodes.find((n) => n.id === "moe.gate");
    if (gate) gate.class = "X";
    b.nodes = b.nodes.filter((n) => n.id !== "moe.experts.reduce");
    b.edges = b.edges.filter((e) => e.to !== "moe.experts.reduce");
    const d = diffIR(ir, b);
    expect([...d.nodes.keys()].sort()).toEqual(["moe.experts.reduce", "moe.gate"]);
    expect(d.nodes.get("moe.experts.reduce")?.mark).toBe("removed");
    const marks = sideMarks(
      d,
      view.views.find((v) => v.id === "L2:moe"),
      "a",
    );
    expect(marks.nodes.get("moe.experts.reduce")).toBe("removed");
    expect(marks.edges.has("e5")).toBe(true);
  });
});
