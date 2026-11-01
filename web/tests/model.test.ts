import type { Eval, IR, View } from "@inferatlas/contracts";
import { describe, expect, it } from "vitest";
import { buildGraph } from "@/components/graph/build";
import { buildEnv, computeAll, evalPointer, kvSweep, memoryLedger, seqCapacity } from "@/lib/model";
import evalJson from "../../fixtures/qwen3-30b-a3b/eval.json";
import irJson from "../../fixtures/qwen3-30b-a3b/ir.json";
import viewJson from "../../fixtures/qwen3-30b-a3b/view.json";

const ir = irJson as unknown as IR;
const oracle = evalJson as unknown as Eval;
const view = viewJson as unknown as View;

describe("fixture eval oracle", () => {
  for (const c of oracle.cases) {
    it(c.id, () => {
      const { env, errors } = buildEnv(ir, c.env as Record<string, number | string>);
      expect(errors).toEqual({});
      for (const [ptr, expected] of Object.entries(c.expected)) {
        const got = evalPointer(ir, env, ptr);
        if (Array.isArray(expected)) expect(got, ptr).toEqual(expected);
        else expect(got as number, ptr).toBeCloseTo(expected as number, 3);
      }
    });
  }
});

describe("env binding", () => {
  it("rejects out-of-domain overrides and keeps defaults", () => {
    const s = buildEnv(ir, { tp: 3, T: 0, dtype_w: "int3" });
    expect(Object.keys(s.errors).sort()).toEqual(["T", "dtype_w", "tp"]);
    expect(s.params.tp).toBe(1);
  });
  it("rejects inconsistent ep and computes with EP disabled", () => {
    const s = buildEnv(ir, { tp: 2, ep: 4 });
    expect(s.errors.ep).toContain("1 or tp×dp");
    expect(s.env.ep).toBe(1);
  });
  it("recomputes a parameter change (metrics, ledger, graph, KV tab) within a frame", () => {
    const run = (i: number) => {
      const state = buildEnv(ir, { tp: [1, 2, 4, 8][i % 4] as number });
      const phase = i % 2 ? "decode" : "prefill";
      const metrics = computeAll(ir, state.env, phase);
      memoryLedger(ir, state.env);
      kvSweep(ir, state);
      for (const graph of view.views) {
        buildGraph({
          ir,
          view,
          graph,
          metrics,
          env: state.env,
          phase,
          shapes: "num",
          heat: "flops",
          overlay: true,
          selected: "attn.o",
        });
      }
    };
    run(0); // warm compiled-closure cache
    const runs = 20;
    const t0 = performance.now();
    for (let i = 0; i < runs; i++) run(i);
    expect((performance.now() - t0) / runs).toBeLessThan(16);
  });
  it("counts sequences in whole KV blocks", () => {
    expect(seqCapacity(1000, { context: 100, block_size: 128 })).toBe(7);
    expect(seqCapacity(1000, { context: 127, block_size: 128 })).toBe(7);
    expect(seqCapacity(1000, { context: 128, block_size: 128 })).toBe(3);
    expect(seqCapacity(1000, { context: 99 })).toBe(10);
  });
});

it("re-evaluates annotation predicates after runtime dtype changes", async () => {
  const { runtimeAnnotations } = await import("../src/lib/model");
  const source = structuredClone(ir);
  const node = source.nodes.find((n) => n.ascend);
  if (!node?.ascend) throw new Error("fixture needs an overlay");
  node.ascend.applies_when = "is_a2 && dtype_w == bf16";
  node.ascend.context = { is_a2: true };
  expect(
    runtimeAnnotations(source, { dtype_w: "bf16" }).nodes.find((n) => n.id === node.id)?.ascend,
  ).toBeTruthy();
  const inactiveEnvs: Record<string, string>[] = [{ dtype_w: "fp16" }, {}];
  for (const env of inactiveEnvs) {
    const fallback = runtimeAnnotations(source, env).nodes.find((n) => n.id === node.id)?.ascend;
    expect(fallback?.note).toBe("Ascend 替换 — 当前参数下无已审阅说明");
    expect(fallback?.src).toEqual(node.ascend.src);
    expect(fallback?.src.some((s) => s.symbol.includes("Ascend"))).toBe(true);
    expect(fallback?.tag).toEqual([]);
    expect(fallback?.annotation_id).toBe("");
  }
  expect(
    runtimeAnnotations(source, { dtype_w: "bf16" }).nodes.find((n) => n.id === node.id)?.ascend,
  ).toEqual(node.ascend);
  expect(node.ascend).toBeTruthy();
});
