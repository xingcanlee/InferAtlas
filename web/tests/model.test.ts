import type { Eval, IR } from "@inferatlas/contracts";
import { describe, expect, it } from "vitest";
import { buildEnv, computeAll, evalPointer, memoryLedger } from "@/lib/model";
import evalJson from "../../fixtures/qwen3-30b-a3b/eval.json";
import irJson from "../../fixtures/qwen3-30b-a3b/ir.json";

const ir = irJson as unknown as IR;
const oracle = evalJson as unknown as Eval;

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
  it("warns on inconsistent ep", () => {
    const s = buildEnv(ir, { tp: 2, ep: 4 });
    expect(s.warnings.some((w) => w.params.includes("ep"))).toBe(true);
  });
  it("recomputes all metrics well under a frame budget", () => {
    const { env } = buildEnv(ir, {});
    computeAll(ir, env, "prefill");
    const t0 = performance.now();
    const runs = 20;
    for (let i = 0; i < runs; i++) {
      const e = buildEnv(ir, { tp: [1, 2, 4, 8][i % 4] as number }).env;
      computeAll(ir, e, i % 2 ? "decode" : "prefill");
      memoryLedger(ir, e);
    }
    expect((performance.now() - t0) / runs).toBeLessThan(16);
  });
});
