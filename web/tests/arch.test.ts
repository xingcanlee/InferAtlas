/** lib/arch helpers over the hand-written arch.json fixtures (fixtures/arch-ui). */
import { describe, expect, it } from "vitest";
import {
  allBands,
  archEnv,
  bandForLayer,
  bandOrdinal,
  blockSegments,
  evaluator,
  layerCount,
  layerPath,
  MAX_STACK_COLUMNS,
  noteMarker,
  notesFor,
  representative,
  represented,
  shortClass,
  slotAt,
  stackColumns,
  stackItems,
  stacks,
} from "@/lib/arch";
import type { Arch, ArchBand, ArchBlock, ArchStackStage } from "@/lib/arch-types";
import deepseekJson from "../../fixtures/arch-ui/deepseek-v3/arch.json";
import denseJson from "../../fixtures/arch-ui/qwen3-8b/arch.json";
import moeJson from "../../fixtures/arch-ui/qwen3-30b-a3b/arch.json";
import nextJson from "../../fixtures/arch-ui/qwen3-next-80b-a3b/arch.json";

const dense = denseJson as unknown as Arch;
const moe = moeJson as unknown as Arch;
const deepseek = deepseekJson as unknown as Arch;
const next = nextJson as unknown as Arch;
const FIXTURES = { dense, moe, deepseek, next };

describe("fixtures follow the contract", () => {
  for (const [name, arch] of Object.entries(FIXTURES)) {
    it(`${name}: every shape evaluates with the default symbols`, () => {
      const ev = evaluator(archEnv(arch, {}).env);
      const shapes: string[][] = [arch.input.shape, arch.output.shape];
      for (const s of arch.stages) {
        if (s.kind === "stack")
          for (const b of s.bands)
            for (const sub of b.block.sublayers) {
              shapes.push(...sub.io.inputs.map((t) => t.shape));
              shapes.push(...sub.io.outputs.map((t) => t.shape));
              shapes.push(...(sub.module?.params ?? []).map((p) => p.shape));
            }
        else shapes.push(...s.io.inputs.map((t) => t.shape), ...s.io.outputs.map((t) => t.shape));
      }
      for (const shape of shapes)
        for (const d of shape) expect(ev(d), `${name}: ${d}`).toBeTypeOf("number");
    });
    it(`${name}: bands tile the layers without gaps`, () => {
      const bands = allBands(arch);
      let next = 0;
      for (const b of bands) {
        expect(b.layers[0]).toBe(next);
        expect(b.count).toBe(b.layers[1] - b.layers[0] + 1);
        next = b.layers[1] + 1;
      }
      expect(next).toBe(layerCount(arch));
    });
  }
  it("uses the real config numbers", () => {
    expect(layerCount(dense)).toBe(36);
    expect(layerCount(moe)).toBe(48);
    expect(layerCount(deepseek)).toBe(61);
    expect(layerCount(next)).toBe(48);
    expect(deepseek.symbols.Rkv?.value).toBe(512);
    expect(next.symbols.E?.value).toBe(512);
  });
});

describe("stack folding", () => {
  it("keeps heterogeneous bands (DeepSeek-V3: 3 dense + 58 MoE)", () => {
    const items = stackItems(stacks(deepseek)[0]?.bands ?? []);
    expect(items.map((i) => (i.kind === "band" ? i.band.layers : i.kind))).toEqual([
      [0, 2],
      [3, 60],
    ]);
  });

  it("folds a periodic pattern (Qwen3-Next: (3 linear + 1 full) × 12)", () => {
    const bands = stacks(next)[0]?.bands ?? [];
    expect(bands).toHaveLength(24);
    const items = stackItems(bands);
    expect(items).toHaveLength(1);
    const g = items[0];
    if (g?.kind !== "pattern") throw new Error("expected a pattern group");
    expect(g.repeats).toBe(12);
    expect(g.period.map((b) => b.count)).toEqual([3, 1]);
    expect(g.layers).toEqual([0, 47]);
    // layer 47 → full-attention member → drawn by the period's full-attention band
    const b47 = bandForLayer(next, 47);
    expect(b47?.id).toBe("full_47");
    expect(representative(items, b47 as never)).toBe(g.period[1]);
    expect(represented(items, g.period[0] as never)).toHaveLength(12);
  });

  it("folds unlabelled alternating bands by structure, ignoring ids and paths", () => {
    const bands = (stacks(next)[0]?.bands ?? []).map((b, i) => ({
      ...b,
      pattern: null,
      // Emitter-style differences that must not break folding.
      block: {
        ...b.block,
        sublayers: b.block.sublayers.map((s) => ({
          ...s,
          id: `${s.id}_${i}`,
          module: s.module && { ...s.module, path: `x.${i}`, source: null },
        })),
      },
    }));
    const items = stackItems(bands);
    expect(items).toHaveLength(1);
    const g = items[0];
    if (g?.kind !== "pattern") throw new Error("expected a pattern group");
    expect(g.repeats).toBe(12);
    expect(g.label).toBe("3× linear attn + 1× GQA attn");
    expect(stackColumns(items)).toBe(2);
  });

  it("non-periodic heterogeneous stacks switch to the compact list", () => {
    const base = stacks(deepseek)[0]?.bands ?? [];
    const dense = base[0] as ArchBand;
    const moeBand = base[1] as ArchBand;
    const counts = [1, 2, 3, 4, 5, 6];
    let start = 0;
    const bands = counts.map((c, i) => {
      const b = {
        ...(i % 2 ? moeBand : dense),
        id: `b${i}`,
        count: c,
        layers: [start, start + c - 1] as [number, number],
      };
      start += c;
      return b;
    });
    const items = stackItems(bands);
    expect(items.every((it) => it.kind === "band")).toBe(true);
    expect(stackColumns(items)).toBeGreaterThan(MAX_STACK_COLUMNS);
  });

  it("does not fold a pattern whose members differ", () => {
    const bands = (stacks(next)[0]?.bands ?? []).slice(0, 3);
    const items = stackItems(bands);
    expect(items[0]?.kind === "pattern" && items[0].repeats).toBe(1);
  });
});

describe("block segments", () => {
  const block = (moe.stages[1] as { bands: { block: ArchBlock }[] }).bands[0]?.block as ArchBlock;

  it("explicit residual adds close the branches", () => {
    const segs = blockSegments(block);
    expect(segs.map((s) => (s.kind === "branch" ? s.body.map((b) => b.id) : s.kind))).toEqual([
      ["input_layernorm", "self_attn"],
      ["post_attention_layernorm", "mlp"],
    ]);
  });

  it("infers ⊕ after mixer/ffn when no residual_add sublayers exist", () => {
    const implicit = {
      ...block,
      sublayers: block.sublayers.filter((s) => s.role !== "residual_add"),
    };
    const segs = blockSegments(implicit);
    expect(segs).toHaveLength(2);
    expect(segs.every((s) => s.kind === "branch" && s.add === undefined)).toBe(true);
  });

  it("post-norm keeps the norm on the main stream", () => {
    const [norm1, attn, add1, norm2, mlp, add2] = block.sublayers;
    const post: ArchBlock = {
      kind: "decoder",
      residual: "post_norm",
      sublayers: [attn, add1, norm1, mlp, add2, norm2] as ArchBlock["sublayers"],
    };
    expect(blockSegments(post).map((s) => s.kind)).toEqual(["branch", "node", "branch", "node"]);
  });

  it("unknown residual style is a plain chain", () => {
    expect(blockSegments({ ...block, residual: "unknown" }).every((s) => s.kind === "node")).toBe(
      true,
    );
  });
});

describe("layer slots", () => {
  it("map ordinals to bands across several stacks and handle no stack", () => {
    const stack = stacks(dense)[0] as ArchStackStage;
    const twoStacks: Arch = {
      ...dense,
      stages: [
        { ...stack, id: "encoder", label: "Encoder" },
        {
          ...stack,
          id: "decoder",
          label: "Decoder",
          bands: stack.bands.map((b) => ({ ...b, id: "d" })),
        },
      ],
    };
    expect(layerCount(twoStacks)).toBe(72);
    const s = slotAt(twoStacks, 40);
    expect(s?.stack.label).toBe("Decoder");
    expect(s?.layer).toBe(4);
    expect(bandOrdinal(twoStacks, s?.band as ArchBand)).toBe(36);
    const none: Arch = { ...dense, stages: dense.stages.filter((x) => x.kind !== "stack") };
    expect(layerCount(none)).toBe(0);
    expect(bandForLayer(none, 0)).toBeUndefined();
  });

  it("short class names for figures, full names in the data (v1.1 §10)", () => {
    const m = bandForLayer(moe, 0)?.block.sublayers.find((x) => x.id === "mlp")?.module;
    expect(m?.class_name).toBe("vllm.model_executor.models.qwen3_moe.Qwen3MoeSparseMoeBlock");
    expect(shortClass(m?.class_name)).toBe("Qwen3MoeSparseMoeBlock");
  });
});

describe("symbols, paths and notes", () => {
  it("evaluates shapes with URL overrides of runtime symbols only", () => {
    const { env, overridden } = archEnv(moe, { T: 8192, tp: 4, bogus: 3, H: 999, E: 64 });
    expect(env.H).toBe(2048);
    expect([...overridden].sort()).toEqual(["T", "tp"]);
    const ev = evaluator(env);
    expect(ev("(Hq+2*Hkv)*D/tp")).toBe(1280);
    expect(ev("T")).toBe(8192);
    expect(ev("unknown_symbol")).toBeUndefined();
  });

  it("rewrites the representative module path to the chosen layer", () => {
    const band = bandForLayer(deepseek, 40);
    const sub = band?.block.sublayers.find((s) => s.id === "mlp");
    expect(layerPath(sub?.module as never, band as never, 40)).toBe("model.layers.40.mlp");
  });

  it("collects notes per target and summarises their severity", () => {
    const n = notesFor(deepseek, ["moe/mlp"]);
    expect(n).toHaveLength(2);
    expect(noteMarker(n)).toEqual({ severity: "warn", count: 2 });
    // Qwen3-Next repeats the same note on every linear band: counted once.
    const items = stackItems(stacks(next)[0]?.bands ?? []);
    const g = items[0];
    if (g?.kind !== "pattern") throw new Error("pattern");
    const targets = represented(items, g.period[0] as never).map((b) => `${b.id}/linear_attn`);
    expect(noteMarker(notesFor(next, targets))).toEqual({ severity: "warn", count: 1 });
  });
});
