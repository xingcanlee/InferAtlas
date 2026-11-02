/**
 * Real emitter output (S1, contract v1.3). 1) A trimmed copy of the H100 DeepSeek-V4-Flash
 * arch.json (≤ 2 params per module; fixtures/arch-ui/build.mjs) is a committed fixture. 2) When
 * INFERATLAS_S1_DATA points at a release-data directory, every arch.json under it goes through
 * the UI helpers and renders (skipped otherwise).
 */
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import type { Arch, ArchTensor, Index } from "@inferatlas/contracts";
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ArchFigure } from "@/components/arch/ArchFigure";
import {
  archEnv,
  bandSummary,
  blockSegments,
  encoderStages,
  evaluator,
  layerCount,
  layerSlots,
  normalizeArch,
  stackColumns,
  stackItems,
  stacks,
  subTarget,
} from "@/lib/arch";
import { archCoverage, cellCoverage } from "@/lib/coverage";
import v4TrimmedJson from "../../fixtures/arch-ui/s1/deepseek-v4-flash/arch.json";

describe("S1 DeepSeek-V4-Flash (trimmed real output, H100)", () => {
  const arch = normalizeArch(v4TrimmedJson as unknown as Arch);
  const bands = stacks(arch)[0]?.bands ?? [];

  it("one label per period, shared by its bands (v1.3 §1); prefix bands unlabelled", () => {
    expect(bands).toHaveLength(42);
    expect(bands.slice(0, 2).map((b) => b.pattern)).toEqual([null, null]);
    expect(new Set(bands.slice(2).map((b) => b.pattern))).toEqual(
      new Set(["1× MQA ÷128 + 1× MQA ÷4 + indexer"]),
    );
    const items = stackItems(bands);
    expect(items.map((it) => it.kind)).toEqual(["band", "band", "pattern"]);
    const g = items[2];
    if (g?.kind !== "pattern") throw new Error("pattern");
    expect(g.period).toHaveLength(2);
    expect(g.repeats).toBe(20);
    expect(g.label).toBe("1× MQA ÷128 + 1× MQA ÷4 + indexer");
    // Covered layers tile the stack.
    const covered = items.reduce(
      (n, it) =>
        n + (it.kind === "band" ? it.band.count : it.members.reduce((m, b) => m + b.count, 0)),
      0,
    );
    expect(covered).toBe(layerCount(arch));
  });

  it("attention variants are head-sharing schemes (v1.3 §5): MQA, locality in extras", () => {
    const variants = new Set(
      bands.flatMap((b) =>
        b.block.sublayers.flatMap((s) =>
          s.component?.kind === "attention" ? [s.component.variant] : [],
        ),
      ),
    );
    expect([...variants]).toEqual(["mqa"]);
    const routers = bands.map(
      (b) =>
        b.block.sublayers.find((s) => s.component?.kind === "moe")?.component as {
          router: { scoring: string };
        },
    );
    // v1.3 §4: the first three layers route by token-id hash.
    expect(routers.slice(0, 2).map((r) => r.router.scoring)).toEqual(["hash", "hash"]);
    expect(routers[2]?.router.scoring).not.toBe("hash");
  });

  it("mHC sublayers form hyper segments; their notes resolve", () => {
    const segs = blockSegments((bands[5] ?? bands[0])?.block as never);
    expect(segs.map((s) => s.kind)).toEqual(["hyper", "hyper"]);
    const targets = new Set([
      ...arch.stages.map((s) => s.id),
      ...bands.flatMap((b) => [b.id, ...b.block.sublayers.map((s) => subTarget(b.id, s.id))]),
    ]);
    for (const n of arch.notes) if (n.target) expect(targets, n.target).toContain(n.target);
  });

  it("renders without throwing", () => {
    const { container } = render(
      <ArchFigure
        arch={arch}
        evaluate={evaluator(archEnv(arch, {}).env)}
        selection={{}}
        onSelectBand={() => {}}
        onSelectSub={() => {}}
        onSelectStage={() => {}}
      />,
    );
    expect(container.querySelectorAll("[data-slot=residual-bus]").length).toBeGreaterThan(0);
    expect(stackColumns(stackItems(bands))).toBeGreaterThan(0);
  });
});

// ------------------------------------------------------------------- full release-data sweep

/**
 * Opt-in only: another worktree's release data may be uncommitted or half-written, so the sweep
 * runs only when INFERATLAS_S1_DATA names a directory (e.g. `.worktrees/S1/.release-data`).
 */
const dataDir = process.env.INFERATLAS_S1_DATA
  ? resolve(process.env.INIT_CWD ?? process.cwd(), process.env.INFERATLAS_S1_DATA)
  : undefined;

function archFiles(dir: string | undefined): string[] {
  if (!dir || !existsSync(dir)) return [];
  const out: string[] = [];
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) out.push(...archFiles(p));
    else if (name === "arch.json") out.push(p);
  }
  return out.sort();
}

const files = archFiles(dataDir);

describe.skipIf(files.length === 0)("S1 release data (all arch.json)", () => {
  for (const file of files) {
    it(relative(dataDir ?? "", file), () => {
      const arch = normalizeArch(JSON.parse(readFileSync(file, "utf8")) as Arch);
      const ev = evaluator(archEnv(arch, {}).env);
      const shapes: { where: string; t: ArchTensor }[] = [];
      const push = (where: string, ts: ArchTensor[] | undefined) =>
        shapes.push(...(ts ?? []).map((t) => ({ where, t })));
      const ids = new Set<string>(arch.stages.map((s) => s.id));
      for (const s of arch.stages) {
        if (s.kind === "stack") {
          const items = stackItems(s.bands);
          expect(items.length).toBeGreaterThan(0);
          for (const b of s.bands) {
            ids.add(b.id);
            bandSummary(b);
            blockSegments(b.block);
            for (const sub of b.block.sublayers) {
              ids.add(subTarget(b.id, sub.id));
              push(`${b.id}/${sub.id}`, sub.io.inputs);
              push(`${b.id}/${sub.id}`, sub.io.outputs);
              push(`${b.id}/${sub.id}`, sub.io.internals);
              push(`${b.id}/${sub.id} params`, sub.module?.params);
            }
          }
        } else {
          push(s.id, s.io.inputs);
          push(s.id, s.io.outputs);
          push(s.id, s.io.internals);
          push(`${s.id} params`, s.module?.params);
        }
      }
      expect(layerSlots(arch).length).toBe(layerCount(arch));
      // VLM stages before the token embedding become the encoder branch, never lost.
      if (encoderStages(arch).length)
        expect(arch.stages.some((s) => s.kind === "embedding")).toBe(true);
      const { container, unmount } = render(
        <ArchFigure
          arch={arch}
          evaluate={ev}
          selection={{}}
          onSelectBand={() => {}}
          onSelectSub={() => {}}
          onSelectStage={() => {}}
        />,
      );
      expect(container.querySelector("[data-testid=arch-figure]")).not.toBeNull();
      unmount();
      const bad = shapes
        .flatMap(({ where, t }) => t.shape.map((d) => ({ where, name: t.name, d })))
        .filter(({ d }) => ev(d) === undefined)
        .map(({ where, name, d }) => `${where} ${name}: ${d}`);
      expect(bad).toEqual([]);
      for (const n of arch.notes) if (n.target) expect(ids, n.target).toContain(n.target);
    });
  }
});

// v1.4 §1: the index's structure coverage is what the UI derives from the cell's arch.json.
const indexFile = dataDir ? join(dataDir, "index.json") : undefined;
const index: Index | undefined =
  indexFile && existsSync(indexFile) ? JSON.parse(readFileSync(indexFile, "utf8")) : undefined;

describe.skipIf(!index)("S1 release data (index coverage)", () => {
  it("every built cell carries coverage equal to archCoverage(arch.json)", () => {
    const built = (index?.cells ?? []).filter((c) => c.status !== "missing");
    expect(built.length).toBeGreaterThan(0);
    for (const cell of built) {
      const where = `${cell.version}/${cell.device}/${cell.model}`;
      const cov = cellCoverage(cell);
      expect(cov, where).toBeDefined();
      const file = join(dataDir ?? "", (cell.data_path ?? "").replace(/^data\//, ""), "arch.json");
      expect(existsSync(file), file).toBe(true);
      const arch = JSON.parse(readFileSync(file, "utf8")) as Arch;
      expect(archCoverage(normalizeArch(arch)), where).toEqual(cov);
    }
  });
});
