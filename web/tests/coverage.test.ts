/** v1.4 §1 structure coverage: index field (defensive read), derivation from arch.json, wording. */
import type { Arch, IndexCell } from "@inferatlas/contracts";
import { describe, expect, it } from "vitest";
import { archCoverage, cellCoverage, coverageText, verificationText } from "@/lib/coverage";
import v4Json from "../../fixtures/arch-ui/deepseek-v4-flash/arch.json";
import denseJson from "../../fixtures/arch-ui/qwen3-8b/arch.json";

const cell = (extra: Record<string, unknown> = {}): IndexCell =>
  ({
    version: "main-verified",
    model: "m",
    device: "A2",
    status: "triage",
    triage_count: 7,
    data_path: "data/x",
    ...extra,
  }) as IndexCell;

describe("structure coverage", () => {
  it("reads IndexCell.coverage when well-formed, ignores it otherwise", () => {
    const cov = {
      sublayers_ruled: 4,
      sublayers_generic: 1,
      notes_warn: 2,
      notes_info: 3,
      arch: true,
    };
    expect(cellCoverage(cell({ coverage: cov }))).toEqual(cov);
    expect(cellCoverage(cell())).toBeUndefined();
    expect(cellCoverage(cell({ coverage: { ...cov, arch: "yes" } }))).toBeUndefined();
    expect(cellCoverage(cell({ coverage: { ...cov, sublayers_ruled: -1 } }))).toBeUndefined();
  });

  it("words full, partial, absent and unknown coverage", () => {
    const base = { notes_warn: 0, notes_info: 0, arch: true };
    expect(coverageText({ ...base, sublayers_ruled: 4, sublayers_generic: 0 })).toMatchObject({
      label: "structure ✓ · 4/4 ruled",
      tone: "full",
    });
    const partial = coverageText({
      ...base,
      sublayers_ruled: 10,
      sublayers_generic: 2,
      notes_warn: 3,
    });
    expect(partial).toMatchObject({ label: "10/12 ruled · 2 generic", tone: "partial" });
    expect(partial.detail).toContain("3 warnings");
    expect(
      coverageText({ ...base, sublayers_ruled: 0, sublayers_generic: 0, arch: false }).label,
    ).toBe("no structure");
    expect(coverageText(undefined)).toMatchObject({ label: "structure", tone: "unknown" });
  });

  it("derives the same numbers from an arch.json", () => {
    const dense = archCoverage(denseJson as unknown as Arch);
    expect(dense.arch).toBe(true);
    expect(dense.sublayers_generic).toBe(0);
    expect(dense.sublayers_ruled).toBeGreaterThan(0);
    // DeepSeek-V4: the mHC ops are generic components with warnings.
    const v4 = archCoverage(v4Json as unknown as Arch);
    expect(v4.sublayers_generic).toBeGreaterThan(0);
    expect(v4.notes_warn).toBe(3);
  });

  it("keeps numeric verification as a secondary wording", () => {
    expect(verificationText(cell()).label).toBe("unverified · 7");
    expect(verificationText(cell()).detail).toContain("Structure is unaffected");
    expect(verificationText(cell({ status: "ok", triage_count: 0 })).label).toBe("verified");
  });
});
