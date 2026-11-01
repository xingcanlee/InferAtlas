import type { Index, IndexCell } from "@inferatlas/contracts";
import { describe, expect, it } from "vitest";
import { type InlineData, singleTarget } from "@/lib/data";
import { cellStatusText, cellUnavailableMessage, devicesOf } from "@/lib/status";

const cell = (over: Partial<IndexCell>): IndexCell => ({
  version: "main-verified",
  model: "deepseek-v3",
  device: "A2",
  status: "missing",
  triage_count: 0,
  data_path: null,
  ...over,
});

describe("cell status messages", () => {
  it("never calls a triage cell missing", () => {
    const msg = cellUnavailableMessage(cell({ status: "triage", triage_count: 668 }));
    expect(msg).toBe(
      "deepseek-v3 @ main-verified on A2: Blocked by 668 unresolved triage items; no numeric data was published.",
    );
    expect(msg).not.toMatch(/missing/i);
    expect(cellStatusText(cell({ status: "triage", triage_count: 1 })).detail).toContain(
      "1 unresolved triage item;",
    );
    expect(cellStatusText(cell({ status: "triage", triage_count: 0 })).detail).toContain(
      "Blocked by unresolved triage items",
    );
  });

  it("distinguishes triage with data, ok and truly missing cells", () => {
    expect(
      cellStatusText(cell({ status: "triage", triage_count: 2, data_path: "data/x" })),
    ).toEqual({ title: "Needs triage", detail: "Published with 2 unresolved triage items." });
    expect(cellStatusText(cell({ status: "ok", data_path: "data/x" })).title).toBe("Verified");
    expect(cellStatusText(cell({ status: "ok" })).detail).toContain("no data bundle");
    expect(cellUnavailableMessage(cell({}))).toBe(
      "deepseek-v3 @ main-verified on A2: No artifacts were generated for this combination.",
    );
    expect(cellStatusText(cell({ status: "missing", triage_count: undefined })).title).toBe(
      "Missing",
    );
  });

  it("lists devices sorted and unique", () => {
    const index = {
      versions: [],
      models: [],
      cells: [cell({ device: "H100" }), cell({ device: "A2" }), cell({ device: "H100" })],
    } as Index;
    expect(devicesOf(index)).toEqual(["A2", "H100"]);
  });
});

describe("single-file default target", () => {
  const data = (cells: IndexCell[], bundles: string[]): InlineData => ({
    index: { versions: [], models: [], cells },
    bundles: Object.fromEntries(bundles.map((b) => [b, { ir: {} as never }])),
  });

  it("opens the inlined cell, preferring ok over triage", () => {
    const triage = cell({ model: "a", status: "triage", data_path: "data/a" });
    const ok = cell({ model: "b", device: "H100", status: "ok", data_path: "data/b" });
    expect(singleTarget(data([triage, ok], ["data/a", "data/b"]))).toEqual({
      version: "main-verified",
      model: "b",
      device: "H100",
    });
    expect(singleTarget(data([triage, ok], ["data/a"]))?.model).toBe("a");
  });

  it("has no target without inlined bundles", () => {
    expect(singleTarget(undefined)).toBeUndefined();
    expect(singleTarget(data([cell({ data_path: "data/a", status: "ok" })], []))).toBeUndefined();
  });
});
