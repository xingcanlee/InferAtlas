/** Contract v1.4 UI semantics: class-level notes, recorded normalisation, model inputs, mHC stages. */
import type { Arch, ArchNote, ArchStackStage } from "@inferatlas/contracts";
import { describe, expect, it, vi } from "vitest";
import {
  CLASS_LEVEL_KINDS,
  mhcStageLabel,
  modelInputs,
  NORMALISED_KIND,
  normalizeArch,
  normalizePatches,
  noteMarker,
  notesAt,
  stageLabel,
  subNotes,
} from "@/lib/arch";
import v4Json from "../../fixtures/arch-ui/deepseek-v4-flash/arch.json";
import denseJson from "../../fixtures/arch-ui/qwen3-8b/arch.json";

const v4 = v4Json as unknown as Arch;
const dense = denseJson as unknown as Arch;
const stackOf = (a: Arch) => a.stages.find((s) => s.kind === "stack") as ArchStackStage;

describe("class-level notes (v1.4 §2)", () => {
  it("applies an unknown-class warning to every instance of the class", () => {
    const bands = stackOf(v4).bands;
    // Emitted once, on band 0's mhc_pre_attn / mhc_post_attn …
    for (const id of ["mhc_pre_ffn", "mhc_post_ffn", "mhc_pre_attn", "mhc_post_attn"])
      for (const b of bands) {
        const m = noteMarker(subNotes(v4, [b], id));
        expect(m, `${b.id}/${id}`).toEqual({ severity: "warn", count: 1 });
      }
  });

  it("keeps band-specific kinds on their target", () => {
    const stack = stackOf(dense);
    const [b0] = stack.bands;
    const golden: ArchNote = {
      severity: "warn",
      target: `${b0?.id}/input_layernorm`,
      text: "Golden diff on layer 0",
      triage_kind: "golden_diff",
    };
    const cls: ArchNote = {
      severity: "warn",
      target: `${b0?.id}/input_layernorm`,
      text: "Unannotated override of RMSNorm",
      triage_kind: "unannotated_override",
    };
    const a = { ...dense, notes: [...dense.notes, golden, cls] } as Arch;
    const norm = notesAt(a, [`${b0?.id}/post_attention_layernorm`]);
    expect(norm).toContain(cls);
    expect(norm).not.toContain(golden);
    expect(notesAt(a, [`${b0?.id}/input_layernorm`])).toEqual(
      expect.arrayContaining([golden, cls]),
    );
    // The final norm stage is the same class: the class-level note reaches it too.
    expect(notesAt(a, ["model.norm"])).toContain(cls);
    expect(CLASS_LEVEL_KINDS.has("golden_diff")).toBe(false);
  });
});

describe("normalizeArch records what it patched", () => {
  it("passes conforming data through without a note", () => {
    expect(normalizePatches(v4)).toEqual([]);
    expect(normalizeArch(v4).notes).toHaveLength(v4.notes.length);
  });

  it("adds a model-level info note and warns in development", () => {
    const warn = vi.spyOn(console, "warn").mockImplementation(() => {});
    const stack = stackOf(dense);
    const old = {
      ...dense,
      flags: undefined,
      stages: dense.stages.map((s) =>
        s === stack
          ? {
              ...stack,
              bands: stack.bands.map((b) => ({
                ...b,
                block: {
                  ...b.block,
                  sublayers: b.block.sublayers.map((x) =>
                    x.component?.kind === "attention"
                      ? {
                          ...x,
                          io: { inputs: x.io.inputs, outputs: x.io.outputs },
                          component: { ...x.component, variant: "sliding_window" },
                        }
                      : x,
                  ),
                },
              })),
            }
          : s,
      ),
    } as unknown as Arch;
    const out = normalizeArch(old);
    const note = out.notes.find((n) => n.triage_kind === NORMALISED_KIND);
    expect(note).toMatchObject({ severity: "info", target: null });
    expect(note?.text).toMatch(/deprecated variant "sliding_window"/);
    expect(note?.text).toMatch(/without internals/);
    expect(note?.text).toMatch(/missing flags/);
    expect(normalizePatches(old)).toHaveLength(3);
    expect(warn).toHaveBeenCalledWith(expect.stringContaining("[arch] normalised"));
    warn.mockRestore();
  });
});

describe("model inputs and mHC stages", () => {
  it("lists v1.4 inputs when present, else the primary input", () => {
    expect(modelInputs(dense)).toEqual([dense.input]);
    const pixels = { name: "pixel_values", shape: ["B", "C", "Himg", "Wimg"], dtype: "bfloat16" };
    const vlm = { ...dense, inputs: [pixels, dense.input] } as unknown as Arch;
    expect(modelInputs(vlm).map((t) => t.name)).toEqual(["pixel_values", "input_ids"]);
    // Malformed entries are ignored.
    expect(modelInputs({ ...dense, inputs: [{ name: 3 }] } as unknown as Arch)).toEqual([
      dense.input,
    ]);
  });

  it("labels stream expansion / merge stages", () => {
    expect(mhcStageLabel("model.hc_expand")).toBe("mHC expand (→ HC streams)");
    expect(mhcStageLabel("model.hc_head_op")).toBe("mHC merge (HC streams →)");
    expect(mhcStageLabel("model.norm")).toBeUndefined();
    const head = v4.stages.find((s) => s.id === "model.hc_head_op");
    expect(head && stageLabel(v4, head)).toBe("mHC merge (HC streams →)");
  });
});
