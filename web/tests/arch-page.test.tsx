/** Architecture tab: default view, layer detail with evaluated shapes, empty states. */
import type { Arch, ArchModuleStage, Index, IR, Report, View } from "@inferatlas/contracts";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it } from "vitest";
import deepseekJson from "../../fixtures/arch-ui/deepseek-v3/arch.json";
import v4Json from "../../fixtures/arch-ui/deepseek-v4-flash/arch.json";
import denseArchJson from "../../fixtures/arch-ui/qwen3-8b/arch.json";
import moeArchJson from "../../fixtures/arch-ui/qwen3-30b-a3b/arch.json";
import nextJson from "../../fixtures/arch-ui/qwen3-next-80b-a3b/arch.json";
import indexJson from "../../fixtures/index.json";
import irJson from "../../fixtures/qwen3-30b-a3b/ir.json";
import reportJson from "../../fixtures/qwen3-30b-a3b/report.json";
import viewJson from "../../fixtures/qwen3-30b-a3b/view.json";

const MOE = "data/main-verified-A2/qwen3-30b-a3b";
const DENSE = "data/main-verified-A2/qwen3-8b";
const DS = "data/main-verified-A2/deepseek-v3";
const NEXT = "data/main-verified-A2/qwen3-next-80b-a3b";
const BAD = "data/main-verified-H100/qwen3-30b-a3b";
const MIXED = "data/main-verified-H100/qwen3-8b";
const V4 = "data/main-verified-A2/deepseek-v4-flash";
const VLM = "data/main-verified-H100/deepseek-v3";

/**
 * Qwen3-8B with a VLM-style vision branch before the token embedding (the shape of the real
 * Llama-4 / MiniMax-M3 data): patch embedding (generic) → vision layers → projector (generic).
 */
function visionArch(): Arch {
  const a = denseArchJson as unknown as Arch;
  const stack = a.stages.find((x) => x.kind === "stack") as Extract<
    Arch["stages"][number],
    { kind: "stack" }
  >;
  const embed = a.stages[0] as ArchModuleStage;
  const generic = (id: string, cls: string, kind: "embedding" | "other") => ({
    ...embed,
    id,
    kind,
    label: kind === "embedding" ? "Token embedding" : id,
    component: { kind: "generic" as const, class_name: `vllm.x.${cls}`, children: [] },
    io: { inputs: [], internals: [], outputs: [] },
    module: { ...embed.module, path: id, class_name: `vllm.x.${cls}`, status: "generic" as const },
  });
  const vision = {
    ...stack,
    id: "vision_model.layers",
    label: "Vision layers",
    bands: stack.bands.map((b) => ({ ...b, id: "vit" })),
  };
  return {
    ...a,
    stages: [
      generic("vision_model.patch_embedding", "PatchEmbed", "embedding"),
      vision,
      generic("multi_modal_projector", "Projector", "other"),
      ...a.stages,
    ],
  } as Arch;
}

/** A non-periodic, heterogeneous stack (6 bands alternating dense / MoE with varying counts). */
function mixedArch(): Arch {
  const ds = deepseekJson as unknown as Arch;
  const stack = ds.stages.find((x) => x.kind === "stack") as Extract<
    Arch["stages"][number],
    { kind: "stack" }
  >;
  const [dense, moe] = stack.bands;
  let start = 0;
  const bands = [1, 2, 3, 4, 5, 6].map((c, i) => {
    const b = {
      ...(i % 2 ? moe : dense),
      id: `mix${i}`,
      count: c,
      pattern: null,
      layers: [start, start + c - 1],
    };
    start += c;
    return b;
  });
  return {
    ...ds,
    stages: ds.stages.map((x) => (x.kind === "stack" ? { ...x, bands } : x)),
  } as Arch;
}

beforeAll(() => {
  const index = indexJson as unknown as Index;
  window.__INFERATLAS__ = {
    index: {
      ...index,
      models: [
        ...index.models,
        {
          id: "deepseek-v3",
          hf_id: "deepseek-ai/DeepSeek-V3",
          label: "DeepSeek V3",
          family: "mla",
        },
        {
          id: "deepseek-v4-flash",
          hf_id: "deepseek-ai/DeepSeek-V4-Flash",
          label: "DeepSeek V4 Flash",
          family: "mla",
        },
        {
          id: "qwen3-next-80b-a3b",
          hf_id: "Qwen/Qwen3-Next",
          label: "Qwen3 Next",
          family: "hybrid",
        },
      ],
      cells: [
        ...index.cells,
        {
          version: "main-verified",
          model: "qwen3-30b-a3b",
          device: "H100",
          status: "ok",
          triage_count: 0,
          data_path: BAD,
        },
        {
          version: "main-verified",
          model: "qwen3-8b",
          device: "H100",
          status: "triage",
          triage_count: 1,
          data_path: MIXED,
        },
        {
          version: "main-verified",
          model: "deepseek-v4-flash",
          device: "A2",
          status: "triage",
          triage_count: 1,
          data_path: V4,
        },
        {
          version: "main-verified",
          model: "deepseek-v3",
          device: "H100",
          status: "ok",
          triage_count: 0,
          data_path: VLM,
        },
        // Structure-only triage cells: still linked and rendered (triage never blocks).
        {
          version: "main-verified",
          model: "deepseek-v3",
          device: "A2",
          status: "triage",
          triage_count: 2,
          data_path: DS,
        },
        {
          version: "main-verified",
          model: "qwen3-next-80b-a3b",
          device: "A2",
          status: "triage",
          triage_count: 1,
          data_path: NEXT,
        },
      ],
    },
    hardware: [],
    bundles: {
      [MOE]: {
        ir: irJson as unknown as IR,
        view: viewJson as unknown as View,
        report: reportJson as unknown as Report,
        arch: moeArchJson as unknown as Arch,
      },
      // A snapshot published before arch.json existed.
      [DENSE]: { ir: irJson as unknown as IR, view: viewJson as unknown as View },
      [DS]: { arch: deepseekJson as unknown as Arch },
      [NEXT]: { arch: nextJson as unknown as Arch },
      // Malformed arch.json next to a good numeric bundle.
      [BAD]: {
        ir: irJson as unknown as IR,
        view: viewJson as unknown as View,
        arch: { schema_version: 1, oops: true } as unknown as Arch,
      },
      [MIXED]: { arch: mixedArch() },
      [V4]: { arch: v4Json as unknown as Arch },
      [VLM]: { arch: visionArch() },
    },
  };
});

async function renderAt(path: string) {
  const { createAppRouter } = await import("@/router");
  const router = createAppRouter(createMemoryHistory({ initialEntries: [path] }));
  render(<RouterProvider router={router} />);
  return router;
}

describe("architecture page", { timeout: 20_000 }, () => {
  it("is the default tab and draws the whole-model figure", async () => {
    await renderAt("/m/main-verified/A2/qwen3-30b-a3b");
    const fig = await screen.findByTestId("arch-figure");
    expect(screen.getByTestId("tab-arch")).toHaveAttribute("data-state", "active");
    expect(within(fig).getByTestId("stage-model.embed_tokens")).toBeInTheDocument();
    expect(within(fig).getByTestId("band-model.layers:band:0")).toBeInTheDocument();
    expect(within(fig).getByText("128E · top-8")).toBeInTheDocument();
    expect(screen.getByTestId("arch-overview")).toBeInTheDocument();
  });

  it("expands a band into per-layer tensor I/O with evaluated shapes", async () => {
    const router = await renderAt("/m/main-verified/A2/qwen3-30b-a3b");
    const header = within(await screen.findByTestId("band-model.layers:band:0")).getByRole(
      "button",
      {
        name: /^layers 0–47/,
      },
    );
    expect(header).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(header);
    expect(router.state.location.search).toMatchObject({ layer: 0 });
    const detail = await screen.findByTestId("layer-detail");
    expect(within(detail).getByTestId("layer-title")).toHaveTextContent("Layer 0");
    const attn = within(detail).getByTestId("sub-card-self_attn");
    const qkv = within(attn)
      .getAllByTestId("param-row")
      .find((r) => r.textContent?.includes("qkv_proj.weight")) as HTMLElement;
    expect(within(qkv).getByTestId("shape-sym")).toHaveTextContent(
      "[(Hq/tp+2*max(1,Hkv/tp))*D, H]",
    );
    expect(within(qkv).getByTestId("shape-num")).toHaveTextContent("[5120, 2048]");
    expect(within(attn).getByTestId("source-link")).toHaveAttribute(
      "href",
      expect.stringContaining("qwen3_moe.py#L244"),
    );
  });

  it("resolves a layer index to its band and follows symbol overrides", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v3?layer=42&T=8192&tp=8");
    const detail = await screen.findByTestId("layer-detail");
    expect(screen.getByTestId("layer-band")).toHaveTextContent("layers 3–60");
    expect(screen.getByTestId("band-model.layers:band:3")).toHaveAttribute("data-selected", "true");
    const mlp = within(detail).getByTestId("sub-card-mlp");
    expect(within(mlp).getByTestId("module-path")).toHaveTextContent("model.layers.42.mlp");
    expect(within(mlp).getAllByTestId("shape-num")[0]).toHaveTextContent("[8192, 7168]");
    const w13 = within(mlp)
      .getAllByTestId("param-row")
      .find((r) => r.textContent?.includes("w13")) as HTMLElement;
    // [E, 4096/tp, 7168] with tp=8 from the URL.
    expect(within(w13).getByTestId("shape-num")).toHaveTextContent("[256, 512, 7168]");
  });

  it("layer selector jumps to any layer (pattern bands resolve to their period)", async () => {
    const router = await renderAt("/m/main-verified/A2/qwen3-next-80b-a3b");
    await screen.findByTestId("pattern-group");
    const input = screen.getByTestId("layer-input");
    await userEvent.type(input, "47{Enter}");
    expect(router.state.location.search).toMatchObject({ layer: 47 });
    expect(await screen.findByTestId("layer-title")).toHaveTextContent("Layer 47");
    // Layer 47 is a full-attention member; the period's full-attention band (layer 3) is drawn.
    expect(screen.getByTestId("band-model.layers:band:3")).toHaveAttribute("data-selected", "true");
    expect(
      within(screen.getByTestId("layer-detail")).getByTestId("sub-card-self_attn"),
    ).toBeTruthy();
  });

  it("shows notes and generic components without blocking the figure", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v4-flash");
    const head = await screen.findByTestId("stage-model.hc_head_op");
    expect(head).toHaveTextContent("HCHeadOp");
    expect(
      within(screen.getByTestId("arch-overview")).getByTestId("note-group-unknown_layer_class"),
    ).toHaveTextContent("No architecture rule describes");
    await userEvent.click(head);
    const card = await screen.findByTestId("stage-card-model.hc_head_op");
    expect(card).toHaveTextContent("generic");
    expect(card).toHaveTextContent("residual_streams");
    // Long qualified names wrap at dots (no mid-word breaks).
    expect(within(card).getByTestId("class-name").querySelectorAll("wbr").length).toBe(4);
  });

  it("structure-only cell: graph tab explains what is missing", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v3?tab=graph");
    expect(await screen.findByTestId("graph-empty")).toBeInTheDocument();
  });

  it("snapshot without arch.json: clear empty state, graph still one click away", async () => {
    await renderAt("/m/main-verified/A2/qwen3-8b");
    expect(await screen.findByTestId("arch-empty")).toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: "Open graph & analysis" }));
    expect(await screen.findByTestId("tab-graph")).toHaveAttribute("data-state", "active");
  });

  it("a malformed arch.json only affects the architecture tab", async () => {
    await renderAt("/m/main-verified/H100/qwen3-30b-a3b");
    expect(await screen.findByTestId("arch-error")).toHaveTextContent("arch.json is malformed");
    await userEvent.click(screen.getByRole("button", { name: "Open graph & analysis" }));
    expect(await screen.findByTestId("graph")).toBeInTheDocument();
  });

  it("a heterogeneous non-periodic stack uses the compact band list", async () => {
    const router = await renderAt("/m/main-verified/H100/qwen3-8b");
    const list = await screen.findByTestId("band-list");
    expect(within(list).getAllByRole("button")).toHaveLength(6);
    // Only one full block is drawn (the first band until one is selected).
    expect(screen.getAllByTestId(/^band-mix\d$/)).toHaveLength(1);
    await userEvent.click(screen.getByTestId("band-card-mix3"));
    expect(router.state.location.search).toMatchObject({ layer: 6 });
    expect(screen.getByTestId("band-card-mix3")).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByTestId("band-mix3")).toBeInTheDocument();
    expect(screen.getByTestId("layer-total")).toHaveTextContent("of 21 layers");
  });

  it("only runtime symbols are editable; model dimensions are read-only", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v3?H=1");
    await userEvent.click(await screen.findByTestId("symbols-toggle"));
    expect(screen.getByTestId("sym-T")).toBeInTheDocument();
    expect(screen.getByTestId("sym-tp")).toBeInTheDocument();
    expect(screen.queryByTestId("sym-H")).toBeNull();
    expect(screen.getByTestId("model-dims")).toHaveTextContent("7,168");
  });

  it("DeepSeek-V4: mHC blocks, compact stack, internals in figure and detail", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v4-flash");
    const list = await screen.findByTestId("band-list");
    expect(within(list).getAllByRole("button")).toHaveLength(3);
    expect(screen.getAllByTestId(/^band-model\.layers:band:[34]$/)).toHaveLength(2);
    expect(document.querySelectorAll("[data-slot=residual-bus]").length).toBe(2);
    // v1.3 §3: targeted mHC warnings badge the mHC box like any sublayer.
    const mhc = screen.getByTestId("sub-model.layers:band:3-mhc_pre_attn");
    expect(mhc.querySelector("[data-slot=note-badge]")).toHaveAttribute("data-severity", "warn");
    // Single-layer members of a pattern group show no "1×".
    expect(
      within(screen.getByTestId("band-model.layers:band:3")).queryByText("repeated 1 times"),
    ).toBeNull();
    // Array order: the HC head merges the streams before the final norm.
    const fig = screen.getByTestId("arch-figure");
    const order = ["stage-model.hc_head_op", "stage-model.norm"].map((id) =>
      [...fig.querySelectorAll("[data-testid]")].findIndex(
        (e) => e.getAttribute("data-testid") === id,
      ),
    );
    expect(order[0]).toBeLessThan(order[1] as number);
    expect(document.querySelectorAll("[data-slot=tensor-chip]")).toHaveLength(0);
    await userEvent.click(screen.getByTestId("internals-toggle"));
    expect(document.querySelectorAll("[data-slot=tensor-chip]").length).toBeGreaterThan(0);
    await userEvent.click(screen.getByTestId("sub-model.layers:band:4-attn"));
    const card = await screen.findByTestId("sub-card-attn");
    const internals = within(card).getByTestId("internals");
    expect(internals).toHaveTextContent("index_scores");
    expect(internals).toHaveTextContent("[1024, 512]"); // kv_compressed = [cdiv(T, 4), D]
    // Outputs are only what leaves the sublayer.
    expect(within(card).getAllByTestId("tensor-row").length).toBeGreaterThan(2);
  });

  it("VLM: the vision branch merges into the token embedding, collapsed until opened", async () => {
    await renderAt("/m/main-verified/H100/deepseek-v3");
    const merge = await screen.findByTestId("figure-merge");
    // The token embedding is the merge node; the branch summarises the encoder chain.
    expect(within(merge).getByTestId("stage-model.embed_tokens")).toBeInTheDocument();
    expect(within(merge).getByTestId("encoder-input")).toHaveTextContent("image");
    const toggle = within(merge).getByRole("button", { name: /Vision encoder/ });
    expect(toggle).toHaveAttribute("aria-expanded", "false");
    expect(within(merge).getByTestId("encoder-branch")).toHaveTextContent(
      "PatchEmbed → 36× GQA attn · dense MLP → Projector",
    );
    expect(screen.queryByTestId("band-vit")).toBeNull();
    expect(screen.getByTestId("band-model.layers:band:0")).toBeInTheDocument();
    await userEvent.click(toggle);
    const inner = await screen.findByRole("button", { name: /Vision layers/ });
    await userEvent.click(inner);
    expect(await screen.findByTestId("band-vit")).toBeInTheDocument();
    // Overview: encoder stages are grouped and relabelled (not "Token embedding").
    const flow = within(screen.getByTestId("dataflow-encoder"));
    expect(flow.getByText("Vision encoder")).toBeInTheDocument();
    expect(flow.getByText("patch embedding")).toBeInTheDocument();
    expect(flow.getByText("multi modal projector")).toBeInTheDocument();
  });

  it("home groups models by family and filters by text and family", async () => {
    const router = await renderAt("/");
    await screen.findByTestId("matrix");
    expect(screen.getByTestId("family-moe")).toBeInTheDocument();
    expect(screen.getByTestId("family-mla")).toBeInTheDocument();
    // The model name is the row's primary link to its model page.
    expect(screen.getByTestId("model-link-qwen3-8b")).toHaveAttribute(
      "href",
      expect.stringContaining("/m/main-verified/A2/qwen3-8b"),
    );
    await userEvent.type(screen.getByTestId("model-filter"), "deepseek");
    expect(router.state.location.search).toMatchObject({ q: "deepseek" });
    expect(screen.queryByTestId("model-row-qwen3-8b")).toBeNull();
    expect(screen.getByTestId("model-row-deepseek-v3")).toBeInTheDocument();
    await userEvent.clear(screen.getByTestId("model-filter"));
    await userEvent.click(within(screen.getByTestId("family-filter")).getByText("Dense"));
    expect(screen.getByTestId("model-row-qwen3-8b")).toBeInTheDocument();
    expect(screen.queryByTestId("model-row-qwen3-30b-a3b")).toBeNull();
  });

  it("home links every cell with data, including triage-only structure cells", async () => {
    await renderAt("/");
    const cell = await screen.findByTestId("cell-main-verified-deepseek-v3");
    expect(cell.tagName).toBe("A");
  });
});
