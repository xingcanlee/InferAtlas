/** Architecture tab: default view, layer detail with evaluated shapes, empty states. */
import type { Index, IR, Report, View } from "@inferatlas/contracts";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it } from "vitest";
import type { Arch } from "@/lib/arch-types";
import deepseekJson from "../../fixtures/arch-ui/deepseek-v3/arch.json";
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
          id: "qwen3-next-80b-a3b",
          hf_id: "Qwen/Qwen3-Next",
          label: "Qwen3 Next",
          family: "hybrid",
        },
      ],
      cells: [
        ...index.cells,
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
    expect(within(fig).getByTestId("stage-embed")).toBeInTheDocument();
    expect(within(fig).getByTestId("band-moe")).toBeInTheDocument();
    expect(within(fig).getByText("128E · top-8")).toBeInTheDocument();
    expect(screen.getByTestId("arch-overview")).toBeInTheDocument();
  });

  it("expands a band into per-layer tensor I/O with evaluated shapes", async () => {
    const router = await renderAt("/m/main-verified/A2/qwen3-30b-a3b");
    const header = within(await screen.findByTestId("band-moe")).getByRole("button", {
      name: /^layers 0–47/,
    });
    expect(header).toHaveAttribute("aria-expanded", "false");
    await userEvent.click(header);
    expect(router.state.location.search).toMatchObject({ layer: 0 });
    const detail = await screen.findByTestId("layer-detail");
    expect(within(detail).getByTestId("layer-title")).toHaveTextContent("Layer 0");
    const attn = within(detail).getByTestId("sub-card-self_attn");
    const qkv = within(attn)
      .getAllByTestId("param-row")
      .find((r) => r.textContent?.includes("qkv_proj.weight")) as HTMLElement;
    expect(within(qkv).getByTestId("shape-sym")).toHaveTextContent("[(Hq+2*Hkv)*D/tp, H]");
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
    expect(screen.getByTestId("band-moe")).toHaveAttribute("data-selected", "true");
    const mlp = within(detail).getByTestId("sub-card-mlp");
    expect(within(mlp).getByTestId("module-path")).toHaveTextContent("model.layers.42.mlp");
    expect(within(mlp).getAllByTestId("shape-num")[0]).toHaveTextContent("[8192, 7168]");
    const w13 = within(mlp)
      .getAllByTestId("param-row")
      .find((r) => r.textContent?.includes("w13")) as HTMLElement;
    expect(within(w13).getByTestId("shape-num")).toHaveTextContent("[256, 512, 7168]");
  });

  it("layer selector jumps to any layer (pattern bands resolve to their period)", async () => {
    const router = await renderAt("/m/main-verified/A2/qwen3-next-80b-a3b");
    await screen.findByTestId("pattern-group");
    const input = screen.getByTestId("layer-input");
    await userEvent.type(input, "47{Enter}");
    expect(router.state.location.search).toMatchObject({ layer: 47 });
    expect(await screen.findByTestId("layer-title")).toHaveTextContent("Layer 47");
    expect(screen.getByTestId("band-full_3")).toHaveAttribute("data-selected", "true");
    expect(
      within(screen.getByTestId("layer-detail")).getByTestId("sub-card-self_attn"),
    ).toBeTruthy();
  });

  it("shows notes and generic components without blocking the figure", async () => {
    await renderAt("/m/main-verified/A2/deepseek-v3");
    const mtp = await screen.findByTestId("stage-mtp");
    expect(mtp).toHaveTextContent("DeepSeekMultiTokenPredictorLayer");
    expect(within(screen.getByTestId("arch-overview")).getByTestId("note-list")).toHaveTextContent(
      "no lowering rule",
    );
    await userEvent.click(mtp);
    expect(await screen.findByTestId("stage-card-mtp")).toHaveTextContent("generic");
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

  it("home links every cell with data, including triage-only structure cells", async () => {
    await renderAt("/");
    const cell = await screen.findByTestId("cell-main-verified-deepseek-v3");
    expect(cell.tagName).toBe("A");
  });
});
