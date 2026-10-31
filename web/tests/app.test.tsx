import type { Index, IR, Report, View } from "@inferatlas/contracts";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it } from "vitest";
import indexJson from "../../fixtures/index.json";
import irJson from "../../fixtures/qwen3-30b-a3b/ir.json";
import reportJson from "../../fixtures/qwen3-30b-a3b/report.json";
import viewJson from "../../fixtures/qwen3-30b-a3b/view.json";

const path = "data/main-verified-A2/qwen3-30b-a3b";

beforeAll(() => {
  // Exercise the single-file data path: everything inlined on window.
  window.__INFERATLAS__ = {
    index: indexJson as unknown as Index,
    hardware: [],
    bundles: {
      [path]: {
        ir: irJson as unknown as IR,
        view: viewJson as unknown as View,
        report: reportJson as unknown as Report,
      },
    },
  };
});

async function renderAt(hash: string) {
  const { createAppRouter } = await import("@/router");
  const router = createAppRouter(createMemoryHistory({ initialEntries: [hash.slice(1)] }));
  render(<RouterProvider router={router} />);
  return router;
}

describe("app", { timeout: 20_000 }, () => {
  it("renders the matrix with status badges", async () => {
    await renderAt("#/");
    const table = await screen.findByTestId("matrix");
    expect(within(table).getAllByText("ok").length).toBeGreaterThan(0);
    expect(within(table).getByText("triage")).toBeInTheDocument();
    expect(within(table).getAllByText("missing")).toHaveLength(2);
  });

  it("model page shows details and recomputes on parameter change", async () => {
    const router = await renderAt(
      "#/m/main-verified/A2/qwen3-30b-a3b?view=L2%3Aattn&node=attn.qkv",
    );
    const detail = await screen.findByTestId("node-detail");
    expect(within(detail).getByText("QKV projection")).toBeInTheDocument();
    // tp=1: out q shape [4096, 32, 128]
    expect(within(detail).getAllByTestId("tensor-shape")[1]).toHaveTextContent("[4096, 32, 128]");
    await userEvent.click(within(screen.getByTestId("param-tp")).getByText("4"));
    expect(router.state.location.search).toMatchObject({ tp: 4 });
    expect(
      within(await screen.findByTestId("node-detail")).getAllByTestId("tensor-shape")[1],
    ).toHaveTextContent("[4096, 8, 128]");
  });

  it("shows the missing-cell error state", async () => {
    await renderAt("#/m/v0.32.0/A2/qwen3-8b");
    expect(await screen.findByText(/has no artifacts/)).toBeInTheDocument();
  });
});
