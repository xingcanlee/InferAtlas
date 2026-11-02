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
    index: {
      ...(indexJson as unknown as Index),
      // A triage cell without a numeric bundle, like the release's blocked DeepSeek cells.
      cells: [
        // v1.4 §1: one cell carries structure coverage, the other relies on the neutral badge.
        ...(indexJson as unknown as Index).cells.map((c) =>
          c.model === "qwen3-8b" && c.data_path
            ? {
                ...c,
                coverage: {
                  sublayers_ruled: 4,
                  sublayers_generic: 0,
                  notes_warn: 2,
                  notes_info: 1,
                  arch: true,
                },
              }
            : c,
        ),
        {
          version: "main-verified",
          model: "qwen3-8b",
          device: "H100",
          status: "triage",
          triage_count: 3,
          data_path: null,
        },
        // A device the pipeline lists but has not built anything for (release 310P / A3 / A5).
        {
          version: "main-verified",
          model: "qwen3-8b",
          device: "A5",
          status: "missing",
          data_path: null,
        },
      ],
    },
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
  it("leads the matrix with structure coverage; verification is secondary", async () => {
    await renderAt("#/");
    const table = await screen.findByTestId("matrix");
    // Coverage from the index (v1.4) …
    const covered = within(table).getByTestId("cell-main-verified-qwen3-8b");
    expect(covered).toHaveTextContent("structure ✓ · 4/4 ruled");
    expect(within(covered).getByTestId("warn-count")).toHaveTextContent("2");
    expect(within(covered).getByTestId("verification")).toHaveTextContent("unverified · 2");
    // … or a neutral badge when the index has none; never a "triage · N" badge.
    const plain = within(table).getByTestId("cell-main-verified-qwen3-30b-a3b");
    expect(within(plain).getByText("structure")).toBeInTheDocument();
    expect(within(plain).getByTestId("verification")).toHaveTextContent("verified");
    expect(within(table).queryByText("triage")).toBeNull();
    // v0.32.0 has no built cell on A2: no column.
    expect(within(table).queryByText("v0.32.0 pair")).toBeNull();
    expect(within(table).queryAllByText("missing")).toHaveLength(0);
    // Devices with only `missing` cells are not offered.
    const devices = within(screen.getByTestId("device-switch"));
    expect(devices.getByText("A2")).toBeInTheDocument();
    expect(devices.getByText("H100")).toBeInTheDocument();
    expect(devices.queryByText("A5")).toBeNull();
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
    expect(await screen.findByTestId("cell-unavailable")).toHaveTextContent(
      "qwen3-8b @ v0.32.0 on A2: No artifacts were generated for this combination.",
    );
    expect(screen.getByText("Missing: qwen3-8b")).toBeInTheDocument();
  });

  it("says triage, not missing, for a triage-only deep link", async () => {
    await renderAt("#/m/main-verified/H100/qwen3-8b");
    const msg = await screen.findByTestId("cell-unavailable");
    expect(msg).toHaveTextContent("Blocked by 3 unresolved triage items");
    expect(msg.textContent).not.toMatch(/missing/i);
    expect(screen.getByText("Needs triage: qwen3-8b")).toBeInTheDocument();
  });
});
