/** An index with no published cells (e.g. every version pair omitted, v1.4 §1) renders a clear empty state. */
import type { Index } from "@inferatlas/contracts";
import { createMemoryHistory, RouterProvider } from "@tanstack/react-router";
import { render, screen } from "@testing-library/react";
import { beforeAll, describe, expect, it } from "vitest";

beforeAll(() => {
  window.__INFERATLAS__ = {
    index: { models: [], versions: [], cells: [] } as unknown as Index,
    hardware: [],
    bundles: {},
  };
});

describe("empty index", { timeout: 20_000 }, () => {
  it("home says nothing is published instead of an empty matrix", async () => {
    const { createAppRouter } = await import("@/router");
    const router = createAppRouter(createMemoryHistory({ initialEntries: ["/"] }));
    render(<RouterProvider router={router} />);
    expect(await screen.findByText("No snapshots published yet")).toBeInTheDocument();
    expect(screen.queryByTestId("matrix")).toBeNull();
    expect(screen.queryByTestId("device-switch")).toBeNull();
  });
});
