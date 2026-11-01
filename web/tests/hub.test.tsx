import type { Index } from "@inferatlas/contracts";
import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { HubPage } from "@/hub/HubPage";
import { exportHref, hubLinks, linkFor } from "@/lib/hub";
import indexJson from "../../fixtures/index.json";

describe("hub links", () => {
  it("maps each version/model/device to a plain relative sibling href", () => {
    const links = hubLinks({
      exports: [
        {
          version: "main-verified",
          model: "qwen3-8b",
          device: "A2",
          file: "inferatlas-qwen3-8b-a2.html",
        },
        {
          version: "main-verified",
          model: "qwen3-8b",
          device: "H100",
          file: "inferatlas-qwen3-8b-h100.html",
        },
      ],
    });
    expect(linkFor(links, { version: "main-verified", model: "qwen3-8b", device: "A2" })).toBe(
      "./inferatlas-qwen3-8b-a2.html",
    );
    expect(linkFor(links, { version: "main-verified", model: "qwen3-8b", device: "H100" })).toBe(
      "./inferatlas-qwen3-8b-h100.html",
    );
    expect(linkFor(links, { version: "v0.32.0", model: "qwen3-8b", device: "A2" })).toBeUndefined();
    expect(hubLinks(undefined).size).toBe(0);
  });

  it("only links bare .html file names beside the hub", () => {
    expect(exportHref("inferatlas-qwen3-30b-a3b-a2.html")).toBe(
      "./inferatlas-qwen3-30b-a3b-a2.html",
    );
    for (const bad of [
      "../x.html",
      "sub/x.html",
      "/abs.html",
      "https://e.com/x.html",
      "x.json",
      "a..html",
      "",
    ]) {
      expect(exportHref(bad), bad).toBeUndefined();
    }
  });
});

describe("hub page", () => {
  it("links exported cells to their sibling file and shows honest status elsewhere", () => {
    const index = structuredClone(indexJson) as unknown as Index;
    index.cells.push({
      version: "main-verified",
      model: "qwen3-30b-a3b",
      device: "H100",
      status: "triage",
      triage_count: 3,
      data_path: null,
    });
    render(
      <HubPage
        data={{
          index,
          hardware: [],
          bundles: {},
          hub: {
            exports: [
              {
                version: "main-verified",
                model: "qwen3-30b-a3b",
                device: "A2",
                file: "inferatlas-qwen3-30b-a3b-a2.html",
              },
            ],
          },
        }}
      />,
    );
    const ok = screen.getByTestId("hub-main-verified-qwen3-30b-a3b-A2");
    expect(ok.tagName).toBe("A");
    expect(ok).toHaveAttribute("href", "./inferatlas-qwen3-30b-a3b-a2.html");
    const triage = screen.getByTestId("hub-main-verified-qwen3-30b-a3b-H100");
    expect(triage.tagName).not.toBe("A");
    expect(within(triage).getByText(/3 unresolved triage items/)).toBeInTheDocument();
    // Has numeric data on the site but no offline export: not a link, says so.
    const dense = screen.getByTestId("hub-main-verified-qwen3-8b-A2");
    expect(dense.tagName).not.toBe("A");
    expect(dense).toHaveTextContent("No offline export");
    // A version pair with nothing generated collapses to one honest line.
    expect(screen.getByText("Nothing has been generated for this version pair yet.")).toBeVisible();
    expect(screen.queryByTestId("hub-v0.32.0-qwen3-8b-A2")).toBeNull();
    expect(screen.getAllByRole("link")).toHaveLength(1);
  });
});
