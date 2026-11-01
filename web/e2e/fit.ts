import type { Page } from "@playwright/test";

/** Nodes outside the React Flow canvas box, plus the camera zoom. */
export async function graphFit(page: Page, testId = "graph") {
  return page.getByTestId(testId).evaluate((graph) => {
    const canvas = graph.querySelector(".react-flow");
    const viewport = graph.querySelector<HTMLElement>(".react-flow__viewport");
    if (!canvas || !viewport) throw new Error("no React Flow canvas");
    const box = canvas.getBoundingClientRect();
    const nodes = [...graph.querySelectorAll<HTMLElement>(".react-flow__node")];
    const outside = nodes
      .filter((node) => {
        const r = node.getBoundingClientRect();
        return (
          r.left < box.left - 0.5 ||
          r.top < box.top - 0.5 ||
          r.right > box.right + 0.5 ||
          r.bottom > box.bottom + 0.5
        );
      })
      .map((node) => node.dataset.id);
    return {
      nodes: nodes.length,
      outside,
      zoom: new DOMMatrix(getComputedStyle(viewport).transform).a,
      canvas: { width: Math.round(box.width), height: Math.round(box.height) },
    };
  });
}
