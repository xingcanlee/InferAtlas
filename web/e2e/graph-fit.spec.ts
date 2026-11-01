import { expect, type Page, test } from "@playwright/test";
import { graphFit } from "./fit";

/**
 * First load of every view, nothing selected, no Fit click: the whole graph must be on screen
 * at a readable zoom. Runs on whatever data the preview serves (demo fixtures or release data).
 */

const VIEWPORTS = [
  { width: 1280, height: 800 },
  { width: 1440, height: 900 },
  { width: 1600, height: 1000 },
  { width: 1920, height: 1080 },
] as const;
const MIN_ZOOM = 0.6;

interface Cell {
  version: string;
  model: string;
  device: string;
  status: string;
  data_path: string | null;
}

async function viewsToCheck(page: Page): Promise<{ route: string; view: string }[]> {
  const index = (await (await page.request.get("./data/index.json")).json()) as { cells: Cell[] };
  const cells = index.cells.filter((c) => c.status === "ok" && c.data_path && c.device === "A2");
  const seen = new Set<string>();
  const out: { route: string; view: string }[] = [];
  for (const c of cells) {
    if (seen.has(c.model) || c.version.startsWith("fixture-prev")) continue;
    seen.add(c.model);
    const res = await page.request.get(`./${c.data_path}/view.json`);
    // Structure-only cells (arch.json without ir/view) have no graph to fit.
    if (!res.ok() || !(res.headers()["content-type"] ?? "").includes("json")) continue;
    const view = (await res.json()) as { views: { id: string }[] };
    for (const v of view.views)
      out.push({ route: `#/m/${c.version}/${c.device}/${c.model}`, view: v.id });
  }
  return out;
}

for (const size of VIEWPORTS) {
  test(`every view fits on first load at ${size.width}×${size.height}`, async ({ page }) => {
    test.setTimeout(90_000);
    const errors: string[] = [];
    page.on("pageerror", (e) => errors.push(e.message));
    page.on("console", (m) => {
      if (m.type() === "error") errors.push(m.text());
    });
    await page.setViewportSize(size);
    await page.goto("./");
    const views = await viewsToCheck(page);
    expect(views.length).toBeGreaterThan(0);
    for (const { route, view } of views) {
      const q = view === "L0" ? "?tab=graph" : `?view=${encodeURIComponent(view)}`;
      await page.goto(`./${route}${q}`);
      await expect(page.locator('[data-testid="graph"] .react-flow__node').first()).toBeVisible();
      // Placement re-runs (debounced) when the canvas settles; poll until stable.
      await expect
        .poll(async () => (await graphFit(page)).outside, { message: `${route} ${view}` })
        .toEqual([]);
      const fit = await graphFit(page);
      expect(fit.nodes, `${route} ${view}`).toBeGreaterThan(0);
      expect(fit.zoom, `${route} ${view} zoom`).toBeGreaterThanOrEqual(MIN_ZOOM);
    }
    expect(errors).toEqual([]);
  });
}

test("analysis panel collapses, expands and remembers the split", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("./#/m/main-verified/A2/qwen3-30b-a3b?tab=graph");
  const toggle = page.getByTestId("analysis-toggle");
  const graphHeight = async () =>
    (await page.getByTestId("graph").boundingBox())?.height ?? Number.NaN;
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  const open = await graphHeight();
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await expect(page.getByTestId("memory-chart")).toHaveCount(0);
  await expect.poll(graphHeight).toBeGreaterThan(open + 100);
  // Whole graph still fits after the canvas grew (debounced re-fit).
  await expect.poll(async () => (await graphFit(page)).outside).toEqual([]);
  await page.reload();
  await expect(page.getByTestId("analysis-toggle")).toHaveAttribute("aria-expanded", "false");
  // Choosing a tab re-opens the panel; keyboard resize goes through the separator.
  await page.getByTestId("tab-kv").click();
  await expect(page.getByTestId("analysis-toggle")).toHaveAttribute("aria-expanded", "true");
  await expect(page.getByTestId("kv-chart")).toBeVisible();
  const before = await graphHeight();
  await page.getByTestId("analysis-resize").focus();
  await page.keyboard.press("ArrowUp");
  await page.keyboard.press("ArrowUp");
  await expect.poll(graphHeight).toBeLessThan(before);
});

test("shrinking the window re-fits the graph", async ({ page }) => {
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.goto("./#/m/main-verified/A2/qwen3-30b-a3b?view=L2%3Aattn");
  await expect(page.getByTestId("analysis-toggle")).toHaveAttribute("aria-expanded", "true");
  await page.setViewportSize({ width: 1280, height: 800 });
  // Without a saved split the analysis panel follows the window and collapses.
  await expect(page.getByTestId("analysis-toggle")).toHaveAttribute("aria-expanded", "false");
  await expect.poll(async () => (await graphFit(page)).outside).toEqual([]);
  expect((await graphFit(page)).zoom).toBeGreaterThanOrEqual(0.6);
});

test("analysis panel re-opens in the same session (button, tab, Enter on divider)", async ({
  page,
}) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("./#/m/main-verified/A2/qwen3-30b-a3b?tab=graph");
  const toggle = page.getByTestId("analysis-toggle");
  const analysisHeight = async () =>
    (await page.getByTestId("analysis").boundingBox())?.height ?? Number.NaN;
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  // Chevron: collapse then expand without a reload.
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect.poll(analysisHeight).toBeGreaterThanOrEqual(180);
  // Tab click re-opens a collapsed panel.
  await toggle.click();
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await page.getByTestId("tab-roofline").click();
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect.poll(analysisHeight).toBeGreaterThanOrEqual(180);
  // Enter on the focused divider toggles both ways.
  await page.getByTestId("analysis-resize").focus();
  await page.keyboard.press("Enter");
  await expect(toggle).toHaveAttribute("aria-expanded", "false");
  await page.keyboard.press("Enter");
  await expect(toggle).toHaveAttribute("aria-expanded", "true");
  await expect.poll(analysisHeight).toBeGreaterThanOrEqual(180);
});

test("view header items do not overlap at 1280×800", async ({ page }) => {
  await page.setViewportSize({ width: 1280, height: 800 });
  for (const q of ["?tab=graph", "?view=L1%3Adecoder", "?view=L2%3Aattn"]) {
    await page.goto(`./#/m/main-verified/A2/qwen3-30b-a3b${q}`);
    const band = await page.getByTestId("band-strip").boundingBox();
    const crumb = await page.getByTestId("current-view").boundingBox();
    const hint = page.getByTestId("view-hint");
    const boxes = [band, crumb, (await hint.isVisible()) ? await hint.boundingBox() : null].filter(
      (b): b is NonNullable<typeof b> => b !== null,
    );
    for (let i = 0; i < boxes.length; i++)
      for (let j = i + 1; j < boxes.length; j++) {
        const [a, b] = [boxes[i], boxes[j]] as [(typeof boxes)[0], (typeof boxes)[0]];
        const overlap =
          a.x < b.x + b.width &&
          b.x < a.x + a.width &&
          a.y < b.y + b.height &&
          b.y < a.y + a.height;
        expect(overlap, `${q} header boxes ${i}/${j} overlap`).toBe(false);
      }
  }
});

test("a user zoom survives a small window resize", async ({ page }) => {
  await page.setViewportSize({ width: 1600, height: 1000 });
  await page.goto("./#/m/main-verified/A2/qwen3-30b-a3b?tab=graph");
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await page.locator(".react-flow__controls-zoomin").click();
  await page.locator(".react-flow__controls-zoomin").click();
  await page.waitForTimeout(400);
  const zoomed = (await graphFit(page)).zoom;
  await page.setViewportSize({ width: 1590, height: 990 });
  await page.waitForTimeout(500);
  expect((await graphFit(page)).zoom).toBeCloseTo(zoomed, 3);
  // Fit view hands the camera back to auto placement.
  await page.locator(".react-flow__controls-fitview").click();
  await page.waitForTimeout(400);
  await expect.poll(async () => (await graphFit(page)).outside).toEqual([]);
});
