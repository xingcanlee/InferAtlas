import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Locator, type Page, test } from "@playwright/test";

/**
 * Phones and tablets (projects iphone-13 390×844, pixel-7 360×800, ipad 768×1024; Chromium with
 * touch + mobile viewport). Every page fits the viewport width (no sideways page scroll), the
 * architecture page pans/zooms with touch and thumb-size controls, its detail panel is a bottom
 * sheet, and interactive targets are ≥ 40px. Runs on the demo build (`pnpm e2e:demo`).
 */
const demo = process.env.INFERATLAS_DEMO === "1";
test.skip(!demo, "mobile checks run on the demo build");

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const MODELS = ["deepseek-v4-flash", "qwen3-next-80b-a3b", "qwen3-30b-a3b", "gpt-oss-120b"];
const url = (model: string, q = "") => `./#/m/main-verified/A2/${model}${q}`;

function watchErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

/** No horizontal page overflow: the document never scrolls sideways. */
async function noOverflow(page: Page, where: string) {
  const o = await page.evaluate(() => ({
    sw: (document.scrollingElement ?? document.documentElement).scrollWidth,
    iw: innerWidth,
  }));
  expect(o.sw, `${where}: scrollWidth ${o.sw} > innerWidth ${o.iw}`).toBeLessThanOrEqual(o.iw);
}

/** Touch target: at least 40×40 CSS px. */
async function tappable(l: Locator, name: string) {
  const box = await l.boundingBox();
  expect(box, `${name} is visible`).not.toBeNull();
  expect(box?.width ?? 0, `${name} width`).toBeGreaterThanOrEqual(40);
  expect(box?.height ?? 0, `${name} height`).toBeGreaterThanOrEqual(40);
}

const scaleOf = async (page: Page) =>
  Number(await page.locator("[data-scale]").first().getAttribute("data-scale"));

/** Two-finger pinch on the canvas, synthesised as touch events (fingers move apart by `by`). */
async function pinch(page: Page, by: number) {
  await page.getByTestId("arch-canvas").evaluate((el, d) => {
    const r = el.getBoundingClientRect();
    const cx = r.left + r.width / 2;
    const cy = r.top + r.height / 2;
    const t = (id: number, x: number) =>
      new Touch({ identifier: id, target: el, clientX: x, clientY: cy });
    const fire = (type: string, touches: Touch[]) =>
      el.dispatchEvent(
        new TouchEvent(type, { touches, changedTouches: touches, bubbles: true, cancelable: true }),
      );
    fire("touchstart", [t(1, cx - 40), t(2, cx + 40)]);
    for (let k = 1; k <= 4; k++)
      fire("touchmove", [t(1, cx - 40 - (k * d) / 8), t(2, cx + 40 + (k * d) / 8)]);
    fire("touchend", []);
  }, by);
}

test("home: readable without sideways scrolling; every cell with data links", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto("./");
  const phone = (page.viewportSize()?.width ?? 0) < 768;
  await expect(page.getByTestId(phone ? "matrix-cards" : "matrix")).toBeVisible();
  await noOverflow(page, "home");
  for (const model of MODELS) {
    const cell = page.getByTestId(`cell-main-verified-${model}`);
    await expect(cell).toHaveAttribute("href", new RegExp(`/m/main-verified/A2/${model}`));
    if (phone) await tappable(cell, `cell ${model}`);
  }
  // Filter chips and the device switch stay within the viewport and are thumb-size on phones.
  for (const id of ["family-filter", "device-switch"]) {
    const box = await page.getByTestId(id).boundingBox();
    expect((box?.x ?? 0) + (box?.width ?? 0)).toBeLessThanOrEqual(page.viewportSize()?.width ?? 0);
    if (phone) await tappable(page.getByTestId(id).getByRole("radio").first(), id);
  }
  if (phone) {
    // The nav collapses into a menu.
    await tappable(page.getByTestId("nav-menu"), "nav menu");
    await page.getByTestId("nav-menu").click();
    await page.getByRole("menuitem", { name: "Diff" }).click();
    await expect(page).toHaveURL(/#\/diff/);
    await noOverflow(page, "diff");
  }
  await page.goto("./");
  await page.getByTestId("model-link-deepseek-v4-flash").click();
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  expect(errors).toEqual([]);
});

for (const model of MODELS) {
  test(`${model}: fits the width, pinch / buttons zoom, no overflow`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.goto(url(model));
    const fig = page.getByTestId("arch-figure");
    await expect(fig).toBeVisible();
    await expect(page.locator('[data-layout="narrow"]')).toBeVisible();
    await noOverflow(page, model);
    // Fit to width by default: the figure's width is inside the canvas.
    const canvas = await page.getByTestId("arch-canvas").boundingBox();
    await expect
      .poll(async () => (await fig.boundingBox())?.width ?? 1e9)
      .toBeLessThanOrEqual((canvas?.width ?? 0) + 1);
    // Thumb-size controls.
    for (const id of ["zoom-in", "zoom-out", "zoom-fit", "open-details", "arch-menu"])
      await tappable(page.getByTestId(id), id);
    await tappable(page.getByRole("button", { name: "Next layer" }), "next layer");
    await tappable(page.getByTestId("tab-graph"), "graph tab");
    // Buttons zoom in and back to fit.
    const fitScale = await scaleOf(page);
    await page.getByTestId("zoom-in").click();
    await expect.poll(() => scaleOf(page)).toBeGreaterThan(fitScale);
    await page.getByTestId("zoom-fit").click();
    await expect.poll(() => scaleOf(page)).toBe(fitScale);
    // Pinch out zooms in; the page itself never scrolls sideways.
    await pinch(page, 200);
    await expect.poll(() => scaleOf(page)).toBeGreaterThan(fitScale);
    await expect(page.getByTestId("arch-canvas")).not.toHaveAttribute("data-zoom", "fit");
    await noOverflow(page, `${model} zoomed`);
    expect(errors).toEqual([]);
  });
}

test("detail bottom sheet: open, select a layer, step, close", async ({ page }) => {
  const errors = watchErrors(page);
  await page.goto(url("deepseek-v4-flash"));
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  // Details button opens the overview.
  await page.getByTestId("open-details").click();
  const sheet = page.getByTestId("detail-sheet");
  await expect(sheet).toBeVisible();
  await expect(sheet.getByTestId("arch-overview")).toBeVisible();
  await noOverflow(page, "sheet overview");
  // A layer from the overview's dataflow list → layer detail inside the sheet.
  await sheet.getByRole("button", { name: /^layers 3–42/ }).click();
  await expect(sheet.getByTestId("layer-detail")).toBeVisible();
  await expect(page).toHaveURL(/layer=3/);
  await tappable(sheet.getByTestId("sheet-next-layer"), "sheet next layer");
  await sheet.getByTestId("sheet-next-layer").click();
  await expect(page).toHaveURL(/layer=4/);
  // Shapes are readable without hover: symbolic = numeric in the sheet.
  await expect(sheet.getByTestId("shape-num").first()).toBeVisible();
  // Text in the sheet never drops below ~11px.
  const smallest = await sheet.evaluate((el) => {
    let min = 99;
    for (const n of el.querySelectorAll("*")) {
      const text = [...n.childNodes].some((c) => c.nodeType === 3 && c.textContent?.trim());
      if (!text || !(n as HTMLElement).offsetParent) continue;
      min = Math.min(min, Number.parseFloat(getComputedStyle(n).fontSize));
    }
    return min;
  });
  expect(smallest).toBeGreaterThanOrEqual(10.5);
  await page.getByRole("button", { name: "Close" }).click();
  await expect(sheet).toBeHidden();
  await expect(page).toHaveURL(/layer=4/);
  // Tapping a sublayer in the figure opens the sheet on that sublayer.
  await page.getByTestId("zoom-in").click();
  await page.locator('[data-testid$="-mhc_pre_ffn"]').first().click();
  await expect(sheet).toBeVisible();
  await expect(sheet.locator('[data-testid="sub-card-mhc_pre_ffn"]')).toHaveAttribute(
    "data-selected",
    "true",
  );
  // v1.4 §2: the class-level warning reaches the ffn mHC op.
  await expect(
    sheet.locator('[data-testid="sub-card-mhc_pre_ffn"] [data-severity="warn"]'),
  ).toHaveCount(1);
  await noOverflow(page, "sheet layer");
  expect(errors).toEqual([]);
});

test("graph tab: a clear note on phones, the workspace on tablets; never overflows", async ({
  page,
}) => {
  await page.goto(url("qwen3-30b-a3b", "?tab=graph"));
  const phone = (page.viewportSize()?.width ?? 0) < 768;
  if (phone) {
    await expect(page.getByTestId("graph-narrow")).toBeVisible();
    await noOverflow(page, "graph note");
    await page.getByTestId("graph-anyway").click();
  }
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await noOverflow(page, "graph workspace");
});

test("landscape phone and offline exports fit the width", async ({ page }) => {
  const { width, height } = page.viewportSize() ?? { width: 0, height: 0 };
  if (width < 768) await page.setViewportSize({ width: height, height: width });
  for (const target of [
    "./",
    url("deepseek-v4-flash"),
    url("qwen3-30b-a3b", "?tab=graph"),
    "./#/diff",
  ]) {
    await page.goto(target);
    await page.waitForLoadState("networkidle");
    await noOverflow(page, `${target} at ${height}×${width}`);
  }
  await page.setViewportSize({ width, height });
  for (const file of ["dist-single/index.html", "dist-hub/index.html"]) {
    const path = join(web, file);
    if (!existsSync(path)) continue;
    await page.goto(`file://${path}`);
    await page.waitForLoadState("load");
    await page.waitForTimeout(300);
    await noOverflow(page, file);
  }
});
