import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

/**
 * Architecture tab (structure first) on the demo fixtures (fixtures/arch-ui). Release data has
 * no arch.json yet, so these run in demo mode only. Screenshots of every fixture figure, light
 * and dark at 1440×900, are written to web/docs/screenshots/u1/.
 */
const demo = process.env.INFERATLAS_DEMO === "1";
test.skip(!demo, "architecture fixtures are demo data");

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "docs", "screenshots", "u1");
const MODELS = ["qwen3-8b", "qwen3-30b-a3b", "deepseek-v3", "qwen3-next-80b-a3b"] as const;
const url = (model: string, q = "") => `./#/m/main-verified/A2/${model}${q}`;

function watchErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(m.text());
  });
  return errors;
}

/**
 * The app persists its theme in localStorage ("inferatlas-ui") and ignores the media query once
 * set, so emulateMedia alone is not enough: seed the stored theme and assert it took effect.
 */
async function applyStoredTheme(page: Page, theme: "light" | "dark") {
  await page.evaluate((t) => {
    localStorage.setItem("inferatlas-ui", JSON.stringify({ state: { theme: t }, version: 0 }));
  }, theme);
  await page.emulateMedia({ colorScheme: theme });
  await page.reload();
  if (theme === "dark") await expect(page.locator("html")).toHaveClass(/\bdark\b/);
  else await expect(page.locator("html")).not.toHaveClass(/\bdark\b/);
}

async function settle(page: Page) {
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(250);
}

test("home links structure-only triage cells", async ({ page }) => {
  await page.goto("./");
  for (const model of MODELS) {
    const cell = page.getByTestId(`cell-main-verified-${model}`);
    await expect(cell).toHaveAttribute("href", new RegExp(`/m/main-verified/A2/${model}`));
  }
  await page.getByTestId("cell-main-verified-deepseek-v3").click();
  await expect(page.getByTestId("arch-figure")).toBeVisible();
});

for (const model of MODELS) {
  test(`${model}: figure fits and screenshots (light + dark)`, async ({ page }) => {
    const errors = watchErrors(page);
    mkdirSync(shots, { recursive: true });
    await page.setViewportSize({ width: 1440, height: 900 });
    await page.goto(url(model));
    for (const scheme of ["light", "dark"] as const) {
      await applyStoredTheme(page, scheme);
      const fig = page.getByTestId("arch-figure");
      await expect(fig).toBeVisible();
      await expect(page.getByTestId("tab-arch")).toHaveAttribute("data-state", "active");
      // Fit mode: the whole figure is inside the canvas at a readable scale.
      const canvas = await page.getByTestId("arch-canvas").boundingBox();
      await expect
        .poll(async () => {
          const box = await fig.boundingBox();
          return (
            !!box &&
            !!canvas &&
            box.y >= canvas.y - 1 &&
            box.y + box.height <= canvas.y + canvas.height + 1 &&
            box.x >= canvas.x - 1 &&
            box.x + box.width <= canvas.x + canvas.width + 1
          );
        })
        .toBe(true);
      const scale = Number(await page.locator("[data-scale]").first().getAttribute("data-scale"));
      // FitBox floor: smallest figure text (10px) stays ≥ 9px.
      expect(scale).toBeGreaterThanOrEqual(0.9);
      await settle(page);
      await page.screenshot({ path: join(shots, `${model}-${scheme}.png`) });
    }
    expect(errors).toEqual([]);
  });
}

test("expand a band: per-layer tensor I/O with numeric shapes", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(url("deepseek-v3"));
  await applyStoredTheme(page, "light");
  const header = page.getByTestId("band-moe").getByRole("button", { name: /^layers 3–60/ });
  await expect(header).toHaveAttribute("aria-expanded", "false");
  await header.click();
  await expect(header).toHaveAttribute("aria-expanded", "true");
  await expect(page).toHaveURL(/layer=3/);
  const detail = page.getByTestId("layer-detail");
  await expect(detail.getByTestId("layer-title")).toHaveText("Layer 3");
  const mlp = detail.getByTestId("sub-card-mlp");
  await expect(mlp).toContainText("MoE 256E top-8 + 1 shared");
  await expect(mlp.getByTestId("tensor-row").first()).toContainText("[T, H]");
  await expect(mlp.getByTestId("tensor-row").first()).toContainText("[4096, 7168]");
  await expect(mlp).toContainText("AscendSharedFusedMoE");
  // Clicking a sublayer in the figure selects its card.
  await page.getByTestId("sub-moe-self_attn").click();
  await expect(detail.getByTestId("sub-card-self_attn")).toHaveAttribute("data-selected", "true");
  await expect(detail.getByTestId("sub-card-self_attn")).toContainText("[4096, 7168]");
  await settle(page);
  await page.screenshot({ path: join(shots, "deepseek-v3-layer3-light.png") });

  // Symbols: T=8192 flows into every evaluated shape.
  await page.getByTestId("symbols-toggle").click();
  await page.getByTestId("sym-T").fill("8192");
  await page.getByTestId("sym-T").press("Enter");
  await expect(page).toHaveURL(/T=8192/);
  await expect(detail.getByTestId("sub-card-self_attn")).toContainText("[8192, 7168]");

  // Esc returns to the overview.
  await page.locator("body").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("arch-overview")).toBeVisible();
});

test("layer selector resolves any layer to its band", async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto(url("qwen3-next-80b-a3b"));
  await expect(page.getByTestId("pattern-group")).toBeVisible();
  const input = page.getByTestId("layer-input");
  await input.fill("23");
  await input.press("Enter");
  await expect(page).toHaveURL(/layer=23/);
  await expect(page.getByTestId("layer-band")).toContainText("layer 23");
  await expect(page.getByTestId("layer-total")).toHaveText("of 48 layers");
  await expect(page.getByTestId("band-full_3")).toHaveAttribute("data-selected", "true");
  await expect(page.getByTestId("layer-detail")).toContainText("model.layers.23.self_attn");
  // ArrowDown steps to the previous layer: a linear-attention band.
  await input.press("ArrowDown");
  await expect(page).toHaveURL(/layer=22/);
  await expect(page.getByTestId("band-linear_0_2")).toHaveAttribute("data-selected", "true");
  await expect(page.getByTestId("layer-detail")).toContainText("Gated DeltaNet");
  await expect(page.getByTestId("layer-detail")).toContainText("model.layers.22.linear_attn");
  await applyStoredTheme(page, "dark");
  await expect(page.getByTestId("layer-detail")).toContainText("model.layers.22.linear_attn");
  await settle(page);
  await page.screenshot({ path: join(shots, "qwen3-next-layer22-dark.png") });
});

test("shape labels reveal numeric values on hover and focus", async ({ page }) => {
  await page.goto(url("qwen3-30b-a3b"));
  const edge = page.locator('[data-slot="tensor-edge"] [data-slot="shape-label"]').first();
  await expect(edge).toHaveAttribute("data-numeric", "[4096, 2048]");
  await edge.hover();
  await expect(page.getByRole("tooltip").first()).toHaveText("[4096, 2048]");
  await edge.click();
  await expect(edge).toHaveAttribute("aria-pressed", "true");
  await expect(edge.getByTestId("shape-text")).toHaveText("[4096, 2048]");
});

test("snapshot without arch.json shows a clear empty state", async ({ page }) => {
  await page.goto("./#/m/fixture-prev/A2/qwen3-30b-a3b");
  await expect(page.getByTestId("arch-empty")).toBeVisible();
  await page.getByRole("button", { name: "Open graph & analysis" }).click();
  await expect(page.getByTestId("node-decoder")).toBeVisible();
});

test("graph tab stays available beside the architecture", async ({ page }) => {
  await page.goto(url("qwen3-30b-a3b"));
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  await page.getByTestId("tab-graph").click();
  await expect(page).toHaveURL(/tab=graph/);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await page.getByTestId("tab-arch").click();
  await expect(page.getByTestId("arch-figure")).toBeVisible();
});
