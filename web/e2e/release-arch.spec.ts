import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

/**
 * Real-data architecture sweep (structure first): every cell of the served release data with a
 * `data_path` opens on the Architecture tab, draws the figure without page/console errors, fits
 * a 1440×900 window at a readable scale and shows per-layer tensor I/O. Screenshots of every A2
 * model (light) and a few dark ones go to web/docs/screenshots/u3/.
 *
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web build
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web exec playwright test release-arch
 *
 * Runs when INFERATLAS_RELEASE=1 or INFERATLAS_DATA_DIR is set (the served `dist/` must be a
 * release build of that data); the cells are read from dist/data/index.json.
 */
const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const shots = join(web, "docs", "screenshots", "u3");
const enabled = process.env.INFERATLAS_RELEASE === "1" || !!process.env.INFERATLAS_DATA_DIR;
const indexFile = join(web, "dist", "data", "index.json");

interface Cell {
  model: string;
  device: string;
  version: string;
  status: string;
  data_path: string | null;
}
interface IndexFile {
  cells: Cell[];
  models: { id: string; family: string; label: string }[];
}

const index: IndexFile | undefined =
  enabled && existsSync(indexFile)
    ? (JSON.parse(readFileSync(indexFile, "utf8")) as IndexFile)
    : undefined;
const cells = (index?.cells ?? []).filter((c) => c.data_path);
const DARK = new Set(["deepseek-v4-flash", "llama-4-scout", "qwen3-next-80b-a3b", "gpt-oss-120b"]);

test.skip(!enabled, "Set INFERATLAS_RELEASE=1 or INFERATLAS_DATA_DIR for the real-data sweep.");

function watchErrors(page: Page) {
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return errors;
}

async function applyStoredTheme(page: Page, theme: "light" | "dark") {
  await page.evaluate((t) => {
    localStorage.setItem("inferatlas-ui", JSON.stringify({ state: { theme: t }, version: 0 }));
  }, theme);
  await page.emulateMedia({ colorScheme: theme });
  await page.reload();
  if (theme === "dark") await expect(page.locator("html")).toHaveClass(/\bdark\b/);
  else await expect(page.locator("html")).not.toHaveClass(/\bdark\b/);
}

async function shoot(page: Page, name: string) {
  mkdirSync(shots, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(250);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}

test("index lists real cells", () => {
  expect(index, `${indexFile} (build the site with release data first)`).toBeDefined();
  expect(cells.length).toBeGreaterThan(0);
});

for (const cell of cells) {
  const { model, device, version } = cell;
  test(`${model} ${device}: architecture renders from real data`, async ({ page }) => {
    const errors = watchErrors(page);
    await page.setViewportSize({ width: 1440, height: 900 });
    const url = `./#/m/${version}/${device}/${model}`;
    await page.goto(url);
    await applyStoredTheme(page, "light");
    const fig = page.getByTestId("arch-figure");
    await expect(fig).toBeVisible();
    await expect(page.getByTestId("tab-arch")).toHaveAttribute("data-state", "active");
    await expect(page.getByTestId("arch-error")).toHaveCount(0);
    // Whole-model dataflow: input and output terminals, at least one decoder band.
    await expect(fig.locator("[data-slot=figure-terminal]").first()).toBeVisible();
    await expect(fig.locator("[data-slot=band]").first()).toBeVisible();
    // Fit: inside the canvas at a readable scale (FitBox floor 0.9).
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
    expect(scale).toBeGreaterThanOrEqual(0.9);
    if (device === "A2") {
      await shoot(page, `${model}-light`);
      if (DARK.has(model)) {
        await applyStoredTheme(page, "dark");
        await expect(fig).toBeVisible();
        await shoot(page, `${model}-dark`);
        await applyStoredTheme(page, "light");
      }
    }

    // Per-layer I/O: the middle decoder layer shows every sublayer's inputs/outputs.
    const total = Number(
      ((await page.getByTestId("layer-total").textContent()) ?? "").match(/\d+/)?.[0] ?? 0,
    );
    expect(total).toBeGreaterThan(0);
    const layer = total - 1 - Math.floor(total / 4);
    await page.goto(`${url}?layer=${layer}`);
    const detail = page.getByTestId("layer-detail");
    await expect(detail).toBeVisible();
    await expect(detail.locator("[data-testid^=sub-card-]").first()).toBeVisible();
    expect(await detail.getByTestId("tensor-row").count()).toBeGreaterThan(2);
    // The selected band is drawn and marked.
    await expect(page.locator("[data-slot=band][data-selected=true]")).toHaveCount(1);
    if (device === "A2" && model === "deepseek-v4-flash") await shoot(page, `${model}-layer`);

    // VLMs: the encoder branch merges into the token embedding and opens on demand.
    const encoder = page.getByTestId("encoder-branch");
    if (await encoder.count()) {
      await page.goto(url);
      await encoder.locator("[data-slot=stack-toggle]").first().click();
      await expect(encoder.locator("[data-slot=figure-node]").first()).toBeVisible();
      if (device === "A2") await shoot(page, `${model}-encoder`);
    }
    expect(errors).toEqual([]);
  });
}

test("home: every real model is grouped by family and links to its architecture", async ({
  page,
}) => {
  const errors = watchErrors(page);
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.goto("./");
  await applyStoredTheme(page, "light");
  const models = index?.models ?? [];
  for (const f of new Set(models.map((m) => m.family)))
    await expect(page.getByTestId(`family-${f}`)).toBeVisible();
  for (const m of models) {
    const hasData = cells.some((c) => c.model === m.id && c.device === "A2");
    const link = page.getByTestId(`model-link-${m.id}`);
    if (hasData) await expect(link).toHaveAttribute("href", new RegExp(`/m/[^/]+/A2/${m.id}$`));
  }
  await shoot(page, "home-light");
  await page.getByTestId("model-filter").fill("llama");
  await expect(page.locator("[data-testid^=model-row-]")).toHaveCount(
    models.filter((m) => /llama/i.test(`${m.id} ${m.label}`)).length,
  );
  await page.getByTestId("model-filter").fill("");
  await page.getByTestId("family-filter").getByText("Vision-language").click();
  await expect(page.locator("[data-testid^=model-row-]")).toHaveCount(
    models.filter((m) => m.family === "vlm").length,
  );
  await shoot(page, "home-vlm-filter");
  const vlm = models.find((m) => m.family === "vlm");
  if (vlm) {
    await page.getByTestId(`model-link-${vlm.id}`).click();
    await expect(page.getByTestId("arch-figure")).toBeVisible();
    await expect(page.getByTestId("figure-merge")).toBeVisible();
  }
  expect(errors).toEqual([]);
});
