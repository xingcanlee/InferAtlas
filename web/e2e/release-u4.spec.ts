import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Browser, expect, type Page, test } from "@playwright/test";

/**
 * Real-data responsive sweep + U4 screenshots (web/docs/screenshots/u4/): every built cell at
 * 390×844 fits the width (no sideways page scroll) without errors; phone (390×844) and tablet
 * (768×1024) screenshots of home, DeepSeek-V4-Flash, Qwen3-Next, Llama-4-Scout and the detail
 * sheet (light, plus dark ones); desktop 1440×900 / 1280×800 captures to confirm the desktop
 * layout did not regress.
 *
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web build
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web exec playwright test release-u4
 */
const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const shots = join(web, "docs", "screenshots", "u4");
const enabled = process.env.INFERATLAS_RELEASE === "1" || !!process.env.INFERATLAS_DATA_DIR;
const indexFile = join(web, "dist", "data", "index.json");
const index = (
  enabled && existsSync(indexFile) ? JSON.parse(readFileSync(indexFile, "utf8")) : { cells: [] }
) as {
  cells: { model: string; device: string; version: string; data_path: string | null }[];
};
const cells = index.cells.filter((c) => c.data_path);

test.skip(!enabled, "Set INFERATLAS_RELEASE=1 or INFERATLAS_DATA_DIR for the real-data sweep.");
test.describe.configure({ timeout: 120_000 });

const BASE = "http://127.0.0.1:4318/";
const MODELS = ["deepseek-v4-flash", "qwen3-next-80b-a3b", "llama-4-scout"];
const url = (model: string, q = "") => `${BASE}#/m/main-verified/A2/${model}${q}`;

async function open(
  browser: Browser,
  size: { width: number; height: number },
  theme: "light" | "dark",
  touch = true,
) {
  const ctx = await browser.newContext({
    viewport: size,
    deviceScaleFactor: 1,
    hasTouch: touch,
    isMobile: touch,
    colorScheme: theme,
  });
  await ctx.addInitScript((t) => {
    localStorage.setItem("inferatlas-ui", JSON.stringify({ state: { theme: t }, version: 0 }));
  }, theme);
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return { ctx, page, errors };
}

async function noOverflow(page: Page, where: string) {
  const o = await page.evaluate(() => ({
    sw: (document.scrollingElement ?? document.documentElement).scrollWidth,
    iw: innerWidth,
  }));
  expect(o.sw, `${where}: scrollWidth ${o.sw} > ${o.iw}`).toBeLessThanOrEqual(o.iw);
}

async function shoot(page: Page, name: string) {
  mkdirSync(shots, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}

test("every real cell fits a 390×844 phone", async ({ browser }) => {
  expect(cells.length).toBeGreaterThan(0);
  const { ctx, page, errors } = await open(browser, { width: 390, height: 844 }, "light");
  for (const c of cells) {
    await page.goto(`${BASE}#/m/${c.version}/${c.device}/${c.model}`);
    await expect(page.getByTestId("arch-figure")).toBeVisible();
    await noOverflow(page, `${c.model} ${c.device}`);
  }
  expect(errors).toEqual([]);
  await ctx.close();
});

for (const size of [
  { width: 390, height: 844 },
  { width: 768, height: 1024 },
]) {
  const tag = `${size.width}x${size.height}`;
  for (const theme of ["light", "dark"] as const) {
    test(`${tag} ${theme}: home, models and detail sheet`, async ({ browser }) => {
      const { ctx, page, errors } = await open(browser, size, theme);
      await page.goto(BASE);
      await expect(page.getByTestId(size.width < 768 ? "matrix-cards" : "matrix")).toBeVisible();
      await noOverflow(page, "home");
      await shoot(page, `${tag}-home-${theme}`);
      for (const model of theme === "light" ? MODELS : ["deepseek-v4-flash"]) {
        await page.goto(url(model));
        await expect(page.getByTestId("arch-figure")).toBeVisible();
        await noOverflow(page, model);
        await shoot(page, `${tag}-${model}-${theme}`);
      }
      // Detail sheet on a DeepSeek-V4 sublayer (class-level mHC warning visible).
      await page.goto(url("deepseek-v4-flash", "?layer=5&sub=mhc_pre_ffn"));
      const sheet = page.getByTestId("detail-sheet");
      await expect(sheet).toBeVisible();
      await expect(sheet.getByTestId("sub-card-mhc_pre_ffn")).toHaveAttribute(
        "data-selected",
        "true",
      );
      await noOverflow(page, "sheet");
      await shoot(page, `${tag}-sheet-${theme}`);
      if (theme === "light") {
        // Llama-4 vision encoder opened, image terminal from arch inputs or muted.
        await page.goto(url("llama-4-scout"));
        await page
          .getByTestId("encoder-branch")
          .locator("[data-slot=stack-toggle]")
          .first()
          .click();
        await expect(page.getByTestId("encoder-input")).toBeVisible();
        await shoot(page, `${tag}-llama-4-scout-encoder-${theme}`);
      }
      expect(errors).toEqual([]);
      await ctx.close();
    });
  }
}

for (const size of [
  { width: 1440, height: 900 },
  { width: 1280, height: 800 },
]) {
  const tag = `${size.width}x${size.height}`;
  test(`desktop ${tag}: layout unchanged (side panel, fit)`, async ({ browser }) => {
    const { ctx, page, errors } = await open(browser, size, "light", false);
    await page.goto(BASE);
    await expect(page.getByTestId("matrix")).toBeVisible();
    await shoot(page, `desktop-${tag}-home`);
    for (const model of MODELS) {
      await page.goto(url(model));
      await expect(page.locator('[data-layout="wide"]')).toBeVisible();
      await expect(page.getByRole("complementary", { name: "Architecture details" })).toBeVisible();
      await expect(page.getByTestId("zoom-controls")).toHaveCount(0);
      await shoot(page, `desktop-${tag}-${model}`);
    }
    expect(errors).toEqual([]);
    await ctx.close();
  });
}
