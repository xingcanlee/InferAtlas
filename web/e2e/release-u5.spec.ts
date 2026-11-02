import { existsSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { type Browser, expect, type Page, test } from "@playwright/test";

/**
 * Real-data checks of the contract v1.4 UI (S3 generator data) + U5 screenshots
 * (web/docs/screenshots/u5/): home coverage badges, DeepSeek-V4 stream expansion / merge, VLM
 * image inputs and labelled vision stages, config-only MTP asides, gpt-oss attention sinks.
 * Desktop 1440×900 (home, deepseek-v4-flash, llama-4-scout, minimax-m3, gpt-oss-120b) and phone
 * 390×844 (home, deepseek-v4-flash).
 *
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web build
 *   INFERATLAS_DATA_DIR=$PWD/.release-data pnpm --filter web exec playwright test release-u5
 */
const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const shots = join(web, "docs", "screenshots", "u5");
const enabled = process.env.INFERATLAS_RELEASE === "1" || !!process.env.INFERATLAS_DATA_DIR;
const indexFile = join(web, "dist", "data", "index.json");
const index = (
  enabled && existsSync(indexFile)
    ? JSON.parse(readFileSync(indexFile, "utf8"))
    : { cells: [], models: [], versions: [] }
) as {
  models: { id: string }[];
  versions: { id: string }[];
  cells: {
    model: string;
    device: string;
    version: string;
    status: string;
    data_path: string | null;
    coverage?: { notes_warn: number } | null;
  }[];
};

test.skip(!enabled, "Set INFERATLAS_RELEASE=1 or INFERATLAS_DATA_DIR for the real-data checks.");
test.describe.configure({ timeout: 120_000 });

const BASE = "http://127.0.0.1:4318/";
const url = (model: string, device = "A2") => `${BASE}#/m/main-verified/${device}/${model}`;

async function open(browser: Browser, size: { width: number; height: number }, touch: boolean) {
  const ctx = await browser.newContext({
    viewport: size,
    deviceScaleFactor: 1,
    hasTouch: touch,
    isMobile: touch,
    colorScheme: "light",
  });
  await ctx.addInitScript(() => {
    localStorage.setItem(
      "inferatlas-ui",
      JSON.stringify({ state: { theme: "light" }, version: 0 }),
    );
  });
  const page = await ctx.newPage();
  const errors: string[] = [];
  page.on("pageerror", (e) => errors.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") errors.push(`console: ${m.text()}`);
  });
  return { ctx, page, errors };
}

async function shoot(page: Page, name: string) {
  mkdirSync(shots, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(400);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}

/** Desktop fit: the whole figure inside the canvas at ≥ 0.9. */
async function fits(page: Page, where: string) {
  const fig = page.getByTestId("arch-figure");
  await expect(fig).toBeVisible();
  // Hash navigation keeps the document: measure the canvas inside the poll (fresh layout).
  await expect
    .poll(async () => {
      const canvas = await page.getByTestId("arch-canvas").boundingBox();
      const box = await fig.boundingBox();
      return (
        !!box &&
        !!canvas &&
        box.y >= canvas.y - 1 &&
        box.y + box.height <= canvas.y + canvas.height + 1 &&
        box.x >= canvas.x - 1 &&
        box.x + box.width <= canvas.x + canvas.width + 1
      );
    }, where)
    .toBe(true);
  const scale = Number(await page.locator("[data-scale]").first().getAttribute("data-scale"));
  expect(scale, where).toBeGreaterThanOrEqual(0.9);
}

test("index: one version, every built cell carries structure coverage", () => {
  expect(index.models).toHaveLength(16);
  expect(index.versions.map((v) => v.id)).toEqual(["main-verified"]);
  const built = index.cells.filter((c) => c.data_path);
  expect(built).toHaveLength(32);
  for (const c of built) expect(c.coverage, `${c.device}/${c.model}`).toBeTruthy();
});

test("desktop 1440×900: v1.4 structure on real data", async ({ browser }) => {
  const { ctx, page, errors } = await open(browser, { width: 1440, height: 900 }, false);

  // Home: coverage leads; warnings are counted; devices without built cells are not offered.
  await page.goto(BASE);
  const table = page.getByTestId("matrix");
  await expect(table).toBeVisible();
  await expect(page.getByTestId("cell-main-verified-qwen3-8b")).toContainText("structure ✓");
  await expect(page.getByTestId("cell-main-verified-deepseek-v4-flash")).toContainText(
    "168/336 ruled · 168 generic",
  );
  await expect(
    page.getByTestId("cell-main-verified-llama-4-scout").getByTestId("warn-count"),
  ).toHaveText(/\b3$/);
  const devices = page.getByTestId("device-switch");
  await expect(devices.getByText("A2", { exact: true })).toBeVisible();
  await expect(devices.getByText("310P")).toHaveCount(0);
  await shoot(page, "desktop-1440x900-home");

  // DeepSeek-V4: stream expansion before the stack, labelled merge, config-only MTP aside.
  await page.goto(url("deepseek-v4-flash"));
  await fits(page, "deepseek-v4-flash");
  const fig = page.getByTestId("arch-figure");
  await expect(fig.getByTestId("stage-model.layers:hc_expand")).toHaveText(
    /Stream expansion \(HC\)/,
  );
  await expect(fig.getByTestId("stage-model.hc_head_op")).toContainText("Stream merge (HC head)");
  await expect(fig.getByTestId("stage-mtp")).toContainText("config only");
  await shoot(page, "desktop-1440x900-deepseek-v4-flash");

  // VLMs: image input from `inputs`, labelled vision stages, fits.
  for (const model of ["llama-4-scout", "minimax-m3"]) {
    await page.goto(url(model));
    await fits(page, model);
    const input = page.getByTestId("encoder-input");
    await expect(input).toContainText("pixel_values");
    await expect(input).not.toHaveAttribute("data-muted");
    await expect(page.getByTestId("encoder-branch")).toContainText("Patch embedding");
    await expect(page.getByTestId("dataflow-encoder")).toContainText("Vision pre-norm");
    await shoot(page, `desktop-1440x900-${model}`);
  }
  await expect(page.getByTestId("stage-mtp")).toContainText("Multi-token prediction");

  // gpt-oss: attention sinks in the attention extras.
  await page.goto(url("gpt-oss-120b"));
  await fits(page, "gpt-oss-120b");
  await expect(page.locator('[data-component="attention"]').first()).toContainText("sinks");
  await shoot(page, "desktop-1440x900-gpt-oss-120b");

  // MTP asides wherever the config declares them (Kimi-K2 pins 0).
  for (const model of ["deepseek-v3", "deepseek-v3-2", "glm-4-5", "glm-4-6"]) {
    await page.goto(url(model));
    await fits(page, model);
    await expect(page.getByTestId("stage-mtp"), model).toContainText("config only");
  }
  await page.goto(url("kimi-k2"));
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  await expect(page.getByTestId("stage-mtp")).toHaveCount(0);

  expect(errors).toEqual([]);
  await ctx.close();
});

test("phone 390×844: home and DeepSeek-V4", async ({ browser }) => {
  const { ctx, page, errors } = await open(browser, { width: 390, height: 844 }, true);
  await page.goto(BASE);
  await expect(page.getByTestId("matrix-cards")).toBeVisible();
  await shoot(page, "390x844-home");
  await page.goto(url("deepseek-v4-flash"));
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  await expect(page.getByTestId("stage-model.layers:hc_expand")).toBeVisible();
  const o = await page.evaluate(() => ({
    sw: (document.scrollingElement ?? document.documentElement).scrollWidth,
    iw: innerWidth,
  }));
  expect(o.sw).toBeLessThanOrEqual(o.iw);
  await shoot(page, "390x844-deepseek-v4-flash");
  expect(errors).toEqual([]);
  await ctx.close();
});
