import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "docs", "screenshots");
const MODEL = "#/m/main-verified/A2/qwen3-30b-a3b";

async function shot(page: Page, name: string) {
  await page.waitForTimeout(250);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}

test("home matrix lists cells and opens a model", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByTestId("matrix")).toBeVisible();
  await expect(page.locator('[data-status="ok"]').first()).toBeVisible();
  await expect(page.locator('[data-status="triage"]').first()).toBeVisible();
  await expect(page.locator('[data-status="missing"]').first()).toBeVisible();
  await shot(page, "home");
  await page.getByTestId("cell-main-verified-qwen3-30b-a3b").click();
  await expect(page.getByTestId("model-title")).toHaveText("Qwen/Qwen3-30B-A3B");
});

test("model page: drill down, select, recompute, overlay", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(`./${MODEL}`);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await expect(page.getByTestId("edge-label").first()).toBeVisible();
  await shot(page, "model-light");

  // drill L0 -> L1 -> L2
  await page.getByTestId("node-decoder").dblclick();
  await expect(page.getByTestId("current-view")).toContainText("L1");
  await page.getByTestId("node-attn").dblclick();
  await expect(page.getByTestId("current-view")).toContainText("L2");
  await page.getByTestId("node-attn.o").click();
  await expect(page.getByTestId("node-detail")).toContainText("RowParallelLinear");
  await expect(page).toHaveURL(/node=attn\.o/);

  // live recompute: tp=4 changes the shard size shown in detail panel
  const resident = page.getByTestId("node-detail").getByText("Resident on this rank").locator("..");
  await expect(resident).toContainText("16 MiB");
  await page.getByTestId("param-tp").getByText("4", { exact: true }).click();
  await expect(page).toHaveURL(/tp=4/);
  await expect(resident).toContainText("4 MiB");

  // Esc goes up a level
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("current-view")).toContainText("L1");

  // overlay badges + comparison card
  await page.getByTestId("overlay-toggle").click();
  await expect(page.getByTestId("ascend-badge").first()).toBeVisible();
  await page.getByTestId("node-norm.input").click();
  await expect(page.getByTestId("ascend-card")).toBeVisible();
  await shot(page, "model-overlay");

  // side views
  await page.getByTestId("tab-kv").click();
  await expect(page.getByTestId("kv-chart")).toBeVisible();
  await shot(page, "model-kv");
  await page.getByTestId("tab-ranks").click();
  await expect(page.getByTestId("rank-lanes")).toBeVisible();
  await shot(page, "model-ranks");
  await page.getByTestId("tab-roofline").click();
  await page.getByTestId("tab-memory").click();
  await expect(page.getByTestId("memory-chart")).toBeVisible();

  // node search
  await page.keyboard.press("/");
  await page.getByTestId("node-search-input").fill("dispatch");
  await page.keyboard.press("Enter");
  await expect(page).toHaveURL(/view=L2%3Amoe|view=L2:moe/);
  await expect(page.getByTestId("node-detail")).toContainText("all_to_all");
});

test("model page dark theme", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto(`./${MODEL}?view=L2%3Amoe&node=moe.w13&overlay=true&tp=2&side=roofline`);
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expect(page.getByTestId("roofline-chart")).toBeVisible();
  await shot(page, "model-dark");
});

test("diff page highlights changes", async ({ page }) => {
  await page.goto("./#/diff");
  await expect(page.getByTestId("change-list")).toBeVisible();
  await page.getByTestId("diff-view-L2:moe").click();
  await expect(page.getByTestId("diff-graph-b").getByTestId("diff-badge").first()).toBeVisible();
  await shot(page, "diff");
});

test("dense model renders its build-time layout", async ({ page }) => {
  await page.goto("./#/m/main-verified/A2/qwen3-8b");
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await expect(page.getByTestId("memory-chart")).toBeVisible();
});

test("keyboard: Space selects a focused node, Enter opens it, Esc returns to its anchor", async ({
  page,
}) => {
  await page.goto(`./${MODEL}`);
  const decoder = page.locator('.react-flow__node[data-id="decoder"]');
  await decoder.focus();
  await page.keyboard.press(" ");
  await expect(page).toHaveURL(/node=decoder/);
  await decoder.focus();
  await page.keyboard.press("Enter");
  await expect(page.getByTestId("current-view")).toContainText("L1");
  await page.keyboard.press("Escape");
  await expect(page.getByTestId("current-view")).toContainText("L0");
  await expect(page).toHaveURL(/node=decoder/);
});

test("ep choices are limited to 1 or tp×dp", async ({ page }) => {
  await page.goto(`./${MODEL}?tp=2`);
  await page.getByTestId("param-ep").click();
  await expect(page.getByTestId("param-ep-4")).toHaveAttribute("data-disabled", "");
  await expect(page.getByTestId("param-ep-2")).not.toHaveAttribute("data-disabled", "");
  await page.getByTestId("param-ep-2").click();
  await expect(page).toHaveURL(/ep=2/);
  // changing tp keeps enabled EP on the tp×dp group
  await page.getByTestId("param-tp").getByText("4", { exact: true }).click();
  await expect(page).toHaveURL(/ep=4/);
});

test("single-file build works from file://", async ({ page }) => {
  const file = join(here, "..", "dist-single", "index.html");
  test.skip(!existsSync(file), "run build:single first");
  await page.goto(`file://${file}#/m/main-verified/A2/qwen3-30b-a3b?view=L1%3Adecoder`);
  await expect(page.getByTestId("node-attn")).toBeVisible();
  await page.getByTestId("node-moe").click();
  await expect(page.getByTestId("node-detail")).toContainText("Sparse MoE");
});
