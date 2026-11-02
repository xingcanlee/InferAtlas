import { existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const shots = join(here, "..", "test-results", "screenshots");
const MODEL = "#/m/main-verified/A2/qwen3-30b-a3b";
const demo = process.env.INFERATLAS_DEMO === "1";

async function shot(page: Page, name: string) {
  await page.waitForTimeout(250);
  await page.screenshot({ path: join(shots, `${name}.png`) });
}

test("home matrix lists cells and opens a model", async ({ page }) => {
  await page.goto("./");
  await expect(page.getByTestId("matrix")).toBeVisible();
  await expect(page.locator('[data-status="ok"]').first()).toBeVisible();
  if (demo) await expect(page.locator('[data-status="triage"]').first()).toBeVisible();
  // Structure first (v1.4 §1): cells lead with a coverage badge; version pairs without any built
  // cell are not columns, so "missing" only shows inside a built column.
  await expect(page.locator("[data-coverage]").first()).toBeVisible();
  await shot(page, "home");
  await page.getByTestId("cell-main-verified-qwen3-30b-a3b").click();
  await expect(page.getByTestId("model-title")).toHaveText("Qwen/Qwen3-30B-A3B");
});

test("model page: drill down, select, recompute, overlay", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(`./${MODEL}?tab=graph&tp=1`);
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
  await expect(page.getByTestId("node-detail")).toContainText("all_gather");
});

test("model page dark theme", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "dark" });
  await page.goto(`./${MODEL}?view=L2%3Amoe&node=moe.w13&overlay=true&tp=2&side=roofline`);
  await expect(page.locator("html")).toHaveClass(/dark/);
  await expect(page.getByTestId("roofline-chart")).toBeVisible();
  await shot(page, "model-dark");
});

test("diff page handles the available versions", async ({ page }) => {
  await page.goto("./#/diff");
  if (demo) {
    await expect(page.getByTestId("change-list")).toBeVisible();
    await page.getByTestId("diff-view-L2:moe").click();
    await expect(page.getByTestId("diff-graph-b").getByTestId("diff-badge").first()).toBeVisible();
  } else {
    await expect(page.getByText("Nothing to compare")).toBeVisible();
  }
  await shot(page, "diff");
});

test("dense model renders its build-time layout", async ({ page }) => {
  await page.goto("./#/m/main-verified/A2/qwen3-8b?tab=graph");
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await expect(page.getByTestId("memory-chart")).toBeVisible();
});

test("keyboard: Space selects a focused node, Enter opens it, Esc returns to its anchor", async ({
  page,
}) => {
  await page.goto(`./${MODEL}?tab=graph`);
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
  await page.goto(`./${MODEL}?tab=graph&tp=2`);
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
  const file =
    process.env.INFERATLAS_RELEASE === "1"
      ? join(here, "../../dist/inferatlas-qwen3-30b-a3b-a2.html")
      : join(here, "..", "dist-single", "index.html");
  expect(existsSync(file), "run inferatlas single before e2e").toBe(true);
  await page.context().setOffline(true);
  await page.goto(`file://${file}#/m/main-verified/A2/qwen3-30b-a3b?view=L1%3Adecoder`);
  await expect(page.getByTestId("node-attn")).toBeVisible();
  await page.getByTestId("node-moe").click();
  await expect(page.getByTestId("node-detail")).toContainText(
    demo ? "Sparse MoE" : "Qwen3MoeSparseMoeBlock",
  );
});

for (const device of demo ? ["A2"] : ["A2", "H100"]) {
  for (const model of ["qwen3-30b-a3b", "qwen3-8b"]) {
    test(`${model} on ${device} has data and a roofline`, async ({ page }) => {
      await page.goto(`./#/m/main-verified/${device}/${model}?side=roofline`);
      await expect(page.getByTestId("roofline-chart")).toBeVisible();
      await expect(page.getByTestId("hw-select")).toContainText(device);
      if (!demo && model === "qwen3-8b") {
        await page.getByTestId("node-decoder").dblclick();
        await page.getByTestId("node-mlp").dblclick();
        await expect(page.getByTestId("node-mlp.gate_up")).toBeVisible();
      }
    });
  }
}

if (!demo) {
  test("dense single-file export works offline", async ({ page }) => {
    const file = join(here, "..", "..", "dist", "inferatlas-qwen3-8b-a2.html");
    expect(existsSync(file), "build the dense single-file export first").toBe(true);
    await page.context().setOffline(true);
    await page.goto(`file://${file}#/m/main-verified/A2/qwen3-8b?view=L2%3Amlp`);
    await expect(page.getByTestId("node-mlp.gate_up")).toBeVisible();
    await page.getByTestId("node-mlp.down").click();
    await expect(page.getByTestId("node-detail")).toContainText("RowParallelLinear");
  });

  test("changing runtime dtype keeps the replacement marker and restores its note", async ({
    page,
  }) => {
    await page.goto(`./${MODEL}?view=L2%3Aattn&node=attn.o&overlay=true`);
    const card = page.getByTestId("ascend-card");
    await expect(card).toBeVisible();
    await expect(card).not.toContainText("当前参数下无已审阅说明");
    await page.getByTestId("param-dtype_w").getByText("fp16", { exact: true }).click();
    await expect(card).toContainText("Ascend 替换 — 当前参数下无已审阅说明");
    await expect(card).toContainText("AscendRowParallelLinear");
    await expect(card.locator('a[href*="vllm-ascend"]')).not.toHaveCount(0);
    await page.getByTestId("param-dtype_w").getByText("bf16", { exact: true }).click();
    await expect(card).not.toContainText("当前参数下无已审阅说明");
  });
}
