import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";
import { graphFit } from "./fit";

const repo = join(dirname(fileURLToPath(import.meta.url)), "../..");
const shots = join(repo, "docs/screenshots/release");
const route = "#/m/main-verified/A2/qwen3-30b-a3b";

test.skip(
  process.env.INFERATLAS_RELEASE !== "1",
  "Set INFERATLAS_RELEASE=1 for release artifacts.",
);

async function screenshot(page: Page, name: string) {
  mkdirSync(shots, { recursive: true });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(300);
  await page.screenshot({ path: join(shots, `${name}.png`), scale: "css" });
}

/** No page scrollbars and, on a model page, every graph node inside the canvas at zoom ≥ 0.6. */
async function fits(page: Page) {
  expect(
    await page.evaluate(() => ({
      horizontal: document.documentElement.scrollWidth > innerWidth,
      vertical: document.documentElement.scrollHeight > innerHeight,
    })),
  ).toEqual({ horizontal: false, vertical: false });
  if ((await page.getByTestId("graph").count()) === 0) return;
  await expect.poll(async () => (await graphFit(page)).outside).toEqual([]);
  expect((await graphFit(page)).zoom).toBeGreaterThanOrEqual(0.6);
}

test("release matrix exposes DeepSeek blockers without numeric data or links", async ({ page }) => {
  const index = await (await page.request.get("./data/index.json")).json();
  expect(index.cells.filter((c: { status: string }) => c.status === "ok")).toHaveLength(4);
  const blocked = index.cells.filter(
    (c: { model: string; version: string; device: string }) =>
      c.model === "deepseek-v3" &&
      c.version === "main-verified" &&
      ["A2", "H100"].includes(c.device),
  );
  expect(blocked).toHaveLength(2);
  for (const cell of blocked) {
    expect(cell.status).toBe("triage");
    expect(cell.triage_count).toBeGreaterThan(0);
    expect(cell.data_path).toBeNull();
  }
  for (const device of ["A2", "H100"]) {
    await page.goto(`./#/?device=${device}`);
    await expect(page.getByTestId("matrix")).toBeVisible();
    await expect(page.locator('[data-status="triage"]')).toHaveCount(1);
    await expect(page.locator('a[href*="deepseek-v3"]')).toHaveCount(0);
    await fits(page);
    await screenshot(page, `home-${device.toLowerCase()}`);
  }
  await page.goto("./#/m/main-verified/A2/deepseek-v3");
  await expect(page.getByText(/deepseek-v3 .*has no artifacts/)).toBeVisible();
  await expect(page.getByTestId("model-title")).toHaveCount(0);
});

test("release model captures light, dark, overlay and smaller desktop", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(`./${route}`);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await fits(page);
  await screenshot(page, "model-light");
  await page.getByTestId("theme-toggle").click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await screenshot(page, "model-dark");
  await page.getByTestId("theme-toggle").click();
  await page.getByTestId("node-decoder").dblclick();
  await page.getByTestId("overlay-toggle").click();
  await page.getByTestId("node-norm.input").click();
  await expect(page.getByTestId("ascend-card")).toBeVisible();
  await screenshot(page, "model-overlay-a2");
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.goto(`./${route}`);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await fits(page);
  await screenshot(page, "model-1280");
  for (const view of ["L1:decoder", "L2:attn", "L2:moe"]) {
    await page.goto(`./${route}?view=${encodeURIComponent(view)}`);
    await expect(page.getByTestId("current-view")).toContainText(view.split(":")[0] as string);
    await fits(page);
  }
  await page.goto(`./${route}`);
  await page.locator(".react-flow__controls-fitview").click();
  await screenshot(page, "model-1280-fit");
  expect((await graphFit(page)).outside).toEqual([]);
  expect(errors).toEqual([]);
});

test("release diff needs two real snapshots", async ({ page }) => {
  await page.goto("./#/diff");
  await expect(page.getByText("Nothing to compare")).toBeVisible();
  await screenshot(page, "diff-unavailable");
});

for (const [model, before, after] of [
  ["qwen3-30b-a3b", "16 MiB", "4 MiB"],
  ["qwen3-8b", "32 MiB", "8 MiB"],
] as const) {
  test(`release ${model} recomputes from file:// with networking disabled`, async ({ page }) => {
    const errors: string[] = [];
    const network: string[] = [];
    page.on("pageerror", (error) => errors.push(error.message));
    page.on("request", (request) => {
      if (/^https?:/.test(request.url())) network.push(request.url());
    });
    await page.context().setOffline(true);
    await page.goto(
      `file://${join(repo, `dist/inferatlas-${model}-a2.html`)}#/m/main-verified/A2/${model}?view=L2%3Aattn&node=attn.o`,
    );
    const resident = page
      .getByTestId("node-detail")
      .getByText("Resident on this rank")
      .locator("..");
    await expect(resident).toContainText(before);
    await page.getByTestId("param-tp").getByText("4", { exact: true }).click();
    await expect(page).toHaveURL(/tp=4/);
    await expect(resident).toContainText(after);
    await screenshot(page, `single-${model}-a2-tp4`);
    expect(errors).toEqual([]);
    expect(network).toEqual([]);
  });
}
