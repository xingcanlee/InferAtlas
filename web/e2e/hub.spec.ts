import { copyFileSync, existsSync, mkdirSync, mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { expect, type Page, test } from "@playwright/test";

const here = dirname(fileURLToPath(import.meta.url));
const web = join(here, "..");
const shots = join(web, "test-results", "screenshots");
const demo = process.env.INFERATLAS_DEMO === "1";
const single =
  process.env.INFERATLAS_RELEASE === "1"
    ? join(web, "../dist/inferatlas-qwen3-30b-a3b-a2.html")
    : join(web, "dist-single", "index.html");
const hub = join(web, "dist-hub", "index.html");

/** Fail on page exceptions, console errors and any HTTP(S) request (offline pages). */
function watch(page: Page) {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`pageerror: ${e.message}`));
  page.on("console", (m) => {
    if (m.type() === "error") problems.push(`console: ${m.text()}`);
  });
  page.on("request", (r) => {
    if (/^https?:/.test(r.url())) problems.push(`request: ${r.url()}`);
  });
  return problems;
}

test("single-file export opens on its embedded model page", async ({ page }) => {
  expect(existsSync(single), "build the single-file export first").toBe(true);
  const problems = watch(page);
  await page.context().setOffline(true);
  await page.goto(`file://${single}`);
  await expect(page.getByTestId("model-title")).toHaveText("Qwen/Qwen3-30B-A3B");
  await expect(page).toHaveURL(/#\/m\/main-verified\/A2\/qwen3-30b-a3b/);
  // Structure first: the export opens on the architecture figure; the graph is one tab away.
  await expect(page.getByTestId("arch-figure")).toBeVisible();
  await page.getByTestId("tab-graph").click();
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await expect(page.getByTestId("matrix")).toHaveCount(0);
  await expect(page.getByTestId("hw-select")).toContainText("A2");
  const all = page.getByTestId("nav-all-models");
  await expect(all).toHaveText("All models");
  await expect(all).toHaveAttribute("href", "./index.html");
  expect(problems).toEqual([]);
});

test.describe("hub", () => {
  test.skip(!demo, "the release hub is produced by the generator");

  test("lists every cell on file:// and links to the sibling export", async ({ page }) => {
    expect(existsSync(hub), "run build:hub:demo first").toBe(true);
    // Lay out a release-like dist/: hub as index.html beside the single-file export.
    const dir = mkdtempSync(join(tmpdir(), "inferatlas-hub-"));
    copyFileSync(hub, join(dir, "index.html"));
    copyFileSync(single, join(dir, "inferatlas-qwen3-30b-a3b-a2.html"));
    const problems = watch(page);
    await page.context().setOffline(true);
    await page.setViewportSize({ width: 1280, height: 800 });
    await page.emulateMedia({ colorScheme: "light" });
    await page.goto(`file://${join(dir, "index.html")}`);

    await expect(page).toHaveTitle("InferAtlas · All models");
    await expect(page.getByRole("heading", { name: "All models" })).toBeVisible();
    await expect(page.getByTestId("hub-matrix")).toHaveCount(2);
    await expect(page.getByTestId("hub-version-main-verified")).toBeVisible();
    const ok = page.getByTestId("hub-main-verified-qwen3-30b-a3b-A2");
    await expect(ok).toHaveAttribute("href", "./inferatlas-qwen3-30b-a3b-a2.html");
    await expect(ok.locator('[data-status="ok"]')).toBeVisible();
    const triage = page.getByTestId("hub-main-verified-qwen3-30b-a3b-H100");
    await expect(triage).toContainText("3 unresolved triage items");
    await expect(page.getByTestId("hub-main-verified-qwen3-8b-A2")).toContainText(
      "No offline export",
    );
    await expect(page.getByTestId("hub-main-verified-qwen3-8b-H100")).toHaveAttribute(
      "title",
      /No artifacts/,
    );
    await expect(page.getByTestId("hub-version-v0.32.0")).toContainText(
      "Nothing has been generated for this version pair yet.",
    );
    await expect(page.locator("main a")).toHaveCount(1);
    mkdirSync(shots, { recursive: true });
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: join(shots, "hub-light.png") });
    await page.getByTestId("theme-toggle").click();
    await expect(page.locator("html")).toHaveClass(/dark/);
    await page.screenshot({ path: join(shots, "hub-dark.png") });

    await ok.click();
    await expect(page).toHaveURL(/inferatlas-qwen3-30b-a3b-a2\.html#\/m\/main-verified\/A2\//);
    await expect(page.getByTestId("model-title")).toHaveText("Qwen/Qwen3-30B-A3B");
    await page.getByTestId("nav-all-models").click();
    await expect(page.getByRole("heading", { name: "All models" })).toBeVisible();
    expect(problems).toEqual([]);
  });

  test("a triage-only deep link says triage, not missing", async ({ page }) => {
    await page.goto("./#/m/main-verified/H100/qwen3-30b-a3b");
    const msg = page.getByTestId("cell-unavailable");
    await expect(msg).toHaveText(
      "qwen3-30b-a3b @ main-verified on H100: Blocked by 3 unresolved triage items; no numeric data was published.",
    );
    await expect(page.getByText("Needs triage: qwen3-30b-a3b")).toBeVisible();
    await expect(page.locator("main").getByText(/missing/i)).toHaveCount(0);
    await expect(page.locator('main [data-status="triage"]')).toBeVisible();
  });
});
