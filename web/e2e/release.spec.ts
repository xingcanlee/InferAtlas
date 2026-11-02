import { mkdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { test as base, expect, type Page } from "@playwright/test";
import { graphFit } from "./fit";

const repo = join(dirname(fileURLToPath(import.meta.url)), "../..");
const shots = join(repo, "docs/screenshots/release");
const route = "#/m/main-verified/A2/qwen3-30b-a3b";

const test = base.extend<{ expectedConsoleErrors: string[] }>({
  expectedConsoleErrors: [
    async ({ page }, use) => {
      const pageErrors: string[] = [];
      const consoleErrors: string[] = [];
      const expected: string[] = [];
      page.on("pageerror", (error) => pageErrors.push(error.message));
      page.on("console", (message) => {
        if (message.type() === "error") consoleErrors.push(message.text());
      });
      await use(expected);
      expect(pageErrors).toEqual([]);
      expect(consoleErrors).toEqual(expected.map((message) => expect.stringContaining(message)));
    },
    { auto: true },
  ],
});

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

test("release matrix links every extracted cell, triage included (structure first)", async ({
  page,
}) => {
  const index = await (await page.request.get("./data/index.json")).json();
  type C = {
    version: string;
    model: string;
    device: string;
    status: string;
    data_path: string | null;
  };
  // Structure first: every ok / triage cell was extracted, so every one carries a data path …
  const built: C[] = index.cells.filter((c: C) => c.status === "ok" || c.status === "triage");
  expect(built.length).toBeGreaterThan(0);
  expect(
    built.filter((c) => !c.data_path).map((c) => `${c.model}@${c.version}/${c.device}`),
  ).toEqual([]);
  // Triage never hides structure.
  expect(built.some((c) => c.status === "triage")).toBe(true);
  for (const device of [...new Set(built.map((c) => c.device))].sort()) {
    await page.goto(`./#/?device=${device}`);
    await expect(page.getByTestId("matrix")).toBeVisible();
    // … and is a link to its model page.
    for (const c of built.filter((x) => x.device === device))
      await expect(page.getByTestId(`cell-${c.version}-${c.model}`)).toHaveAttribute(
        "href",
        new RegExp(`/m/${c.version}/${device}/${c.model}$`),
      );
    await fits(page);
    await screenshot(page, `home-${device.toLowerCase()}`);
  }
});

test("release model captures light, dark, overlay and smaller desktop", async ({ page }) => {
  await page.emulateMedia({ colorScheme: "light" });
  await page.goto(`./${route}?tab=graph`);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await expect(
    page.getByTestId("param-tp").getByRole("radio", { name: "2", exact: true }),
  ).toHaveAttribute("aria-checked", "true");
  await expect(page.getByText(/no KV cache fits/i)).toHaveCount(0);
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
  await page.goto(`./${route}?tab=graph`);
  await expect(page.getByTestId("node-decoder")).toBeVisible();
  await fits(page);
  await screenshot(page, "model-1280");
  for (const view of ["L1:decoder", "L2:attn", "L2:moe"]) {
    await page.goto(`./${route}?view=${encodeURIComponent(view)}`);
    await expect(page.getByTestId("current-view")).toContainText(view.split(":")[0] as string);
    await fits(page);
  }
  await page.goto(`./${route}?tab=graph`);
  await page.locator(".react-flow__controls-fitview").click();
  await screenshot(page, "model-1280-fit");
  expect((await graphFit(page)).outside).toEqual([]);
});

test("release diff needs two real snapshots", async ({ page }) => {
  await page.goto("./#/diff");
  await expect(page.getByText("Nothing to compare")).toBeVisible();
  await screenshot(page, "diff-unavailable");
});

test("real release hub links every export and returns offline", async ({ page }) => {
  const network: string[] = [];
  page.on("request", (request) => {
    if (/^https?:/.test(request.url())) network.push(request.url());
  });
  await page.context().setOffline(true);
  await page.setViewportSize({ width: 1280, height: 800 });
  await page.emulateMedia({ colorScheme: "light" });
  const hub = `file://${join(repo, "dist/index.html")}`;
  await page.goto(hub);
  await expect(page).toHaveTitle("InferAtlas · All models");
  await expect(page.locator("main a")).toHaveCount(4);
  for (const device of ["A2", "H100"]) {
    const blocked = page.getByTestId(`hub-main-verified-deepseek-v3-${device}`);
    await expect(blocked).toContainText(/Blocked by \d+ unresolved triage items?/);
    await expect(blocked.locator('[data-status="triage"]')).toBeVisible();
    await expect(blocked).not.toHaveAttribute("href");
  }
  await fits(page);
  await screenshot(page, "hub-1280");
  await page.getByTestId("theme-toggle").click();
  await expect(page.locator("html")).toHaveClass(/dark/);
  await screenshot(page, "hub-dark-1280");
  await page.getByTestId("theme-toggle").click();

  for (const device of ["A2", "H100"]) {
    for (const model of ["qwen3-30b-a3b", "qwen3-8b"]) {
      const filename = `inferatlas-${model}-${device.toLowerCase()}.html`;
      const link = page.getByTestId(`hub-main-verified-${model}-${device}`);
      await expect(link).toHaveAttribute("href", `./${filename}`);
      await link.click();
      await expect(page.getByTestId("model-title")).toHaveText(
        model === "qwen3-30b-a3b" ? "Qwen/Qwen3-30B-A3B" : "Qwen/Qwen3-8B",
      );
      await expect(page).toHaveURL(
        new RegExp(`${filename}#\\/m\\/main-verified\\/${device}\\/${model}`),
      );
      await expect(page.getByTestId("node-decoder")).toBeVisible();
      const tp = device === "A2" && model === "qwen3-30b-a3b" ? "2" : "1";
      await expect(
        page.getByTestId("param-tp").getByRole("radio", { name: tp, exact: true }),
      ).toHaveAttribute("aria-checked", "true");
      await expect(page.getByText(/no KV cache fits/i)).toHaveCount(0);
      await page.getByTestId("nav-all-models").click();
      await expect(page).toHaveURL(hub);
      await expect(page.getByRole("heading", { name: "All models" })).toBeVisible();
      await expect(page.locator("main a")).toHaveCount(4);
    }
  }
  expect(network).toEqual([]);
});

for (const device of ["A2", "H100"]) {
  for (const [model, before, after] of [
    ["qwen3-30b-a3b", "16 MiB", "4 MiB"],
    ["qwen3-8b", "32 MiB", "8 MiB"],
  ] as const) {
    test(`release ${model} ${device} recomputes from file:// with networking disabled`, async ({
      page,
    }) => {
      const network: string[] = [];
      page.on("request", (request) => {
        if (/^https?:/.test(request.url())) network.push(request.url());
      });
      await page.context().setOffline(true);
      await page.goto(
        `file://${join(repo, `dist/inferatlas-${model}-${device.toLowerCase()}.html`)}#/m/main-verified/${device}/${model}?tp=1&view=L2%3Aattn&node=attn.o`,
      );
      const resident = page
        .getByTestId("node-detail")
        .getByText("Resident on this rank")
        .locator("..");
      await expect(resident).toContainText(before);
      await page.getByTestId("param-tp").getByText("4", { exact: true }).click();
      await expect(page).toHaveURL(/tp=4/);
      await expect(resident).toContainText(after);
      await screenshot(page, `single-${model}-${device.toLowerCase()}-tp4`);
      expect(network).toEqual([]);
    });
  }
}
