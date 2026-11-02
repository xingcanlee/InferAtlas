import { defineConfig, devices } from "@playwright/test";

const PORT = 4318;
const MOBILE = /mobile\.spec\.ts/;

/**
 * Desktop specs run in Desktop Chrome at 1600×1000 (each spec sets its own window where it
 * matters). `mobile.spec.ts` runs on phone/tablet emulation (touch, mobile viewport meta) at the
 * sizes the site is designed for: iPhone 13 390×844, an Android phone at 360×800 and an iPad in
 * portrait at 768×1024. All projects use Chromium so the suite needs one browser.
 */
export default defineConfig({
  testDir: "e2e",
  timeout: 30_000,
  fullyParallel: false,
  reporter: [["list"]],
  use: {
    ...devices["Desktop Chrome"],
    baseURL: `http://127.0.0.1:${PORT}/`,
    viewport: { width: 1600, height: 1000 },
    deviceScaleFactor: 1,
  },
  projects: [
    { name: "desktop", testIgnore: MOBILE },
    {
      name: "iphone-13",
      testMatch: MOBILE,
      use: { ...devices["iPhone 13"], browserName: "chromium", deviceScaleFactor: 1 },
    },
    {
      name: "pixel-7",
      testMatch: MOBILE,
      use: { ...devices["Pixel 7"], viewport: { width: 360, height: 800 }, deviceScaleFactor: 1 },
    },
    {
      name: "ipad",
      testMatch: MOBILE,
      use: {
        ...devices["iPad (gen 7)"],
        browserName: "chromium",
        viewport: { width: 768, height: 1024 },
        deviceScaleFactor: 1,
      },
    },
  ],
  webServer: {
    command: `pnpm exec vite preview --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
