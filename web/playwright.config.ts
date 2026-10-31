import { defineConfig, devices } from "@playwright/test";

const PORT = 4318;

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
  webServer: {
    command: `pnpm exec vite preview --port ${PORT} --strictPort --host 127.0.0.1`,
    url: `http://127.0.0.1:${PORT}/`,
    reuseExistingServer: !process.env.CI,
    timeout: 60_000,
  },
});
