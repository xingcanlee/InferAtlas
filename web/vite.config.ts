/// <reference types="vitest/config" />
import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "public", "data");

/** Inline index + every snapshot bundle as `window.__INFERATLAS__` (single-file mode). */
function inlineData(): Plugin {
  const onlyModel = process.env.INFERATLAS_SINGLE_MODEL;
  return {
    name: "inferatlas-inline-data",
    transformIndexHtml() {
      const read = (p: string) => JSON.parse(readFileSync(p, "utf8")) as unknown;
      const index = read(join(dataDir, "index.json")) as {
        cells: { data_path?: string | null; model: string }[];
      };
      if (onlyModel) index.cells = index.cells.filter((c) => c.model === onlyModel);
      const bundles: Record<string, Record<string, unknown>> = {};
      for (const cell of index.cells) {
        if (!cell.data_path) continue;
        const dir = join(here, "public", cell.data_path);
        if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
        const bundle: Record<string, unknown> = {};
        for (const f of readdirSync(dir)) {
          if (["ir.json", "view.json", "report.json"].includes(f)) {
            bundle[f.replace(".json", "")] = read(join(dir, f));
          }
        }
        bundles[cell.data_path] = bundle;
      }
      const hwPath = join(dataDir, "hardware.json");
      const payload = {
        index,
        hardware: existsSync(hwPath) ? read(hwPath) : [],
        bundles,
      };
      // Escape "<" so data can never terminate the script element.
      const json = JSON.stringify(payload).replace(/</g, "\\u003c");
      return [
        {
          tag: "script",
          children: `window.__INFERATLAS__=${json};`,
          injectTo: "head-prepend",
        },
      ];
    },
  };
}

export default defineConfig(({ mode }) => {
  const single = mode === "single";
  return {
    base: "./",
    plugins: [react(), tailwindcss(), ...(single ? [inlineData(), viteSingleFile()] : [])],
    publicDir: single ? false : "public",
    resolve: {
      alias: {
        "@": resolve(here, "src"),
        "@inferatlas/contracts": resolve(here, "../packages/contracts/src/index.ts"),
      },
    },
    build: {
      outDir: single ? "dist-single" : "dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 2048,
      assetsInlineLimit: single ? Number.MAX_SAFE_INTEGER : 4096,
    },
    test: {
      environment: "jsdom",
      setupFiles: ["./tests/setup.ts"],
      include: ["tests/**/*.test.{ts,tsx}"],
      css: false,
      testTimeout: 20_000,
    },
    server: { fs: { allow: [resolve(here, "..")] } },
  };
});
