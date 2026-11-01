/// <reference types="vitest/config" />
import { copyFileSync, existsSync, mkdirSync, readdirSync, readFileSync, statSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, type Plugin } from "vite";
import { viteSingleFile } from "vite-plugin-singlefile";

const here = dirname(fileURLToPath(import.meta.url));
const dataDir = join(here, "public", "data");

/**
 * Inline data as `window.__INFERATLAS__`.
 * - single: index + every snapshot bundle (optionally one model via INFERATLAS_SINGLE_MODEL).
 * - hub: index + hardware + hub.json (the sibling exports), no bundles; entry is src/hub/main.tsx.
 * The built HTML is copied to INFERATLAS_SINGLE_OUTPUT / INFERATLAS_HUB_OUTPUT when set.
 */
function inlineData(mode: "single" | "hub"): Plugin {
  const hub = mode === "hub";
  const onlyModel = hub ? undefined : process.env.INFERATLAS_SINGLE_MODEL;
  // Optional, for local/demo builds whose data holds several versions of the model.
  const onlyVersion = hub ? undefined : process.env.INFERATLAS_SINGLE_VERSION;
  return {
    name: "inferatlas-inline-data",
    writeBundle(options) {
      const output = hub ? process.env.INFERATLAS_HUB_OUTPUT : process.env.INFERATLAS_SINGLE_OUTPUT;
      if (output) {
        mkdirSync(dirname(output), { recursive: true });
        copyFileSync(join(options.dir ?? `dist-${mode}`, "index.html"), output);
      }
    },
    transformIndexHtml() {
      const read = (p: string) => JSON.parse(readFileSync(p, "utf8")) as unknown;
      const index = read(join(dataDir, "index.json")) as {
        cells: { data_path?: string | null; model: string; version: string }[];
        models: { id: string }[];
        versions: { id: string }[];
      };
      if (onlyModel) {
        index.cells = index.cells.filter((c) => c.model === onlyModel);
        index.models = index.models.filter((m) => m.id === onlyModel);
      }
      if (onlyVersion) {
        index.cells = index.cells.filter((c) => c.version === onlyVersion);
        index.versions = index.versions.filter((v) => v.id === onlyVersion);
      }
      const bundles: Record<string, Record<string, unknown>> = {};
      for (const cell of hub ? [] : index.cells) {
        if (!cell.data_path) continue;
        const dir = join(here, "public", cell.data_path);
        if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
        const bundle: Record<string, unknown> = {};
        for (const f of readdirSync(dir)) {
          if (["ir.json", "view.json", "report.json", "arch.json"].includes(f)) {
            bundle[f.replace(".json", "")] = read(join(dir, f));
          }
        }
        bundles[cell.data_path] = bundle;
      }
      const hwPath = join(dataDir, "hardware.json");
      const hubPath = join(dataDir, "hub.json");
      if (hub && !existsSync(hubPath)) {
        throw new Error(`[inferatlas] hub mode needs ${hubPath} ({"exports":[...]})`);
      }
      const payload = {
        index,
        hardware: existsSync(hwPath) ? read(hwPath) : [],
        bundles,
        ...(hub ? { hub: read(hubPath) } : {}),
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

/** The hub is a small landing page: its own entry, without the router, graph or model code. */
function hubEntry(): Plugin {
  return {
    name: "inferatlas-hub-entry",
    transformIndexHtml: {
      order: "pre",
      handler: (html) =>
        html
          .replace("/src/main.tsx", "/src/hub/main.tsx")
          .replace("<title>InferAtlas</title>", "<title>InferAtlas · All models</title>"),
    },
  };
}

export default defineConfig(({ mode }) => {
  const inlined = mode === "single" || mode === "hub";
  return {
    base: "./",
    plugins: [
      react(),
      tailwindcss(),
      ...(mode === "hub" ? [hubEntry()] : []),
      ...(inlined ? [inlineData(mode as "single" | "hub"), viteSingleFile()] : []),
    ],
    publicDir: inlined ? false : "public",
    resolve: {
      alias: [
        { find: /^@\//, replacement: `${resolve(here, "src")}/` },
        {
          find: /^@inferatlas\/contracts$/,
          replacement: resolve(here, "../packages/contracts/src/index.ts"),
        },
        // Source, not dist: the package is copy-in friendly and needs no build for dev.
        {
          find: /^@inferatlas\/arch-ui$/,
          replacement: resolve(here, "../packages/arch-ui/src/index.ts"),
        },
      ],
    },
    build: {
      outDir: inlined ? `dist-${mode}` : "dist",
      emptyOutDir: true,
      chunkSizeWarningLimit: 2048,
      assetsInlineLimit: inlined ? Number.MAX_SAFE_INTEGER : 4096,
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
