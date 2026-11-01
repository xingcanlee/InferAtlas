#!/usr/bin/env node
/**
 * arch.json fixtures for the architecture UI, copied from REAL generator output (contract v1.3).
 *
 *   node fixtures/arch-ui/build.mjs [<release-data dir>]   # default: $INFERATLAS_DATA_DIR or
 *                                                          # <repo>/.release-data
 *
 * The source is a site-ready data directory as written by
 * `inferatlas site --data-dir .release-data` (index.json + <snapshot>/<model>/arch.json; see
 * docs/progress/S1.md "How to run"). That directory is gitignored, so the copies below are the
 * committed, schema-validated snapshot of it (generator/tests/test_schema_contracts.py checks
 * every fixtures/**\/arch.json against schemas/arch.schema.json).
 *
 * - fixtures/arch-ui/<model>/arch.json — byte-for-byte copies of the A2 cells. `web/scripts/
 *   sync-data.mjs --demo` installs them into the demo site; web unit tests and e2e use them.
 * - fixtures/arch-ui/s1/deepseek-v4-flash/arch.json — the H100 DeepSeek-V4-Flash cell trimmed for
 *   unit tests: every module keeps at most two params (the rest of the structure — bands,
 *   sublayers, IO, internals, notes — is untouched, so it stays schema-valid).
 */
import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const src = resolve(
  process.argv[2] ?? process.env.INFERATLAS_DATA_DIR ?? join(repo, ".release-data"),
);

/** Models the UI fixtures cover (dense, MoE, MLA, mHC/compressed, SWA alternation, hybrid). */
const MODELS = [
  "deepseek-v3",
  "deepseek-v4-flash",
  "gpt-oss-120b",
  "qwen3-8b",
  "qwen3-30b-a3b",
  "qwen3-next-80b-a3b",
];
const DEVICE = "A2";

if (!existsSync(join(src, "index.json"))) {
  console.error(`[arch-ui fixtures] ${src} is not a release-data directory (no index.json)`);
  process.exit(1);
}
const index = JSON.parse(readFileSync(join(src, "index.json"), "utf8"));
const archPath = (model, device) => {
  const cell = index.cells.find((c) => c.model === model && c.device === device && c.data_path);
  if (!cell) throw new Error(`no ${model} × ${device} cell in ${src}/index.json`);
  // data_path is "data/<snapshot>/<model>" relative to the site root = the data dir's parent.
  return join(src, cell.data_path.replace(/^data\//, ""), "arch.json");
};

for (const model of MODELS) {
  const from = archPath(model, DEVICE);
  const to = join(here, model, "arch.json");
  mkdirSync(dirname(to), { recursive: true });
  copyFileSync(from, to);
  console.log(`[arch-ui fixtures] ${from} -> ${to}`);
}

// Trimmed H100 DeepSeek-V4-Flash (unit tests import it; ≤ 2 params per module).
{
  const arch = JSON.parse(readFileSync(archPath("deepseek-v4-flash", "H100"), "utf8"));
  const trim = (m) => {
    if (m) m.params = m.params.slice(0, 2);
  };
  for (const stage of arch.stages) {
    if (stage.kind !== "stack") trim(stage.module);
    else for (const b of stage.bands) for (const s of b.block.sublayers) trim(s.module);
  }
  const to = join(here, "s1", "deepseek-v4-flash", "arch.json");
  mkdirSync(dirname(to), { recursive: true });
  writeFileSync(to, `${JSON.stringify(arch)}\n`);
  console.log(`[arch-ui fixtures] trimmed H100 deepseek-v4-flash -> ${to}`);
}
