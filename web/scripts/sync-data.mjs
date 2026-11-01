#!/usr/bin/env node
import { spawnSync } from "node:child_process";
/**
 * Populate web/public/data/ (dev-plan §3.7 layout):
 *   data/index.json, data/hardware.json, data/<version>-<device>/<model>/{ir,view,report}.json
 *
 * Release:  INFERATLAS_DATA_DIR=<pipeline output dir> node scripts/sync-data.mjs
 *           Copies real data. Without INFERATLAS_DATA_DIR (and without --demo) this FAILS, so a
 *           release build can never ship fixtures.
 * Demo:     node scripts/sync-data.mjs --demo   (dev server, build:demo, e2e)
 *           Copies contract fixtures, lays out views with packages/layout when built, and derives
 *           ONE clearly-labelled synthetic snapshot ("fixture-prev") so the diff page has two
 *           versions. None of this is publication data.
 */
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "..");
const repo = resolve(web, "..");
const out = join(web, "public", "data");

const demo = process.argv.includes("--demo") || process.env.INFERATLAS_DEMO === "1";

if (process.env.INFERATLAS_DATA_DIR) {
  // Real pipeline output: a directory containing index.json, hardware.json and data paths.
  // Relative paths are taken from where the user ran pnpm (INIT_CWD), not web/.
  const src = resolve(process.env.INIT_CWD ?? process.cwd(), process.env.INFERATLAS_DATA_DIR);
  if (!existsSync(join(src, "index.json"))) {
    console.error(`[sync-data] ${src} has no index.json`);
    process.exit(1);
  }
  if (src === out || src.startsWith(`${out}${sep}`) || out.startsWith(`${src}${sep}`)) {
    console.error("[sync-data] release input must be separate from web/public/data");
    process.exit(1);
  }
  rmSync(out, { recursive: true, force: true });
  cpSync(src, out, { recursive: true });
  console.log(`[sync-data] copied ${src} -> ${out}`);
  process.exit(0);
}

if (!demo) {
  console.error(
    "[sync-data] INFERATLAS_DATA_DIR is required for a release build (directory with index.json,\n" +
      "hardware.json and the data/<snapshot>/<model>/ paths it names).\n" +
      "For local fixture/demo data use `pnpm --filter web build:demo` or `dev`.",
  );
  process.exit(1);
}

// Prefer the real build-time layout (packages/layout) over the fixture's hand-made views.
const layoutCli = join(repo, "packages", "layout", "dist", "cli.js");
const layout = (irPath, outPath) => {
  if (!existsSync(layoutCli)) return false;
  const r = spawnSync(process.execPath, [layoutCli, irPath, "-o", outPath], { encoding: "utf8" });
  if (r.status !== 0) {
    console.warn(`[sync-data] layout failed for ${irPath}; using fixture view\n${r.stderr}`);
    return false;
  }
  return true;
};

const fx = join(repo, "fixtures");
const readJson = (p) => JSON.parse(readFileSync(p, "utf8"));
const writeJson = (p, v) => {
  mkdirSync(dirname(p), { recursive: true });
  writeFileSync(p, `${JSON.stringify(v, null, 2)}\n`);
};

rmSync(out, { recursive: true, force: true });
mkdirSync(out, { recursive: true });

// 1. Real fixture snapshots at the paths the fixture index names.
const index = readJson(join(fx, "index.json"));
const sources = {
  "qwen3-30b-a3b": join(fx, "qwen3-30b-a3b"),
  "qwen3-8b": join(fx, "qwen3-8b"),
};
for (const cell of index.cells) {
  if (!cell.data_path) continue;
  const src = sources[cell.model];
  if (!src) continue;
  const dst = join(web, "public", cell.data_path);
  for (const f of ["ir.json", "view.json", "report.json"]) {
    if (existsSync(join(src, f))) cpSync(join(src, f), join(dst, f));
  }
  layout(join(dst, "ir.json"), join(dst, "view.json"));
}

// 2. Synthetic previous snapshot for the diff page (fixture-derived, labelled as such).
const prevPath = "data/fixture-prev-A2/qwen3-30b-a3b";
{
  const ir = readJson(join(fx, "qwen3-30b-a3b", "ir.json"));
  const view = readJson(join(fx, "qwen3-30b-a3b", "view.json"));
  const report = readJson(join(fx, "qwen3-30b-a3b", "report.json"));
  const prevSnap = (s) => ({ ...s, vllm: { ...s.vllm, tag: "fixture-prev" } });
  ir.snapshot = prevSnap(ir.snapshot);
  report.snapshot = prevSnap(report.snapshot);
  ir.notes = [
    "SYNTHETIC: derived from the Qwen3-30B-A3B fixture by web/scripts/sync-fixtures.mjs to exercise the diff page. Not a real upstream snapshot.",
    ...(ir.notes ?? []),
  ];
  // a) MoE TP reduce did not exist as an explicit node.
  const dropped = "moe.experts.reduce";
  ir.nodes = ir.nodes.filter((n) => n.id !== dropped);
  ir.edges = ir.edges.filter((e) => e.from !== dropped && e.to !== dropped);
  for (const v of view.views) {
    v.nodes = v.nodes.filter((n) => n.ir_node !== dropped);
    v.edges = v.edges.filter((e) => e.from !== dropped && e.to !== dropped);
  }
  // b) Output projection had no Ascend replacement; router was a ReplicatedLinear.
  for (const n of ir.nodes) {
    if (n.id === "attn.o") n.ascend = null;
    if (n.id === "moe.gate") {
      n.class = "vllm.model_executor.layers.linear.ReplicatedLinear";
      n.src = { ...n.src, symbol: n.class, file: "vllm/model_executor/layers/linear.py" };
    }
    if (n.id === "attn.core") n.flops.decode = n.flops.decode.replace("(context+1)", "context");
  }
  writeJson(join(web, "public", prevPath, "ir.json"), ir);
  writeJson(join(web, "public", prevPath, "view.json"), view);
  layout(join(web, "public", prevPath, "ir.json"), join(web, "public", prevPath, "view.json"));
  writeJson(join(web, "public", prevPath, "report.json"), report);

  index.versions.push({
    ascend: "main",
    id: "fixture-prev",
    label: "synthetic · diff demo",
    vllm: "fixture-prev (synthetic)",
  });
  index.versions.sort((a, b) => a.id.localeCompare(b.id));
  index.cells.push({
    data_path: prevPath,
    device: "A2",
    model: "qwen3-30b-a3b",
    status: "ok",
    triage_count: 0,
    version: "fixture-prev",
  });
  index.cells.sort((a, b) =>
    `${a.version}\0${a.model}\0${a.device}`.localeCompare(`${b.version}\0${b.model}\0${b.device}`),
  );
}
writeJson(join(out, "index.json"), index);

// 3. Hardware presets (roofline) from catalog.yaml; null peaks stay null.
const catalog = parse(readFileSync(join(repo, "catalog.yaml"), "utf8"));
writeJson(join(out, "hardware.json"), catalog.hardware_presets ?? []);

console.log(`[sync-data] wrote DEMO fixture data to ${out}`);
