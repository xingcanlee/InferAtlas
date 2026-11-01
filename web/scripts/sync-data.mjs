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
 *           versions, synthetic H100 status-only cells (one triage-only, like a blocked release
 *           cell) and a demo hub.json for `build:hub:demo`. None of this is publication data.
 *           Hand-written arch.json fixtures (fixtures/arch-ui/<model>/arch.json) are copied into
 *           every main-verified cell of their model; models without IR fixtures (DeepSeek-V3,
 *           Qwen3-Next) get structure-only triage cells. The synthetic fixture-prev snapshot has
 *           no arch.json, which exercises the architecture tab's empty state.
 */
import {
  cpSync,
  existsSync,
  mkdirSync,
  readdirSync,
  readFileSync,
  rmSync,
  writeFileSync,
} from "node:fs";
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

// 1b. Hand-written arch.json fixtures (structure first): copy into the model's main-verified
//     cell, creating a structure-only cell (no ir.json) for models without IR fixtures.
const ARCH_LABELS = {
  "deepseek-v3": "DeepSeek V3",
  "deepseek-v4-flash": "DeepSeek V4 Flash",
  "gpt-oss-120b": "gpt-oss 120B",
  "qwen3-next-80b-a3b": "Qwen3 Next 80B A3B",
};
const archFx = join(fx, "arch-ui");
for (const model of existsSync(archFx) ? readdirSync(archFx).sort() : []) {
  const file = join(archFx, model, "arch.json");
  if (!existsSync(file)) continue;
  const arch = readJson(file);
  let cell = index.cells.find(
    (c) => c.model === model && c.version === "main-verified" && c.device === arch.device,
  );
  if (!cell) {
    const warns = new Set(arch.notes.filter((n) => n.severity === "warn").map((n) => n.text)).size;
    cell = {
      data_path: `data/main-verified-${arch.device}/${model}`,
      device: arch.device,
      model,
      status: warns ? "triage" : "ok",
      triage_count: warns,
      version: "main-verified",
    };
    index.cells.push(cell);
  }
  if (!index.models.some((m) => m.id === model)) {
    index.models.push({
      family: arch.model.family,
      hf_id: arch.model.hf_id,
      id: model,
      label: ARCH_LABELS[model] ?? model,
    });
  }
  if (cell.data_path) cpSync(file, join(web, "public", cell.data_path, "arch.json"));
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
  // c) Status-only H100 column: a triage cell without a numeric bundle (like the release's
  //    blocked DeepSeek cells) exercises the honest triage deep-link message and hub cells.
  for (const version of ["main-verified", "v0.32.0"]) {
    for (const model of ["qwen3-30b-a3b", "qwen3-8b"]) {
      const triage = version === "main-verified" && model === "qwen3-30b-a3b";
      index.cells.push({
        data_path: null,
        device: "H100",
        model,
        status: triage ? "triage" : "missing",
        triage_count: triage ? 3 : 0,
        version,
      });
    }
  }
  index.cells.sort((a, b) =>
    `${a.version}\0${a.model}\0${a.device}`.localeCompare(`${b.version}\0${b.model}\0${b.device}`),
  );
}
writeJson(join(out, "index.json"), index);

// Demo hub manifest: the one export `build:single:demo` produces, under its release file name.
writeJson(join(out, "hub.json"), {
  exports: [
    {
      version: "main-verified",
      model: "qwen3-30b-a3b",
      device: "A2",
      file: "inferatlas-qwen3-30b-a3b-a2.html",
    },
  ],
});

// 3. Hardware presets (roofline) from catalog.yaml; null peaks stay null.
const catalog = parse(readFileSync(join(repo, "catalog.yaml"), "utf8"));
writeJson(join(out, "hardware.json"), catalog.hardware_presets ?? []);

console.log(`[sync-data] wrote DEMO fixture data to ${out}`);
