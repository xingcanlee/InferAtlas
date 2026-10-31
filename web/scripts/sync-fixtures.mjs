#!/usr/bin/env node
/**
 * Copy contract fixtures into web/public/data/ for dev/build (dev-plan §3.7 layout):
 *   data/index.json, data/hardware.json, data/<version>-<device>/<model>/{ir,view,report}.json
 *
 * Also derives ONE clearly-labelled synthetic snapshot ("fixture-prev") from the
 * Qwen3-30B-A3B fixture so the version diff page has two versions to compare. It is not
 * publication data. With INFERATLAS_DATA_DIR=<dir> the real pipeline output is copied instead.
 */
import { cpSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parse } from "yaml";

const here = dirname(fileURLToPath(import.meta.url));
const web = resolve(here, "..");
const repo = resolve(web, "..");
const out = join(web, "public", "data");

if (process.env.INFERATLAS_DATA_DIR) {
  // Real pipeline output: a directory containing index.json, hardware.json and data paths.
  const src = resolve(process.env.INFERATLAS_DATA_DIR);
  rmSync(out, { recursive: true, force: true });
  cpSync(src, out, { recursive: true });
  console.log(`[sync-data] copied ${src} -> ${out}`);
  process.exit(0);
}

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
  for (const f of ["ir.json", "view.json", "report.json"]) {
    if (existsSync(join(src, f))) cpSync(join(src, f), join(web, "public", cell.data_path, f));
  }
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

console.log(`[sync-data] wrote fixtures to ${out}`);
