import type { HardwarePreset, Index, IndexCell, IR, Report, View } from "@inferatlas/contracts";
import type { Arch } from "@/lib/arch-types";
import type { HubManifest } from "@/lib/hub";
import { cellUnavailableMessage } from "@/lib/status";

export type Hardware = HardwarePreset;

export interface Bundle {
  dataPath: string;
  ir: IR;
  view?: View | undefined;
  report?: Report | undefined;
}

export interface InlineData {
  index: Index;
  hardware?: Hardware[];
  /** `ir` may be absent for structure-only cells that ship just `arch.json`. */
  bundles: Record<string, { ir?: IR; view?: View; report?: Report; arch?: Arch }>;
  /** Hub build only: the single-file exports that sit beside the hub HTML. */
  hub?: HubManifest;
}

declare global {
  interface Window {
    __INFERATLAS__?: InlineData;
  }
}

export class DataError extends Error {
  constructor(
    message: string,
    readonly kind: "not_found" | "invalid" | "network" = "invalid",
    /** The matrix cell this error is about, when there is one (e.g. a triage-only deep link). */
    readonly cell?: IndexCell,
  ) {
    super(message);
  }
}

export const inline = (): InlineData | undefined =>
  typeof window === "undefined" ? undefined : window.__INFERATLAS__;

async function fetchJson<T>(path: string, optional: true): Promise<T | undefined>;
async function fetchJson<T>(path: string, optional?: false): Promise<T>;
async function fetchJson<T>(path: string, optional = false): Promise<T | undefined> {
  let res: Response;
  try {
    res = await fetch(path);
  } catch (e) {
    if (optional) return undefined;
    throw new DataError(`Could not fetch ${path}: ${String(e)}`, "network");
  }
  const type = res.headers.get("content-type") ?? "";
  // Dev servers answer unknown paths with index.html; treat non-JSON as missing.
  if (!res.ok || type.includes("text/html")) {
    if (optional) return undefined;
    throw new DataError(`${path} is not available (${res.status})`, "not_found");
  }
  try {
    return (await res.json()) as T;
  } catch {
    throw new DataError(`${path} is not valid JSON`, "invalid");
  }
}

let indexPromise: Promise<Index> | undefined;
export function loadIndex(): Promise<Index> {
  const data = inline();
  if (data) return Promise.resolve(data.index);
  indexPromise ??= fetchJson<Index>("data/index.json").catch((e) => {
    indexPromise = undefined;
    throw e;
  });
  return indexPromise;
}

let hwPromise: Promise<Hardware[]> | undefined;
export function loadHardware(): Promise<Hardware[]> {
  const data = inline();
  if (data) return Promise.resolve(data.hardware ?? []);
  hwPromise ??= fetchJson<Hardware[]>("data/hardware.json", true).then((h) => h ?? []);
  return hwPromise;
}

const bundleCache = new Map<string, Promise<Bundle>>();
export function loadBundle(dataPath: string): Promise<Bundle> {
  const data = inline();
  if (data) {
    const b = data.bundles[dataPath];
    if (!b?.ir) return Promise.reject(new DataError(`${dataPath} is not inlined`, "not_found"));
    return Promise.resolve({ dataPath, ir: b.ir, view: b.view, report: b.report });
  }
  let p = bundleCache.get(dataPath);
  if (!p) {
    const base = dataPath.replace(/\/+$/, "");
    p = Promise.all([
      fetchJson<IR>(`${base}/ir.json`),
      fetchJson<View>(`${base}/view.json`, true),
      fetchJson<Report>(`${base}/report.json`, true),
    ]).then(([ir, view, report]) => {
      if (!ir || !Array.isArray(ir.nodes)) throw new DataError(`${base}/ir.json is malformed`);
      return { dataPath, ir, view, report };
    });
    p.catch(() => bundleCache.delete(dataPath));
    bundleCache.set(dataPath, p);
  }
  return p;
}

export function findCell(
  index: Index,
  version: string,
  model: string,
  device: string,
): IndexCell | undefined {
  return index.cells.find((c) => c.version === version && c.model === model && c.device === device);
}

const archCache = new Map<string, Promise<Arch | undefined>>();
/** `arch.json` of a data path, or undefined when the snapshot has none yet. */
export function loadArch(dataPath: string): Promise<Arch | undefined> {
  const data = inline();
  if (data) return Promise.resolve(data.bundles[dataPath]?.arch);
  let p = archCache.get(dataPath);
  if (!p) {
    const base = dataPath.replace(/\/+$/, "");
    p = fetchJson<Arch>(`${base}/arch.json`, true).then((a) => {
      if (a && (!Array.isArray(a.stages) || !a.model))
        throw new DataError(`${base}/arch.json is malformed`);
      return a;
    });
    p.catch(() => archCache.delete(dataPath));
    archCache.set(dataPath, p);
  }
  return p;
}

/**
 * A cell's data: the numeric bundle (ir/view/report) and/or the structure (`arch.json`).
 * Structure first: a cell with only `arch.json` still opens; one with only `ir.json` shows the
 * architecture tab's empty state.
 */
export async function loadCellBundle(
  version: string,
  model: string,
  device: string,
): Promise<{ index: Index; cell: IndexCell; bundle: Bundle | undefined; arch: Arch | undefined }> {
  const index = await loadIndex();
  const cell = findCell(index, version, model, device);
  if (!cell)
    throw new DataError(`No matrix cell for ${model} @ ${version} on ${device}`, "not_found");
  if (!cell.data_path) {
    throw new DataError(cellUnavailableMessage(cell), "not_found", cell);
  }
  const path = cell.data_path;
  const [bundle, arch] = await Promise.all([
    loadBundle(path).catch((e: unknown) => {
      if (e instanceof DataError && e.kind === "not_found") return undefined;
      throw e;
    }),
    loadArch(path),
  ]);
  if (!bundle && !arch)
    throw new DataError(`${path} has neither ir.json nor arch.json`, "not_found");
  return { index, cell, bundle, arch };
}

/**
 * The cell a single-file export opens on: the first inlined cell, preferring `ok` over triage.
 * Undefined when nothing is inlined (the regular site, or an empty export).
 */
export function singleTarget(
  data: InlineData | undefined,
): { version: string; model: string; device: string } | undefined {
  if (!data) return undefined;
  const inlined = data.index.cells.filter((c) => c.data_path && data.bundles[c.data_path]);
  const cell = inlined.find((c) => c.status === "ok") ?? inlined[0];
  return cell && { version: cell.version, model: cell.model, device: cell.device };
}

export function srcUrl(src: { repo: string; commit: string; file: string; line: number }): string {
  const repo =
    src.repo === "vllm"
      ? "vllm-project/vllm"
      : src.repo === "vllm-ascend"
        ? "vllm-project/vllm-ascend"
        : "pytorch/pytorch";
  return `https://github.com/${repo}/blob/${src.commit}/${src.file}#L${src.line}`;
}
