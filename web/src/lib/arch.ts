/**
 * Pure helpers over `arch.json`: stack folding (bands → periodic pattern groups), residual
 * segments of a block, layer → band resolution, symbol environment and note lookup.
 * Generic over the contract — nothing here knows about a particular model.
 */
import type { ExprEnv } from "@inferatlas/contracts";
import type {
  Arch,
  ArchBand,
  ArchBlock,
  ArchComponent,
  ArchModule,
  ArchModuleStage,
  ArchNote,
  ArchStackStage,
  ArchSublayer,
} from "./arch-types";
import { tryEvaluate } from "./expr";

export const isStack = (s: Arch["stages"][number]): s is ArchStackStage => s.kind === "stack";
export const moduleStages = (arch: Arch): ArchModuleStage[] =>
  arch.stages.filter((s): s is ArchModuleStage => s.kind !== "stack");
export const stacks = (arch: Arch): ArchStackStage[] => arch.stages.filter(isStack);

export function allBands(arch: Arch): ArchBand[] {
  return stacks(arch).flatMap((s) => s.bands);
}

export function layerCount(arch: Arch): number {
  return allBands(arch).reduce((n, b) => Math.max(n, b.layers[1] + 1), 0);
}

export function bandForLayer(arch: Arch, layer: number): ArchBand | undefined {
  return allBands(arch).find((b) => b.layers[0] <= layer && layer <= b.layers[1]);
}

export const layerRange = ([a, b]: [number, number]): string =>
  a === b ? `layer ${a}` : `layers ${a}–${b}`;

// ------------------------------------------------------------------------------- folding

/** Structural identity of a block, ignoring module paths, sources and band ids. */
export function blockSignature(block: ArchBlock): string {
  return JSON.stringify([
    block.residual,
    block.sublayers.map((s) => [s.id, s.role, s.component, s.module?.class_name ?? null]),
  ]);
}

export type StackItem =
  | { kind: "band"; band: ArchBand }
  | {
      kind: "pattern";
      label: string;
      /** Every member band, in order. */
      members: ArchBand[];
      /** The bands of one period (drawn once). */
      period: ArchBand[];
      repeats: number;
      layers: [number, number];
    };

/**
 * Consecutive bands sharing a non-null `pattern` label fold into one group drawn as
 * "period × repeats" when the run is periodic (same block structure and layer counts).
 */
export function stackItems(bands: ArchBand[]): StackItem[] {
  const out: StackItem[] = [];
  let i = 0;
  while (i < bands.length) {
    const b = bands[i] as ArchBand;
    if (!b.pattern) {
      out.push({ kind: "band", band: b });
      i++;
      continue;
    }
    let j = i;
    while (j < bands.length && bands[j]?.pattern === b.pattern) j++;
    const members = bands.slice(i, j);
    const sigs = members.map((m) => `${m.count}:${blockSignature(m.block)}`);
    let p = members.length;
    for (let cand = 1; cand <= members.length; cand++) {
      if (members.length % cand) continue;
      if (sigs.every((s, k) => s === sigs[k % cand])) {
        p = cand;
        break;
      }
    }
    const last = members[members.length - 1] as ArchBand;
    out.push({
      kind: "pattern",
      label: b.pattern,
      members,
      period: members.slice(0, p),
      repeats: members.length / p,
      layers: [b.layers[0], last.layers[1]],
    });
    i = j;
  }
  return out;
}

/** The band drawn in the figure for `band` (its period representative inside a pattern). */
export function representative(items: StackItem[], band: ArchBand): ArchBand {
  for (const it of items) {
    if (it.kind !== "pattern") continue;
    const k = it.members.indexOf(band);
    if (k >= 0) return it.period[k % it.period.length] as ArchBand;
  }
  return band;
}

/** All member bands a drawn (representative) band stands for. */
export function represented(items: StackItem[], band: ArchBand): ArchBand[] {
  for (const it of items) {
    if (it.kind !== "pattern") continue;
    const k = it.period.indexOf(band);
    if (k >= 0) return it.members.filter((_, i) => i % it.period.length === k);
  }
  return [band];
}

// ------------------------------------------------------------------------------ segments

export type Segment =
  | { kind: "branch"; body: ArchSublayer[]; add: ArchSublayer | undefined }
  | { kind: "node"; sublayer: ArchSublayer };

/**
 * Split a block into residual branches. Explicit `residual_add` sublayers close a branch; when
 * a block has none, a ⊕ is inferred after each mixer/ffn (pre/post/sandwich styles). For
 * post-norm blocks a norm directly after ⊕ stays on the main stream (outside the branch).
 */
export function blockSegments(block: ArchBlock): Segment[] {
  const subs = block.sublayers;
  if (block.residual === "unknown") return subs.map((s) => ({ kind: "node", sublayer: s }));
  const explicit = subs.some((s) => s.role === "residual_add");
  const out: Segment[] = [];
  let buf: ArchSublayer[] = [];
  const close = (add: ArchSublayer | undefined) => {
    out.push({ kind: "branch", body: buf, add });
    buf = [];
  };
  for (let i = 0; i < subs.length; i++) {
    const s = subs[i] as ArchSublayer;
    if (s.role === "residual_add") {
      close(s);
      continue;
    }
    if (block.residual === "post_norm" && buf.length === 0 && s.role === "norm" && out.length) {
      out.push({ kind: "node", sublayer: s });
      continue;
    }
    buf.push(s);
    if (!explicit && (s.role === "mixer" || s.role === "ffn")) {
      // Sandwich: the post-norm right after the mixer/ffn belongs inside the branch.
      const next = subs[i + 1];
      if (block.residual === "sandwich" && next?.role === "norm") {
        buf.push(next);
        i++;
      }
      close(undefined);
    }
  }
  for (const s of buf) out.push({ kind: "node", sublayer: s });
  return out;
}

// ----------------------------------------------------------------------------- symbols

export interface ArchEnv {
  env: ExprEnv;
  /** Symbols whose value comes from a URL override. */
  overridden: Set<string>;
}

export function archEnv(arch: Arch, overrides: Record<string, unknown>): ArchEnv {
  const env: Record<string, number> = {};
  const overridden = new Set<string>();
  for (const [k, s] of Object.entries(arch.symbols)) {
    const o = overrides[k];
    const n = typeof o === "number" ? o : typeof o === "string" ? Number(o) : Number.NaN;
    if (Number.isFinite(n) && n > 0 && n !== s.value) {
      env[k] = n;
      overridden.add(k);
    } else env[k] = s.value;
  }
  return { env, overridden };
}

export function evaluator(env: ExprEnv): (expr: string) => number | undefined {
  return (expr) => {
    const r = tryEvaluate(expr, env);
    return r.ok ? r.value : undefined;
  };
}

// ------------------------------------------------------------------------------- notes

export const subTarget = (bandId: string, subId: string) => `${bandId}/${subId}`;

export function notesFor(arch: Arch, targets: string[]): ArchNote[] {
  const set = new Set(targets);
  return arch.notes.filter((n) => n.target !== null && set.has(n.target));
}

/** De-duplicated by text: a note repeated on every member band is shown once. */
export function uniqueNotes(notes: ArchNote[]): ArchNote[] {
  const seen = new Set<string>();
  return notes.filter((n) => {
    const k = `${n.severity}\0${n.text}`;
    if (seen.has(k)) return false;
    seen.add(k);
    return true;
  });
}

export function noteMarker(
  notes: ArchNote[],
): { severity: "info" | "warn"; count: number } | undefined {
  const u = uniqueNotes(notes);
  if (!u.length) return undefined;
  return { severity: u.some((n) => n.severity === "warn") ? "warn" : "info", count: u.length };
}

// --------------------------------------------------------------------------- display

/** Module path for a concrete layer (band blocks carry their first layer's path). */
export function layerPath(module: ArchModule, band: ArchBand, layer: number): string {
  return module.path.replace(
    new RegExp(`(^|\\.)layers\\.${band.layers[0]}(?=\\.|$)`),
    `$1layers.${layer}`,
  );
}

const ATTN: Record<string, string> = {
  mha: "MHA",
  gqa: "GQA",
  mqa: "MQA",
  mla: "MLA",
  linear: "linear",
  sliding_window: "sliding window",
  sparse: "sparse",
  other: "attention",
};

/** One-line human summary of a component ("GQA attention", "MoE 128E top-8"). */
export function componentSummary(c: ArchComponent | null, role?: string): string {
  if (!c) return role === "residual_add" ? "Residual add" : (role ?? "module");
  switch (c.kind) {
    case "norm":
      return c.variant === "rmsnorm" ? "RMSNorm" : c.variant === "layernorm" ? "LayerNorm" : "Norm";
    case "attention":
      return c.variant === "linear"
        ? `Linear attention${c.extras.kind === "gated_deltanet" ? " (Gated DeltaNet)" : ""}`
        : `${ATTN[c.variant]} attention`;
    case "mlp":
      return `MLP (${c.variant === "swiglu" ? "SwiGLU" : c.variant === "geglu" ? "GeGLU" : c.variant})`;
    case "moe":
      return `MoE ${c.experts}E top-${c.top_k}${c.shared_experts ? ` + ${c.shared_experts} shared` : ""}`;
    case "generic":
      return c.class_name;
    case "embedding":
      return "Token embedding";
    case "lm_head":
      return "LM head";
  }
}

/** Short band subtitle from its mixer + ffn ("GQA · MoE"). */
export function bandSummary(band: ArchBand): string {
  const mixer = band.block.sublayers.find((s) => s.role === "mixer")?.component;
  const ffn = band.block.sublayers.find((s) => s.role === "ffn")?.component;
  const m =
    mixer?.kind === "attention"
      ? mixer.variant === "linear"
        ? "linear attn"
        : `${ATTN[mixer.variant]} attn`
      : mixer?.kind === "generic"
        ? mixer.class_name
        : undefined;
  const f =
    ffn?.kind === "moe"
      ? "MoE"
      : ffn?.kind === "mlp"
        ? "dense MLP"
        : ffn?.kind === "generic"
          ? ffn.class_name
          : undefined;
  return [m, f].filter(Boolean).join(" · ");
}
