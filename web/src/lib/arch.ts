/**
 * Pure helpers over `arch.json`: stack folding (bands → periodic pattern groups), residual
 * segments of a block, layer → band resolution, symbol environment and note lookup.
 * Generic over the contract — nothing here knows about a particular model.
 */
import type {
  Arch,
  ArchBand,
  ArchBlock,
  ArchComponent,
  ArchIO,
  ArchModule,
  ArchModuleStage,
  ArchNote,
  ArchStackStage,
  ArchSublayer,
  ArchTensor,
  ExprEnv,
} from "@inferatlas/contracts";
import { tryEvaluate } from "./expr";

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

export const isStack = (s: Arch["stages"][number]): s is ArchStackStage => s.kind === "stack";
export const moduleStages = (arch: Arch): ArchModuleStage[] =>
  arch.stages.filter((s): s is ArchModuleStage => s.kind !== "stack");
export const stacks = (arch: Arch): ArchStackStage[] => arch.stages.filter(isStack);

export function allBands(arch: Arch): ArchBand[] {
  return stacks(arch).flatMap((s) => s.bands);
}

/** Last segment of a fully qualified class name (figures); detail views show the full name. */
export const shortClass = (cls: string | null | undefined): string =>
  cls ? (cls.split(".").at(-1) ?? cls) : "";

// --------------------------------------------------------------------------- layer slots

/**
 * One entry per decoder layer across every stack, in dataflow order. The selector and the
 * `?layer=` URL key use the ordinal, so models with several stacks (whose layer indices may
 * restart) and models without any stack both work; `layer` is the index within its band's stack.
 */
export interface LayerSlot {
  ordinal: number;
  stack: ArchStackStage;
  band: ArchBand;
  layer: number;
}

const slotCache = new WeakMap<Arch, LayerSlot[]>();
export function layerSlots(arch: Arch): LayerSlot[] {
  let slots = slotCache.get(arch);
  if (!slots) {
    slots = [];
    for (const stack of stacks(arch))
      for (const band of stack.bands)
        for (let layer = band.layers[0]; layer <= band.layers[1]; layer++)
          slots.push({ ordinal: slots.length, stack, band, layer });
    slotCache.set(arch, slots);
  }
  return slots;
}

export const layerCount = (arch: Arch): number => layerSlots(arch).length;

export const slotAt = (arch: Arch, ordinal: number): LayerSlot | undefined =>
  layerSlots(arch)[ordinal];

/** Band containing the layer with this ordinal (== layer index for single-stack models). */
export const bandForLayer = (arch: Arch, ordinal: number): ArchBand | undefined =>
  slotAt(arch, ordinal)?.band;

/** Ordinal of a band's first layer. */
export const bandOrdinal = (arch: Arch, band: ArchBand): number =>
  layerSlots(arch).find((s) => s.band === band)?.ordinal ?? 0;

export const layerRange = ([a, b]: [number, number]): string =>
  a === b ? `layer ${a}` : `layers ${a}–${b}`;

// ------------------------------------------------------------------------------- folding

function componentSignature(c: ArchComponent | null): unknown {
  if (!c) return null;
  switch (c.kind) {
    case "attention":
      // v1.2 §3: compression ratio and indexer distinguish otherwise identical attention.
      return [
        c.kind,
        c.variant,
        c.extras.sliding_window != null,
        c.extras.compress_ratio ?? null,
        c.extras.indexer ?? null,
      ];
    case "mlp":
    case "norm":
      return [c.kind, c.variant];
    case "moe":
      return [c.kind, c.router.scoring];
    case "generic":
      return [c.kind, shortClass(c.class_name)];
    default:
      return [c.kind];
  }
}

/**
 * Structural identity of a block (contract v1.1 §4): residual style plus, per sublayer, role,
 * component kind/variant and short class. Ids, module paths, sources and sizes are ignored.
 */
export function blockSignature(block: ArchBlock): string {
  return JSON.stringify([
    block.residual,
    block.sublayers.map((s) => [
      s.role,
      componentSignature(s.component),
      shortClass(s.module?.class_name),
    ]),
  ]);
}

const sameBand = (a: ArchBand, b: ArchBand) =>
  a.count === b.count && blockSignature(a.block) === blockSignature(b.block);

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

/** Longest structural period considered when folding unlabelled runs. */
const MAX_PERIOD = 6;

/** Short mixer name of a band ("linear attn", "GQA attn"). */
export function mixerLabel(band: ArchBand): string {
  const c = band.block.sublayers.find((s) => s.role === "mixer")?.component;
  if (c?.kind === "attention") return attentionLabel(c);
  if (c?.kind === "generic") return shortClass(c.class_name);
  return "block";
}

/** "GQA attn", "linear attn", "MQA attn ÷4 + indexer" (compressed attention, v1.2 §3). */
export function attentionLabel(c: Extract<ArchComponent, { kind: "attention" }>): string {
  const base = c.variant === "linear" ? "linear attn" : `${ATTN[c.variant]} attn`;
  const r = c.extras.compress_ratio;
  const comp = typeof r === "number" && r > 1 ? ` ÷${r}` : "";
  const swa = c.extras.sliding_window != null && c.extras.sliding_window !== false ? " · SWA" : "";
  return `${base}${comp}${c.extras.indexer === true ? " + indexer" : ""}${swa}`;
}

const autoLabel = (period: ArchBand[]) =>
  period.map((b) => `${b.count}× ${mixerLabel(b)}`).join(" + ");

/** Best fold of `bands[i..end)`: the period covering the most bands with ≥ 2 repeats. */
function foldAt(bands: ArchBand[], i: number, end: number): { p: number; r: number } | undefined {
  let best: { p: number; r: number } | undefined;
  for (let p = 1; p <= MAX_PERIOD && i + 2 * p <= end; p++) {
    let r = 1;
    while (
      i + (r + 1) * p <= end &&
      Array.from({ length: p }, (_, k) => k).every((k) =>
        sameBand(bands[i + r * p + k] as ArchBand, bands[i + k] as ArchBand),
      )
    )
      r++;
    if (r >= 2 && (!best || r * p > best.r * best.p)) best = { p, r };
  }
  return best;
}

function group(members: ArchBand[], period: ArchBand[], label: string): StackItem {
  const first = members[0] as ArchBand;
  const last = members[members.length - 1] as ArchBand;
  return {
    kind: "pattern",
    label,
    members,
    period,
    repeats: members.length / period.length,
    layers: [first.layers[0], last.layers[1]],
  };
}

/**
 * Fold a stack's bands for drawing. Consecutive bands sharing a `pattern` label form one group
 * (period = smallest structural repeat). Unlabelled runs are folded by structural signature
 * when they repeat at least twice (e.g. an emitter that splits a hybrid stack into alternating
 * bands without labelling them).
 */
export function stackItems(bands: ArchBand[]): StackItem[] {
  // v1.3 §1: a label only groups when ≥ 2 consecutive bands share it; a single-member labelled
  // run (emitters that label every band individually) is a plain band and folds structurally.
  const label = (k: number): string | null => {
    const p = bands[k]?.pattern;
    if (!p) return null;
    return bands[k - 1]?.pattern === p || bands[k + 1]?.pattern === p ? p : null;
  };
  const out: StackItem[] = [];
  let i = 0;
  while (i < bands.length) {
    const b = bands[i] as ArchBand;
    if (label(i)) {
      let j = i;
      while (j < bands.length && bands[j]?.pattern === b.pattern) j++;
      const members = bands.slice(i, j);
      let p = members.length;
      for (let cand = 1; cand < members.length; cand++) {
        if (members.length % cand) continue;
        if (members.every((m, k) => sameBand(m, members[k % cand] as ArchBand))) {
          p = cand;
          break;
        }
      }
      out.push(group(members, members.slice(0, p), b.pattern as string));
      i = j;
      continue;
    }
    let end = i;
    while (end < bands.length && !label(end)) end++;
    const f = foldAt(bands, i, end);
    if (f) {
      const period = bands.slice(i, i + f.p);
      out.push(group(bands.slice(i, i + f.p * f.r), period, autoLabel(period)));
      i += f.p * f.r;
    } else {
      out.push({ kind: "band", band: b });
      i++;
    }
  }
  return out;
}

/** The band the figure draws for `band`: its period representative inside a pattern group. */
export function drawnBand(arch: Arch, band: ArchBand): ArchBand {
  const stack = stacks(arch).find((s) => s.bands.includes(band));
  return stack ? representative(stackItems(stack.bands), band) : band;
}

/**
 * Most block columns drawn side by side; beyond this the stack shows one card per item (band or
 * folded pattern) and draws only the selected item (default: the one covering most layers).
 */
export const MAX_STACK_COLUMNS = 3;

export const itemBands = (it: StackItem): ArchBand[] =>
  it.kind === "band" ? [it.band] : it.members;
export const itemLayers = (it: StackItem): [number, number] =>
  it.kind === "band" ? it.band.layers : it.layers;

/** The item drawn in compact mode: the one holding `band`, else the one with most layers. */
export function compactFocus(
  items: StackItem[],
  band: ArchBand | undefined,
): StackItem | undefined {
  const holding =
    band &&
    items.find(
      (it) => itemBands(it).includes(band) || (it.kind === "pattern" && it.period.includes(band)),
    );
  if (holding) return holding;
  const size = (it: StackItem) => itemLayers(it)[1] - itemLayers(it)[0];
  return [...items].sort((a, b) => size(b) - size(a))[0];
}

/** Number of blocks the folded stack would draw side by side. */
export const stackColumns = (items: StackItem[]): number =>
  items.reduce((n, it) => n + (it.kind === "band" ? 1 : it.period.length), 0);

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
  | { kind: "node"; sublayer: ArchSublayer }
  /** v1.2 §4 mHC: `pre` reads the residual streams, `post` writes back (both `mhc_*`). */
  | {
      kind: "hyper";
      pre: ArchSublayer | undefined;
      body: ArchSublayer[];
      post: ArchSublayer | undefined;
    };

export const isMhc = (s: ArchSublayer): boolean => s.role === "other" && /^mhc/i.test(s.id);

/**
 * Hyper-connection blocks: each `mhc_*` sublayer closes the open segment and/or opens the next
 * one, in dataflow order (pre … post, or a fused post/pre that does both).
 */
function hyperSegments(subs: ArchSublayer[]): Segment[] {
  const out: Segment[] = [];
  let pre: ArchSublayer | undefined;
  let body: ArchSublayer[] = [];
  let open = false;
  for (const s of subs) {
    if (!isMhc(s)) {
      body.push(s);
      continue;
    }
    const isPost = /post/i.test(s.id);
    const isPre = /pre/i.test(s.id);
    if (open && (isPost || !isPre)) {
      out.push({ kind: "hyper", pre, body, post: s });
      body = [];
      pre = undefined;
      open = false;
      if (!(isPre && isPost)) continue;
    }
    if (body.length) {
      out.push(...body.map((b): Segment => ({ kind: "node", sublayer: b })));
      body = [];
    }
    pre = s;
    open = true;
  }
  if (open) out.push({ kind: "hyper", pre, body, post: undefined });
  else out.push(...body.map((b): Segment => ({ kind: "node", sublayer: b })));
  return out;
}

/**
 * Split a block into residual branches. Explicit `residual_add` sublayers close a branch; when
 * a block has none, a ⊕ is inferred after each mixer/ffn (pre/post/sandwich styles). For
 * post-norm blocks a norm directly after ⊕ stays on the main stream (outside the branch).
 */
export function blockSegments(block: ArchBlock): Segment[] {
  const subs = block.sublayers;
  if (block.residual === "hyper_connection") return hyperSegments(subs);
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

// --------------------------------------------------------------------------- normalize

type Attn = Extract<ArchComponent, { kind: "attention" }>;

/**
 * v1.3 §5: `variant` is the head-sharing scheme (mha/gqa/mqa/mla) or `linear`; locality and
 * sparsity live in extras. Older data used `sliding_window` / `sparse` variants: infer the
 * head-sharing scheme from the head counts and keep the locality in extras.
 */
export function normalizeAttention(c: Attn): Attn {
  if (c.variant !== "sliding_window" && c.variant !== "sparse") return c;
  const variant =
    c.extras.kv_lora_rank !== undefined
      ? "mla"
      : c.heads_kv === 1
        ? "mqa"
        : c.heads_kv === c.heads_q
          ? "mha"
          : "gqa";
  const extras = { ...c.extras };
  // `true` = "windowed, size unknown": a UI-only placeholder on this in-memory view. The
  // normalised arch is never serialised or exported; source data is left untouched.
  if (c.variant === "sliding_window" && extras.sliding_window === undefined)
    extras.sliding_window = true;
  return { ...c, variant, extras };
}

const archNorm = new WeakMap<Arch, Arch>();
const archPatches = new WeakMap<Arch, string[]>();

/** Model-level info note kind added when `normalizeArch` had to patch the data. */
export const NORMALISED_KIND = "ui_normalised";

const plural = (n: number, w: string) => `${n} ${w}${n === 1 ? "" : "s"}`;

/**
 * Contract-compatible view of older `arch.json` (deprecated attention variants, missing v1.2
 * `internals` / `flags`, missing v1.1 `placement`); cached per object. Current generator output
 * passes through structurally unchanged. Whatever had to be patched is recorded
 * (`normalizePatches`), surfaced as a model-level info note ("Normalised for display: …") and,
 * in development, logged with `console.warn`, so generator regressions stay visible.
 */
export function normalizeArch(arch: Arch): Arch {
  const hit = archNorm.get(arch);
  if (hit) return hit;
  const variants = new Map<string, number>();
  let internals = 0;
  let placement = 0;
  const fix = (c: ArchComponent | null | undefined) => {
    if (c?.kind !== "attention") return c;
    const n = normalizeAttention(c);
    if (n !== c) variants.set(c.variant, (variants.get(c.variant) ?? 0) + 1);
    return n;
  };
  const fixIO = (io: ArchIO): ArchIO => {
    if (io.internals) return io;
    internals++;
    return { ...io, internals: [] };
  };
  const stages = arch.stages.map((st) => {
    if (st.kind === "stack")
      return {
        ...st,
        bands: st.bands.map((b) => ({
          ...b,
          block: {
            ...b.block,
            sublayers: b.block.sublayers.map((x) => ({
              ...x,
              io: fixIO(x.io),
              component: fix(x.component) ?? null,
            })),
          },
        })),
      };
    if (st.placement == null) placement++;
    return {
      ...st,
      io: fixIO(st.io),
      component: fix(st.component) ?? null,
      // Legacy default (v1.1 §5): `other` stages without a placement are side branches.
      placement: st.placement ?? (st.kind === "other" ? "aside" : "spine"),
    };
  });
  const patches: string[] = [];
  for (const [v, n] of variants)
    patches.push(`${plural(n, "attention component")} with deprecated variant "${v}"`);
  if (internals) patches.push(`${plural(internals, "IO record")} without internals`);
  if (placement) patches.push(`${plural(placement, "stage")} without placement`);
  if (!arch.flags) patches.push("missing flags");
  if (!arch.notes) patches.push("missing notes");
  const notes = [...(arch.notes ?? [])];
  if (patches.length) {
    notes.push({
      severity: "info",
      target: null,
      text: `Normalised for display (older or non-conforming arch.json): ${patches.join("; ")}.`,
      triage_kind: NORMALISED_KIND,
    });
    if (import.meta.env?.DEV)
      console.warn(`[arch] normalised ${arch.model?.id ?? "arch.json"}: ${patches.join("; ")}`);
  }
  const out: Arch = { ...arch, flags: arch.flags ?? [], notes, stages: stages as Arch["stages"] };
  archNorm.set(arch, out);
  archPatches.set(arch, patches);
  return out;
}

/** What `normalizeArch` patched in `arch` (empty for conforming data). */
export function normalizePatches(arch: Arch): string[] {
  normalizeArch(arch);
  return archPatches.get(arch) ?? [];
}

// ----------------------------------------------------------------------------- symbols

/**
 * Runtime parameters the user may change; everything else in `symbols` is a model dimension
 * from the config and is shown read-only.
 */
export const RUNTIME_SYMBOLS: ReadonlySet<string> = new Set([
  "T",
  "S",
  "tp",
  "dp",
  "ep",
  "pp",
  "batch",
  "context",
]);

export interface ArchEnv {
  env: ExprEnv;
  /** Symbols whose value comes from a URL override. */
  overridden: Set<string>;
}

export function archEnv(arch: Arch, overrides: Record<string, unknown>): ArchEnv {
  const env: Record<string, number> = {};
  const overridden = new Set<string>();
  for (const [k, s] of Object.entries(arch.symbols)) {
    const o = RUNTIME_SYMBOLS.has(k) ? overrides[k] : undefined;
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

/** Raw lookup: notes whose `target` is one of `targets` (no class-level expansion). */
export function notesFor(arch: Arch, targets: string[]): ArchNote[] {
  const set = new Set(targets);
  return arch.notes.filter((n) => n.target !== null && set.has(n.target));
}

/**
 * v1.4 §2: note kinds that describe a **class**, not a place. The emitter de-duplicates them per
 * (class, kind) and targets the first instance; they apply to every instance of that class in
 * the model. Band-specific kinds (e.g. `golden_diff`) apply only to their target.
 */
export const CLASS_LEVEL_KINDS: ReadonlySet<string> = new Set([
  "unknown_layer_class",
  "unannotated_override",
  "conditional_override",
]);

interface NoteSite {
  cls: string | undefined;
  generic: boolean;
}

/** Every place a note can target: module stages (`<stage id>`) and band sublayers. */
function noteSites(arch: Arch): Map<string, NoteSite> {
  const out = new Map<string, NoteSite>();
  for (const st of arch.stages) {
    if (st.kind === "stack") {
      for (const b of st.bands)
        for (const s of b.block.sublayers)
          out.set(subTarget(b.id, s.id), {
            cls: s.module?.class_name ?? undefined,
            generic: s.component?.kind === "generic",
          });
    } else
      out.set(st.id, {
        cls: st.module?.class_name ?? undefined,
        generic: st.component?.kind === "generic",
      });
  }
  return out;
}

/** Class-level note (v1.4 §2), incl. the plain "generic component" info note of a class. */
export function isClassLevelNote(note: ArchNote, site: NoteSite | undefined): boolean {
  if (note.triage_kind) return CLASS_LEVEL_KINDS.has(note.triage_kind);
  return note.severity === "info" && (!!site?.generic || /^Generic component\b/.test(note.text));
}

const noteIndexCache = new WeakMap<Arch, Map<string, ArchNote[]>>();

/**
 * Notes per target key, with class-level notes copied to every instance of their class. Model-
 * level notes (`target: null`) are not indexed (see `modelNotes`).
 */
export function noteIndex(arch: Arch): Map<string, ArchNote[]> {
  const hit = noteIndexCache.get(arch);
  if (hit) return hit;
  const sites = noteSites(arch);
  const byClass = new Map<string, string[]>();
  for (const [key, site] of sites)
    if (site.cls) byClass.set(site.cls, [...(byClass.get(site.cls) ?? []), key]);
  const out = new Map<string, ArchNote[]>();
  const add = (key: string, n: ArchNote) => {
    const list = out.get(key);
    if (list) list.push(n);
    else out.set(key, [n]);
  };
  for (const n of arch.notes) {
    if (n.target === null) continue;
    const site = sites.get(n.target);
    const keys =
      site?.cls && isClassLevelNote(n, site) ? (byClass.get(site.cls) ?? [n.target]) : [n.target];
    for (const k of keys) add(k, n);
  }
  noteIndexCache.set(arch, out);
  return out;
}

/** Notes that apply to any of `keys` (class-level notes included), in `arch.notes` order. */
export function notesAt(arch: Arch, keys: string[]): ArchNote[] {
  const idx = noteIndex(arch);
  const hit = new Set(keys.flatMap((k) => idx.get(k) ?? []));
  return arch.notes.filter((n) => hit.has(n));
}

/** Notes of sublayer `subId` in `bands` (a drawn band and the pattern members it stands for). */
export const subNotes = (arch: Arch, bands: ArchBand[], subId: string): ArchNote[] =>
  notesAt(
    arch,
    bands.map((b) => subTarget(b.id, subId)),
  );

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

/**
 * Figure badge for a component: only targeted **warn** notes (v1.2 §2: warn = changes the
 * figure). Info notes live in the panels, never on the figure.
 */
export function noteMarker(
  notes: ArchNote[],
): { severity: "info" | "warn"; count: number } | undefined {
  const u = uniqueNotes(notes).filter((n) => n.severity === "warn");
  return u.length ? { severity: "warn", count: u.length } : undefined;
}

export interface NoteGroup {
  /** `triage_kind`, or `null` for plain notes. */
  kind: string | null;
  notes: ArchNote[];
  warn: number;
}

/** Notes grouped by `triage_kind` (warn-heavy groups first, plain info notes last). */
export function groupNotes(notes: ArchNote[]): NoteGroup[] {
  const groups = new Map<string | null, ArchNote[]>();
  for (const n of uniqueNotes(notes)) {
    const k = n.triage_kind ?? null;
    groups.set(k, [...(groups.get(k) ?? []), n]);
  }
  return [...groups.entries()]
    .map(([kind, ns]) => ({
      kind,
      notes: ns,
      warn: ns.filter((n) => n.severity === "warn").length,
    }))
    .sort(
      (a, b) =>
        Number(a.kind === null) - Number(b.kind === null) ||
        b.warn - a.warn ||
        String(a.kind).localeCompare(String(b.kind)),
    );
}

/** Model- and stack-level notes: `null` target, or a stack stage id. */
export function modelNotes(arch: Arch): ArchNote[] {
  const stackIds = new Set(stacks(arch).map((s) => s.id));
  return arch.notes.filter((n) => n.target === null || stackIds.has(n.target));
}

// ------------------------------------------------------------------------------- stages

/** `other` stages drawn as side branches (v1.1 §5; legacy default aside). */
export const isAsideStage = (s: Arch["stages"][number]): boolean =>
  s.kind === "other" && (s.placement ?? "aside") === "aside";

/**
 * Stage order (v1.3 §2) is the array order: the emitter writes true dataflow order (e.g. an mHC
 * `hc_head_op` before the final norm). Aside stages are drawn as side branches, not on the spine.
 */
export const spineStages = (arch: Arch): Arch["stages"] =>
  arch.stages.filter((s) => !isAsideStage(s));

/**
 * The stage where token ids enter the language model: an `embedding` stage whose component is a
 * vocabulary embedding (not a generic patch embedding), the last such before the primary stack.
 */
export function tokenEmbedding(arch: Arch): ArchModuleStage | undefined {
  const spine = spineStages(arch);
  const primary = primaryStack(arch);
  const end = primary ? spine.indexOf(primary) : spine.length;
  return spine
    .slice(0, end)
    .filter(
      (s): s is ArchModuleStage => s.kind === "embedding" && s.component?.kind === "embedding",
    )
    .at(-1);
}

/**
 * Spine stages before the token embedding form a second input branch (vision tower + adapter /
 * projector of a VLM): they consume another input (pixels) and their features are merged into
 * the token embeddings. Drawn left of the spine, merging into it (paper-style VLM figure).
 * Empty for text-only models.
 */
export function encoderStages(arch: Arch): Arch["stages"] {
  const tok = tokenEmbedding(arch);
  if (!tok) return [];
  const spine = spineStages(arch);
  return spine.slice(0, spine.indexOf(tok));
}

/** "Vision encoder" when the branch's ids/classes say so, otherwise "Encoder". */
export function encoderTitle(stages: Arch["stages"]): string {
  const text = stages
    .map((s) => `${s.id} ${s.kind === "stack" ? s.label : s.module.class_name}`)
    .join(" ");
  return /vision|visual|image|pixel|patch|\bvit\b/i.test(text) ? "Vision encoder" : "Encoder";
}

/**
 * Every model input (v1.4 §3 `inputs`: multimodal models list e.g. `pixel_values` next to
 * `input_ids`); falls back to the primary text `input` for data without the field.
 */
export function modelInputs(arch: Arch): ArchTensor[] {
  const list = (arch as Arch & { inputs?: unknown }).inputs;
  const ok = Array.isArray(list)
    ? list.filter(
        (t): t is ArchTensor =>
          !!t &&
          typeof t === "object" &&
          typeof (t as ArchTensor).name === "string" &&
          Array.isArray((t as ArchTensor).shape),
      )
    : [];
  return ok.length ? ok : [arch.input];
}

/**
 * A stage is its own module when `module.path` is the stage id. v1.4 synthetic stages are not:
 * the spine `hc_expand` is a tensor op whose provenance is the containing model, and a
 * config-only MTP aside (flag `config_only`, empty path) names the model class it is declared by.
 * Their class name must not be drawn as the node's class.
 */
export function ownsModule(stage: ArchModuleStage): boolean {
  return !!stage.module?.path && stage.module.path === stage.id;
}

/** Declared by the model config but not constructed by vLLM (v1.4 MTP asides). */
export function isConfigOnly(stage: ArchModuleStage): boolean {
  return stage.module?.flags?.includes("config_only") ?? false;
}

/** mHC stream expansion / merge stages (v1.4 §4, v1.3 §2): `hc_expand`, `hc_head_op`. */
export function mhcStageLabel(id: string): string | undefined {
  // `model.hc_head_op`, `model.layers:hc_expand` (v1.4 synthetic spine expansion).
  const last = id.split(/[.:]/).at(-1) ?? id;
  if (/^hc_expand/i.test(last)) return "mHC expand (→ HC streams)";
  if (/^hc_head/i.test(last)) return "mHC merge (HC streams →)";
  return undefined;
}

const humanize = (id: string): string =>
  (id.split(".").at(-1) ?? id).replace(/[_-]+/g, " ").replace(/\s+/g, " ").trim();

/**
 * Display label of a stage. The emitter's defaults ("Token embedding", "Final norm") are only
 * right on the language spine: inside an encoder branch (vision patch embedding, pre/post
 * layernorms) the module name is used instead, as it is for labels that just repeat the id.
 */
export function stageLabel(arch: Arch, stage: Arch["stages"][number]): string {
  const label = stage.label?.trim();
  if (!label || label === stage.id) return mhcStageLabel(stage.id) ?? humanize(stage.id);
  if (stage.kind !== "stack" && encoderStages(arch).includes(stage)) {
    if (/^(token embedding|final norm)$/i.test(label)) return humanize(stage.id);
  }
  return label;
}

/** The decoder stack feeding the head: the last stack in dataflow order. */
export function primaryStack(arch: Arch): ArchStackStage | undefined {
  return stacks(arch).at(-1);
}

// ---------------------------------------------------------------------------- internals

/** Priority of internals for figure chips (v1.3 §6 canonical names; `*_lora_latent` tolerated). */
const KEY_INTERNALS = [
  /^q$/,
  /^k$/,
  /^v$/,
  /^q_(lora_)?latent$/,
  /^kv_(lora_)?latent$/,
  /^kv_compressed$/,
  /^index_scores$/,
  /^qkvz$/,
  /^recurrent_state$/,
  /^router_logits$/,
  /^topk_ids$/,
  /^topk_weights$/,
];

/** A few representative internal tensors for figure chips (q/k/v, latents, routing). */
export function keyInternals(sub: ArchSublayer, max = 3) {
  const all = sub.io.internals ?? [];
  const picked = KEY_INTERNALS.flatMap((re) => all.filter((t) => re.test(t.name)));
  return [...new Set(picked)].slice(0, max);
}

// --------------------------------------------------------------------------- display

/** Module path for a concrete layer (band blocks carry their first layer's path). */
export function layerPath(module: ArchModule, band: ArchBand, layer: number): string {
  return module.path.replace(
    new RegExp(`(^|\\.)layers\\.${band.layers[0]}(?=\\.|$)`),
    `$1layers.${layer}`,
  );
}

/** One-line human summary of a component ("GQA attention", "MoE 128E top-8"). */
export function componentSummary(c: ArchComponent | null, role?: string, id?: string): string {
  if (id && /^mhc/i.test(id))
    return /post/i.test(id) ? "mHC write-back (H_post · H_res)" : "mHC read (H_pre)";
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
      return `MoE ${c.experts}E top-${c.top_k}${c.shared_experts ? ` + ${c.shared_experts} shared` : ""} · ${c.router.scoring} router`;
    case "generic":
      return shortClass(c.class_name);
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
      ? attentionLabel(mixer)
      : mixer?.kind === "generic"
        ? shortClass(mixer.class_name)
        : undefined;
  const f =
    ffn?.kind === "moe"
      ? "MoE"
      : ffn?.kind === "mlp"
        ? "dense MLP"
        : ffn?.kind === "generic"
          ? shortClass(ffn.class_name)
          : undefined;
  return [m, f].filter(Boolean).join(" · ");
}
