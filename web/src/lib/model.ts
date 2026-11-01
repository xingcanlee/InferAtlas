/**
 * Environment binding (docs/expr-dsl.md "IR binding") and per-node metric evaluation.
 * Everything numeric is derived from IR expression strings through `lib/expr`.
 */
import type { ExprEnv, IR, IRNode, ParamSpec, PhaseShape } from "@inferatlas/contracts";
import { compile, evaluate, free, tryEvaluate } from "./expr";

export type Phase = "prefill" | "decode";
export type ParamValue = number | string | boolean;
export type Overrides = Record<string, ParamValue | undefined>;

/** A computed quantity: value or the evaluation error. */
export type Q = { v: number; e?: undefined } | { v?: undefined; e: string };

const q = (src: string | undefined | null, env: ExprEnv): Q => {
  if (src === undefined || src === null) return { e: "absent" };
  const r = tryEvaluate(src, env);
  return r.ok ? { v: r.value } : { e: r.error };
};

// ------------------------------------------------------------------ symbols & params

export function evalSymbols(ir: IR): {
  values: Record<string, number>;
  errors: Record<string, string>;
} {
  const values: Record<string, number> = {};
  const errors: Record<string, string> = {};
  const pending = new Map(Object.entries(ir.symbols));
  const deps = new Map<string, string[]>();
  for (const [k, src] of pending) {
    try {
      deps.set(k, free(src));
    } catch (e) {
      errors[k] = String((e as Error).message);
      pending.delete(k);
    }
  }
  let progress = true;
  while (pending.size && progress) {
    progress = false;
    for (const [k, src] of pending) {
      const d = deps.get(k) ?? [];
      if (d.every((x) => x in values || !pending.has(x))) {
        const r = tryEvaluate(src, values);
        if (r.ok) values[k] = r.value;
        else errors[k] = r.error;
        pending.delete(k);
        progress = true;
      }
    }
  }
  for (const k of pending.keys()) errors[k] = "dependency cycle";
  return { values, errors };
}

export function paramDefault(spec: ParamSpec, symbols: ExprEnv): ParamValue {
  switch (spec.kind) {
    case "boolean":
      return spec.default === "true";
    case "dtype":
    case "enum":
      return spec.default;
    default:
      return evaluate(spec.default, symbols);
  }
}

export interface Bound {
  min?: number;
  max?: number;
  step?: number;
}

export function paramBounds(spec: ParamSpec, symbols: ExprEnv): Bound {
  const b: Bound = {};
  const ev = (s: string | null | undefined) => {
    if (!s) return undefined;
    const r = tryEvaluate(s, symbols);
    return r.ok ? r.value : undefined;
  };
  const min = ev(spec.min);
  const max = ev(spec.max);
  const step = ev(spec.step);
  if (min !== undefined) b.min = min;
  if (max !== undefined) b.max = max;
  if (step !== undefined) b.step = step;
  return b;
}

/** Coerce a raw (URL) value into the param's kind and validate its domain. */
export function validateParam(
  spec: ParamSpec,
  raw: ParamValue,
  symbols: ExprEnv,
): { value: ParamValue } | { error: string } {
  if (spec.kind === "boolean") {
    if (typeof raw === "boolean") return { value: raw };
    if (raw === "true" || raw === "false") return { value: raw === "true" };
    return { error: "expected true/false" };
  }
  if (spec.kind === "dtype" || spec.kind === "enum") {
    const s = String(raw);
    if (spec.choices && !spec.choices.includes(s))
      return { error: `must be one of ${spec.choices.join(", ")}` };
    return { value: s };
  }
  const n = typeof raw === "number" ? raw : Number(raw);
  if (!Number.isFinite(n)) return { error: "not a number" };
  if (spec.kind === "integer" && !Number.isInteger(n)) return { error: "must be an integer" };
  if (spec.choices && !spec.choices.some((c) => Number(c) === n)) {
    return { error: `must be one of ${spec.choices.join(", ")}` };
  }
  const b = paramBounds(spec, symbols);
  if (b.min !== undefined && n < b.min) return { error: `must be ≥ ${b.min}` };
  if (b.max !== undefined && n > b.max) return { error: `must be ≤ ${b.max}` };
  return { value: n };
}

export interface EnvState {
  env: ExprEnv;
  symbols: Record<string, number>;
  defaults: Record<string, ParamValue>;
  /** Effective values of runtime params (after validation). */
  params: Record<string, ParamValue>;
  /** Override validation errors, keyed by param. */
  errors: Record<string, string>;
  /** Cross-parameter consistency warnings (advisory). */
  warnings: { params: string[]; message: string }[];
}

export function buildEnv(ir: IR, overrides: Overrides = {}): EnvState {
  const { values: symbols } = evalSymbols(ir);
  const defaults: Record<string, ParamValue> = {};
  const params: Record<string, ParamValue> = {};
  const errors: Record<string, string> = {};
  for (const [name, spec] of Object.entries(ir.params)) {
    let d: ParamValue;
    try {
      d = paramDefault(spec, symbols);
    } catch (e) {
      errors[name] = `default: ${(e as Error).message}`;
      continue;
    }
    defaults[name] = d;
    params[name] = d;
    const raw = overrides[name];
    if (raw === undefined || raw === "") continue;
    const r = validateParam(spec, raw, symbols);
    if ("error" in r) errors[name] = r.error;
    else params[name] = r.value;
  }
  const env: ExprEnv = { ...symbols, ...params };
  const ep = Number(params.ep ?? 1);
  if (ep !== 1 && ep !== Number(params.tp ?? 1) * Number(params.dp ?? 1)) {
    errors.ep = "EP must be 1 or tp×dp";
    params.ep = 1;
    env.ep = 1;
  }
  for (const [name, spec] of Object.entries(ir.params)) {
    if (!spec.constraint) continue;
    const result = tryEvaluate(spec.constraint, env);
    if (!result.ok || result.value === 0) errors[name] = `violates ${spec.constraint}`;
  }
  return { env, symbols, defaults, params, errors, warnings: consistency(params, symbols) };
}

/** Fixed source facts are retained, while runtime params (e.g. dtype_w) are re-evaluated. */
export function runtimeAnnotations(ir: IR, env: ExprEnv): IR {
  return {
    ...ir,
    nodes: ir.nodes.map((node) => {
      const annotation = node.ascend;
      if (!annotation?.applies_when) return node;
      const result = tryEvaluate(annotation.applies_when, { ...env, ...annotation.context });
      return result.ok && result.value !== 0 ? node : { ...node, ascend: null };
    }),
  };
}

function consistency(
  p: Record<string, ParamValue>,
  s: Record<string, number>,
): EnvState["warnings"] {
  const w: EnvState["warnings"] = [];
  const n = (k: string) => (typeof p[k] === "number" ? (p[k] as number) : undefined);
  const tp = n("tp");
  const dp = n("dp");
  const ep = n("ep");
  const pp = n("pp");
  if (ep !== undefined && ep > 1 && tp !== undefined && dp !== undefined && ep !== tp * dp) {
    w.push({
      params: ["ep", "tp", "dp"],
      message: `EP must equal tp×dp = ${tp * dp} when enabled`,
    });
  }
  if (ep !== undefined && ep > 1 && s.E !== undefined && s.E % ep !== 0) {
    w.push({ params: ["ep"], message: `E=${s.E} experts are not divisible by ep=${ep}` });
  }
  if (tp !== undefined && s.Hq !== undefined && s.Hq % tp !== 0) {
    w.push({ params: ["tp"], message: `tp must divide Hq=${s.Hq}` });
  }
  if (pp !== undefined && s.L !== undefined && s.L % pp !== 0) {
    w.push({ params: ["pp"], message: `pp must divide L=${s.L}` });
  }
  const T = n("T");
  const batch = n("batch");
  if (T !== undefined && batch !== undefined && T % batch !== 0) {
    w.push({ params: ["T", "batch"], message: "T must be divisible by batch" });
  }
  return w;
}

// ------------------------------------------------------------------ shapes

export type Shape = string[] | PhaseShape;

export function isPhaseShape(s: Shape | null | undefined): s is PhaseShape {
  return !!s && !Array.isArray(s);
}

/** Symbolic dims for a phase; arrays keep their T placeholders (decode substitutes T=S on eval). */
export function shapeDims(shape: Shape, phase: Phase): string[] {
  return isPhaseShape(shape) ? shape[phase] : shape;
}

export function phaseEnv(env: ExprEnv, phase: Phase, shape?: Shape | null): ExprEnv {
  if (phase === "decode" && (!shape || Array.isArray(shape)) && "S" in env) {
    return { ...env, T: env.S as number };
  }
  return env;
}

export function evalShape(shape: Shape, env: ExprEnv, phase: Phase): Q[] {
  const e = phaseEnv(env, phase, shape);
  return shapeDims(shape, phase).map((d) => q(d, e));
}

// ------------------------------------------------------------------ node metrics

export interface NodeMetrics {
  weight: Q;
  flops: Q;
  bytes: Q;
  intensity?: number;
  comm?: { active: boolean; logical: Q; bus: Q };
}

export function nodeMetrics(node: IRNode, env: ExprEnv, phase: Phase): NodeMetrics {
  const m: NodeMetrics = {
    weight: q(node.weight_bytes, env),
    flops: q(node.flops?.[phase], env),
    bytes: q(node.bytes_moved?.[phase], env),
  };
  if (m.flops.v !== undefined && m.bytes.v !== undefined && m.bytes.v > 0 && m.flops.v > 0) {
    m.intensity = m.flops.v / m.bytes.v;
  }
  if (node.comm) {
    const when = node.comm.when
      ? tryEvaluate(node.comm.when, env)
      : { ok: true as const, value: 1 };
    m.comm = {
      active: when.ok ? when.value !== 0 : false,
      logical: q(node.comm.logical_bytes?.[phase], env),
      bus: q(node.comm.bus_bytes?.[phase], env),
    };
  }
  return m;
}

export function computeAll(ir: IR, env: ExprEnv, phase: Phase): Map<string, NodeMetrics> {
  const out = new Map<string, NodeMetrics>();
  for (const n of ir.nodes) out.set(n.id, nodeMetrics(n, env, phase));
  return out;
}

/** Same expression evaluated with all parallel degrees = 1 (the unsharded size). */
export function unshardedEnv(env: ExprEnv): ExprEnv {
  const e = { ...env };
  for (const k of ["tp", "pp", "dp", "ep"]) if (k in e) e[k] = 1;
  return e;
}

export interface MemoryLedger {
  weights: Q;
  activation: Q;
  nonTorch: Q;
  reserved: Q;
  kvPerToken: Q;
  kvTokens: Q;
  kvBytes: Q;
  budget: Q;
  hbm: Q;
}

export function memoryLedger(ir: IR, env: ExprEnv): MemoryLedger {
  const m = ir.memory;
  const kvPerToken = q(m.kv_per_token, env);
  const kvTokens = q(m.kv_tokens, env);
  const kvBytes: Q =
    kvPerToken.v !== undefined && kvTokens.v !== undefined
      ? { v: kvPerToken.v * kvTokens.v }
      : { e: kvPerToken.e ?? kvTokens.e ?? "?" };
  return {
    weights: q(m.weights, env),
    activation: q(m.activation_peak, env),
    nonTorch: q(m.non_torch, env),
    reserved: q(m.reserved, env),
    kvPerToken,
    kvTokens,
    kvBytes,
    budget:
      "hbm_bytes" in env ? q("hbm_bytes*gpu_memory_utilization", env) : { e: "no hbm_bytes param" },
    hbm: "hbm_bytes" in env ? q("hbm_bytes", env) : { e: "no hbm_bytes param" },
  };
}

// ------------------------------------------------------------------ JSON pointers (eval oracle)

/** Evaluate an eval.json pointer such as `/nodes/attn.o.reduce/comm/bus_bytes/prefill`. */
export function evalPointer(ir: IR, env: ExprEnv, pointer: string): number | number[] {
  const parts = pointer
    .split("/")
    .slice(1)
    .map((p) => p.replace(/~1/g, "/").replace(/~0/g, "~"));
  let cur: unknown = ir;
  if (parts[0] === "nodes") {
    cur = ir.nodes.find((n) => n.id === parts[1]);
    parts.splice(0, 2);
  }
  let phase: Phase = "prefill";
  for (const p of parts) {
    if (p === "decode" || p === "prefill") phase = p;
    if (cur === null || cur === undefined) throw new Error(`missing ${pointer}`);
    cur = Array.isArray(cur) ? cur[Number(p)] : (cur as Record<string, unknown>)[p];
  }
  if (typeof cur === "string") return evaluate(cur, env);
  if (Array.isArray(cur) || (cur && typeof cur === "object")) {
    return evalShape(cur as Shape, env, phase).map((x) => {
      if (x.v === undefined) throw new Error(x.e);
      return x.v;
    });
  }
  throw new Error(`unsupported pointer ${pointer}`);
}

export { compile };

/**
 * Concurrent sequences that fit the KV capacity: each sequence holds whole cache blocks, so
 * `floor(kv_tokens / (ceil((context+1)/block_size) * block_size))`. Without a block_size
 * param the token count is used unrounded.
 */
export function seqCapacity(kvTokens: number | undefined, env: ExprEnv): number | undefined {
  const ctx = env.context;
  if (kvTokens === undefined || typeof ctx !== "number") return undefined;
  const block = typeof env.block_size === "number" && env.block_size > 0 ? env.block_size : 1;
  const perSeq = Math.ceil((ctx + 1) / block) * block;
  return Math.floor(kvTokens / Math.max(1, perSeq));
}

export interface KvSweepRow {
  tp: number;
  weights: number;
  activation: number;
  other: number;
  kv: number;
  tokens: number;
  seqs: number;
}

/**
 * Per-rank memory split for every TP choice, keeping the user's other overrides (KV tab).
 * EP stays legal: enabled EP follows ep = tp*dp, disabled stays 1.
 */
export function kvSweep(ir: IR, state: EnvState): KvSweepRow[] {
  const spec = ir.params.tp;
  const choices = (spec?.choices ?? ["1", "2", "4", "8"]).map(Number);
  const overrides: Overrides = {};
  for (const [k, v] of Object.entries(state.params)) if (v !== state.defaults[k]) overrides[k] = v;
  const dp = Number(state.params.dp ?? 1);
  const ep = Number(state.params.ep ?? 1);
  return choices.map((tp) => {
    const e = buildEnv(ir, {
      ...overrides,
      tp,
      ...("ep" in ir.params ? { ep: ep > 1 ? tp * dp : 1 } : {}),
    });
    const l = memoryLedger(ir, e.env);
    return {
      tp,
      weights: l.weights.v ?? 0,
      activation: l.activation.v ?? 0,
      other: (l.nonTorch.v ?? 0) + (l.reserved.v ?? 0),
      kv: l.kvBytes.v ?? 0,
      tokens: l.kvTokens.v ?? 0,
      seqs: seqCapacity(l.kvTokens.v ?? 0, e.env) ?? 0,
    };
  });
}
