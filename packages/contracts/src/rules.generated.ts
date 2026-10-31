/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Work for one decode step of S local-attention sequences.
 */
export type Decode = string;
/**
 * Work per prefill chunk of T local-attention tokens.
 */
export type Prefill = string;
/**
 * Local-rank estimated link traffic, includes collective algorithm factor; zero when inactive.
 */
export type BusBytes = string;
/**
 * Parallel group name (tp/ep/pp/dp) or a rule-declared group.
 */
export type Group = string;
/**
 * Local-rank logical tensor bytes per prefill invocation; use T=S for decode.
 */
export type LogicalBytes = string;
/**
 * Ring/dispatch/routing assumptions, especially EP uniform routing.
 */
export type Notes = string[];
/**
 * Logical collective; fused kernels retain semantic communication metadata.
 */
export type Op =
  "all_reduce" | "all_gather" | "reduce_scatter" | "all_to_all" | "send_recv" | "none";
/**
 * Activation condition; inactive communication contributes zero.
 */
export type When = string;
/**
 * Stable rule key; multiple matching rules are an error, never pick by file order.
 */
export type Id = string;
/**
 * PEP 440 SpecifierSet for vllm-ascend; empty means any.
 */
export type Ascend = string;
/**
 * Exact qualified class or exact short class name; qualified match wins, no glob/regex matching.
 */
export type Class = string;
/**
 * PEP 440 SpecifierSet; empty means any. For development commits use base package version plus snapshot identity.
 */
export type Vllm = string;
/**
 * Additional DSL applicability predicate over declared symbols/attrs/params.
 */
export type When1 = string;
/**
 * Assumptions and explanation of non-obvious formula terms.
 */
export type Notes1 = string[];
/**
 * Dtype symbol or builtin dtype name, consumed by bytes().
 */
export type Dtype = string;
/**
 * Local input/output slot name; multiple inputs/outputs stay separate.
 */
export type Name = string;
/**
 * Local-rank dimensions as DSL expressions, in dimension order.
 */
export type Shape = string[];
/**
 * Output tensor slots with expression dimensions, same representation as IR.
 */
export type Out = TensorSpec[];
/**
 * Resident local-rank weight bytes expression, including scale/padding overhead when modelled.
 */
export type Weight = string;
/**
 * Rule YAML files contain lists sorted by id; no implicit version-range directory semantics.
 */
export type Rules = Rule[];

export interface Rule {
  bytes_moved: PhaseExpr;
  /**
   * Collective metadata or null; same representation as IR.
   */
  comm?: Comm | null;
  flops: PhaseExpr1;
  id: Id;
  match: RuleMatch;
  notes?: Notes1;
  out: Out;
  shard?: Shard;
  weight: Weight;
}
/**
 * Phase-specific estimated HBM byte expressions.
 */
export interface PhaseExpr {
  decode: Decode;
  prefill: Prefill;
}
export interface Comm {
  bus_bytes: BusBytes;
  group: Group;
  logical_bytes: LogicalBytes;
  notes?: Notes;
  op: Op;
  when?: When;
}
/**
 * Phase-specific FLOP expressions, not a scalar shorthand.
 */
export interface PhaseExpr1 {
  decode: Decode;
  prefill: Prefill;
}
/**
 * Class, upstream version ranges and optional predicate.
 */
export interface RuleMatch {
  ascend?: Ascend;
  class: Class;
  vllm?: Vllm;
  when?: When1;
}
export interface TensorSpec {
  dtype?: Dtype;
  name: Name;
  shape: Shape;
}
/**
 * Parallel axis to shard dimension/strategy, converted into IR parallel axes.
 */
export interface Shard {
  [k: string]: string;
}
