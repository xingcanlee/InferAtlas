/* Generated from schemas; edit generator/inferatlas/schema instead. */

/**
 * Stable named evaluation case.
 */
export type Id = string;
/**
 * Golden numeric examples in stable case id order; no evaluator implementation in Python.
 */
export type Cases = EvalCase[];
/**
 * Fixture IR snapshot.model.hf_id.
 */
export type Model = string;

export interface Eval {
  cases: Cases;
  model: Model;
}
export interface EvalCase {
  env: Env;
  expected: Expected;
  id: Id;
}
/**
 * Overrides of IR params; bind IR symbols and defaults first. Raw values, not expressions.
 */
export interface Env {
  [k: string]: number | string | boolean;
}
/**
 * Expected values keyed by memory.<field> or nodes.<id>.<field>; shape values are arrays, e.g. nodes.attn.qkv.out.0.shape.
 */
export interface Expected {
  [k: string]: number | number[];
}
