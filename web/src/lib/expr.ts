/**
 * Expression adapter over `@inferatlas/expr` (T2). The rest of the UI talks only to this
 * module so evaluation stays in one place: compiled closures are cached per source string,
 * so live recomputation never re-parses.
 */
import type { ExprEnv } from "@inferatlas/contracts";
import { compile as compileExpr, format as formatExpr, free as freeExpr } from "@inferatlas/expr";

const compiled = new Map<string, (env: ExprEnv) => number>();

/** Compile once per distinct formula; parse/arity errors are thrown here. */
export function compile(src: string): (env: ExprEnv) => number {
  let fn = compiled.get(src);
  if (!fn) {
    fn = compileExpr(src);
    compiled.set(src, fn);
  }
  return fn;
}

export function evaluate(src: string, env: ExprEnv): number {
  return compile(src)(env);
}

export function format(src: string, mode: "symbolic" | "numeric", env?: ExprEnv): string {
  return formatExpr(src, mode, env);
}

export function free(src: string): string[] {
  return freeExpr(src);
}

export type EvalResult = { ok: true; value: number } | { ok: false; error: string; code?: string };

export function tryEvaluate(src: string, env: ExprEnv): EvalResult {
  try {
    return { ok: true, value: evaluate(src, env) };
  } catch (e) {
    const err = e as { message?: string; code?: string };
    return { ok: false, error: err.message ?? String(e), code: err.code };
  }
}
