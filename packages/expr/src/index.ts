// biome-ignore-all lint/style/noNonNullAssertion: Builtin arity and dtype membership are checked before indexing.
import type { Ast, ExprEnv, ExprEvaluator } from "@inferatlas/contracts";
import { parse as parseGenerated } from "./parser.js";

export type { Ast, ExprEnv, ExprEvaluator } from "@inferatlas/contracts";
export type ErrorCode = "syntax" | "unknown_symbol" | "type" | "arity" | "domain" | "non_finite";
export interface Position {
  /** Zero-based source offset; line and column are one-based. AST-only errors use the origin. */
  offset: number;
  line: number;
  column: number;
}

export class ExprError extends Error {
  constructor(
    public readonly code: ErrorCode,
    message: string,
    public readonly position: Position = { offset: 0, line: 1, column: 1 },
  ) {
    super(`${message} at ${position.line}:${position.column}`);
    this.name = "ExprError";
  }
}

export class UnknownSymbolError extends ExprError {
  constructor(
    public readonly symbol: string,
    position?: Position,
  ) {
    super("unknown_symbol", `Unknown identifier: ${symbol}`, position);
    this.name = "UnknownSymbolError";
  }
}

type Value = number | string | boolean;
type Runner = (env: ExprEnv) => Value;
const dtypeBytes: Readonly<Record<string, number>> = Object.freeze({
  fp32: 4,
  bf16: 2,
  fp16: 2,
  fp8: 1,
  int8: 1,
  int4: 0.5,
  int32: 4,
  int64: 8,
});
const functions = new Set(["max", "min", "ceil", "floor", "cdiv", "round_up", "bytes"]);
const reserved = new Set([...Object.keys(dtypeBytes), ...functions, "true", "false"]);
const positions = new WeakMap<Ast, Position>();
const astCache = new Map<string, Ast>();

function fail(code: ErrorCode, message: string, ast: Ast): never {
  throw new ExprError(code, message, positions.get(ast));
}

function finite(value: number, ast: Ast): number {
  if (!Number.isFinite(value)) fail("non_finite", "Expected a finite number", ast);
  return value;
}

function numeric(value: Value, ast: Ast): number {
  if (typeof value !== "number") fail("type", "Expected a number", ast);
  return finite(value, ast);
}

function truth(value: Value, ast: Ast): boolean {
  if (typeof value === "boolean") return value;
  return numeric(value, ast) !== 0;
}

function children(ast: Ast): Ast[] {
  switch (ast.type) {
    case "unary":
      return [ast.argument];
    case "binary":
      return [ast.left, ast.right];
    case "conditional":
      return [ast.test, ast.consequent, ast.alternate];
    case "call":
      return ast.args;
    default:
      return [];
  }
}

// Check every branch before execution, including branches that will be skipped.
function check(ast: Ast): void {
  if (!ast || typeof ast !== "object") fail("syntax", "Expected an AST node", ast);
  switch (ast.type) {
    case "number":
      numeric(ast.value, ast);
      break;
    case "boolean":
      if (typeof ast.value !== "boolean") fail("type", "Expected a boolean literal", ast);
      break;
    case "identifier":
      if (typeof ast.name !== "string" || !/^[a-zA-Z_][a-zA-Z0-9_]*$/.test(ast.name)) {
        fail("syntax", "Invalid identifier", ast);
      }
      if (["NaN", "Infinity", "true", "false"].includes(ast.name)) {
        fail("syntax", `${ast.name} is not a DSL identifier`, ast);
      }
      if (functions.has(ast.name)) fail("syntax", "Function names require parentheses", ast);
      break;
    case "unary":
      if (!["+", "-", "!"].includes(ast.op)) fail("syntax", "Unknown unary operator", ast);
      break;
    case "binary":
      if (
        !["+", "-", "*", "/", "%", "**", "<", "<=", ">", ">=", "==", "!=", "&&", "||"].includes(
          ast.op,
        )
      ) {
        fail("syntax", "Unknown binary operator", ast);
      }
      break;
    case "conditional":
      break;
    case "call": {
      if (!functions.has(ast.name)) fail("syntax", "Unknown function", ast);
      if (!Array.isArray(ast.args)) fail("syntax", "Expected function arguments", ast);
      const n = ast.args.length;
      const valid =
        ast.name === "max" || ast.name === "min"
          ? n >= 1
          : ast.name === "cdiv" || ast.name === "round_up"
            ? n === 2
            : n === 1;
      if (!valid) fail("arity", `Invalid arity for ${ast.name}`, ast);
      break;
    }
    default:
      fail("syntax", "Unknown AST node type", ast);
  }
  for (const child of children(ast)) check(child);
}

function freeze(ast: Ast): Ast {
  for (const child of children(ast)) freeze(child);
  if (ast.type === "call") Object.freeze(ast.args);
  return Object.freeze(ast);
}

/** Cached ASTs are deeply frozen; source positions are kept outside the wire AST. */
export function parse(src: string): Ast {
  const cached = astCache.get(src);
  if (cached) return cached;
  let ast: Ast;
  try {
    ast = parseGenerated(src, { positions }) as Ast;
  } catch (error) {
    const detail = error as { message: string; location?: { start: Position } };
    throw new ExprError("syntax", detail.message, detail.location?.start);
  }
  check(ast);
  freeze(ast);
  astCache.set(src, ast);
  return ast;
}

function checkEnv(env: ExprEnv): void {
  for (const key of Object.getOwnPropertyNames(env)) {
    if (reserved.has(key)) throw new ExprError("type", `Cannot rebind builtin: ${key}`);
  }
}

function runner(ast: Ast): Runner {
  switch (ast.type) {
    case "number":
    case "boolean":
      return () => ast.value;
    case "identifier": {
      const name = ast.name;
      if (Object.hasOwn(dtypeBytes, name)) return () => name;
      return (env) => {
        if (!Object.hasOwn(env, name)) throw new UnknownSymbolError(name, positions.get(ast));
        const value = env[name];
        if (typeof value === "number") return finite(value, ast);
        if (typeof value === "boolean" || typeof value === "string") return value;
        return fail("type", `Invalid value for ${name}`, ast);
      };
    }
    case "unary": {
      const arg = runner(ast.argument);
      if (ast.op === "!") return (env) => !truth(arg(env), ast.argument);
      return (env) =>
        ast.op === "-" ? -numeric(arg(env), ast.argument) : numeric(arg(env), ast.argument);
    }
    case "conditional": {
      const test = runner(ast.test),
        yes = runner(ast.consequent),
        no = runner(ast.alternate);
      return (env) => (truth(test(env), ast.test) ? yes(env) : no(env));
    }
    case "binary": {
      const left = runner(ast.left),
        right = runner(ast.right);
      if (ast.op === "&&")
        return (env) => truth(left(env), ast.left) && truth(right(env), ast.right);
      if (ast.op === "||")
        return (env) => truth(left(env), ast.left) || truth(right(env), ast.right);
      if (ast.op === "==") return (env) => left(env) === right(env);
      if (ast.op === "!=") return (env) => left(env) !== right(env);
      return (env) => {
        const a = numeric(left(env), ast.left),
          b = numeric(right(env), ast.right);
        switch (ast.op) {
          case "+":
            return finite(a + b, ast);
          case "-":
            return finite(a - b, ast);
          case "*":
            return finite(a * b, ast);
          case "/":
          case "%":
            if (b === 0) fail("domain", "Division or remainder by zero", ast);
            return finite(ast.op === "/" ? a / b : a % b, ast);
          case "**":
            return finite(a ** b, ast);
          case "<":
            return a < b;
          case "<=":
            return a <= b;
          case ">":
            return a > b;
          case ">=":
            return a >= b;
          default:
            return fail("syntax", "Unknown binary operator", ast);
        }
      };
    }
    case "call": {
      const args = ast.args.map(runner);
      if (ast.name === "bytes")
        return (env) => {
          const value = args[0]!(env);
          if (typeof value !== "string") fail("type", "bytes expects a dtype string", ast);
          if (!Object.hasOwn(dtypeBytes, value)) fail("domain", `Unknown dtype: ${value}`, ast);
          return dtypeBytes[value]!;
        };
      return (env) => {
        const values = args.map((arg, i) => numeric(arg(env), ast.args[i]!));
        const a = values[0]!,
          b = values[1]!;
        switch (ast.name) {
          case "max":
            return Math.max(...values);
          case "min":
            return Math.min(...values);
          case "ceil":
            return Math.ceil(a);
          case "floor":
            return Math.floor(a);
          case "cdiv":
          case "round_up":
            if (a < 0 || b <= 0) fail("domain", `${ast.name} requires x >= 0 and divisor > 0`, ast);
            return finite(Math.ceil(a / b) * (ast.name === "round_up" ? b : 1), ast);
          default:
            return fail("syntax", "Unknown function", ast);
        }
      };
    }
  }
}

function executable(ast: Ast): (env: ExprEnv) => number {
  const run = runner(ast);
  return (env) => {
    checkEnv(env);
    const value = run(env);
    const result = typeof value === "boolean" ? Number(value) : numeric(value, ast);
    return result === 0 ? 0 : result;
  };
}

/** Compile closures once for hot paths. Never uses eval or the Function constructor. */
export function compile(src: string): (env: ExprEnv) => number {
  return executable(parse(src));
}

export function evaluate(src: string | Ast, env: ExprEnv): number {
  const ast = typeof src === "string" ? parse(src) : src;
  if (typeof src !== "string") check(ast);
  return executable(ast)(env);
}

export function free(src: string): string[] {
  const names = new Set<string>();
  function visit(ast: Ast): void {
    if (ast.type === "identifier" && !reserved.has(ast.name)) names.add(ast.name);
    for (const child of children(ast)) visit(child);
  }
  visit(parse(src));
  return [...names].sort();
}

function symbolic(ast: Ast): string {
  switch (ast.type) {
    case "number":
    case "boolean":
      return String(ast.value);
    case "identifier":
      return ast.name;
    case "unary":
      return `(${ast.op}${symbolic(ast.argument)})`;
    case "binary":
      return `(${symbolic(ast.left)} ${ast.op} ${symbolic(ast.right)})`;
    case "conditional":
      return `(${symbolic(ast.test)} ? ${symbolic(ast.consequent)} : ${symbolic(ast.alternate)})`;
    case "call":
      return `${ast.name}(${ast.args.map(symbolic).join(", ")})`;
  }
}

function formatNumber(value: number, precision: number): string {
  if (!Number.isInteger(precision) || precision < 1 || precision > 100) {
    throw new ExprError("domain", "Precision must be an integer from 1 to 100");
  }
  const rounded = value.toPrecision(precision);
  // Rounding near Number.MAX_VALUE can exceed the representable range.
  return Number.isFinite(Number(rounded))
    ? String(Number(rounded))
    : rounded.replace(/\.?0+e/, "e");
}

export function format(
  src: string,
  mode: "symbolic" | "numeric",
  env: ExprEnv = {},
  precision = 4,
): string {
  return mode === "symbolic" ? symbolic(parse(src)) : formatNumber(evaluate(src, env), precision);
}

/** Display-only units with significant-digit rounding after scaling. */
export function formatHuman(
  src: string,
  unit: "bytes" | "FLOP" | "GFLOP" | "TFLOP",
  env: ExprEnv = {},
  precision = 4,
): string {
  const value = evaluate(src, env);
  if (unit !== "bytes") {
    const scale = unit === "TFLOP" ? 1e12 : unit === "GFLOP" ? 1e9 : 1;
    return `${formatNumber(value / scale, precision)} ${unit}`;
  }
  const index = Math.min(3, Math.max(0, Math.floor(Math.log2(Math.abs(value) || 1) / 10)));
  return `${formatNumber(value / 1024 ** index, precision)} ${["B", "KiB", "MiB", "GiB"][index]}`;
}

export const evaluator = { parse, evaluate, free, format } satisfies ExprEvaluator;
