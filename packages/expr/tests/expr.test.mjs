import { execFileSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import {
  compile,
  ExprError,
  evaluate,
  evaluator,
  format,
  formatHuman,
  free,
  parse,
  UnknownSymbolError,
} from "../dist/index.js";

const root = new URL("../../../", import.meta.url);
const load = (path) => JSON.parse(readFileSync(new URL(path, root), "utf8"));
const vectors = load("packages/expr/test-vectors.json");

describe("shared DSL vectors", () => {
  it("contains at least 150 cases", () => expect(vectors.length).toBeGreaterThanOrEqual(150));
  it.each(vectors.map((vector, index) => ({ ...vector, index })))("$index: $expr", (vector) => {
    const { expr, env, value, error } = vector;
    if (error) {
      for (const run of [
        () => evaluate(expr, env),
        () => compile(expr)(env),
        () => evaluate(parse(expr), env),
      ]) {
        expect(run).toThrow(ExprError);
        try {
          run();
        } catch (caught) {
          expect(caught.code).toBe(error);
          expect(caught.position.offset).toBeGreaterThanOrEqual(0);
          expect(caught.position.line).toBeGreaterThanOrEqual(1);
          expect(caught.position.column).toBeGreaterThanOrEqual(1);
        }
      }
    } else {
      expect(evaluate(expr, env)).toBe(value);
      expect(compile(expr)(env)).toBe(value);
      expect(evaluate(parse(expr), env)).toBe(value);
      expect(format(expr, "numeric", env)).toBe(String(value));
      expect(evaluate(format(expr, "symbolic"), env)).toBe(value);
      if (vector.free) expect(free(expr)).toEqual(vector.free);
      if (vector.symbolic) expect(format(expr, "symbolic")).toBe(vector.symbolic);
    }
  });
});

it("matches every oracle pointer and prevents fixture-vector drift", () => {
  const ir = load("fixtures/qwen3-30b-a3b/ir.json");
  const oracle = load("fixtures/qwen3-30b-a3b/eval.json");
  const projection = { ...ir, nodes: Object.fromEntries(ir.nodes.map((n) => [n.id, n])) };
  const resolve = (object, path) =>
    path
      .split("/")
      .slice(1)
      .reduce((obj, key) => obj[key.replaceAll("~1", "/").replaceAll("~0", "~")], object);
  const env = Object.fromEntries(
    Object.entries(ir.symbols).map(([key, expr]) => [key, evaluate(expr, {})]),
  );
  for (const [key, param] of Object.entries(ir.params)) {
    env[key] = ["dtype", "enum"].includes(param.kind)
      ? param.default
      : evaluate(param.default, env);
  }
  for (const test of oracle.cases) {
    for (const [pointer, expected] of Object.entries(test.expected)) {
      const expression = resolve(projection, pointer);
      const expressions = Array.isArray(expression) ? expression : [expression];
      const values = Array.isArray(expected) ? expected : [expected];
      for (const [index, expr] of expressions.entries()) {
        const path = Array.isArray(expression) ? `${pointer}/${index}` : pointer;
        const vector = vectors.find(
          (v) => v.fixture?.case === test.id && v.fixture.pointer === path,
        );
        expect(vector).toMatchObject({ expr, env: { ...env, ...test.env }, value: values[index] });
        expect(evaluate(expr, { ...env, ...test.env })).toBe(values[index]);
      }
    }
  }
});

it("evaluates every fixture metric and tensor dimension at TP1 and TP4", () => {
  const ir = load("fixtures/qwen3-30b-a3b/ir.json");
  const expressions = Object.values(ir.memory).filter((value) => typeof value === "string");
  for (const node of ir.nodes) {
    expressions.push(
      node.weight_bytes,
      ...Object.values(node.flops),
      ...Object.values(node.bytes_moved),
    );
    if (node.comm) {
      expressions.push(
        node.comm.when,
        ...Object.values(node.comm.logical_bytes),
        ...Object.values(node.comm.bus_bytes),
      );
    }
    for (const slot of [...node.in, ...node.out]) {
      expressions.push(
        ...(Array.isArray(slot.shape) ? slot.shape : [...slot.shape.prefill, ...slot.shape.decode]),
      );
    }
  }
  const env = vectors.find((vector) => vector.fixture).env;
  for (const tp of [1, 4]) {
    for (const src of expressions) {
      expect(free(src).every((name) => Object.hasOwn(env, name))).toBe(true);
      expect(Number.isFinite(compile(src)({ ...env, tp }))).toBe(true);
    }
  }
});

it("keeps cached ASTs immutable and positions out of the wire AST", () => {
  const ast = parse("max(x, 1)");
  expect(parse("max(x, 1)")).toBe(ast);
  expect(ast).toEqual({
    type: "call",
    name: "max",
    args: [
      { type: "identifier", name: "x" },
      { type: "number", value: 1 },
    ],
  });
  expect(() => {
    ast.args[0].name = "bad";
  }).toThrow();
  const env = Object.freeze({ x: 2 });
  expect(evaluate(ast, env)).toBe(2);
  expect(evaluator.evaluate("1", {})).toBe(1);
});

it("reports precise multiline source positions and typed unknown identifiers", () => {
  expect.assertions(5);
  try {
    compile("\n  3 + missing")({});
  } catch (error) {
    expect(error).toBeInstanceOf(UnknownSymbolError);
    expect(error.symbol).toBe("missing");
    expect(error.position).toEqual({ offset: 7, line: 2, column: 7 });
  }
  try {
    parse("1 +\n @");
  } catch (error) {
    expect(error.code).toBe("syntax");
    expect(error.position).toEqual({ offset: 5, line: 2, column: 2 });
  }
});

it("checks AST input even in lazy branches", () => {
  expect(() =>
    evaluate(
      {
        type: "conditional",
        test: { type: "boolean", value: false },
        consequent: { type: "call", name: "max", args: [] },
        alternate: { type: "number", value: 1 },
      },
      {},
    ),
  ).toThrow(expect.objectContaining({ code: "arity" }));
  expect(() => evaluate({ type: "number", value: Infinity }, {})).toThrow(
    expect.objectContaining({ code: "non_finite" }),
  );
  expect(() => evaluate("x", { x: NaN })).toThrow(expect.objectContaining({ code: "non_finite" }));
  expect(() => evaluate("x", { x: null })).toThrow(expect.objectContaining({ code: "type" }));
  expect(() => evaluate("1", Object.defineProperty({}, "bf16", { value: "bf16" }))).toThrow(
    expect.objectContaining({ code: "type" }),
  );
  expect(() => evaluate("x", Object.create({ x: 1 }))).toThrow(
    expect.objectContaining({ code: "unknown_symbol" }),
  );
});

it("compiled closures read each environment without capturing its values", () => {
  const run = compile("x > 0 ? x * 2 : -x");
  expect(run({ x: 4 })).toBe(8);
  expect(run({ x: -3 })).toBe(3);
  expect(free("flag ? bytes(dtype_w) : max(z, z, a)")).toEqual(["a", "dtype_w", "flag", "z"]);
  expect(format("x + 1", "symbolic", { fp32: 99 })).toBe("(x + 1)");
  expect(format("-0", "numeric")).toBe("0");
  expect(Object.is(evaluate("-0", {}), -0)).toBe(false);
});

it.each([
  ["0", "bytes", "0 B"],
  [".5", "bytes", "0.5 B"],
  ["1023", "bytes", "1023 B"],
  ["1024", "bytes", "1 KiB"],
  ["1536", "bytes", "1.5 KiB"],
  ["1048576", "bytes", "1 MiB"],
  ["1073741824", "bytes", "1 GiB"],
  ["-2048", "bytes", "-2 KiB"],
  ["1e9", "GFLOP", "1 GFLOP"],
  ["2e12", "TFLOP", "2 TFLOP"],
  ["12", "FLOP", "12 FLOP"],
])("formats display units: %s %s", (src, unit, expected) => {
  expect(formatHuman(src, unit)).toBe(expected);
});

it("evaluates 10,000 compiled expressions in under 50 ms in Node", () => {
  const output = execFileSync(process.execPath, ["scripts/benchmark.mjs"], {
    cwd: fileURLToPath(new URL("../", import.meta.url)),
    encoding: "utf8",
  });
  expect(JSON.parse(output).medianMs).toBeLessThan(50);
});
