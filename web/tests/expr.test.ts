import { describe, expect, it } from "vitest";
import { evaluate, format, free } from "@/lib/expr";

describe("expr adapter", () => {
  it.each([
    ["-2**2", -4],
    ["2**-2", 0.25],
    ["2**3**2", 512],
    ["cdiv(7, 2)", 4],
    ["round_up(151936, 64)", 151936],
    ["bytes(int4)", 0.5],
    ["1 == true", 0],
    ["tp > 1 ? 1/0 : 3", 3],
    ["-7 % 3", -1],
  ])("%s = %d", (src, v) => {
    expect(evaluate(src, { tp: 1 })).toBe(v);
  });
  it("errors", () => {
    expect(() => evaluate("x+1", {})).toThrow();
    expect(() => evaluate("1/0", {})).toThrow();
    expect(() => evaluate("max()", {})).toThrow();
    expect(() => evaluate("1 2", {})).toThrow();
    expect(() => evaluate("1 < 2 < 3", {})).toThrow();
  });
  it("free and format", () => {
    expect(free("bytes(dtype_w)*H + max(1, Hkv/tp)")).toEqual(["H", "Hkv", "dtype_w", "tp"]);
    expect(evaluate(format("(a+b)*c", "symbolic"), { a: 1, b: 2, c: 3 })).toBe(9);
    expect(format("H*2", "numeric", { H: 3 })).toBe("6");
  });
});
