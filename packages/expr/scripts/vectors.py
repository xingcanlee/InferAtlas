"""Deterministically assemble explicit test cases and the independently pinned fixture oracle."""

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VECTORS = []


def value(expr, expected, env=None, **extra):
    VECTORS.append({"expr": expr, "env": env or {}, "value": expected, **extra})


def error(expr, code, env=None):
    VECTORS.append({"expr": expr, "env": env or {}, "error": code})


for expr, expected in [
    ("0", 0),
    ("123", 123),
    ("0012", 12),
    ("1.", 1),
    (".5", 0.5),
    ("1e3", 1000),
    ("2E-2", 0.02),
    (".5e+2", 50),
    ("1.e2", 100),
    ("-0", 0),
    ("+2", 2),
    ("--2", 2),
    ("+-2", -2),
    ("-+2", -2),
    ("1+2*3", 7),
    ("(1+2)*3", 9),
    ("9-3-2", 4),
    ("24/4/2", 3),
    ("8%3*2", 4),
    ("8/(3-1)", 4),
    ("2**3**2", 512),
    ("(2**3)**2", 64),
    ("-2**2", -4),
    ("(-2)**2", 4),
    ("2**-2", 0.25),
    ("-2**-2", -0.25),
    ("2*-3**2", -18),
    ("2**3*4", 32),
    ("2*3**2", 18),
    ("!2**0", 0),
    ("2**-2**2", 0.0625),
    ("5%2", 1),
    ("-5%2", -1),
    ("5%-2", 1),
    ("-5%-2", -1),
    ("5.5%2", 1.5),
    ("0**0", 1),
    ("1/4", 0.25),
    ("9007199254740991+1", 9007199254740992),
    ("1e-300*1e-30", 0),
    ("true", 1),
    ("false", 0),
    ("!false", 1),
    ("!!3", 1),
    ("!0", 1),
    ("true || false && false", 1),
    ("(true || false) && false", 0),
    ("1<2", 1),
    ("2<=2", 1),
    ("2>1", 1),
    ("2>=2", 1),
    ("2>2", 0),
    ("1+2==3", 1),
    ("1!=2", 1),
    ("1==true", 0),
    ("0==false", 0),
    ("true==true", 1),
    ("true!=1", 1),
    ("1==1.0", 1),
    ("1<2 == true", 1),
    ("1==1 && 2>1", 1),
    ("0 || 3", 1),
    ("2 && 3", 1),
    ("0 && missing", 0),
    ("1 || missing", 1),
    ("false && 1/0", 0),
    ("true || bytes(3)", 1),
    ("true ? 7 : missing", 7),
    ("false ? 1/0 : 8", 8),
    ("1 ? 2 : 3", 2),
    ("0 ? 2 : 3", 3),
    ("-1 ? 2 : 3", 2),
    ("false ? 1 : true ? 2 : 3", 2),
    ("true ? false ? 1 : 2 : 3", 2),
    ("0 || 1 ? 4 : 5", 4),
    ("true ? bf16 : fp16 == fp16", 0),
    ("max(1)", 1),
    ("max(1,3,2,-4)", 3),
    ("min(1)", 1),
    ("min(1,3,2,-4)", -4),
    ("max(min(2,3),1)", 2),
    ("ceil(1.1)", 2),
    ("ceil(-1.1)", -1),
    ("floor(1.9)", 1),
    ("floor(-1.1)", -2),
    ("ceil(0)", 0),
    ("floor(2)", 2),
    ("cdiv(0,3)", 0),
    ("cdiv(6,3)", 2),
    ("cdiv(7,3)", 3),
    ("cdiv(.5,.2)", 3),
    ("round_up(0,64)", 0),
    ("round_up(64,64)", 64),
    ("round_up(65,64)", 128),
    ("round_up(2.1,.5)", 2.5),
    ("round_up(cdiv(17,2),8)", 16),
    ("bf16==fp16", 0),
    ("bf16!=fp32", 1),
    ("\t1 +\r\n 2 ", 3),
    ("_x2 + foo", 9),
]:
    # A ternary selecting a string is a type error even if the other branch is boolean.
    if expr == "true ? bf16 : fp16 == fp16":
        error(expr, "type")
    else:
        value(expr, expected, {"_x2": 4, "foo": 5} if "_x2" in expr else None)

for dtype, size in {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
    "int32": 4,
    "int64": 8,
}.items():
    value(f"bytes({dtype})", size)
    value("bytes(dtype_w)", size, {"dtype_w": dtype}, free=["dtype_w"])
    value(f"dtype_w == {dtype}", 1, {"dtype_w": dtype})
    error("1", "type", {dtype: dtype})

value(
    "a ? z + a : bytes(dtype_w) + z",
    5,
    {"a": 1, "z": 4, "dtype_w": "fp16"},
    free=["a", "dtype_w", "z"],
)
value("true ? 1 : missing", 1, free=["missing"])
value("trueish + false_", 5, {"trueish": 2, "false_": 3}, free=["false_", "trueish"])
value("max_count + ceil2", 3, {"max_count": 1, "ceil2": 2})
value("a+b*2", 8, {"a": 2, "b": 3}, symbolic="(a + (b * 2))")
value(" 2 ** -2 ", 0.25, symbolic="(2 ** (-2))")
value("true?ceil(1.1):0", 2, symbolic="(true ? ceil(1.1) : 0)")
value("bytes( bf16 )", 2, symbolic="bytes(bf16)")
value("1e2 + .5", 100.5, symbolic="(100 + 0.5)")

for expr in [
    "",
    "max",
    "min",
    "ceil",
    "floor",
    "cdiv",
    "round_up",
    "bytes",
    "max + 1",
    "true ? 1 : bytes",
    "false && ceil",
    "max(1, min)",
    " ",
    "1+",
    "(1",
    "1)",
    "1 2",
    "1e",
    "1e+",
    ".",
    "0x10",
    "'bf16'",
    '"bf16"',
    "[1]",
    "{}",
    "x.y",
    "x[0]",
    "a=1",
    "2//1",
    "/*x*/1",
    "1 # hi",
    "abs(1)",
    "f(1)",
    "maxx(1)",
    "true(1)",
    "1?2",
    "1?:2",
    "1?2:",
    "1,2",
    "max(1,)",
    "max(,1)",
    "1 & 2",
    "1 | 2",
    "1 === 1",
    "NaN",
    "Infinity",
    "2***3",
    "2**",
    "é",
]:
    error(expr, "syntax")
for expr in [
    "max()",
    "min()",
    "ceil()",
    "ceil(1,2)",
    "floor()",
    "floor(1,2)",
    "cdiv(1)",
    "cdiv(1,2,3)",
    "round_up()",
    "round_up(1,2,3)",
    "bytes()",
    "bytes(bf16,fp16)",
    "true ? 1 : max()",
    "false && ceil()",
]:
    error(expr, "arity")
for expr in [
    "1/0",
    "1/-0",
    "1%0",
    "cdiv(-1,2)",
    "cdiv(1,0)",
    "cdiv(1,-1)",
    "round_up(-1,2)",
    "round_up(1,0)",
    "round_up(1,-1)",
]:
    error(expr, "domain")
for expr in [
    "true+1",
    "false*2",
    "-true",
    "true<2",
    "1<2<3",
    "bf16>fp16",
    "bf16 ? 1 : 0",
    "bf16 && true",
    "false || bf16",
    "!bf16",
    "bytes(2)",
    "bytes(true)",
    "max(1,true)",
    "min(bf16)",
    "ceil(false)",
    "floor(fp16)",
    "cdiv(true,2)",
    "round_up(1,false)",
    "bf16",
]:
    error(expr, "type")
for expr in [
    "1e309",
    "1e308*2",
    "0**-1",
    "(-1)**.5",
    "1e308**2",
    "cdiv(1e308,1e-308)",
    "round_up(1e308,1e-308)",
    "false ? 1e309 : 0",
]:
    error(expr, "non_finite")
for expr in ["missing", "toString", "constructor", "__proto__", "true && unknown"]:
    error(expr, "unknown_symbol")
error("bytes(dtype_w)", "domain", {"dtype_w": "bfloat16"})
error("x+1", "type", {"x": "2"})
for name in [
    "max",
    "min",
    "ceil",
    "floor",
    "cdiv",
    "round_up",
    "bytes",
    "true",
    "false",
]:
    error("1", "type", {name: 1})

ir = json.loads((ROOT / "fixtures/qwen3-30b-a3b/ir.json").read_text())
oracle = json.loads((ROOT / "fixtures/qwen3-30b-a3b/eval.json").read_text())
projection = {**ir, "nodes": {node["id"]: node for node in ir["nodes"]}}
base_env = {key: json.loads(expr) for key, expr in ir["symbols"].items()}
for key, param in ir["params"].items():
    base_env[key] = (
        param["default"]
        if param["kind"] in {"dtype", "enum"}
        else json.loads(param["default"])
    )
for case in oracle["cases"]:
    env = {**base_env, **case["env"]}
    for pointer, expected in sorted(case["expected"].items()):
        expr = projection
        for key in pointer.split("/")[1:]:
            key = key.replace("~1", "/").replace("~0", "~")
            expr = expr[int(key)] if isinstance(expr, list) else expr[key]
        exprs = (
            list(zip(expr, expected, strict=True))
            if isinstance(expr, list)
            else [(expr, expected)]
        )
        for index, (source, result) in enumerate(exprs):
            path = f"{pointer}/{index}" if isinstance(expr, list) else pointer
            value(source, result, env, fixture={"case": case["id"], "pointer": path})

formatted = subprocess.run(
    [
        "pnpm",
        "exec",
        "biome",
        "format",
        "--stdin-file-path=packages/expr/test-vectors.json",
    ],
    input=json.dumps(VECTORS, sort_keys=True, indent=2, ensure_ascii=False) + "\n",
    text=True,
    capture_output=True,
    check=True,
    cwd=ROOT,
).stdout
(ROOT / "packages/expr/test-vectors.json").write_text(formatted)
print(f"Wrote {len(VECTORS)} vectors")
