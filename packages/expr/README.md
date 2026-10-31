# @inferatlas/expr

The sole production evaluator for the [InferAtlas DSL](../../docs/expr-dsl.md).
ESM, tree-shakeable named exports, no runtime imports or dynamic code generation.
`@inferatlas/contracts` is a package dependency because the public declarations re-export its types.
Peggy is a build dependency; its generated standalone parser is committed and regenerated
at build time. [Peggy's documentation](https://peggyjs.org/documentation.html) describes
this runtime-free generation mode.

```ts
import { compile, evaluate, format, formatHuman, free, parse } from "@inferatlas/expr";

const src = "T * H / tp * bytes(dtype_a)";
const env = { T: 4096, H: 2048, tp: 4, dtype_a: "bf16" };
evaluate(src, env);             // 4194304
compile(src)(env);              // 4194304; reuse this closure when controls change
free(src);                     // ["H", "T", "dtype_a", "tp"]
format(src, "numeric", env);   // "4194000" (4 significant digits)
format(src, "numeric", env, 7); // "4194304"
formatHuman(src, "bytes", env); // "4 MiB"
```

`parse(src)` returns the exact contracts `Ast` union, deeply frozen and cached by the
complete source string. `evaluate` also accepts an AST and checks all branches for valid
node types, operators, literal types, arity and finite literals, including skipped branches.
Bare function names such as `max + 1` or `bytes` fail with `syntax`; functions require calls.
The exported `evaluator` object satisfies `ExprEvaluator`.
`compile` builds closures over the AST once, without `eval` or `new Function`; it reads each
supplied environment afresh. Inputs are not mutated. Cache entries live for the module's
lifetime; consumers should compile a finite set of IR formulas rather than arbitrary streams.

Symbolic formatting fully parenthesizes operators, uses one space around binary/ternary
operators and after commas, and retains identifier names. It ignores the environment.
`format(src, "numeric", env?, precision = 4)` rounds to significant digits and trims trailing
fractional zeros, with no locale separators or units. Precision must be an integer from 1 to
100; invalid precision raises `ExprError` with code `domain`. Negative zero prints as `0`.
For example, `format("1 / 3", "numeric")` returns `"0.3333"`; precision 6 returns `"0.333333"`.
`evaluate` and `compile` retain full precision; precision 17 in `format` preserves the numeric
result when parsed back. Symbolic formatting ignores precision.

`formatHuman(src, "bytes", env?, precision = 4)` selects B/KiB/MiB/GiB (capped at GiB), and
`"FLOP"`, `"GFLOP"`, `"TFLOP"` select explicit decimal FLOP scales. It rounds after scaling
with the same precision and zero trimming: `formatHuman("1234567890", "bytes")` returns
`"1.15 GiB"`, or `"1.14978 GiB"` at precision 6. Scientific notation may appear for extreme
magnitudes. These review-requested formatting semantics supersede the original exact numeric
formatting description in `docs/expr-dsl.md`; that T0-owned document is unchanged here.

`ExprError.code` is one of `syntax`, `unknown_symbol`, `type`, `arity`, `domain`,
`non_finite`. Every error has `position: {offset, line, column}` (zero-based offset,
one-based line/column). Missing identifiers throw `UnknownSymbolError` with `symbol`.
Positions are stored outside the AST in a WeakMap. Environment binding errors and external
ASTs without source metadata use `{offset: 0, line: 1, column: 1}`. Builtin names cannot be
rebound, inherited environment properties are ignored, and unused branches are lazy.
Error codes are the cross-language contract; positions and error prose are not. Peggy and
Lark may report different offsets for the same invalid expression.

The Python package `inferatlas.expr` exposes only `validate(src) -> list[str]` and
`ExprSyntaxError`. It uses [Lark](https://lark-parser.readthedocs.io/en/stable/grammar.html)
for syntax/arity validation and returns sorted unique free identifiers across all branches.
Runtime types, arithmetic domains, and environment resolution belong to TS. The private
`_reference.py` interpreter is imported only by pytest for cross-language checks; production
must never use it to evaluate IR metrics.

Run from the repository root:

```sh
pnpm install --frozen-lockfile --registry=https://registry.npmmirror.com
pnpm --filter @inferatlas/contracts build
pnpm --filter @inferatlas/expr test
uv run --project generator pytest generator/tests/test_expr.py
pnpm --filter @inferatlas/expr benchmark
pnpm exec biome check packages/expr
uv run --project generator ruff check generator/inferatlas/expr generator/tests/test_expr.py
```

`test-vectors.json` has 330 cases shared by Vitest/pytest. Standard keys are
`expr`, `env`, and either `value` or an error-code string `error`. Optional `free` and
`symbolic` pin those API outputs. `fixture` records an oracle case id and JSON pointer.
All six Qwen3-MoE oracle cases are represented, including TP1/TP4 memory, weight, FLOP,
shape and phase-specific communication expectations. The Vitest test checks vector drift
against `ir.json` and `eval.json` directly. Fixture values are copied, never recomputed
by the implementation under test.

Regenerate vectors with `uv run --project generator python packages/expr/scripts/vectors.py`;
regenerate the parser with `pnpm --filter @inferatlas/expr generate`. Both scripts produce
stable bytes, formatted by the pinned workspace Biome. The generated parser's lint rules
are suppressed; edit `src/grammar.peggy`, not the generated JavaScript.

The benchmark compiles 10,000 distinct sources built from the real fixture formulas, then
measures a full recomputation in standalone Node, including environment/type/domain checks.
Parsing/compilation is excluded. It asserts a median below 50 ms over seven batches after
three warmup batches and verifies the checksum each time. The Vitest suite runs this gate.
