# Golden format

Each model directory contains `ir.json` (the IR contract) and `eval.json` (the Eval
contract). The initial `qwen3-30b-a3b` baseline is normalized from the reviewed T0
fixtures, including their six named TP/DP/EP parameter points. TP1 and TP4 include
rank-local memory, weights, attention shapes/work and communication; the additional
cases isolate EP/DP communication. This is a curated regression baseline, not NPU
calibration or complete coverage of all parameter combinations.

Eval `env` values override declared runtime parameters. `expected` keys are JSON
pointers into an evaluated IR projection whose `nodes` are indexed by id. Dots in
node ids are literal; escape slashes and tildes as `~1` and `~0`. A pointer selects
one numeric expression or one array of shape expressions. Comm quantities retain
their `prefill`/`decode` components. Each case needs at least one pointer.

From the repository root:

```sh
uv run --project generator inferatlas golden check --ir fixtures/qwen3-30b-a3b/ir.json
uv run --project generator inferatlas golden update --ir path/to/ir.json
uv run --project generator inferatlas golden update --ir path/to/ir.json --grid path/to/eval.json
```

`--directory` selects an alternate baseline directory; the default is
`golden/<lowercase-model-basename>`. Update evaluates all requested pointers before
writing either file. Check compares the whole IR and re-evaluates the committed
grid; it prints differences per node/case and field and exits 3 on drift. All JSON
uses sorted keys, stable case ordering, wire aliases and no timestamps. Inspect and
commit an update like any other reviewed source change.

Evaluation first invokes `node packages/expr/dist/cli.js`, then falls back with a
warning to `inferatlas.expr.evaluate(expression, env)` if available. While both T2
entry points are absent, it uses the independent `fixtures/.../eval.json` oracle
only for matching fixture symbols, parameter definitions, expressions and env
points. It refuses to invent values for changed formulas or new grid points.
Verification reports this last mode as an evaluation **skip**, never fresh runtime
evidence. No second DSL evaluator is implemented here.

The Node CLI protocol and all pipeline integration signatures are recorded in
`docs/progress/T6.md` and requested in `docs/requests/T6.md`.
