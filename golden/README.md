# Golden format

Each `golden/<lowercase-model-basename>/<device>/` directory contains `ir.json`
(the real pipeline IR) and `eval.json` (fresh Node evaluations). H100 is the
baseline without Ascend extraction; A2 includes observed Ascend replacements and
source-anchored annotations. The old fixture-seeded model-level golden is removed.
These are CPU/meta regression baselines, not measured hardware calibration.

The default grid is derived from the real IR and covers TP 1/2/4/8. It evaluates
memory expressions, node weights/work/traffic, input/output shapes, collective
conditions and both prefill/decode communication quantities. Custom grids can add
EP/DP points; all parameter choices and cross-parameter constraints are enforced.
Eval `env` values override declared runtime parameters. `expected` keys are JSON
pointers into an IR projection whose nodes are indexed by id. Dots are literal;
escape slashes/tildes as `~1`/`~0`. Each pointer selects an expression or shape array.

From the repository root:

```sh
pnpm --filter @inferatlas/contracts build
pnpm --filter @inferatlas/expr build
uv run --project generator inferatlas build --model Qwen/Qwen3-30B-A3B --device A2 --output .cache/review
# A first build without its golden exits 3 and leaves reviewable artifacts.
uv run --project generator inferatlas golden update --ir .cache/review/ir.json
uv run --project generator inferatlas golden check --ir .cache/review/ir.json
# Optional explicit grid or baseline destination:
uv run --project generator inferatlas golden update --ir .cache/review/ir.json --grid path/to/eval.json --directory path/to/baseline
```

Update reuses an existing grid or derives pointers from the input IR, then evaluates
all values before writing either file. It never seeds expected numbers from a
fixture. Check compares the full IR and freshly evaluated committed grid, prints
field differences, and exits 3 on drift. JSON uses sorted keys, stable ordering,
wire aliases and no timestamps. Review updates like source changes.

`verify/eval.mjs` imports T2's `evaluate` from `packages/expr/dist/index.js`.
There is no Python evaluator or numeric fixture fallback. An installed but broken
Node evaluator fails. Missing Node/dist fails verification; explicit fixture mode
permits structural-only checking, prints SKIPPED, and remains unpublishable.
`golden update` always refuses that unavailable backend.

For the 30B BF16 A2 model, TP1 legitimately has zero KV capacity: 56.9 GiB of
weights exceeds the 53.6 GiB utilization budget. Use TP4 for calibration examples.
See `docs/progress/T6.md` for the four real builds and exact commands.
