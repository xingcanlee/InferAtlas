# @inferatlas/layout

Build-time Node layout from validated IR to curated `view.json`; no model imports,
weights, browser, DOM, or expression evaluation.

From the repository root (Node 22+, pnpm 10):

```sh
# Until the root lockfile owner incorporates this package's dependency changes:
pnpm install --lockfile=false --registry=https://registry.npmmirror.com
pnpm --filter @inferatlas/contracts build
pnpm --filter @inferatlas/layout build
pnpm --filter @inferatlas/layout inferatlas-layout ../../fixtures/qwen3-30b-a3b/ir.json -o /tmp/view.json
# Direct Node entry point (the package also declares the inferatlas-layout executable):
node packages/layout/dist/cli.js fixtures/qwen3-30b-a3b/ir.json -o /tmp/view.json
pnpm --filter @inferatlas/layout test
pnpm --filter @inferatlas/layout lint
```

```ts
import { layoutIR, serializeView, rankBoundaryLanes, checkBudget } from "@inferatlas/layout";

const view = await layoutIR(ir, { onWarning: (message) => console.warn(message) });
const json = serializeView(view); // sorted keys, stable arrays, trailing newline
const lanes = rankBoundaryLanes(view.views[0], ir);
const issues = checkBudget(view.views[0], 0); // caller supplies current highlight count
```

`layoutIR(unknown, {onWarning?})` checks Draft 2020-12 schemas and reference/hierarchy
invariants, then returns `Promise<View>`. Errors reject the promise; CLI errors exit 1
without writing output. Warnings use stderr by default. The compiled package loads
the repository's authoritative `schemas/{ir,view}.schema.json` relative to itself;
keep that repository structure when deploying this private build tool.

Dense/MoE templates use the semantic ids in `docs/expr-dsl.md`, with attention/MLP
class recognition as a fallback. L0 folds descendants into trunk summaries and labels
layer bands with their inclusive symbolic range. L1 is `L1:<band>` and exposes the
representative decoder's norm/attention/MLP groups. L2 attention merges Q/K head norms
and rotary application into their semantic parent; L2 MLP/MoE flattens implementation
wrappers into projection/activation/router/expert operations and collectives. Separate
Q/K/V tensor shapes retain separate edges. No output node is invented without an IR id.

`collapsed_of` is the sorted **entire represented scope**, including visible anchors
and suppressed descendants. The difference from visible `ir_node`s is the fidelity
ledger for details. Every edge has same-kind `ir_edges` provenance and a copied symbolic
shape. Internal collapsed data paths retain intervening collectives; KV loops are shown
only in their detailed view, and residual edges remain residual.

ELK 0.11.0 runs layered/ORTHOGONAL with seed 1, sorted inputs, fixed padding/spacing,
DOWN for L0 and RIGHT for L1/L2. Node sizes use a frozen 14px sans advance table, 8px
mono advances for symbolic shapes, and conservative 14px advances for unknown glyphs.
Multiline labels reserve space; all exported geometry snaps to 4px. The renderer can
draw neutral dashed rank-boundary segments from `rankBoundaryLanes`: each visible
collective straddles a segment perpendicular to the data flow. Segments are local
collective boundaries, not a physical rank assignment; the View schema has no lane
or style field. The helper derives them from `ir_node`, so `view.json` stays compliant.
No accent colors are emitted. `checkBudget` reports >9 nodes, >12 edges or >2 caller
accents; `assertBudget` and layout fail instead of truncating an oversized view.

MLA/hybrid/VLM are explicit generic L0/L1 fallbacks with a warning. Missing internals
also warn and do not generate made-up drill-downs. The Qwen3-8B fixture currently has
no decoder children: its L1 repeats the known summary, while full dense attention/MLP
templates are exercised with synthetic IR in tests. See `docs/requests/T4.md`.

Tests cover both fixtures, separate CLI processes, reordered input arrays, schema,
routes/attachments/grid/budgets, provenance, multiple bands, all fallbacks, invalid
inputs, collapsed communication and the fidelity ledger. Generated views live only
in `test/__snapshots__/*.view.json.snap`. Refresh with
`pnpm --filter @inferatlas/layout exec vitest run --update`; The `.snap` suffix prevents Biome from changing generated
JSON formatting, so these remain the exact serializer output.
