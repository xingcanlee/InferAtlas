# @inferatlas/layout

Build-time Node layout from validated IR to curated `view.json`; no model imports,
weights, browser, DOM, or expression evaluation.

From the repository root (Node 22.14+, pnpm 10):

```sh
pnpm install --frozen-lockfile --registry=https://registry.npmmirror.com
pnpm --filter @inferatlas/contracts build
pnpm --filter @inferatlas/layout build
pnpm --filter @inferatlas/layout inferatlas-layout ../../fixtures/qwen3-30b-a3b/ir.json -o /tmp/view.json
node packages/layout/dist/cli.js fixtures/qwen3-30b-a3b/ir.json -o /tmp/view.json
pnpm --filter @inferatlas/layout test
pnpm --filter @inferatlas/layout lint
```

```ts
import { layoutIR, serializeView, layerBandTargets, rankBoundaryLanes, checkBudget } from "@inferatlas/layout";

const view = await layoutIR(ir, { onWarning: (message) => console.warn(message) });
const json = serializeView(view); // sorted keys, stable arrays, trailing newline
const bands = layerBandTargets(ir, decoderNode.ir_node); // [{ label, target: "L1:<band>" }]
const lanes = rankBoundaryLanes(view.views[0], ir);
const issues = checkBudget(view.views[0], 0); // caller supplies current highlight count
```

`layoutIR(unknown, {onWarning?})` checks Draft 2020-12 schemas and reference/hierarchy
invariants, then returns `Promise<View>`. Errors reject the promise; CLI errors exit 1
without writing output. Warnings use stderr by default. Schema lookup resolves the
`@inferatlas/contracts` package through Node's `findPackageJSON`, follows its workspace
symlink, and locates the authoritative `schemas/` alongside `packages/`. For deployments
with a different structure, set `INFERATLAS_SCHEMA_DIR` to the directory containing
`ir.schema.json` and `view.schema.json`. This also works from an unrelated working directory.

Dense/MoE templates use the semantic ids in `docs/expr-dsl.md`, with attention/MLP
class recognition as a fallback. L0 folds descendants into trunk summaries. Consecutive
numeric layer ranges share one existing decoder anchor, labelled with a band strip
such as `[dense ×3][MoE ×58]`. Segments wrap two per line; `layerBandTargets(ir, ir_node)`
provides the matching strip labels and ordered `L1:<band>` targets without extending
View. Every band's L1/L2 remains available. Single bands retain their inclusive symbolic
range. If L0 still exceeds nine nodes, embedding/logits collectives fold into their
upstream neighbours, with their names in the neighbour's label and edges in provenance.

L1 exposes the representative decoder's norm/attention/MLP groups. L2 attention merges
Q/K head norms and rotary application into their semantic parent; L2 MLP/MoE flattens
implementation wrappers into projection/activation/router/expert operations and
collectives. Separate Q/K/V tensor shapes retain separate edges. No output node is
invented without an IR id.

`collapsed_of` is the sorted **entire represented scope**, including visible anchors
and suppressed descendants. The difference from visible `ir_node`s is the fidelity
ledger for details. Every edge has same-kind `ir_edges` provenance and a copied symbolic
shape. Internal collapsed data paths retain intervening collectives; KV loops are shown
only in their detailed view, and residual edges remain residual. An in-scope edge with
an unrepresented endpoint produces a warning containing its view and IR edge id.
Intentional internal collapse and same-route coalescing do not warn.

ELK 0.11.0 runs layered/ORTHOGONAL with seed 1, sorted inputs, DOWN flow, MULTI_EDGE
wrapping, inline shape labels, and BRANDES_KOEPF/BALANCED placement favouring straight
edges. Long shapes wrap one dimension per line. ELK's wrapping can produce diagonal
or tiny segments around label dummies; such a result deterministically retries ELK
with wrapping OFF. Fixture and synthetic dense/four-band views have aspect ratios
between 1:3 and 3:1, fit 1200×800 at scale >=0.6, and have no segment <=8px.

Node sizes use a frozen 14px sans advance table, 8px mono advances for symbolic shapes,
and conservative 14px advances for unknown glyphs. Multiline labels reserve space;
all exported geometry snaps to 4px. `rankBoundaryLanes(view, ir, direction?)` derives
local neutral segments perpendicular to the template's flow. Its optional explicit
`DOWN`/`RIGHT` direction supports custom templates. Segments do not claim physical rank
placement; the View schema has no lane or style field. No accent colours are emitted.

`checkBudget` reports >9 nodes, >12 edges or >2 caller accents; `assertBudget` rejects
excesses. Supported templates enforce that budget after merging. MLA/hybrid/VLM remain
warned generic L0/L1 fallbacks; an oversized generic view collapses to one labelled
summary with its complete scope in `collapsed_of`, preserving schema validity. Missing
internals also warn. The Qwen3-8B fixture currently has no decoder children: its L1
repeats the known summary, while synthetic IR exercises full dense L1/L2 templates.
See `docs/requests/T4.md` for the outstanding fixture request.

Tests cover deterministic fixture snapshots, separate CLI processes, reordered input,
geometry, provenance, budgets, four contiguous bands, all fallback families, invalid
inputs, folded communication, dropped-edge warnings, relocated schema resolution and
schema-directory overrides. Refresh snapshots with
`pnpm --filter @inferatlas/layout exec vitest run --update`. The `.snap` suffix prevents
Biome from changing the generated JSON's exact serializer formatting.
