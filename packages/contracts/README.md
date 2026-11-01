# Shared contracts (T0)

`generator/inferatlas/schema/models.py` is the only wire-schema source. Run `just schemas`
(or `pnpm schemas`) to export Draft 2020-12 JSON Schema and regenerate the committed TS
files. `pnpm --filter @inferatlas/contracts gen` regenerates TS only. Every JSON schema is
self-contained; `$defs` are local. Field descriptions carry normative units/meaning. Import
public types from `@inferatlas/contracts`; do not edit generated files. The strict TS 5 base
lives here in `tsconfig.base.json` because T0 does not own a root tsconfig path.

| Schema / TS root | Producer → consumer |
| --- | --- |
| snapshot / Snapshot | T1 resolve → every artifact |
| catalog / Catalog | catalog.yaml → T1 resolve, UI preset selection |
| extract / Extract | T1 → T3; ordinary factual numbers, sorted module paths |
| ir / IR | T3 → T4/F1/T6; symbolic quantities, stable node IDs |
| view / View | T4 → F1; precomputed 4px orthogonal layouts |
| triage / Triage | T3/T5/T6 → maintainers/publication gate |
| annotation / Annotation | one reviewed annotation |
| annotations / Annotations | YAML list of Annotation, T5 → T3 |
| rule / Rule | one lowering rule |
| rules / Rules | YAML list of Rule, T3 → T3 loader |
| report / Report | T6 verify → UI/publication gate |
| eval / Eval | fixture/golden numeric examples → T2/T3/T6 |
| index / Index | T6/T7 → F1 version × model × device matrix |

Always use `model_dump(mode="json", by_alias=True)` / `model_dump_json(by_alias=True)`:
`class`, `in` and `from` are wire keys; their Python attributes are `class_name`, `inputs`,
and `from_node`. Undeclared keys fail validation. Optional overlay/comm/source fields can be
omitted or null. Raw HF config/whitelisted facts allow JSON values, including null. Model
schemas emit the serialization representation. Pydantic also checks IR reference integrity,
parent cycles and view routes/bounds; JSON Schema cannot express these cross-field checks.
Consumers should validate with the model as well as the schema at Python boundaries.

`IR.in`/`out` slots are arrays of `{name, shape, dtype}`; shape is a string array or
`{prefill: string[], decode: string[]}` for explicit phase shapes; they are not a
single flattened shape. Rule outputs use the same representation, and Rule.flops and
bytes_moved and Comm.logical_bytes/bus_bytes always have `{prefill, decode}` (no scalar shorthand). `Rule.weight` becomes
`IRNode.weight_bytes`; `shard: {tp: 'in'}` becomes `parallel: {tp: {dim: 'in'}}`.
Qualified exact class matches take precedence over short exact matches; within that tier,
multiple applicable rules are an error. `match.vllm/ascend` are packaging SpecifierSet ranges,
not SHA comparisons; resolver provides package versions separately from immutable commits.
An empty range matches any version. Conditions use the DSL and declared context only.
Unknown classes or conditions block the affected model via triage.

Arrays with identities serialize in lexical id order, module arrays in path order, and
triage items by `(kind, symbol|id|model)`. Tensor slots, dimensions, routed points,
layer_types and ordered execution data retain semantic order. JSON object keys are sorted;
no timestamps or absolute host paths. Index cells sort by `(version, model, device)`.
Version pairs are explicit, never a Cartesian product of upstream releases. Index `data_path`
is relative to the site's public root and names the directory containing `ir.json`,
`view.json`, `report.json`. Missing cells and triage cells without a numeric bundle have null
paths. Paths in fixtures/index.json are
synthetic matrix examples, not directories guaranteed to exist in this fixture tree.

The structured Snapshot identity is reused in triage/report (the older string example in
design §5.2 is superseded by dev-plan §3.1). Inline catalog configurations are pinned by
`inline:sha256:<digest>` of UTF-8 `json.dumps(config, sort_keys=True, separators=(',', ':'))`;
no HF lookup is needed. Remote configurations should use an HF commit SHA. Snapshot Git
commits must be 40 lowercase hex digits, and source refs use repo-relative files with
script-resolved one-based declaration lines. No agent-authored line numbers in annotations.

Annotation anchors: locate the target AST class/def, select full physical lines starting at
the first decorator (or declaration if undecorated), through `end_lineno` inclusive; normalize
CRLF/CR to LF, ensure a final LF, preserve indentation/comments/other whitespace, UTF-8 encode,
and prefix its SHA-256 hex digest with `sha256:`. Exclude preceding comments/docstrings outside
the target. Any hash change invalidates the annotation, emits stale_annotation, and prevents
applying its note/decision. `target`/`resolves_to` remain symbols, resolved to SourceRef by
scripts. The fixture annotation is illustrative and is explicitly **not** maintainer approval.

The shared Ast/ExprEvaluator contract is handwritten only in `src/expr.ts`; T2 implements
it. See `docs/expr-dsl.md` for grammar, domains, errors, dtype bytes, env binding, and metric
accounting. Additional parallel axes belong in `IRNode.parallel`, not additional numeric
properties. Geometry numbers and source line numbers are not DSL expressions.

Fixtures are curated development inputs, never publication evidence. The MoE numeric oracle
matches the measured TP1/TP4 experiment; EP/dp>1, traffic and activation are estimates. The
dense fixture intentionally has only collapsed top-level nodes and zero placeholder work
counts. `fixtures/qwen3-30b-a3b/triage.json` has two independent sample errors to exercise UI;
it is not a claim that the `ok` matrix sample has passed a real pipeline. Production
publication requires empty triage and verification with no failed checks/diffs/broken links;
a skipped calibration is allowed, clearly marked as uncalibrated.

Rule.context binds GLOBAL module sizes, while extraction preserves per-rank facts.
Rule.emits creates `<module-node-id>.<suffix>` nodes; no duplicate compute-node comm.
See the generic row-linear o_proj/down_proj examples and path-to-template-id mapping in
`docs/expr-dsl.md`. View ids require `L1:<band>`; ViewEdge.ir_edges/kind/label preserve
IR provenance. Eval expected keys use JSON pointers with nodes indexed by stable id.
