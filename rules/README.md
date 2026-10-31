# T3 rules and symbolic lowering

`vllm.yaml` covers unquantized Qwen3 dense/MoE at the pinned vLLM development
commit and its paired 0.32 base version. Class lookup prefers qualified names,
then short names, then caller-supplied MRO bases in order. Overlapping version
ranges are ambiguous; file order never resolves them. Missing classes and
nonconstant applicability predicates block publication through triage.

YAML entries expand through `inferatlas.lower.load_rules()` before contract
validation. `inherits` names another rule id and replaces fields explicitly
specified by the child. Ascend entries carry `tags: [ascend]` and inherit the
baseline mathematics. Cycles, duplicate ids and missing bases fail loading.
Each expanded `RuleSpec.contract()` conforms to `schemas/rule.schema.json`.
Authoring-only `parameter_shapes` and `kv_per_token` formulas are validated
metadata; they are excluded from the public Rule wire contract. Weight tensor
inputs in IR expose parameter shapes without extending IR. `LowerResult`
also exposes `parameter_shapes[module_path][parameter_name]` for cross-checks.

Context sizes are GLOBAL. Recover sizes using observed extraction TP, then
retain runtime `tp`/`ep` in emitted expressions. Constructor globals and config
are cross-checked with local partitions/shapes. Vocabulary pads before TP;
KV heads replicate above the KV head count. Replicated head norms and gates
stay replicated. Without EP, expert intermediate TP is `tp*dp`; with EP,
whole experts shard over `ep=tp*dp`. Uniform MoE compute uses tokens from DP
replicas. Ring bus factors are `2*(n-1)/n` for all-reduce and `(n-1)/n` for
all-gather/reduce-scatter under the full-result logical tensor convention.
Default logits gather instead uses a local-shard logical tensor and reports
rank-average outgoing bytes; `use_all_gather` changes its op and factor.

IR contains representative layers, layer bands, group summaries, parameter
slots, KV/residual/data edges and explicit communication emissions. Expert
GEMMs use `moe.w13`, `moe.act`, `moe.w2`. Groups summarize descendants; use
`memory.weights` for authoritative totals. PP controls cover equal-size
partitions only; multiple bands restrict PP to one. Production extracts must
contain every configured layer; all layers in a band must share parameter and
module semantics. Shared, parameterless RoPE instances may appear only once
in `named_modules`. Partial curated fixtures repeat their representative layer.

Memory capacity subtracts weights, a profiling-token activation estimate,
configured non-torch bytes and reserved bytes, then floors to whole blocks.
Sampler warmup includes `max_num_seqs*V*bytes(fp32)`. The HBM control defaults
to the fixture's decimal 64 GB; callers must bind the selected catalog hardware
preset. Graph capture, fragmentation, backend scratch, non-parameter buffers
and unconfigured communication workspace remain explicit exclusions. Runtime
controls require joint validation: TP/head/vocab divisibility, `ep=1` or
`ep=tp*dp` dividing E, expert-intermediate divisibility, `T%batch=0`, equal PP.

`lower(extract, rules=..., vllm_version=..., ascend_version=..., mro=...)`
returns `LowerResult(ir, triage, parameter_shapes)`. `coverage_gate(triage)` is
true only for empty blocking triage. `result.write(directory)` serializes
sorted JSON without timestamps. Context/parameter/path failures use blocking
`golden_diff` items because the current triage schema has no binding-error kind.
Known-stale installed package metadata never overrides the pinned source
commit. No runtime Python DSL evaluator is shipped. Test helpers use Lark as
a reference while T2 is developed; tests consume real T1 cache extracts when
available. The annotations stage supplies reviewed Ascend significance; these
rules do not invent review identities.

Run from repository root:

```sh
uv run --project generator python -m inferatlas.lower \
  --extract fixtures/qwen3-30b-a3b/extract.json --out /tmp/t3-moe
uv run --project generator python -m inferatlas.lower \
  --model qwen3-8b --snapshot fixtures/qwen3-30b-a3b/snapshot.json \
  --out /tmp/t3-dense
uv run --project generator pytest generator/tests/test_lower*.py generator/tests/test_rules*.py
```

Config projections use pinned local catalog configs and AST-resolved class
locations from `INFERATLAS_SRC` or the provided read-only upstream source root.
They are marked curated fixtures, not CPU/meta facts. No weights or NPU are
used. Set `INFERATLAS_LOWER_EXTRACTS` to an additional extract-cache directory
to run every-parameter shape and total-byte checks on CPU/meta output.
