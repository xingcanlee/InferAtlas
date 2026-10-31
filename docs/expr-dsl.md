# InferAtlas expression DSL, contract v1

Pydantic/JSON Schema describes expression **strings**, not evaluated numbers. T2 supplies a
Peggy parser/evaluator in `packages/expr` and a Lark syntax/symbol checker in
`generator/inferatlas/expr`. Python must not implement the runtime evaluator. Both consume
`packages/expr/test-vectors.json` (`[{expr, env, value | error}]`). The AST and evaluator API
are exported by `@inferatlas/contracts` from `src/expr.ts`. There is no JavaScript `eval`,
property access, indexing, assignment, arbitrary function call, or implicit host access.

## Grammar (EBNF)

Whitespace (`space`, `tab`, CR, LF) is ignored between tokens. Identifiers and names are
case-sensitive. Match the entire input; trailing tokens, empty input and comments fail.

```ebnf
expression  = conditional ;
conditional = logical_or, [ "?", expression, ":", conditional ] ;
logical_or  = logical_and, { "||", logical_and } ;
logical_and = equality, { "&&", equality } ;
equality    = comparison, { ("==" | "!="), comparison } ;
comparison  = sum, { ("<=" | ">=" | "<" | ">"), sum } ;
sum         = product, { ("+" | "-"), product } ;
product     = unary, { ("*" | "/" | "%"), unary } ;
unary       = ("+" | "-" | "!"), unary | power ;
power       = primary, [ "**", unary ] ;
primary     = number | "true" | "false" | call | identifier | "(", expression, ")" ;
call        = function, "(", [ expression, { ",", expression } ], ")" ;
function    = "max" | "min" | "ceil" | "floor" | "cdiv" | "round_up" | "bytes" ;
identifier  = (letter | "_"), { letter | digit | "_" } ;
number      = (digits, [".", [digits]] | ".", digits), [exponent] ;
exponent    = ("e" | "E"), ["+" | "-"], digits ;
digits      = digit, {digit} ;
digit       = "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" | "9" ;
letter      = "A" … "Z" | "a" … "z" ;
```

Use longest-token matching (`**` before `*`, `<=` before `<`, `!=` before `!`, and whole
identifier boundaries for keywords). The exponent sign belongs to a number, while a leading
sign is always unary. Decimal leading zeroes are accepted as decimal, never octal. Hex,
strings in quotes, array/object literals, `NaN` and `Infinity` are not grammar productions.
Only the listed function names may be called; identifiers cannot be callable.

## Precedence and values

From lowest to highest: `?:`, `||`, `&&`, `== !=`, `< <= > >=`, `+ -`, `* / %`, unary
`+ - !`, `**`, primary. Binary operators associate left except `**`, which associates right;
ternary associates right. `-2**2` is `-(2**2)` = -4; `2**-2` = 0.25; `2**3**2` = 512.
The grammar is normative if a precedence description seems ambiguous.

Numbers use IEEE-754 double. Integers are exact only through 2^53; consumers must keep integer
counts within that bound when exactness matters. Numeric operations require numbers, without
string or boolean coercion. Equality is strict by type (`1 == true` is false); dtype strings
can be compared for equality. Ordering comparisons require numbers. Chained comparisons
associate left and therefore generally fail type validation (`1 < 2 < 3` compares a boolean
to a number); write `1 < x && x < 3`.

`true`/`false` are booleans. Conditions accept booleans or finite numbers (zero is false,
nonzero true); strings in condition position are errors. `!`, comparisons, `&&`, and `||`
produce booleans. `&&`, `||`, and `?:` evaluate lazily, so an unselected branch cannot cause
a division-by-zero or missing-symbol error. They return booleans for logical operators and
the selected value for ternary. `evaluate` converts a final boolean to 1/0; a final string is
an error. `bytes` is the only numeric conversion of a dtype string.

Non-finite literals/results, division or remainder by zero, invalid function arity/type/domain,
unknown function, and a missing identifier on the evaluated path are errors. The sign of `%`
follows its left operand (JavaScript remainder). No silent fallback, clamping, or unknown
symbol = zero. Error **categories** are `syntax`, `unknown_symbol`, `type`, `arity`, `domain`,
`non_finite`; T2 should expose these as a stable error `code`. Error prose is not a contract.
Validate arity when parsing so malformed calls in unselected branches still fail.

## Builtins

| Function | Arity and meaning |
| --- | --- |
| `max(a, …)`, `min(a, …)` | At least one finite number; largest/smallest value. |
| `ceil(x)`, `floor(x)` | One finite number; standard rounding toward +∞ / −∞. |
| `cdiv(x, y)` | Two numbers, `x >= 0`, `y > 0`; `ceil(x / y)`. |
| `round_up(x, m)` | Two numbers, `x >= 0`, `m > 0`; `ceil(x / m) * m`. |
| `bytes(dtype)` | One dtype string; bytes per scalar from the table below. |

`fp32`, `bf16`, `fp16`, `fp8`, `int8`, `int4`, `int32`, `int64` are reserved builtin identifier constants whose
values are their own strings. They are usable directly (`bytes(bf16)`) and in equality
(`dtype_w == int4`). Environment dtype values use the same spellings, without `torch.` prefixes.
These names, function names, and boolean literals cannot be rebound in the environment.

| Dtype | Bytes |
| --- | ---: |
| fp32 | 4 |
| bf16 | 2 |
| fp16 | 2 |
| fp8 | 1 |
| int8 | 1 |
| int4 | 0.5 |
| int32 | 4 |
| int64 | 8 |

Fractional scalar bytes do not imply half-byte addressability. Storage expressions must round
up packed byte totals and add quantization scale/zero overhead explicitly. Torch extraction
facts retain names such as `bfloat16`; lowering maps them to DSL dtype names.

## API behavior

- `parse(src): Ast` is pure; ASTs use exactly the discriminated union in contracts. Dtypes
  parse as `identifier`, booleans as `boolean`, numbers as `number`; no source-location fields.
- `evaluate(src | Ast, env: Record<string, number|string|boolean>): number` is synchronous,
  pure, finite on success, and does not mutate the AST/environment. AST input gets the same
  type/domain/arity checks as parsed input. Invalid reserved bindings fail with `type`.
- `free(src): string[]` returns **all** non-builtin identifiers across all branches, sorted,
  unique, without an environment. Function names, boolean literals and dtype constants are
  excluded; `bytes(dtype_w)` returns `["dtype_w"]`.
- `format(src, 'symbolic', env?)` prints a deterministic parseable expression preserving
  meaning; no algebraic rewrites required, unused whitespace removed, operators consistently
  spaced. Its exact parenthesis style belongs to T2 shared vectors. Environment is ignored.
- `format(src, 'numeric', env?)` calls evaluate with `env ?? {}` and returns the shortest
  JavaScript round-trip decimal (`Number.toString()`), with negative zero normalized to `0`.
  No locale, units, commas or precision truncation. Errors propagate.

## IR binding and accounting

`IR.symbols` maps config identifiers to expression literals (e.g. `H: "2048"`); resolve them
before runtime params and reject dependency cycles. `IR.params` maps names to type/domain
metadata: numeric defaults/bounds are expression strings; boolean defaults are `"true"` or
`"false"`; dtype/enum defaults are raw strings. Build the env from symbols, parsed defaults,
then validated user overrides. Symbols are immutable and cannot be overridden by controls.
Bounds may refer to config symbols; choices are raw literals (numeric choices are decimal
strings). T2 owns the evaluator; consumers own environment binding/domain validation.

The standard config symbols are `H` hidden size, `L` decoder count, `E` routed experts,
`topk` active experts per token, `Hq` query heads, `Hkv` KV heads, `D` head dimension,
`V` vocabulary, `I_moe` expert intermediate size (dense models use `I`). They are **not**
reserved DSL constants. Every formula dependency must be declared by the IR or rule context.

`tp`, `pp`, `dp`, `ep` are positive integer **degrees** in IR. `ep=1` disables expert
parallelism; for this vLLM revision an enabled flattened EP group has `ep=tp*dp` (PCP=1).
`Extract.parallel.ep` is the observed upstream **boolean** enable flag. Without EP, vLLM
flattens MoE intermediate TP over `tp*dp`; with EP, experts shard over `ep` and the expert
intermediate dimension is whole. T1/T3 must not interpret the boolean as a numeric degree.
Data-parallel replicas are not a multiplier of rank-local resident weight memory.

`T` is prefill chunk tokens per attention DP replica, `S` is decode sequences, `batch` is
equal-length prefill sequences, and `context` is existing tokens per decode sequence. They
are different quantities. Tensor `shape` may be an array (decode substitutes `T=S`) or an explicit
`{prefill: [...], decode: [...]}` object. Select the latter before evaluating dimensions;
never substitute `T=S` in an explicit phase shape. LM-head input/output uses explicit
phase shapes: `[batch,H]` / `[S,H]` and `[batch,round_up(V,64)/tp]` /
`[S,round_up(V,64)/tp]`. Prefill selects the last token of each sequence. Decode work
expressions already use `S`. `max_num_batched_tokens` and `max_num_seqs` control profiling peaks.

Metrics are local-rank values. One multiply-add is two FLOPs. A group summarizes children;
`decoder` summarizes resident layers, while its children describe one representative layer.
Use `memory.weights` for authoritative total bytes, never sum groups plus descendants. KV
uses `max(1,Hkv/tp)` heads per rank, or a separate MLA-specific formula. Quantization/layout
padding and tied weights require explicit rules. `memory.kv_tokens` means **capacity**, not
occupancy, and rounds down to whole cache blocks after configured deductions; exclusions
remain visible in `memory.unmodelled`.

Comm `logical_bytes` and `bus_bytes` are both explicit `PhaseExpr` objects
`{prefill, decode}`; evaluate the chosen phase directly, without T substitution. Logical
bytes count the tensor whether or not the collective is active; bus bytes are zero when inactive. `when=false` contributes zero traffic. Ring all-reduce bus bytes are
`2*(n-1)/n * logical_bytes`; all-gather/reduce-scatter are `(n-1)/n * logical_bytes` with the
logical tensor convention stated by the rule. EP dispatch/combine assume uniform routing,
count outgoing payload once per rank, and exclude metadata/padding unless explicitly added.
HBM bytes moved exclude communication bus bytes. Hardware throughput is dense vendor peak,
not measured latency. Raw extraction shapes, catalog specs, evaluation results and view
coordinates are JSON numbers; IR metrics, dimensions, layer indices and config symbols are
expression strings. Source declaration line numbers remain integer provenance everywhere.


## Rule context and stable emissions

Rules declare required module identifiers in `context`; these are a separate namespace from
IR config symbols and runtime parameters. Lower binds them before checking `match.when`,
then substitutes those bindings into emitted IR expressions. Context sizes are **GLOBAL,
unsharded** sizes. The rule expressions divide by `tp` themselves. Never copy a local
extraction shape into `in_features` and then divide it again. A missing/ambiguous binding or
a disagreement between config and observed attrs becomes triage, never a guessed size.

Extraction preserves observed facts: partition sizes, attention head counts and parameter
shapes are per-rank. Some upstream constructor attrs (`input_size`, `output_size`) already
contain global sizes; they remain unchanged facts. Normalize according to the matched
class, not an attribute-name heuristic:

| Context identifier | Binding / recovery by lower |
| --- | --- |
| `in_features`, `out_features` | RowParallelLinear: `input_size_per_partition * extract.parallel.tp`, `output_size_per_partition`; ColumnParallelLinear: local input, local output × observed TP. Prefer constructor global sizes/config when available and cross-check local weights. ReplicatedLinear: both local sizes are global. |
| `reduce_results`, `gather_output` | Actual module boolean attrs; defaults only from the pinned class constructor. |
| `num_heads` | Config query-head count, or observed local heads × observed TP. |
| `num_kv_heads` | Config KV-head count; cross-check `max(1, global/observed TP)`. Replication loses information, so local heads × TP is not valid when TP exceeds KV heads. |
| `head_size` | Observed `head_size`/`head_dim` or config head_dim; not sharded. |
| `num_experts`, `top_k` | Config global expert count and active experts per token; cross-check local expert count × EP, and module top_k. |
| `intermediate_size` | Config dense/expert intermediate size; or local partition × observed expert TP (`tp*dp` without EP, 1 with EP for this revision). |

Use **extraction** TP to recover a constant; retain **runtime** `tp` in rule formulas for
interactive recomputation. Every free identifier must be builtin, IR symbol/param, or a
listed context binding. Context names cannot shadow IR symbols/params or builtins.

One generic RowParallelLinear rule is demonstrated in `fixtures/rule.json` and
`fixtures/rules.json`. It uses `in_features/tp*out_features*bytes(dtype_w)` for weights,
`2*T*in_features/tp*out_features` / `2*S*in_features/tp*out_features` for FLOPs and
`[T,out_features]` for output. Its compute node has no `comm`. It declares:

```json
{"emits": [{"suffix": "reduce", "kind": "comm", "comm": {
  "op": "all_reduce", "group": "tp", "when": "reduce_results && tp>1",
  "logical_bytes": {"prefill": "T*out_features*bytes(dtype_a)", "decode": "S*out_features*bytes(dtype_a)"},
  "bus_bytes": {"prefill": "reduce_results && tp>1 ? 2*(tp-1)/tp*T*out_features*bytes(dtype_a) : 0", "decode": "reduce_results && tp>1 ? 2*(tp-1)/tp*S*out_features*bytes(dtype_a) : 0"}
}}]}
```

Worked Qwen3-8B examples at TP4, bf16 (T tokens prefill, S decode sequences):

| Module | Global context | Observed local weight | Weight bytes | Prefill FLOPs |
| --- | --- | --- | ---: | --- |
| `self_attn.o_proj` | in_features=4096 (=Hq*D), out_features=4096 (=H), reduce_results=true | [4096,1024] | 8388608 | 8388608*T |
| `mlp.down_proj` | in_features=12288 (=I), out_features=4096 (=H), reduce_results=true | [4096,3072] | 25165824 | 25165824*T |

Both emit logical bytes `T*4096*2` / `S*4096*2`, bus bytes 1.5 times that at TP4.
For Qwen3-30B-A3B o_proj the same rule binds in_features=4096, out_features=2048;
the supplied TP4 extract observes [2048,1024], not [2048,256].

A rule MAY emit additional communication nodes. Each emission has a unique simple `suffix`;
IR id is `<module-node-id>.<suffix>`. Metadata and matched rule id are inherited from the
module. Place its collective only on the emitted node, with explicit graph edges to express
execution order. A rule with `emits` cannot also attach `comm` to its compute node. Inactive
comm nodes remain in IR with `when=false` and zero bus bytes; views may hide them.

Module-node ids are stable semantic **template** ids, independent of layer index and layout.
Lower strips `model.layers.<n>.` for a representative band, then uses this mapping:

| Module path | Template id |
| --- | --- |
| `model.embed_tokens` | `embed` (emitted reduction `embed.reduce`) |
| `self_attn` / `self_attn.qkv_proj` | `attn` / `attn.qkv` |
| `self_attn.o_proj` | `attn.o` (emitted reduction `attn.o.reduce`) |
| `self_attn.q_norm`, `k_norm`, `rotary_emb`, `attn` | `attn.q_norm`, `attn.k_norm`, `attn.rope`, `attn.core` |
| `input_layernorm` / `post_attention_layernorm` | `norm.input` / `norm.post` |
| `mlp` (dense) / `mlp.gate_up_proj` / `mlp.down_proj` | `mlp` / `mlp.gate_up` / `mlp.down` (reduction `mlp.down.reduce`) |
| `mlp` (MoE) / `mlp.gate` / `mlp.experts` | `moe` / `moe.gate` / `moe.experts` |
| `mlp.experts.routed_experts` / `.quant_method` | `moe.experts.weights` / `moe.experts.method` |
| `model.norm` / `lm_head` / `logits_processor` | `final_norm` / `lm_head` / `logits` (collection `logits.gather`, sampling `output`) |

MoE runner emissions are `moe.experts.dispatch`, `.combine`, `.reduce`; internal expert
GEMM/activation template ids are `moe.w13`, `moe.act`, `moe.w2`. These are curated semantic
operations on the routed-expert module. Synthetic summaries such as `decoder` and
`attn.qk_norm` have no independent module identity. Multiple bands that need different
semantics prefix their node ids with `<band-id>.`; the single-band fixtures use unprefixed
ids. Unknown module paths require an explicit family mapping or triage.

For pinned vLLM 544f182, this fixture excludes PCP/SP and uses the default unfused output
reduction path. The default `all2all_backend` is `allgather_reducescatter`
(`config/parallel.py`); naive dispatch/combine activates whenever `dp>1`
(`fused_moe/runner/moe_runner.py`), regardless of whether EP is enabled.
The approved communication contract represents dispatch as **all_gather** and combine as
**reduce_scatter** over the flattened **tp*dp** EP group, including `ep=1`.
Each has logical bytes `T*H*bytes(dtype_a)` / `S*H*bytes(dtype_a)` and bus bytes
`(tp*dp-1)/(tp*dp)` times that tensor, gated by `dp>1`.
The group size for these default collectives is `tp*dp`, not the `ep` enable degree.
Expert FLOPs and workspace use the `T*dp` / `S*dp` gathered tokens, with local expert
ownership divided by `ep` when EP is enabled, or intermediate TP `tp*dp` otherwise.

`lower(..., moe_a2a=False)` exposes backend selection as a boolean **lowering parameter**,
also available as `python -m inferatlas.lower --moe-a2a`. The default is `false`.
Selecting `true` makes the same dispatch/combine nodes use **all_to_all**, logical bytes
`T*topk*H*bytes(dtype_a)` / `S*topk*H*bytes(dtype_a)` and uniform-routing bus factor
`(ep-1)/ep`, gated by `ep>1 && dp>1`. The backend requires enabled EP and DP for traffic.
`Comm.op` is a static enum in the IR schema, so switching backend requires lowering again;
TP/DP/EP degrees remain interactive expressions in either output. Reusing the existing
node IDs keeps layout budgets and drill-down links stable.
These logical tensors omit router scores, expert IDs, padding, and transport scratch.

Final MoE output all-reduce activates for `tp>1 || ep>1` over **TP**, of size `tp`,
with logical bytes `T*H*bytes(dtype_a)` / `S*H*bytes(dtype_a)`
(`fused_moe/runner/moe_runner.py`). At tp=1 the call has no bus traffic.
Fused backends that already reduce output must lower the equivalent semantics without a
duplicate collective. Communication remains an uncalibrated uniform-routing estimate.

VocabParallelEmbedding emits `embed.reduce` for tp>1, logical bytes `T*H*bytes(dtype_a)` /
`S*H*bytes(dtype_a)`. LogitsProcessor emits `logits.gather`: default op `gather` to TP rank 0,
logical bytes **local shard** `rows*round_up(V,64)/tp*bytes(dtype_a)`, with rows=batch/S.
Gather bus bytes are the rank-average outgoing payload `(tp-1)/tp * local shard`;
rank 0 sends zero and receives `(tp-1)*local shard`. A platform selecting `use_all_gather`
must emit `all_gather` with `(tp-1)*local shard` outgoing bytes per rank. This differs from
the usual full-result all-gather logical-size convention. The default head output dtype is
model dtype, represented by dtype_a here; sampler fp32 conversion/workspace is separate.
Explicit fp32 head overrides need corresponding logits slots/traffic dtype changes.

View ids are `L0`, `L1:<band-id>` (e.g. `L1:decoder`), and `L2:<node-id>`; bare `L1` fails.
View edges carry sorted `ir_edges`, semantic `kind` (data/kv/residual), and optional `label`
that copies a represented IR edge's **shape expression** (array or phase-shape object).
Collapsed routes may reference several IR edges. The producer checks those references and
preserves edge kind; residual/KV edges cannot be merged into data routes.

Eval expected keys are JSON pointers, e.g. `/nodes/attn.o.reduce/comm/bus_bytes/prefill`.
First index evaluated `nodes` by stable id, not array position; all other arrays use numeric
indices, e.g. `/nodes/attn.qkv/out/0/shape`. Dots in ids are literal. Escape `~` and `/` as
`~0` and `~1`; reject dotted-path keys. Bind defaults, then case overrides before evaluation.
