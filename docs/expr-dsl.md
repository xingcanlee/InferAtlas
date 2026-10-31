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

`fp32`, `bf16`, `fp16`, `fp8`, `int8`, `int4` are reserved builtin identifier constants whose
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
are different quantities. Decode phase tensor dimensions use `T=S`; decode work expressions
already use `S`. `max_num_batched_tokens` and `max_num_seqs` control profiling peaks.

Metrics are local-rank values. One multiply-add is two FLOPs. A group summarizes children;
`decoder` summarizes resident layers, while its children describe one representative layer.
Use `memory.weights` for authoritative total bytes, never sum groups plus descendants. KV
uses `max(1,Hkv/tp)` heads per rank, or a separate MLA-specific formula. Quantization/layout
padding and tied weights require explicit rules. `memory.kv_tokens` means **capacity**, not
occupancy, and rounds down to whole cache blocks after configured deductions; exclusions
remain visible in `memory.unmodelled`.

Comm logical bytes count a local-rank logical tensor per prefill invocation (substitute `T=S`
for decode). `when=false` contributes zero. Ring all-reduce bus bytes are
`2*(n-1)/n * logical_bytes`; all-gather/reduce-scatter are `(n-1)/n * logical_bytes` with the
logical tensor convention stated by the rule. EP dispatch/combine assume uniform routing,
count outgoing payload once per rank, and exclude metadata/padding unless explicitly added.
HBM bytes moved exclude communication bus bytes. Hardware throughput is dense vendor peak,
not measured latency. Raw extraction shapes, catalog specs, evaluation results and view
coordinates are JSON numbers; IR metrics, dimensions, layer indices and config symbols are
expression strings. Source declaration line numbers remain integer provenance everywhere.
