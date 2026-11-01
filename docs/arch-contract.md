# Direction v2 — structure first (2026-11-01)

The product goal is **model structure and tensor I/O**: whole-model dataflow and, for every
layer, what goes in and what comes out (symbolic + numeric shapes). Memory ledgers, rooflines,
offline single-file exports and release automation are secondary and must never hide structure.

## Rules
1. **Triage never blocks structure.** Any cell whose extraction produced a module tree is shown.
   Unknown classes, unannotated Ascend overrides, unknown hardware capacity, golden diffs are
   displayed as badges/notes on the affected component, not as a reason to hide the cell.
   Numeric memory/roofline panels may be marked "unverified" independently.
2. **No rule ⇒ generic component, never nothing.** A class without a lowering rule is rendered
   as a `generic` component with its class name, source link, children, parameter shapes and
   whatever I/O shapes can be derived.
3. **Paper-style figures.** The architecture view looks like the figures in LLM papers
   (Transformer / Llama / DeepSeek-V3 / Mixtral): a vertical stack Embedding → [Block × N] →
   Norm → LM Head, a block drawn as Norm → Mixer → ⊕ → Norm → FFN/MoE → ⊕ with residual lines,
   MoE drawn as router + expert grid (+ shared expert), heterogeneous stacks drawn as bands
   (e.g. "layers 0–2 dense MLP", "layers 3–60 MoE"; "3 × linear-attn + 1 × full-attn" patterns).
4. **Shared components.** Visual primitives live in one shadcn-style package
   (`packages/arch-ui`): Tailwind tokens, `cva` variants, composable, copy-in friendly. Pages
   compose them; no per-model drawing code.

## `arch.json` (one per model × device × version snapshot)

Emitted by the generator next to `ir.json`/`view.json`/`report.json` as `arch.json`.
Shapes are lists of expression strings in the existing expr DSL (`docs/expr-dsl.md`) over the
symbols below; the UI evaluates them numerically with the current parameter values.

```jsonc
{
  "schema_version": 1,
  "model": { "id": "qwen3-30b-a3b", "hf_id": "Qwen/Qwen3-30B-A3B",
             "architecture": "Qwen3MoeForCausalLM", "family": "moe" },
  "device": "A2",
  "symbols": { "T": {"value": 4096, "description": "tokens in the batch"},
               "H": {"value": 2048, "description": "hidden size"}, "...": {} },
  "input":  { "name": "input_ids", "shape": ["T"], "dtype": "int64" },
  "output": { "name": "logits", "shape": ["T", "V"], "dtype": "bfloat16" },
  "stages": [                         // ordered top-level dataflow
    { "id": "embed", "kind": "embedding", "label": "Token embedding", "module": Module, "io": IO },
    { "id": "layers", "kind": "stack", "label": "Decoder layers", "bands": [Band, ...] },
    { "id": "final_norm", "kind": "norm", "module": Module, "io": IO, "component": Component },
    { "id": "lm_head", "kind": "lm_head", "module": Module, "io": IO },
    { "id": "...", "kind": "other", "module": Module, "io": IO }      // e.g. MTP head
  ],
  "notes": [ { "severity": "info|warn", "target": "<component id>|null", "text": "...",
               "triage_kind": "unknown_layer_class|conditional_override|...|null" } ]
}
```

- **Band**: `{ "id", "layers": [start, end] (inclusive), "count", "pattern": null | "label",
  "block": Block }`. Consecutive layers with identical block structure form one band; a periodic
  pattern (e.g. Qwen3-Next 3 linear + 1 full) may be expressed as several bands or a `pattern`.
- **Block**: `{ "kind": "decoder", "residual": "pre_norm"|"post_norm"|"sandwich"|"unknown",
  "sublayers": [Sublayer, ...] }` in dataflow order.
- **Sublayer**: `{ "id", "role": "norm"|"mixer"|"ffn"|"residual_add"|"other",
  "component": Component, "io": IO, "module": Module|null }`.
- **Component** (discriminated by `kind`):
  - `norm`: `{variant: "rmsnorm"|"layernorm"|"other", eps?}`
  - `attention`: `{variant: "mha"|"gqa"|"mqa"|"mla"|"linear"|"sliding_window"|"sparse"|"other",
    heads_q, heads_kv, head_dim, rope: {kind, theta?, scaling?}|null, qk_norm: bool,
    extras: {…}}` (MLA: `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `v_head_dim`, …)
  - `mlp`: `{variant: "swiglu"|"geglu"|"gelu"|"other", intermediate}`
  - `moe`: `{experts, top_k, shared_experts, router: {scoring, norm_topk?, groups?},
    expert: {variant, intermediate}, shared: {variant, intermediate}|null}`
  - `generic`: `{class_name, children: [{name, class_name}]}` — fallback for unknown classes
  - `embedding` / `lm_head`: `{vocab, hidden, tied: bool}`
- **IO**: `{ "inputs": [Tensor], "outputs": [Tensor] }`, Tensor =
  `{ "name", "shape": [expr], "dtype" }`.
- **Module**: `{ "path": "model.layers.0.self_attn", "class_name", "source": {file, line, url},
  "status": "ruled"|"generic"|"ascend_override"|"unannotated_override",
  "replaced": {"baseline_class", "ascend_class"}|null,
  "params": [{ "name", "shape": [expr], "dtype" }] }`.

All lists are deterministically ordered; no timestamps.

## Ownership for this round
- **S1 (Codex)** — pydantic model → JSON Schema → TS contract for `arch.json`; emitter (bands,
  per-sublayer IO, generic fallback, status/notes); triage becomes non-blocking for structure;
  site shows every extracted cell; goldens for Qwen3-8B / Qwen3-30B-A3B.
- **S2 (Codex)** — more models: catalog entries (config.json only, no weights) for current major
  families that vLLM 544f182 registers (DeepSeek-V3.2/V4, Kimi-K2, MiniMax-M2/M3, GLM-4.x MoE,
  gpt-oss, Qwen3-Next, Llama-4, …); make CPU extraction succeed for them on baseline and Ascend
  (e.g. redirect NPU-only constructor allocations to meta).
- **U1 (Claude)** — `packages/arch-ui` component library + the architecture page (whole-model
  dataflow and per-layer I/O), built against fixtures that follow this contract.

## Contract decisions v1.1 (reconciling S1 emitter and U1 UI)
1. `residual_add` sublayers: `component: null`, `module: null`; emitted for every residual style
   (pre_norm, post_norm, sandwich), in dataflow order.
2. Sublayer ids are **band-relative and layer-independent** (e.g. `input_layernorm`, `self_attn`,
   `residual_1`, `post_attention_layernorm`, `mlp`, `residual_2`); the layer-specific path lives in
   `module.path` (path of the band's first layer; UI substitutes the selected layer index).
3. Note `target`: `"<band_id>/<sublayer_id>"` for sublayers (applies to the whole band),
   `"<stage_id>"` for stages, `null` for model-level. `text` is human-readable prose, never raw
   JSON; `triage_kind` carries the machine kind.
4. Periodic stacks: consecutive bands that form a repeating period share the same non-null
   `pattern` label (e.g. `"3× linear + 1× full"`). The UI additionally folds by structural
   signature (role + component kind/variant + class), ignoring ids/paths/sources.
5. Stages: containers (`model`, root) and `LogitsProcessor` are not stages. `other` stages carry
   `placement: "spine" | "aside"` (MTP / draft heads = aside).
6. `moe.router.groups`: `{ "n_group": int, "topk_group": int } | null`.
7. Attention `extras` keys (all optional): MLA `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dim`,
   `qk_rope_head_dim`, `v_head_dim`; sparse `index_topk`; sliding `sliding_window`; linear
   attention `kind` (e.g. `"gated_deltanet"`), `num_k_heads`, `num_v_heads`, `head_k_dim`,
   `head_v_dim`, `conv_kernel`, `output_gate`. For linear attention, `heads_q/heads_kv/head_dim`
   describe the linear mixer itself, not the full-attention config.
8. `rope`: `{ kind, theta?, scaling?, partial_rotary_factor? }` (partial factor read from the
   top-level config when present).
9. Symbols include runtime `T` and `tp` (and `dp`/`ep` when relevant); IO and param shapes that are
   TP-sharded are written with `/tp` (e.g. `Hq*D/tp`), so the UI can recompute them.
10. `class_name` is fully qualified; UI shows the last segment in figures and the full name in detail.
11. TS type names are those exported by `@inferatlas/contracts` (S1); the UI adapts to them.

## Contract decisions v1.2
1. IO gains `internals: [Tensor]` (default `[]`): tensors produced inside a component that are
   neither its inputs nor its outputs (q/k/v/o, MLA latents, router_logits, topk ids/weights,
   per-expert gate/up/down). `outputs` holds only what leaves the sublayer (usually `[T, H]`).
2. Notes are de-duplicated: one note per (class, triage_kind) at model or stack level, plus targeted
   notes only where the sublayer itself is generic/unannotated. vLLM building blocks inside a ruled
   component (parallel linears, Attention, rotary, activation, MoERunner, ModuleList,
   LogitsProcessor) produce no warning. Severity `warn` only for things that change the figure.
3. Attention `extras` may include `compress_ratio` and `indexer: bool` (DeepSeek-V4 style
   compressed/sparse attention); these participate in band signatures and pattern labels.
4. Block `residual` may be `"hyper_connection"` (DeepSeek-V4 mHC); `mhc_*` sublayers then have role
   `"other"` and are ordered in dataflow position.
5. Stage order is canonical: embedding(s) → stack(s) (each with its own pre/post norms) → final norm
   → lm_head → spine others; aside others last.

## Contract decisions v1.3
1. **Pattern labels are per period**: every band of one repeating period carries the same label,
   e.g. `"1× MQA ÷128 + 1× MQA ÷4 + indexer"`; non-repeating bands use `pattern: null`. The UI
   additionally folds structurally and treats single-member labelled runs as plain bands.
2. **Stage order is the array order.** The emitter writes stages in true dataflow order (e.g.
   DeepSeek-V4 `hc_head_op` — merging the residual streams — sits before the final norm as a spine
   `other` stage); the UI renders in array order and does not re-sort.
3. **mHC sublayers**: `role: "other"`, `component: generic` (keeps class/source), ids
   `mhc_pre_attn`, `mhc_post_attn`, `mhc_pre_ffn`, `mhc_post_ffn`; residual stream tensor name
   `residual_streams` with shape `[T, HC, H]`; the UI shows their notes/status like any sublayer.
4. **Hash routing**: layers routed by token-id hash use `router.scoring: "hash"`.
5. **Attention `variant` = head-sharing scheme** (`mha` | `gqa` | `mqa` | `mla`) or `linear` for
   non-softmax mixers. Locality/sparsity are extras: `sliding_window`, `compress_ratio`, `indexer`,
   `index_topk`, plus `o_lora_rank`, `o_groups`, `chunk_size`. Variants `sliding_window` and
   `sparse` are deprecated (still accepted for old data).
6. **Canonical internal tensor names**: attention `q`, `k`, `v`, `k_rope`, `q_latent`,
   `kv_latent`, `kv_compressed`, `index_scores`, `o`; MoE `router_logits`, `topk_ids`,
   `topk_weights`, `expert_input`, `expert_gate_up`, `expert_output`, `shared_output`; MLP
   `gate_up`, `act`; linear attention `qkvz`, `ba`, `conv_out`, `recurrent_state`.
