# Ascend annotations

`a2/qwen3-moe.yaml` contains Chinese draft notes anchored to Ascend commit
`24a76e84394ae3427195e3715fc08e90769d398c`, paired with vLLM commit
`544f182d9f0c9bfe8bd7645525951d5e3692a542`. All entries have
`reviewed_by: pending`; no human approval or NPU measurements are implied.

There are nine distinct replacement implementations in the experiment's twelve
layer-module swaps: RMSNorm is shared by input/post-attention and Q/K norms.
The endpoint embedding, head and logits implementations are also covered.
Separate predicates distinguish ordinary kernels, NZ storage, and custom parallel
modes. The patches cover the MoE factory, Qwen MoE attention forward, routing
capture, and redundant expert mapping. API/scheduler patches and patches for other
architectures, such as MLA and Mamba, are outside this model's overlay.

## Python API

Run Python with `uv run --project generator python`. Import the public API from
`inferatlas.annotations`:

```python
from pathlib import Path
from inferatlas.annotations import SourceTree, load_annotations, check_annotations

sources = {
    "vllm_ascend": SourceTree(
        Path(source_root) / "vllm-ascend", "vllm-ascend", snapshot.ascend.commit
    ),
    "vllm": SourceTree(Path(source_root) / "vllm", "vllm", snapshot.vllm.commit),
}
annotations = load_annotations("annotations/a2")
resolved, stale_items = check_annotations(annotations, sources)
```

The loader uses `AnchoredAnnotation`, a local Pydantic subclass of the shared
Annotation model with optional `anchors: [{target, anchor_hash}]` for additional
implementation dependencies. The shared contract cannot yet express these
anchors; its adoption is requested in `docs/requests/T5.md`. Until then, load these
files through `load_annotations`, not the shared Annotations model. Both the base
fields and the extension have schema validation tests. The loader rejects
duplicate ids across files and returns entries ordered by id. YAML duplicate keys
and unknown fields (including inside anchors) fail. The caller performs T2's
syntax/symbol checks on `applies_when` in its declared lowering context; the
loader does not implement a second DSL parser or evaluator.

`SourceTree` reads Git blobs at the supplied immutable full commit, ignoring
uncommitted edits and the checkout's HEAD. It locates module/package files and
class, sync/async function or nested method declarations with AST only. Imported
aliases, missing declarations, or ambiguous conditional definitions fail closed;
use the defining qualified symbol, not an import alias. `SourceAnchor.src` is a
contract SourceRef whose line is the declaration line; its `source` also includes
decorators. Upstream modules are never imported.

`anchor_hash(anchor.source)` removes comment-only lines before dedenting, then
tokenizes and hashes a canonical JSON list of token types/values. This includes
comments at column zero inside indented method fragments; comment-like text
inside strings is preserved. Comments, insignificant newlines, token positions and
formatting spaces are omitted; indent/dedent and logical newline tokens remain.
String literals/docstrings and decorators remain significant. This makes normal
whitespace/comment edits stable while preserving code and block changes. Hashes
include the `sha256:` prefix. Each hash covers one declaration and its nested
definitions. List relevant external implementations as additional anchors.

To compute a draft's anchor, call `sources[package].resolve(target).anchor_hash`;
never author a digest or a source line manually. Every committed annotation uses
the replacement class/function as both `target` and `resolves_to`. Linear notes
add the post-load method, NZ policy/conversion helpers and parallel selector;
the row parallel note anchors the independent output projection implementation.
RoPE also hashes the module-level derived-cache helper. Expert notes include the
NZ policy/conversion helpers alongside the class containing post-load processing.
If `resolves_to` differs, it is linked; add it to `anchors` if its changes must
invalidate the note too. Unlisted helper definitions are not recursively hashed.

`check_annotations` returns `(list[ResolvedAnnotation], list[StaleAnnotation])`.
The primary target and all extra anchors must match. A change in any declaration
excludes the entire annotation from the usable list. One stale item per annotation
reports the first changed target in symbol order, with that target's old/new
hashes, preserving the existing triage wire format. Append stale items to the
snapshot's existing Triage.items. Repeated anchor targets are rejected. Missing
targets, helpers or replacements raise ValueError and fail the build because the
triage schema has no unresolved-target variant.

`apply_annotations(ir, resolved, bindings=..., active_ids=...)` returns a deep IR
copy and leaves all inputs untouched. `bindings` maps node id to annotation id;
the caller obtains the relationship from actual extraction class swaps and
patched-symbol facts. `active_ids` is an explicit set selected by the shared TS
predicate evaluator, or a caller's already-established fixed snapshot context.
No default condition selection is assumed. Unknown/stale ids, unknown nodes,
duplicate ids and active source commits inconsistent with the IR snapshot fail.
Inactive bindings preserve existing Ascend facts, including extracted replacement
source links and OOT tags; unbound nodes are copied unchanged. Active notes set
`ascend.level/note/annotation_id`, merge tags, and add source links while retaining
extracted links. Tags and complete SourceRefs are sorted/deduplicated, and copies
share no mutable source objects with inputs. Apply mode changes to the original
extracted/lowered IR; this merge API cannot reconstruct which old links/tags came
from an earlier annotation. It does not rewrite weight/communication formulas:
lowering must model any noted semantic changes.

The wire format supports one annotation per node. Select a mutually exclusive
lane, put separately relevant patches on their own semantic node, or retain a
collision as triage; do not silently pick a note by severity or overwrite it by
file order. The factory patch may bind the MoE block while its class-swap note
binds the runner; the attention patch may bind the attention group while its
linear notes bind projections.

## Predicate context for the Qwen MoE drafts

These are explicit declarations integration must derive from facts, not
environment-name heuristics. Missing context is an error/conditional triage,
never false. Bind booleans as DSL boolean literals in symbols/context. Runtime
dependent selection must be recomputed when the corresponding setting changes.

| Predicate | Source-backed meaning |
| --- | --- |
| `is_a2` | Selected snapshot/profile device is A2. |
| `is_qwen3_moe` | Pinned model architecture is Qwen3MoeForCausalLM / model_type qwen3_moe. |
| `unquantized` | No active quantization config/method for the matched implementation. BF16 weight dtype alone does not prove this. |
| `dtype_w` | Standard IR weight dtype symbol; these notes target `bf16`. |
| `custom_parallel` | For Qwen's `o_proj`, `get_parallel_op` chooses `OProjRowParallelOp` when `oproj_tp_enable()` is true. Derive from the resolved prefix/settings, not FlashComm. The pinned column selector only customizes `wo_a` or `gate_up_proj`, so QKV has no custom-parallel lane or predicate. |
| `linear_nz` | For that unquantized linear's loaded weight, `_should_trans_nz` is true and no explicit skip flag bypasses conversion. This refers to the intended loaded NPU weight, never the meta tensor's always-false conversion test. |
| `vocab_parallel_override` | The matched embedding/head plan is FINE_GRAINED or PCP_X_TP rather than standard TP/replicated. For logits use its actual head's plan. |
| `mtp_draft` | Active forward is an MTP draft path; ordinary rotary-cache note excludes its optional position all-gather. |
| `use_mrope` | Actual attention rotary implementation is AscendMRotaryEmbedding. The Qwen text model uses the ordinary branch. |
| `capture_routing` | RoutedExpertsCapturer is bound/enabled in the active model execution. The function being patched alone does not prove it runs. |
| `redundant_experts` | Active EPLB configuration allocates redundant physical experts. |

Conditional custom-parallel and vocabulary notes summarize the possible changes
and require integration's actual branch to supply formulas. They are not evidence
that every listed collective executes in every mode. A2 MoE runner/container
notes describe the changed execution/communication implementation; singleton
groups do not imply real network traffic. Gate output dtype differs from upstream
in this pinned model because upstream leaves `out_dtype` unset while Ascend
defaults it to fp32.

Expert matrix transposition runs unconditionally after loading. Forced NZ casts
occur only when `enable_fused_mc2` is true and MegaMoE is not selected. With fused
MC2 disabled, the code calls `maybe_trans_nz`, so conversion depends on NZ policy;
the note does not claim all experts always use NZ. These are source facts, not
evidence of a post-load conversion having run in a meta build.

## Validation

From the repo root, with `INFERATLAS_SRC` pointing to the directory containing
`vllm/` and `vllm-ascend/`, run:

```sh
UV_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/ uv run --project generator pytest generator/tests/test_annotations*.py
uv run --project generator ruff check generator/inferatlas/annotations generator/tests/test_annotations*.py
uv run --project generator ruff format --check generator/inferatlas/annotations generator/tests/test_annotations*.py
```

Source tests also discover the experiment clones in a normal checkout or this
project's worktree layout. Without local clones those integration checks skip
explicitly; unit tests build temporary Git repositories and never use the network.
