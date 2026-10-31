# Reviewing an Ascend overlay

Start with extraction's CustomOp table and runtime patched-symbol inventory. AST literals cross-check registration; semistructured patch comments are explanatory, not an authoritative target list. Use upstream CPU mocks; never import the plugin directly on an unprepared CPU process.

| Level | Evidence |
| --- | --- |
| KERNEL | Same logical shapes, dtypes, storage and collectives; device kernel or equivalent fusion changes. Hidden by default. |
| LAYOUT | Weight transposition, NZ storage, cache arrangement or derived buffers change without changing the logical operation. Note post-load versus meta behavior explicitly. |
| SEMANTIC | Activation dtype/shape, collective group/order, routing, or graph structure changes. Retain communication metadata even when fused with matmul. |

Tags describe mechanism, separately from significance: `OOT` for an observed class replacement; `PATCH` for a runtime patch; `QUANT` for quantization behavior; `FUSED` only when the applicable branch actually fuses operations. Do not label every patched function as fused.

Read the replacement's constructor, forward, post-load processing and called layout/communication helpers, plus the original implementation. Bind conditions from the selected hardware profile, pinned model config, quantization config, environment and `additional_config`. Different modes may need mutually exclusive notes for the same target. Unknown or unmodeled modes stay in triage.

The A2 Qwen MoE annotations in `annotations/a2/` cover the experiment's repeated RMSNorm swaps using a shared class annotation. Their helper predicates are defined in `annotations/README.md`; they must be explicitly supplied by integration, never inferred from the annotation filename. BF16/unquantized notes make no claims about quantized paths or actual NPU performance.

The `patch_qwen3vl` file also replaces the ordinary Qwen MoE attention forward. Its ordinary rotary branch is not the MRoPE fusion branch. `patch_fused_moe` replaces model-imported factory bindings, so inspect the factory result as well as CustomOp classes. Routed-expert capture is relevant only when routing capture is enabled; scheduler, API and unrelated model patches do not belong on the Qwen MoE graph.

Hash the target def/class using the annotation resolver and tokenizer normalizer. Each hash covers decorators, docstrings and nested declarations. Add relevant external implementations as `anchors: [{target, anchor_hash}]` in the shared Annotation contract, each computed by the same resolver; any changed anchor invalidates the whole note. Use the stale item's `target` to locate the changed declaration, then recompute every anchor for the affected annotation via script rather than replacing hash values. Linear NZ notes must cover post-load processing and NZ policy/conversion helpers; rotary derived-buffer notes must cover the module-level cache helper. QKV has no custom-parallel lane in the pinned column selector. Expert transposition always runs; forced NZ requires fused MC2 without MegaMoE, while the unfused path follows NZ policy. Pending notes are drafts, not human approval. No note should claim measured memory or kernel timing from CPU mocks.
