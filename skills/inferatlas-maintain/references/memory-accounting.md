# Memory and golden review

Use pinned config and extracted parameter shapes as facts; use symbolic rules for adjustable parallel and workload settings. Compare golden changes only after checking snapshot identity and rule coverage.

The vLLM cache budget subtracts loaded weights, profile-run peak activation, non-torch allocations and graph reservations from the utilization-limited device budget. Divide the remaining budget by actual page bytes across resident cache groups and round down to complete blocks. Capacity and current occupancy are different quantities. Keep exclusions visible instead of inventing constants.

- Weights: use direct parameters without recursive double counting; include vocab padding, KV head replication, tied embedding ownership, quantization scales/zeros and reviewed storage padding. Meta shapes do not measure loaded NZ allocations.
- KV: grouped-query attention and MLA have different storage contracts. MLA latent cache is replicated across TP unless source proves otherwise; Ascend may separate rope/nope caches. Hybrid layers require distinct cache groups. Read actual backend/layout selection before reusing an attention rule.
- Activations: profiling uses `max_num_batched_tokens`; context length is not the prefill-profile token count. Sampler/logits buffers depend on `max_num_seqs`, vocabulary and their actual dtypes and may dominate the peak.
- Parallelism: report rank-local weights and the heaviest PP rank. Effective expert TP and EP must come from source-backed groups. Avoid summing a group summary with its children.
- Communication: logical tensor bytes and bus traffic have different formulas; do not charge collective traffic as HBM bytes moved. EP routing estimates need an explicit distribution assumption. A fused kernel does not erase communication.
- Unknown allocations: graph capture, allocator fragmentation and HCCL workspace require measured calibration or explicit exclusions. CPU/mock success is not calibration.

For calibration, compare the startup log's model-loading memory, available KV memory, token capacity and maximum concurrency using consistent units. Missing calibration should be visible. A golden discrepancy can be a real dtype/layout change even when parameter counts stay equal; inspect intermediate dtype and post-load transformations before blaming the evaluator.
