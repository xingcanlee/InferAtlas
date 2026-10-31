# Parallel layers to inspect

Read declarations in `vllm.model_executor.layers` at the snapshot commit, together with the model constructor's arguments. CustomOp registration alone does not prove a layer is reachable.

| Layer | Accounting and communication checks |
| --- | --- |
| ColumnParallelLinear | Splits output features; input is replicated. Check whether `gather_output` reconstructs the output. |
| RowParallelLinear | Splits input features; check `input_is_parallel` and `reduce_results`. A reduction must be counted once. |
| QKVParallelLinear | Query heads shard while KV heads may replicate when TP exceeds the KV head count. Do not divide every packed output dimension uniformly. |
| MergedColumnParallelLinear | Each packed projection shards independently; preserve the logical projection boundaries. |
| ReplicatedLinear / GateLinear | Router weights are replicated. Verify parameter and output dtypes separately; Ascend's default logits dtype can differ. |
| VocabParallelEmbedding / ParallelLMHead | Apply padding before vocabulary sharding. Check tied weights and the actual communication group; independent head TP and PCP can override global TP. |
| MoERunner / RoutedExperts | The runner owns execution, the container owns expert weights. Without EP, the expert intermediate dimension follows effective MoE TP; with EP, experts distribute across the flattened expert group. |
| RMSNorm / RotaryEmbedding | Small parameters do not imply zero buffers. Check residual fusion, quantization bias, cos/sin caches and persistent versus nonpersistent ownership. |

`Extract.parallel.ep` is a boolean enable flag; the IR's `ep` is a degree. Use the actual extracted group sizes, never convert the boolean directly into a shard divisor. PP accounting uses each rank's resident layers and reports the heaviest rank, not an average. DP replicas do not multiply rank-local weights.

The CPU experiment uses a real singleton gloo group and a TP coordinator shim for meta construction. Do not replace it with FakeProcessGroup: upstream initialization can deadlock. Read `experiments/a2_qwen3moe/` before changing extraction assumptions. Meta verifies logical parameter shapes, not post-load NZ padding or NPU kernel behavior.
