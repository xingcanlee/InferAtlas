"""Experiment 1: build Qwen3-MoE with plain vLLM (no Ascend plugin) on the meta device.

Checks
  - plain vLLM (VLLM_TARGET_DEVICE=empty build) can construct the model on CPU/meta;
  - parameter count matches the closed-form count;
  - with --tp N (FakeProcessGroup), per-rank shapes are the sharded ones.

Usage: python -I exp1_vllm_meta.py --tp 1
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import dump, expected_params, run_stage, summarize_model, write_model_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--tp", type=int, default=1)
args = ap.parse_args()

# Make sure the Ascend plugin is NOT loaded even if it is importable.
os.environ["VLLM_PLUGINS"] = ""
# No GPU/NPU: select vLLM's CPU platform so DeviceConfig has a device type.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

result: dict = {"tp": args.tp, "expected": expected_params()}
ctx: dict = {}


def stage_env():
    import torch
    import vllm

    from vllm.platforms import current_platform

    return {
        "torch": torch.__version__,
        "vllm": vllm.__version__,
        "platform": type(current_platform).__name__,
        "device_type": getattr(current_platform, "device_type", None),
    }


def stage_config():
    from vllm.engine.arg_utils import EngineArgs

    model_dir = write_model_dir()
    ea = EngineArgs(
        model=model_dir,
        skip_tokenizer_init=True,
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
    )
    ctx["vllm_config"] = ea.create_engine_config()
    pc = ctx["vllm_config"].parallel_config
    return {"model_dir": model_dir, "tp": pc.tensor_parallel_size, "world_size": pc.world_size}


def stage_dist():
    """Real 1-rank gloo init; for tp>1, present rank 0 of an N-rank TP group via a shim.

    A FakeProcessGroup does not survive vLLM's init handshakes (it hangs in
    collective setup), and layer construction only needs the TP world size and
    rank. So we initialize a genuine single-process group and override just
    those two attributes on the TP GroupCoordinator.
    """
    import vllm.distributed.parallel_state as ps
    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(ctx["vllm_config"]):
        ps.init_distributed_environment(
            world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29571", backend="gloo"
        )
        ps.ensure_model_parallel_initialized(1, 1)
    if args.tp > 1:

        class _ShimTP:
            def __init__(self, real, n):
                self._real, self.world_size, self.rank_in_group = real, n, 0
                self.ranks, self.rank = list(range(n)), 0

            def __getattr__(self, k):
                return getattr(self._real, k)

        ps._TP = _ShimTP(ps._TP, args.tp)
    return {"tp_world_size": ps.get_tensor_model_parallel_world_size(), "tp_rank": ps.get_tensor_model_parallel_rank()}


def stage_build():
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model

    vc = ctx["vllm_config"]
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vc), set_default_torch_dtype(vc.model_config.dtype), torch.device("meta"):
        model = initialize_model(vllm_config=vc)
    ctx["model"] = model
    s = summarize_model(model)
    exp = result["expected"]["total"]
    # With TP>1, embed/lm_head/attn/experts are sharded, so only check tp=1 exactly.
    if args.tp == 1:
        s["matches_expected"] = s["num_params"] == exp
        s["diff_vs_expected"] = s["num_params"] - exp
    return s


run_stage(result, "env", stage_env)
if run_stage(result, "config", stage_config)["ok"]:
    if run_stage(result, "dist", stage_dist)["ok"]:
        run_stage(result, "build", stage_build)

dump(f"exp1_vllm_meta_tp{args.tp}", result)
