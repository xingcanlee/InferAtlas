"""Experiment 3: build Qwen3-MoE on meta with the Ascend CustomOps + patches active (mocked NPU).

Question: are parameter shapes still trustworthy when torch_npu is a MagicMock?
Compare against exp1 (plain vLLM): same param names/shapes => trustworthy;
MagicMock leaks (non-int shapes) or shape diffs => use the exp1 fallback.

Usage: python -I exp3_ascend_meta.py --device A2   (run exp1 --tp 1 first)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import OUT, dump, expected_params, run_stage, summarize_model, write_model_dir  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="A2")
args = ap.parse_args()

# vllm_ascend is on sys.path but not installed, so its platform plugin is not
# registered; use vLLM's CPU platform. CustomOp OOT registration is platform-independent.
os.environ.setdefault("VLLM_TARGET_DEVICE", "cpu")

result: dict = {"device": args.device, "expected": expected_params()}
ctx: dict = {}


def stage_mock():
    ns = __import__("mock_env").install(args.device)
    ns["adapt_patch"](True)
    ns["adapt_patch"](False)
    ns["register_ascend_customop"]()
    from vllm.platforms import current_platform

    return {"platform": type(current_platform).__name__}


def stage_config():
    from vllm.engine.arg_utils import EngineArgs

    ea = EngineArgs(
        model=write_model_dir(),
        skip_tokenizer_init=True,
        enforce_eager=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
    )
    vc = ea.create_engine_config()
    ctx["vllm_config"] = vc
    try:
        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(vc)
        return {"ascend_config": "initialized"}
    except Exception as e:  # recorded, not fatal
        return {"ascend_config": f"init failed: {type(e).__name__}: {e}"}


def stage_dist():
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment

    from vllm.config import set_current_vllm_config

    with set_current_vllm_config(ctx["vllm_config"]):
        _init_dist()


def _init_dist():
    from vllm.distributed.parallel_state import ensure_model_parallel_initialized, init_distributed_environment

    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method="tcp://127.0.0.1:29572", backend="gloo"
    )
    ensure_model_parallel_initialized(1, 1)


def stage_build():
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model

    vc = ctx["vllm_config"]
    from vllm.utils.torch_utils import set_default_torch_dtype

    with set_current_vllm_config(vc), set_default_torch_dtype(vc.model_config.dtype), torch.device("meta"):
        model = initialize_model(vllm_config=vc)
    return summarize_model(model)


def stage_compare():
    base_p = OUT / "exp1_vllm_meta_tp1.json"
    if not base_p.exists():
        return {"skipped": "run exp1 --tp 1 first"}
    base = json.loads(base_p.read_text()).get("build", {})
    cur = result["build"]
    bp, cp = base.get("layer0_params", {}), cur.get("layer0_params", {})
    shape_diff = {
        k: {"vllm": bp.get(k, {}).get("shape"), "ascend": cp.get(k, {}).get("shape")}
        for k in sorted(set(bp) | set(cp))
        if bp.get(k, {}).get("shape") != cp.get(k, {}).get("shape")
    }
    bc, cc = base.get("layer0_classes", {}), cur.get("layer0_classes", {})
    class_swaps = {k: {"vllm": bc.get(k), "ascend": cc.get(k)} for k in sorted(set(bc) | set(cc)) if bc.get(k) != cc.get(k)}
    return {
        "num_params": {"vllm": base.get("num_params"), "ascend": cur.get("num_params")},
        "layer0_shape_diff": shape_diff,
        "layer0_class_swaps": class_swaps,
        "bad_params": cur.get("bad_params"),
    }


if run_stage(result, "mock", stage_mock)["ok"] and run_stage(result, "config", stage_config)["ok"]:
    if run_stage(result, "dist", stage_dist)["ok"] and run_stage(result, "build", stage_build)["ok"]:
        run_stage(result, "compare", stage_compare)

dump(f"exp3_ascend_meta_{args.device}", result)
