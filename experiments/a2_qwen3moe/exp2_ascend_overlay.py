"""Experiment 2: extract the Ascend overlay facts on CPU under upstream mocks.

Outputs
  - hardware profile of the selected device (A2 by default);
  - REGISTERED_ASCEND_OPS (CustomOp name -> Ascend class);
  - the set of vllm / torch symbols actually patched, found by code origin:
    a function whose __code__ lives in vllm_ascend, or a class defined in
    vllm_ascend, reachable from a vllm.* / torch.* module namespace
    (functools.wraps keeps __module__, but not __code__, so wrappers are caught);
  - coverage of the documented targets in patch/__init__.py vs detected ones.

Usage: python -I exp2_ascend_overlay.py --device A2
"""

from __future__ import annotations

import argparse
import inspect
import os
import re
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import VLLM_ASCEND, dump, run_stage  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="A2", choices=["A2", "A3", "A5", "_310P", "310P"])
args = ap.parse_args()

SCAN_PREFIXES = ("vllm.", "torch.distributed", "torch.accelerator", "xgrammar")
result: dict = {"device": args.device}
ctx: dict = {}


def _origin(obj) -> str | None:
    """Return the vllm_ascend source file that defines obj, if any."""
    try:
        code = getattr(obj, "__code__", None)
    except Exception:  # lazy-import proxies (e.g. vllm audio extras)
        return None
    if isinstance(code, types.CodeType) and "vllm_ascend" in code.co_filename:
        return code.co_filename.split("vllm_ascend", 1)[1].lstrip("/")
    if inspect.isclass(obj) and (getattr(obj, "__module__", None) or "").startswith("vllm_ascend"):
        return obj.__module__
    return None


def scan() -> dict[str, str]:
    """Map 'module.attr[.member]' -> vllm_ascend origin for every foreign symbol."""
    found: dict[str, str] = {}
    for mname, mod in list(sys.modules.items()):
        if not mname.startswith(SCAN_PREFIXES) or not isinstance(mod, type(sys)):
            continue
        for attr, obj in list(vars(mod).items()):
            o = _origin(obj)
            if o:
                found[f"{mname}.{attr}"] = o
            # Methods replaced on upstream classes defined in this module.
            if inspect.isclass(obj) and getattr(obj, "__module__", "") == mname:
                for mem, val in list(vars(obj).items()):
                    fn = getattr(val, "__func__", val)
                    o2 = _origin(fn)
                    if o2:
                        found[f"{mname}.{attr}.{mem}"] = o2
    return found


def stage_mock():
    ns = __import__("mock_env").install("_310P" if args.device == "310P" else args.device)
    ctx["ns"] = ns
    ctx["before"] = scan()
    return {"foreign_symbols_before_patch": len(ctx["before"])}


def stage_profile():
    from vllm_ascend.device.hardware_profile import get_current_hardware_profile

    p = get_current_hardware_profile()
    return {
        "device_type": str(p._device_type),
        "attention_backend_family": str(p.attention_backend_family),
        "weight_layout_policy": str(p.weight_layout_policy),
        "moe_comm_policy": str(p.moe_comm_policy),
        "quantization_backend_family": str(p.quantization_backend_family),
        "capabilities": sorted(c.name for c in p.capabilities),
    }


def stage_patch_platform():
    ctx["ns"]["adapt_patch"](True)


def stage_patch_worker():
    ctx["ns"]["adapt_patch"](False)


def stage_customop():
    import vllm_ascend.utils as u

    ctx["ns"]["register_ascend_customop"]()
    ops = {k: f"{v.__module__}.{v.__qualname__}" for k, v in u.REGISTERED_ASCEND_OPS.items()}
    from vllm.model_executor.custom_op import CustomOp

    oot = getattr(CustomOp, "op_registry_oot", None)
    return {"registered": ops, "count": len(ops), "oot_registry_keys": sorted(oot) if oot else None}


def stage_diff():
    after = scan()
    patched = {k: v for k, v in after.items() if ctx["before"].get(k) != v}
    by_file: dict[str, list[str]] = {}
    for sym, f in sorted(patched.items()):
        by_file.setdefault(f, []).append(sym)
    ctx["patched"] = patched
    return {"patched_count": len(patched), "by_origin": by_file}


def stage_doc_coverage():
    doc = (VLLM_ASCEND / "vllm_ascend/patch/__init__.py").read_text()
    documented = sorted(set(re.findall(r"`((?:vllm|torch|xgrammar)\.[\w\.]+)`", doc)))
    detected = set(ctx.get("patched", {}))
    # A documented target counts as detected if any detected symbol equals or extends it.
    hit = [d for d in documented if any(s == d or s.startswith(d + ".") for s in detected)]
    miss = [d for d in documented if d not in hit]
    undocumented = sorted(
        s for s in detected if not any(s == d or s.startswith(d + ".") or d.startswith(s + ".") for d in documented)
    )
    return {
        "documented": len(documented),
        "documented_and_detected": len(hit),
        "documented_not_detected": miss,
        "detected_not_documented": undocumented,
    }


if run_stage(result, "mock", stage_mock)["ok"]:
    run_stage(result, "profile", stage_profile)
    run_stage(result, "patch_platform", stage_patch_platform)
    run_stage(result, "patch_worker", stage_patch_worker)
    run_stage(result, "customop", stage_customop)
    run_stage(result, "diff", stage_diff)
    run_stage(result, "doc_coverage", stage_doc_coverage)

dump(f"exp2_ascend_overlay_{args.device}", result)
