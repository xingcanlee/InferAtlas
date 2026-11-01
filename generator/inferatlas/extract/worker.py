"""CPU/meta subprocess. Keep imports stdlib-only until the upstream paths are set.

Run as a file with python -I: isolated mode ignores PYTHONPATH, so upstream
paths are installed explicitly from the host's commit-checked request.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import json
import os
import platform
import sys
import tempfile
import textwrap
import types
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

ATTRS = (
    "input_size",
    "output_size",
    "input_size_per_partition",
    "output_size_per_partition",
    "output_sizes",
    "output_partition_sizes",
    "num_heads",
    "num_kv_heads",
    "head_size",
    "head_dim",
    "total_num_heads",
    "total_num_kv_heads",
    "tp_size",
    "tp_rank",
    "ep_size",
    "ep_rank",
    "num_experts",
    "global_num_experts",
    "local_num_experts",
    "num_local_experts",
    "top_k",
    "intermediate_size",
    "intermediate_size_per_partition",
    "hidden_size",
    "vocab_size",
    "org_vocab_size",
    "num_embeddings",
    "org_num_embeddings",
    "num_embeddings_per_partition",
    "num_embeddings_padded",
    "org_vocab_size_padded",
    "padding_size",
    "embedding_dim",
    "reduce_results",
    "gather_output",
    "skip_bias_add",
    "disable_tp",
    "use_ep",
    "q_size",
    "kv_size",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "scaling",
    "eps",
    "variance_epsilon",
    "max_position_embeddings",
    "rotary_dim",
    "is_neox_style",
    "layer_type",
    "sliding_window",
    "num_shared_experts",
    "is_sparse",
    "compress_ratio",
    "index_topk",
    "index_n_heads",
    "index_head_dim",
)
SCAN_PREFIXES = ("vllm.", "torch.distributed", "torch.accelerator", "xgrammar")


def fqn(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


def source_ref(cls, request):
    """inspect starts at decorators; AST supplies the actual class/def line."""
    try:
        filename = inspect.getsourcefile(cls)
    except TypeError:  # Built-in bases of runtime-generated classes have no source.
        return None
    if filename is None:
        return None
    path = Path(filename).resolve()
    for repo, root in request["sources"].items():
        if path.is_relative_to(Path(root).resolve()):
            try:
                lines, first = inspect.getsourcelines(cls)
            except OSError:
                # CustomOp can generate a runtime subclass; link its declared ancestor.
                return next(
                    (ref for base in cls.__mro__[1:] if (ref := source_ref(base, request))),
                    None,
                )
            tree = ast.parse(textwrap.dedent("".join(lines)))
            declaration = next(
                node
                for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == cls.__name__
            )
            revision = request["snapshot"]["vllm" if repo == "vllm" else "ascend"]
            return {
                "repo": repo,
                "file": path.relative_to(root).as_posix(),
                "line": first + declaration.lineno - 1,
                "commit": revision["commit"],
                "symbol": fqn(cls),
            }
    # External containers have no vLLM/Ascend provenance in this snapshot.
    return None


def json_attr(value):
    if value is None or type(value) in (bool, int, float, str):
        return value
    if type(value) in (list, tuple):
        return [json_attr(item) for item in value]
    raise TypeError(f"non-JSON attribute: {type(value).__name__}")


def flatten(model, request):
    import torch

    rows = []
    seen_params = set()
    sources = {}
    for path, module in sorted(model.named_modules()):
        cls = type(module)
        if cls not in sources:
            sources[cls] = source_ref(cls, request)
        attrs = {}
        for name in ATTRS:
            try:
                value = getattr(module, name)
                if not inspect.isroutine(value):
                    attrs[name] = json_attr(value)
            except AttributeError:
                pass
        method = getattr(module, "quant_method", None)
        if method is not None:
            attrs["quant_method"] = fqn(type(method))
        params = []
        for name, param in sorted(module.named_parameters(recurse=False)):
            # Tied weights must be counted once, just as model.named_parameters().
            if id(param) in seen_params:
                continue
            seen_params.add(id(param))
            if not isinstance(param, torch.Tensor) or not param.is_meta:
                raise TypeError(f"non-meta parameter: {path}.{name}")
            if not all(type(size) is int for size in param.shape):
                raise TypeError(f"non-integer shape: {path}.{name}")
            params.append(
                {
                    "name": name,
                    "shape": list(param.shape),
                    "dtype": str(param.dtype).removeprefix("torch."),
                }
            )
        rows.append(
            {
                "path": path,
                "class": fqn(cls),
                "src": sources[cls],
                "attrs": attrs,
                "params": params,
            }
        )
    return rows


class _ShimTP:
    """Verified one-rank gloo coordinator exposing only construction-time ranks."""

    def __init__(self, real, n):
        self._real, self.world_size, self.rank_in_group = real, n, 0
        self.ranks, self.rank = list(range(n)), 0

    def __getattr__(self, key):
        return getattr(self._real, key)


def engine_config(request):
    from vllm.engine.arg_utils import EngineArgs

    def structure_config(config):
        # Kernel/packing selection needs actual hardware; construct logical BF16 weights.
        for section in (config, config.get_text_config()):
            if hasattr(section, "quantization_config"):
                delattr(section, "quantization_config")
        return config

    return EngineArgs(
        model=request["model_dir"],
        skip_tokenizer_init=True,
        tensor_parallel_size=request["parallel"]["tp"],
        enable_expert_parallel=request["parallel"]["ep"],
        enforce_eager=True,
        load_format="dummy",
        dtype="bfloat16",
        max_model_len=4096,
        hf_overrides=structure_config,
    ).create_engine_config()


def init_dist(config, request, ascend=False):
    import vllm.distributed.parallel_state as ps
    from vllm.config import set_current_vllm_config

    # A file rendezvous avoids fixed TCP ports and permits concurrent workers.
    with tempfile.TemporaryDirectory(prefix="inferatlas-gloo-") as directory:
        with set_current_vllm_config(config):
            ps.init_distributed_environment(
                world_size=1,
                rank=0,
                local_rank=0,
                distributed_init_method=Path(directory, "store").as_uri(),
                backend="gloo",
            )
            ps.ensure_model_parallel_initialized(1, 1)
            if ascend:
                from vllm_ascend.distributed import parallel_state as ascend_ps

                # Initialize real one-rank groups before exposing construction-time sizes.
                parallel = copy.copy(config.parallel_config)
                parallel.tensor_parallel_size = 1
                ascend_ps.init_ascend_model_parallel(parallel)
    tp = request["parallel"]["tp"]
    if tp > 1:
        ps._TP = _ShimTP(ps._TP, tp)
        if request["parallel"]["ep"] and ps._EP is not None:
            ps._EP = _ShimTP(ps._EP, tp)
        if ascend:
            ascend_ps._MC2 = _ShimTP(ascend_ps._MC2, tp)


def build(config):
    import torch
    from vllm.config import set_current_vllm_config
    from vllm.model_executor.model_loader.utils import initialize_model
    from vllm.utils.torch_utils import set_default_torch_dtype

    with (
        set_current_vllm_config(config),
        set_default_torch_dtype(config.model_config.dtype),
        torch.device("meta"),
        constructor_devices(),
        constructor_runtime(),
    ):
        return initialize_model(vllm_config=config)


def constructor_devices():
    """Keep explicit accelerator allocations inside the CPU/meta sandbox too."""
    import torch
    from torch.overrides import TorchFunctionMode

    def redirect(value):
        if isinstance(value, (str, torch.device)) and str(value).split(":")[0] in (
            "npu",
            "cuda",
            "xpu",
            "privateuseone",
        ):
            return "meta"
        return value

    class ConstructorDevices(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            kwargs = dict(kwargs or {})
            if "device" in kwargs:
                kwargs["device"] = redirect(kwargs["device"])
                if func.__name__ in ("empty", "empty_like") and str(kwargs["device"]) == "cpu":
                    kwargs["device"] = "meta"
            if func.__name__ in ("to", "to_empty"):
                args = tuple(redirect(arg) for arg in args)
            return func(*args, **kwargs)

    return ConstructorDevices()


@contextmanager
def constructor_runtime():
    """Use CPU attention plumbing and inert events; never execute accelerator kernels."""
    import torch
    from vllm.model_executor.layers.fused_moe import unquantized_fused_moe_method as moe
    from vllm.model_executor.layers.fused_moe.experts.cpu_moe import CPUUnquantizedExperts
    from vllm.model_executor.layers.fused_moe.oracle.unquantized import UnquantizedMoeBackend
    from vllm.platforms import current_platform

    original = current_platform.get_attn_backend_cls
    select_moe = moe.select_unquantized_moe_backend

    def backend(selected_backend, attn_selector_config, num_heads=None):
        return original(
            selected_backend,
            attn_selector_config._replace(use_sparse=False),
            num_heads,
        )

    def moe_backend(moe_config):
        try:
            return select_moe(moe_config)
        except NotImplementedError:
            # Weight creation is shared even when CPU cannot execute the activation.
            return UnquantizedMoeBackend.CPU, CPUUnquantizedExperts

    with (
        patch.object(type(current_platform), "get_attn_backend_cls", side_effect=backend),
        patch.object(torch.cuda, "Event"),
        patch.object(moe, "select_unquantized_moe_backend", side_effect=moe_backend),
    ):
        yield


def decoder_layer_types(config, model):
    """Read instantiated attention policy without requiring per-model rules."""
    text = config.get_text_config()
    declared = getattr(text, "layer_types", None)
    if declared:
        return list(declared)
    layers = next(
        (
            module
            for path, module in model.named_modules()
            if path.endswith("layers") and len(list(module.children())) == text.num_hidden_layers
        ),
        None,
    )
    if layers is None:
        return ["unknown"] * text.num_hidden_layers
    sparse = getattr(text, "sparse_attention_config", None) or {}
    sparse_layers = sparse.get("sparse_attention_freq", [])
    result = []
    for index, layer in enumerate(layers.children()):
        attention = [module for module in layer.modules() if hasattr(module, "attn_backend")]
        if (index < len(sparse_layers) and sparse_layers[index]) or any(
            getattr(module, "is_sparse", False) is True for module in layer.modules()
        ):
            result.append("sparse_attention")
        elif any(getattr(module, "sliding_window", None) for module in attention):
            result.append("sliding_attention")
        elif attention:
            result.append("full_attention")
        else:
            result.append("unknown")
    return result


def merge_overlay(baseline, actual, notes):
    """Preserve baseline identities where aligned; emit every actual Ascend module."""
    baseline = {row["path"]: row for row in baseline}
    actual = {row["path"]: row for row in actual}
    for path in sorted(baseline.keys() - actual.keys()):
        notes.append(f"Ascend removed baseline module: {path}")
    rows = []
    for path, replacement in sorted(actual.items()):
        row = baseline.get(path)
        if row is None or row["params"] != replacement["params"]:
            kind = "added module" if row is None else "changed parameter shapes"
            notes.append(f"Ascend {kind}: {path}; emitting actual Ascend structure.")
            row = replacement
        elif row["class"] != replacement["class"]:
            row["ascend"] = {
                "class": replacement["class"],
                "src": replacement["src"],
                "attrs": replacement["attrs"],
            }
        rows.append(row)
    return rows


def install_mock(root, device):
    # Match the verified experiment: reuse upstream mocks, never invent NPU stubs.
    info = types.ModuleType("vllm_ascend._build_info")
    info.__spec__ = importlib.util.spec_from_loader(info.__name__, loader=None)
    info.__device_type__ = "_310P" if device == "310P" else device
    sys.modules[info.__name__] = info
    path = root / "tests/ut/conftest.py"
    source = path.read_text()
    for block in ("\nadapt_patch()\nadapt_patch(True)\n", "\nregister_ascend_customop()\n"):
        if source.count(block) != 1:
            raise RuntimeError(f"upstream conftest changed; cannot strip {block!r}")
        source = source.replace(block, "\n")
    namespace = {"__name__": "inferatlas_mock_conftest", "__file__": str(path)}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def ascend_version(request):
    from setuptools_scm import get_version

    revision = request["snapshot"]["vllm"]
    # Fresh source-only checkouts lack _version.py; use upstream's SCM versioning.
    # Shallow, tagless clones yield SCM's placeholder (e.g. 0.1.dev1), not a release.
    # Ignore stale generated metadata and omit local build dates/dirty suffixes.
    os.environ["VLLM_VERSION"] = (
        get_version(root=request["sources"]["vllm"], local_scheme="no-local-version")
        if revision["tag"] == "main"
        else revision["tag"].removeprefix("v")
    )


def origin(obj, root):
    try:
        code = getattr(obj, "__code__", None)
    except Exception:  # Lazy optional-dependency proxies can raise on getattr.
        return None
    if isinstance(code, types.CodeType) and "vllm_ascend" in code.co_filename:
        return Path(code.co_filename).resolve().relative_to(root).as_posix()
    if inspect.isclass(obj) and (getattr(obj, "__module__", None) or "").startswith("vllm_ascend"):
        return Path(inspect.getfile(obj)).resolve().relative_to(root).as_posix()
    return None


def scan(root):
    found = {}
    for name, module in list(sys.modules.items()):
        if not name.startswith(SCAN_PREFIXES) or not isinstance(module, types.ModuleType):
            continue
        for attr, obj in list(vars(module).items()):
            file = origin(obj, root)
            if file:
                found[f"{name}.{attr}"] = (id(obj), file)
            if inspect.isclass(obj) and getattr(obj, "__module__", "") == name:
                for member, value in list(vars(obj).items()):
                    function = getattr(value, "__func__", value)
                    file = origin(function, root)
                    if file:
                        found[f"{name}.{attr}.{member}"] = (id(function), file)
    return found


def ascend_overlay(request):
    root = Path(request["sources"]["vllm-ascend"])
    namespace = install_mock(root, request["device"])
    before = scan(root)
    namespace["adapt_patch"](True)
    namespace["adapt_patch"](False)
    namespace["register_ascend_customop"]()
    from vllm_ascend.device.hardware_profile import get_current_hardware_profile
    from vllm_ascend.utils import REGISTERED_ASCEND_OPS

    profile = get_current_hardware_profile()
    after = scan(root)
    return {
        "device": request["device"],
        "profile": {
            "device_type": str(profile._device_type),
            **{
                key: str(getattr(profile, key))
                for key in (
                    "attention_backend_family",
                    "weight_layout_policy",
                    "moe_comm_policy",
                    "quantization_backend_family",
                )
            },
            "capabilities": sorted(cap.name for cap in profile.capabilities),
        },
        "custom_ops": {name: fqn(cls) for name, cls in sorted(REGISTERED_ASCEND_OPS.items())},
        "patched": {
            name: file
            for name, (identity, file) in sorted(after.items())
            if before.get(name) != (identity, file)
        },
    }


def run(request):
    os.environ["VLLM_TARGET_DEVICE"] = "cpu"
    os.environ["VLLM_PLUGINS"] = ""
    for root in request["sources"].values():
        sys.path.insert(0, root)
    import torch
    import vllm

    config = json.loads(Path(request["model_dir"], "config.json").read_text())
    vc = engine_config(request)
    init_dist(vc, request)
    baseline = build(vc)
    modules = flatten(baseline, request)
    layer_types = decoder_layer_types(vc.model_config.hf_config, baseline)
    overlay = None
    notes = [
        "CPU/meta construction only; no weights or forward pass.",
        "TP/EP coordinators expose rank zero shapes over a real one-rank gloo group.",
        "External container sources are null; shared parameters are emitted once.",
        "Accelerator allocations use meta; CUDA events are inert and attention uses CPU "
        "backend plumbing, including sparse attention. No backend forward is executed.",
        "Runtime-generated classes link to their nearest declared ancestor when available.",
        "Uninitialized explicit CPU allocations use meta; unsupported MoE execution "
        "backends use the upstream CPU expert weight constructor only.",
    ]
    if config.get("quantization_config") or config.get("text_config", {}).get(
        "quantization_config"
    ):
        notes.append(
            "Structure uses logical BF16 parameters; source quantization_config is preserved "
            "but packed weights, scales and hardware quantization kernels are not constructed."
        )
    if request["device"]:
        import vllm.distributed.parallel_state as ps

        ascend_version(request)
        # Recreate groups after patching, matching exp3's patch-before-init order.
        ps.destroy_model_parallel()
        ps.destroy_distributed_environment()
        overlay = ascend_overlay(request)
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        # get_rope caches instances across builds, bypassing new CustomOp registrations.
        _ROPE_DICT.clear()
        overlay_config = engine_config(request)
        from vllm_ascend.ascend_config import init_ascend_config

        init_ascend_config(overlay_config)
        init_dist(overlay_config, request, ascend=True)
        actual = flatten(build(overlay_config), request)
        modules = merge_overlay(modules, actual, notes)
    return {
        "snapshot": request["snapshot"],
        "fixture": False,
        "config": config,
        "parallel": request["parallel"],
        "modules": modules,
        "layer_types": layer_types,
        "ascend": overlay,
        "env": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "vllm": vllm.__version__,
        },
        "notes": notes,
    }


if __name__ == "__main__":
    request_path, response_path = map(Path, sys.argv[1:])
    try:
        result = run(json.loads(request_path.read_text()))
    except Exception as error:
        response_path.write_text(
            json.dumps(
                {"error": {"exception_type": type(error).__name__, "message": str(error)}},
                sort_keys=True,
            )
            + "\n"
        )
        raise
    response_path.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
