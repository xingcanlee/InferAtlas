"""Load vllm-ascend under its own upstream CPU unit-test mocks (tests/ut/conftest.py).

We exec the upstream conftest with the three side-effect calls removed
(adapt_patch() / adapt_patch(True) / register_ascend_customop()), so the
caller decides when to apply patches and can observe state before and after.
"""

from __future__ import annotations

import importlib.util
import sys
import types

from common import VLLM_ASCEND

_STRIP = [
    "\nadapt_patch()\nadapt_patch(True)\n",
    "\nregister_ascend_customop()\n",
]


def install(device_type: str = "A2") -> dict:
    """Install upstream mocks and select the Ascend device type. Returns the exec namespace."""
    root = str(VLLM_ASCEND)
    if root not in sys.path:
        sys.path.insert(0, root)

    # Select the hardware profile before anything imports vllm_ascend.device.
    bi = types.ModuleType("vllm_ascend._build_info")
    bi.__spec__ = importlib.util.spec_from_loader("vllm_ascend._build_info", loader=None)
    bi.__device_type__ = device_type
    sys.modules["vllm_ascend._build_info"] = bi

    src = (VLLM_ASCEND / "tests" / "ut" / "conftest.py").read_text()
    for s in _STRIP:
        if s not in src:
            raise RuntimeError(f"upstream conftest changed; cannot strip {s!r}")
        src = src.replace(s, "\n")
    ns: dict = {"__name__": "inferatlas_mock_conftest", "__file__": "conftest.py"}
    exec(compile(src, str(VLLM_ASCEND / "tests/ut/conftest.py"), "exec"), ns)
    return ns
