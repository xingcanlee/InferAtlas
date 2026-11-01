"""Host-side extraction API; torch/vLLM are loaded only in an isolated worker."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

from inferatlas.resolve import (
    ROOT,
    checkout_sources,
    experiment_dir,
    get_model,
    write_config,
)
from inferatlas.schema import Extract, Snapshot
from inferatlas.schema.models import CatalogModel, Device, ExtractParallel


class ExtractionError(RuntimeError):
    """A CPU/meta worker failed; no successful output was published."""

    def __init__(self, message: str, exception_type: str = "ExtractionError"):
        super().__init__(message)
        self.exception_type = exception_type


def extract(
    snapshot: Snapshot,
    model: str | CatalogModel,
    tp: int = 1,
    ep: bool = False,
    device: Device | None = None,
    out_path: str | Path | None = None,
) -> Extract:
    """Extract rank zero, with EP degree equal to TP when enabled (PP=DP=1).

    ``device=None`` constructs the baseline. A device constructs both baseline
    and Ascend models in one subprocess, preserving baseline source identities.
    The CPU interpreter is selected by INFERATLAS_PY; it needs no generator deps.
    """
    snapshot = Snapshot.model_validate(snapshot)
    parallel = ExtractParallel(tp=tp, ep=ep)
    entry = get_model(model)
    if (entry.hf_id, entry.config_rev) != (snapshot.model.hf_id, snapshot.model.config_rev):
        raise ValueError("model/config identity does not match snapshot")
    if device is not None and (device != snapshot.device or device == "H100"):
        raise ValueError("Ascend overlay device must match the snapshot's Ascend profile")
    python = Path(os.environ.get("INFERATLAS_PY") or experiment_dir() / ".venv/bin/python")
    if not python.is_file():
        raise FileNotFoundError(f"CPU interpreter missing: {python}; set INFERATLAS_PY")
    sources = checkout_sources(snapshot)
    worker = Path(__file__).with_name("worker.py")
    env = {
        **{key: value for key, value in os.environ.items() if not key.startswith("VLLM_")},
        "PYTHONPATH": str(worker.parent),
        "VLLM_TARGET_DEVICE": "cpu",
        "VLLM_PLUGINS": "",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
    }
    with tempfile.TemporaryDirectory(prefix="inferatlas-extract-") as temporary:
        scratch = Path(temporary)
        write_config(entry, scratch)
        request = {
            "snapshot": snapshot.model_dump(mode="json"),
            "parallel": parallel.model_dump(),
            "device": device,
            "sources": {name: str(path) for name, path in sources.items()},
            "model_dir": str(scratch),
        }
        request_path, response_path = scratch / "request.json", scratch / "response.json"
        request_path.write_text(json.dumps(request, sort_keys=True))
        try:
            result = subprocess.run(
                [str(python), "-I", str(worker), str(request_path), str(response_path)],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired as error:
            raise ExtractionError(
                "CPU/meta worker exceeded 300 seconds", "TimeoutExpired"
            ) from error
        if result.returncode:
            if response_path.is_file():
                failure = json.loads(response_path.read_text()).get("error")
                if failure:
                    raise ExtractionError(failure["message"], failure["exception_type"])
            raise ExtractionError(
                f"CPU/meta worker exited {result.returncode} without a structured error"
            )
        value = Extract.model_validate_json(response_path.read_text())
    if value.snapshot != snapshot or value.parallel != parallel or value.config != entry.config:
        raise ExtractionError("worker response identity/config does not match request")
    if (value.ascend is None) != (device is None):
        raise ExtractionError("worker response overlay does not match request")
    if out_path is not None:
        destination = Path(out_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Publish only complete, validated data, even if a previous output exists.
        with tempfile.NamedTemporaryFile(mode="w", dir=destination.parent, delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(
                    json.dumps(
                        value.model_dump(mode="json", by_alias=True), sort_keys=True, indent=2
                    )
                    + "\n"
                )
                stream.close()
                umask = os.umask(0)
                os.umask(umask)
                pending.chmod(0o666 & ~umask)
                pending.replace(destination)
            finally:
                pending.unlink(missing_ok=True)
    return value


__all__ = ["ExtractionError", "extract"]
