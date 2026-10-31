"""Parse pinned vLLM/Ascend startup logs; no hardware is accessed here."""

import math
import re

from inferatlas.schema.models import Check

_NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_PATTERNS = {
    # Ascend prints GB but divides consumed_memory by 2**30.
    "weight_bytes": (
        rf"(?:Model loading took\s+{_NUMBER}\s+GiB|Loading model weights took\s+{_NUMBER}\s+GB)",
        2**30,
    ),
    "kv_tokens": (rf"(?:GPU|NPU)(?: \+ host)? KV cache size:\s*{_NUMBER}\s+tokens", 1),
    "available_kv_bytes": (
        r"Available KV cache memory:\s*(-?[0-9][0-9,]*(?:\.[0-9]+)?)\s+GiB",
        2**30,
    ),
    "maximum_concurrency": (
        rf"Maximum concurrency for [0-9,]+ tokens per request:\s*{_NUMBER}x",
        1,
    ),
}


def parse_vllm_logs(text: str) -> dict[str, float]:
    """Use the last occurrence of each metric, converting binary GiB to bytes."""
    metrics = {}
    for key, (pattern, scale) in sorted(_PATTERNS.items()):
        matches = list(re.finditer(pattern, text))
        if matches:
            value = next(group for group in matches[-1].groups() if group is not None)
            metrics[key] = float(value.replace(",", "")) * scale
    return metrics


def compare_calibration(
    predicted: dict[str, float], observed: dict[str, float], tolerance: float = 0.1
) -> list[Check]:
    if not 0 <= tolerance <= 1:
        raise ValueError("calibration tolerance must be between zero and one")
    checks = []
    for key in sorted(observed):
        actual = observed[key]
        if isinstance(actual, bool) or not isinstance(actual, (int, float)):
            raise ValueError(f"invalid calibration metric: {key}")
        if not math.isfinite(actual) or (actual < 0 and key != "available_kv_bytes"):
            raise ValueError(f"invalid calibration metric: {key}")
        if key not in predicted:
            checks.append(Check(id=f"calibration.{key}", status="skip", message="No estimate."))
            continue
        estimate = predicted[key]
        if not math.isfinite(estimate) or (estimate < 0 and key != "available_kv_bytes"):
            raise ValueError(f"invalid estimate: {key}")
        delta = abs(estimate - actual)
        limit = tolerance * abs(actual)
        passed = delta <= limit or math.isclose(delta, limit)
        checks.append(
            Check(
                id=f"calibration.{key}",
                status="pass" if passed else "fail",
                message=f"estimate={estimate:g}, observed={actual:g}, tolerance={tolerance:.0%}",
            )
        )
    return checks or [Check(id="calibration", status="skip", message="No calibration metrics.")]
