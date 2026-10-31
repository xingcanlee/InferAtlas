"""Optional startup-log calibration; no hardware is accessed here."""

import math
import re

from inferatlas.schema.models import Check

_NUMBER = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
_PATTERNS = {
    "weight_bytes": (rf"Model loading took\s+{_NUMBER}\s+GiB", 2**30),
    "gpu_blocks": (rf"# GPU blocks:\s*{_NUMBER}", 1),
    "kv_tokens": (rf"(?:GPU\s+)?KV cache size:\s*{_NUMBER}\s+tokens", 1),
    "available_kv_bytes": (rf"Available KV cache memory:\s*{_NUMBER}\s+GiB", 2**30),
}


def parse_vllm_logs(text: str) -> dict[str, float]:
    """Use the last occurrence of each metric, converting GiB to bytes."""
    metrics = {}
    for key, (pattern, scale) in sorted(_PATTERNS.items()):
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            metrics[key] = float(matches[-1].replace(",", "")) * scale
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
        if not math.isfinite(actual) or actual < 0:
            raise ValueError(f"invalid calibration metric: {key}")
        if key not in predicted:
            checks.append(Check(id=f"calibration.{key}", status="skip", message="No estimate."))
            continue
        estimate = predicted[key]
        if not math.isfinite(estimate) or estimate < 0:
            raise ValueError(f"invalid estimate: {key}")
        delta = abs(estimate - actual)
        passed = delta <= tolerance * actual or math.isclose(delta, tolerance * actual)
        checks.append(
            Check(
                id=f"calibration.{key}",
                status="pass" if passed else "fail",
                message=f"estimate={estimate:g}, observed={actual:g}, tolerance={tolerance:.0%}",
            )
        )
    return checks or [Check(id="calibration", status="skip", message="No calibration metrics.")]
