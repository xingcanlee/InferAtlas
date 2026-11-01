"""Source, golden and optional calibration verification."""

import math
import subprocess
from pathlib import Path

from inferatlas.schema import IR, Extract, Report, Triage
from inferatlas.schema.models import Check

from .calibration import compare_calibration, parse_vllm_logs
from .golden import Evaluator, check_golden, golden_path
from .io import read_json
from .source import check_sources


def verify_ir(
    ir: IR,
    *,
    extracts: list[Extract] | None = None,
    triage: Triage | None = None,
    repos: dict[str, Path],
    golden_dir: Path | None = None,
    calibration: dict[str, float] | None = None,
    evaluator: Evaluator | None = None,
    calibration_env: dict | None = None,
    fixture_stages: set[str] | None = None,
    allow_fixtures: bool = False,
) -> Report:
    checks = [Check(id="schema", status="pass", message="Input contracts validate.")]
    extracts = extracts or []
    if any(extract.snapshot != ir.snapshot for extract in extracts):
        raise ValueError("extract snapshot disagrees with IR")
    if triage is not None and triage.snapshot != ir.snapshot:
        raise ValueError("triage snapshot disagrees with IR")
    checks.append(
        Check(
            id="coverage",
            status="fail" if triage and triage.items else "pass",
            message=f"{len(triage.items) if triage else 0} blocking triage items.",
        )
        if triage is not None
        else Check(
            id="coverage", status="skip", message="No triage supplied; coverage is unverified."
        )
    )
    failures = check_sources([ir, *extracts], ir.snapshot, repos)
    checks.append(
        Check(
            id="source_links",
            status="fail" if failures else "pass",
            message=f"{len(failures)} invalid immutable source links.",
        )
    )
    evaluator = evaluator or Evaluator()
    blocked = bool(triage and triage.items)
    blocked_message = "Blocked by triage; resolve it first."
    for extract in sorted(extracts, key=lambda item: item.parallel.tp):
        check_id = f"weights.tp{extract.parallel.tp}"
        if blocked:
            checks.append(Check(id=check_id, status="skip", message=blocked_message))
            continue
        try:
            sizes = {"bfloat16": 2, "float16": 2, "float32": 4}
            observed = sum(
                math.prod(param.shape) * sizes[param.dtype]
                for module in extract.modules
                for param in module.params
            )
            env = evaluator.bind(
                ir,
                {
                    "tp": extract.parallel.tp,
                    "pp": extract.parallel.pp,
                    "dp": extract.parallel.dp,
                    "ep": extract.parallel.tp * extract.parallel.dp if extract.parallel.ep else 1,
                },
            )
            predicted = evaluator.many([ir.memory.weights], env)[0]
            equal = predicted == observed
            checks.append(
                Check(
                    id=check_id,
                    status="pass" if equal else "fail",
                    message=f"IR {predicted} bytes; extracted rank-zero parameters {observed} bytes.",
                )
            )
        except (ValueError, KeyError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            checks.append(Check(id=check_id, status="fail", message=str(error)))
    diffs = []
    if blocked:
        checks.append(Check(id="golden", status="skip", message=blocked_message))
    else:
        try:
            diffs = check_golden(golden_dir or golden_path(ir), ir, evaluator)
            checks.append(
                Check(
                    id="golden",
                    status="fail" if diffs else "pass",
                    message=f"{len(diffs)} golden differences.",
                )
            )
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
            message = "Golden inputs unavailable." if isinstance(error, OSError) else str(error)
            checks.append(Check(id="golden", status="fail", message=message))
    unavailable = evaluator.backend != "node"
    checks.append(
        Check(
            id="evaluation",
            status="skip"
            if blocked
            else (("skip" if allow_fixtures else "fail") if unavailable else "pass"),
            message=blocked_message
            if blocked
            else (
                "SKIPPED numeric evaluation; node/dist unavailable."
                if unavailable
                else "Evaluated using node."
            ),
        )
    )
    if blocked:
        checks.append(Check(id="calibration", status="skip", message=blocked_message))
    elif calibration is None:
        checks.append(Check(id="calibration", status="skip", message="No calibration supplied."))
    else:
        try:
            overrides = dict(calibration_env or {})
            max_len = overrides.pop("max_model_len", None)
            if max_len is not None and (type(max_len) is not int or max_len <= 0):
                raise ValueError("max_model_len must be a positive integer")
            env = evaluator.bind(ir, overrides)
            available = (
                f"hbm_bytes*gpu_memory_utilization-({ir.memory.weights})"
                f"-({ir.memory.activation_peak})-({ir.memory.non_torch})"
                f"-({ir.memory.reserved})"
            )
            weights, tokens, available_bytes = evaluator.many(
                [ir.memory.weights, ir.memory.kv_tokens, available], env
            )
            predicted = {
                "weight_bytes": weights,
                "kv_tokens": tokens,
                "available_kv_bytes": available_bytes,
            }
            if max_len is not None:
                predicted["maximum_concurrency"] = tokens / max_len
            checks.extend(compare_calibration(predicted, calibration))
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            checks.append(Check(id="calibration", status="fail", message=str(error)))
    fixtures = set(fixture_stages or ())
    if any(extract.fixture for extract in extracts):
        fixtures.add("extract")
    if evaluator.backend != "node" and allow_fixtures:
        fixtures.add("evaluation")
    checks.append(
        Check(
            id="provenance",
            status="fail" if fixtures else "pass",
            message="Fixture stages: " + ", ".join(sorted(fixtures))
            if fixtures
            else "All stages used production backends.",
        )
    )
    return Report(
        snapshot=ir.snapshot,
        checks=sorted(checks, key=lambda check: (check.id, check.node or "")),
        golden_diff=diffs,
        source_link_failures=failures,
    )


def exit_code(report: Report, triage: Triage | None = None) -> int:
    if any(check.status == "fail" and check.id != "coverage" for check in report.checks):
        return 3
    if (triage and triage.items) or any(
        check.id == "coverage" and check.status == "fail" for check in report.checks
    ):
        return 2
    return 0


__all__ = [
    "Evaluator",
    "check_golden",
    "check_sources",
    "compare_calibration",
    "exit_code",
    "parse_vllm_logs",
    "read_json",
    "verify_ir",
]
