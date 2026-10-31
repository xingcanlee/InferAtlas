"""Source, golden and optional calibration verification."""

import subprocess
from pathlib import Path

from inferatlas.schema import IR, Extract, Report, Triage
from inferatlas.schema.models import Check

from .calibration import compare_calibration, parse_vllm_logs
from .golden import Evaluator, check_golden, model_slug
from .io import ROOT, read_json
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
    diffs = []
    try:
        diffs = check_golden(golden_dir or ROOT / "golden" / model_slug(ir), ir, evaluator)
        checks.append(
            Check(
                id="golden",
                status="fail" if diffs else "pass",
                message=f"{len(diffs)} golden differences.",
            )
        )
        checks.append(
            Check(
                id="evaluation",
                status="skip" if evaluator.backend == "fixture" else "pass",
                message=(
                    "Fixture numeric oracle; runtime evaluator unavailable."
                    if evaluator.backend == "fixture"
                    else f"Evaluated using {evaluator.backend}."
                ),
            )
        )
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        message = "Golden inputs unavailable." if isinstance(error, OSError) else str(error)
        checks.append(Check(id="golden", status="fail", message=message))
    if calibration is None:
        checks.append(Check(id="calibration", status="skip", message="No calibration supplied."))
    else:
        try:
            env = evaluator.bind(ir, {})
            weights, tokens = evaluator.many([ir.memory.weights, ir.memory.kv_tokens], env)
            checks.extend(
                compare_calibration({"weight_bytes": weights, "kv_tokens": tokens}, calibration)
            )
        except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
            checks.append(Check(id="calibration", status="fail", message=str(error)))
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
