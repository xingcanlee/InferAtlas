"""Choose a usable first view with the shared Node memory evaluator."""

from inferatlas.schema import IR
from inferatlas.schema.models import Check

from .golden import Evaluator, ParameterDomainError

TP_DEFAULT_RULE = (
    "Build default: smallest valid TP choice with memory.kv_tokens > 0 on the snapshot "
    "device, keeping all other defaults; if none fits, use the smallest valid TP "
    "and retain the zero-capacity warning."
)


def select_default_tp(ir: IR, evaluator: Evaluator) -> Check:
    if evaluator.backend != "node":
        return Check(
            id="default_parameters",
            status="skip",
            message="Node evaluator unavailable; kept the extracted default parameters.",
        )
    candidates = []
    for tp in sorted({evaluator.scalar(choice, {}) for choice in ir.params["tp"].choices or []}):
        try:
            env = evaluator.bind(ir, {"tp": tp})
        except ParameterDomainError:
            # A TP choice may violate a cross-parameter constraint at the other defaults.
            continue
        tokens = evaluator.many([ir.memory.kv_tokens], env)[0]
        candidates.append((tp, tokens))
        if tokens > 0:
            break
    if not candidates:
        raise ValueError("no valid TP choice at the other default parameters")
    tp, tokens = next((point for point in candidates if point[1] > 0), candidates[0])
    ir.params["tp"].default = f"{tp:g}"
    if TP_DEFAULT_RULE not in ir.params["tp"].description:
        ir.params["tp"].description += " " + TP_DEFAULT_RULE
    return Check(
        id="default_parameters",
        status="pass",
        message=f"{TP_DEFAULT_RULE} Selected {ir.snapshot.device}: tp={tp}, "
        f"KV capacity={tokens:g} tokens."
        + (" No TP choice has positive KV capacity." if tokens <= 0 else ""),
    )
