"""Lark syntax and free-symbol validation. Production evaluation is TypeScript-only."""

import math
from pathlib import Path

from lark import Lark, Tree, UnexpectedInput

DTYPE_BYTES = {
    "fp32": 4,
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
    "int32": 4,
    "int64": 8,
}
FUNCTIONS = {"max", "min", "ceil", "floor", "cdiv", "round_up", "bytes"}
RESERVED = set(DTYPE_BYTES) | FUNCTIONS | {"true", "false"}
_PARSER = Lark(
    Path(__file__).with_name("grammar.lark").read_text(),
    parser="lalr",
    propagate_positions=True,
    maybe_placeholders=False,
)

__all__ = ["ExprSyntaxError", "validate"]


class ExprSyntaxError(ValueError):
    """Invalid DSL expression; offsets are zero-based, lines/columns one-based."""

    def __init__(self, message, *, code="syntax", offset=0, line=1, column=1):
        self.code = code
        self.offset = offset
        self.line = line
        self.column = column
        self.position = {"offset": offset, "line": line, "column": column}
        super().__init__(f"{message} at {line}:{column}")


def _error(message, tree, code="syntax"):
    meta = tree.meta
    raise ExprSyntaxError(
        message, code=code, offset=meta.start_pos, line=meta.line, column=meta.column
    )


def _parse(src: str) -> Tree:
    try:
        tree = _PARSER.parse(src)
    except UnexpectedInput as exc:
        # Lark may attach EOF to the previous token; report the end of the source.
        at_eof = getattr(getattr(exc, "token", None), "type", None) == "$END"
        offset = len(src) if at_eof or exc.pos_in_stream < 0 else exc.pos_in_stream
        line = src.count("\n", 0, offset) + 1
        column = offset - src.rfind("\n", 0, offset)
        raise ExprSyntaxError(
            "Invalid expression", offset=offset, line=line, column=column
        ) from exc
    for node in tree.iter_subtrees_topdown():
        if node.data == "number" and not math.isfinite(float(node.children[0])):
            _error("Non-finite literal", node, "non_finite")
        if node.data == "identifier" and str(node.children[0]) in {"NaN", "Infinity"}:
            _error("Non-finite names are not DSL literals", node)
        if node.data == "identifier" and str(node.children[0]) in FUNCTIONS:
            _error("Function names require parentheses", node)
        if node.data == "call":
            name = str(node.children[0])
            if name not in FUNCTIONS:
                _error("Unknown function", node)
            count = len(node.children[1].children) if len(node.children) > 1 else 0
            expected = 2 if name in {"cdiv", "round_up"} else 1
            valid = count >= 1 if name in {"max", "min"} else count == expected
            if not valid:
                _error(f"Invalid arity for {name}", node, "arity")
    return tree


def validate(src: str) -> list[str]:
    """Return sorted, unique free symbols in all branches, or raise ExprSyntaxError.

    Runtime types, environment bindings and arithmetic domains are checked by the TS
    evaluator. This validator does not evaluate formulas or discard unselected branches.
    """
    tree = _parse(src)
    return sorted(
        {
            str(node.children[0])
            for node in tree.iter_subtrees_topdown()
            if node.data == "identifier" and str(node.children[0]) not in RESERVED
        }
    )
