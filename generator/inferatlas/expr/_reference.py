"""TEST ONLY oracle for shared vectors, never imported by the production pipeline.

The browser/Node implementation is the sole production evaluator. Keeping this small
interpreter independent catches precedence, laziness and numeric-semantics drift.
"""

import math

from . import DTYPE_BYTES, RESERVED, ExprSyntaxError, _error, _parse


def evaluate_reference(src, env):
    tree = _parse(src)
    if RESERVED.intersection(env):
        raise ExprSyntaxError("Cannot rebind builtin", code="type")

    def number(value, node):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            _error("Expected number", node, "type")
        if not math.isfinite(value):
            _error("Non-finite number", node, "non_finite")
        return float(value)

    def truth(value, node):
        return value if isinstance(value, bool) else number(value, node) != 0

    def interpret(node):
        parts = node.children
        kind = node.data
        if kind == "number":
            return float(parts[0])
        if kind == "boolean":
            return parts[0] == "true"
        if kind == "identifier":
            name = str(parts[0])
            if name in DTYPE_BYTES:
                return name
            if name not in env:
                _error(f"Unknown symbol {name}", node, "unknown_symbol")
            value = env[name]
            if not isinstance(value, (str, bool)):
                number(value, node)
                return float(value)
            return value
        if kind == "conditional":
            return run(parts[1] if truth(run(parts[0]), parts[0]) else parts[2])
        if kind == "unary":
            op, arg = parts
            value = run(arg)
            if op == "!":
                return not truth(value, arg)
            value = number(value, arg)
            return -value if op == "-" else value
        if kind in {"binary", "power"}:
            left, op, right = parts if kind == "binary" else (parts[0], "**", parts[1])
            a = run(left)
            if op == "&&":
                return truth(a, left) and truth(run(right), right)
            if op == "||":
                return truth(a, left) or truth(run(right), right)
            b = run(right)
            if op in {"==", "!="}:
                same_type = (isinstance(a, bool) == isinstance(b, bool)) and (
                    isinstance(a, str) == isinstance(b, str)
                )
                equal = same_type and a == b
                return equal if op == "==" else not equal
            a, b = number(a, left), number(b, right)
            if op in {"/", "%"} and b == 0:
                _error("Division by zero", node, "domain")
            if op == "+":
                return a + b
            if op == "-":
                return a - b
            if op == "*":
                return a * b
            if op == "/":
                return a / b
            if op == "%":
                return math.fmod(a, b)
            if op == "**":
                try:
                    return math.pow(a, b)
                except (ValueError, OverflowError):
                    _error("Non-finite power", node, "non_finite")
            return {"<": a < b, "<=": a <= b, ">": a > b, ">=": a >= b}[op]
        if kind == "call":
            name = str(parts[0])
            args = parts[1].children
            values = [run(arg) for arg in args]
            if name == "bytes":
                value = values[0]
                if not isinstance(value, str):
                    _error("Expected dtype string", node, "type")
                if value not in DTYPE_BYTES:
                    _error("Unknown dtype", node, "domain")
                return DTYPE_BYTES[value]
            values = [number(value, arg) for value, arg in zip(values, args, strict=True)]
            a = values[0]
            if name == "max":
                return max(values)
            if name == "min":
                return min(values)
            if name == "ceil":
                return math.ceil(a)
            if name == "floor":
                return math.floor(a)
            b = values[1]
            if a < 0 or b <= 0:
                _error("Invalid rounding domain", node, "domain")
            ratio = number(a / b, node)
            return math.ceil(ratio) * (b if name == "round_up" else 1)
        raise AssertionError(kind)

    # Check each intermediate result as well as the final value.
    def run(node):
        result = interpret(node)
        if not isinstance(result, (str, bool)):
            number(result, node)
        return result

    value = run(tree)
    return int(value) if isinstance(value, bool) else number(value, tree)
