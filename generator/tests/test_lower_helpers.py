"""Test-only reference arithmetic until T2's evaluator can be consumed in integration.

Lark parses the normative operators; this helper is not imported by production code.
"""

import math
import operator
from functools import lru_cache

from lark import Lark, Token

GRAMMAR = r"""
?start: conditional
?conditional: logical_or | logical_or "?" conditional ":" conditional -> choose
?logical_or: logical_and | logical_or "||" logical_and -> logical_or
?logical_and: equality | logical_and "&&" equality -> logical_and
?equality: comparison | equality "==" comparison -> eq | equality "!=" comparison -> ne
?comparison: sum | comparison "<" sum -> lt | comparison ">" sum -> gt
           | comparison "<=" sum -> le | comparison ">=" sum -> ge
?sum: product | sum "+" product -> add | sum "-" product -> sub
?product: unary | product "*" unary -> mul | product "/" unary -> div | product "%" unary -> mod
?unary: power | "-" unary -> neg | "+" unary -> pos | "!" unary -> logical_not
?power: primary | primary "**" unary -> pow
?primary: NUMBER -> number | IDENT "(" [conditional ("," conditional)*] ")" -> call
        | IDENT -> identifier | "(" conditional ")"
IDENT: /[A-Za-z_][A-Za-z_0-9]*/
NUMBER: /(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?/
%import common.WS
%ignore WS
"""
PARSER = Lark(GRAMMAR, parser="lalr")
DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp32": 4,
    "fp8": 1,
    "int8": 1,
    "int4": 0.5,
    "int32": 4,
    "int64": 8,
}
OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
    "mod": operator.mod,
    "pow": operator.pow,
    "lt": operator.lt,
    "gt": operator.gt,
    "le": operator.le,
    "ge": operator.ge,
    "eq": operator.eq,
    "ne": operator.ne,
    "neg": operator.neg,
    "pos": operator.pos,
    "logical_not": operator.not_,
}


@lru_cache(maxsize=4096)
def parse(expression):
    return PARSER.parse(expression)


def evaluate(expression, env):
    def visit(node):
        if isinstance(node, Token):
            return str(node)
        name, children = node.data, node.children
        if name == "number":
            return float(children[0])
        if name == "identifier":
            key = str(children[0])
            if key in DTYPE_BYTES:
                return key
            if key in {"true", "false"}:
                return key == "true"
            return env[key]
        if name == "choose":
            return visit(children[1] if visit(children[0]) else children[2])
        if name == "logical_and":
            return bool(visit(children[0]) and visit(children[1]))
        if name == "logical_or":
            return bool(visit(children[0]) or visit(children[1]))
        if name == "call":
            fn = str(children[0])
            args = [visit(c) for c in children[1:]]
            return {
                "max": max,
                "min": min,
                "ceil": math.ceil,
                "floor": math.floor,
                "cdiv": lambda x, y: math.ceil(x / y),
                "round_up": lambda x, y: math.ceil(x / y) * y,
                "bytes": lambda dtype: DTYPE_BYTES[dtype],
            }[fn](*args)
        return OPS[name](*(visit(c) for c in children))

    return visit(parse(expression))


def environment(ir, **overrides):
    env = {}
    for key, expr in ir.symbols.items():
        env[key] = evaluate(expr, env)
    for key, spec in ir.params.items():
        env[key] = spec.default if spec.kind in {"dtype", "enum"} else evaluate(spec.default, env)
    return env | overrides
