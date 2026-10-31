"""Golden IR and numeric grids. Evaluation belongs to T2, never to this module."""

import importlib
import json
import logging
import math
import re
import shutil
import subprocess
from pathlib import Path

from inferatlas.schema import IR, Eval
from inferatlas.schema.models import EvalCase, GoldenDifference

from .io import ROOT, read_json, wire, write_json

BUILTINS = set("max min ceil floor round_up cdiv bytes true false".split()) | set(
    "fp32 bf16 fp16 fp8 int8 int4 int32 int64".split()
)


def escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def projection(ir: IR) -> dict:
    document = wire(ir)
    document["nodes"] = {node["id"]: node for node in document["nodes"]}
    return document


def pointer(document, path: str):
    for part in path.removeprefix("/").split("/"):
        part = part.replace("~1", "/").replace("~0", "~")
        document = document[int(part)] if isinstance(document, list) else document[part]
    return document


def differences(expected, actual, path="") -> list[GoldenDifference]:
    """Diff nodes/cases by id so ordering does not hide the field that changed."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        result = []
        for key in sorted(expected.keys() | actual.keys()):
            child = f"{path}/{escape(key)}"
            if key not in expected or key not in actual:
                result.append(
                    GoldenDifference(path=child, expected=expected.get(key), actual=actual.get(key))
                )
            else:
                result.extend(differences(expected[key], actual[key], child))
        return result
    if isinstance(expected, list) and isinstance(actual, list):
        if all(isinstance(v, dict) and "id" in v for v in expected + actual):
            return differences({v["id"]: v for v in expected}, {v["id"]: v for v in actual}, path)
        result = []
        for index in range(max(len(expected), len(actual))):
            if index >= len(expected) or index >= len(actual):
                result.append(
                    GoldenDifference(
                        path=f"{path}/{index}",
                        expected=expected[index] if index < len(expected) else None,
                        actual=actual[index] if index < len(actual) else None,
                    )
                )
            else:
                result.extend(differences(expected[index], actual[index], f"{path}/{index}"))
        return result
    equal = expected == actual
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        equal = math.isclose(expected, actual, rel_tol=1e-12, abs_tol=1e-9)
    return [] if equal else [GoldenDifference(path=path, expected=expected, actual=actual)]


def format_diff(items: list[GoldenDifference]) -> str:
    return "\n".join(
        f"{item.path}: {json.dumps(item.expected, sort_keys=True, ensure_ascii=False)}"
        f" -> {json.dumps(item.actual, sort_keys=True, ensure_ascii=False)}"
        for item in sorted(items, key=lambda item: item.path)
    )


class Evaluator:
    """Small process adapter; an installed-but-broken evaluator is a failure."""

    def __init__(self, root: Path = ROOT, warn=None):
        self.root = root
        self.warn = warn or logging.getLogger(__name__).warning
        self.cli = root / "packages/expr/dist/cli.js"
        self.reference = None
        if self.cli.is_file() and shutil.which("node"):
            self.backend = "node"
            return
        self.warn("Expression CLI unavailable; falling back to Python reference evaluator.")
        module = importlib.import_module("inferatlas.expr")
        self.reference = getattr(module, "evaluate", None)
        self.backend = "python" if callable(self.reference) else "fixture"
        if self.backend == "fixture":
            # shortcut: only fixture points are evaluable, remove when T2's runtime CLI is available.
            self.warn(
                "Python reference evaluator unavailable; using fixture oracle only for matching "
                "expressions and parameter points. No new values can be evaluated."
            )

    def many(self, expressions: list[str], env: dict) -> list[float]:
        if self.backend == "node":
            result = subprocess.run(
                ["node", str(self.cli)],
                cwd=self.root,
                input=json.dumps({"expressions": expressions, "env": env}, allow_nan=False),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
            if result.returncode:
                raise ValueError(f"expression CLI failed: {result.stderr.strip()}")
            payload = json.loads(result.stdout)
            values = payload.get("values") if isinstance(payload, dict) else payload
        elif self.backend == "python":
            values = [self.reference(expression, env) for expression in expressions]
        else:
            raise ValueError("no evaluator for new expressions; build @inferatlas/expr first")
        if not isinstance(values, list) or len(values) != len(expressions):
            raise ValueError("expression evaluator returned the wrong number of values")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
            raise ValueError("expression evaluator returned a non-finite/non-numeric value")
        return values

    def scalar(self, expression: str, env: dict):
        # Literal decoding is environment binding, not a second DSL evaluator.
        try:
            value = json.loads(expression)
        except ValueError:
            value = None
        if type(value) in (int, float, bool):
            if not math.isfinite(value):
                raise ValueError("non-finite environment literal")
            return value
        return self.many([expression], env)[0]

    def bind(self, ir: IR, overrides: dict) -> dict:
        if set(overrides) - ir.params.keys():
            raise ValueError("grid may override declared runtime parameters only")
        if (ir.symbols.keys() | ir.params.keys()) & BUILTINS:
            raise ValueError("reserved DSL identifier cannot be bound")
        if ir.symbols.keys() & ir.params.keys():
            raise ValueError("runtime parameter shadows immutable symbol")
        env = {}

        def resolve(bindings):
            pending = dict(bindings)
            while pending:
                ready = [
                    name
                    for name, expr in sorted(pending.items())
                    if set(re.findall(r"\b[A-Za-z_]\w*\b", expr)) - BUILTINS <= env.keys()
                ]
                if not ready:
                    raise ValueError("cyclic or unknown environment symbol")
                for name in ready:
                    env[name] = self.scalar(pending.pop(name), env)

        resolve(ir.symbols)
        pending = {}
        for name, param in ir.params.items():
            if name in overrides:
                continue
            if param.kind in ("dtype", "enum"):
                env[name] = param.default
            else:
                pending[name] = param.default
        env.update(overrides)
        resolve(pending)
        for name, param in sorted(ir.params.items()):
            value = env[name]
            if param.kind in ("integer", "number"):
                if type(value) not in (int, float) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if param.kind == "integer" and int(value) != value:
                    raise ValueError(f"{name} must be an integer")
                for bound, direction in ((param.min, -1), (param.max, 1)):
                    if bound is not None and direction * value > direction * self.scalar(
                        bound, env
                    ):
                        raise ValueError(f"{name} outside parameter domain")
                if param.step is not None:
                    step = self.scalar(param.step, env)
                    origin = self.scalar(param.min, env) if param.min is not None else 0
                    if step <= 0 or not math.isclose(
                        (value - origin) / step, round((value - origin) / step), abs_tol=1e-9
                    ):
                        raise ValueError(f"{name} outside parameter step")
            elif param.kind == "boolean" and type(value) is not bool:
                raise ValueError(f"{name} must be a boolean")
            elif param.kind in ("dtype", "enum") and not isinstance(value, str):
                raise ValueError(f"{name} must be a string")
            if param.choices:
                choices = [
                    self.scalar(c, env) if param.kind in ("integer", "number") else c
                    for c in param.choices
                ]
                if value not in choices:
                    raise ValueError(f"{name} outside parameter choices")
        return env


def evaluate_grid(ir: IR, grid: Eval, evaluator: Evaluator) -> Eval:
    if grid.model != ir.snapshot.model.hf_id:
        raise ValueError("evaluation grid model disagrees with IR")
    if len({case.id for case in grid.cases}) != len(grid.cases) or not grid.cases:
        raise ValueError("grid requires unique, nonempty cases")
    if any(not case.expected for case in grid.cases):
        raise ValueError("each grid case needs at least one expression pointer")
    document = projection(ir)
    oracle_ir = oracle = None
    if evaluator.backend == "fixture":
        fixture = evaluator.root / "fixtures" / model_slug(ir)
        oracle_ir = IR.model_validate(read_json(fixture / "ir.json"))
        oracle = Eval.model_validate(read_json(fixture / "eval.json"))
        if ir.symbols != oracle_ir.symbols or ir.params != oracle_ir.params:
            raise ValueError("fixture oracle cannot evaluate changed symbols/parameters")
    cases = []
    for case in sorted(grid.cases, key=lambda case: case.id):
        expected = {}
        if evaluator.backend == "fixture":
            match = next((item for item in oracle.cases if item.env == case.env), None)
            if match is None:
                raise ValueError(f"fixture oracle has no parameter point: {case.id}")
            for path in sorted(case.expected):
                if pointer(document, path) != pointer(projection(oracle_ir), path):
                    raise ValueError(f"fixture oracle cannot evaluate changed expression: {path}")
                if path not in match.expected:
                    raise ValueError(f"fixture oracle has no numeric value: {path}")
                expected[path] = match.expected[path]
        else:
            env = evaluator.bind(ir, case.env)
            expressions = []
            fields = []
            for path in sorted(case.expected):
                expr = pointer(document, path)
                values = expr if isinstance(expr, list) else [expr]
                if not values or not all(isinstance(value, str) for value in values):
                    raise ValueError(f"grid pointer must select expression(s): {path}")
                fields.append((path, len(values), isinstance(expr, list)))
                expressions.extend(values)
            values = evaluator.many(expressions, env)
            start = 0
            for path, size, array in fields:
                expected[path] = values[start : start + size] if array else values[start]
                start += size
        cases.append(EvalCase(id=case.id, env=case.env, expected=expected))
    return Eval(model=grid.model, cases=cases)


def model_slug(ir: IR) -> str:
    return ir.snapshot.model.hf_id.rsplit("/", 1)[-1].lower()


def update_golden(directory: Path, ir: IR, grid: Eval, evaluator: Evaluator) -> Eval:
    values = evaluate_grid(ir, grid, evaluator)
    write_json(directory / "ir.json", ir)
    write_json(directory / "eval.json", values)
    return values


def check_golden(directory: Path, ir: IR, evaluator: Evaluator) -> list[GoldenDifference]:
    baseline = IR.model_validate(read_json(directory / "ir.json"))
    grid = Eval.model_validate(read_json(directory / "eval.json"))
    result = differences(projection(baseline), projection(ir), "/ir")
    # Structural changes still get a useful per-field diff when no evaluator is installed.
    try:
        evaluated = evaluate_grid(ir, grid, evaluator)
    except ValueError:
        if result and evaluator.backend == "fixture":
            return sorted(result, key=lambda item: item.path)
        raise
    result.extend(differences(wire(grid), wire(evaluated), "/eval"))
    return sorted(result, key=lambda item: item.path)
