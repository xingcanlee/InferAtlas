"""Golden IR and numeric grids. Evaluation belongs to T2, never to this module."""

import json
import logging
import math
import shutil
import subprocess
from pathlib import Path

from inferatlas.expr import RESERVED, validate
from inferatlas.schema import IR, Eval
from inferatlas.schema.models import EvalCase, GoldenDifference

from .io import ROOT, read_json, wire, write_json


def escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def projection(ir: IR) -> dict:
    document = wire(ir)
    ids = [node["id"] for node in document["nodes"]]
    if len(ids) == len(set(ids)):
        document["nodes"] = {node["id"]: node for node in document["nodes"]}
    return document


def pointer(document, path: str):
    try:
        for part in path.removeprefix("/").split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            document = document[int(part)] if isinstance(document, list) else document[part]
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ValueError(f"golden grid pointer unavailable: {path}") from error
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
            duplicate_ids = []
            for values in (expected, actual):
                ids = [v["id"] for v in values]
                duplicate_ids.append(sorted({key for key in ids if ids.count(key) > 1}))
            if any(duplicate_ids):
                return [
                    GoldenDifference(
                        path=f"{path}/duplicate_ids",
                        expected=duplicate_ids[0],
                        actual=duplicate_ids[1],
                    )
                ]
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
        self.cli = Path(__file__).with_name("eval.mjs")
        self.backend = (
            "node"
            if (root / "packages/expr/dist/index.js").is_file() and shutil.which("node")
            else "fixture"
        )
        if self.backend == "fixture":
            self.warn(
                "SKIPPED numeric evaluation: node or @inferatlas/expr dist unavailable; "
                "run pnpm --filter @inferatlas/expr build."
            )

    def many(self, expressions: list[str], env: dict) -> list[float]:
        if self.backend == "node":
            result = subprocess.run(
                ["node", str(self.cli), str(self.root / "packages/expr/dist/index.js")],
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
        if (ir.symbols.keys() | ir.params.keys()) & RESERVED:
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
                    if set(validate(expr)) <= env.keys()
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
        for name, param in sorted(ir.params.items()):
            if param.constraint and not self.scalar(param.constraint, env):
                raise ValueError(f"{name} violates parameter constraint: {param.constraint}")
        return env


def evaluate_grid(ir: IR, grid: Eval, evaluator: Evaluator) -> Eval:
    if grid.model != ir.snapshot.model.hf_id:
        raise ValueError("evaluation grid model disagrees with IR")
    if len({case.id for case in grid.cases}) != len(grid.cases) or not grid.cases:
        raise ValueError("grid requires unique, nonempty cases")
    if any(not case.expected for case in grid.cases):
        raise ValueError("each grid case needs at least one expression pointer")
    document = projection(ir)
    if evaluator.backend != "node":
        raise ValueError("SKIPPED numeric evaluation: node or @inferatlas/expr dist unavailable")
    cases = []
    for case in sorted(grid.cases, key=lambda case: case.id):
        expected = {}
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


def golden_path(ir: IR) -> Path:
    return ROOT / "golden" / model_slug(ir) / ir.snapshot.device


def default_grid(ir: IR) -> Eval:
    """Seed pointers from the real IR; expected numbers are always computed by Node."""
    fields = {}

    def collect(value, path):
        if isinstance(value, str) or isinstance(value, list):
            fields[path] = 0 if isinstance(value, str) else [0] * len(value)
        elif isinstance(value, dict):
            for key, child in sorted(value.items()):
                collect(child, f"{path}/{escape(key)}")

    document = projection(ir)
    for key in ("weights", "kv_per_token", "kv_tokens", "activation_peak", "non_torch", "reserved"):
        collect(document["memory"][key], f"/memory/{key}")
    for node in document["nodes"].values():
        path = f"/nodes/{escape(node['id'])}"
        for key in ("weight_bytes", "flops", "bytes_moved"):
            collect(node[key], f"{path}/{key}")
        if node["comm"]:
            for key in ("when", "logical_bytes", "bus_bytes"):
                collect(node["comm"][key], f"{path}/comm/{key}")
        for key in ("in", "out"):
            for index, tensor in enumerate(node[key]):
                collect(tensor["shape"], f"{path}/{key}/{index}/shape")
    return Eval(
        model=ir.snapshot.model.hf_id,
        cases=[EvalCase(id=f"tp{tp}-bf16", env={"tp": tp}, expected=fields) for tp in (1, 2, 4, 8)],
    )


def update_golden(directory: Path, ir: IR, grid: Eval, evaluator: Evaluator) -> Eval:
    if evaluator.backend != "node":
        raise ValueError("golden update refuses fixture backend; build @inferatlas/expr first")
    values = evaluate_grid(ir, grid, evaluator)
    write_json(directory / "ir.json", ir)
    write_json(directory / "eval.json", values)
    return values


def check_golden(directory: Path, ir: IR, evaluator: Evaluator) -> list[GoldenDifference]:
    baseline = IR.model_validate(read_json(directory / "ir.json"))
    grid = Eval.model_validate(read_json(directory / "eval.json"))
    result = differences(projection(baseline), projection(ir), "/ir")
    # Duplicate identifiers must survive ID projection and cannot be evaluated by pointer.
    for expected, actual, path in (
        (wire(baseline)["nodes"], wire(ir)["nodes"], "/ir/nodes"),
        (wire(grid)["cases"], wire(grid)["cases"], "/eval/cases"),
    ):
        duplicate_diffs = [
            item
            for item in differences(expected, actual, path)
            if item.path.endswith("/duplicate_ids")
        ]
        if duplicate_diffs:
            return sorted(duplicate_diffs, key=lambda item: item.path)
    if evaluator.backend == "node":
        evaluated = evaluate_grid(ir, grid, evaluator)
        result.extend(differences(wire(grid), wire(evaluated), "/eval"))
    return sorted(result, key=lambda item: item.path)
