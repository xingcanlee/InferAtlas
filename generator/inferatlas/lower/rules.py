"""Validated YAML inheritance and deterministic class/version matching."""

from pathlib import Path
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pydantic import Field
from ruamel.yaml import YAML

from inferatlas.schema.models import Rule

DEFAULT_RULES = Path(__file__).resolve().parents[3] / "rules"


# Explicitly reviewed containers, engine sampling, and Ascend replacement modules.
SHORT_CLASS_MODULES = frozenset(
    {
        "torch.nn.modules.container",
        "vllm.model_executor.models.qwen2",
        "vllm.model_executor.models.qwen3",
        "vllm.model_executor.models.qwen3_moe",
        "vllm.v1.sample.sampler",
        "vllm_ascend.ops.linear",
        "vllm_ascend.ops.activation",
        "vllm_ascend.ops.layernorm",
        "vllm_ascend.ops.rotary_embedding",
        "vllm_ascend.ops.vocab_parallel_embedding",
        "vllm_ascend.ops.fused_moe.fused_moe",
        "vllm_ascend.ops.fused_moe.gate_linear",
        "vllm_ascend.ops.fused_moe.routed_experts",
    }
)


class RuleSpec(Rule):
    """Authoring metadata is expanded before exporting the public Rule contract."""

    inherits: str | None = None
    tags: list[str] = Field(default_factory=list)
    parameter_shapes: dict[str, list[str]] = Field(default_factory=dict)
    kv_per_token: str | None = None

    def contract(self) -> Rule:
        return Rule.model_validate(
            self.model_dump(exclude={"inherits", "tags", "parameter_shapes", "kv_per_token"})
        )


class RuleMatchError(ValueError):
    def __init__(self, symbol: str, candidates: list[RuleSpec]):
        self.symbol = symbol
        self.candidates = candidates
        super().__init__(f"{symbol}: expected one rule, found {[r.id for r in candidates]}")


def load_rules(path: str | Path = DEFAULT_RULES) -> list[RuleSpec]:
    """Load sorted YAML files; reject duplicate ids, missing bases and inheritance cycles."""
    path = Path(path)
    files = sorted(path.glob("*.yaml")) if path.is_dir() else [path]
    raw: dict[str, dict[str, Any]] = {}
    for file in files:
        entries = YAML(typ="safe").load(file.read_text())
        if not isinstance(entries, list):
            raise ValueError(f"{file.name}: rules must be a list")
        for item in entries:
            if item["id"] in raw:
                raise ValueError(f"duplicate rule id: {item['id']}")
            raw[item["id"]] = item
    resolved: dict[str, RuleSpec] = {}

    def resolve(key: str, visiting: frozenset[str] = frozenset()) -> RuleSpec:
        if key in resolved:
            return resolved[key]
        if key in visiting:
            raise ValueError(f"rule inheritance cycle: {key}")
        if key not in raw:
            raise ValueError(f"unknown inherited rule: {key}")
        item = raw[key]
        base = item.get("inherits")
        data = resolve(base, visiting | {key}).model_dump(by_alias=True) if base else {}
        data.update(item)
        rule = RuleSpec.model_validate(data)
        SpecifierSet(rule.match.vllm)
        SpecifierSet(rule.match.ascend)
        resolved[key] = rule
        return rule

    return [resolve(key) for key in sorted(raw)]


def match_rule(
    symbol: str,
    rules: list[RuleSpec],
    *,
    vllm_version: str,
    ascend_version: str | None = None,
    mro: tuple[str, ...] = (),
) -> RuleSpec:
    """Exact qualified, then short class, then each supplied MRO base in order.

    No upstream import is needed. MRO facts must come from extraction/the caller.
    Nonconstant predicates require reviewed resolution and are returned as ambiguity.
    """
    version = Version(vllm_version)
    ascend = Version(ascend_version) if ascend_version else None
    for cls in dict.fromkeys((symbol, *mro)):
        namespace, _, short = cls.rpartition(".")
        names = [cls]
        if namespace.startswith("vllm.model_executor.layers.") or namespace in SHORT_CLASS_MODULES:
            names.append(short)
        for name in dict.fromkeys(names):
            candidates = [
                r
                for r in rules
                if r.match.class_name == name
                and SpecifierSet(r.match.vllm).contains(version, prereleases=True)
                and (
                    not r.match.ascend
                    or ascend is not None
                    and SpecifierSet(r.match.ascend).contains(ascend, prereleases=True)
                )
                and r.match.when != "false"
            ]
            if candidates:
                if len(candidates) != 1 or candidates[0].match.when != "true":
                    raise RuleMatchError(symbol, candidates)
                return candidates[0]
    raise RuleMatchError(symbol, [])
