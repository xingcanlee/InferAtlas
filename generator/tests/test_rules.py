"""Rule authoring, precedence, inheritance and version boundaries."""

import copy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from ruamel.yaml import YAML
from test_lower_helpers import environment, evaluate

from inferatlas.lower import RuleMatchError, load_rules, match_rule
from inferatlas.lower.context import bind_context, substitute
from inferatlas.schema.models import Extract, Rule, Rules

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def rules():
    return load_rules(ROOT / "rules")


def test_all_expanded_rules_follow_contract(rules):
    schema = json.loads((ROOT / "schemas/rule.schema.json").read_text())
    for r in rules:
        exported = r.contract().model_dump(by_alias=True, mode="json")
        Draft202012Validator(schema).validate(exported)
        assert Rule.model_validate(exported) == r.contract()
    Rules.model_validate([r.contract() for r in rules])
    assert [r.id for r in rules] == sorted({r.id for r in rules})


def test_twelve_measured_ascend_swaps_inherit_math(rules):
    experiment = json.loads(
        (ROOT / "experiments/a2_qwen3moe/out/exp3_ascend_meta_A2.json").read_text()
    )
    swaps = experiment["compare"]["layer0_class_swaps"]
    assert len(swaps) == 12
    for swap in swaps.values():
        base = match_rule(swap["vllm"], rules, vllm_version="0.32.0")
        replacement = match_rule(swap["ascend"], rules, vllm_version="0.32.0")
        assert replacement.inherits == base.id
        assert replacement.tags == ["ascend"]
        for field in ("weight", "flops", "out", "parameter_shapes", "emits"):
            assert getattr(base, field) == getattr(replacement, field)


def test_specific_class_precedes_base_and_mro(rules):
    qkv = match_rule(
        "vllm.model_executor.layers.linear.QKVParallelLinear",
        rules,
        vllm_version="0.32.0",
        mro=("ColumnParallelLinear",),
    )
    assert qkv.id == "linear.qkv"
    assert (
        match_rule(
            "pkg.NewLinear",
            rules,
            vllm_version="0.32.0",
            mro=("vllm.model_executor.layers.linear.RowParallelLinear",),
        ).id
        == "linear.row"
    )
    base = next(r for r in rules if r.id == "linear.row")
    qualified = base.model_copy(
        update={
            "id": "qualified",
            "match": base.match.model_copy(update={"class_name": "pkg.RowParallelLinear"}),
        }
    )
    assert (
        match_rule("pkg.RowParallelLinear", [*rules, qualified], vllm_version="0.32.0").id
        == "qualified"
    )


@pytest.mark.parametrize("version", ["0.18.9", "0.33.0", "1.0"])
def test_unsupported_version_is_not_guessed(rules, version):
    with pytest.raises(RuleMatchError):
        match_rule("RowParallelLinear", rules, vllm_version=version)


def test_overlaps_are_ambiguous_even_with_narrower_version_range(rules):
    original = next(r for r in rules if r.id == "linear.row")
    other = original.model_copy(
        update={
            "id": "overlap",
            "match": original.match.model_copy(update={"vllm": ">=0.32,<0.33"}),
        }
    )
    with pytest.raises(RuleMatchError) as error:
        match_rule("RowParallelLinear", [*rules, other], vllm_version="0.32.0")
    assert {r.id for r in error.value.candidates} == {"linear.row", "overlap"}


def test_ascend_version_and_predicates(rules):
    base = next(r for r in rules if r.id == "linear.row")
    restricted = base.model_copy(
        update={"match": base.match.model_copy(update={"ascend": ">=0.32,<0.33"})}
    )
    with pytest.raises(RuleMatchError):
        match_rule("RowParallelLinear", [restricted], vllm_version="0.32.0")
    assert (
        match_rule(
            "RowParallelLinear", [restricted], vllm_version="0.32.0", ascend_version="0.32.1"
        )
        == restricted
    )
    conditional = base.model_copy(update={"match": base.match.model_copy(update={"when": "tp>1"})})
    with pytest.raises(RuleMatchError) as error:
        match_rule("RowParallelLinear", [conditional], vllm_version="0.32.0")
    assert error.value.candidates == [conditional]


@pytest.mark.parametrize(
    "contents,match",
    [
        ([{"id": "a", "inherits": "b"}, {"id": "b", "inherits": "a"}], "cycle"),
        ([{"id": "a", "inherits": "missing"}], "unknown inherited"),
        ([{"id": "a"}, {"id": "a"}], "duplicate rule"),
    ],
)
def test_invalid_inheritance_is_rejected(tmp_path, contents, match):
    file = tmp_path / "rules.yaml"
    with file.open("w") as stream:
        YAML().dump(contents, stream)
    with pytest.raises(ValueError, match=match):
        load_rules(file)


def test_row_context_is_global_and_reusable_for_attention_and_dense_down(rules):
    from inferatlas.lower import lower

    extract = Extract.model_validate_json(
        (ROOT / "fixtures/qwen3-30b-a3b/extract-tp4.json").read_text()
    )
    row = next(r for r in rules if r.id == "linear.row")
    module = next(m for m in extract.modules if m.path.endswith("self_attn.o_proj"))
    assert bind_context(module, extract, row)["in_features"] == "4096"
    result = lower(extract)
    env = environment(result.ir)
    weight = substitute(row.weight, bind_context(module, extract, row))
    assert evaluate(weight, env) == 2048 * 1024 * 2
    # Same generic row rule, independently reconstructed local dense down-projection.
    dense = extract.model_copy(deep=True)
    dense.config.update(hidden_size=4096, intermediate_size=12288)
    down = module.model_copy(deep=True)
    down.path = "model.layers.0.mlp.down_proj"
    down.attrs.update(
        input_size=12288,
        output_size=4096,
        input_size_per_partition=3072,
        output_size_per_partition=4096,
    )
    down.params[0].shape = [4096, 3072]
    bindings = bind_context(down, dense, row)
    assert bindings["in_features"] == "12288"
    assert bindings["out_features"] == "H"
    assert evaluate(substitute(row.weight, bindings), env | {"H": 4096}) == 4096 * 3072 * 2


def test_rule_shapes_and_bytes_match_all_fixture_parameters(rules):
    from inferatlas.lower import lower

    for file in (ROOT / "fixtures/qwen3-30b-a3b").glob("extract*.json"):
        extract = Extract.model_validate_json(file.read_text())
        env = environment(lower(extract).ir)
        for module in extract.modules:
            rule = match_rule(module.class_name, rules, vllm_version="0.32.0")
            context = bind_context(module, extract, rule)
            expected = 0
            for param in module.params:
                shape = [
                    evaluate(substitute(x, context), env) for x in rule.parameter_shapes[param.name]
                ]
                assert shape == param.shape, (module.path, param.name)
                import math

                expected += math.prod(param.shape) * 2
            assert evaluate(substitute(rule.weight, context), env) == expected, module.path


def test_bias_and_vocabulary_padding(rules):
    from inferatlas.lower import lower

    data = json.loads((ROOT / "fixtures/qwen3-30b-a3b/extract.json").read_text())
    data["config"]["vocab_size"] = 151937
    for module in data["modules"]:
        if module["path"] in {"lm_head", "model.embed_tokens"}:
            module["params"][0]["shape"][0] = 152000
    row = next(m for m in data["modules"] if m["path"].endswith("o_proj"))
    row["params"].append({"name": "bias", "shape": [2048], "dtype": "bfloat16"})
    r = lower(data)
    assert not r.triage.items
    env = environment(r.ir, tp=4)
    n = next(n for n in r.ir.nodes if n.id == "attn.o")
    assert evaluate(n.weight_bytes, env) == 2048 * 1024 * 2 + 2048 * 2
    embed = next(n for n in r.ir.nodes if n.id == "embed")
    assert evaluate(embed.weight_bytes, env) == 38000 * 2048 * 2
    assert r.parameter_shapes[row["path"]]["bias"] == ["H"]
    bad = copy.deepcopy(data)
    next(m for m in bad["modules"] if m["path"] == "model.embed_tokens")["params"][0]["shape"][
        0
    ] -= 1
    assert lower(bad).triage.items


@pytest.mark.parametrize(
    "symbol",
    [
        "private.RowParallelLinear",
        "private.Attention",
        "vllm.other.RMSNorm",
        "vllm.model_executor.layers_evil.linear.RowParallelLinear",
    ],
)
def test_private_short_class_collisions_fail_closed(rules, symbol):
    with pytest.raises(RuleMatchError):
        match_rule(symbol, rules, vllm_version="0.32.0")
