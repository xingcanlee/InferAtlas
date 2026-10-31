"""Symbolic lowering against independent extraction, byte ledgers and graph contracts."""

import copy
import json
import math
import os
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from ruamel.yaml import YAML
from test_lower_helpers import environment, evaluate, parse

from inferatlas.lower import coverage_gate, load_rules, lower
from inferatlas.lower.catalog import catalog_extract, lower_catalog
from inferatlas.schema.models import Catalog, Extract, Snapshot

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "fixtures/qwen3-30b-a3b/extract.json"
SOURCE = Path("/home/cyberspace/projects/InferAtlas/experiments/a2_qwen3moe/src")


@pytest.fixture
def data():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def catalog():
    return Catalog.model_validate(YAML(typ="safe").load(ROOT / "catalog.yaml"))


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    if SOURCE.is_dir():
        return SOURCE
    # Portable AST-only declarations for the config projection boundary. No vLLM import.
    path = tmp_path_factory.mktemp("sources")
    classes = {
        "models/qwen3": ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3DecoderLayer", "Qwen3Attention"],
        "models/qwen2": ["Qwen2MLP"],
        "layers/vocab_parallel_embedding": ["VocabParallelEmbedding", "ParallelLMHead"],
        "layers/layernorm": ["RMSNorm"],
        "layers/logits_processor": ["LogitsProcessor"],
        "layers/linear": ["QKVParallelLinear", "RowParallelLinear", "MergedColumnParallelLinear"],
        "layers/rotary_embedding/base": ["RotaryEmbedding"],
        "layers/rotary_embedding/common": ["ApplyRotaryEmb"],
        "layers/attention/attention": ["Attention"],
        "layers/activation": ["SiluAndMul"],
    }
    for name, names in classes.items():
        file = path / "vllm/vllm/model_executor" / f"{name}.py"
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text("\n".join(f"class {name}:\n    pass\n" for name in names))
    file = path / "vllm/vllm/v1/sample/sampler.py"
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text("class Sampler:\n    pass\n")
    return path


@pytest.mark.parametrize(
    "filename,total", [("extract.json", 61064245248), ("extract-tp4.json", 15285252096)]
)
def test_moe_schema_exact_weights_shapes_and_determinism(filename, total, tmp_path):
    path = FIXTURE.with_name(filename)
    extract = Extract.model_validate_json(path.read_text())
    result = lower(extract)
    assert coverage_gate(result.triage)
    env = environment(result.ir)
    assert evaluate(result.ir.memory.weights, env) == total
    for module in extract.modules:
        for param in module.params:
            assert [
                evaluate(x, env) for x in result.parameter_shapes[module.path][param.name]
            ] == param.shape
    for name, model in [("ir", result.ir), ("triage", result.triage)]:
        schema = json.loads((ROOT / f"schemas/{name}.schema.json").read_text())
        Draft202012Validator(schema).validate(model.model_dump(by_alias=True, mode="json"))
    result.write(tmp_path)
    first = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    reversed_extract = extract.model_copy(update={"modules": list(reversed(extract.modules))})
    lower(reversed_extract).write(tmp_path)
    assert first == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_all_expressions_resolve_in_both_phases(data, tp):
    result = lower(data)
    env = environment(result.ir, tp=tp, T=256, batch=2, S=3, context=1024)
    for node in result.ir.nodes:
        for phase in ("prefill", "decode"):
            for expr in [
                node.weight_bytes,
                getattr(node.flops, phase),
                getattr(node.bytes_moved, phase),
            ]:
                value = evaluate(expr, env)
                assert math.isfinite(value) and value >= 0, (node.id, expr)
            for tensor in [*node.inputs, *node.out]:
                shape = tensor.shape
                local = (
                    env | {"T": env["S"]} if isinstance(shape, list) and phase == "decode" else env
                )
                dims = shape if isinstance(shape, list) else getattr(shape, phase)
                assert all(evaluate(d, local) >= 0 for d in dims)
            if node.comm:
                assert math.isfinite(evaluate(getattr(node.comm.bus_bytes, phase), env))
                assert evaluate(getattr(node.comm.bus_bytes, phase), env) >= 0
                assert parse(node.comm.when)
    for field in (
        "weights",
        "activation_peak",
        "kv_per_token",
        "kv_tokens",
        "non_torch",
        "reserved",
    ):
        assert math.isfinite(evaluate(getattr(result.ir.memory, field), env))


def test_dense_catalog_ir(catalog, source, tmp_path):
    model = next(m for m in catalog.models if m.id == "qwen3-8b")
    snapshot = Snapshot.model_validate_json(
        (ROOT / "fixtures/qwen3-30b-a3b/snapshot.json").read_text()
    )
    projected = catalog_extract(model, snapshot, source)
    assert projected.fixture
    result = lower_catalog(model, snapshot, source)
    assert result.ir.views.family == "dense"
    assert coverage_gate(result.triage)
    assert evaluate(result.ir.memory.weights, environment(result.ir)) == 16381470720
    for tp in [1, 2, 4, 8]:
        env = environment(result.ir, tp=tp)
        expected = (
            2 * 151936 * 4096 / tp
            + 4096
            + 36
            * (
                4096 * (32 / tp + 2 * max(1, 8 / tp)) * 128
                + 4096 * 4096 / tp
                + 2 * 128
                + 3 * 4096 * 12288 / tp
                + 2 * 4096
            )
        ) * 2
        assert evaluate(result.ir.memory.weights, env) == expected
    nodes = {n.id: n for n in result.ir.nodes}
    assert nodes["mlp.down"].rule == "linear.row"
    assert nodes["mlp.gate_up"].rule == "linear.merged"
    assert nodes["mlp"].class_name.endswith(".Qwen2MLP")
    assert nodes["mlp.down.reduce"].comm.op == "all_reduce"
    result.write(tmp_path)
    Draft202012Validator(json.loads((ROOT / "schemas/ir.schema.json").read_text())).validate(
        json.loads((tmp_path / "ir.json").read_text())
    )


def test_kv_replication_capacity_budget_and_activation_profiling(data):
    r = lower(data)
    e = environment(
        r.ir,
        tp=8,
        hbm_bytes=80_000_000_000,
        max_num_batched_tokens=1024,
        max_num_seqs=2,
        non_torch_bytes=1000,
        reserved_bytes=2000,
    )
    per_token = 2 * 48 * 1 * 128 * 2
    assert evaluate(r.ir.memory.kv_per_token, e) == per_token
    weight = evaluate(r.ir.memory.weights, e)
    peak = evaluate(r.ir.memory.activation_peak, e)
    expected = (
        math.floor(max(0, e["hbm_bytes"] * 0.9 - weight - peak - 3000) / (per_token * 128)) * 128
    )
    assert evaluate(r.ir.memory.kv_tokens, e) == expected
    assert evaluate(r.ir.memory.kv_tokens, e | {"hbm_bytes": 1}) == 0
    assert evaluate(r.ir.memory.activation_peak, e | {"context": 8192}) == peak
    assert evaluate(r.ir.memory.activation_peak, e | {"max_num_batched_tokens": 2048}) > peak
    assert (
        evaluate(r.ir.memory.activation_peak, e | {"max_num_seqs": 100000}) == 100000 * 151936 * 4
    )
    assert {"graph capture workspace", "allocator fragmentation"} <= set(r.ir.memory.unmodelled)


def test_lm_head_uses_batch_not_token_count_and_tied_storage(data):
    plain = lower(data)
    data["config"]["tie_word_embeddings"] = True
    tied = lower(data)
    env = environment(tied.ir, tp=4, T=4096, batch=2, S=3)
    nodes = {n.id: n for n in tied.ir.nodes}
    assert evaluate(nodes["lm_head"].weight_bytes, env) == 0
    assert (
        evaluate(plain.ir.memory.weights, env) - evaluate(tied.ir.memory.weights, env)
        == 151936 / 4 * 2048 * 2
    )
    assert evaluate(nodes["lm_head"].flops.prefill, env) == 2 * 2 * 2048 * 151936 / 4
    assert evaluate(nodes["lm_head"].flops.decode, env) == 2 * 3 * 2048 * 151936 / 4
    assert nodes["lm_head"].out[0].shape.prefill[0] == "batch"
    assert nodes["lm_head"].inputs[0].shape.decode == ["S", "H"]


@pytest.mark.parametrize("tp,dp,ep", [(1, 1, 1), (4, 1, 1), (4, 1, 4), (1, 4, 4), (2, 2, 4)])
def test_communication_and_moe_flops(data, tp, dp, ep):
    r = lower(data)
    nodes = {n.id: n for n in r.ir.nodes}
    e = environment(r.ir, tp=tp, dp=dp, ep=ep, T=256, S=3, batch=2)
    for phase, tokens in [("prefill", 256), ("decode", 3)]:
        logical = tokens * 2048 * 2
        for name in ["attn.o.reduce", "embed.reduce", "moe.experts.reduce"]:
            comm = nodes[name].comm
            assert evaluate(getattr(comm.logical_bytes, phase), e) == logical
            assert evaluate(getattr(comm.bus_bytes, phase), e) == 2 * (tp - 1) / tp * logical
        for name in ["moe.experts.dispatch", "moe.experts.combine"]:
            comm = nodes[name].comm
            assert evaluate(getattr(comm.bus_bytes, phase), e) == (
                (ep - 1) / ep * tokens * 8 * 2048 * 2 if ep > 1 and dp > 1 else 0
            )
        rows = 2 if phase == "prefill" else 3
        assert (
            evaluate(getattr(nodes["logits.gather"].comm.logical_bytes, phase), e)
            == rows * 151936 / tp * 2
        )
        assert (
            evaluate(getattr(nodes["logits.gather"].comm.bus_bytes, phase), e)
            == (tp - 1) / tp * rows * 151936 / tp * 2
        )
        degree = ep if ep > 1 else tp * dp
        expected = 6 * tokens * dp * 8 * 2048 * 768 / degree + 4 * tokens * dp * 8 * 768 / degree
        assert evaluate(getattr(nodes["moe.experts.weights"].flops, phase), e) == expected
    assert all(nodes[n].comm is None for n in ["attn.o", "moe.experts", "embed", "logits"])
    assert any(e.kind == "kv" for e in r.ir.edges)
    assert {"moe.w13", "moe.act", "moe.w2"} <= nodes.keys()


def test_group_totals_and_pp_heaviest_stage(data):
    r = lower(data)
    e = environment(r.ir)
    nodes = {n.id: n for n in r.ir.nodes}
    endpoints = sum(evaluate(nodes[n].weight_bytes, e) for n in ["embed", "lm_head", "final_norm"])
    layer = evaluate(nodes["decoder"].weight_bytes, e) / 48
    assert evaluate(r.ir.memory.weights, e) == 48 * layer + endpoints
    for pp in [2, 4, 8]:
        env = e | {"pp": pp}
        expected = layer * 48 / pp + max(
            evaluate(nodes["embed"].weight_bytes, e),
            evaluate(nodes["lm_head"].weight_bytes, e)
            + evaluate(nodes["final_norm"].weight_bytes, e),
        )
        assert evaluate(r.ir.memory.weights, env) == expected


def test_unknown_ambiguity_invalid_global_binding_are_blocking(data):
    unknown = copy.deepcopy(data)
    unknown["modules"][0]["class"] = "private.NewModel"
    unknown["modules"].append(copy.deepcopy(unknown["modules"][0]))
    result = lower(unknown)
    assert not coverage_gate(result.triage)
    assert any(i.kind == "unknown_layer_class" for i in result.triage.items)
    rules = load_rules()
    base = next(r for r in rules if r.id == "linear.row")
    result = lower(data, [*rules, base.model_copy(update={"id": "other.row"})])
    assert any(i.kind == "conditional_override" for i in result.triage.items)
    invalid = copy.deepcopy(data)
    row = next(m for m in invalid["modules"] if m["path"].endswith("o_proj"))
    row["attrs"]["input_size_per_partition"] = 123
    result = lower(invalid)
    assert any(i.kind == "golden_diff" and "disagrees" in i.diff for i in result.triage.items)


def test_nonconstant_predicate_is_triaged(data):
    rules = load_rules()
    base = next(r for r in rules if r.id == "linear.row")
    conditional = base.model_copy(update={"match": base.match.model_copy(update={"when": "tp>1"})})
    r = lower(data, [conditional if r.id == base.id else r for r in rules])
    assert any(
        i.kind == "conditional_override" and i.conditions == ["tp>1"] for i in r.triage.items
    )


def test_alternate_logits_all_gather(data):
    next(m for m in data["modules"] if m["path"] == "logits_processor")["attrs"][
        "use_all_gather"
    ] = True
    r = lower(data)
    comm = next(n.comm for n in r.ir.nodes if n.id == "logits.gather")
    e = environment(r.ir, tp=4, batch=2)
    assert comm.op == "all_gather"
    assert evaluate(comm.bus_bytes.prefill, e) == 3 * evaluate(comm.logical_bytes.prefill, e)


def test_bands_stable_ids_and_layer_types(data):
    data["config"]["num_hidden_layers"] = 4
    data["layer_types"] = [
        "full_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    second = []
    for m in data["modules"]:
        if m["path"].startswith("model.layers.0"):
            m = copy.deepcopy(m)
            m["path"] = m["path"].replace("model.layers.0", "model.layers.2", 1)
            second.append(m)
    data["modules"] += second
    r = lower(data)
    assert coverage_gate(r.triage)
    assert [(g.id, g.layers) for g in r.ir.groups] == [
        ("band_0_1", ("0", "1")),
        ("band_2_3", ("2", "3")),
    ]
    assert {"band_0_1.attn.qkv", "band_2_3.attn.qkv"} <= {n.id for n in r.ir.nodes}
    assert r.ir.params["pp"].choices == ["1"]
    assert all(n.parent is None or n.parent in {x.id for x in r.ir.nodes} for n in r.ir.nodes)


def test_incomplete_production_extract_blocks(data):
    data["fixture"] = False
    r = lower(data)
    assert not coverage_gate(r.triage)
    assert any(
        "missing configured decoder" in i.diff for i in r.triage.items if i.kind == "golden_diff"
    )


def test_cached_real_extracts():
    directories = [
        ROOT.parents[1] / ".cache/extract",
        Path("/home/cyberspace/projects/InferAtlas/.cache/extract"),
    ]
    if os.environ.get("INFERATLAS_LOWER_EXTRACTS"):
        directories.append(Path(os.environ["INFERATLAS_LOWER_EXTRACTS"]))
    files = sorted({p for d in directories if d.exists() for p in d.rglob("*.json")})
    checked = 0
    for file in files:
        data = json.loads(file.read_text())
        if (
            not isinstance(data, dict)
            or not {"snapshot", "modules", "config", "layer_types"} <= data.keys()
        ):
            continue
        if data["config"].get("model_type") not in {"qwen3", "qwen3_moe"}:
            continue
        extract = Extract.model_validate(data)
        r = lower(extract)
        assert coverage_gate(r.triage), (file, r.triage.items)
        env = environment(r.ir)
        actual = 0
        for module in extract.modules:
            for param in module.params:
                assert [
                    evaluate(x, env) for x in r.parameter_shapes[module.path][param.name]
                ] == param.shape, (file, module.path, param.name)
                actual += math.prod(param.shape) * 2
        assert evaluate(r.ir.memory.weights, env) == actual, (file, actual)
        checked += 1
    if not checked:
        pytest.skip(
            "T1 real extract cache not present; set INFERATLAS_LOWER_EXTRACTS for CPU/meta extracts"
        )


@pytest.mark.parametrize(
    "mutation", ["missing_weight", "bad_bias", "bad_bool", "bad_kv_heads", "unknown_ascend"]
)
def test_invalid_facts_never_publish(data, mutation):
    row = next(m for m in data["modules"] if m["path"].endswith("o_proj"))
    if mutation == "missing_weight":
        row["params"] = []
    elif mutation == "bad_bias":
        row["params"].append({"name": "bias", "shape": [1], "dtype": "bfloat16"})
    elif mutation == "bad_bool":
        row["attrs"]["reduce_results"] = "yes"
    elif mutation == "bad_kv_heads":
        qkv = next(m for m in data["modules"] if m["path"].endswith("qkv_proj"))
        qkv["attrs"]["total_num_kv_heads"] = 8
    elif mutation == "unknown_ascend":
        row["ascend"]["class"] = "private.UnknownAscendLinear"
    result = lower(data)
    assert not coverage_gate(result.triage)
    # Even failed lowering returns wire-valid, reviewable IR alongside its triage.
    Draft202012Validator(json.loads((ROOT / "schemas/ir.schema.json").read_text())).validate(
        result.ir.model_dump(by_alias=True, mode="json")
    )


def test_unmatched_whole_version_still_returns_triage(data):
    result = lower(data, vllm_version="1.0")
    assert not coverage_gate(result.triage)
    assert not result.ir.nodes
    assert not result.ir.groups


def test_cli_writes_outputs_before_coverage_failure(data, tmp_path):
    import typer
    from typer.testing import CliRunner

    from inferatlas.lower.__main__ import main

    app = typer.Typer()
    app.command()(main)
    file = tmp_path / "extract.json"
    data["modules"][0]["class"] = "private.NewModel"
    file.write_text(json.dumps(data))
    output = tmp_path / "output"
    result = CliRunner().invoke(app, ["--extract", str(file), "--out", str(output)])
    assert result.exit_code == 1, result.output
    assert json.loads((output / "triage.json").read_text())["items"]
    assert (output / "ir.json").exists()


def test_missing_descendant_and_changed_collective_block_compression(data):
    data["fixture"] = False
    data["config"]["num_hidden_layers"] = 2
    data["layer_types"] = ["full_attention"] * 2
    second = []
    for m in data["modules"]:
        if m["path"].startswith("model.layers.0"):
            m = copy.deepcopy(m)
            m["path"] = m["path"].replace("model.layers.0", "model.layers.1", 1)
            second.append(m)
    data["modules"] += second
    assert coverage_gate(lower(data).triage)
    row = next(m for m in data["modules"] if m["path"] == "model.layers.1.self_attn.o_proj")
    row["attrs"]["reduce_results"] = False
    assert not coverage_gate(lower(data).triage)
    data["modules"] = [m for m in data["modules"] if m["path"] != "model.layers.1.self_attn.attn"]
    assert not coverage_gate(lower(data).triage)
