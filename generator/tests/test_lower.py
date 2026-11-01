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
SOURCE = Path(os.environ["INFERATLAS_SRC"]) if os.environ.get("INFERATLAS_SRC") else None


@pytest.fixture
def data():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(scope="module")
def catalog():
    return Catalog.model_validate(YAML(typ="safe").load(ROOT / "catalog.yaml"))


@pytest.fixture(scope="module")
def source():
    if SOURCE is None or not SOURCE.is_dir():
        pytest.skip("Set INFERATLAS_SRC to the directory containing pinned upstream clones")
    return SOURCE


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
        evaluate(r.ir.memory.activation_peak, e | {"max_num_seqs": 100000}) == 100000 * 151936 * 22
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


@pytest.mark.parametrize(
    "tp,dp,ep", [(1, 1, 1), (4, 1, 1), (4, 1, 4), (1, 4, 1), (1, 4, 4), (2, 2, 1), (2, 2, 4)]
)
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
                (tp * dp - 1) / (tp * dp) * logical if dp > 1 else 0
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
    assert not coverage_gate(r.triage)
    assert any(
        "Unsupported layer_types" in i.diff for i in r.triage.items if i.kind == "golden_diff"
    )
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
        ROOT / ".cache/extract",
        ROOT.parents[1] / ".cache/extract",
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
    assert len(result.triage.items) == 1
    item = result.triage.items[0]
    assert item.kind == "version_mismatch"
    assert item.vllm_version == "1.0"
    assert item.expected_ranges == [">=0.32,<0.33"]
    assert item.message == "No rules support vLLM 1.0"


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


@pytest.mark.parametrize("tp,dp,ep", [(1, 1, 1), (4, 1, 1), (4, 1, 4), (2, 2, 1), (2, 2, 4)])
def test_expert_traffic_is_transferred_to_children(data, tp, dp, ep):
    from inferatlas.lower.context import bind_context, substitute

    r = lower(data)
    nodes = {n.id: n for n in r.ir.nodes}
    rule = next(rule for rule in load_rules() if rule.id == "moe.weights")
    extract = Extract.model_validate(data)
    module = next(m for m in extract.modules if m.path.endswith("routed_experts"))
    bindings = bind_context(module, extract, rule)
    for phase, tokens in [("prefill", 256), ("decode", 1)]:
        env = environment(r.ir, tp=tp, dp=dp, ep=ep, T=256, S=1)
        children = [nodes["moe." + key] for key in ["w13", "act", "w2"]]
        total = sum(evaluate(getattr(n.bytes_moved, phase), env) for n in children)
        assert total == pytest.approx(
            evaluate(substitute(getattr(rule.bytes_moved, phase), bindings), env)
        )
        assert total == pytest.approx(
            evaluate(getattr(nodes["moe.experts.weights"].bytes_moved, phase), env)
        )
        touched = min(128 / (ep if ep > 1 else 1), tokens * dp * 8 / (ep if ep > 1 else 1))
        inter = 768 / (1 if ep > 1 else tp * dp)
        routed = tokens * dp * 8 / (ep if ep > 1 else 1)
        expected = [
            2 * touched * 2048 * inter * 2 + routed * (2048 + 2 * inter) * 2,
            routed * 3 * inter * 2,
            touched * 2048 * inter * 2 + routed * (inter + 2048) * 2,
        ]
        assert [evaluate(getattr(n.bytes_moved, phase), env) for n in children] == expected
    # A2 dense bf16 ridge point: 320 TFLOP/s / 1.6 TB/s = 200 FLOP/byte.
    env = environment(r.ir, tp=4, S=1, dp=1, ep=1)
    assert (
        0
        < evaluate(nodes["moe.w13"].flops.decode, env)
        / evaluate(nodes["moe.w13"].bytes_moved.decode, env)
        < 200
    )


@pytest.mark.parametrize("tp,dp,ep", [(4, 1, 4), (2, 2, 1), (2, 2, 4), (1, 4, 4)])
def test_moe_backend_collectives_are_exclusive(data, tp, dp, ep):
    for a2a in [False, True]:
        r = lower(data, moe_a2a=a2a)
        nodes = {n.id: n for n in r.ir.nodes}
        assert not any(n.endswith("_a2a") for n in nodes)
        for phase, tokens in [("prefill", 256), ("decode", 3)]:
            env = environment(r.ir, tp=tp, dp=dp, ep=ep, T=256, S=3)
            for name, op in [("dispatch", "all_gather"), ("combine", "reduce_scatter")]:
                comm = nodes["moe.experts." + name].comm
                assert comm.op == ("all_to_all" if a2a else op)
                assert comm.group == "ep"
                active = dp > 1 and (ep > 1 if a2a else True)
                logical = tokens * (8 if a2a else 1) * 2048 * 2
                degree = ep if a2a else tp * dp
                assert evaluate(getattr(comm.logical_bytes, phase), env) == logical
                assert evaluate(comm.when, env) == active
                assert evaluate(getattr(comm.bus_bytes, phase), env) == (
                    (degree - 1) / degree * logical if active else 0
                )


def test_missing_context_key_names_module_and_key(data):
    # Config symbols remain available but this module's expert binding cannot be resolved.
    del data["config"]["num_experts"]
    result = lower(data)
    assert not coverage_gate(result.triage)
    assert any(
        "model.layers.0.mlp.gate: missing key num_experts" in i.diff
        for i in result.triage.items
        if i.kind == "golden_diff"
    )


@pytest.mark.parametrize("active", [True, False])
def test_sliding_window_blocks_only_when_active(data, active):
    data["config"].update(sliding_window=4096, use_sliding_window=active)
    assert coverage_gate(lower(data).triage) is not active


@pytest.mark.parametrize(
    "tag,metadata", [("main", "0.1.dev1+gstale"), ("main", "0.32.0"), ("v0.31.0", "0.32.0")]
)
def test_unknown_or_stale_version_has_one_diagnostic(data, tag, metadata):
    data["snapshot"]["vllm"].update(commit="0" * 40, tag=tag)
    data["env"]["vllm"] = metadata
    result = lower(data)
    assert len(result.triage.items) == 1
    item = result.triage.items[0]
    assert item.kind == "version_mismatch"
    assert item.vllm_version == ("0.31.0" if tag == "v0.31.0" else "main")
    assert item.expected_ranges == [">=0.32,<0.33"]
    assert item.message
    assert not coverage_gate(result.triage)


def test_rope_hierarchy_and_dense_gather(catalog, source):
    model = next(m for m in catalog.models if m.id == "qwen3-8b")
    snapshot = Snapshot.model_validate_json(FIXTURE.with_name("snapshot.json").read_text())
    projected = catalog_extract(model, snapshot, source)
    result = lower(projected)
    nodes = {n.id: n for n in result.ir.nodes}
    assert nodes["attn.rope.apply"].parent == "attn.rope"
    assert "mlp.gate_up.gather" not in nodes
    env = environment(result.ir, tp=4, T=32, S=3)
    assert nodes["attn.rope"].kind == "group"
    assert evaluate(nodes["attn.rope"].bytes_moved.decode, env) == evaluate(
        nodes["attn.rope.apply"].bytes_moved.decode, env
    )
    assert evaluate(nodes["mlp.act"].bytes_moved.prefill, env) == 32 * 3 * 12288 / 4 * 2
    assert (
        evaluate(nodes["attn.rope.apply"].bytes_moved.decode, env)
        == 2 * 3 * (32 / 4 + 8 / 4) * 128 * 2
    )
    next(m for m in projected.modules if m.path.endswith("gate_up_proj")).attrs["gather_output"] = (
        True
    )
    real = lower(projected)
    assert any(e.from_node == "mlp.gate_up.gather" and e.to == "mlp.act" for e in real.ir.edges)


def test_moe_activation_peak_tracks_tp_and_routed_hidden_workspace(data):
    result = lower(data)
    env = environment(result.ir, tp=4, dp=2, ep=1, max_num_batched_tokens=1024, max_num_seqs=1)
    expected = (
        1024
        * (
            4 * 2048
            + (32 / 4 + 2 * max(1, 4 / 4)) * 128
            + 2 * 8 * (2 * 2048 + 3 * 768 / (4 * 2))
            + 2 * (2048 + 128)
        )
        * 2
    )
    assert evaluate(result.ir.memory.activation_peak, env) == expected
    assert evaluate(result.ir.memory.activation_peak, env | {"tp": 8}) < expected


def test_deepseek_v3_extract_blocks_coverage():
    directories = [ROOT / ".cache/extract", ROOT.parents[1] / ".cache/extract"]
    if os.environ.get("INFERATLAS_LOWER_EXTRACTS"):
        directories.insert(0, Path(os.environ["INFERATLAS_LOWER_EXTRACTS"]))
    path = next((p for d in directories for p in sorted(d.glob("deepseek-v3/*.json"))), None)
    if path is None:
        pytest.skip(
            "Set INFERATLAS_LOWER_EXTRACTS to a cache containing deepseek-v3 CPU/meta extracts"
        )
    result = lower(path)
    assert not coverage_gate(result.triage)
    assert any(
        i.kind == "unknown_layer_class" and "Deepseek" in i.symbol for i in result.triage.items
    )


@pytest.mark.parametrize("a2a", [False, True])
def test_cli_selects_moe_backend(tmp_path, a2a):
    import typer
    from typer.testing import CliRunner

    from inferatlas.lower.__main__ import main

    app = typer.Typer()
    app.command()(main)
    args = ["--extract", str(FIXTURE), "--out", str(tmp_path)]
    if a2a:
        args.append("--moe-a2a")
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    ir = json.loads((tmp_path / "ir.json").read_text())
    dispatch = next(n for n in ir["nodes"] if n["id"] == "moe.experts.dispatch")
    assert dispatch["comm"]["op"] == ("all_to_all" if a2a else "all_gather")
