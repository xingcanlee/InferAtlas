"""Real CPU/meta integration checks. Each extraction gets its own subprocess."""

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from t6_runtime import node_backend as node_backend

from inferatlas.extract import ExtractionError, extract
from inferatlas.resolve import ROOT, experiment_dir, load_catalog, resolve, source_dir

pytestmark = [
    pytest.mark.extract,
    pytest.mark.skipif(
        not Path(os.environ.get("INFERATLAS_PY") or experiment_dir() / ".venv/bin/python").is_file()
        or not (source_dir() / "vllm/vllm").is_dir()
        or not (source_dir() / "vllm-ascend/tests/ut/conftest.py").is_file(),
        reason="external CPU interpreter/upstream sources unavailable",
    ),
]


@pytest.fixture(scope="session")
def observed(tmp_path_factory):
    directory = tmp_path_factory.mktemp("extract")

    @lru_cache
    def run(model, tp=1, ep=False, device=None):
        name = f"tp{tp}" + ("-ep" if ep else "") + (f"-{device}" if device else "")
        snapshot = resolve(
            model, device=device or "H100", out_path=directory / model / "snapshot.json"
        )
        return extract(snapshot, model, tp, ep, device, directory / model / f"{name}.json")

    return run


def parameters(value):
    return {
        f"{module.path}.{param.name}": param.shape
        for module in value.modules
        for param in module.params
    }


@pytest.mark.parametrize("empty_annotations", [False, True])
def test_build_annotation_coverage_gates_verification(
    observed, tmp_path, monkeypatch, node_backend, empty_annotations
):
    from typer.testing import CliRunner

    import inferatlas.annotations
    import inferatlas.cli as cli

    # Replay real CPU facts through the production build stages.
    facts = observed("qwen3-8b", device="A2")
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES", raising=False)
    monkeypatch.setattr(cli, "extract_stage", lambda *args: facts)
    if empty_annotations:
        monkeypatch.setattr(inferatlas.annotations, "load_annotations", lambda path: [])
    result = CliRunner().invoke(
        cli.app, ["build", "--model", "qwen3-8b", "--tp", "1", "--output", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output or repr(result.exception)
    report = json.loads((tmp_path / "report.json").read_text())
    checks = {c["id"]: c for c in report["checks"]}
    if empty_annotations:
        assert "default_parameters" not in checks
        for name in ("weights.tp1", "golden", "evaluation", "calibration"):
            assert checks[name]["status"] == "skip"
            assert "blocked by triage" in checks[name]["message"].lower()
        assert (tmp_path / "view.json").is_file()
        assert checks["layout"]["status"] == "pass"
    else:
        assert checks["default_parameters"]["status"] == "pass"
        assert checks["layout"]["status"] == "pass"
        assert (tmp_path / "view.json").is_file()
    assert (tmp_path / "arch.json").is_file()
    assert report["numeric_verified"] is (not empty_annotations)
    assert not [d for d in report["golden_diff"] if not d["path"].startswith("/arch")]
    assert not any(
        c["status"] == "fail"
        for c in report["checks"]
        if c["id"] not in {"coverage", "arch_golden"}
    )
    triage = json.loads((tmp_path / "triage.json").read_text())
    assert bool(triage["items"]) == empty_annotations
    if empty_annotations:
        assert any(item.get("reason") == "unannotated_override" for item in triage["items"])


def param_count(value):
    return sum(math.prod(shape) for shape in parameters(value).values())


def assert_sources(value):
    refs = [
        ref
        for module in value.modules
        for ref in (module.src, module.ascend.src if module.ascend else None)
        if ref
    ]
    assert refs
    for ref in refs:
        root = source_dir() / ref.repo
        text = (root / ref.file).read_text().splitlines()[ref.line - 1].strip()
        symbol = ref.symbol.rsplit(".", 1)[-1]
        assert re.match(rf"(?:class|(?:async )?def) {re.escape(symbol)}\b", text), (ref, text)
        assert (
            ref.commit
            == (value.snapshot.vllm if ref.repo == "vllm" else value.snapshot.ascend).commit
        )


@pytest.mark.parametrize("model", ["qwen3-8b", "qwen3-30b-a3b"])
@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_qwen_matrix(observed, model, tp):
    value = observed(model, tp)
    assert value.fixture is False
    assert [m.path for m in value.modules] == sorted(m.path for m in value.modules)
    assert len(value.layer_types) == value.config["num_hidden_layers"]
    assert {p.dtype for m in value.modules for p in m.params} == {"bfloat16"}
    assert_sources(value)
    schema = json.loads((ROOT / "schemas/extract.schema.json").read_text())
    Draft202012Validator(schema).validate(value.model_dump(mode="json", by_alias=True))
    c = value.config
    qkv = next(m for m in value.modules if m.path == "model.layers.0.self_attn.qkv_proj")
    kv_heads = max(1, c["num_key_value_heads"] // tp)
    assert qkv.attrs["num_kv_heads"] == kv_heads
    assert qkv.attrs["output_partition_sizes"] == [
        c["num_attention_heads"] // tp * c["head_dim"],
        kv_heads * c["head_dim"],
        kv_heads * c["head_dim"],
    ]


def test_moe_count_and_tp4_shapes_match_verified_experiment(observed):
    assert param_count(observed("qwen3-30b-a3b")) == 30_532_122_624
    value = observed("qwen3-30b-a3b", 4)
    measured = json.loads(
        (ROOT / "experiments/a2_qwen3moe/out/exp1_vllm_meta_tp4.json").read_text()
    )
    layer0 = {
        name: shape
        for name, shape in parameters(value).items()
        if name.startswith("model.layers.0.")
    }
    assert layer0 == {name: p["shape"] for name, p in measured["build"]["layer0_params"].items()}


def test_dense_count_matches_config_closed_form(observed):
    value = observed("qwen3-8b")
    c = value.config
    h, d, q, kv = (
        c[k] for k in ("hidden_size", "head_dim", "num_attention_heads", "num_key_value_heads")
    )
    attn = h * (q + 2 * kv) * d + q * d * h + 2 * d
    mlp = 3 * h * c["intermediate_size"]
    endpoints = (
        math.ceil(c["vocab_size"] / 64) * 64 * h * (1 if c["tie_word_embeddings"] else 2) + h
    )
    expected = c["num_hidden_layers"] * (attn + mlp + 2 * h) + endpoints
    assert param_count(value) == expected


@pytest.mark.parametrize("model", ["qwen3-8b", "qwen3-30b-a3b"])
@pytest.mark.parametrize("tp", [1, 4])
def test_a2_overlay_preserves_parameters_and_baseline_sources(observed, model, tp):
    value, base = observed(model, tp, device="A2"), observed(model, tp)
    assert parameters(value) == parameters(base)
    assert {m.path: m.attrs for m in value.modules} == {m.path: m.attrs for m in base.modules}
    assert [(m.path, m.class_name, m.src) for m in value.modules] == [
        (m.path, m.class_name, m.src) for m in base.modules
    ]
    assert len(value.ascend.custom_ops) == 31
    assert len(value.ascend.patched) == 130
    assert len(value.ascend.profile["capabilities"]) == 26
    assert_sources(value)
    if model == "qwen3-30b-a3b":
        experiment = json.loads(
            (ROOT / "experiments/a2_qwen3moe/out/exp3_ascend_meta_A2.json").read_text()
        )
        assert {
            m.path: {"vllm": m.class_name, "ascend": m.ascend.class_name}
            for m in value.modules
            if m.ascend and m.path.startswith("model.layers.0.")
        } == experiment["compare"]["layer0_class_swaps"]
        fixture = json.loads((ROOT / "fixtures/qwen3-30b-a3b/extract.json").read_text())
        expected = dict(fixture["ascend"]["patched"])
        # exp2 had no CPU platform selected; this worker always selects CPU.
        expected["vllm.models.deepseek_v4.cpu.model.FusedMoEFactory"] = expected.pop(
            "vllm.models.deepseek_v4.nvidia.model.FusedMoEFactory"
        )
        assert value.ascend.patched == expected
        assert value.ascend.custom_ops == fixture["ascend"]["custom_ops"]


def test_ep_shards_experts_instead_of_intermediate_dimension(observed):
    value = observed("qwen3-30b-a3b", 4, True)
    params = parameters(value)
    prefix = "model.layers.0.mlp.experts.routed_experts."
    assert params[prefix + "w13_weight"] == [32, 1536, 2048]
    assert params[prefix + "w2_weight"] == [32, 2048, 768]
    module = next(m for m in value.modules if m.path == prefix[:-1])
    assert module.attrs["local_num_experts"] == 32
    assert module.attrs["intermediate_size_per_partition"] == 768
    assert module.attrs["use_ep"] is True
    assert value.parallel.ep is True


def test_ep_overlay_initializes_mc2_but_needs_an_npu_process_group(tmp_path):
    snapshot = resolve("qwen3-30b-a3b", out_path=tmp_path / "snapshot.json")
    output = tmp_path / "extract.json"
    output.write_text("previous output")
    with pytest.raises(
        ExtractionError, match="No backend type associated with device type npu"
    ) as exc:
        extract(snapshot, "qwen3-30b-a3b", 4, True, "A2", output)
    assert exc.value.exception_type == "RuntimeError"
    assert "mc2 group is not initialized" not in str(exc.value)
    assert output.read_text() == "previous output"


def test_deepseek_v3_mla_and_moe_construct_on_meta(observed):
    value = observed("deepseek-v3")
    params = parameters(value)
    assert len(value.layer_types) == 61
    assert params["model.layers.0.self_attn.fused_qkv_a_proj.weight"] == [2112, 7168]
    assert params["model.layers.0.self_attn.q_b_proj.weight"] == [24576, 1536]
    assert params["model.layers.0.self_attn.kv_b_proj.weight"] == [32768, 512]
    assert params["model.layers.3.mlp.experts.routed_experts.w13_weight"] == [256, 4096, 7168]
    assert_sources(value)


def test_real_worker_is_byte_deterministic(observed, tmp_path):
    value = observed("qwen3-30b-a3b", 4, device="A2")
    output = tmp_path / "second.json"
    extract(value.snapshot, "qwen3-30b-a3b", 4, False, "A2", output)
    first = tmp_path / "first.json"
    extract(value.snapshot, "qwen3-30b-a3b", 4, False, "A2", first)
    assert output.read_bytes() == first.read_bytes()


@pytest.mark.parametrize("model", [m.id for m in load_catalog().models])
@pytest.mark.parametrize("device", ["H100", "A2"])
def test_complete_catalog_structure(observed, model, device):
    value = observed(model, device=None if device == "H100" else device)
    assert value.snapshot.device == device
    assert value.modules and not value.fixture
    assert [module.path for module in value.modules] == sorted(
        module.path for module in value.modules
    )
    text = value.config.get("text_config", value.config)
    assert len(value.layer_types) == text["num_hidden_layers"]
    assert_sources(value)
    assert all(
        isinstance(dim, int) and dim > 0
        for module in value.modules
        for param in module.params
        for dim in param.shape
    )
    if value.config.get("quantization_config") or text.get("quantization_config"):
        assert any("logical BF16" in note for note in value.notes)
    if device == "A2":
        assert value.ascend.device == "A2"
        assert value.ascend.custom_ops
    if model == "deepseek-v3-2":
        assert "sparse_attention" in value.layer_types
        assert any("indexer" in module.path and module.params for module in value.modules)
    if model == "deepseek-v4-flash":
        assert "compressed_sparse_attention" in value.layer_types
        assert any("compressor" in module.path and module.params for module in value.modules)
    if model == "qwen3-next-80b-a3b":
        assert set(value.layer_types) == {"linear_attention", "full_attention"}
        assert any("linear_attn" in module.path and module.params for module in value.modules)
    if model == "minimax-m3":
        assert set(value.layer_types) == {"full_attention", "sparse_attention"}
        assert any("index_q_norm" in module.path and module.params for module in value.modules)


def test_constructor_device_redirects_are_scoped(tmp_path):
    import subprocess

    from inferatlas.extract import worker

    script = f"""
import sys
sys.path.insert(0, {str(Path(worker.__file__).parent)!r})
import torch
from worker import constructor_devices
with constructor_devices():
    for device in ('cuda', 'cuda:0', 'privateuseone', 'npu', 'npu:1', 'cpu'):
        assert torch.empty(2, 3, device=device).is_meta, device
    assert torch.ones(2, device='cuda').is_meta
    assert torch.ones(2).to('npu').is_meta
    assert torch.tensor([3], device='cpu').item() == 3
assert torch.empty(2, device='cpu').device.type == 'cpu'
"""
    python = Path(os.environ.get("INFERATLAS_PY") or experiment_dir() / ".venv/bin/python")
    result = subprocess.run([str(python), "-I", "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
