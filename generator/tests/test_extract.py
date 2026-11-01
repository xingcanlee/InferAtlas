"""Worker protocol and provenance checks that need no torch/vLLM environment."""

import importlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from inferatlas.extract import ExtractionError, worker
from inferatlas.resolve import ROOT, get_model
from inferatlas.schema import Extract

host = importlib.import_module("inferatlas.extract")
SAMPLE = Path(__file__).parent / "fixtures_t1/extract-qwen3-moe-layer0.json"


@pytest.fixture
def sample():
    return Extract.model_validate_json(SAMPLE.read_text())


def test_sample_is_sorted_and_contains_observed_shapes(sample):
    assert sample.fixture
    assert [m.path for m in sample.modules] == sorted(m.path for m in sample.modules)
    assert len(sample.ascend.custom_ops) == 31
    assert sum(m.ascend is not None for m in sample.modules) == 12
    assert all(m.params == sorted(m.params, key=lambda p: p.name) for m in sample.modules)


def mock_worker(monkeypatch, tmp_path, sample, response=None, returncode=0):
    monkeypatch.setenv("INFERATLAS_PY", sys.executable)
    monkeypatch.setattr(host, "checkout_sources", lambda snapshot: {"vllm": tmp_path})

    def run(command, **kwargs):
        assert command[1] == "-I"
        assert kwargs["env"]["VLLM_TARGET_DEVICE"] == "cpu"
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        assert kwargs["env"]["VLLM_PLUGINS"] == ""
        assert {key for key in kwargs["env"] if key.startswith("VLLM_")} == {
            "VLLM_TARGET_DEVICE",
            "VLLM_PLUGINS",
        }
        assert kwargs["env"]["PYTHONPATH"] == str(Path(worker.__file__).parent)
        request = json.loads(Path(command[-2]).read_text())
        assert json.loads(Path(request["model_dir"], "config.json").read_text()) == sample.config
        Path(command[-1]).write_text(json.dumps(response or sample.model_dump(by_alias=True)))
        return subprocess.CompletedProcess(command, returncode, "worker logs", "worker failure")

    monkeypatch.setattr(host.subprocess, "run", run)


def test_host_uses_isolated_worker_and_validates_before_publishing(monkeypatch, tmp_path, sample):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "1,2")
    monkeypatch.setenv("VLLM_USE_V1", "0")
    mock_worker(monkeypatch, tmp_path, sample)
    destination = tmp_path / "out.json"
    result = host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2", destination)
    assert result == sample
    assert Extract.model_validate_json(destination.read_text()) == result
    first = destination.read_bytes()
    host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2", destination)
    assert destination.read_bytes() == first


@pytest.mark.parametrize("mask", [0o022, 0o027, 0o077])
def test_published_output_permissions_follow_umask(monkeypatch, tmp_path, sample, mask):
    mock_worker(monkeypatch, tmp_path, sample)
    destination = tmp_path / "out.json"
    previous = os.umask(mask)
    try:
        for _ in range(2):
            host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2", destination)
            assert stat.S_IMODE(destination.stat().st_mode) == 0o666 & ~mask
    finally:
        os.umask(previous)


@pytest.mark.parametrize("failure", ["schema", "identity", "process", "timeout"])
def test_failure_preserves_previous_output(monkeypatch, tmp_path, sample, failure):
    response = sample.model_dump(by_alias=True)
    if failure == "schema":
        response["unknown"] = True
    elif failure == "identity":
        response["parallel"]["tp"] = 2
    mock_worker(monkeypatch, tmp_path, sample, response, int(failure == "process"))
    if failure == "timeout":

        def timeout(*args, **kwargs):
            raise subprocess.TimeoutExpired("worker", 300)

        monkeypatch.setattr(host.subprocess, "run", timeout)
    destination = tmp_path / "out.json"
    destination.write_text("previous valid output")
    with pytest.raises((ExtractionError, ValueError)):
        host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2", destination)
    assert destination.read_text() == "previous valid output"


def test_host_rejects_inconsistent_identity_and_device(sample):
    with pytest.raises(ValueError, match="identity"):
        host.extract(sample.snapshot, "qwen3-8b")
    with pytest.raises(ValueError, match="device"):
        host.extract(sample.snapshot, "qwen3-30b-a3b", device="A3")
    with pytest.raises(ValidationError):
        host.extract(sample.snapshot, "qwen3-30b-a3b", tp=0)


def test_worker_failure_preserves_original_exception_class(monkeypatch, tmp_path, sample):
    mock_worker(
        monkeypatch,
        tmp_path,
        sample,
        response={
            "error": {"exception_type": "NotImplementedError", "message": "rotary cache needs NPU"}
        },
        returncode=1,
    )
    with pytest.raises(ExtractionError, match="rotary cache needs NPU") as caught:
        host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2")
    assert caught.value.exception_type == "NotImplementedError"
    assert caught.value.structured


def test_source_ref_points_at_declaration_after_decorator(tmp_path):
    path = tmp_path / "decorated.py"
    path.write_text("def decorate(cls): return cls\n\n@decorate\nclass Example:\n    pass\n")
    spec = importlib.util.spec_from_file_location("inferatlas_test_decorated", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        request = {"sources": {"vllm": str(tmp_path)}, "snapshot": {"vllm": {"commit": "a" * 40}}}
        source = worker.source_ref(module.Example, request)
        assert source["line"] == 4
        assert source["file"] == "decorated.py"
        assert source["symbol"].endswith(".Example")
    finally:
        del sys.modules[spec.name]


def test_patch_scan_guards_lazy_proxies_and_mock_code(tmp_path):
    class Lazy:
        def __getattr__(self, name):
            raise RuntimeError("optional dependency unavailable")

    assert worker.origin(Lazy(), tmp_path) is None
    assert worker.origin(MagicMock(), tmp_path) is None
    assert (
        worker.origin(SimpleNamespace(__code__=SimpleNamespace(co_filename=123)), tmp_path) is None
    )
    namespace = {}
    filename = tmp_path / "vllm_ascend/patch/test.py"
    exec(compile("def replacement(): pass", str(filename), "exec"), namespace)
    assert worker.origin(namespace["replacement"], tmp_path) == "vllm_ascend/patch/test.py"


def test_attrs_reject_mock_contamination():
    with pytest.raises(TypeError, match="non-JSON"):
        worker.json_attr(MagicMock())
    assert worker.json_attr((1, [2], None)) == [1, [2], None]


@pytest.mark.parametrize("tag,expected", [("main", "0.33.dev1"), ("v0.32.0", "0.32.0")])
def test_ascend_source_only_version_uses_scm_or_explicit_release(monkeypatch, tag, expected):
    calls = []

    def version(**kwargs):
        calls.append(kwargs)
        return "0.33.dev1"

    monkeypatch.setitem(sys.modules, "setuptools_scm", SimpleNamespace(get_version=version))
    monkeypatch.setenv("VLLM_VERSION", "stale-generated-version")
    worker.ascend_version({"snapshot": {"vllm": {"tag": tag}}, "sources": {"vllm": "/source"}})
    assert os.environ["VLLM_VERSION"] == expected
    assert calls == (
        [dict(root="/source", local_scheme="no-local-version")] if tag == "main" else []
    )


def test_sample_matches_current_pinned_moe_config(sample):
    assert sample.config == get_model("qwen3-30b-a3b").config
    assert sample.snapshot.model.config_rev == get_model("qwen3-30b-a3b").config_rev
    assert (ROOT / "schemas/extract.schema.json").is_file()


@pytest.mark.parametrize("response", [None, "{", '{"error": {"message": 42}}'])
def test_unstructured_failure_warns_with_stderr_tail(monkeypatch, tmp_path, sample, response):
    monkeypatch.setenv("INFERATLAS_PY", sys.executable)
    monkeypatch.setattr(host, "checkout_sources", lambda snapshot: {"vllm": tmp_path})

    def run(command, **kw):
        if response is not None:
            Path(command[-1]).write_text(response)
        return subprocess.CompletedProcess(command, 1, "", "discard-me" + "x" * 16384)

    monkeypatch.setattr(host.subprocess, "run", run)
    with pytest.warns(UserWarning, match="last 16KB") as warnings:
        with pytest.raises(ExtractionError) as caught:
            host.extract(sample.snapshot, "qwen3-30b-a3b", 4, False, "A2")
    assert not caught.value.structured
    message = str(warnings[0].message)
    assert "discard-me" not in message
    assert message.endswith("x" * 16384)


def test_stderr_warning_is_bounded_in_bytes():
    with pytest.warns(UserWarning) as warnings:
        host.warn_stderr("discard-me" + "中" * 10000)
    tail = str(warnings[0].message).split("\n", 1)[1]
    assert "discard-me" not in tail
    assert len(tail.encode()) <= 16386  # A split UTF-8 codepoint becomes one replacement character.


def test_overlay_emits_actual_tree_when_ascend_changes_structure():
    def row(path, cls="Baseline", shape=8):
        return dict(
            path=path,
            **{"class": cls},
            src=None,
            attrs={},
            params=[dict(name="weight", shape=[shape], dtype="bfloat16")],
        )

    baseline = [row("removed"), row("same"), row("changed")]
    actual = [row("same", "Ascend"), row("added", "Ascend"), row("changed", "Ascend", 16)]
    notes = []
    merged = worker.merge_overlay(baseline, actual, notes)
    assert [r["path"] for r in merged] == ["added", "changed", "same"]
    assert merged[0] == actual[1]
    assert merged[1]["params"][0]["shape"] == [16]
    assert merged[1]["class"] == "Baseline"
    assert merged[1]["ascend"]["class"] == "Ascend"
    assert merged[1]["flags"] == ["ascend_params_changed"]
    assert merged[0]["flags"] == ["ascend_added"]
    assert merged[2]["class"] == "Baseline"
    assert merged[2]["ascend"]["class"] == "Ascend"
    assert notes == [
        "Ascend removed baseline module: removed",
        "Ascend added module: added; emitting actual Ascend structure.",
        "Ascend changed parameter shapes: changed; emitting actual Ascend structure.",
    ]


def test_layer_types_preserve_unknown_structure_and_declared_hybrid_policy():
    text = SimpleNamespace(
        num_hidden_layers=4, layer_types=["linear_attention"] * 3 + ["full_attention"]
    )
    config = SimpleNamespace(get_text_config=lambda: text)
    assert worker.decoder_layer_types(config, None) == text.layer_types
    text.layer_types = None
    model = SimpleNamespace(named_modules=lambda: [])
    assert worker.decoder_layer_types(config, model) == ["unknown"] * 4


def test_dynamic_class_source_links_to_declared_ancestor(tmp_path):
    path = tmp_path / "dynamic.py"
    path.write_text("class Declared: pass\nDynamic = type('Dynamic', (Declared,), {})\n")
    spec = importlib.util.spec_from_file_location("inferatlas_test_dynamic", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        request = {"sources": {"vllm": str(tmp_path)}, "snapshot": {"vllm": {"commit": "a" * 40}}}
        ref = worker.source_ref(module.Dynamic, request)
        assert ref["line"] == 1
        assert ref["symbol"].endswith(".Declared")
    finally:
        del sys.modules[spec.name]
