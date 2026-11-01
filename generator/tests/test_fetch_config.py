"""Offline checks for config-only provenance and reproducible catalog updates."""

import hashlib
import importlib.util
import json
import re
from io import BytesIO
from urllib.error import HTTPError

import pytest
from ruamel.yaml import YAML

from inferatlas.resolve import ROOT, load_catalog

SPEC = importlib.util.spec_from_file_location("fetch_config", ROOT / "scripts/fetch-config.py")
fetch_config = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_config)


@pytest.mark.parametrize(
    "model",
    [m for m in load_catalog().models if m.id not in ("qwen3-8b", "qwen3-30b-a3b")],
    ids=lambda m: m.id,
)
def test_pinned_download_matches_catalog_and_raw_hash(model):
    directory = ROOT / "fixtures" / model.id
    raw = (directory / "config.json").read_bytes()
    source = json.loads((directory / "config.source.json").read_text())
    assert json.loads(raw) == model.config
    assert source["hf_id"] == model.hf_id
    assert re.fullmatch(r"[0-9a-f]{40}", source["revision"])
    assert source["sha256"] == hashlib.sha256(raw).hexdigest()
    assert model.config_rev == "inline:sha256:" + source["canonical_sha256"]
    assert fetch_config.canonical_hash(model.config) == source["canonical_sha256"]
    assert source["source_url"].endswith(f"/{source['revision']}/config.json")
    catalog_text = (ROOT / "catalog.yaml").read_text()
    assert f"# source_url: {source['source_url']}" in catalog_text


def test_fetch_uses_mirror_then_hf_and_requires_immutable_revision(monkeypatch):
    calls = []
    raw = b'{"model_type":"example"}'

    def open_url(url, timeout):
        calls.append(url)
        assert timeout == 45
        if "hf-mirror" in url:
            raise HTTPError(url, 503, "unavailable", {}, None)
        response = BytesIO(raw)
        response.headers = {"x-repo-commit": "a" * 40}
        return response

    monkeypatch.setattr(fetch_config, "urlopen", open_url)
    content, revision, source = fetch_config.fetch("org/model", "main")
    assert content == raw and revision == "a" * 40
    assert calls == [f"{host}/org/model/resolve/main/config.json" for host in fetch_config.HOSTS]
    assert source == f"https://huggingface.co/org/model/resolve/{revision}/config.json"

    def mutable_url(url, timeout):
        response = BytesIO(raw)
        response.headers = {}
        response.url = url
        return response

    monkeypatch.setattr(fetch_config, "urlopen", mutable_url)
    with pytest.raises(RuntimeError, match="immutable commit"):
        fetch_config.fetch("org/model", "main")


def test_catalog_update_is_deterministic_and_comments_follow_sorted_models(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_config, "ROOT", tmp_path)
    path = tmp_path / "catalog.yaml"
    path.write_text("devices: []\nmodels: []\nversion_pairs: []\n")
    for slug in ("z-model", "a-model"):
        directory = tmp_path / "fixtures" / slug
        directory.mkdir(parents=True)
        (directory / "config.source.json").write_text(json.dumps({"hf_id": slug}))
        fetch_config.update_catalog(slug, slug, "dense", {"model_type": slug}, "a" * 64)
    before = path.read_bytes()
    fetch_config.update_catalog("z-model", "z-model", "dense", {"model_type": "z-model"}, "a" * 64)
    assert before == path.read_bytes()
    assert [m["id"] for m in YAML(typ="safe").load(path)["models"]] == ["a-model", "z-model"]
    assert path.read_text().count("# hf_id: a-model") == 1
    assert path.read_text().count("# hf_id: z-model") == 1


def test_required_model_families_are_pinned():
    assert {
        "deepseek-v3",
        "deepseek-v3-2",
        "deepseek-v4-flash",
        "kimi-k2",
        "minimax-m2",
        "minimax-m3",
        "glm-4-5",
        "glm-4-6",
        "gpt-oss-120b",
        "qwen3-next-80b-a3b",
        "llama-4-scout",
        "qwen3-235b-a22b",
        "mixtral-8x7b",
        "llama-3-1-8b",
    } <= {model.id for model in load_catalog().models}


def test_catalog_architectures_exist_in_pinned_registry():
    import ast

    from inferatlas.resolve import source_dir

    path = source_dir() / "vllm/vllm/model_executor/models/registry.py"
    if not path.is_file():
        pytest.skip("read-only pinned upstream registry unavailable")
    architectures = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name)
            and target.id in ("_TEXT_GENERATION_MODELS", "_MULTIMODAL_MODELS")
            for target in node.targets
        ):
            architectures.update(ast.literal_eval(node.value))
    for model in load_catalog().models:
        assert set(model.config["architectures"]) <= architectures
        assert set(model.config.get("text_config", {}).get("architectures", [])) <= architectures


@pytest.mark.parametrize("bad_body", [b"<html>bad gateway</html>", b"[]", b'{"error":"missing"}'])
def test_bad_mirror_body_falls_back(monkeypatch, bad_body):
    def open_url(url, timeout):
        response = BytesIO(bad_body if "hf-mirror" in url else b'{"model_type":"example"}')
        response.headers = {"x-repo-commit": "a" * 40}
        return response

    monkeypatch.setattr(fetch_config, "urlopen", open_url)
    assert fetch_config.fetch("org/model", "main")[0] == b'{"model_type":"example"}'
