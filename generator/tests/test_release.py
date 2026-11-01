"""Failure triage and the deterministic single → manifest → hub/release interface."""

import subprocess
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError
from test_cli import FIXTURE, invoke, site_build
from test_cli import isolated_sources as isolated_sources

import inferatlas.cli as cli
from inferatlas.exports import ExportManifest
from inferatlas.extract import ExtractionError
from inferatlas.lower import coverage_gate
from inferatlas.schema import Index, Report, Snapshot, Triage
from inferatlas.verify import exit_code
from inferatlas.verify.io import ROOT, read_json, write_json


@pytest.mark.parametrize("stage", ["resolve", "extract"])
def test_failure_is_deterministic_blocking_triage_and_site_cell(tmp_path, monkeypatch, stage):
    snapshot = Snapshot.model_validate(read_json(FIXTURE / "snapshot.json"))
    monkeypatch.setattr(cli, "resolve_stage", lambda *args: snapshot)
    attempts = iter(["/tmp/inferatlas-12345", "/tmp/inferatlas-67890"])

    def fail(*args):
        message = (
            f"NPU rotary cache: {next(attempts)}/config.json 0xAb123 "
            + "2026-11-01T12:34:56.123Z\n"
            + "x" * 1200
        )
        if stage == "resolve":
            raise subprocess.CalledProcessError(1, message)
        raise ExtractionError(message, "NotImplementedError", structured=True)

    monkeypatch.setattr(cli, f"{stage}_stage", fail)
    directory = tmp_path / "build/pinned/qwen3-30b-a3b"
    directory.mkdir(parents=True)
    (directory / "ir.json").write_text("stale")
    before = None
    for _ in range(2):
        invoke("build", "--output", directory, code=2)
        triage = Triage.model_validate(read_json(directory / "triage.json"))
        assert (triage.snapshot is None) == (stage == "resolve")
        item = triage.items[0]
        assert item.kind == "extraction_failure"
        assert (item.stage, item.exception_type) == (
            stage,
            "CalledProcessError" if stage == "resolve" else "NotImplementedError",
        )
        assert (item.model, item.device, item.version) == (cli.DEFAULT_MODEL, "A2", "main-verified")
        assert len(item.message) == 1000 and item.message.endswith("...")
        assert (
            "/tmp/" not in item.message
            and "0xAb" not in item.message
            and "2026-" not in item.message
        )
        assert not coverage_gate(triage)
        assert (
            exit_code(
                Report(snapshot=snapshot, checks=[], golden_diff=[], source_link_failures=[]),
                triage,
            )
            == 2
        )
        Draft202012Validator(read_json(ROOT / "schemas/triage.schema.json")).validate(
            triage.model_dump(mode="json")
        )
        actual = {p.name: p.read_bytes() for p in directory.iterdir()}
        assert "ir.json" not in actual
        if before is not None:
            assert actual == before
        before = actual
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    invoke(
        "site",
        "--build-root",
        tmp_path / "build",
        "--data-dir",
        tmp_path / "data",
        "--include-blocked",
        code=2,
    )
    index = Index.model_validate(read_json(tmp_path / "data/index.json"))
    cell = next(
        c
        for c in index.cells
        if c.model == "qwen3-30b-a3b" and c.device == "A2" and c.version == "main-verified"
    )
    assert (cell.status, cell.triage_count, cell.data_path) == ("triage", 1, None)


def test_null_snapshot_only_allowed_for_resolve_failures():
    value = read_json(FIXTURE / "triage.json") | {"snapshot": None}
    with pytest.raises(ValidationError, match="resolve extraction_failure"):
        Triage.model_validate(value)


def manifest(directory, model, device, status, code, count=0, exists=False):
    filename = f"inferatlas-{model}-{device.lower()}.html"
    value = ExportManifest(
        version="main-verified",
        model=model,
        device=device,
        file=filename,
        status=status,
        triage_count=count,
        exit_code=code,
    )
    write_json(directory / ".inferatlas" / f"{filename}.json", value)
    if exists:
        (directory / filename).write_text("<!doctype html>single")
    return filename


def test_hub_real_matrix_links_and_deterministic_data(tmp_path, monkeypatch):
    directory = tmp_path / "exports"
    output = directory / "index.html"
    ok = manifest(directory, "qwen3-8b", "H100", "ok", 0, exists=True)
    blocked = manifest(directory, "qwen3-30b-a3b", "A2", "triage", 2, 3, exists=True)
    manifest(directory, "deepseek-v3", "A2", "triage", 2, 1)
    manifest(directory, "qwen3-8b", "A2", "ok", 0)  # deleted export
    (directory / "orphan.html").write_text("unknown status")
    captures = []

    def run(command, **kwargs):
        assert command == ["pnpm", "--filter", "web", "build:hub"]
        assert kwargs["cwd"] == ROOT and kwargs["check"] and kwargs["timeout"] == 300
        data = Path(kwargs["env"]["INFERATLAS_DATA_DIR"])
        generated = Path(kwargs["env"]["INFERATLAS_HUB_OUTPUT"])
        assert data.is_absolute() and generated.is_absolute()
        captures.append({p.name: p.read_bytes() for p in data.iterdir()})
        index = Index.model_validate(read_json(data / "index.json"))
        assert len(index.versions) == 1 and len(index.models) == 3 and len(index.cells) == 15
        assert all(c.data_path is None for c in index.cells)
        by_key = {(c.model, c.device): c for c in index.cells}
        assert by_key["qwen3-8b", "H100"].status == "ok"
        assert by_key["qwen3-8b", "A2"].status == "missing"
        assert by_key["deepseek-v3", "A2"].triage_count == 1
        assert by_key["deepseek-v3", "A2"].status == "triage"
        assert read_json(data / "hub.json")["exports"] == [
            dict(
                version="main-verified",
                model="qwen3-30b-a3b",
                device="A2",
                file=blocked,
            ),
            dict(version="main-verified", model="qwen3-8b", device="H100", file=ok),
        ]
        assert [p["id"] for p in read_json(data / "hardware.json")] == sorted(
            p.id for p in cli.catalog().hardware_presets
        )
        generated.write_text("<!doctype html>hub")

    monkeypatch.setattr(cli.subprocess, "run", run)
    for _ in range(2):
        invoke("hub", "--dir", directory, "--output", output, code=3)
        assert output.read_text() == "<!doctype html>hub"
    assert captures[0] == captures[1]


@pytest.mark.parametrize("failure", ["missing_output", "process"])
def test_hub_checks_web_output_and_preserves_old_hub(tmp_path, monkeypatch, failure):
    output = tmp_path / "index.html"
    output.write_text("previous")
    manifest(tmp_path, "qwen3-8b", "A2", "ok", 0, exists=True)

    def run(command, **kwargs):
        if failure == "process":
            raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("hub", "--dir", tmp_path, code=3)
    assert output.read_text() == "previous"


def test_single_without_bundle_records_triage_and_removes_stale_html(tmp_path, monkeypatch):
    directory = tmp_path / "build"
    write_json(directory / "triage.json", {"items": [{"kind": "extraction_failure"}]})
    output = tmp_path / "exports/custom.html"
    output.parent.mkdir()
    output.write_text("stale")
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 2))
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: pytest.fail("web ran"))
    invoke("single", "--output", output, "--include-blocked", code=2)
    assert not output.exists()
    assert read_json(output.parent / ".inferatlas/custom.html.json") == dict(
        version="main-verified",
        model="qwen3-30b-a3b",
        device="A2",
        file="custom.html",
        status="triage",
        triage_count=1,
        exit_code=2,
    )
    assert (output.parent / ".inferatlas/.gitignore").read_text() == "*\n"


def test_release_continues_all_cells_and_hub_with_worst_exit(tmp_path, monkeypatch):
    builds, web_calls = [], []

    def build(model, device, pair, output, tp, ep):
        builds.append((model, device))
        directory = tmp_path / "build" / model / device
        if (model, device) == ("qwen3-30b-a3b", "A2"):
            raise AssertionError("broken lower stage")
        blocked = model == "deepseek-v3"
        write_json(
            directory / "triage.json",
            {"items": [dict(kind="extraction_failure")] if blocked else []},
        )
        if not blocked or device == "H100":
            for name in ("ir", "view", "report"):
                write_json(directory / f"{name}.json", {})
        return directory, 2 if blocked else 0

    def run(command, **kwargs):
        web_calls.append(command[-1])
        env = kwargs["env"]
        if command[-1] == "build:hub":
            data = Path(env["INFERATLAS_DATA_DIR"])
            index = Index.model_validate(read_json(data / "index.json"))
            assert len(read_json(data / "hub.json")["exports"]) == 4
            by_key = {(c.model, c.device): c for c in index.cells}
            assert by_key["qwen3-30b-a3b", "A2"].status == "missing"
            assert by_key["deepseek-v3", "A2"].status == "triage"
            Path(env["INFERATLAS_HUB_OUTPUT"]).write_text("hub")
        else:
            Path(env["INFERATLAS_SINGLE_OUTPUT"]).write_text("single")

    monkeypatch.setattr(cli, "build_pipeline", build)
    monkeypatch.setattr(cli.subprocess, "run", run)
    out = tmp_path / "dist"
    invoke("release", "--out", out, code=3)
    assert builds == [
        (m.id, device)
        for m in sorted(cli.catalog().models, key=lambda m: m.id)
        for device in ["A2", "H100"]
    ]
    assert web_calls == ["build:single"] * 4 + ["build:hub"]
    assert (out / "index.html").read_text() == "hub"
    assert read_json(out / ".inferatlas/inferatlas-qwen3-30b-a3b-a2.html.json")["exit_code"] == 3


@pytest.mark.parametrize(
    "args",
    [
        ["--models", "Qwen/Qwen3-8B", "qwen3-30b-a3b", "--devices", "H100", "A2"],
        ["--models=Qwen/Qwen3-8B", "qwen3-30b-a3b", "--devices=H100", "A2"],
        [
            "--models",
            "qwen3-30b-a3b",
            "--models",
            "Qwen/Qwen3-8B",
            "--devices",
            "H100",
            "--devices",
            "A2",
        ],
    ],
)
def test_release_accepts_list_spellings_and_hf_ids(tmp_path, monkeypatch, args):
    calls = []

    def single(**kwargs):
        calls.append((kwargs["model"], kwargs["device"]))
        assert kwargs["include_blocked"]
        raise cli.typer.Exit(2)

    monkeypatch.setattr(cli, "single", single)
    monkeypatch.setattr(cli, "hub_export", lambda *args: 0)
    invoke("release", *args, "--out", tmp_path, code=2)
    assert calls == [(m, d) for m in ["qwen3-30b-a3b", "qwen3-8b"] for d in ["A2", "H100"]]


@pytest.mark.parametrize("stage", ["resolve", "extract"])
@pytest.mark.parametrize(
    "error",
    [
        ValueError("identity/TP/config mismatch"),
        ValueError("catalog config hash mismatch"),
        ValueError("cached checkout has local changes"),
        TypeError("broken call"),
        AttributeError("broken attribute"),
        ExtractionError("worker did not start"),
        ExtractionError("bug in worker", "TypeError", structured=True),
        ExtractionError("invalid worker data", "ValidationError", structured=True),
    ],
)
def test_early_failures_preserve_previous_build_and_exit_three(tmp_path, monkeypatch, stage, error):
    snapshot = Snapshot.model_validate(read_json(FIXTURE / "snapshot.json"))
    monkeypatch.setattr(cli, "resolve_stage", lambda *args: snapshot)
    output = tmp_path / "good-build"
    write_json(output / "ir.json", {"previous": "good"})

    def fail(*args):
        raise error

    monkeypatch.setattr(cli, f"{stage}_stage", fail)
    invoke("build", "--output", output, code=3)
    assert list(output.iterdir()) == [output / "ir.json"]
    assert read_json(output / "ir.json") == {"previous": "good"}


def test_build_requires_explicit_fixture_permission(tmp_path, monkeypatch):
    monkeypatch.delenv("INFERATLAS_ALLOW_FIXTURES")
    output = tmp_path / "build"
    write_json(output / "ir.json", {"previous": "good"})
    assert "--allow-fixtures" in invoke("build", "--output", output, code=3).output
    assert read_json(output / "ir.json") == {"previous": "good"}


@pytest.mark.parametrize("stage", ["resolve", "extract"])
@pytest.mark.parametrize("invalid_schema", [False, True])
def test_real_stage_identity_checks_exit_three(tmp_path, monkeypatch, stage, invalid_schema):
    import inferatlas.extract
    import inferatlas.resolve

    if stage == "resolve":
        snapshot = read_json(FIXTURE / "snapshot.json")
        snapshot["device"] = "A3"
        monkeypatch.setattr(
            inferatlas.resolve, "resolve", lambda **kw: {} if invalid_schema else snapshot
        )
    else:
        value = read_json(FIXTURE / "extract.json")
        value["parallel"]["tp"] = 2
        monkeypatch.setattr(
            inferatlas.extract, "extract", lambda *args, **kw: {} if invalid_schema else value
        )
    result = invoke("build", "--output", tmp_path / "build", code=3)
    assert ("validation errors" if invalid_schema else "disagrees") in result.output
    assert not (tmp_path / "build").exists()


@pytest.mark.parametrize("failure_first", [False, True])
def test_site_prefers_resolved_build_in_both_orders(tmp_path, monkeypatch, failure_first):
    def fail(*args):
        raise subprocess.CalledProcessError(1, ["git", "fetch"])

    monkeypatch.setattr(cli, "resolve_stage", fail)
    unresolved = tmp_path / "build/main-verified-A2-unresolved/qwen3-30b-a3b"
    if failure_first:
        invoke("build", "--output", unresolved, code=2)
    site_build(tmp_path, 0)
    if not failure_first:
        invoke("build", "--output", unresolved, code=2)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kw: None)
    invoke("site", "--build-root", tmp_path / "build", "--data-dir", tmp_path / "data")
    index = Index.model_validate(read_json(tmp_path / "data/index.json"))
    assert (
        next(
            c
            for c in index.cells
            if c.model == "qwen3-30b-a3b" and c.device == "A2" and c.version == "main-verified"
        ).status
        == "ok"
    )


@pytest.mark.parametrize(
    "filename",
    [
        "bad name.html",
        "你好.html",
        "-bad.html",
        ".bad.html",
        "bad.htm",
        "bad.HTML",
        "a.html\n",
        "bad..name.html",
    ],
)
def test_export_filename_rejected_early(tmp_path, monkeypatch, filename):
    with pytest.raises(ValidationError):
        ExportManifest(
            version="main-verified",
            model="qwen3-8b",
            device="A2",
            file=filename,
            status="ok",
            triage_count=0,
            exit_code=0,
        )
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: pytest.fail("build ran"))
    result = invoke("single", "--output", tmp_path / filename, code=3)
    assert "filename must match ASCII" in result.output
    assert not (tmp_path / ".inferatlas").exists()


def test_invalid_device_does_not_mask_single_error(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: pytest.fail("build ran"))
    result = invoke("single", "--device", "bogus", "--output", tmp_path / "a.html", code=3)
    assert "unknown catalog device: bogus" in result.output
    assert not (tmp_path / ".inferatlas").exists()


@pytest.mark.parametrize("output", ["elsewhere/index.html", "hub.html"])
def test_hub_requires_index_beside_exports(tmp_path, monkeypatch, output):
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kw: pytest.fail("web ran"))
    assert (
        "index.html inside --dir"
        in invoke("hub", "--dir", tmp_path, "--output", tmp_path / output, code=3).output
    )


def test_empty_hub_fails_and_preserves_previous_output(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("old")
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kw: pytest.fail("web ran"))
    assert "zero exports" in invoke("hub", "--dir", tmp_path, code=3).output
    assert (tmp_path / "index.html").read_text() == "old"


def test_standalone_hub_skips_foreign_and_prefers_default(tmp_path, monkeypatch):
    default = manifest(tmp_path, "qwen3-8b", "A2", "ok", 0, exists=True)
    value = read_json(tmp_path / ".inferatlas" / f"{default}.json")
    for name in ["aaa.html", "zzz.html"]:
        write_json(tmp_path / ".inferatlas" / f"{name}.json", value | {"file": name})
        (tmp_path / name).write_text("custom")
    for field, foreign in [("model", "removed"), ("version", "removed-pair"), ("device", "bogus")]:
        write_json(tmp_path / ".inferatlas" / f"foreign-{field}.json", value | {field: foreign})
    (tmp_path / ".inferatlas/broken.json").write_text("{")
    # Valid other version pairs can be placed in their own matrix.
    other_pair = next(p.id for p in cli.catalog().version_pairs if p.id != value["version"])
    write_json(tmp_path / ".inferatlas/other.json", value | {"version": other_pair})

    def run(command, **kw):
        data = Path(kw["env"]["INFERATLAS_DATA_DIR"])
        exports = read_json(data / "hub.json")["exports"]
        assert len(exports) == 2 and all(e["file"] == default for e in exports)
        assert {e["version"] for e in exports} == {"main-verified", other_pair}
        Path(kw["env"]["INFERATLAS_HUB_OUTPUT"]).write_text("hub")

    monkeypatch.setattr(cli.subprocess, "run", run)
    result = invoke("hub", "--dir", tmp_path)
    assert "duplicate" in result.output and "Skipping" in result.output


def test_release_uses_only_manifests_written_this_run(tmp_path, monkeypatch):
    default = manifest(tmp_path, "qwen3-8b", "A2", "ok", 0, exists=True)
    value = read_json(tmp_path / ".inferatlas" / f"{default}.json")
    write_json(tmp_path / ".inferatlas/custom.html.json", value | {"file": "custom.html"})
    (tmp_path / "custom.html").write_text("custom")
    manifest(tmp_path, "deepseek-v3", "H100", "missing", 3)
    write_json(tmp_path / ".inferatlas/other-pair.json", value | {"version": "v0.32.0"})
    write_json(tmp_path / ".inferatlas/removed.json", value | {"model": "removed"})

    def single(**kw):
        assert not (tmp_path / ".inferatlas" / f"{default}.json").exists()
        manifest(tmp_path, "qwen3-8b", "A2", "ok", 0, exists=True)
        raise cli.typer.Exit(0)

    def run(command, **kw):
        data = Path(kw["env"]["INFERATLAS_DATA_DIR"])
        assert read_json(data / "hub.json")["exports"] == [
            dict(version="main-verified", model="qwen3-8b", device="A2", file=default)
        ]
        index = Index.model_validate(read_json(data / "index.json"))
        assert len(index.versions) == 1
        assert all(
            c.status == "missing" for c in index.cells if (c.model, c.device) != ("qwen3-8b", "A2")
        )
        Path(kw["env"]["INFERATLAS_HUB_OUTPUT"]).write_text("hub")

    monkeypatch.setattr(cli, "single", single)
    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("release", "--models", "qwen3-8b", "--devices", "A2", "--out", tmp_path)


def test_single_reserves_index_for_the_hub(tmp_path, monkeypatch):
    output = tmp_path / "index.html"
    output.write_text("previous hub")
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: pytest.fail("build ran"))
    assert "reserved for the hub" in invoke("single", "--output", output, code=3).output
    assert output.read_text() == "previous hub"
