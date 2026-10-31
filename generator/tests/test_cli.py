"""CliRunner pipeline tests; source/link arithmetic is tested independently."""

import subprocess
from pathlib import Path

import pytest
from typer.testing import CliRunner

import inferatlas.cli as cli
import inferatlas.verify as verification
from inferatlas.schema import IR, Extract, Index, Report, Snapshot, View
from inferatlas.verify.io import ROOT, read_json, write_json

FIXTURE = ROOT / "fixtures/qwen3-30b-a3b"
runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "source_root", lambda root: tmp_path / "sources")
    monkeypatch.setattr(verification, "check_sources", lambda *args: [])


def invoke(*args, code=0):
    result = runner.invoke(cli.app, [str(arg) for arg in args])
    assert result.exit_code == code, result.output or repr(result.exception)
    return result


def test_command_help_and_version():
    help_text = invoke("--help").output
    for name in (
        "resolve",
        "extract",
        "lower",
        "annotate",
        "verify",
        "layout",
        "build",
        "site",
        "single",
        "golden",
    ):
        assert name in help_text
    assert invoke("version").output.strip() == "0.1.0"


def test_individual_stages_and_exit_codes(tmp_path):
    snapshot, extract = tmp_path / "snapshot.json", tmp_path / "extract.json"
    graph, triage, report = (tmp_path / name for name in ("ir.json", "triage.json", "report.json"))
    assert "WARNING" in invoke("resolve", "--model", "qwen3-30b-a3b", "--output", snapshot).output
    Snapshot.model_validate(read_json(snapshot))
    invoke("extract", "--snapshot", snapshot, "--tp", "4", "--output", extract)
    assert Extract.model_validate(read_json(extract)).parallel.tp == 4
    invoke("lower", "--extract", extract, "--output", graph, "--triage-output", triage, code=2)
    invoke(
        "annotate",
        "--ir",
        graph,
        "--triage",
        triage,
        "--output",
        graph,
        "--triage-output",
        triage,
        code=2,
    )
    invoke("verify", "--ir", graph, "--triage", triage, "--output", report, code=2)
    assert len(Report.model_validate(read_json(report)).checks) >= 5
    issues = read_json(triage)
    issues["items"] = []
    write_json(triage, issues)
    invoke("verify", "--ir", graph, "--triage", triage, "--output", report)
    data = read_json(graph)
    data["nodes"][0]["label"] = "changed"
    write_json(graph, data)
    result = invoke("verify", "--ir", graph, "--triage", triage, "--output", report, code=3)
    assert "/ir/nodes/" in result.output and "/label" in result.output
    assert Report.model_validate(read_json(report)).golden_diff


def test_build_fixture_pipeline_is_complete_and_deterministic(tmp_path):
    result = invoke(
        "build",
        "--model",
        cli.DEFAULT_MODEL,
        "--device",
        "A2",
        "--pair",
        "main-verified",
        "--output",
        tmp_path,
        code=2,
    )
    assert "falling back to fixtures" in result.output
    expected = {
        "snapshot.json",
        "extract.json",
        "extract-tp4.json",
        "ir.json",
        "triage.json",
        "report.json",
        "view.json",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    first = {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    invoke("build", "--output", tmp_path, "--tp", "4", "--tp", "1", code=2)
    assert first == {path.name: path.read_bytes() for path in tmp_path.iterdir()}
    assert View.model_validate(read_json(tmp_path / "view.json")).views[1].id.startswith("L1:")


def test_build_records_layout_failure_in_report(tmp_path, monkeypatch):
    def fail(*args):
        raise ValueError("layout unavailable")

    monkeypatch.setattr(cli, "layout_stage", fail)
    invoke("build", "--output", tmp_path, code=3)
    report = Report.model_validate(read_json(tmp_path / "report.json"))
    assert next(check for check in report.checks if check.id == "layout").status == "fail"


@pytest.mark.parametrize(
    "arguments, message",
    [
        (["resolve", "--model", "unknown/model"], "unknown catalog model"),
        (["resolve", "--device", "A3"], "pinned main-verified/device"),
        (["resolve", "--pair", "v0.32.0"], "pinned main-verified/device"),
        (["resolve", "--pair", "unknown"], "unknown version pair"),
        (["build", "--tp", "8"], "no fixture"),
        (["verify", "--ir", "absent.json"], "ERROR"),
    ],
)
def test_failures_are_exit_three(tmp_path, arguments, message, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if arguments[0] == "build":
        arguments = [*arguments, "--output", str(tmp_path)]
    result = invoke(*arguments, code=3)
    assert message in result.output


def test_golden_update_and_check(tmp_path):
    invoke("golden", "update", "--ir", FIXTURE / "ir.json", "--directory", tmp_path)
    assert (
        "Golden matches"
        in invoke("golden", "check", "--ir", FIXTURE / "ir.json", "--directory", tmp_path).output
    )
    data = read_json(tmp_path / "ir.json")
    data["nodes"][0]["label"] = "old label"
    write_json(tmp_path / "ir.json", data)
    result = invoke("golden", "check", "--ir", FIXTURE / "ir.json", "--directory", tmp_path, code=3)
    assert "/label" in result.output and "old label" in result.output


def test_present_extract_hook_receives_contract_and_does_not_hide_errors(monkeypatch):
    import inferatlas.extract

    snapshot = Snapshot.model_validate(read_json(FIXTURE / "snapshot.json"))
    observed = []

    def extract_snapshot(value, **kwargs):
        observed.append((value, kwargs))
        return read_json(FIXTURE / "extract-tp4.json")

    monkeypatch.setattr(inferatlas.extract, "extract_snapshot", extract_snapshot, raising=False)
    result = cli.extract_stage(snapshot, 4)
    assert result.parallel.tp == 4
    assert observed[0][0] == snapshot and observed[0][1]["tp"] == 4

    def broken(*args, **kwargs):
        raise ValueError("real extraction failed")

    monkeypatch.setattr(inferatlas.extract, "extract_snapshot", broken)
    with pytest.raises(ValueError, match="real extraction failed"):
        cli.extract_stage(snapshot, 4)


def test_fixture_lowering_rejects_changed_facts():
    extract = Extract.model_validate(read_json(FIXTURE / "extract.json"))
    extract.modules[0].attrs["invented"] = 1
    with pytest.raises(ValueError, match="changed extraction facts"):
        cli.lower_stage([extract])


def test_layout_runs_pnpm_contract(tmp_path, monkeypatch):
    executable = tmp_path / "packages/layout/dist/cli.js"
    executable.parent.mkdir(parents=True)
    executable.touch()
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    graph = tmp_path / "ir.json"
    write_json(graph, read_json(FIXTURE / "ir.json"))
    output = tmp_path / "view.json"
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        write_json(output, read_json(FIXTURE / "view.json"))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("layout", "--ir", graph, "--output", output)
    assert commands == [
        [
            "pnpm",
            "--filter",
            "@inferatlas/layout",
            "exec",
            "inferatlas-layout",
            "--input",
            str(graph),
            "--output",
            str(output),
        ]
    ]


def test_site_copies_data_and_preserves_triage_status(tmp_path, monkeypatch):
    build_root = tmp_path / "build"
    directory = build_root / "pinned-snapshot/qwen3-30b-a3b"
    invoke("build", "--output", directory, code=2)
    commands = []
    monkeypatch.setattr(cli.subprocess, "run", lambda command, **kwargs: commands.append(command))
    data_dir = tmp_path / "public/data"
    invoke("site", "--build-root", build_root, "--data-dir", data_dir, code=2)
    index = Index.model_validate(read_json(data_dir / "index.json"))
    cell = next(cell for cell in index.cells if cell.status == "triage")
    assert cell.triage_count == 2
    assert (data_dir.parent / cell.data_path / "view.json").is_file()
    assert len(index.cells) == 3 * 2 * 4
    assert any(cell.status == "missing" and cell.data_path is None for cell in index.cells)
    assert commands == [["pnpm", "--filter", "web", "build"]]


def test_single_handoff_and_failed_build_preserve_existing_file(tmp_path, monkeypatch):
    directory = tmp_path / "model-build"
    directory.mkdir()
    monkeypatch.setattr(cli, "build_pipeline", lambda *args: (directory, 2))
    output = tmp_path / "atlas.html"
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(kwargs["env"]["INFERATLAS_SINGLE_OUTPUT"]).write_text("<!doctype html><p>Atlas</p>")

    monkeypatch.setattr(cli.subprocess, "run", run)
    invoke("single", "--output", output, code=2)
    assert output.read_text().startswith("<!doctype html>")
    assert calls[0][0] == ["pnpm", "--filter", "web", "build:single"]
    assert calls[0][1]["env"]["INFERATLAS_DATA_DIR"] == str(directory)
    monkeypatch.setattr(cli.subprocess, "run", lambda *args, **kwargs: None)
    invoke("single", "--output", output, code=3)
    assert output.read_text().startswith("<!doctype html>")


def test_parse_logs_command(tmp_path):
    log, output = tmp_path / "startup.log", tmp_path / "calibration.json"
    log.write_text("INFO Model loading took 10 GiB\n# GPU blocks: 100\n")
    invoke("parse-logs", log, "--output", output)
    assert read_json(output) == {"weight_bytes": 10 * 2**30, "gpu_blocks": 100}


def test_verify_missing_golden_still_writes_failure_report(tmp_path):
    report = tmp_path / "report.json"
    invoke(
        "verify",
        "--ir",
        FIXTURE / "ir.json",
        "--golden-dir",
        tmp_path / "missing",
        "--output",
        report,
        code=3,
    )
    result = Report.model_validate(read_json(report))
    assert any(c.id == "golden" and c.status == "fail" for c in result.checks)


def test_layout_fallback_rejects_changed_graph(tmp_path):
    graph = IR.model_validate(read_json(FIXTURE / "ir.json"))
    graph.nodes[0].label = "not the fixture"
    path = tmp_path / "ir.json"
    write_json(path, graph)
    result = invoke("layout", "--ir", path, "--output", tmp_path / "view.json", code=3)
    assert "cannot represent changed IR" in result.output
