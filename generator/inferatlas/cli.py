"""Typer orchestration with explicit, identity-preserving fixture fallbacks."""

import hashlib
import importlib
import os
import shutil
import subprocess
import tempfile
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from ruamel.yaml import YAML

from inferatlas import __version__
from inferatlas.schema import IR, Catalog, Eval, Extract, Index, Report, Snapshot, Triage, View
from inferatlas.schema.models import Check
from inferatlas.verify import Evaluator, exit_code, parse_vllm_logs, verify_ir
from inferatlas.verify.golden import check_golden, format_diff, model_slug, update_golden
from inferatlas.verify.io import ROOT, read_json, source_root, write_json

app = typer.Typer(
    no_args_is_help=True, help="Generate source-linked inference architecture atlases."
)
golden_app = typer.Typer(
    no_args_is_help=True, help="Update or check committed IR and numeric grids."
)
app.add_typer(golden_app, name="golden")
DEFAULT_MODEL = "Qwen/Qwen3-30B-A3B"
ModelOption = Annotated[str, typer.Option(help="Catalog model slug or HF repository id.")]
DeviceOption = Annotated[str, typer.Option(help="Hardware profile, e.g. A2.")]
PairOption = Annotated[str, typer.Option(help="Explicit catalog version-pair id.")]


def warning(message: str) -> None:
    typer.echo(f"WARNING: {message}", err=True)


def guarded(function):
    @wraps(function)
    def run(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except typer.Exit:
            raise
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            ImportError,
            RuntimeError,
            subprocess.SubprocessError,
        ) as e:
            typer.echo(f"ERROR: {e}", err=True)
            raise typer.Exit(3) from e

    return run


def optional(stage: str, name: str):
    module_name = f"inferatlas.{stage}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        module = None
    function = getattr(module, name, None)
    if not callable(function):
        warning(f"{module_name}.{name} unavailable; falling back to fixtures/.")
        return None
    return function


def catalog() -> Catalog:
    return Catalog.model_validate(YAML(typ="safe").load(ROOT / "catalog.yaml"))


def catalog_model(name: str):
    matches = [model for model in catalog().models if name in (model.id, model.hf_id)]
    if not matches:
        raise ValueError(f"unknown catalog model: {name}")
    return matches[0]


def fixture(name: str, filename: str) -> Path:
    path = ROOT / "fixtures" / catalog_model(name).id / filename
    if not path.is_file():
        raise ValueError(f"no fixture for {name}/{filename}; implement the owning stage first")
    return path


def resolve_stage(model: str, device: str, pair: str) -> Snapshot:
    selected = catalog_model(model)
    pairs = {p.id: p for p in catalog().version_pairs}
    if pair not in pairs:
        raise ValueError(f"unknown version pair: {pair}")
    if function := optional("resolve", "resolve_snapshot"):
        result = function(
            model=selected.hf_id,
            device=device,
            pair=pair,
            catalog_path=ROOT / "catalog.yaml",
            source_root=source_root(ROOT),
        )
        snapshot = Snapshot.model_validate(result)
        version = pairs[pair]
        if (
            snapshot.model.hf_id != selected.hf_id
            or snapshot.model.config_rev != selected.config_rev
            or snapshot.device != device
            or version.vllm not in (snapshot.vllm.tag, snapshot.vllm.commit)
            or version.ascend not in (snapshot.ascend.tag, snapshot.ascend.commit)
        ):
            raise ValueError("resolved snapshot disagrees with requested catalog identity")
        return snapshot
    path = ROOT / "fixtures" / selected.id / "snapshot.json"
    snapshot = (
        Snapshot.model_validate(read_json(path))
        if path.is_file()
        else IR.model_validate(read_json(fixture(model, "ir.json"))).snapshot
    )
    # Never label A2/main fixture facts as a different hardware or source revision.
    if pair != "main-verified" or device != snapshot.device:
        raise ValueError("fixture snapshot supports its pinned main-verified/device pair only")
    return snapshot


def extract_stage(snapshot: Snapshot, tp: int) -> Extract:
    if tp < 1:
        raise ValueError("tp must be positive")
    if function := optional("extract", "extract_snapshot"):
        python = os.environ.get("INFERATLAS_PY")
        value = function(
            snapshot, tp=tp, source_root=source_root(ROOT), python=Path(python) if python else None
        )
        result = Extract.model_validate(value)
    else:
        name = "extract.json" if tp == 1 else f"extract-tp{tp}.json"
        result = Extract.model_validate(read_json(fixture(snapshot.model.hf_id, name)))
    if result.snapshot != snapshot or result.parallel.tp != tp:
        raise ValueError("extract identity/TP disagrees with requested snapshot")
    return result


def lower_stage(extracts: list[Extract]) -> tuple[IR, Triage]:
    if not extracts or any(item.snapshot != extracts[0].snapshot for item in extracts):
        raise ValueError("lower requires extracts from one snapshot")
    if function := optional("lower", "lower_extracts"):
        ir, triage = function(extracts, rules_dir=ROOT / "rules")
    else:
        if not all(item.fixture for item in extracts):
            raise ValueError("fixture lowering requires fixture extracts")
        model = extracts[0].snapshot.model.hf_id
        for item in extracts:
            original = Extract.model_validate(
                read_json(
                    fixture(
                        model,
                        "extract.json"
                        if item.parallel.tp == 1
                        else f"extract-tp{item.parallel.tp}.json",
                    )
                )
            )
            if original != item:
                raise ValueError("fixture lowering cannot consume changed extraction facts")
        ir = read_json(fixture(model, "ir.json"))
        triage = read_json(fixture(model, "triage.json"))
    result, issues = IR.model_validate(ir), Triage.model_validate(triage)
    if result.snapshot != extracts[0].snapshot or issues.snapshot != result.snapshot:
        raise ValueError("lower outputs disagree with input snapshot")
    return result, issues


def annotate_stage(ir: IR, extracts: list[Extract], triage: Triage) -> tuple[IR, Triage]:
    if function := optional("annotations", "annotate_ir"):
        value, new_triage = function(
            ir,
            extracts=extracts,
            annotations_dir=ROOT / "annotations",
            source_root=source_root(ROOT),
        )
        result = IR.model_validate(value)
        additional = Triage.model_validate(new_triage)
        if result.snapshot != ir.snapshot or additional.snapshot != ir.snapshot:
            raise ValueError("annotation outputs disagree with IR snapshot")
        merged = {item.model_dump_json(): item for item in [*triage.items, *additional.items]}
        issues = sorted(
            merged.values(),
            key=lambda item: (
                item.kind,
                getattr(item, "symbol", getattr(item, "id", getattr(item, "model", ""))),
            ),
        )
        return result, Triage(snapshot=ir.snapshot, items=issues)
    original = IR.model_validate(read_json(fixture(ir.snapshot.model.hf_id, "ir.json")))
    if ir != original:
        raise ValueError("fixture annotation fallback requires the unchanged, already annotated IR")
    return ir, triage


def layout_stage(ir_path: Path, output: Path) -> View:
    ir = IR.model_validate(read_json(ir_path))
    executable = ROOT / "packages/layout/dist/cli.js"
    if executable.is_file():
        subprocess.run(
            [
                "pnpm",
                "--filter",
                "@inferatlas/layout",
                "exec",
                "inferatlas-layout",
                "--input",
                str(ir_path.resolve()),
                "--output",
                str(output.resolve()),
            ],
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        result = View.model_validate(read_json(output))
    else:
        warning("Layout CLI unavailable; falling back to fixtures/.")
        original = IR.model_validate(read_json(fixture(ir.snapshot.model.hf_id, "ir.json")))
        if ir != original:
            raise ValueError("fixture layout cannot represent changed IR")
        result = View.model_validate(read_json(fixture(ir.snapshot.model.hf_id, "view.json")))
    validate_view(ir, result)
    write_json(output, result)
    return result


def validate_view(ir: IR, view: View) -> None:
    """View has no snapshot field; its provenance is checked against the paired IR."""
    nodes = {node.id for node in ir.nodes}
    edges = {edge.id: edge for edge in ir.edges}
    for graph in view.views:
        if (
            any(node.ir_node not in nodes for node in graph.nodes)
            or set(graph.collapsed_of) - nodes
        ):
            raise ValueError("view references missing IR nodes")
        for edge in graph.edges:
            if any(key not in edges for key in edge.ir_edges):
                raise ValueError("view references missing IR edges")
            represented = [edges[key] for key in edge.ir_edges]
            if any(item.kind != edge.kind for item in represented):
                raise ValueError("view edge kind disagrees with IR")
            if edge.label is not None and not any(
                item.tensor.shape == edge.label for item in represented if item.tensor is not None
            ):
                raise ValueError("view edge label disagrees with IR")


def report_stage(
    ir: IR,
    extracts: list[Extract],
    triage: Triage | None,
    golden_dir: Path | None = None,
    calibration: Path | None = None,
) -> Report:
    src = source_root(ROOT)
    return verify_ir(
        ir,
        extracts=extracts,
        triage=triage,
        repos={name: src / name for name in ("vllm", "vllm-ascend", "pytorch")},
        golden_dir=golden_dir or ROOT / "golden" / model_slug(ir),
        calibration=read_json(calibration) if calibration else None,
        evaluator=Evaluator(ROOT, warning),
    )


def snapshot_key(snapshot: Snapshot, pair: str) -> str:
    config = hashlib.sha256(snapshot.model.config_rev.encode()).hexdigest()[:12]
    return (
        f"{pair}-{snapshot.device}-{snapshot.vllm.commit[:12]}-"
        f"{snapshot.ascend.commit[:12]}-{config}-g{snapshot.generator_version}"
    )


def build_pipeline(
    model: str, device: str, pair: str, output: Path | None, tp: list[int]
) -> tuple[Path, int]:
    if not tp:
        raise ValueError("build needs at least one TP point")
    snapshot = resolve_stage(model, device, pair)
    directory = (
        output or ROOT / ".cache/build" / snapshot_key(snapshot, pair) / catalog_model(model).id
    )
    write_json(directory / "snapshot.json", snapshot)
    extracts = [extract_stage(snapshot, degree) for degree in sorted(set(tp))]
    for extract in extracts:
        name = (
            "extract.json" if extract.parallel.tp == 1 else f"extract-tp{extract.parallel.tp}.json"
        )
        write_json(directory / name, extract)
    ir, triage = lower_stage(extracts)
    ir, triage = annotate_stage(ir, extracts, triage)
    write_json(directory / "ir.json", ir)
    write_json(directory / "triage.json", triage)
    report = report_stage(ir, extracts, triage)
    try:
        layout_stage(directory / "ir.json", directory / "view.json")
        report.checks.append(
            Check(id="layout", status="pass", message="View validates against IR.")
        )
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        warning(f"Layout failed: {error}")
        report.checks.append(Check(id="layout", status="fail", message="Layout generation failed."))
    report.checks.sort(key=lambda check: (check.id, check.node or ""))
    write_json(directory / "report.json", report)
    typer.echo(str(directory))
    return directory, exit_code(report, triage)


@app.command()
def version() -> None:
    """Print the generator version."""
    typer.echo(__version__)


@app.command()
@guarded
def resolve(
    model: ModelOption = DEFAULT_MODEL,
    device: DeviceOption = "A2",
    pair: PairOption = "main-verified",
    output: Path = Path("snapshot.json"),
) -> None:
    """Resolve a pinned model/device/version identity."""
    write_json(output, resolve_stage(model, device, pair))


@app.command()
@guarded
def extract(
    snapshot: Path = Path("snapshot.json"), tp: int = 1, output: Path = Path("extract.json")
) -> None:
    """Extract CPU/meta facts for one TP degree."""
    write_json(output, extract_stage(Snapshot.model_validate(read_json(snapshot)), tp))


@app.command()
@guarded
def lower(
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    output: Path = Path("ir.json"),
    triage_output: Path = Path("triage.json"),
) -> None:
    """Lower one or more extracts into symbolic IR and blocking triage."""
    ir, triage = lower_stage(
        [Extract.model_validate(read_json(path)) for path in extract or [Path("extract.json")]]
    )
    write_json(output, ir)
    write_json(triage_output, triage)
    raise typer.Exit(2 if triage.items else 0)


@app.command()
@guarded
def annotate(
    ir: Path = Path("ir.json"),
    triage: Path = Path("triage.json"),
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    output: Path = Path("ir.json"),
    triage_output: Path = Path("triage.json"),
) -> None:
    """Apply reviewed Ascend annotations and merge unresolved triage."""
    graph = IR.model_validate(read_json(ir))
    issues = Triage.model_validate(read_json(triage))
    if issues.snapshot != graph.snapshot:
        raise ValueError("triage snapshot disagrees with IR")
    graph, issues = annotate_stage(
        graph, [Extract.model_validate(read_json(path)) for path in extract or []], issues
    )
    write_json(output, graph)
    write_json(triage_output, issues)
    raise typer.Exit(2 if issues.items else 0)


@app.command()
@guarded
def verify(
    ir: Path = Path("ir.json"),
    triage: Path | None = None,
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    golden_dir: Path | None = None,
    calibration: Path | None = None,
    output: Path = Path("report.json"),
) -> None:
    """Write a schema-valid report; exit 2 for triage or 3 for failed verification."""
    issues = Triage.model_validate(read_json(triage)) if triage else None
    report = report_stage(
        IR.model_validate(read_json(ir)),
        [Extract.model_validate(read_json(path)) for path in extract or []],
        issues,
        golden_dir,
        calibration,
    )
    write_json(output, report)
    if report.golden_diff:
        typer.echo(format_diff(report.golden_diff))
    raise typer.Exit(exit_code(report, issues))


@golden_app.command("update")
@guarded
def golden_update(
    ir: Path = Path("ir.json"), directory: Path | None = None, grid: Path | None = None
) -> None:
    """Re-evaluate a grid before replacing committed golden files."""
    graph = IR.model_validate(read_json(ir))
    directory = directory or ROOT / "golden" / model_slug(graph)
    grid = grid or (
        directory / "eval.json"
        if (directory / "eval.json").is_file()
        else fixture(graph.snapshot.model.hf_id, "eval.json")
    )
    update_golden(directory, graph, Eval.model_validate(read_json(grid)), Evaluator(ROOT, warning))
    typer.echo("Golden updated.")


@golden_app.command("check")
@guarded
def golden_check(ir: Path = Path("ir.json"), directory: Path | None = None) -> None:
    """Compare IR and fresh evaluations, printing differences per node/field."""
    graph = IR.model_validate(read_json(ir))
    diffs = check_golden(
        directory or ROOT / "golden" / model_slug(graph), graph, Evaluator(ROOT, warning)
    )
    typer.echo(format_diff(diffs) if diffs else "Golden matches.")
    raise typer.Exit(3 if diffs else 0)


@app.command()
@guarded
def layout(ir: Path = Path("ir.json"), output: Path = Path("view.json")) -> None:
    """Run the pnpm layout CLI, with a fixture fallback during integration."""
    layout_stage(ir, output)


@app.command()
@guarded
def build(
    model: ModelOption = DEFAULT_MODEL,
    device: DeviceOption = "A2",
    pair: PairOption = "main-verified",
    output: Path | None = None,
    tp: Annotated[list[int] | None, typer.Option()] = None,
) -> None:
    """Run resolve, extract, lower, annotate, verify and layout."""
    _, code = build_pipeline(model, device, pair, output, tp or [1, 4])
    raise typer.Exit(code)


@app.command("parse-logs")
@guarded
def parse_logs(log: Path, output: Path = Path("calibration.json")) -> None:
    """Parse vLLM startup logs into unit-normalized calibration JSON."""
    write_json(output, parse_vllm_logs(log.read_text(encoding="utf-8")))


@app.command()
@guarded
def site(build_root: Path | None = None, data_dir: Path | None = None) -> None:
    """Copy build outputs, generate the version/model index and build the web app."""
    build_root = build_root or ROOT / ".cache/build"
    data_dir = data_dir or ROOT / "web/public/data"
    data = catalog()
    cells = {}
    code = 0
    for path in sorted(build_root.glob("*/*/ir.json")):
        directory = path.parent
        ir = IR.model_validate(read_json(path))
        report = Report.model_validate(read_json(directory / "report.json"))
        triage = Triage.model_validate(read_json(directory / "triage.json"))
        view = View.model_validate(read_json(directory / "view.json"))
        if not (ir.snapshot == report.snapshot == triage.snapshot):
            raise ValueError("build artifacts have inconsistent snapshot identities")
        validate_view(ir, view)
        model = catalog_model(ir.snapshot.model.hf_id)
        pairs = [
            p
            for p in data.version_pairs
            if (
                p.vllm in (ir.snapshot.vllm.tag, ir.snapshot.vllm.commit)
                and p.ascend in (ir.snapshot.ascend.tag, ir.snapshot.ascend.commit)
            )
        ]
        if len(pairs) != 1:
            raise ValueError("build snapshot does not identify a unique catalog pair")
        pair = pairs[0]
        key = (pair.id, model.id, ir.snapshot.device)
        if key in cells:
            raise ValueError("multiple builds occupy the same version/model/device cell")
        status = exit_code(report, triage)
        code = max(code, status)
        relative = Path(directory.parent.name) / model.id
        for name, value in (("ir", ir), ("view", view), ("report", report)):
            write_json(data_dir / relative / f"{name}.json", value)
        cells[key] = dict(
            version=pair.id,
            model=model.id,
            device=ir.snapshot.device,
            status="ok" if status == 0 else "triage",
            data_path=f"data/{relative.as_posix()}",
            triage_count=len(triage.items),
        )
    if not cells:
        raise ValueError("no build outputs found; run inferatlas build first")
    devices = sorted({device.id for device in data.devices} | {c["device"] for c in cells.values()})
    for pair in data.version_pairs:
        for model in data.models:
            for device in devices:
                cells.setdefault(
                    (pair.id, model.id, device),
                    dict(
                        version=pair.id,
                        model=model.id,
                        device=device,
                        status="missing",
                        data_path=None,
                        triage_count=0,
                    ),
                )
    index = Index(
        versions=[
            dict(id=p.id, label=p.id, vllm=p.vllm, ascend=p.ascend)
            for p in sorted(data.version_pairs, key=lambda p: p.id)
        ],
        models=[
            dict(id=m.id, hf_id=m.hf_id, label=m.hf_id.split("/")[-1], family=m.family)
            for m in sorted(data.models, key=lambda m: m.id)
        ],
        cells=[cells[k] for k in sorted(cells)],
    )
    write_json(data_dir / "index.json", index)
    subprocess.run(["pnpm", "--filter", "web", "build"], cwd=ROOT, check=True, timeout=300)
    raise typer.Exit(code)


@app.command()
@guarded
def single(
    model: ModelOption = DEFAULT_MODEL,
    device: DeviceOption = "A2",
    pair: PairOption = "main-verified",
    output: Path | None = None,
) -> None:
    """Build a model and ask the web single-file plugin to inline its artifacts."""
    directory, code = build_pipeline(model, device, pair, None, [1, 4])
    if code == 3:
        raise typer.Exit(code)
    output = output or ROOT / "dist" / f"inferatlas-{catalog_model(model).id}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "INFERATLAS_DATA_DIR": str(directory.resolve()), "INFERATLAS_SINGLE": "1"}
    # A fresh destination prevents stale HTML from masquerading as build success.
    with tempfile.TemporaryDirectory(prefix="inferatlas-single-") as temporary:
        generated = Path(temporary) / "index.html"
        env["INFERATLAS_SINGLE_OUTPUT"] = str(generated)
        subprocess.run(
            ["pnpm", "--filter", "web", "build:single"], cwd=ROOT, env=env, check=True, timeout=300
        )
        if not generated.is_file():
            raise ValueError("web build:single did not produce INFERATLAS_SINGLE_OUTPUT")
        shutil.copyfile(generated, output)
    typer.echo(str(output))
    raise typer.Exit(code)
