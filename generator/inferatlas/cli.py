"""Typer orchestration with explicit, identity-preserving fixture fallbacks."""

import hashlib
import importlib
import os
import shutil
import subprocess
import tempfile
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Annotated

import typer
from ruamel.yaml import YAML

from inferatlas import __version__
from inferatlas.expr import validate
from inferatlas.schema import IR, Catalog, Eval, Extract, Index, Report, Snapshot, Triage, View
from inferatlas.schema.models import Check, ConditionalOverride
from inferatlas.verify import Evaluator, exit_code, parse_vllm_logs, verify_ir
from inferatlas.verify.golden import (
    check_golden,
    default_grid,
    format_diff,
    golden_path,
    update_golden,
)
from inferatlas.verify.io import ROOT, publish_directory, read_json, source_root, write_json

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
AllowFixtures = Annotated[
    bool,
    typer.Option(
        "--allow-fixtures",
        envvar="INFERATLAS_ALLOW_FIXTURES",
        help="Allow fixture artifacts for development; reports remain unpublishable.",
    ),
]
_used_fixtures: ContextVar[set[str] | None] = ContextVar("used_fixtures", default=None)
_allow_fixtures: ContextVar[bool] = ContextVar("allow_fixtures", default=False)


def fixture_use(stage: str) -> None:
    if not (_allow_fixtures.get() or os.environ.get("INFERATLAS_ALLOW_FIXTURES") == "1"):
        raise RuntimeError(f"{stage} unavailable; use --allow-fixtures only for development")
    used = _used_fixtures.get()
    if used is not None:
        used.add(stage)
    warning(f"{stage} unavailable; falling back to fixtures/ (unpublishable).")


def inherit_provenance(*paths: Path) -> None:
    used = _used_fixtures.get()
    if used is not None:
        for path in paths:
            if path.resolve().is_relative_to(ROOT / "fixtures"):
                used.add("input")
            for sidecar in (path.with_suffix(".provenance.json"), path.parent / "provenance.json"):
                if sidecar.is_file():
                    used.update(read_json(sidecar)["fixtures"])


def save_artifact(path: Path, value) -> None:
    write_json(path, value)
    write_json(
        path.with_suffix(".provenance.json"), {"fixtures": sorted(_used_fixtures.get() or [])}
    )


PairOption = Annotated[str, typer.Option(help="Explicit catalog version-pair id.")]


def warning(message: str) -> None:
    typer.echo(f"WARNING: {message}", err=True)


def guarded(function):
    @wraps(function)
    def run(*args, **kwargs):
        used_token = _used_fixtures.set(set())
        allow_token = _allow_fixtures.set(kwargs.get("allow_fixtures", False))
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
        finally:
            _used_fixtures.reset(used_token)
            _allow_fixtures.reset(allow_token)

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
        fixture_use(stage)
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
    if function := optional("resolve", "resolve"):
        result = function(
            model=selected.hf_id,
            device=device,
            version=pair,
            catalog_path=ROOT / "catalog.yaml",
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


def extract_stage(snapshot: Snapshot, tp: int, ep: bool = False) -> Extract:
    if tp < 1:
        raise ValueError("tp must be positive")
    if function := optional("extract", "extract"):
        value = function(
            snapshot,
            snapshot.model.hf_id,
            tp=tp,
            ep=ep,
            device=None if snapshot.device == "H100" else snapshot.device,
        )
        result = Extract.model_validate(value)
    else:
        name = "extract.json" if tp == 1 else f"extract-tp{tp}.json"
        result = Extract.model_validate(read_json(fixture(snapshot.model.hf_id, name)))
    if result.snapshot != snapshot or result.parallel.tp != tp or result.parallel.ep != ep:
        raise ValueError("extract identity/TP disagrees with requested snapshot")
    return result


def lower_stage(extracts: list[Extract]) -> tuple[IR, Triage]:
    if not extracts or any(item.snapshot != extracts[0].snapshot for item in extracts):
        raise ValueError("lower requires extracts from one snapshot")
    if function := optional("lower", "lower"):
        ordered = sorted(extracts, key=lambda item: item.parallel.tp)
        if ordered[0].parallel.tp != 1:
            raise ValueError("lower needs the tp=1 extract as primary")
        results = [
            function(item, rules=ROOT / "rules", vllm_version=None, ascend_version=None, mro=None)
            for item in ordered
        ]
        if any(
            result.ir.snapshot != ordered[0].snapshot
            or result.triage.snapshot != ordered[0].snapshot
            for result in results
        ):
            raise ValueError("lower outputs disagree with input snapshot")
        ir, triage = results[0].ir, results[0].triage
        merged = {
            item.model_dump_json(): item for result in results for item in result.triage.items
        }
        triage = Triage(snapshot=ir.snapshot, items=[merged[key] for key in sorted(merged)])
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
    if any(item.fixture for item in extracts) and _used_fixtures.get() is not None:
        _used_fixtures.get().add("extract")
    result, issues = IR.model_validate(ir), Triage.model_validate(triage)
    if result.snapshot != extracts[0].snapshot or issues.snapshot != result.snapshot:
        raise ValueError("lower outputs disagree with input snapshot")
    return result, issues


def annotation_context(primary: Extract, fact) -> dict:
    """Bind facts plus the pinned worker defaults documented in annotations/README.md."""
    profile = primary.ascend.profile if primary.ascend else {}
    attrs = fact.attrs
    pinned_defaults = (primary.snapshot.vllm.commit, primary.snapshot.ascend.commit) == (
        "544f182d9f0c9bfe8bd7645525951d5e3692a542",
        "24a76e84394ae3427195e3715fc08e90769d398c",
    )
    prefix = attrs.get("prefix", fact.path)
    finegrained = profile.get("finegrained_tp_config", {})
    custom_parallel = (
        ("o_proj" in prefix and finegrained.get("oproj_tensor_parallel_size", 0) > 1)
        or (
            any(part in prefix for part in ("gate_up_proj", "down_proj"))
            and finegrained.get("mlp_tensor_parallel_size", 0) > 1
        )
    ) and not attrs.get("disable_tp", False)
    policy = profile.get("weight_layout_policy", "").rsplit(".", 1)[-1]
    context = {
        "is_a2": primary.snapshot.device == "A2",
        "is_qwen3_moe": primary.config.get("model_type") == "qwen3_moe"
        or "Qwen3MoeForCausalLM" in primary.config.get("architectures", []),
        "is_qwen3_dense": primary.config.get("model_type") == "qwen3"
        or "Qwen3ForCausalLM" in primary.config.get("architectures", []),
        "unquantized": not primary.config.get("quantization_config")
        and (not attrs.get("quant_method") or "Unquantized" in attrs["quant_method"]),
        "custom_parallel": custom_parallel,
        "vocab_parallel_override": attrs.get("parallel_mode", "STANDARD")
        in ("FINE_GRAINED", "PCP_X_TP"),
        "mtp_draft": False,
        "use_mrope": any(
            module.ascend and module.ascend.class_name.endswith(".AscendMRotaryEmbedding")
            for module in primary.modules
        ),
        "capture_routing": False,
        "redundant_experts": any(
            module.attrs.get("global_num_experts", 0) > primary.config.get("num_experts", 0)
            for module in primary.modules
        ),
        "fused_mc2": bool(profile.get("enable_fused_mc2", False)),
    }
    if policy in ("FORCE_NZ", "CONFIGURABLE"):
        context["linear_nz"] = (
            policy == "FORCE_NZ" or profile.get("weight_nz_mode", 1) == 2
        ) and not (
            attrs.get("skip_weight_nz_conversion", False)
            or "conv1d" in prefix
            or any(
                param.dtype == "float32" or 1 in param.shape[-2:]
                for param in fact.params
                if param.name == "weight"
            )
        )
    if not pinned_defaults:
        # These settings are not exported by the current extractor; never guess for new pairs.
        for name in (
            "custom_parallel",
            "vocab_parallel_override",
            "mtp_draft",
            "capture_routing",
            "fused_mc2",
        ):
            context.pop(name)
        if policy != "FORCE_NZ":
            context.pop("linear_nz", None)
    return context


def annotate_stage(ir: IR, extracts: list[Extract], triage: Triage) -> tuple[IR, Triage]:
    if ir.snapshot.device == "H100":
        return ir, triage
    if load := optional("annotations", "load_annotations"):
        if not extracts:
            raise ValueError("Ascend annotations require extraction facts; supply --extract")
        module = importlib.import_module("inferatlas.annotations")
        src = source_root(ROOT)
        sources = {
            "vllm": module.SourceTree(src / "vllm", "vllm", ir.snapshot.vllm.commit),
            "vllm_ascend": module.SourceTree(
                src / "vllm-ascend", "vllm-ascend", ir.snapshot.ascend.commit
            ),
        }
        resolved, stale = module.check_annotations(load(ROOT / "annotations"), sources)
        evaluator = Evaluator(ROOT, warning)
        env = evaluator.bind(ir, {})
        active = set()
        evaluated = {}
        pending = list(stale)
        bindings = {}
        contexts_by_node = {}
        for node in ir.nodes:
            observed = set()
            facts = [
                (extract, fact)
                for extract in sorted(extracts, key=lambda item: item.parallel.tp)
                for fact in extract.modules
                if fact.class_name == node.class_name
            ]
            # The same Linear class implements both attention and MLP projections.
            for label, suffix in (
                ("attn.o", "o_proj"),
                ("mlp.gate_up", "gate_up_proj"),
                ("mlp.down", "down_proj"),
            ):
                if f".{label}." in f".{node.id}.":
                    facts = [pair for pair in facts if pair[1].path.endswith("." + suffix)]
            for _, module_fact in facts:
                if module_fact.ascend:
                    observed.add(module_fact.ascend.class_name)
            contexts = [annotation_context(extract, fact) for extract, fact in facts]
            context_agrees = (
                bool(contexts)
                and all(ctx == contexts[0] for ctx in contexts)
                and all(any(source is extract for source, _ in facts) for extract in extracts)
            )
            swaps = observed.copy()
            for extract in extracts:
                if extract.ascend:
                    for symbol, file in extract.ascend.patched.items():
                        if symbol == node.class_name or symbol.startswith(node.class_name + "."):
                            observed.update(
                                ref.symbol
                                for item in resolved
                                for ref in item.src
                                if ref.repo == "vllm-ascend" and ref.file == file
                            )
            candidates = []
            for item in resolved:
                ann = item.annotation
                if ann.resolves_to not in observed:
                    continue
                outcomes = set()
                for fixed_context in contexts:
                    context = {**env, **fixed_context}
                    if set(validate(ann.applies_when)) <= context.keys():
                        key = (ann.applies_when, tuple(sorted(context.items())))
                        if key not in evaluated:
                            evaluated[key] = bool(evaluator.many([ann.applies_when], context)[0])
                        outcomes.add(evaluated[key])
                    else:
                        outcomes.add(None)
                if outcomes == {True} and context_agrees:
                    candidates.append(ann)
                elif outcomes != {False} or not context_agrees:
                    pending.append(
                        ConditionalOverride(
                            kind="conditional_override",
                            symbol=node.class_name,
                            candidates=[ann.resolves_to],
                            conditions=[ann.applies_when],
                        )
                    )
            if len(candidates) == 1:
                bindings[node.id] = candidates[0].id
                active.add(candidates[0].id)
                contexts_by_node[node.id] = contexts[0]
            elif len(candidates) > 1:
                pending.append(
                    ConditionalOverride(
                        kind="conditional_override",
                        symbol=node.class_name,
                        candidates=sorted(ann.resolves_to for ann in candidates),
                        conditions=sorted(ann.applies_when for ann in candidates),
                    )
                )
            covered = {ann.resolves_to for ann in candidates}
            for replacement in sorted(swaps - covered):
                pending.append(
                    ConditionalOverride(
                        kind="conditional_override",
                        reason="unannotated_override",
                        symbol=node.class_name,
                        candidates=[replacement],
                        conditions=[],
                    )
                )
        represented = {node.class_name for node in ir.nodes}
        for extract in extracts:
            for fact in extract.modules:
                if fact.ascend and fact.class_name not in represented:
                    pending.append(
                        ConditionalOverride(
                            kind="conditional_override",
                            reason="unannotated_override",
                            symbol=fact.class_name,
                            candidates=[fact.ascend.class_name],
                            conditions=[],
                        )
                    )
        result = IR.model_validate(
            module.apply_annotations(ir, resolved, bindings=bindings, active_ids=active)
        )
        if result.snapshot != ir.snapshot:
            raise ValueError("annotation output disagrees with IR snapshot")
        for node in result.nodes:
            if node.id in bindings and node.ascend:
                node.ascend.context = contexts_by_node[node.id]
        # Preserve newer stale.target when available; legacy contracts have no such field.
        merged = {item.model_dump_json(): item for item in [*triage.items, *pending]}
        return result, Triage(snapshot=ir.snapshot, items=[merged[key] for key in sorted(merged)])
    original = IR.model_validate(read_json(fixture(ir.snapshot.model.hf_id, "ir.json")))
    if ir != original:
        raise ValueError("fixture annotation fallback requires the unchanged, already annotated IR")
    return ir, triage


def layout_stage(ir_path: Path, output: Path) -> View:
    ir = IR.model_validate(read_json(ir_path))
    executable = ROOT / "packages/layout/dist/cli.js"
    if not executable.is_file() and (ROOT / "packages/layout/src").is_dir():
        subprocess.run(
            ["pnpm", "--filter", "@inferatlas/layout", "build"], cwd=ROOT, check=True, timeout=120
        )
    if executable.is_file():
        subprocess.run(
            ["node", str(executable), str(ir_path.resolve()), "-o", str(output.resolve())],
            cwd=ROOT,
            check=True,
            timeout=120,
        )
        result = View.model_validate(read_json(output))
    else:
        fixture_use("layout")
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
    calibration_env: Path | None = None,
) -> Report:
    src = source_root(ROOT)
    return verify_ir(
        ir,
        extracts=extracts,
        triage=triage,
        repos={name: src / name for name in ("vllm", "vllm-ascend", "pytorch")},
        golden_dir=golden_dir or golden_path(ir),
        calibration=read_json(calibration) if calibration else None,
        evaluator=Evaluator(ROOT, warning),
        calibration_env=read_json(calibration_env) if calibration_env else None,
        fixture_stages=_used_fixtures.get() or set(),
        allow_fixtures=_allow_fixtures.get() or os.environ.get("INFERATLAS_ALLOW_FIXTURES") == "1",
    )


def snapshot_key(snapshot: Snapshot, pair: str) -> str:
    config = hashlib.sha256(snapshot.model.config_rev.encode()).hexdigest()[:12]
    return (
        f"{pair}-{snapshot.device}-{snapshot.vllm.commit[:12]}-"
        f"{snapshot.ascend.commit[:12]}-{config}-g{snapshot.generator_version}"
    )


def build_pipeline(
    model: str, device: str, pair: str, output: Path | None, tp: list[int], ep: bool = False
) -> tuple[Path, int]:
    if not tp:
        raise ValueError("build needs at least one TP point")
    snapshot = resolve_stage(model, device, pair)
    directory = (
        output or ROOT / ".cache/build" / snapshot_key(snapshot, pair) / catalog_model(model).id
    )
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".inferatlas-build-", dir=directory.parent) as scratch:
        pending = Path(scratch) / "artifacts"
        write_json(pending / "snapshot.json", snapshot)
        extracts = [extract_stage(snapshot, degree, ep) for degree in sorted(set(tp))]
        for extract in extracts:
            name = (
                "extract.json"
                if extract.parallel.tp == 1
                else f"extract-tp{extract.parallel.tp}.json"
            )
            write_json(pending / name, extract)
        ir, triage = lower_stage(extracts)
        ir, triage = annotate_stage(ir, extracts, triage)
        write_json(pending / "ir.json", ir)
        write_json(pending / "triage.json", triage)
        report = report_stage(ir, extracts, triage)
        try:
            layout_stage(pending / "ir.json", pending / "view.json")
            report.checks.append(
                Check(id="layout", status="pass", message="View validates against IR.")
            )
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
            (pending / "view.json").unlink(missing_ok=True)
            warning(f"Layout failed: {error}")
            report.checks.append(
                Check(id="layout", status="fail", message="Layout generation failed.")
            )
        if _used_fixtures.get():
            provenance = next(check for check in report.checks if check.id == "provenance")
            provenance.status = "fail"
            provenance.message = "Fixture stages: " + ", ".join(sorted(_used_fixtures.get()))
        report.checks.sort(key=lambda check: (check.id, check.node or ""))
        write_json(pending / "report.json", report)
        write_json(pending / "provenance.json", {"fixtures": sorted(_used_fixtures.get() or [])})
        publish_directory(pending, directory)
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
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Resolve a pinned model/device/version identity."""
    save_artifact(output, resolve_stage(model, device, pair))


@app.command()
@guarded
def extract(
    snapshot: Path = Path("snapshot.json"),
    tp: int = 1,
    ep: bool = False,
    output: Path = Path("extract.json"),
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Extract CPU/meta facts for one TP degree."""
    inherit_provenance(snapshot)
    save_artifact(output, extract_stage(Snapshot.model_validate(read_json(snapshot)), tp, ep))


@app.command()
@guarded
def lower(
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    output: Path = Path("ir.json"),
    triage_output: Path = Path("triage.json"),
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Lower one or more extracts into symbolic IR and blocking triage."""
    inherit_provenance(*(extract or [Path("extract.json")]))
    ir, triage = lower_stage(
        [Extract.model_validate(read_json(path)) for path in extract or [Path("extract.json")]]
    )
    save_artifact(output, ir)
    save_artifact(triage_output, triage)
    raise typer.Exit(2 if triage.items else 0)


@app.command()
@guarded
def annotate(
    ir: Path = Path("ir.json"),
    triage: Path = Path("triage.json"),
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    output: Path = Path("ir.json"),
    triage_output: Path = Path("triage.json"),
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Apply reviewed Ascend annotations and merge unresolved triage."""
    inherit_provenance(ir, triage, *(extract or []))
    graph = IR.model_validate(read_json(ir))
    issues = Triage.model_validate(read_json(triage))
    if issues.snapshot != graph.snapshot:
        raise ValueError("triage snapshot disagrees with IR")
    graph, issues = annotate_stage(
        graph, [Extract.model_validate(read_json(path)) for path in extract or []], issues
    )
    save_artifact(output, graph)
    save_artifact(triage_output, issues)
    raise typer.Exit(2 if issues.items else 0)


@app.command()
@guarded
def verify(
    ir: Path = Path("ir.json"),
    triage: Path | None = None,
    extract: Annotated[list[Path] | None, typer.Option()] = None,
    golden_dir: Path | None = None,
    calibration: Path | None = None,
    calibration_env: Path | None = None,
    output: Path = Path("report.json"),
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Write a schema-valid report; exit 2 for triage or 3 for failed verification."""
    inherit_provenance(ir, *(extract or []))
    issues = Triage.model_validate(read_json(triage)) if triage else None
    report = report_stage(
        IR.model_validate(read_json(ir)),
        [Extract.model_validate(read_json(path)) for path in extract or []],
        issues,
        golden_dir,
        calibration,
        calibration_env,
    )
    write_json(output, report)
    if report.golden_diff:
        typer.echo(format_diff(report.golden_diff))
    raise typer.Exit(exit_code(report, issues))


@golden_app.command("update")
@guarded
def golden_update(
    ir: Path = Path("ir.json"),
    directory: Path | None = None,
    grid: Path | None = None,
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Re-evaluate a grid before replacing committed golden files."""
    graph = IR.model_validate(read_json(ir))
    directory = directory or golden_path(graph)
    if grid is None and (directory / "eval.json").is_file():
        grid = directory / "eval.json"
    values = Eval.model_validate(read_json(grid)) if grid else default_grid(graph)
    update_golden(directory, graph, values, Evaluator(ROOT, warning))
    typer.echo("Golden updated.")


@golden_app.command("check")
@guarded
def golden_check(
    ir: Path = Path("ir.json"), directory: Path | None = None, allow_fixtures: AllowFixtures = False
) -> None:
    """Compare IR and fresh evaluations, printing differences per node/field."""
    graph = IR.model_validate(read_json(ir))
    evaluator = Evaluator(ROOT, warning)
    diffs = check_golden(directory or golden_path(graph), graph, evaluator)
    if evaluator.backend != "node":
        typer.echo("SKIPPED numeric golden evaluation.")
        if not allow_fixtures and os.environ.get("INFERATLAS_ALLOW_FIXTURES") != "1":
            raise typer.Exit(3)
    typer.echo(
        format_diff(diffs)
        if diffs
        else ("Golden matches." if evaluator.backend == "node" else "Structural golden matches.")
    )
    raise typer.Exit(3 if diffs else 0)


@app.command()
@guarded
def layout(
    ir: Path = Path("ir.json"),
    output: Path = Path("view.json"),
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Run the Node layout CLI; fixture mode requires explicit opt-in."""
    inherit_provenance(ir)
    layout_stage(ir, output)
    save_artifact(output, View.model_validate(read_json(output)))


@app.command()
@guarded
def build(
    model: ModelOption = DEFAULT_MODEL,
    device: DeviceOption = "A2",
    pair: PairOption = "main-verified",
    output: Path | None = None,
    tp: Annotated[list[int] | None, typer.Option()] = None,
    ep: bool = False,
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Run resolve, extract, lower, annotate, verify and layout."""
    _, code = build_pipeline(model, device, pair, output, tp or [1, 4], ep)
    raise typer.Exit(code)


@app.command("parse-logs")
@guarded
def parse_logs(log: Path, output: Path = Path("calibration.json")) -> None:
    """Parse vLLM startup logs into unit-normalized calibration JSON."""
    write_json(output, parse_vllm_logs(log.read_text(encoding="utf-8")))


@app.command()
@guarded
def site(
    build_root: Path | None = None, data_dir: Path | None = None, include_blocked: bool = False
) -> None:
    """Copy build outputs, generate the version/model index and build the web app."""
    build_root = build_root or ROOT / ".cache/build"
    data = catalog()
    cells = {}
    artifacts = []
    code = 0
    for path in sorted(build_root.glob("*/*/ir.json")):
        directory = path.parent
        ir = IR.model_validate(read_json(path))
        report = Report.model_validate(read_json(directory / "report.json"))
        triage = Triage.model_validate(read_json(directory / "triage.json"))
        if not (ir.snapshot == report.snapshot == triage.snapshot):
            raise ValueError("build artifacts have inconsistent snapshot identities")
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
        provenance = [check for check in report.checks if check.id == "provenance"]
        if not provenance or any(c.status != "pass" for c in provenance):
            raise ValueError(f"site refuses fixture or unverified provenance: {directory}")
        if status == 3:
            raise ValueError(f"site refuses failed verification: {directory}")
        view = View.model_validate(read_json(directory / "view.json"))
        validate_view(ir, view)
        code = max(code, status)
        relative = Path(directory.parent.name) / model.id
        for name, value in (("ir", ir), ("view", view), ("report", report)):
            artifacts.append((relative / f"{name}.json", value))
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
    if code == 2 and not include_blocked:
        typer.echo("ERROR: site blocked by triage; use --include-blocked for review.", err=True)
        raise typer.Exit(2)
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
    if data_dir is not None:
        data_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="inferatlas-site-", dir=data_dir.parent if data_dir is not None else None
    ) as temporary:
        fresh_data = Path(temporary) / "data"
        for path, value in artifacts:
            write_json(fresh_data / path, value)
        write_json(fresh_data / "index.json", index)
        write_json(
            fresh_data / "hardware.json",
            [p.model_dump(mode="json") for p in sorted(data.hardware_presets, key=lambda p: p.id)],
        )
        subprocess.run(
            ["pnpm", "--filter", "web", "build"],
            cwd=ROOT,
            env={**os.environ, "INFERATLAS_DATA_DIR": str(fresh_data.resolve())},
            check=True,
            timeout=300,
        )
        if data_dir is not None:
            publish_directory(fresh_data, data_dir)
    raise typer.Exit(code)


@app.command()
@guarded
def single(
    model: ModelOption = DEFAULT_MODEL,
    device: DeviceOption = "A2",
    pair: PairOption = "main-verified",
    output: Path | None = None,
    include_blocked: bool = False,
    ep: bool = False,
    allow_fixtures: AllowFixtures = False,
) -> None:
    """Build a model and ask the web single-file plugin to inline its artifacts."""
    directory, code = build_pipeline(model, device, pair, None, [1, 4], ep)
    if code == 3 or (code == 2 and not include_blocked):
        raise typer.Exit(code)
    output = output or ROOT / "dist" / f"inferatlas-{catalog_model(model).id}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    # A fresh destination prevents stale HTML from masquerading as build success.
    with tempfile.TemporaryDirectory(prefix="inferatlas-single-") as temporary:
        selected = catalog_model(model)
        data = Path(temporary) / "data"
        relative = Path(f"{pair}-{device}") / selected.id
        for name in ("ir", "view", "report"):
            write_json(data / relative / f"{name}.json", read_json(directory / f"{name}.json"))
        write_json(
            data / "hardware.json", [p.model_dump(mode="json") for p in catalog().hardware_presets]
        )
        version = next(p for p in catalog().version_pairs if p.id == pair)
        write_json(
            data / "index.json",
            Index(
                versions=[dict(id=pair, label=pair, vllm=version.vllm, ascend=version.ascend)],
                models=[
                    dict(
                        id=selected.id,
                        hf_id=selected.hf_id,
                        label=selected.hf_id.split("/")[-1],
                        family=selected.family,
                    )
                ],
                cells=[
                    dict(
                        version=pair,
                        model=selected.id,
                        device=device,
                        status="ok" if code == 0 else "triage",
                        data_path=f"data/{relative.as_posix()}",
                        triage_count=len(read_json(directory / "triage.json")["items"]),
                    )
                ],
            ),
        )
        env = {
            **os.environ,
            "INFERATLAS_DATA_DIR": str(data),
            "INFERATLAS_SINGLE_MODEL": selected.id,
        }
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
