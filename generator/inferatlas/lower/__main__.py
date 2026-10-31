"""T3-local entry point; the repository-wide CLI is owned by T6."""

from pathlib import Path
from typing import Annotated

import typer
from ruamel.yaml import YAML

from inferatlas.schema.models import Catalog, Snapshot

from .catalog import lower_catalog
from .pipeline import coverage_gate, lower

DEFAULT_CATALOG = Path(__file__).resolve().parents[3] / "catalog.yaml"


def main(
    out: Annotated[Path, typer.Option(help="Directory receiving ir.json and triage.json.")],
    extract: Annotated[Path | None, typer.Option(exists=True)] = None,
    model: Annotated[
        str | None, typer.Option(help="Catalog id for a config-only Qwen3 projection.")
    ] = None,
    snapshot: Annotated[Path | None, typer.Option(exists=True)] = None,
    catalog: Annotated[Path, typer.Option(exists=True)] = DEFAULT_CATALOG,
    source_root: Annotated[Path | None, typer.Option(exists=True)] = None,
    moe_a2a: Annotated[
        bool,
        typer.Option(help="Select an all-to-all kernel backend instead of default gather/scatter."),
    ] = False,
):
    if (extract is None) == (model is None):
        raise typer.BadParameter("Choose exactly one of --extract or --model")
    if extract is not None:
        result = lower(extract, moe_a2a=moe_a2a)
    else:
        if snapshot is None:
            raise typer.BadParameter("--model requires --snapshot")
        models = Catalog.model_validate(YAML(typ="safe").load(catalog)).models
        selected = next((m for m in models if m.id == model), None)
        if selected is None:
            raise typer.BadParameter(f"Unknown catalog model: {model}")
        result = lower_catalog(
            selected,
            Snapshot.model_validate_json(snapshot.read_text()),
            source_root,
            moe_a2a=moe_a2a,
        )
    result.write(out)
    typer.echo(f"{len(result.ir.nodes)} nodes; {len(result.triage.items)} blocking triage items")
    if not coverage_gate(result.triage):
        raise typer.Exit(1)


if __name__ == "__main__":
    typer.run(main)
