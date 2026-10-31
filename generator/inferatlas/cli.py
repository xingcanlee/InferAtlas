"""CLI shell. Pipeline commands are supplied by T6."""

import typer

from inferatlas import __version__

app = typer.Typer(
    no_args_is_help=True, help="Generate source-linked inference architecture atlases."
)


@app.callback()
def main() -> None:
    """InferAtlas pipeline command group (commands supplied by owning tasks)."""


@app.command()
def version() -> None:
    """Print the generator version."""
    typer.echo(__version__)
