"""Command-line entry point (`amber`). Subcommands are added by later tasks."""

import logging
from typing import Annotated

import typer

from amber import config

app = typer.Typer(
    help="Amber: five yous compete to predict your clothing picks.",
    no_args_is_help=True,
)
ingest_app = typer.Typer(
    help="Build the item catalogs (thumbnails + items.parquet) under the data dir.",
    no_args_is_help=True,
)
app.add_typer(ingest_app, name="ingest")


@app.callback()
def main() -> None:
    """Amber command-line interface."""


@ingest_app.callback()
def ingest() -> None:
    """Build the item catalogs."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )


@ingest_app.command()
def clothes(
    limit: Annotated[
        int | None,
        typer.Option(min=1, help="Stop after this many items (for a quick trial)."),
    ] = None,
) -> None:
    """Stream the clothing catalog from Hugging Face into <data dir>/clothes."""
    from amber.data.clothes import ingest_clothes  # heavy deps; keep `--help` fast

    out_dir = config.DATA_DIR / "clothes"
    items = ingest_clothes(out_dir, limit=limit)
    typer.echo(f"{len(items)} clothing items in {out_dir}")
