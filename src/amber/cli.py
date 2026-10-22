"""Command-line entry point (`amber`). Subcommands are added by later tasks."""

import atexit
import logging
import os
import sys
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
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("amber").setLevel(logging.INFO)  # progress, without HTTP chatter


def _exit_without_teardown() -> None:
    """Leave now, after running Python-level exit handlers but skipping C++ teardown.

    A pyarrow/datasets stream that is abandoned half-way (as `--limit` does) leaves an
    async read pending in Arrow's thread pool; at interpreter exit the pool's destructor
    then waits on it forever (observed with pyarrow 26 / datasets 5.1). A stream that is
    read to the end exits normally.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    atexit._run_exitfuncs()
    os._exit(0)


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
    if limit is not None:
        _exit_without_teardown()
