"""Command-line entry point (`amber`). Subcommands are added by later tasks."""

import atexit
import logging
import os
import sys
import traceback
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


def _exit_without_teardown(code: int = 0) -> None:
    """Leave now with `code`, after running Python-level exit handlers but skipping C++ teardown.

    A pyarrow/datasets stream that is abandoned half-way leaves an async read pending in
    Arrow's thread pool; at interpreter exit the pool's destructor then waits on it
    forever (observed with pyarrow 26 / datasets 5.1). A stream read to the end exits
    normally. A stream is abandoned by `--limit`, but equally by Ctrl-C, a network error
    or a full disk, so every way out of the streaming call that is not a clean full read
    comes through here.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    atexit._run_exitfuncs()
    sys.stdout.flush()  # exit handlers may have printed too
    sys.stderr.flush()
    os._exit(code)


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
    try:
        items = ingest_clothes(out_dir, limit=limit)
    except BaseException:  # noqa: BLE001 - Ctrl-C, network error, disk full: stream abandoned
        traceback.print_exc()
        _exit_without_teardown(1)
        raise typer.Exit(1) from None  # only reached when the exit above is stubbed
    typer.echo(f"{len(items)} clothing items in {out_dir}")
    if limit is not None:
        _exit_without_teardown(0)


@ingest_app.command()
def art(
    n: Annotated[
        int,
        typer.Option(
            min=1, help="How many artworks to collect (about 1 request per second)."
        ),
    ] = 2000,
) -> None:
    """Fetch public-domain artworks from the Art Institute of Chicago into <data dir>/art."""
    from amber.data.art import ingest_art

    out_dir = config.DATA_DIR / "art"
    items = ingest_art(out_dir, n=n)
    typer.echo(f"{len(items)} artworks in {out_dir}")
