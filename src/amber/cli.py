"""Command-line entry point (`amber`). Subcommands are added by later tasks."""

import atexit
import logging
import os
import sys
import time
import traceback
from enum import StrEnum
from pathlib import Path
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
            min=1,
            help="How many artworks to collect. Each artwork type gets its own search "
            "(and its own window of results) and they are taken in turns, so the mix "
            "is balanced. About 1 request per second.",
        ),
    ] = 2000,
) -> None:
    """Fetch public-domain artworks from the Art Institute of Chicago into <data dir>/art."""
    from amber.data.art import ingest_art

    out_dir = config.DATA_DIR / "art"
    items = ingest_art(out_dir, n=n)
    typer.echo(f"{len(items)} artworks in {out_dir}")


# --- the embedding space ----------------------------------------------------------------


class _EmbedDomain(StrEnum):
    clothes = "clothes"
    art = "art"
    all = "all"


class _Domain(StrEnum):
    clothes = "clothes"
    art = "art"


_ModelOption = Annotated[str, typer.Option(help="open_clip model name.")]
_PretrainedOption = Annotated[
    str, typer.Option(help="open_clip pretrained weights tag.")
]
_CONTACT_SHEET_COLUMNS = 4
_CONTACT_SHEET_ROWS = 3  # 4 x 3 = the top 12


def _log_progress() -> None:
    logging.basicConfig(
        level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("amber").setLevel(logging.INFO)


def _fail(message: str) -> typer.Exit:
    typer.echo(message, err=True)
    return typer.Exit(1)


@app.command()
def embed(
    model: _ModelOption = config.MODEL_NAME,
    pretrained: _PretrainedOption = config.MODEL_PRETRAINED,
    domain: Annotated[
        _EmbedDomain, typer.Option(help="Which catalog to embed.")
    ] = _EmbedDomain.all,
) -> None:
    """Embed every thumbnail and save the vectors + PCA bases under <data dir>/<domain>/emb/."""
    _log_progress()
    domains = ["clothes", "art"] if domain is _EmbedDomain.all else [domain.value]
    todo = []
    for name in domains:
        if (config.DATA_DIR / name / "items.parquet").exists():
            todo.append(name)
        elif domain is _EmbedDomain.all:
            typer.echo(f"skipping {name}: not ingested yet (run `amber ingest {name}`)")
        else:
            raise _fail(
                f"No {name} catalog in {config.DATA_DIR}: run `amber ingest {name}`."
            )

    from amber.space import encoder as encoder_module  # loads torch; keep `--help` fast
    from amber.space.index import embed_dir, embed_domain

    encoder = encoder_module.ClipEncoder(model, pretrained)
    for name in todo:
        started = time.perf_counter()
        index = embed_domain(name, encoder, data_dir=config.DATA_DIR)
        seconds = time.perf_counter() - started
        typer.echo(
            f"{name}: {len(index)} items embedded in {seconds:.1f}s "
            f"(dim {encoder.dim}) -> {embed_dir(name, encoder, config.DATA_DIR)}"
        )


@app.command("eval-space")
def eval_space(
    model: _ModelOption = config.MODEL_NAME,
    pretrained: _PretrainedOption = config.MODEL_PRETRAINED,
    min_group: Annotated[
        int, typer.Option(min=1, help="Smallest (colour, article type) group to score.")
    ] = 20,
) -> None:
    """Score a model's clothes embeddings: can text queries find items by colour + type?"""
    from amber.data.catalog import load_items
    from amber.space import encoder as encoder_module
    from amber.space.evaluate import label_retrieval_score
    from amber.space.index import VectorIndex, embed_dir

    encoder = encoder_module.ClipEncoder(model, pretrained)
    directory = embed_dir("clothes", encoder, config.DATA_DIR)
    if not directory.exists():
        raise _fail(f"No embeddings for {model}/{pretrained}: run `amber embed` first.")
    index = VectorIndex.load(directory)
    items = load_items("clothes", data_dir=config.DATA_DIR)
    try:
        score = label_retrieval_score(index, items, encoder, k=20, min_group=min_group)
    except ValueError as err:
        raise _fail(f"{err}; lower --min-group.") from err
    typer.echo(
        f"{model}/{pretrained}: P@20 = {score:.4f} (dim {encoder.dim}, {len(index)} items)"
    )


def _contact_sheet(paths: list[Path], dest: Path) -> None:
    """Tile thumbnails (left to right, top to bottom) into one JPEG at `dest`."""
    from PIL import Image, ImageOps

    cell_w, cell_h = 216, 288
    columns = _CONTACT_SHEET_COLUMNS
    rows = max(1, -(-len(paths) // columns))
    sheet = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    for i, path in enumerate(paths):
        with Image.open(path) as im:
            tile = ImageOps.pad(im.convert("RGB"), (cell_w, cell_h), color="white")
        sheet.paste(tile, ((i % columns) * cell_w, (i // columns) * cell_h))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, format="JPEG", quality=85)


@app.command("search-text")
def search_text(
    query: Annotated[str, typer.Argument(help="What to look for, in words.")],
    domain: Annotated[
        _Domain, typer.Option(help="Which catalog to search.")
    ] = _Domain.clothes,
    model: _ModelOption = config.MODEL_NAME,
    pretrained: _PretrainedOption = config.MODEL_PRETRAINED,
) -> None:
    """Show the 12 best matches for a text query as <data dir>/contact_sheet.jpg."""
    from amber.data.catalog import load_items, thumb_path
    from amber.space import encoder as encoder_module
    from amber.space.index import VectorIndex, embed_dir

    encoder = encoder_module.ClipEncoder(model, pretrained)
    directory = embed_dir(domain.value, encoder, config.DATA_DIR)
    if not directory.exists():
        raise _fail(
            f"No {domain.value} embeddings for {model}/{pretrained}: run `amber embed` first."
        )
    index = VectorIndex.load(directory)
    titles = load_items(domain.value, data_dir=config.DATA_DIR).set_index("id")["title"]

    direction = encoder.embed_texts([query])[0]
    hits = index.top_k(
        direction, _CONTACT_SHEET_COLUMNS * _CONTACT_SHEET_ROWS, use_centered=False
    )
    for item_id, score in hits:
        typer.echo(f"{item_id:>8}  {score:.3f}  {titles.get(item_id, '')}")
    dest = config.DATA_DIR / "contact_sheet.jpg"
    _contact_sheet([thumb_path(i, data_dir=config.DATA_DIR) for i, _ in hits], dest)
    typer.echo(f"wrote {dest}")


@app.command("shrink-thumbs")
def shrink_thumbs(
    domain: Annotated[
        str,
        typer.Option(
            help="Catalog to shrink (only 'clothes': art keeps its aspect ratios)."
        ),
    ] = "clothes",
) -> None:
    """Re-encode the clothing thumbnails in place at the configured size and quality."""
    from amber.data.files import THUMB_QUALITY, shrink_thumbnails

    if domain != "clothes":
        raise _fail("Only the clothes thumbnails can be shrunk.")
    thumbs = config.DATA_DIR / "clothes" / "thumbs"
    count = shrink_thumbnails(thumbs, config.THUMB_SIZE, THUMB_QUALITY)
    typer.echo(f"shrunk {count} thumbnails in {thumbs}")


_BENCH_TEMPERATURES = (0.05, 0.2)
_BENCH_PRIOR_VARS = (0.1, 1.0, 10.0)


@app.command("bench-active")
def bench_active(
    n_users: Annotated[
        int, typer.Option(min=1, help="Simulated people per table cell.")
    ] = 30,
    n_train: Annotated[
        int, typer.Option(min=1, help="Pairs each simulated person answers.")
    ] = 40,
    n_test: Annotated[
        int, typer.Option(min=1, help="Held-out pairs each model is scored on.")
    ] = 200,
    seed: Annotated[int, typer.Option(help="Random seed.")] = config.SEED,
    model: _ModelOption = config.MODEL_NAME,
    pretrained: _PretrainedOption = config.MODEL_PRETRAINED,
) -> None:
    """Simulate people on the clothes basis: does choosing pairs actively beat random ones?

    Prints held-out pairwise accuracy for strategy (active, random) x temperature
    (0.05, 0.2) x prior variance (0.1, 1, 10), as a markdown table.
    """
    from amber.prefs.simulate import run_benchmark
    from amber.space.basis import Basis
    from amber.space.index import VectorIndex

    directory = config.DATA_DIR / "clothes" / "emb" / f"{model}__{pretrained}"
    if not (directory / "basis").exists():
        raise _fail(
            f"No clothes embeddings for {model}/{pretrained}: run `amber embed` first."
        )
    index = VectorIndex.load(directory)
    X = Basis.load(directory / "basis").project(index.centered(index.ids))
    dup_vectors = index.vectors

    started = time.perf_counter()
    typer.echo(
        f"{len(index)} items, {X.shape[1]} dims; {n_users} users x {n_train} train "
        f"pairs x {n_test} test pairs, seed {seed}"
    )
    typer.echo("")
    typer.echo("| temperature | prior_var | active | random | active - random |")
    typer.echo("| --- | --- | --- | --- | --- |")
    for temperature in _BENCH_TEMPERATURES:
        for prior_var in _BENCH_PRIOR_VARS:
            accuracy = {
                strategy: run_benchmark(
                    X,
                    dup_vectors,
                    n_users=n_users,
                    n_train=n_train,
                    n_test=n_test,
                    strategy=strategy,
                    temperature=temperature,
                    seed=seed,
                    prior_var=prior_var,
                )
                for strategy in ("active", "random")
            }
            typer.echo(
                f"| {temperature} | {prior_var:g} | {accuracy['active']:.3f} | "
                f"{accuracy['random']:.3f} | {accuracy['active'] - accuracy['random']:+.3f} |"
            )
    chance = run_benchmark(
        X,
        dup_vectors,
        n_users=n_users,
        n_train=n_train,
        n_test=n_test,
        strategy="active",
        temperature=_BENCH_TEMPERATURES[0],
        seed=seed,
        random_clicker=True,
    )
    typer.echo("")
    typer.echo(f"coin-flip clicker (active, default prior_var): {chance:.3f}")
    typer.echo(f"took {time.perf_counter() - started:.0f}s")
