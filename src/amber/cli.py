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
decide_app = typer.Typer(
    help="Decision engine: predict your own Spotify skips from your data export.",
    no_args_is_help=True,
)
app.add_typer(decide_app, name="decide")
topics_app = typer.Typer(
    help="Topics: what your activity is about and how much each topic matters to you.",
    no_args_is_help=True,
)
app.add_typer(topics_app, name="topics")

# Where `amber decide eval --publish` copies the aggregates-only report.
PUBLIC_RESULTS_PATH = config.REPO_ROOT / "docs" / "results" / "spotify-v1.md"


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


@app.command()
def serve(
    port: Annotated[
        int, typer.Option(min=1, max=65535, help="Port to listen on.")
    ] = 8000,
    host: Annotated[str, typer.Option(help="Address to bind.")] = "127.0.0.1",
) -> None:
    """Serve the API (and web/dist at / if it has been built)."""
    import uvicorn  # keep `--help` fast

    from amber.api.app import create_app
    from amber.context import load_context

    try:
        context = load_context(config.DATA_DIR)
    except FileNotFoundError as err:
        raise _fail(
            f"Missing data under {config.DATA_DIR} ({err}): run `amber ingest clothes`, "
            "`amber ingest art` and `amber embed` first."
        ) from err
    uvicorn.run(create_app(context), host=host, port=port)


# --- the decision engine ----------------------------------------------------------------


class _FakeKind(StrEnum):
    personal = "personal"
    null = "null"


class _ExportFormat(StrEnum):
    extended = "extended"
    account = "account"


def _export_dir(export_dir: Path | None) -> Path:
    return export_dir or config.DATA_DIR / "exports" / "spotify"


@decide_app.command("fake")
def decide_fake(
    dest: Annotated[Path, typer.Argument(help="Folder to write the fake export into.")],
    kind: Annotated[
        _FakeKind, typer.Option(help="personal has real skip structure; null has none.")
    ] = _FakeKind.personal,
    fmt: Annotated[_ExportFormat, typer.Option(help="Spotify export format.")] = (
        _ExportFormat.extended
    ),
    n_plays: Annotated[int, typer.Option(min=1, help="Plays to generate.")] = 6000,
) -> None:
    """Write a fake Spotify export with planted (or no) skip structure."""
    from amber.decide.fake import write_fake_export

    out = write_fake_export(dest, kind.value, n_plays=n_plays, fmt=fmt.value)
    typer.echo(f"wrote a fake {kind.value} {fmt.value} export ({n_plays} plays) to {out}")


@decide_app.command("ingest")
def decide_ingest(
    export_dir: Annotated[
        Path | None,
        typer.Argument(help="Spotify export folder (default: <data dir>/exports/spotify)."),
    ] = None,
) -> None:
    """Parse the export into <data dir>/decide/decisions.parquet."""
    from amber.decide.pipeline import load_plays
    from amber.decide.report import counts_line

    source = _export_dir(export_dir)
    try:
        df = load_plays(source)
    except ValueError as err:
        raise _fail(str(err)) from err
    out_dir = config.DATA_DIR / "decide"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "decisions.parquet")
    typer.echo(
        f"{len(df)} plays from {df['ts'].min():%Y-%m-%d} to {df['ts'].max():%Y-%m-%d} "
        f"-> {out_dir / 'decisions.parquet'}"
    )
    typer.echo(counts_line(df.attrs))


class _SourcesFormat(StrEnum):
    html = "html"
    json = "json"


@decide_app.command("fake-sources")
def decide_fake_sources(
    dest: Annotated[Path, typer.Argument(help="Exports folder to write google/ and instagram/ into.")],
    fmt: Annotated[_SourcesFormat, typer.Option(help="Google Takeout format.")] = (
        _SourcesFormat.html
    ),
) -> None:
    """Write fake Google Takeout and Instagram exports (made-up data) for demos and tests."""
    from amber.decide.fake_sources import write_fake_sources

    out = write_fake_sources(dest, fmt=fmt.value)
    typer.echo(f"wrote fake Google ({fmt.value}) and Instagram exports to {out}")


@decide_app.command("sources")
def decide_sources(
    export_dir: Annotated[
        Path | None,
        typer.Argument(
            help="Exports folder holding google/, meta/ and instagram/ (default: <data dir>/exports)."
        ),
    ] = None,
) -> None:
    """Read Google and Instagram exports into <data dir>/decide/events.parquet and audit them.

    Prints counts and dates only: never a title, query, URL, username or id.
    """
    from amber.decide.sources import audit_lines, load_sources

    source = export_dir or config.DATA_DIR / "exports"
    try:
        df = load_sources(source)
    except ValueError as err:
        raise _fail(str(err)) from err
    out_dir = config.DATA_DIR / "decide"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "events.parquet")
    for line in audit_lines(df):
        typer.echo(line)
    typer.echo(f"-> {out_dir / 'events.parquet'}")


# Where `amber decide tasks --publish` copies the aggregates-only report.
TASKS_RESULTS_PATH = config.REPO_ROOT / "docs" / "results" / "sources-v2.md"


@decide_app.command("tasks")
def decide_tasks(
    publish: Annotated[
        bool,
        typer.Option(
            help="Also copy the aggregates-only report to docs/results "
            "(refused for synthetic data)."
        ),
    ] = False,
) -> None:
    """Score the frozen v2 decisions (YouTube, Instagram) on <data dir>/decide/events.parquet.

    Uses <data dir>/profile/life.yaml for local time and life context when present.
    Prints and writes aggregates only.
    """
    import json

    import pandas as pd

    from amber.decide.life import load_timeline
    from amber.decide.tasks import is_synthetic_events, render_tasks_md, run_tasks

    path = config.DATA_DIR / "decide" / "events.parquet"
    if not path.exists():
        raise _fail(f"no event log at {path}: run `amber decide sources` first")
    events = pd.read_parquet(path)
    life = config.DATA_DIR / "profile" / "life.yaml"
    tl = load_timeline(life) if life.exists() else None
    typer.echo("life context: " + ("on" if tl else "off (no profile/life.yaml)"))
    results = run_tasks(events, tl)
    md = render_tasks_md(results)
    out_dir = config.DATA_DIR / "decide" / "tasks"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "public.md").write_text(md, encoding="utf-8")
    (out_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    typer.echo(md)
    typer.echo(f"wrote {out_dir / 'public.md'} and {out_dir / 'results.json'}")
    if publish:
        if is_synthetic_events(events):
            raise _fail(
                "refusing to --publish: this event log is synthetic (fake). "
                "Run on real exports to publish."
            )
        TASKS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        TASKS_RESULTS_PATH.write_text(md, encoding="utf-8")
        typer.echo(f"published {TASKS_RESULTS_PATH}")


@decide_app.command("eval")
def decide_eval(
    export_dir: Annotated[
        Path | None,
        typer.Argument(help="Spotify export folder (default: <data dir>/exports/spotify)."),
    ] = None,
    tz: Annotated[str, typer.Option(help="Your timezone, e.g. Europe/Berlin.")] = "UTC",
    publish: Annotated[
        bool,
        typer.Option(
            help="Also copy the aggregates-only report to docs/results "
            "(refused for synthetic data)."
        ),
    ] = False,
) -> None:
    """Evaluate the decision ladder; write private reports to <data dir>/decide."""
    from amber.decide.pipeline import run_pipeline
    from amber.decide.report import counts_line, headline

    out_dir = config.DATA_DIR / "decide"
    try:
        result = run_pipeline(_export_dir(export_dir), out_dir, tz=tz)
    except ValueError as err:
        raise _fail(str(err)) from err
    typer.echo(headline(result))
    typer.echo(counts_line(result))
    typer.echo(f"wrote {out_dir / 'report.md'} (private) and {out_dir / 'public.md'}")
    if publish:
        if result["synthetic"]:
            raise _fail(
                "refusing to --publish: this run used synthetic (fake) data. "
                "Evaluate a real export to publish."
            )
        PUBLIC_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        PUBLIC_RESULTS_PATH.write_text(
            (out_dir / "public.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        typer.echo(f"published {PUBLIC_RESULTS_PATH}")


class _TopicEncoder(StrEnum):
    minilm = "minilm"
    hash = "hash"


@topics_app.command("build")
def topics_build(
    encoder: Annotated[
        _TopicEncoder, typer.Option(help="minilm (local sentence model) or hash (no model).")
    ] = _TopicEncoder.minilm,
    min_docs: Annotated[int, typer.Option(min=1, help="Smallest topic kept.")] = 30,
) -> None:
    """Cluster <data dir>/decide/events.parquet into topics and score their relevance.

    Writes <data dir>/topics/. Prints counts only: never a title, query or topic name.
    """
    import pandas as pd

    from amber.topics.embed import CachedEncoder, HashingEncoder, MiniLMEncoder
    from amber.topics.pipeline import build_topics

    path = config.DATA_DIR / "decide" / "events.parquet"
    if not path.exists():
        raise _fail(f"no event log at {path}: run `amber decide sources` first")
    out_dir = config.DATA_DIR / "topics"
    base = MiniLMEncoder() if encoder is _TopicEncoder.minilm else HashingEncoder()
    enc = CachedEncoder(base, out_dir / "embeddings.npz")
    try:
        summary = build_topics(
            pd.read_parquet(path), enc, out_dir,
            life_path=config.DATA_DIR / "profile" / "life.yaml", min_docs=min_docs,
        )
    except ValueError as err:
        raise _fail(str(err)) from err
    for k, v in summary.items():
        typer.echo(f"{k}: {v}")
    typer.echo(f"-> {out_dir} (names and texts stay there; open the veto screen to see them)")
