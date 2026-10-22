"""Stream the clothing catalog from Hugging Face into thumbnails + `items.parquet`.

Nothing is downloaded whole: rows are streamed once, filtered to the kept master
categories, and only a `THUMB_SIZE` JPEG per item is written to disk. The run is
resumable: a row whose thumbnail already exists is not re-encoded, but its metadata is
still collected, so thumbnails written by an interrupted run never lose their rows.
"""

import io
import logging
import math
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

from amber import config
from amber.data.files import clean_text, save_thumbnail, write_catalog

logger = logging.getLogger(__name__)

CLOTHES_DATASET = "benitomartin/fashion-product-images-small-384x512"
PROGRESS_EVERY = 2000

# items.parquet column -> Hugging Face field (string columns; missing becomes "")
_TEXT_FIELDS = {
    "title": "productDisplayName",
    "gender": "gender",
    "master_category": "masterCategory",
    "sub_category": "subCategory",
    "article_type": "articleType",
    "base_colour": "baseColour",
    "season": "season",
    "usage": "usage",
}
COLUMNS = ["id", *_TEXT_FIELDS, "year"]


def _stream_rows() -> Iterable[dict]:
    """The train split streamed from the Hub, with images left undecoded.

    `decode=False` yields `{"bytes": ..., "path": ...}` instead of PIL images, so a corrupt
    image fails inside our per-row handler rather than killing the whole stream.
    """
    import datasets  # heavy import; only needed when actually streaming

    stream = datasets.load_dataset(CLOTHES_DATASET, split="train", streaming=True)
    return stream.cast_column("image", datasets.Image(decode=False))


def _year(value: object) -> int | None:
    """The source year may be an int, a float, a NaN or missing; null when unusable."""
    try:
        year = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return int(year) if math.isfinite(year) else None


def _thumbnail(image: object) -> Image.Image:
    """Decode (from PIL or `datasets.Image(decode=False)` bytes) and fit to THUMB_SIZE.

    `fit` centre-crops to the target aspect ratio before resizing, so the exact size is
    guaranteed without distorting the product (a no-op crop for the usual 3:4 source).
    """
    if isinstance(image, dict):
        data = image.get("bytes")
        if not data:
            raise ValueError("image has no bytes")
        with Image.open(io.BytesIO(data)) as im:
            rgb = im.convert("RGB")
    else:
        rgb = image.convert("RGB")  # type: ignore[attr-defined]
    return ImageOps.fit(rgb, config.THUMB_SIZE, Image.Resampling.LANCZOS)


def _frame(records: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(records, columns=COLUMNS)
    dtypes = {column: "str" for column in COLUMNS if column != "year"}
    return df.astype({**dtypes, "year": "Int64"})  # year: nullable, may be missing


def ingest_clothes(
    out_dir: Path, rows: Iterable[dict] | None = None, limit: int | None = None
) -> pd.DataFrame:
    """Build `out_dir/items.parquet` and `out_dir/thumbs/<n>.jpg`; return the table.

    `rows` are Hugging Face-shaped dicts (`id, gender, masterCategory, subCategory,
    articleType, baseColour, season, year, usage, productDisplayName, image`); `None`
    streams the dataset. Rows outside the kept master categories are dropped; rows whose
    image cannot be decoded are logged and skipped. `limit` stops after that many items
    have been collected in this run (useful for quick trials). The returned/written table
    also keeps rows from a previous run's parquet, de-duplicated by `id`. If the run is
    interrupted (an exception from the stream, Ctrl-C), the rows collected so far are
    saved before the exception propagates.
    """
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive number of items")
    out_dir = Path(out_dir)
    thumbs = out_dir / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    parquet = out_dir / "items.parquet"
    if rows is None:
        rows = _stream_rows()

    records: list[dict] = []
    skipped = 0
    try:
        for row in rows:
            if (
                clean_text(row.get("masterCategory"))
                not in config.KEEP_MASTER_CATEGORIES
            ):
                continue
            try:
                n = int(row["id"])
            except (KeyError, TypeError, ValueError):
                logger.warning("skipping row without a usable id: %r", row.get("id"))
                skipped += 1
                continue
            dest = thumbs / f"{n}.jpg"
            if not dest.exists():
                try:
                    thumb = _thumbnail(row["image"])
                except Exception as exc:  # noqa: BLE001 - PIL raises many types; skip the item
                    logger.warning("skipping c:%d, image not decodable: %s", n, exc)
                    skipped += 1
                    continue
                save_thumbnail(thumb, dest)
            records.append(
                {
                    "id": f"c:{n}",
                    **{
                        col: clean_text(row.get(field))
                        for col, field in _TEXT_FIELDS.items()
                    },
                    "year": _year(row.get("year")),
                }
            )
            if len(records) % PROGRESS_EVERY == 0:
                logger.info("%d items collected", len(records))
            if limit is not None and len(records) >= limit:
                break
    except BaseException:
        # network error, full disk, Ctrl-C: keep the metadata of the thumbnails written
        # so far (the next run would rebuild it anyway, but only after re-streaming)
        if records:
            items = write_catalog(parquet, thumbs, _frame(records))
            logger.warning("interrupted; saved %d items to the catalog", len(items))
        raise

    items = write_catalog(parquet, thumbs, _frame(records))
    logger.info("%d items in catalog (%d skipped this run)", len(items), skipped)
    return items
