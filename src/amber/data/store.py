"""Helpers shared by the catalog ingest commands: text cleanup, atomic thumbnail and
`items.parquet` writes, and the merge with whatever an earlier run left behind.

Item ids are `"<domain letter>:<n>"` (`"c:12"`, `"a:9"`) and the thumbnail of an item lives
at `<thumbs dir>/<n>.jpg`, as in `amber.data.catalog`.
"""

import math
import os
from pathlib import Path

import pandas as pd
from PIL import Image

THUMB_QUALITY = 85


def clean_text(value: object) -> str:
    """A source field as a stripped string; missing (None or NaN) becomes ""."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip()


def save_thumbnail(thumb: Image.Image, dest: Path) -> None:
    """Write via a temp file so a killed run never leaves a half-written thumbnail,
    which the resume check would otherwise treat as done."""
    tmp = dest.with_name(f".{dest.name}.tmp")
    thumb.save(tmp, format="JPEG", quality=THUMB_QUALITY)
    os.replace(tmp, dest)


def merge_items(
    existing: pd.DataFrame | None, new: pd.DataFrame, thumbs: Path
) -> pd.DataFrame:
    """Existing + new rows, one per id (new wins), only those that have a thumbnail,
    ordered by numeric id so the table does not depend on fetch order."""
    merged = new if existing is None else pd.concat([existing, new], ignore_index=True)
    merged = merged.drop_duplicates("id", keep="last")
    has_thumb = merged["id"].map(lambda i: (thumbs / f"{i[2:]}.jpg").exists())
    merged = merged.loc[has_thumb.astype(bool)]  # astype: an empty map() is not a mask
    order = merged["id"].str.slice(2).astype("int64").sort_values(kind="stable").index
    return merged.loc[order].reset_index(drop=True)


def write_catalog(parquet: Path, thumbs: Path, new: pd.DataFrame) -> pd.DataFrame:
    """Merge `new` into the existing parquet (if any) and write it atomically."""
    existing = pd.read_parquet(parquet) if parquet.exists() else None
    items = merge_items(existing, new, thumbs)
    tmp = parquet.with_name(f"{parquet.name}.tmp")
    items.to_parquet(tmp, index=False)
    os.replace(tmp, parquet)
    return items
