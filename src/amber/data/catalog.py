"""Lookups over the ingested catalogs: items table, thumbnail path, id -> domain.

Item ids are `"c:<n>"` (clothes) or `"a:<n>"` (art); thumbnails live at
`<data_dir>/<domain>/thumbs/<n>.jpg` and the table at `<data_dir>/<domain>/items.parquet`.
"""

import re
from pathlib import Path
from typing import Literal

import pandas as pd

from amber import config

Domain = Literal["clothes", "art"]

_DOMAINS: dict[str, Domain] = {"c": "clothes", "a": "art"}
_ITEM_ID = re.compile(r"([ca]):([0-9]+)")


def _parse(item_id: str) -> tuple[Domain, str]:
    match = _ITEM_ID.fullmatch(item_id) if isinstance(item_id, str) else None
    if match is None:
        raise ValueError(f"bad item id {item_id!r}: expected 'c:<n>' or 'a:<n>'")
    return _DOMAINS[match.group(1)], match.group(2)


def domain_of(item_id: str) -> Domain:
    """`"c:12"` -> `"clothes"`, `"a:9"` -> `"art"`; anything else raises ValueError."""
    return _parse(item_id)[0]


def thumb_path(item_id: str, data_dir: Path | None = None) -> Path:
    """Where the thumbnail of `item_id` lives. `data_dir` defaults to `config.DATA_DIR`."""
    domain, n = _parse(item_id)
    root = config.DATA_DIR if data_dir is None else Path(data_dir)
    return root / domain / "thumbs" / f"{n}.jpg"


def load_items(domain: Domain, data_dir: Path | None = None) -> pd.DataFrame:
    """The `items.parquet` table of a domain. `data_dir` defaults to `config.DATA_DIR`."""
    if domain not in ("clothes", "art"):
        raise ValueError(f"unknown domain {domain!r}: expected 'clothes' or 'art'")
    root = config.DATA_DIR if data_dir is None else Path(data_dir)
    return pd.read_parquet(root / domain / "items.parquet")
