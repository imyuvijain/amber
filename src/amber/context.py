"""The app context: everything a request needs, loaded once at startup."""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from amber import config
from amber.config import DATA_DIR, DEFAULT_SETTINGS, Settings
from amber.data.catalog import load_items
from amber.space.basis import Basis
from amber.space.encoder import ClipEncoder, Encoder
from amber.space.index import VectorIndex, embed_dir
from amber.store.db import connect

DOMAINS = ("clothes", "art")


@dataclass
class AppContext:
    conn: sqlite3.Connection
    encoder: Encoder
    items: dict[str, pd.DataFrame]
    index: dict[str, VectorIndex]
    basis: dict[str, Basis]
    basis2d: dict[str, Basis]
    settings: Settings
    data_dir: Path


def load_context(
    data_dir: Path = DATA_DIR,
    settings: Settings = DEFAULT_SETTINGS,
    encoder: Encoder | None = None,
) -> AppContext:
    """Load catalogs, embeddings and bases for both domains and open `<data_dir>/amber.db`.

    Embeddings are read from `<data_dir>/<domain>/emb/<model_name>__<pretrained>/`, so the
    encoder decides which embedding space is used. With `encoder=None` the configured CLIP
    model is built.
    """
    data_dir = Path(data_dir)
    if encoder is None:
        encoder = ClipEncoder(config.MODEL_NAME, config.MODEL_PRETRAINED)
    items, index, basis, basis2d = {}, {}, {}, {}
    for domain in DOMAINS:
        emb = embed_dir(domain, encoder, data_dir)
        items[domain] = load_items(domain, data_dir=data_dir)
        index[domain] = VectorIndex.load(emb)
        basis[domain] = Basis.load(emb / "basis")
        basis2d[domain] = Basis.load(emb / "basis2d")
    return AppContext(
        conn=connect(data_dir / "amber.db"),
        encoder=encoder,
        items=items,
        index=index,
        basis=basis,
        basis2d=basis2d,
        settings=settings,
        data_dir=data_dir,
    )
