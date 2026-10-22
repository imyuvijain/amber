"""Shared pytest fixtures; the fakes themselves live in `fakes.py`."""

from pathlib import Path

import pytest
from fakes import FakeEncoder, build_fixture_data


@pytest.fixture
def fake_encoder() -> FakeEncoder:
    return FakeEncoder(dim=16)


@pytest.fixture
def fixture_data(tmp_path: Path) -> Path:
    """A fresh fixture data dir (clothes + art catalogs with thumbnails) under tmp_path."""
    return build_fixture_data(tmp_path / "data")


@pytest.fixture
def ctx(fixture_data: Path):
    """An `AppContext` over the fixture catalogs, FakeEncoder embeddings and an in-memory DB."""
    from amber.config import Settings
    from amber.context import load_context
    from amber.space.index import embed_domain
    from amber.store.db import connect

    encoder = FakeEncoder()
    for domain in ("clothes", "art"):
        embed_domain(domain, encoder, data_dir=fixture_data)
    context = load_context(fixture_data, Settings(6, 40, 2, 7), encoder)
    context.conn.close()  # load_context opened <data_dir>/amber.db; tests use memory
    context.conn = connect(":memory:")
    yield context
    context.conn.close()


@pytest.fixture
def user(ctx) -> str:
    from amber.store.db import create_user

    return create_user(ctx.conn, "test")
