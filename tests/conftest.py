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
