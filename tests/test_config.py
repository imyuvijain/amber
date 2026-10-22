import dataclasses
import importlib
from pathlib import Path

import pytest

from amber import config


def test_constants():
    assert config.SEED == 7 and config.PCA_DIM == 32 and config.NEAR_DUP_COS == 0.95
    assert config.KEEP_MASTER_CATEGORIES == ("Apparel", "Footwear", "Accessories")
    assert config.THUMB_SIZE == (216, 288)
    assert config.DEFAULT_SETTINGS == config.Settings(30, 40, 20, 10, 7)


def test_remaining_constants():
    assert config.DEFAULT_SECTIONS == ("Apparel", "Footwear")
    assert config.NEUTRAL_PROMPT == "a photo of a piece of clothing"
    assert config.SEARCH_CANDIDATES == 200
    assert config.SEARCH_RESULTS == 24
    assert config.SEARCH_LAMBDA == 1.0
    assert isinstance(config.MODEL_NAME, str) and config.MODEL_NAME
    assert isinstance(config.MODEL_PRETRAINED, str) and config.MODEL_PRETRAINED


def test_settings_fields_and_frozen():
    s = config.DEFAULT_SETTINGS
    assert (s.art_train_pairs, s.clothes_train_pairs, s.showdown_pairs) == (30, 40, 20)
    assert (s.retest_pairs, s.retest_min_days) == (10, 7)
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.showdown_pairs = 1  # type: ignore[misc]


def test_data_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    try:
        importlib.reload(config)
        assert config.DATA_DIR == tmp_path and config.DB_PATH == tmp_path / "amber.db"
    finally:
        monkeypatch.delenv("AMBER_DATA_DIR", raising=False)
        importlib.reload(config)


def test_data_dir_defaults_to_repo_data(monkeypatch):
    monkeypatch.delenv("AMBER_DATA_DIR", raising=False)
    importlib.reload(config)
    repo = Path(__file__).resolve().parents[1]
    assert (
        config.DATA_DIR == repo / "data"
        and config.DB_PATH == repo / "data" / "amber.db"
    )
