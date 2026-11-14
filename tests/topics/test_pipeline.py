import numpy as np
import pandas as pd
import pytest
from decide.test_cli_sources import PRIVATE
from typer.testing import CliRunner

from amber import config
from amber.cli import app
from amber.topics.embed import CachedEncoder, HashingEncoder


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    r = CliRunner()
    assert r.invoke(app, ["decide", "fake-sources", str(tmp_path / "exports")]).exit_code == 0
    assert r.invoke(app, ["decide", "sources"]).exit_code == 0
    return tmp_path


def test_hashing_encoder_is_unit_and_deterministic():
    e = HashingEncoder()
    a = e.embed_texts(["bread recipe", "bread recipe", "quantum physics"])
    assert np.allclose(np.linalg.norm(a, axis=1), 1)
    assert np.allclose(a[0], a[1]) and a[0] @ a[2] < 0.5


def test_cached_encoder_only_embeds_new_texts(tmp_path):
    calls = []

    class Spy(HashingEncoder):
        def embed_texts(self, texts):
            calls.append(list(texts))
            return super().embed_texts(texts)

    enc = CachedEncoder(Spy(), tmp_path / "cache.npz")
    first = enc.embed_texts(["a b", "c d"])
    enc.save()
    enc2 = CachedEncoder(Spy(), tmp_path / "cache.npz")
    again = enc2.embed_texts(["c d", "e f"])
    assert calls[-1] == ["e f"]
    assert np.allclose(first[1], again[0])


def test_topics_build_writes_private_outputs_and_prints_counts_only(data_dir):
    res = CliRunner().invoke(app, ["topics", "build", "--encoder", "hash", "--min-docs", "3"])
    assert res.exit_code == 0, res.output
    out = data_dir / "topics"
    topics = pd.read_parquet(out / "topics.parquet")
    assert len(topics) >= 1
    assert {"name", "relevance", "steady", "drivers"} <= set(topics.columns)
    for name in topics["name"]:
        assert name not in res.output
    for s in PRIVATE:
        assert s.lower() not in res.output.lower()
    assert "topics:" in res.output


def test_topics_build_needs_event_log(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    res = CliRunner().invoke(app, ["topics", "build", "--encoder", "hash"])
    assert res.exit_code != 0 and "decide sources" in res.output
