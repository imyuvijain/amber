import pandas as pd
import pytest
from typer.testing import CliRunner

from amber import config
from amber.cli import app
from amber.decide.events import EVENT_COLUMNS

PRIVATE = [
    "Fake Video",
    "Fake Channel",
    "fake_user",
    "fake_us",
    "fake query",
    "fake yt query",
    "fakevid",
    "example.com",
    "instagram.com",
    "youtube.com",
    "UCfake",
    "Fake Page",
    "fake prompt",
    "fake message",
]


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # read at call time
    return tmp_path


def test_help_lists_new_commands():
    out = CliRunner().invoke(app, ["decide", "--help"]).output
    assert "sources" in out and "fake-sources" in out


@pytest.mark.parametrize("fmt", ["html", "json"])
def test_fake_sources_then_sources_default_dir(data_dir, fmt):
    runner = CliRunner()
    exports = data_dir / "exports"
    r = runner.invoke(app, ["decide", "fake-sources", str(exports), "--fmt", fmt])
    assert r.exit_code == 0, r.output
    assert any((exports / "google").rglob(f"MyActivity.{fmt}"))
    assert any((exports / "instagram").rglob("liked_posts.json"))

    r = runner.invoke(app, ["decide", "sources"])
    assert r.exit_code == 0, r.output
    for word in ("youtube", "instagram", "watch", "active months", "skipped files", "bad rows"):
        assert word in r.output
    for needle in PRIVATE:
        assert needle not in r.output
    df = pd.read_parquet(data_dir / "decide" / "events.parquet")
    assert list(df.columns) == EVENT_COLUMNS
    assert len(df) > 500


def test_fake_sources_defaults_to_html(data_dir):
    dest = data_dir / "x"
    r = CliRunner().invoke(app, ["decide", "fake-sources", str(dest)])
    assert r.exit_code == 0, r.output
    assert any(dest.rglob("MyActivity.html")) and not any(dest.rglob("MyActivity.json"))


def test_sources_explicit_dir(data_dir):
    runner = CliRunner()
    dest = data_dir / "elsewhere"
    assert runner.invoke(app, ["decide", "fake-sources", str(dest)]).exit_code == 0
    r = runner.invoke(app, ["decide", "sources", str(dest)])
    assert r.exit_code == 0, r.output
    assert (data_dir / "decide" / "events.parquet").exists()


def test_sources_fails_clearly_without_events(data_dir):
    (data_dir / "exports").mkdir()
    r = CliRunner().invoke(app, ["decide", "sources"])
    assert r.exit_code != 0
    assert "no events" in r.output
    assert not (data_dir / "decide" / "events.parquet").exists()
