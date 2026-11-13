import pandas as pd
import pytest
from typer.testing import CliRunner

from amber import cli, config
from amber.cli import app


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # read at call time
    return tmp_path


def test_decide_help_lists_commands():
    result = CliRunner().invoke(app, ["decide", "--help"])
    assert result.exit_code == 0
    for name in ("fake", "ingest", "eval"):
        assert name in result.output


def test_cli_fake_ingest_eval(data_dir):
    runner = CliRunner()
    export = data_dir / "exports" / "spotify"
    r = runner.invoke(
        app, ["decide", "fake", str(export), "--kind", "personal", "--n-plays", "1500"]
    )
    assert r.exit_code == 0, r.output
    assert any(export.glob("Streaming_History_Audio_*.json"))

    r = runner.invoke(app, ["decide", "ingest"])
    assert r.exit_code == 0, r.output
    assert "plays" in r.output and "2024-" in r.output
    parquet = data_dir / "decide" / "decisions.parquet"
    assert len(pd.read_parquet(parquet)) > 1000

    r = runner.invoke(app, ["decide", "eval", "--tz", "Europe/Berlin"])
    assert r.exit_code == 0, r.output
    assert "Remembering you" in r.output
    assert (data_dir / "decide" / "public.md").exists()
    assert (data_dir / "decide" / "report.md").exists()


def test_eval_publish_copies_public_report(data_dir, monkeypatch):
    target = data_dir / "published" / "spotify-v1.md"
    monkeypatch.setattr(cli, "PUBLIC_RESULTS_PATH", target)
    runner = CliRunner()
    export = data_dir / "exports" / "spotify"
    assert runner.invoke(
        app, ["decide", "fake", str(export), "--kind", "null", "--n-plays", "800"]
    ).exit_code == 0
    r = runner.invoke(app, ["decide", "eval", "--publish"])
    assert r.exit_code == 0, r.output
    assert target.read_text() == (data_dir / "decide" / "public.md").read_text()


def test_fake_account_format(data_dir):
    dest = data_dir / "acct"
    r = CliRunner().invoke(
        app, ["decide", "fake", str(dest), "--fmt", "account", "--n-plays", "300"]
    )
    assert r.exit_code == 0, r.output
    assert any(dest.glob("StreamingHistory_music_*.json"))


def test_missing_export_is_a_clean_error(data_dir):
    for cmd in ("ingest", "eval"):
        r = CliRunner().invoke(app, ["decide", cmd, str(data_dir / "nothing")])
        assert r.exit_code == 1
        assert "no Spotify streaming history files" in r.output
