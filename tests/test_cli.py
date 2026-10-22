import pandas as pd
from typer.testing import CliRunner

from amber import cli, config
from amber.cli import app


def test_cli_help_lists_usage():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "Usage" in result.output


def test_ingest_help_lists_clothes():
    result = CliRunner().invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0 and "clothes" in result.output


def test_ingest_clothes_streams_into_data_dir_and_prints_count(monkeypatch, tmp_path):
    calls = []

    def fake_ingest(out_dir, rows=None, limit=None):
        calls.append((out_dir, rows, limit))
        return pd.DataFrame({"id": ["c:1", "c:2", "c:3"]})

    monkeypatch.setattr("amber.data.clothes.ingest_clothes", fake_ingest)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # read at call time
    monkeypatch.setattr(cli, "_exit_without_teardown", lambda: None)
    result = CliRunner().invoke(app, ["ingest", "clothes", "--limit", "7"])
    assert result.exit_code == 0
    assert calls == [(tmp_path / "clothes", None, 7)]  # rows=None means: stream
    assert result.output.split()[0] == "3"

    calls.clear()
    assert CliRunner().invoke(app, ["ingest", "clothes"]).exit_code == 0
    assert calls == [(tmp_path / "clothes", None, None)]


def test_ingest_clothes_skips_interpreter_teardown_only_for_limited_runs(
    monkeypatch, tmp_path
):
    # An abandoned pyarrow/datasets stream deadlocks the process at interpreter exit, so
    # a --limit run (which stops mid-stream) exits explicitly; a full run ends normally.
    monkeypatch.setattr(
        "amber.data.clothes.ingest_clothes",
        lambda out_dir, rows=None, limit=None: pd.DataFrame({"id": []}),
    )
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    exits = []
    monkeypatch.setattr(cli, "_exit_without_teardown", lambda: exits.append(1))
    CliRunner().invoke(app, ["ingest", "clothes"])
    assert exits == []
    CliRunner().invoke(app, ["ingest", "clothes", "--limit", "5"])
    assert exits == [1]


def test_ingest_clothes_rejects_non_positive_limit():
    result = CliRunner().invoke(app, ["ingest", "clothes", "--limit", "0"])
    assert result.exit_code != 0
