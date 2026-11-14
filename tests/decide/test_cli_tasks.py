import json

import pytest
from typer.testing import CliRunner

from amber import cli, config
from amber.cli import app

from .test_cli_sources import PRIVATE


@pytest.fixture
def data_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(cli, "TASKS_RESULTS_PATH", tmp_path / "published.md")
    return tmp_path


def _sources(data_dir):
    r = CliRunner()
    assert r.invoke(app, ["decide", "fake-sources", str(data_dir / "exports")]).exit_code == 0
    assert r.invoke(app, ["decide", "sources"]).exit_code == 0


def test_tasks_needs_event_log(data_dir):
    res = CliRunner().invoke(app, ["decide", "tasks"])
    assert res.exit_code != 0
    assert "decide sources" in res.output


def test_tasks_on_fake_sources_prints_aggregates_only(data_dir):
    _sources(data_dir)
    res = CliRunner().invoke(app, ["decide", "tasks"])
    assert res.exit_code == 0, res.output
    assert "life context: off" in res.output
    out = (data_dir / "decide" / "tasks" / "public.md").read_text()
    for s in PRIVATE:
        assert s.lower() not in res.output.lower()
        assert s.lower() not in out.lower()
    results = json.loads((data_dir / "decide" / "tasks" / "results.json").read_text())
    assert {r["task"] for r in results} == {
        "yt_channel_return", "ig_creator_return", "reels_feedback"
    }


def test_tasks_publish_refuses_synthetic(data_dir):
    _sources(data_dir)
    res = CliRunner().invoke(app, ["decide", "tasks", "--publish"])
    assert res.exit_code != 0
    assert "synthetic" in res.output
    assert not (data_dir / "published.md").exists()
