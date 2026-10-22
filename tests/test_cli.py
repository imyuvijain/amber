import os
import re
import subprocess
import sys

import pandas as pd
import pytest
from typer.testing import CliRunner

from amber import cli, config
from amber.cli import app


def test_cli_help_lists_usage():
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0 and "Usage" in result.output


def test_ingest_help_lists_clothes():
    result = CliRunner().invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0 and "clothes" in result.output


def test_ingest_help_lists_art():
    result = CliRunner().invoke(app, ["ingest", "--help"])
    assert result.exit_code == 0 and re.search(r"\bart\b", result.output)


def test_ingest_clothes_streams_into_data_dir_and_prints_count(monkeypatch, tmp_path):
    calls = []

    def fake_ingest(out_dir, rows=None, limit=None):
        calls.append((out_dir, rows, limit))
        return pd.DataFrame({"id": ["c:1", "c:2", "c:3"]})

    monkeypatch.setattr("amber.data.clothes.ingest_clothes", fake_ingest)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # read at call time
    monkeypatch.setattr(cli, "_exit_without_teardown", lambda code=0: None)
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
    monkeypatch.setattr(
        cli, "_exit_without_teardown", lambda code=0: exits.append(code)
    )
    CliRunner().invoke(app, ["ingest", "clothes"])
    assert exits == []
    CliRunner().invoke(app, ["ingest", "clothes", "--limit", "5"])
    assert exits == [0]


@pytest.mark.parametrize("error", [OSError("disk full"), KeyboardInterrupt()])
@pytest.mark.parametrize("args", [[], ["--limit", "5"]])
def test_ingest_clothes_failure_exits_nonzero_without_teardown(
    monkeypatch, tmp_path, error, args
):
    # A network error, disk-full OSError or Ctrl-C abandons the stream just like --limit
    # does, so a failed run must also skip interpreter teardown (with a non-zero code).
    def boom(out_dir, rows=None, limit=None):
        raise error

    monkeypatch.setattr("amber.data.clothes.ingest_clothes", boom)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    exits = []
    monkeypatch.setattr(
        cli, "_exit_without_teardown", lambda code=0: exits.append(code)
    )
    result = CliRunner().invoke(app, ["ingest", "clothes", *args])
    assert exits == [1]
    assert result.exit_code == 1


def _run_python(code, env=None, timeout=120):
    return subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_exit_without_teardown_flushes_output_runs_exit_handlers_and_sets_code():
    # os._exit drops anything still buffered, so check the real thing in a subprocess
    # (stdout/stderr are pipes here, i.e. block-buffered).
    proc = _run_python(
        "import atexit, sys\n"
        "from amber.cli import _exit_without_teardown\n"
        "atexit.register(lambda: print('exit handler ran'))\n"
        "print('buffered stdout'); sys.stderr.write('buffered stderr\\n')\n"
        "_exit_without_teardown(3)\n"
        "print('unreachable')\n"
    )
    assert proc.returncode == 3
    assert "buffered stdout" in proc.stdout and "exit handler ran" in proc.stdout
    assert "unreachable" not in proc.stdout
    assert "buffered stderr" in proc.stderr


@pytest.mark.parametrize("failure", ["OSError('disk full')", "KeyboardInterrupt()"])
def test_ingest_clothes_failed_stream_prints_traceback_and_exits_1(tmp_path, failure):
    # The whole CLI in a subprocess, with the Hub stream replaced by one that yields a
    # row and then fails (no network; AMBER_DATA_DIR keeps it away from the real data dir).
    proc = _run_python(
        "import datasets\n"
        "class Stream:\n"
        "    def cast_column(self, name, feature):\n"
        "        return self\n"
        "    def __iter__(self):\n"
        "        yield {'id': 1, 'masterCategory': 'Personal Care', 'image': None}\n"
        f"        raise {failure}\n"
        "datasets.load_dataset = lambda *a, **k: Stream()\n"
        "from amber.cli import app\n"
        "app(['ingest', 'clothes'])\n"
        "print('unreachable')\n",
        env={"AMBER_DATA_DIR": str(tmp_path)},
    )
    assert proc.returncode == 1, proc.stderr
    assert "Traceback" in proc.stderr and failure.split("(")[0] in proc.stderr
    assert "unreachable" not in proc.stdout


def test_ingest_clothes_rejects_non_positive_limit():
    result = CliRunner().invoke(app, ["ingest", "clothes", "--limit", "0"])
    assert result.exit_code != 0


def test_ingest_art_fetches_into_data_dir_and_prints_count(monkeypatch, tmp_path):
    calls = []

    def fake_ingest(out_dir, n=2000, client=None, min_interval=1.0):
        calls.append((out_dir, n, client, min_interval))
        return pd.DataFrame({"id": ["a:1", "a:2"]})

    monkeypatch.setattr("amber.data.art.ingest_art", fake_ingest)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)  # read at call time
    result = CliRunner().invoke(app, ["ingest", "art", "--n", "40"])
    assert result.exit_code == 0
    assert calls == [(tmp_path / "art", 40, None, 1.0)]  # no client: it makes its own
    assert result.output.split()[0] == "2"

    calls.clear()
    assert CliRunner().invoke(app, ["ingest", "art"]).exit_code == 0
    assert calls == [(tmp_path / "art", 2000, None, 1.0)]  # --n defaults to 2000


def test_ingest_art_failure_exits_nonzero(monkeypatch, tmp_path):
    def boom(out_dir, n=2000, client=None, min_interval=1.0):
        raise OSError("disk full")

    monkeypatch.setattr("amber.data.art.ingest_art", boom)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    result = CliRunner().invoke(app, ["ingest", "art"])
    assert result.exit_code != 0 and isinstance(result.exception, OSError)


def test_ingest_art_rejects_non_positive_n():
    result = CliRunner().invoke(app, ["ingest", "art", "--n", "0"])
    assert result.exit_code != 0 and "Invalid value" in result.output
