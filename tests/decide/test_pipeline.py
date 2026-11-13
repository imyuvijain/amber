import copy
import json
import re

import pandas as pd
import pytest

from amber.decide.fake import write_fake_export
from amber.decide.pipeline import MIN_PLAYS, run_pipeline
from amber.decide.report import headline, render_private, render_public
from amber.decide.spotify import load_export

N_PLAYS = 5000


def _gap(result, lower, higher):
    return next(g for g in result["gaps"] if g["lower"] == lower and g["higher"] == higher)


@pytest.fixture(scope="module")
def fakes(tmp_path_factory):
    """Run the whole pipeline once per fake user and share the outputs."""
    out = {}
    for kind in ("personal", "null"):
        root = tmp_path_factory.mktemp(kind)
        export = write_fake_export(root / "export", kind, n_plays=N_PLAYS)
        out_dir = root / "out"
        result = run_pipeline(export, out_dir)
        out[kind] = {"export": export, "out": out_dir, "result": result}
    return out


def test_personal_user_memory_beats_context(fakes):
    gap = _gap(fakes["personal"]["result"], "context", "memory")
    assert gap["ci"][0] > 0, gap


def test_null_user_nothing_beats_base(fakes):
    for gap in fakes["null"]["result"]["gaps"]:
        ci_contains_zero = gap["ci"][0] <= 0 <= gap["ci"][1]
        assert ci_contains_zero or gap["ll_gain"] < 0.01, gap


def test_pipeline_writes_outputs_and_uses_the_frozen_holdout(fakes):
    run = fakes["personal"]
    for name in ("decisions.parquet", "report.md", "public.md"):
        assert (run["out"] / name).exists()
    result = run["result"]
    assert result["n_holdout"] < result["n_total"]
    assert abs(result["n_holdout"] / result["n_total"] - 0.2) < 0.01
    decisions = pd.read_parquet(run["out"] / "decisions.parquet")
    assert len(decisions) == result["n_total"]
    assert {"p_base", "p_context", "p_memory", "p_recency", "skip"} <= set(decisions)


def test_public_report_has_no_names(fakes):
    run = fakes["personal"]
    public = (run["out"] / "public.md").read_text()
    names, ips = set(), set()
    for path in run["export"].glob("*.json"):
        for row in json.loads(path.read_text()):
            names.update(
                row[k]
                for k in (
                    "master_metadata_track_name",
                    "master_metadata_album_artist_name",
                    "master_metadata_album_album_name",
                    "spotify_track_uri",
                )
                if row.get(k)
            )
            ips.add(row["ip_addr"])
    assert names and ips
    for s in names | ips:
        assert s not in public
    assert "spotify:track:" not in public and "Fake" not in public
    assert not re.search(r"\d{4}-\d{2}-\d{2}", public)  # months only
    assert re.search(r"\d{4}-\d{2}", public)


def test_private_report_lists_artists_and_stays_out_of_public(fakes):
    run = fakes["personal"]
    private = (run["out"] / "report.md").read_text()
    assert "Fake Artist" in private
    assert "most skipped" in private.lower() and "most kept" in private.lower()
    assert "Fake Artist" not in (run["out"] / "public.md").read_text()


def test_headline_wording(fakes):
    base = fakes["personal"]["result"]
    period = ("2024-01", "2025-12")
    for lo, word in ((0.01, "beats"), (-0.01, "does not beat"), (0.0, "does not beat")):
        result = copy.deepcopy(base)
        _gap(result, "context", "memory")["ci"] = (lo, 0.2)
        line = headline(result)
        assert line.startswith(f"Remembering you {word} context-only listening habits")
        assert ("Remembering you beats" in line) == (lo > 0)
        assert line in render_public(result, period)


def test_headline_template_values(fakes):
    result = fakes["personal"]["result"]
    gap = _gap(result, "context", "memory")
    expected = (
        f"Remembering you beats context-only listening habits: log-loss gain "
        f"{gap['ll_gain']:.3f} (95% CI {gap['ci'][0]:.3f} to {gap['ci'][1]:.3f}) "
        f"on {result['n_holdout']} held-out plays, base skip rate "
        f"{result['base_rate_holdout']:.0%}."
    )
    assert headline(result) == expected


def test_nan_renders_as_na(fakes):
    result = copy.deepcopy(fakes["null"]["result"])
    for r in result["rungs"].values():
        r["auc"] = float("nan")
    _gap(result, "context", "memory")["ci"] = (float("nan"), float("nan"))
    text = render_public(result, ("2024-01", "2025-12"))
    assert "n/a" in text and "nan" not in text.lower()
    assert "does not beat" in text


def test_render_private_caps_artist_lists(fakes):
    result = fakes["personal"]["result"]
    df = load_export(fakes["personal"]["export"])
    text = render_private(result, df)
    assert text.count("Fake Artist") <= 20


def test_too_few_plays_is_a_clear_error(tmp_path):
    export = write_fake_export(tmp_path / "e", "personal", n_plays=MIN_PLAYS - 20)
    with pytest.raises(ValueError, match="at least"):
        run_pipeline(export, tmp_path / "out")


def test_empty_export_is_a_clear_error(tmp_path):
    export = tmp_path / "e"
    export.mkdir()
    (export / "Streaming_History_Audio_x.json").write_text("not json")
    with pytest.raises(ValueError, match="no plays"):
        run_pipeline(export, tmp_path / "out")


def test_bad_timezone_is_a_value_error(tmp_path):
    export = write_fake_export(tmp_path / "e", "null", n_plays=200)
    with pytest.raises(ValueError, match="timezone"):
        run_pipeline(export, tmp_path / "out", tz="Mars/Olympus")
