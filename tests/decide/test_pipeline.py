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


def test_null_user_context_has_no_discrimination(fakes):
    """Guard against duration leaking into context features (AUC ~0.5 expected)."""
    auc = fakes["null"]["result"]["rungs"]["context"]["auc"]
    assert auc < 0.6, auc


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
    assert "skip most" in private and "keep most" in private
    assert "Fake Artist" not in (run["out"] / "public.md").read_text()


def _set(result, memory_ci_lo, base_ci_lo):
    result = copy.deepcopy(result)
    _gap(result, "context", "memory")["ci"] = (memory_ci_lo, 0.2)
    result["memory_vs_base"]["ci"] = (base_ci_lo, 0.3)
    return result


def test_headline_wording(fakes):
    base = fakes["personal"]["result"]
    period = ("2024-01", "2025-12")
    cases = (
        (0.01, 0.02, "beats"),
        (-0.01, 0.02, "does not beat"),
        (0.0, 0.02, "does not beat"),
        # memory beats context but memory does not beat base: still not a win
        (0.01, -0.02, "does not beat"),
        (0.01, 0.0, "does not beat"),
    )
    for ctx_lo, base_lo, word in cases:
        result = _set(base, ctx_lo, base_lo)
        line = headline(result)
        assert line.startswith(f"Remembering you {word} context-only listening habits")
        assert ("Remembering you beats" in line) == (word == "beats")
        assert line in render_public(result, period)


def test_headline_template_values(fakes):
    result = fakes["personal"]["result"]
    gap = _gap(result, "context", "memory")
    mvb = result["memory_vs_base"]
    verb = "beats" if gap["ci"][0] > 0 and mvb["ci"][0] > 0 else "does not beat"
    expected = (
        f"Remembering you {verb} context-only listening habits: log-loss gain "
        f"{gap['ll_gain']:.3f} (95% CI {gap['ci'][0]:.3f} to {gap['ci'][1]:.3f}) "
        f"on {result['n_holdout']} held-out plays, base skip rate "
        f"{result['base_rate_holdout']:.0%}; memory versus base-rate-only gain "
        f"{mvb['ll_gain']:.3f} (95% CI {mvb['ci'][0]:.3f} to {mvb['ci'][1]:.3f})."
    )
    assert headline(result) == expected


def test_nan_renders_as_na(fakes):
    result = copy.deepcopy(fakes["null"]["result"])
    for r in result["rungs"].values():
        r["auc"] = float("nan")
    _gap(result, "context", "memory")["ci"] = (float("nan"), float("nan"))
    result["memory_vs_base"]["ci"] = (float("nan"), float("nan"))
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


def test_reports_state_the_pause_limitation(fakes):
    for name in ("public.md", "report.md"):
        text = (fakes["personal"]["out"] / name).read_text()
        assert "ms_played excludes pauses" in text


def test_period_uses_local_months():
    from amber.decide.report import period

    df = pd.DataFrame({"ts": pd.to_datetime(["2024-01-31 23:30", "2024-03-01 00:30"], utc=True)})
    assert period(df) == ("2024-01", "2024-03")
    assert period(df, "Europe/Berlin") == ("2024-02", "2024-03")


def test_result_flags_synthetic_and_counts(fakes):
    result = fakes["personal"]["result"]
    assert result["synthetic"] is True
    assert result["source"] == "spotify_extended"
    assert isinstance(result["n_near_duplicates"], int)
    assert isinstance(result["n_overlap"], int)
    assert result["n_overlap"] == 0  # the fake never overlaps itself


def test_public_report_names_its_source(fakes):
    for name in ("public.md", "report.md"):
        text = (fakes["personal"]["out"] / name).read_text()
        assert "spotify_extended" in text
        assert "forward button" in text  # the label rule, in words
        assert "synthetic" in text.lower()


def test_report_shows_duplicate_and_overlap_counts(fakes):
    result = copy.deepcopy(fakes["personal"]["result"])
    result["n_near_duplicates"], result["n_overlap"] = 12, 7
    period = ("2024-01", "2025-12")
    public = render_public(result, period)
    assert "12 near-duplicate" in public and "7 plays overlap" in public
    df = load_export(fakes["personal"]["export"])
    private = render_private(result, df)
    assert "12 near-duplicate" in private and "7 plays overlap" in private


def test_account_report_names_its_label_rule(tmp_path):
    export = write_fake_export(tmp_path / "e", "null", n_plays=400, fmt="account")
    result = run_pipeline(export, tmp_path / "out")
    text = (tmp_path / "out" / "public.md").read_text()
    assert result["source"] == "spotify_account"
    assert "spotify_account" in text and "30 seconds" in text


def test_real_looking_data_is_not_flagged_synthetic(tmp_path, realish):
    fake = write_fake_export(tmp_path / "fake", "null", n_plays=400)
    result = run_pipeline(realish(fake, tmp_path / "real"), tmp_path / "out")
    assert result["synthetic"] is False
    assert "synthetic" not in (tmp_path / "out" / "public.md").read_text().lower()
