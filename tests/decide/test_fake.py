import json

import pandas as pd
import pytest

from amber.decide.fake import write_fake_export
from amber.decide.spotify import DECISION_COLUMNS, load_export


@pytest.mark.parametrize("fmt", ["extended", "account"])
def test_round_trip_both_formats(tmp_path, fmt):
    out = write_fake_export(tmp_path, "personal", n_plays=1000, fmt=fmt)
    assert out == tmp_path
    df = load_export(out)
    assert 950 <= len(df) <= 1000
    assert list(df.columns) == DECISION_COLUMNS


def test_null_skip_rate(tmp_path):
    df = load_export(write_fake_export(tmp_path, "null", n_plays=4000))
    assert abs(df["skip"].mean() - 0.30) < 0.03


def test_personal_has_artist_signal(tmp_path):
    df = load_export(write_fake_export(tmp_path, "personal", n_plays=4000))
    stats = df.groupby("artist_id")["skip"].agg(["mean", "size"])
    assert (stats["size"] >= 30).sum() >= 30  # artists are played in balance
    rates = stats[stats["size"] >= 30]["mean"]
    assert rates.max() - rates.min() > 0.6


def test_null_has_no_artist_signal(tmp_path):
    df = load_export(write_fake_export(tmp_path, "null", n_plays=4000))
    stats = df.groupby("artist_id")["skip"].agg(["mean", "size"])
    rates = stats[stats["size"] >= 30]["mean"]
    assert rates.max() - rates.min() < 0.3


def test_deterministic(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        write_fake_export(d, "personal", n_plays=500, seed=3)
    files_a = sorted(p.name for p in a.iterdir())
    assert files_a == sorted(p.name for p in b.iterdir())
    for name in files_a:
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_different_seed_differs(tmp_path):
    write_fake_export(tmp_path / "a", "personal", n_plays=500, seed=1)
    write_fake_export(tmp_path / "b", "personal", n_plays=500, seed=2)
    name = "Streaming_History_Audio_2024-2025_0.json"
    assert (tmp_path / "a" / name).read_bytes() != (tmp_path / "b" / name).read_bytes()


def test_sessions_and_extended_raw_fields(tmp_path):
    write_fake_export(tmp_path, "personal", n_plays=4000)
    rows = []
    for f in sorted(tmp_path.glob("Streaming_History_Audio_*.json")):
        rows += json.loads(f.read_text())
    assert len(rows) == 4000
    assert all("ip_addr" in r and "conn_country" in r for r in rows)
    podcasts = [r for r in rows if r["spotify_episode_uri"] is not None]
    assert 0.005 < len(podcasts) / len(rows) < 0.04
    assert all(r["master_metadata_track_name"] is None for r in podcasts)
    assert 0.002 < sum(r["incognito_mode"] for r in rows) / len(rows) < 0.03
    assert {r["platform"] for r in rows} == {"android", "ios", "windows", "web_player"}
    for r in rows:
        if r["reason_end"] == "fwdbtn":
            assert r["ms_played"] < 30_000

    ts = sorted(r["ts"] for r in rows)  # ISO strings sort chronologically
    gaps = pd.Series(pd.to_datetime(ts, utc=True)).diff().dropna().dt.total_seconds()
    assert ((gaps < 600) | (gaps >= 7200)).all()  # within a session vs between
    sizes = (gaps >= 7200).sum() + 1
    assert 4000 / 40 <= sizes <= 4000 / 5 + 1


def test_account_format_files_and_fields(tmp_path):
    write_fake_export(tmp_path, "null", n_plays=300, fmt="account")
    files = sorted(tmp_path.glob("StreamingHistory_music_*.json"))
    assert files
    assert not list(tmp_path.glob("Streaming_History_Audio_*.json"))
    row = json.loads(files[0].read_text())[0]
    assert set(row) == {"endTime", "artistName", "trackName", "msPlayed"}


def test_fake_exports_are_detectably_synthetic(tmp_path):
    from amber.decide.fake import is_synthetic
    from amber.decide.spotify import load_export

    for fmt in ("extended", "account"):
        dest = write_fake_export(tmp_path / fmt, "null", n_plays=200, fmt=fmt)
        assert is_synthetic(load_export(dest))


def test_realish_data_is_not_synthetic(tmp_path, realish):
    from amber.decide.fake import is_synthetic
    from amber.decide.spotify import load_export

    fake = write_fake_export(tmp_path / "fake", "null", n_plays=200)
    assert not is_synthetic(load_export(realish(fake, tmp_path / "real")))
