import time

import numpy as np
import pandas as pd
import pytest

from amber.decide.fake import write_fake_export
from amber.decide.features import (
    CONTEXT_COLS,
    DECAY_COLS,
    MEMORY_COLS,
    build_features,
)
from amber.decide.spotify import DECISION_COLUMNS, load_export

ALL_COLS = CONTEXT_COLS + MEMORY_COLS + DECAY_COLS


def _rows(specs):
    """specs: (ts string, artist, item, skip[, seconds played]) -> a decision frame.

    ``ts`` is the end time; ``start_ts`` is ``ts`` minus the seconds played
    (default 0, so a play starts and ends at ``ts``).
    """
    n = len(specs)
    ts = pd.to_datetime([s[0] for s in specs], utc=True)
    played = pd.to_timedelta([s[4] if len(s) > 4 else 0 for s in specs], unit="s")
    return pd.DataFrame(
        {
            "ts": ts,
            "start_ts": ts - played,
            "source": "spotify_extended",
            "item_id": [s[2] for s in specs],
            "artist_id": [s[1] for s in specs],
            "album_id": "alb",
            "platform": "Android OS 12",
            "shuffle": pd.array([True] * n, dtype="boolean"),
            "reason_start": "trackdone",
            "offline": pd.array([False] * n, dtype="boolean"),
            "skip": [s[3] for s in specs],
        }
    )[DECISION_COLUMNS]


@pytest.fixture(scope="module")
def personal(tmp_path_factory):
    path = write_fake_export(tmp_path_factory.mktemp("fake"), "personal", n_plays=2000)
    return load_export(path)


def test_shape_index_and_finite(personal):
    feats = build_features(personal)
    assert list(feats.columns) == ALL_COLS
    assert feats.index.equals(personal.index)
    assert np.isfinite(feats.to_numpy(dtype=float)).all()


def test_no_future_leakage(personal):
    flipped = personal.copy()
    flipped.loc[flipped.index >= 500, "skip"] = ~flipped.loc[
        flipped.index >= 500, "skip"
    ]
    a = build_features(personal)
    b = build_features(flipped)
    pd.testing.assert_frame_equal(a.iloc[:501], b.iloc[:501])
    same_ts = personal["ts"] == personal["ts"].iloc[500]
    pd.testing.assert_frame_equal(a[same_ts], b[same_ts])


def test_same_ts_invisible():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", False),
            ("2024-01-01 11:00", "a", "t2", True),
            ("2024-01-01 11:00", "a", "t3", False),
        ]
    )
    feats = build_features(df)
    assert feats.loc[1, "artist_skip_s"] == feats.loc[2, "artist_skip_s"]
    assert feats.loc[1, "artist_dskip_s"] == feats.loc[2, "artist_dskip_s"]
    assert feats.loc[1, "prev_skip"] == feats.loc[2, "prev_skip"]
    assert feats.loc[1, "session_pos_log"] == feats.loc[2, "session_pos_log"]


def test_cold_start_finite():
    feats = build_features(_rows([("2024-01-01 10:00", "a", "t1", True)]))
    assert np.isfinite(feats.to_numpy(dtype=float)).all()
    assert feats.loc[0, "artist_skip_s"] == 0.5
    assert feats.loc[0, "track_skip_s"] == 0.5
    assert feats.loc[0, "artist_days_log"] == pytest.approx(np.log1p(365))
    assert feats.loc[0, "session_pos_log"] == 0
    assert feats.loc[0, "prev_skip"] == 0


def test_timezone_shifts_hour():
    df = _rows([("2024-01-01 20:00", "a", "t1", False)])
    utc = build_features(df, tz="UTC").loc[0]
    ist = build_features(df, tz="Asia/Kolkata").loc[0]
    assert utc["hour_sin"] != ist["hour_sin"]
    hour = np.arctan2(ist["hour_sin"], ist["hour_cos"]) * 24 / (2 * np.pi)
    assert hour == pytest.approx(1.5)
    # 2024-01-01 is a Monday in UTC but Tuesday in Kolkata: weekend uses local date.
    sat = _rows([("2024-01-06 20:00", "a", "t1", False)])  # Saturday UTC, Sunday IST
    assert build_features(sat, tz="Pacific/Auckland").loc[0, "weekend"] == 1
    fri = _rows([("2024-01-05 20:00", "a", "t1", False)])  # Friday UTC, Saturday IST
    assert build_features(fri, tz="UTC").loc[0, "weekend"] == 0
    assert build_features(fri, tz="Asia/Kolkata").loc[0, "weekend"] == 1


def test_session_reset():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", True),
            ("2024-01-01 10:03", "a", "t2", True),
            ("2024-01-01 10:06", "a", "t3", False),
            ("2024-01-01 10:37", "a", "t4", True),  # 31 minute gap
            ("2024-01-01 10:40", "a", "t5", True),
        ]
    )
    feats = build_features(df)
    assert feats.loc[2, "session_pos_log"] == pytest.approx(np.log1p(2))
    assert feats.loc[2, "prev_skip"] == 1
    assert feats.loc[3, "session_pos_log"] == 0
    assert feats.loc[3, "prev_skip"] == 0
    assert feats.loc[3, "prev_gap_min_log"] == pytest.approx(np.log1p(3))  # t3 after t2
    assert feats.loc[4, "prev_gap_min_log"] == pytest.approx(np.log1p(31))  # t4 after t3
    assert feats.loc[4, "session_pos_log"] == pytest.approx(np.log1p(1))
    assert feats.loc[4, "prev_skip"] == 1


def test_prev_gap_capped_and_zero_without_history():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", True),
            ("2024-02-01 10:00", "a", "t2", True),
            ("2024-02-01 10:05", "a", "t3", True),
        ]
    )
    f = build_features(df)
    assert f.loc[0, "prev_gap_min_log"] == 0  # no previous play
    assert f.loc[1, "prev_gap_min_log"] == 0  # previous play has no predecessor
    assert f.loc[2, "prev_gap_min_log"] == pytest.approx(np.log1p(1440))


def test_prev_gap_is_shared_within_a_ts_group():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", True),
            ("2024-01-01 10:07", "a", "t2", True),
            ("2024-01-01 10:10", "a", "t3", True),
            ("2024-01-01 10:10", "a", "t4", False),
        ]
    )
    f = build_features(df)
    assert f.loc[2, "prev_gap_min_log"] == f.loc[3, "prev_gap_min_log"]
    assert f.loc[2, "prev_gap_min_log"] == pytest.approx(np.log1p(7))


def test_gap_excludes_own_duration():
    """ts is the end time: a long play must not look different from a short one."""
    short = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", False),
            ("2024-01-01 10:10", "a", "t2", True, 5),  # started 10:09:55
        ]
    )
    long = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", False),
            ("2024-01-01 10:10", "a", "t2", False, 5),
        ]
    )
    long.loc[1, "ts"] = pd.Timestamp("2024-01-01 10:13", tz="UTC")
    long.loc[1, "start_ts"] = short.loc[1, "start_ts"]  # same start, longer play
    a, b = build_features(short), build_features(long)
    pd.testing.assert_series_equal(a.loc[1], b.loc[1])
    # a third play sees the second one's start, never its end
    for df in (short, long):
        df.loc[2] = df.loc[1]
        df.loc[2, ["ts", "start_ts"]] = pd.Timestamp("2024-01-01 11:00", tz="UTC")
    a, b = build_features(short), build_features(long)
    assert a.loc[2, "prev_gap_min_log"] == b.loc[2, "prev_gap_min_log"]
    assert a.loc[2, "prev_gap_min_log"] == pytest.approx(np.log1p(9 + 55 / 60))


def test_hour_and_session_use_start_time():
    df = _rows(
        [
            ("2024-01-01 10:00:00", "a", "t1", False),
            ("2024-01-01 11:00:10", "a", "t2", True, 20 * 60),  # started 10:40:10
        ]
    )
    f = build_features(df)
    assert f.loc[1, "session_pos_log"] == 0  # 40 min from the last end: new session
    hour = np.arctan2(f.loc[1, "hour_sin"], f.loc[1, "hour_cos"]) * 24 / (2 * np.pi)
    assert hour == pytest.approx(10 + 40 / 60)  # minute resolution


def test_overlapping_play_has_zero_gap():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", False),
            ("2024-01-01 10:01", "a", "t2", True, 300),  # started before t1 ended
        ]
    )
    f = build_features(df)
    assert f.loc[1, "artist_days_log"] == 0
    assert np.isfinite(f.to_numpy(dtype=float)).all()


def test_missing_start_ts_is_an_error(personal):
    with pytest.raises(ValueError, match="start_ts"):
        build_features(personal.drop(columns="start_ts"))


def test_memory_smoothing_and_decay():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", True),
            ("2024-01-31 10:00", "a", "t1", False),  # 30 days later: one half-life
        ]
    )
    f = build_features(df)
    g = (1 + 1) / (1 + 2)  # one earlier row, one skip
    assert f.loc[1, "artist_skip_s"] == pytest.approx((1 + 5 * g) / (1 + 5))
    assert f.loc[1, "track_skip_s"] == pytest.approx((1 + 3 * g) / (1 + 3))
    assert f.loc[1, "artist_n_log"] == pytest.approx(np.log1p(1))
    assert f.loc[1, "artist_days_log"] == pytest.approx(np.log1p(30))
    assert f.loc[1, "artist_dskip_s"] == pytest.approx((0.5 + 5 * g) / (0.5 + 5))
    assert f.loc[1, "artist_dn_log"] == pytest.approx(np.log1p(0.5))
    longer = build_features(df, half_life_days=60.0)
    assert longer.loc[1, "artist_dn_log"] == pytest.approx(np.log1p(0.5**0.5))


def test_platform_shuffle_and_reason_encoding():
    df = _rows([("2024-01-01 10:00", "a", "t1", False)] * 1)
    df = pd.concat([df] * 6, ignore_index=True)
    df["ts"] = df["ts"] + pd.to_timedelta(range(6), unit="s")
    df["platform"] = [
        "Android OS 12",
        "iOS 17",
        "Windows 10",
        "web_player",
        "Sonos",
        None,
    ]
    df["reason_start"] = ["clickrow", "trackdone", "fwdbtn", "playbtn", "appload", None]
    df["shuffle"] = pd.array([True, False, pd.NA, True, True, True], dtype="boolean")
    f = build_features(df)
    plat = f[[c for c in CONTEXT_COLS if c.startswith("platform_")]]
    assert plat.sum(axis=1).tolist() == [1] * 6
    assert plat.idxmax(axis=1).tolist() == [
        "platform_android",
        "platform_ios",
        "platform_desktop",
        "platform_web",
        "platform_other",
        "platform_other",
    ]
    rs = f[[c for c in CONTEXT_COLS if c.startswith("rs_")]]
    assert rs.idxmax(axis=1).tolist()[:5] == [
        "rs_clickrow",
        "rs_trackdone",
        "rs_fwdbtn",
        "rs_playbtn",
        "rs_other",
    ]
    assert rs.iloc[5].sum() == 0
    assert f["shuffle"].tolist() == [1, 0, 0, 1, 1, 1]
    assert f["shuffle_known"].tolist() == [1, 1, 0, 1, 1, 1]


def test_account_format_is_finite(tmp_path):
    df = load_export(
        write_fake_export(tmp_path, "personal", n_plays=500, fmt="account")
    )
    feats = build_features(df)
    assert np.isfinite(feats.to_numpy(dtype=float)).all()
    assert (feats["shuffle_known"] == 0).all()


def test_300k_rows_under_30s():
    n = 300_000
    rng = np.random.default_rng(0)
    ts = pd.Timestamp("2020-01-01", tz="UTC") + pd.to_timedelta(
        np.sort(rng.integers(0, 3 * 365 * 86400, n)), unit="s"
    )
    df = pd.DataFrame(
        {
            "ts": ts,
            "start_ts": ts,
            "source": "spotify_extended",
            "item_id": rng.integers(0, 20_000, n).astype(str),
            "artist_id": rng.integers(0, 2_000, n).astype(str),
            "album_id": "x",
            "platform": "android",
            "shuffle": pd.array(rng.integers(0, 2, n).astype(bool), dtype="boolean"),
            "reason_start": "trackdone",
            "offline": pd.array([False] * n, dtype="boolean"),
            "skip": rng.random(n) < 0.3,
        }
    )[DECISION_COLUMNS]
    start = time.perf_counter()
    feats = build_features(df)
    assert time.perf_counter() - start < 30
    assert len(feats) == n


def test_unsorted_input_raises(personal):
    shuffled = personal.sample(frac=1.0, random_state=1)
    with pytest.raises(ValueError, match="sorted"):
        build_features(shuffled)


@pytest.mark.parametrize(
    ("platform", "column"),
    [
        ("web_player osx 11.3.0;chrome", "platform_web"),
        ("web_player windows 10;chrome", "platform_web"),
        ("Android OS 12", "platform_android"),
        ("iOS 16.1 (iPhone14,2)", "platform_ios"),
        ("Windows 10 (10.0.19045; x64)", "platform_desktop"),
    ],
)
def test_real_platform_strings(platform, column):
    df = _rows([("2024-01-01 10:00", "a", "t1", False)])
    df["platform"] = platform
    row = build_features(df).loc[0]
    plat = row[[c for c in CONTEXT_COLS if c.startswith("platform_")]]
    assert plat.idxmax() == column
    assert plat.sum() == 1


def test_missing_ids_share_history():
    df = _rows(
        [
            ("2024-01-01 10:00", "a", "t1", True),
            ("2024-01-01 10:01", "a", "t2", False),
        ]
    )
    df["artist_id"] = pd.Series([None, None], dtype=object)
    df["item_id"] = pd.Series([None, None], dtype=object)
    feats = build_features(df)
    assert feats.loc[1, "artist_n_log"] > 0
    assert feats.loc[1, "track_n_log"] > 0
    assert np.isfinite(feats.to_numpy(dtype=float)).all()
