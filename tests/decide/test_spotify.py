import json
from pathlib import Path

import pandas as pd
import pytest

from amber.decide.spotify import DECISION_COLUMNS, load_export


def ext_row(ts="2024-01-01T12:00:00Z", uri="spotify:track:a", **over):
    row = {
        "ts": ts,
        "platform": "iOS 17",
        "ms_played": 1000,
        "conn_country": "DE",
        "ip_addr": "1.2.3.4",
        "master_metadata_track_name": "Song",
        "master_metadata_album_artist_name": "Artist",
        "master_metadata_album_album_name": "Album",
        "spotify_track_uri": uri,
        "spotify_episode_uri": None,
        "reason_start": "trackdone",
        "reason_end": "trackdone",
        "shuffle": True,
        "skipped": None,
        "offline": False,
        "incognito_mode": False,
    }
    row.update(over)
    return row


def acc_row(end="2024-01-01 12:00", ms=180000, artist="Artist", track="Song"):
    return {"endTime": end, "artistName": artist, "trackName": track, "msPlayed": ms}


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows))


def test_extended_label_rule(tmp_path):
    write(
        tmp_path / "Streaming_History_Audio_2024.json",
        [
            ext_row("2024-01-01T12:00:00Z", "spotify:track:a", reason_end="fwdbtn"),
            ext_row("2024-01-01T12:01:00Z", "spotify:track:b", reason_end="trackdone"),
            ext_row("2024-01-01T12:02:00Z", "spotify:track:c", reason_end="endplay"),
            ext_row(
                "2024-01-01T12:03:00Z",
                "spotify:track:d",
                reason_end="trackdone",
                skipped=True,
            ),
        ],
    )
    df = load_export(tmp_path)
    assert df["skip"].tolist() == [True, False, False, True]
    assert df["skip"].dtype == bool
    assert (df["source"] == "spotify_extended").all()
    assert df["item_id"].tolist() == [f"spotify:track:{c}" for c in "abcd"]
    assert (df["artist_id"] == "Artist").all()
    assert (df["album_id"] == "Album").all()
    assert df["shuffle"].tolist() == [True] * 4
    assert df["offline"].tolist() == [False] * 4
    assert df["reason_start"].tolist() == ["trackdone"] * 4


def test_account_label_rule(tmp_path):
    write(
        tmp_path / "StreamingHistory_music_0.json",
        [
            acc_row("2024-01-01 12:00", 29999, track="a"),
            acc_row("2024-01-01 12:01", 30000, track="b"),
            acc_row("2024-01-01 12:02", 180000, track="c"),
        ],
    )
    df = load_export(tmp_path)
    assert df["skip"].tolist() == [True, False, False]
    assert (df["source"] == "spotify_account").all()
    assert df["item_id"].tolist() == ["Artist|a", "Artist|b", "Artist|c"]
    assert df["artist_id"].tolist() == ["Artist"] * 3
    assert df["album_id"].isna().all()
    assert df["shuffle"].isna().all()
    assert df["offline"].isna().all()
    assert df["reason_start"].isna().all()
    assert str(df["ts"].dt.tz) == "UTC"
    assert df["ts"].iloc[0] == pd.Timestamp("2024-01-01 12:00", tz="UTC")


def test_drops_private_and_non_music(tmp_path):
    write(
        tmp_path / "Streaming_History_Audio_2024.json",
        [
            ext_row("2024-01-01T12:00:00Z", "spotify:track:keep"),
            ext_row(
                "2024-01-01T12:01:00Z",
                None,
                spotify_episode_uri="spotify:episode:e",
                master_metadata_track_name=None,
            ),
            ext_row(
                "2024-01-01T12:02:00Z",
                "spotify:track:ep",
                spotify_episode_uri="spotify:episode:e",
            ),
            ext_row(
                "2024-01-01T12:03:00Z",
                "spotify:track:null",
                master_metadata_track_name=None,
            ),
            ext_row(
                "2024-01-01T12:04:00Z", "spotify:track:priv", incognito_mode=True
            ),
        ],
    )
    df = load_export(tmp_path)
    assert list(df.columns) == DECISION_COLUMNS
    assert "ip_addr" not in df.columns and "conn_country" not in df.columns
    assert df["item_id"].tolist() == ["spotify:track:keep"]


def test_prefers_extended_and_dedups(tmp_path):
    rows = [
        ext_row("2024-01-01T12:00:00Z", "spotify:track:a"),
        ext_row("2024-01-01T12:01:00Z", "spotify:track:b"),
    ]
    write(tmp_path / "Streaming_History_Audio_2024.json", rows)
    write(tmp_path / "Streaming_History_Audio_2024 (1).json", rows)
    write(
        tmp_path / "Spotify Extended Streaming History" / "Streaming_History_Audio_2025.json",
        rows + [ext_row("2024-01-01T12:02:00Z", "spotify:track:c")],
    )
    write(tmp_path / "StreamingHistory_music_0.json", [acc_row(track="x")])
    df = load_export(tmp_path)
    assert (df["source"] == "spotify_extended").all()
    assert not df.duplicated(["ts", "item_id"]).any()
    assert df["item_id"].tolist() == [f"spotify:track:{c}" for c in "abc"]


def test_sorted_and_utc(tmp_path):
    write(
        tmp_path / "Streaming_History_Audio_2024.json",
        [
            ext_row("2024-01-03T00:00:00Z", "spotify:track:c"),
            ext_row("2024-01-01T00:00:00Z", "spotify:track:a"),
            ext_row("2024-01-02T00:00:00Z", "spotify:track:b"),
        ],
    )
    df = load_export(tmp_path)
    assert df["ts"].is_monotonic_increasing
    assert str(df["ts"].dt.tz) == "UTC"
    assert df.index.tolist() == [0, 1, 2]
    assert df["item_id"].tolist() == [f"spotify:track:{c}" for c in "abc"]


def test_equal_ts_keeps_file_order(tmp_path):
    write(
        tmp_path / "Streaming_History_Audio_2024.json",
        [
            ext_row("2024-01-01T00:00:00Z", "spotify:track:z"),
            ext_row("2024-01-01T00:00:00Z", "spotify:track:a"),
        ],
    )
    df = load_export(tmp_path)
    assert df["item_id"].tolist() == ["spotify:track:z", "spotify:track:a"]


def test_unreadable_json_is_skipped_with_warning(tmp_path, caplog):
    write(tmp_path / "Streaming_History_Audio_2024.json", [ext_row()])
    (tmp_path / "Streaming_History_Audio_2025.json").write_text("{not json")
    with caplog.at_level("WARNING"):
        df = load_export(tmp_path)
    assert len(df) == 1
    assert "Streaming_History_Audio_2025.json" in caplog.text


def test_empty_folder_raises(tmp_path):
    with pytest.raises(ValueError, match="no Spotify streaming history files"):
        load_export(tmp_path)


def test_start_ts_is_ts_minus_ms_played(tmp_path):
    write(
        tmp_path / "Streaming_History_Audio_2024.json",
        [
            ext_row("2024-01-01T12:00:00Z", "spotify:track:a", ms_played=90_500),
            ext_row("2024-01-01T12:05:00Z", "spotify:track:b", ms_played=None),
            ext_row("2024-01-01T12:10:00Z", "spotify:track:c", ms_played="oops"),
        ],
    )
    df = load_export(tmp_path)
    assert list(df.columns[:2]) == ["ts", "start_ts"]
    assert df.loc[0, "ts"] - df.loc[0, "start_ts"] == pd.Timedelta(milliseconds=90_500)
    assert (df.loc[1:, "start_ts"] == df.loc[1:, "ts"]).all()
    assert df["start_ts"].dtype == df["ts"].dtype


def test_start_ts_account_format(tmp_path):
    write(
        tmp_path / "StreamingHistory_music_0.json",
        [acc_row("2024-01-01 12:00", ms=180_000)],
    )
    df = load_export(tmp_path)
    assert df.loc[0, "ts"] - df.loc[0, "start_ts"] == pd.Timedelta(minutes=3)
