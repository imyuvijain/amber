"""Parse a Spotify data export into one row per music play.

Two export formats exist. The extended streaming history
(``Streaming_History_Audio_*.json``) is richer, so when it is present the
account-data files (``StreamingHistory_music_*.json``) are ignored. Private
fields (``ip_addr``, ``conn_country``) are never copied out of the raw rows.
"""

import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)

DECISION_COLUMNS: list[str] = [
    "ts",
    "start_ts",
    "source",
    "item_id",
    "artist_id",
    "album_id",
    "platform",
    "shuffle",
    "reason_start",
    "offline",
    "skip",
]

ACCOUNT_SKIP_MS = 30_000  # account format: a play shorter than this is a skip


def _read_rows(files: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for f in files:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            log.warning("skipping unreadable Spotify file %s: %s", f, e)
            continue
        if not isinstance(data, list):
            log.warning("skipping %s: expected a JSON list", f)
            continue
        rows.extend(r for r in data if isinstance(r, dict))
    return rows


def _start_ts(ts: pd.Series, ms_played: pd.Series) -> pd.Series:
    """``ts`` is when the play ended; the decision happened ``ms_played`` earlier.

    A missing or non-numeric ``ms_played`` leaves ``start_ts == ts``.
    """
    ms = pd.to_numeric(ms_played, errors="coerce").fillna(0).clip(lower=0)
    return ts - pd.to_timedelta(ms, unit="ms")


def _extended_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in (
        "ts",
        "spotify_track_uri",
        "spotify_episode_uri",
        "master_metadata_track_name",
        "master_metadata_album_artist_name",
        "master_metadata_album_album_name",
        "platform",
        "shuffle",
        "reason_start",
        "reason_end",
        "skipped",
        "offline",
        "incognito_mode",
        "ms_played",
    ):
        if col not in df:
            df[col] = None
    keep = (
        df["spotify_episode_uri"].isna()
        & df["master_metadata_track_name"].notna()
        & df["spotify_track_uri"].notna()
        & ~df["incognito_mode"].fillna(False).astype(bool)
    )
    df = df[keep]
    skipped = df["skipped"].fillna(False).astype(bool)
    ts = pd.to_datetime(df["ts"], utc=True)
    return pd.DataFrame(
        {
            "ts": ts,
            "start_ts": _start_ts(ts, df["ms_played"]),
            "source": "spotify_extended",
            "item_id": df["spotify_track_uri"],
            "artist_id": df["master_metadata_album_artist_name"],
            "album_id": df["master_metadata_album_album_name"],
            "platform": df["platform"],
            "shuffle": df["shuffle"].astype("boolean"),
            "reason_start": df["reason_start"],
            "offline": df["offline"].astype("boolean"),
            "skip": (df["reason_end"] == "fwdbtn") | skipped,
        }
    )


def _account_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in ("endTime", "artistName", "trackName", "msPlayed"):
        if col not in df:
            df[col] = None
    df = df[df["trackName"].notna() & df["endTime"].notna()]
    na = pd.Series(pd.NA, index=df.index, dtype=object)
    ts = pd.to_datetime(df["endTime"], utc=True)
    return pd.DataFrame(
        {
            "ts": ts,
            "start_ts": _start_ts(ts, df["msPlayed"]),
            "source": "spotify_account",
            "item_id": df["artistName"].astype(str) + "|" + df["trackName"].astype(str),
            "artist_id": df["artistName"],
            "album_id": na,
            "platform": na,
            "shuffle": pd.Series(pd.NA, index=df.index, dtype="boolean"),
            "reason_start": na,
            "offline": pd.Series(pd.NA, index=df.index, dtype="boolean"),
            "skip": pd.to_numeric(df["msPlayed"]) < ACCOUNT_SKIP_MS,
        }
    )


def load_export(path: Path) -> pd.DataFrame:
    """Load every music play from a Spotify export folder (searched recursively).

    Returns exactly ``DECISION_COLUMNS``, sorted by ``ts`` (stable), with a fresh
    index. Podcasts, rows without track metadata and private-session plays are
    dropped, as are duplicate ``(ts, item_id)`` rows from overlapping exports.
    """
    path = Path(path)
    extended = sorted(path.rglob("Streaming_History_Audio_*.json"))
    account = sorted(path.rglob("StreamingHistory_music_*.json"))
    if not extended and not account:
        raise ValueError("no Spotify streaming history files")

    if extended:
        df = _extended_frame(_read_rows(extended))
    else:
        df = _account_frame(_read_rows(account))

    df = df[DECISION_COLUMNS].drop_duplicates(["ts", "item_id"], keep="first")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df["skip"] = df["skip"].astype(bool)
    return df
