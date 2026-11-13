"""Fake Spotify exports with planted skip structure, for tests and demos.

Writes real-format JSON files (the same names and fields ``load_export`` reads)
so the whole decision pipeline can be checked against a known truth:

* ``personal``: skips depend on the artist, on shuffle, on the hour and on how the
  play started, and the artist tastes flip for four artists halfway through.
* ``null``: every play skips with p = 0.3, whatever the context. A model that
  finds signal here is fooling itself.

The raw rows also carry ``ip_addr`` and ``conn_country`` (the loader must drop
them) plus podcast and private-session rows (the loader must skip them). The
fake user's local time is UTC. Output is a pure function of the arguments.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import numpy as np

PLATFORMS = ["android", "ios", "windows", "web_player"]
N_ARTISTS = 40
N_HATED = 5
N_LOVED = 5
N_SWAPPED = 2  # hated and loved artists that trade places at the halfway point
P_HATED, P_LOVED, P_NULL = 0.9, 0.05, 0.3
SHUFFLE_BUMP, NIGHT_BUMP = 0.15, 0.10
NIGHT_HOURS = 6  # hours 0-5 count as night
P_PODCAST, P_INCOGNITO = 0.02, 0.01
START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_S = 730 * 86_400  # about two years
SESSION_GAP_S = 2 * 3600
SKIP_MS = 30_000  # a skip is shorter than this
CHUNK = 3000  # rows per file
COUNTRIES = ["DE", "US", "GB", "FR"]


def _catalog(rng: np.random.Generator) -> list[list[dict]]:
    """40 artists with 1-15 tracks each (name, uri, album and full length)."""
    catalog = []
    for a in range(N_ARTISTS):
        n_tracks = int(rng.integers(1, 16))
        catalog.append(
            [
                {
                    "artist": f"Fake Artist {a:02d}",
                    "track": f"Fake Track {a:02d}-{t:02d}",
                    "album": f"Fake Album {a:02d}-{t % 3}",
                    "uri": f"spotify:track:fake{a:02d}x{t:02d}",
                    "ms": int(rng.integers(120_000, 300_000)),
                }
                for t in range(n_tracks)
            ]
        )
    return catalog


def _artist_p(rng: np.random.Generator, kind: str):
    """Skip probability per artist before and after the halfway swap."""
    beta = rng.beta(2, 5, N_ARTISTS)  # drawn for both kinds so the artist set matches
    order = rng.permutation(N_ARTISTS)
    hated, loved = order[:N_HATED], order[N_HATED : N_HATED + N_LOVED]
    first = beta.copy()
    first[hated], first[loved] = P_HATED, P_LOVED
    second = first.copy()
    second[hated[:N_SWAPPED]], second[loved[:N_SWAPPED]] = P_LOVED, P_HATED
    if kind == "null":
        first = second = np.full(N_ARTISTS, P_NULL)
    return first, second


def _session_sizes(rng: np.random.Generator, n_plays: int) -> list[int]:
    sizes: list[int] = []
    left = n_plays
    while left > 0:
        size = int(rng.integers(5, 41))
        if left - size < 5:  # never leave a runt session behind
            size = left
        sizes.append(size)
        left -= size
    return sizes


def _plays(kind: str, n_plays: int, seed: int, fmt: str) -> list[dict]:
    rng = np.random.default_rng(seed)
    catalog = _catalog(rng)
    p_first, p_second = _artist_p(rng, kind)
    sizes = _session_sizes(rng, n_plays)
    starts = np.sort(rng.integers(0, WINDOW_S, len(sizes)))
    t0 = int(START.timestamp())
    plays: list[dict] = []
    session_end = -SESSION_GAP_S
    for size, start in zip(sizes, starts):
        t = max(int(start), session_end + SESSION_GAP_S)
        platform = PLATFORMS[int(rng.integers(len(PLATFORMS)))]
        country = COUNTRIES[int(rng.integers(len(COUNTRIES)))]
        ip = f"203.0.113.{int(rng.integers(1, 255))}"
        shuffle = bool(rng.random() < 0.4)
        prev_skip = False
        for i in range(size):
            idx = len(plays)
            p_art = p_first if idx < n_plays // 2 else p_second
            artist = int(rng.integers(N_ARTISTS))
            track = catalog[artist][int(rng.integers(len(catalog[artist])))]
            if i == 0:
                reason_start = "clickrow" if rng.random() < 0.5 else "playbtn"
            elif rng.random() < 0.1:
                reason_start = "clickrow"
            else:
                reason_start = "fwdbtn" if prev_skip else "trackdone"
            hour = ((t0 + t) % 86_400) // 3600
            p = float(p_art[artist])
            if kind == "personal":
                p += NIGHT_BUMP * (hour < NIGHT_HOURS)
                if fmt == "extended":  # the account format has no such fields
                    p += SHUFFLE_BUMP * shuffle
            p = min(max(p, 0.01), 0.99)
            if kind == "personal" and fmt == "extended" and reason_start == "clickrow":
                p = max(p / 2, 0.01)
            skip = bool(rng.random() < p)
            ms = int(rng.integers(1_000, SKIP_MS)) if skip else track["ms"]
            podcast = bool(rng.random() < P_PODCAST)
            incognito = bool(rng.random() < P_INCOGNITO)
            gap = int(rng.integers(5, 121))
            t += ms // 1000 + gap if i else ms // 1000
            plays.append(
                {
                    "ts": t0 + t,
                    "platform": platform,
                    "country": country,
                    "ip": ip,
                    "shuffle": shuffle,
                    "reason_start": reason_start,
                    "track": track,
                    "skip": skip,
                    "ms": ms,
                    "podcast": podcast,
                    "incognito": incognito,
                }
            )
            prev_skip = skip
        session_end = t
    return plays


def _iso(ts: int, fmt: str) -> str:
    return datetime.fromtimestamp(ts, UTC).strftime(fmt)


def _extended_row(p: dict) -> dict:
    track = None if p["podcast"] else p["track"]
    return {
        "ts": _iso(p["ts"], "%Y-%m-%dT%H:%M:%SZ"),
        "username": "fake_user",
        "platform": p["platform"],
        "ms_played": p["ms"],
        "conn_country": p["country"],
        "ip_addr": p["ip"],
        "master_metadata_track_name": track and track["track"],
        "master_metadata_album_artist_name": track and track["artist"],
        "master_metadata_album_album_name": track and track["album"],
        "spotify_track_uri": track and track["uri"],
        "episode_name": "Fake Episode" if p["podcast"] else None,
        "episode_show_name": "Fake Show" if p["podcast"] else None,
        "spotify_episode_uri": "spotify:episode:fakeep" if p["podcast"] else None,
        "reason_start": p["reason_start"],
        "reason_end": "fwdbtn" if p["skip"] else "trackdone",
        "shuffle": p["shuffle"],
        "skipped": p["skip"],
        "offline": False,
        "incognito_mode": p["incognito"],
    }


def _account_row(p: dict) -> dict:
    return {
        "endTime": _iso(p["ts"], "%Y-%m-%d %H:%M"),
        "artistName": p["track"]["artist"],
        "trackName": p["track"]["track"],
        "msPlayed": p["ms"],
    }


def write_fake_export(
    dest: Path,
    kind: Literal["personal", "null"],
    n_plays: int = 6000,
    seed: int = 7,
    fmt: Literal["extended", "account"] = "extended",
) -> Path:
    """Write a fake export into ``dest`` (created if needed) and return it.

    The extended format keeps podcast and private-session rows (about 2% and 1%)
    for the loader to drop. The account format has neither field, so those plays
    are left out. It has no shuffle or start-reason fields, so those effects are
    not planted there (only the artist signal and time of day are).
    """
    if kind not in ("personal", "null"):
        raise ValueError(f"unknown kind: {kind!r}")
    if fmt not in ("extended", "account"):
        raise ValueError(f"unknown fmt: {fmt!r}")
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    plays = _plays(kind, n_plays, seed, fmt)
    if fmt == "extended":
        rows = [_extended_row(p) for p in plays]
        name = "Streaming_History_Audio_2024-2025_{}.json"
    else:
        rows = [_account_row(p) for p in plays if not (p["podcast"] or p["incognito"])]
        name = "StreamingHistory_music_{}.json"
    for i in range(0, max(len(rows), 1), CHUNK):
        text = json.dumps(rows[i : i + CHUNK], indent=2) + "\n"
        (dest / name.format(i // CHUNK)).write_text(text, encoding="utf-8")
    return dest
