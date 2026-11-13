"""Prequential features: a play sees only plays with a strictly earlier ``ts``.

One pass in time order, grouped by ``ts``. For each group the features of every
row are computed first, and only then are the group's labels added to the state,
so plays that share a timestamp never see each other's labels.
"""

import math

import numpy as np
import pandas as pd

SESSION_GAP_S = 30 * 60
GAP_CAP_MIN = 1440
NEVER_PLAYED_DAYS = 365
UNKNOWN = "<unknown>"  # stable key for missing artist or track ids
ARTIST_K = 5.0
TRACK_K = 3.0
REASONS = ["clickrow", "trackdone", "fwdbtn", "playbtn"]
PLATFORMS = ["android", "ios", "desktop", "web", "other"]

CONTEXT_COLS: list[str] = (
    ["hour_sin", "hour_cos", "weekend"]
    + [f"platform_{p}" for p in PLATFORMS]
    + ["shuffle", "shuffle_known"]
    + [f"rs_{r}" for r in REASONS]
    + ["rs_other", "session_pos_log", "gap_min_log", "prev_skip"]
)
MEMORY_COLS: list[str] = [
    "artist_n_log",
    "track_n_log",
    "artist_skip_s",
    "track_skip_s",
    "artist_days_log",
    "track_days_log",
]
DECAY_COLS: list[str] = ["artist_dskip_s", "track_dskip_s", "artist_dn_log"]


def _platform(value) -> str:
    if pd.isna(value):
        return "other"
    p = str(value).lower()
    if "android" in p:
        return "android"
    if "ios" in p or "iphone" in p:
        return "ios"
    if "web" in p:
        return "web"
    if any(k in p for k in ("windows", "osx", "os x", "mac", "linux")):
        return "desktop"
    return "other"


def _context(df: pd.DataFrame, tz: str) -> pd.DataFrame:
    """The columns that depend only on the row itself (no history)."""
    local = df["ts"].dt.tz_convert(tz)
    hour = local.dt.hour + local.dt.minute / 60
    out = pd.DataFrame(index=df.index)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekend"] = (local.dt.day_of_week >= 5).astype(float)
    platform = df["platform"].map(_platform)
    for p in PLATFORMS:
        out[f"platform_{p}"] = (platform == p).astype(float)
    shuffle = df["shuffle"]
    out["shuffle"] = shuffle.map(lambda v: 0.0 if pd.isna(v) else float(bool(v)))
    out["shuffle_known"] = shuffle.notna().astype(float)
    reason = df["reason_start"].astype(object)
    for r in REASONS:
        out[f"rs_{r}"] = (reason == r).astype(float)
    out["rs_other"] = (reason.notna() & ~reason.isin(REASONS)).astype(float)
    return out


def build_features(
    df: pd.DataFrame, tz: str = "UTC", half_life_days: float = 30.0
) -> pd.DataFrame:
    """Context, memory and decay features for every play, without look-ahead.

    ``df`` must be sorted by ``ts`` (as ``load_export`` returns it). The result
    has the same index as ``df`` and only finite floats.
    """
    if not df["ts"].is_monotonic_increasing:
        raise ValueError("df must be sorted by ts (use load_export's order)")
    n = len(df)
    context = _context(df, tz)
    t = (
        (df["ts"] - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)
    ).to_numpy()  # epoch s
    skip = df["skip"].to_numpy(dtype=bool)
    artists = (
        df["artist_id"].astype(object).where(df["artist_id"].notna(), UNKNOWN).tolist()
    )
    tracks = df["item_id"].astype(object).where(df["item_id"].notna(), UNKNOWN).tolist()

    # per-row outputs of the pass: session (3), memory (6), decay (3)
    out = np.zeros((n, 12))
    # per-key state: [n, skips, last_t, decayed_n, decayed_skips]
    artist_state: dict = {}
    track_state: dict = {}
    decay_rate = math.log(2) / (half_life_days * 86400)

    n_seen = skips_seen = 0
    last_t = None  # ts of the previous group
    in_session = 0  # plays in the current session, before the current group
    prev_skip = 0.0  # label of the last play of the previous group
    log1p, exp = math.log1p, math.exp

    def read(state, key, now, k, g):
        s = state.get(key)
        if s is None:
            prior = g
            return 0.0, prior, log1p(NEVER_PLAYED_DAYS), prior, 0.0
        count, skips, seen_t, dn, ds = s
        f = exp(-decay_rate * (now - seen_t))
        smooth = (skips + k * g) / (count + k)
        dsmooth = (ds * f + k * g) / (dn * f + k)
        return log1p(count), smooth, log1p((now - seen_t) / 86400), dsmooth, dn * f

    start = 0
    while start < n:
        now = t[start]
        stop = start + 1
        while stop < n and t[stop] == now:
            stop += 1

        if last_t is None or now - last_t > SESSION_GAP_S:
            in_session, prev_skip = 0, 0.0
        gap_min = (
            GAP_CAP_MIN if last_t is None else min((now - last_t) / 60, GAP_CAP_MIN)
        )
        session = (log1p(in_session), log1p(gap_min), prev_skip)
        g = (skips_seen + 1) / (n_seen + 2)

        for i in range(start, stop):
            a_n, a_s, a_days, a_ds, a_dn = read(
                artist_state, artists[i], now, ARTIST_K, g
            )
            t_n, t_s, t_days, t_ds, _ = read(track_state, tracks[i], now, TRACK_K, g)
            out[i] = (
                *session,
                a_n,
                t_n,
                a_s,
                t_s,
                a_days,
                t_days,
                a_ds,
                t_ds,
                log1p(a_dn),
            )

        for i in range(start, stop):
            s = int(skip[i])
            for state, key in ((artist_state, artists[i]), (track_state, tracks[i])):
                old = state.get(key)
                if old is None:
                    state[key] = [1, s, now, 1.0, float(s)]
                else:
                    f = exp(-decay_rate * (now - old[2]))
                    state[key] = [
                        old[0] + 1,
                        old[1] + s,
                        now,
                        old[3] * f + 1,
                        old[4] * f + s,
                    ]
            skips_seen += s
        n_seen += stop - start
        in_session += stop - start
        prev_skip = float(skip[stop - 1])
        last_t = now
        start = stop

    result = context.copy()
    names = ["session_pos_log", "gap_min_log", "prev_skip"] + MEMORY_COLS + DECAY_COLS
    for j, name in enumerate(names):
        result[name] = out[:, j]
    return result[CONTEXT_COLS + MEMORY_COLS + DECAY_COLS]
