"""Markdown reports for a decision evaluation.

The public report holds aggregates only: no artist, track or album strings, no
ids, dates as YYYY-MM. The private report adds the user's most skipped and most
kept artists and belongs under the git-ignored data dir.
"""

import math

import pandas as pd

MIN_ARTIST_PLAYS = 20
TOP_ARTISTS = 10
LIMITATIONS = (
    "Spotify's ms_played excludes pauses, so the session boundary and time of day "
    "for a play can be shifted by that play's own pause time."
)
LABEL_RULES = {
    "spotify_extended": (
        "a play counts as a skip when it ended with the forward button or Spotify "
        "marked it skipped"
    ),
    "spotify_account": "a play counts as a skip when it lasted under 30 seconds",
}
RUNG_NOTES = {
    "base": "running skip rate, nothing else",
    "context": "hour, weekday, platform, shuffle, how the play started, session position",
    "memory": "context plus smoothed per-artist and per-track skip history",
    "recency": "memory plus recency-weighted history",
}


def _num(x: float, spec: str = ".3f") -> str:
    return "n/a" if x is None or math.isnan(x) else format(x, spec)


def _ci(ci: tuple[float, float]) -> str:
    return f"{_num(ci[0])} to {_num(ci[1])}"


def _gap(result: dict, lower: str, higher: str) -> dict:
    return next(
        g for g in result["gaps"] if g["lower"] == lower and g["higher"] == higher
    )


def period(df: pd.DataFrame, tz: str = "UTC") -> tuple[str, str]:
    """First and last month (YYYY-MM, in ``tz``) of the plays."""
    ts = df["ts"].dt.tz_convert(tz)
    return ts.min().strftime("%Y-%m"), ts.max().strftime("%Y-%m")


def headline(result: dict) -> str:
    """The one-sentence verdict: does remembering you beat context alone?

    "beats" needs both the context-to-memory gain and the memory-versus-base gain
    to have a 95% CI lower bound above zero.
    """
    gap = _gap(result, "context", "memory")
    lo, hi = gap["ci"]
    mvb = result["memory_vs_base"]
    wins = lo > 0 and mvb["ci"][0] > 0
    verb = "beats" if wins else "does not beat"
    return (
        f"Remembering you {verb} context-only listening habits: log-loss gain "
        f"{_num(gap['ll_gain'])} (95% CI {_num(lo)} to {_num(hi)}) on "
        f"{result['n_holdout']} held-out plays, base skip rate "
        f"{_num(result['base_rate_holdout'], '.0%')}; memory versus base-rate-only "
        f"gain {_num(mvb['ll_gain'])} (95% CI {_ci(mvb['ci'])})."
    )


def counts_line(result: dict) -> str:
    """Aggregate counts of the two timing oddities the loader found."""
    return (
        f"{result.get('n_near_duplicates', 0)} near-duplicate plays dropped (same "
        f"item again within 60 s); {result.get('n_overlap', 0)} plays overlap the "
        "previous play by more than 5 s (kept)."
    )


def source_line(result: dict) -> str:
    source = result.get("source", "unknown")
    rule = LABEL_RULES.get(source, "see the project README")
    synthetic = " Synthetic data, not a real person." if result.get("synthetic") else ""
    return f"Source: {source}; {rule}.{synthetic}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + " --- |" * len(header)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def render_public(result: dict, period: tuple[str, str]) -> str:
    """Aggregates-only report; ``period`` is (first, last) month as YYYY-MM."""
    lines = [
        "# Spotify skip prediction",
        "",
        headline(result),
        "",
        source_line(result),
        "",
        (
            f"Plays from {period[0]} to {period[1]}: {result['n_total']} in all, "
            f"the last {result['n_holdout']} held out. Every prediction uses only "
            "strictly earlier plays. Gains are log-loss differences (positive means "
            "the higher rung is better) with day-block bootstrap 95% CIs. "
            + counts_line(result)
        ),
        "",
        "## Ladder on held-out plays",
        "",
    ]
    rows = [
        [
            name,
            _num(r["log_loss"]),
            _ci(r["ll_ci"]),
            _num(r["brier"]),
            _num(r["auc"]),
            _num(r["accuracy"], ".1%"),
            RUNG_NOTES.get(name, ""),
        ]
        for name, r in result["rungs"].items()
    ]
    lines += _table(
        ["rung", "log-loss", "95% CI", "Brier", "AUC", "accuracy", "uses"], rows
    )
    lines += [
        "",
        f"Base skip rate on held-out plays: {_num(result['base_rate_holdout'], '.1%')}.",
        "",
        "## What each step adds",
        "",
    ]
    lines += _table(
        ["from", "to", "log-loss gain", "95% CI"],
        [
            [g["lower"], g["higher"], _num(g["ll_gain"]), _ci(g["ci"])]
            for g in result["gaps"]
        ],
    )
    d = result["drift"]
    lines += [
        "",
        "## Frozen versus updating (memory rung)",
        "",
        (
            f"A model frozen at the split has log-loss {_num(d['frozen_ll'])}; the "
            f"same model updating after every play has {_num(d['updating_ll'])}. "
            f"Gain from updating: {_num(d['gain'])} (95% CI {_ci(d['ci'])})."
        ),
        "",
        "## Calibration of the best rung",
        "",
    ]
    cal = result["calibration"]
    lines += _table(
        ["predicted bin", "plays", "mean predicted", "observed skip rate"],
        [
            [
                f"{_num(r.bin_low, '.1f')}-{_num(r.bin_high, '.1f')}",
                str(int(r.n)),
                _num(r.mean_p),
                _num(r.skip_rate),
            ]
            for r in cal.itertuples()
        ],
    )
    lines += ["", "## Known limitations", "", LIMITATIONS]
    return "\n".join(lines) + "\n"


def _artist_rows(df: pd.DataFrame) -> tuple[list[list[str]], list[list[str]]]:
    stats = (
        df.groupby("artist_id")["skip"].agg(["size", "mean"]).reset_index()
    )
    stats = stats[stats["size"] >= MIN_ARTIST_PLAYS].sort_values(
        ["mean", "artist_id"], ascending=[False, True]
    )
    skipped = stats.head(TOP_ARTISTS)
    kept = stats.drop(skipped.index).tail(TOP_ARTISTS).iloc[::-1]

    def rows(frame: pd.DataFrame) -> list[list[str]]:
        return [
            [str(r.artist_id).replace("|", "/"), str(int(r.size)), f"{r.mean:.0%}"]
            for r in frame.itertuples()
        ]

    return rows(skipped), rows(kept)


def render_private(result: dict, df: pd.DataFrame, tz: str = "UTC") -> str:
    """The public report plus the most skipped and most kept artists (names!)."""
    skipped, kept = _artist_rows(df)
    header = ["artist", "plays", "skip rate"]
    lines = [
        render_public(result, period(df, tz)).rstrip("\n"),
        "",
        f"## Artists you skip most (at least {MIN_ARTIST_PLAYS} plays)",
        "",
        *_table(header, skipped),
        "",
        f"## Artists you keep most (at least {MIN_ARTIST_PLAYS} plays)",
        "",
        *_table(header, kept),
    ]
    return "\n".join(lines) + "\n"
