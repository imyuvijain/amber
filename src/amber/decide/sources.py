"""Combine the Google, YouTube and Instagram exports into one event log, and audit it.

``load_sources`` scans ``google/`` (Takeout) and ``meta/`` and ``instagram/`` (Meta
export) under one exports folder. ``audit_lines`` summarises the result with counts and
dates only: it never includes a title, query, URL, username or id.
"""

from pathlib import Path

import pandas as pd

from amber.decide.events import COUNT_KEYS, DEDUP_KEY, EVENT_COLUMNS
from amber.decide.google_takeout import load_google
from amber.decide.instagram import load_instagram

__all__ = ["EVENT_COLUMNS", "audit_lines", "load_sources"]

GOOGLE_DIRS = ("google",)
META_DIRS = ("meta", "instagram")


def load_sources(export_dir: Path) -> pd.DataFrame:
    """One event log from every export under ``export_dir``.

    Rows are sorted by ``ts`` with exact ``(ts, source, kind, item_id)`` duplicates
    dropped (so the same data in both ``meta/`` and ``instagram/`` counts once).
    ``df.attrs`` sums the loaders' ``n_files``, ``n_skipped_files``, ``n_bad_rows`` and
    ``n_skipped_rows``. Raises ``ValueError`` if the folder is missing or has no events.
    """
    export_dir = Path(export_dir)
    if not export_dir.is_dir():
        raise ValueError(f"{export_dir} is not a folder")
    frames = [
        loader(export_dir / name)
        for names, loader in ((GOOGLE_DIRS, load_google), (META_DIRS, load_instagram))
        for name in names
        if (export_dir / name).is_dir()
    ]
    stats = {k: sum(f.attrs.get(k, 0) for f in frames) for k in COUNT_KEYS}
    nonempty = [f for f in frames if len(f)]
    if not nonempty:
        raise ValueError(
            f"no events found under {export_dir} (looked in "
            + ", ".join(f"{d}/" for d in (*GOOGLE_DIRS, *META_DIRS))
            + ")"
        )
    df = pd.concat(nonempty, ignore_index=True)
    df = df.drop_duplicates(DEDUP_KEY, keep="first")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.attrs.update(stats)
    return df


def audit_lines(df: pd.DataFrame) -> list[str]:
    """An aggregate audit as printable lines: counts and dates, never any content."""
    lines = [f"{'source':<10} {'kind':<18} {'rows':>7}"]
    by_kind = df.groupby(["source", "kind"]).size()
    lines += [f"{src:<10} {kind:<18} {n:>7}" for (src, kind), n in by_kind.items()]
    lines += [
        "",
        f"{'source':<10} {'first':<10} {'last':<10} {'rows':>7} {'active months':>14}",
    ]
    # Months are counted on UTC dates (no timezone is assumed for the owner).
    month = df["ts"].dt.strftime("%Y-%m")
    for src, g in df.groupby("source"):
        lines.append(
            f"{src:<10} {g['ts'].min():%Y-%m-%d} {g['ts'].max():%Y-%m-%d} "
            f"{len(g):>7} {month[g.index].nunique():>14}"
        )
    lines.append(
        f"{'all':<10} {df['ts'].min():%Y-%m-%d} {df['ts'].max():%Y-%m-%d} "
        f"{len(df):>7} {month.nunique():>14}"
    )
    counts = (
        f"files read: {df.attrs.get('n_files', 0)}, "
        f"skipped files: {df.attrs.get('n_skipped_files', 0)}, "
        f"bad rows: {df.attrs.get('n_bad_rows', 0)}, "
        f"skipped rows (products off the allow-list): {df.attrs.get('n_skipped_rows', 0)}"
    )
    return [*lines, "", f"ad rows: {int(df['from_ads'].sum())}", counts]
