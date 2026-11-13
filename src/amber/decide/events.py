"""The shared event-log schema for the non-Spotify sources (Google, YouTube, Instagram).

One row per thing the person did, with a time. ``text`` and ``url`` are private
and must never be printed or committed; ``item_id`` and ``creator_id`` can hold
private values too (a search query, a username).
"""

import pandas as pd

EVENT_COLUMNS: list[str] = [
    "ts",
    "source",
    "kind",
    "item_id",
    "creator_id",
    "text",
    "url",
    "from_ads",
]
DEDUP_KEY: list[str] = ["ts", "source", "kind", "item_id"]
COUNT_KEYS: tuple[str, ...] = (
    "n_files",
    "n_skipped_files",
    "n_bad_rows",
    "n_skipped_rows",
)


def events_frame(rows: list[dict], stats: dict) -> pd.DataFrame:
    """Build a clean event frame: exact columns, UTC ``ts``, deduped, sorted by ``ts``."""
    df = pd.DataFrame(rows, columns=EVENT_COLUMNS)
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    df["from_ads"] = df["from_ads"].fillna(False).astype(bool)
    df = df.drop_duplicates(DEDUP_KEY, keep="first")
    df = df.sort_values("ts", kind="stable").reset_index(drop=True)
    df.attrs.update({k: stats.get(k, 0) for k in COUNT_KEYS})
    return df
