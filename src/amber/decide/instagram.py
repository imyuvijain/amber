"""Parse an Instagram (Meta) JSON export into event rows.

Files are found recursively and mapped to a kind **by file stem** (an allow-list).
Any other file, including ``messages/``, ``media/`` and contacts, is never opened:
it is only counted as skipped. Instagram nests records in ``string_list_data`` or
``string_map_data`` and the keys drift between versions, so one generic record
walker accepts either structure.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from amber.decide.events import events_frame

log = logging.getLogger(__name__)

SOURCE = "instagram"
STEM_KINDS: dict[str, str] = {
    "liked_posts": "like",
    "likes_media_likes": "like",
    "saved_posts": "save",
    "story_likes": "story_interaction",
    "polls": "story_interaction",
    "quizzes": "story_interaction",
    "emoji_sliders": "story_interaction",
    "questions": "story_interaction",
    "posts_viewed": "view_post",
    "videos_watched": "view_video",
    "ads_viewed": "ad_view",
    "ads_clicked": "ad_click",
    "following": "follow",
    "account_searches": "ig_search",
    "word_or_phrase_searches": "ig_search",
    "tag_searches": "ig_search",
    "link_history": "link_open",
}
STEM_PREFIXES: dict[str, str] = {"post_comments": "comment", "comments": "comment"}
ADS_KINDS = {"ad_view", "ad_click"}


def fix_mojibake(s):
    """Repair UTF-8 bytes that were stored as Latin-1; leave anything else alone."""
    if not isinstance(s, str):
        return s
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def kind_for_stem(stem: str) -> str | None:
    stem = stem.casefold()
    if stem in STEM_KINDS:
        return STEM_KINDS[stem]
    for prefix, kind in STEM_PREFIXES.items():
        if stem.startswith(prefix):
            return kind
    return None


def _read_json(f: Path):
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("skipping unreadable Instagram file %s: %s", f, e)
        return None


def _records(data) -> list | None:
    """A bare list, or the lists held by a top-level dict; None if neither."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        lists = [v for v in data.values() if isinstance(v, list)]
        if lists:
            return [r for lst in lists for r in lst]
    return None


def _items(record: dict) -> list[dict]:
    """Every inner dict of ``string_list_data`` and ``string_map_data``, in order."""
    out: list[dict] = []
    sld = record.get("string_list_data")
    if isinstance(sld, list):
        out += [x for x in sld if isinstance(x, dict)]
    smd = record.get("string_map_data")
    if isinstance(smd, dict):
        out += [x for x in smd.values() if isinstance(x, dict)]
    return out


def _map_value(record: dict, *keys: str):
    smd = record.get("string_map_data")
    if not isinstance(smd, dict):
        return None
    wanted = {k.casefold() for k in keys}
    for k, v in smd.items():
        if k.casefold() in wanted and isinstance(v, dict):
            return v.get("value") or None
    return None


def _timestamp(record: dict) -> pd.Timestamp | None:
    candidates = [x.get("timestamp") for x in _items(record)] + [
        record.get("timestamp")
    ]
    for t in candidates:
        if isinstance(t, (int, float)) and not isinstance(t, bool) and t > 0:
            return pd.Timestamp(int(t), unit="s", tz="UTC")
    return None


def _first_str(*values) -> str | None:
    for v in values:
        if isinstance(v, str) and v:
            return fix_mojibake(v)
    return None


def _row(record: dict, kind: str, ts: pd.Timestamp) -> dict:
    items = _items(record)
    href = _first_str(*[x.get("href") for x in items], record.get("href"))
    value = _first_str(*[x.get("value") for x in items][:1])
    title = _first_str(record.get("title"))
    author = _first_str(_map_value(record, "Author", "Media Owner"))
    if kind == "ig_search":
        text = _first_str(_map_value(record, "Search"), value, title)
        item_id, creator = href or text, None
    else:
        text = _first_str(_map_value(record, "Comment", "Search"), value, title)
        item_id, creator = href, _first_str(title, author, value)
    return {
        "ts": ts,
        "source": SOURCE,
        "kind": kind,
        "item_id": item_id,
        "creator_id": creator,
        "text": text,
        "url": href,
        "from_ads": kind in ADS_KINDS,
    }


def load_instagram(path: Path) -> pd.DataFrame:
    """Load Instagram activity from an export folder (searched recursively).

    Returns ``EVENT_COLUMNS`` sorted by ``ts`` with duplicates dropped. ``df.attrs``
    carries ``n_files`` (parsed), ``n_skipped_files`` (not allow-listed, unreadable
    or not a record container) and ``n_bad_rows`` (no timestamp or not an object).
    """
    stats = {"n_files": 0, "n_skipped_files": 0, "n_bad_rows": 0}
    rows: list[dict] = []
    for f in sorted(Path(path).rglob("*.json")):
        kind = kind_for_stem(f.stem)
        if kind is None or not f.is_file():
            stats["n_skipped_files"] += 1
            continue
        records = _records(_read_json(f))
        if records is None:
            stats["n_skipped_files"] += 1
            continue
        stats["n_files"] += 1
        for rec in records:
            ts = _timestamp(rec) if isinstance(rec, dict) else None
            if ts is None:
                stats["n_bad_rows"] += 1
                continue
            rows.append(_row(rec, kind, ts))
    return events_frame(rows, stats)
