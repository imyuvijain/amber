"""Parse a Google Takeout export (My Activity and YouTube history) into event rows.

Files are found recursively **by filename** (``MyActivity``, ``watch-history`` and
``search-history``, as ``.html`` (Takeout's default) or ``.json``), never by folder path,
because Takeout folder names are localised. The one exception is a folder named
``Gemini Apps``, which is never opened. The kind of a row comes from its URL, not
from its (localised) title; a YouTube "Liked" verb (HTML only) makes it a ``like``. Only an allow-list of My Activity products is read;
anything else (Gemini Apps, Chrome, ...) is skipped and counted.
"""

import json
import logging
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from amber.decide.events import events_frame
from amber.decide.google_html import parse_activity_html

log = logging.getLogger(__name__)

ACTIVITY_STEM = "MyActivity"
YOUTUBE_STEMS = ("watch-history", "search-history")
EXTENSIONS = (".html", ".json")
NEVER_OPEN_FOLDERS = {"gemini apps"}
ALLOWED_PRODUCTS = {
    p.casefold()
    for p in (
        "Search",
        "YouTube",
        "Discover",
        "Image Search",
        "Video Search",
        "Maps",
        "Shopping",
        "Google Play Store",
        "Flights",
        "Hotels",
        "AI Mode",
    )
}
ADS_MARKER = "from google ads"


def _find(path: Path) -> tuple[list[Path], list[Path]]:
    """``(files to read, files never opened)``."""
    found = {
        f
        for stem in (ACTIVITY_STEM, *YOUTUBE_STEMS)
        for ext in EXTENSIONS
        for f in path.rglob(stem + ext)
        if f.is_file()
    }
    never = {f for f in found if NEVER_OPEN_FOLDERS & {p.casefold() for p in f.parts}}
    return sorted(found - never), sorted(never)


def _first(params: dict, key: str) -> str | None:
    values = params.get(key)
    return values[0] if values else None


def _host(parsed) -> str:
    return (parsed.hostname or "").lower()


def _is_google_host(host: str) -> bool:
    return host.startswith("google.") or ".google." in host


def _youtube(parsed, params) -> tuple[str, str | None] | None:
    host = _host(parsed)
    if host == "youtu.be":
        return "watch", parsed.path.strip("/").split("/")[0] or None
    if host == "youtube.com" or host.endswith(".youtube.com"):
        if parsed.path == "/watch" and _first(params, "v"):
            return "watch", _first(params, "v")
        if parsed.path == "/results" and _first(params, "search_query"):
            return "search", _first(params, "search_query")
    return None


def _classify(row: dict, product: str) -> tuple[str, str, str | None]:
    """Return ``(source, kind, item_id)`` for one allow-listed activity row."""
    url = row.get("titleUrl")
    url = url if isinstance(url, str) and url else None
    is_youtube = product.casefold() == "youtube"
    verb = row.get("verb")
    liked = is_youtube and isinstance(verb, str) and verb.casefold().startswith("liked")
    if url:
        parsed = urlparse(url)
        params = parse_qs(parsed.query)
        hit = _youtube(parsed, params)
        if liked:
            return "youtube", "like", hit[1] if hit and hit[0] == "watch" else url
        if hit:
            return "youtube", hit[0], hit[1]
        host = _host(parsed)
        if _is_google_host(host) and parsed.path == "/search" and _first(params, "q"):
            return "google", "search", _first(params, "q")
        if product.casefold() == "search":
            target = _first(params, "q") if _is_google_host(host) and parsed.path == "/url" else None
            return "google", "visit", target or url
    elif liked:
        return "youtube", "like", None
    elif is_youtube:
        return "youtube", "watch", None  # a removed video has no URL
    source = "youtube" if is_youtube else "google"
    if product.casefold() == "discover":
        return source, "discover", None
    return source, "activity", None


def _product(row: dict, default: str | None) -> str | None:
    header = row.get("header")
    if isinstance(header, str) and header:
        return header
    products = row.get("products")
    if isinstance(products, list) and products and isinstance(products[0], str):
        return products[0]
    return default


def _creator(row: dict) -> str | None:
    subtitles = row.get("subtitles")
    if not (isinstance(subtitles, list) and subtitles and isinstance(subtitles[0], dict)):
        return None
    first = subtitles[0]
    for key in ("url", "name"):
        if isinstance(first.get(key), str) and first[key]:
            return first[key]
    return None


def _from_ads(row: dict) -> bool:
    details = row.get("details")
    if not isinstance(details, list):
        return False
    for d in details:
        name = d.get("name") if isinstance(d, dict) else d
        if isinstance(name, str) and ADS_MARKER in name.casefold():
            return True
    return False


def _ts(value) -> pd.Timestamp | None:
    if isinstance(value, pd.Timestamp):
        return value
    if not isinstance(value, str):
        return None
    ts = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(ts) else ts


def _read_entries(f: Path) -> list | None:
    """The raw activity rows of one file; None if it cannot be read as an activity file."""
    try:
        text = f.read_text(encoding="utf-8")
        if f.suffix.casefold() == ".html":
            entries = parse_activity_html(text)
            return entries or None  # a page with no entries is not a Takeout activity page
        data = json.loads(text)
    except (OSError, ValueError) as e:  # includes UnicodeDecodeError
        log.warning("skipping unreadable Google file %s: %s", f, e)
        return None
    if not isinstance(data, list):
        log.warning("skipping %s: expected a JSON list", f)
        return None
    return data


def _default_product(f: Path) -> str | None:
    """Product for a row with no header: YouTube history files, else the parent folder."""
    if f.stem in YOUTUBE_STEMS:
        return "YouTube"
    return f.parent.name if f.suffix.casefold() == ".html" else None


def load_google(path: Path) -> pd.DataFrame:
    """Load Google and YouTube activity from a Takeout folder (searched recursively).

    Reads both the HTML (default) and the JSON Takeout formats. Returns
    ``EVENT_COLUMNS`` sorted by ``ts`` with duplicates dropped. ``df.attrs`` carries
    ``n_files`` (parsed), ``n_skipped_files`` (unreadable, or under a ``Gemini Apps``
    folder and never opened), ``n_bad_rows`` (not an object, or no valid time) and
    ``n_skipped_rows`` (products off the allow-list).
    """
    stats = {"n_files": 0, "n_skipped_files": 0, "n_bad_rows": 0, "n_skipped_rows": 0}
    rows: list[dict] = []
    files, never_opened = _find(Path(path))
    stats["n_skipped_files"] += len(never_opened)
    for f in files:
        data = _read_entries(f)
        if data is None:
            stats["n_skipped_files"] += 1
            continue
        stats["n_files"] += 1
        default = _default_product(f)
        for r in data:
            if not isinstance(r, dict):
                stats["n_bad_rows"] += 1
                continue
            ts = _ts(r.get("time"))
            if ts is None:
                stats["n_bad_rows"] += 1
                continue
            product = _product(r, default)
            if product is None or product.casefold() not in ALLOWED_PRODUCTS:
                stats["n_skipped_rows"] += 1
                continue
            source, kind, item_id = _classify(r, product)
            title, url = r.get("title"), r.get("titleUrl")
            rows.append(
                {
                    "ts": ts,
                    "source": source,
                    "kind": kind,
                    "item_id": item_id,
                    "creator_id": _creator(r),
                    "text": title if isinstance(title, str) else None,
                    "url": url if isinstance(url, str) else None,
                    "from_ads": _from_ads(r),
                }
            )
    return events_frame(rows, stats)
