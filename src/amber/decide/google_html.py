"""Parse Google Takeout's HTML activity pages (``MyActivity.html`` and friends).

Takeout's default format is HTML, not JSON. Each entry is one
``<div class="outer-cell ...">`` block: a header cell with the product, a first
content cell (verb, a title link, an optional channel link, then a date line) and a
caption with ``Products`` and optional ``Details``. This module turns every entry into
a dict shaped like a JSON My Activity row (``header``, ``title``, ``titleUrl``,
``subtitles``, ``time``, ``details``) plus the ``verb``, so one classifier handles both
formats. It uses regex plus ``html.unescape`` (no extra dependency) and is linear in
the file size.

Dates carry a timezone *abbreviation* (``Nov 13, 2026, 2:47:58 PM EST``, with a
U+202F before the AM/PM). Each is read as the fixed UTC offset below (EST is
always UTC-5); an abbreviation not listed is a bad row. ``IST`` is read as India, not Israel or Ireland.
"""

import html
import re
from datetime import datetime, timedelta, timezone

import pandas as pd

# Each abbreviation is read literally as a fixed UTC offset in hours. Takeout prints
# one label for the whole file (a real Waterloo account shows "EST" even in July), so
# mapping EST to New York's DST rules would shift summer rows by an hour.
TZ_ABBREVIATIONS: dict[str, float] = {
    "EST": -5, "EDT": -4,
    "CST": -6, "CDT": -5,
    "MST": -7, "MDT": -6,
    "PST": -8, "PDT": -7,
    "UTC": 0, "GMT": 0,
    "IST": 5.5,
    "JST": 9,
    "CET": 1, "CEST": 2,
    "BST": 1,
}
_ZONES = {abbr: timezone(timedelta(hours=h)) for abbr, h in TZ_ABBREVIATIONS.items()}

_OUTER = re.compile(r'<div class="outer-cell[^"]*">')
_HEADER = re.compile(r'class="header-cell[^"]*">(.*?)</div>', re.DOTALL)
_CONTENT = re.compile(
    r'class="content-cell[^"]*mdl-typography--body-1">(.*?)</div>', re.DOTALL
)
_CAPTION = re.compile(
    r'class="content-cell[^"]*mdl-typography--caption">(.*?)</div>', re.DOTALL
)
_BR = re.compile(r"<br\s*/?>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_LINK = re.compile(
    r'<a\s[^>]*?href="([^"]*)"[^>]*>(.*?)</a>', re.DOTALL | re.IGNORECASE
)
_STAMP = re.compile(r"^(?P<dt>.+?[AP]M)\s+(?P<tz>[A-Za-z]+)$")
_STAMP_FORMAT = "%b %d, %Y, %I:%M:%S %p"


def _text(fragment: str) -> str:
    """Strip tags, unescape entities and collapse the odd spaces Takeout uses."""
    s = html.unescape(_TAG.sub("", fragment))
    return " ".join(s.replace(" ", " ").replace("\xa0", " ").replace(" ", " ").split())


def parse_stamp(raw: str) -> pd.Timestamp | None:
    """``Nov 13, 2026, 2:47:58 PM EST`` to a UTC timestamp; None if unreadable."""
    m = _STAMP.match(_text(raw))
    if not m:
        return None
    zone = _ZONES.get(m["tz"].upper())
    if zone is None:
        return None
    try:
        naive = datetime.strptime(m["dt"], _STAMP_FORMAT)
    except ValueError:
        return None
    return pd.Timestamp(naive.replace(tzinfo=zone)).tz_convert("UTC")


def _details(caption: str) -> list[dict]:
    if "Details:" not in caption:
        return []
    section = caption.split("Details:", 1)[1].split("<b>", 1)[0]
    return [
        {"name": t} for t in map(_text, _BR.split(section.replace("</b>", "", 1))) if t
    ]


def _entry(chunk: str) -> dict:
    entry: dict = {"verb": None, "title": None, "titleUrl": None, "time": None}
    m = _HEADER.search(chunk)
    header = _text(m[1]) if m else ""
    if header:
        entry["header"] = header
    m = _CONTENT.search(chunk)
    segments = [s for s in _BR.split(m[1]) if _text(s)] if m else []
    if segments:
        entry["time"] = parse_stamp(segments[-1])
        body = segments[:-1]
        if body:
            link = _LINK.search(body[0])
            if link:
                entry["verb"] = _text(body[0][: link.start()])
                entry["titleUrl"] = html.unescape(link[1])
                entry["title"] = _text(link[2]) or None
            else:
                entry["title"] = entry["verb"] = _text(body[0])
        for seg in body[1:]:
            link = _LINK.search(seg)
            if link:
                sub = {"url": html.unescape(link[1]), "name": _text(link[2])}
                entry["subtitles"] = [sub]
                break
    m = _CAPTION.search(chunk)
    if m:
        entry["details"] = _details(m[1])
    return entry


def parse_activity_html(text: str) -> list[dict]:
    """Every activity entry in one Takeout HTML page, in file order."""
    return [_entry(chunk) for chunk in _OUTER.split(text)[1:]]
