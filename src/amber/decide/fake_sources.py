"""Fake Google Takeout and Instagram exports, for tests and demos.

Writes real-layout files that ``load_google`` and ``load_instagram`` read:

* ``google/``: My Activity (Search, YouTube, Discover) and the YouTube history files,
  as HTML (Takeout's default) or JSON, plus a ``Gemini Apps`` folder the loader must
  never open. The YouTube history repeats the My Activity watches, so the cross-file
  dedup has something to do.
* ``instagram/``: JSON in both record structures (``string_list_data`` and
  ``string_map_data``), ads files, some mojibake, and a ``messages/`` file the loader
  must never parse.

Everything is made up ("Fake Video 0001", "fake_user_03"). Output is a pure function of
the arguments.
"""

import html
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus
from zoneinfo import ZoneInfo

import numpy as np

START = datetime(2024, 1, 1, tzinfo=UTC)
WINDOW_S = 730 * 86_400
LOCAL_ZONE = ZoneInfo("America/New_York")
N_CHANNELS = 15
N_USERS = 25
NNBSP = " "


def _times(rng: np.random.Generator, n: int) -> list[int]:
    t0 = int(START.timestamp())
    return sorted(int(t0 + x) for x in rng.integers(0, WINDOW_S, n))


def _mojibake(s: str) -> str:
    """What Instagram does to non-ASCII text: UTF-8 bytes shown as Latin-1."""
    return s.encode("utf-8").decode("latin-1")


# ---- Google ---------------------------------------------------------------------------


def _google_events(rng: np.random.Generator, scale: float) -> dict[str, list[dict]]:
    def n(base: int) -> int:
        return max(int(base * scale), 1)

    weights = 1 / np.arange(1, N_CHANNELS + 1)
    weights = weights / weights.sum()
    yt: list[dict] = []
    for i, ts in enumerate(_times(rng, n(400))):
        chan = int(rng.choice(N_CHANNELS, p=weights))
        yt.append(
            {
                "ts": ts,
                "verb": "Watched",
                "title": f"Fake Video {i:04d}",
                "url": f"https://www.youtube.com/watch?v=fakevid{i:05d}",
                "channel": (
                    f"Fake Channel {chan:02d}",
                    f"https://www.youtube.com/channel/UCfake{chan:02d}",
                ),
                "ads": False,
            }
        )
    yt_all = list(yt)
    for ev in yt[::10]:
        yt_all.append({**ev, "verb": "Liked", "ts": ev["ts"] + 30})
    yt_all.append(
        {
            "ts": _times(rng, 1)[0],
            "verb": "Watched",
            "title": "a video that has been removed",
            "url": None,
            "channel": None,
            "ads": False,
        }
    )
    for ts in _times(rng, n(40)):
        q = f"fake yt query {int(rng.integers(30)):02d}"
        yt_all.append(
            {
                "ts": ts,
                "verb": "Searched for",
                "title": q,
                "url": f"https://www.youtube.com/results?search_query={quote_plus(q)}",
                "channel": None,
                "ads": False,
            }
        )
    search: list[dict] = []
    for ts in _times(rng, n(150)):
        q = f"fake query {int(rng.integers(60)):02d}"
        search.append(
            {
                "ts": ts,
                "verb": "Searched for",
                "title": q,
                "url": f"https://www.google.com/search?q={quote_plus(q)}&hl=en",
                "channel": None,
                "ads": bool(rng.random() < 0.05),
            }
        )
    for i, ts in enumerate(_times(rng, n(60))):
        target = f"https://example.com/fake/{i:03d}?a=1&b=2"
        search.append(
            {
                "ts": ts,
                "verb": "Visited",
                "title": f"Fake Page {i:03d}",
                "url": f"https://www.google.com/url?q={target}&usg=fake",
                "channel": None,
                "ads": False,
            }
        )
    discover = [
        {
            "ts": ts,
            "verb": "Viewed",
            "title": f"Fake Card {i:03d}",
            "url": f"https://example.com/card/{i:03d}",
            "channel": None,
            "ads": False,
        }
        for i, ts in enumerate(_times(rng, n(30)))
    ]
    gemini = [
        {
            "ts": ts,
            "verb": "Prompted",
            "title": f"fake prompt {i}",
            "url": None,
            "channel": None,
            "ads": False,
        }
        for i, ts in enumerate(_times(rng, n(20)))
    ]
    return {
        "Search": search,
        "YouTube": yt_all,
        "Discover": discover,
        "Gemini Apps": gemini,
        "watch": yt,
    }


def _html_stamp(ts: int) -> str:
    local = datetime.fromtimestamp(ts, UTC).astimezone(LOCAL_ZONE)
    return local.strftime("%b %-d, %Y, %-I:%M:%S") + NNBSP + local.strftime("%p %Z")


def _html_entry(product: str, ev: dict) -> str:
    if ev["url"]:
        link = f'<a href="{html.escape(ev["url"])}">{html.escape(ev["title"])}</a>'
        body = f"{ev['verb']}&nbsp;{link}<br>"
    else:
        body = (
            f"{html.escape(ev['title'])}<br>"
            if ev["verb"] != "Watched"
            else (f"Watched {html.escape(ev['title'])}<br>")
        )
    if ev["channel"]:
        name, url = ev["channel"]
        body += f'<a href="{url}">{html.escape(name)}</a><br>'
    caption = f"<b>Products:</b><br>&emsp;{product}<br>"
    if ev["ads"]:
        caption += "<b>Details:</b><br>&emsp;From Google Ads<br>"
    return (
        '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp"><div class="mdl-grid">'
        '<div class="header-cell mdl-cell mdl-cell--12-col">'
        f'<p class="mdl-typography--title">{product}<br></p></div>'
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
        f"{body}{_html_stamp(ev['ts'])}</div>"
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 '
        'mdl-typography--text-right"></div>'
        f'<div class="content-cell mdl-cell mdl-cell--12-col mdl-typography--caption">{caption}</div>'
        "</div></div>"
    )


def _json_row(product: str, ev: dict) -> dict | None:
    if (
        ev["verb"] == "Liked"
    ):  # the JSON format has no verb, so a like is just another watch
        return None
    row: dict = {
        "header": product,
        "title": f"{ev['verb']} {ev['title']}",
        "time": datetime.fromtimestamp(ev["ts"], UTC).strftime(
            "%Y-%m-%dT%H:%M:%S.123Z"
        ),
        "products": [product],
    }
    if ev["url"]:
        row["titleUrl"] = ev["url"]
    if ev["channel"]:
        row["subtitles"] = [{"name": ev["channel"][0], "url": ev["channel"][1]}]
    if ev["ads"]:
        row["details"] = [{"name": "From Google Ads"}]
    return row


def _write_activity(path: Path, product: str, events: list[dict], fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "html":
        body = "".join(_html_entry(product, ev) for ev in events)
        text = (
            '<!DOCTYPE html><html><head><meta charset="utf-8"><title>My Activity</title></head>'
            f'<body><div class="mdl-grid">{body}</div></body></html>'
        )
    else:
        rows = [r for ev in events if (r := _json_row(product, ev))]
        text = json.dumps(rows, indent=2)
    path.write_text(text + "\n", encoding="utf-8")


def _write_google(dest: Path, rng: np.random.Generator, scale: float, fmt: str) -> None:
    events = _google_events(rng, scale)
    ext = "html" if fmt == "html" else "json"
    root = dest / "google" / "Takeout"
    for product in ("Search", "YouTube", "Discover", "Gemini Apps"):
        _write_activity(
            root / "My Activity" / product / f"MyActivity.{ext}",
            product,
            events[product],
            fmt,
        )
    history = root / "YouTube and YouTube Music" / "history"
    _write_activity(history / f"watch-history.{ext}", "YouTube", events["watch"], fmt)


# ---- Instagram ------------------------------------------------------------------------


def _user(rng: np.random.Generator) -> str:
    i = int(rng.integers(N_USERS))
    return _mojibake(f"fake_usér_{i:02d}") if i % 7 == 0 else f"fake_user_{i:02d}"


def _list_records(rng: np.random.Generator, n: int, tag: str) -> list[dict]:
    return [
        {
            "title": _user(rng),
            "string_list_data": [
                {
                    "href": f"https://www.instagram.com/p/FAKE{tag}{i:05d}/",
                    "value": "",
                    "timestamp": ts,
                }
            ],
        }
        for i, ts in enumerate(_times(rng, n))
    ]


def _map_records(
    rng: np.random.Generator, n: int, tag: str, **extra: str
) -> list[dict]:
    return [
        {
            "string_map_data": {
                "Author": {
                    "value": _user(rng),
                    "href": f"https://www.instagram.com/p/FAKE{tag}{i:05d}/",
                },
                "Time": {"timestamp": ts},
                **{k: {"value": v} for k, v in extra.items()},
            }
        }
        for i, ts in enumerate(_times(rng, n))
    ]


def _write_instagram(dest: Path, rng: np.random.Generator, scale: float) -> None:
    def n(base: int) -> int:
        return max(int(base * scale), 1)

    root = dest / "instagram"
    activity = root / "your_instagram_activity"
    files: dict[Path, dict] = {
        activity / "likes" / "liked_posts.json": {
            "likes_media_likes": _list_records(rng, n(120), "L")
        },
        activity / "saved" / "saved_posts.json": {
            "saved_saved_media": _list_records(rng, n(20), "S")
        },
        activity / "comments" / "post_comments_1.json": {
            "comments": _map_records(rng, n(20), "C", Comment="fake comment")
        },
        activity / "story_interactions" / "story_likes.json": {
            "story_activities_story_likes": _list_records(rng, n(30), "T")
        },
        root / "ads_information" / "ads_and_topics" / "posts_viewed.json": {
            "impressions_history_posts_seen": _map_records(rng, n(200), "V")
        },
        root / "ads_information" / "ads_and_topics" / "videos_watched.json": {
            "impressions_history_videos_watched": _map_records(rng, n(80), "W")
        },
        root / "ads_information" / "ads_and_topics" / "ads_viewed.json": {
            "impressions_history_ads_seen": _map_records(rng, n(60), "A")
        },
        root / "ads_information" / "ads_and_topics" / "ads_clicked.json": {
            "impressions_history_ads_clicked": _list_records(rng, n(6), "K")
        },
        root / "connections" / "followers_and_following" / "following.json": {
            "relationships_following": _list_records(rng, n(25), "F")
        },
        root / "logged_information" / "recent_searches" / "account_searches.json": {
            "searches_user": [
                {
                    "string_map_data": {
                        "Search": {"value": f"fake_search_{i}"},
                        "Time": {"timestamp": ts},
                    }
                }
                for i, ts in enumerate(_times(rng, n(15)))
            ]
        },
        # Must never be opened: messages are not on the allow-list.
        root
        / "your_instagram_activity"
        / "messages"
        / "inbox"
        / "fake_user_00_1"
        / "message_1.json": {
            "messages": [
                {
                    "sender_name": "fake_user_00",
                    "content": "fake message",
                    "timestamp_ms": 1,
                }
            ]
        },
    }
    for path, data in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def write_fake_sources(
    dest: Path,
    seed: int = 7,
    fmt: Literal["html", "json"] = "html",
    scale: float = 1.0,
) -> Path:
    """Write fake Google and Instagram exports under ``dest`` and return it.

    ``dest/google`` holds the Takeout (``fmt`` picks HTML, the real default, or JSON),
    ``dest/instagram`` holds the Instagram JSON. ``scale`` multiplies the event counts.
    """
    if fmt not in ("html", "json"):
        raise ValueError(f"unknown fmt: {fmt!r}")
    if scale <= 0:
        raise ValueError("scale must be positive")
    dest = Path(dest)
    rng = np.random.default_rng(seed)
    _write_google(dest, rng, scale, fmt)
    _write_instagram(dest, rng, scale)
    return dest
