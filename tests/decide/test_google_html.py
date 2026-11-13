import pytest

from amber.decide.events import EVENT_COLUMNS
from amber.decide.google_takeout import load_google

NNBSP = " "
PAGE = "<html><head><title>My Activity</title></head><body><div class=\"mdl-grid\">{}</div></body></html>"


def entry(header, content, caption=None):
    cap = caption if caption is not None else f"<b>Products:</b><br>&emsp;{header}<br>"
    return (
        '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
        '<div class="mdl-grid">'
        '<div class="header-cell mdl-cell mdl-cell--12-col">'
        f'<p class="mdl-typography--title">{header}<br></p></div>'
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
        f"{content}</div>"
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1 '
        'mdl-typography--text-right"></div>'
        '<div class="content-cell mdl-cell mdl-cell--12-col mdl-typography--caption">'
        f"{cap}</div></div></div>"
    )


def date(s="Nov 13, 2026, 2:47:58 PM EST"):
    return s.replace(" PM", f"{NNBSP}PM").replace(" AM", f"{NNBSP}AM")


def page(tmp_path, rel, *entries):
    f = tmp_path / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(PAGE.format("".join(entries)), encoding="utf-8")
    return f


def watch(vid="abc123", title="Fake Video", chan="Fake Channel", when=None, verb="Watched"):
    return entry(
        "YouTube",
        f'{verb}&nbsp;<a href="https://www.youtube.com/watch?v={vid}">{title}</a><br>'
        f'<a href="https://www.youtube.com/channel/UCfake1">{chan}</a><br>{date(when or "Nov 13, 2026, 2:47:58 PM EST")}',
    )


def test_watch_entry(tmp_path):
    page(tmp_path, "Takeout/My Activity/YouTube/MyActivity.html", watch())
    df = load_google(tmp_path)
    assert list(df.columns) == EVENT_COLUMNS
    r = df.iloc[0]
    assert (r["source"], r["kind"], r["item_id"]) == ("youtube", "watch", "abc123")
    assert r["text"] == "Fake Video"
    assert r["creator_id"] == "https://www.youtube.com/channel/UCfake1"
    assert r["url"] == "https://www.youtube.com/watch?v=abc123"
    assert not r["from_ads"]
    # 2:47:58 PM EST -> 19:47:58 UTC (wall clock in America/New_York)
    assert r["ts"].isoformat() == "2026-11-13T19:47:58+00:00"
    assert df.attrs["n_files"] == 1 and df.attrs["n_bad_rows"] == 0


def test_liked_verb_gives_like_kind(tmp_path):
    page(
        tmp_path,
        "My Activity/YouTube/MyActivity.html",
        watch(vid="liked1", verb="Liked"),
        watch(vid="viewed1", verb="Viewed", when="Nov 14, 2026, 2:47:58 PM EST"),
    )
    df = load_google(tmp_path)
    assert list(df["kind"]) == ["like", "watch"]
    assert df.iloc[0]["item_id"] == "liked1"
    assert df.iloc[0]["source"] == "youtube"


def test_search_and_amp_hrefs(tmp_path):
    page(
        tmp_path,
        "My Activity/Search/MyActivity.html",
        entry(
            "Search",
            'Searched for&nbsp;<a href="https://www.google.com/search?q=red+panda&amp;hl=en">red panda</a><br>'
            + date("Nov 1, 2026, 9:05:00 AM EST"),
        ),
        entry(
            "Search",
            'Visited&nbsp;<a href="https://www.google.com/url?q=https://example.com/a?x=1&amp;y=2&amp;usg=zz">Fake Page</a><br>'
            + date("Nov 2, 2026, 9:05:00 AM EST"),
        ),
    )
    df = load_google(tmp_path)
    a, b = df.iloc[0], df.iloc[1]
    assert (a["source"], a["kind"], a["item_id"]) == ("google", "search", "red panda")
    assert (b["kind"], b["item_id"]) == ("visit", "https://example.com/a?x=1")
    assert a["ts"].isoformat() == "2026-11-01T14:05:00+00:00"


def test_youtube_search_and_post(tmp_path):
    page(
        tmp_path,
        "My Activity/YouTube/MyActivity.html",
        entry(
            "YouTube",
            'Searched for&nbsp;<a href="https://www.youtube.com/results?search_query=lo+fi">lo fi</a><br>'
            + date(),
        ),
        entry(
            "YouTube",
            'Viewed&nbsp;<a href="https://www.youtube.com/post/Ugkx123">A post</a><br>'
            + date("Nov 14, 2026, 1:00:00 PM EST"),
        ),
    )
    df = load_google(tmp_path)
    assert (df.iloc[0]["kind"], df.iloc[0]["item_id"]) == ("search", "lo fi")
    assert (df.iloc[1]["kind"], df.iloc[1]["source"]) == ("activity", "youtube")


def test_no_link_entries(tmp_path):
    page(
        tmp_path,
        "My Activity/YouTube/MyActivity.html",
        entry("YouTube", "Watched a video that has been removed<br>" + date()),
        entry("Maps", "Used Maps<br>" + date("Nov 3, 2026, 8:00:00 AM EST")),
    )
    df = load_google(tmp_path)  # sorted by time: the Maps entry is earlier
    b, a = df.iloc[0], df.iloc[1]
    assert (a["source"], a["kind"]) == ("youtube", "watch")
    assert a["item_id"] is None or a["item_id"] != a["item_id"]
    assert (b["source"], b["kind"]) == ("google", "activity")
    assert b["text"] == "Used Maps"


def test_from_ads_detail(tmp_path):
    page(
        tmp_path,
        "My Activity/Search/MyActivity.html",
        entry(
            "Search",
            'Searched for&nbsp;<a href="https://www.google.com/search?q=a">a</a><br>' + date(),
            caption="<b>Products:</b><br>&emsp;Search<br><b>Details:</b><br>&emsp;From Google Ads<br>",
        ),
        entry(
            "Search",
            'Searched for&nbsp;<a href="https://www.google.com/search?q=b">b</a><br>'
            + date("Nov 14, 2026, 2:47:58 PM EST"),
        ),
    )
    df = load_google(tmp_path)
    assert list(df["from_ads"]) == [True, False]


def test_product_fallback_to_parent_folder_and_allow_list(tmp_path):
    bare = (
        '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
        "Used Maps<br>" + date() + "</div></div>"
    )
    for sub in ("Maps", "Chrome"):
        f = tmp_path / "My Activity" / sub / "MyActivity.html"
        f.parent.mkdir(parents=True)
        f.write_text(bare, encoding="utf-8")
    df = load_google(tmp_path)
    assert len(df) == 1
    assert df.attrs["n_skipped_rows"] == 1


def test_gemini_folder_never_opened(tmp_path, monkeypatch):
    f = page(
        tmp_path,
        "Takeout/My Activity/Gemini Apps/MyActivity.html",
        entry("Gemini Apps", "Prompted SECRET PROMPT<br>" + date()),
    )
    (tmp_path / "Takeout/My Activity/Gemini Apps/MyActivity.json").write_text("[]")
    from pathlib import Path

    real = Path.read_text
    opened = []

    def spy(self, *a, **k):
        opened.append(str(self))
        return real(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", spy)
    df = load_google(tmp_path)
    assert not any("Gemini" in p for p in opened)
    assert f.exists() and len(df) == 0
    assert df.attrs["n_skipped_files"] == 2


def test_watch_and_search_history_html(tmp_path):
    page(tmp_path, "Takeout/YouTube/history/watch-history.html", watch(vid="w1"))
    page(
        tmp_path,
        "Takeout/YouTube/history/search-history.html",
        entry(
            "YouTube",
            'Searched for&nbsp;<a href="https://www.youtube.com/results?search_query=q">q</a><br>'
            + date("Nov 15, 2026, 1:00:00 AM EST"),
        ),
    )
    df = load_google(tmp_path)
    assert list(df["kind"]) == ["watch", "search"]


def test_history_html_without_header_defaults_to_youtube(tmp_path):
    bare = (
        '<div class="outer-cell mdl-cell mdl-cell--12-col mdl-shadow--2dp">'
        '<div class="content-cell mdl-cell mdl-cell--6-col mdl-typography--body-1">'
        'Watched&nbsp;<a href="https://youtube.com/watch?v=nohdr">T</a><br>' + date() + "</div></div>"
    )
    f = tmp_path / "Takeout" / "history" / "watch-history.html"
    f.parent.mkdir(parents=True)
    f.write_text(bare, encoding="utf-8")
    df = load_google(tmp_path)
    assert (df.iloc[0]["source"], df.iloc[0]["item_id"]) == ("youtube", "nohdr")


@pytest.mark.parametrize(
    "stamp,utc",
    [
        ("Jul 4, 2026, 12:30:00 AM EDT", "2026-07-04T04:30:00+00:00"),
        ("Jan 4, 2026, 12:30:00 PM PST", "2026-01-04T20:30:00+00:00"),
        ("Jan 4, 2026, 9:00:00 AM UTC", "2026-01-04T09:00:00+00:00"),
        ("Jan 4, 2026, 9:00:00 AM IST", "2026-01-04T03:30:00+00:00"),
        ("Jul 4, 2026, 9:00:00 AM CEST", "2026-07-04T07:00:00+00:00"),
        ("Jan 4, 2026, 9:00:00 AM JST", "2026-01-04T00:00:00+00:00"),
    ],
)
def test_timezone_abbreviations(tmp_path, stamp, utc):
    page(tmp_path, "My Activity/Search/MyActivity.html", entry("Search", "Used Search<br>" + date(stamp)))
    assert load_google(tmp_path).iloc[0]["ts"].isoformat() == utc


def test_unknown_timezone_and_missing_date_are_bad_rows(tmp_path):
    page(
        tmp_path,
        "My Activity/Search/MyActivity.html",
        entry("Search", "Used Search<br>" + date("Nov 13, 2026, 2:47:58 PM XYZ")),
        entry("Search", "Used Search<br>no date here"),
        entry("Search", "Used Search<br>" + date()),
    )
    df = load_google(tmp_path)
    assert len(df) == 1
    assert df.attrs["n_bad_rows"] == 2


def test_html_and_json_mix_and_non_takeout_html(tmp_path):
    page(tmp_path, "a/YouTube/MyActivity.html", watch(vid="h1"))
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "MyActivity.json").write_text(
        '[{"header":"YouTube","time":"2024-01-01T00:00:00Z","titleUrl":"https://youtu.be/j1"}]'
    )
    (tmp_path / "c").mkdir()
    (tmp_path / "c" / "MyActivity.html").write_text("<html>nothing</html>")
    df = load_google(tmp_path)
    assert sorted(df["item_id"]) == ["h1", "j1"]
    assert df.attrs["n_files"] == 2 and df.attrs["n_skipped_files"] == 1


def test_many_entries_is_fast(tmp_path):
    page(
        tmp_path,
        "My Activity/YouTube/MyActivity.html",
        *[watch(vid=f"v{i}", when=f"Nov 13, 2026, 2:{i % 60:02d}:{(i // 60) % 60:02d} PM EST") for i in range(3000)],
    )
    assert len(load_google(tmp_path)) == 3000
