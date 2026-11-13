import json

import pandas as pd
import pytest

from amber.decide import google_takeout
from amber.decide.events import EVENT_COLUMNS
from amber.decide.google_takeout import load_google

Z = "2024-01-01T12:00:00.123Z"


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def row(**kw):
    base = {"header": "Search", "title": "t", "time": Z, "products": ["Search"]}
    base.update(kw)
    return base


def load(tmp_path, rows, name="Takeout/My Activity/Search/MyActivity.json"):
    write(tmp_path / name, rows)
    return load_google(tmp_path)


def test_columns_and_ts_are_utc(tmp_path):
    df = load(tmp_path, [row(titleUrl="https://www.google.com/search?q=cats")])
    assert list(df.columns) == EVENT_COLUMNS
    assert str(df["ts"].dt.tz) == "UTC"
    assert df.attrs["n_files"] == 1
    assert df.attrs["n_skipped_files"] == 0
    assert df.attrs["n_bad_rows"] == 0


def test_youtube_watch_url(tmp_path):
    df = load(
        tmp_path,
        [
            row(
                header="YouTube",
                products=["YouTube"],
                title="Watched Some Video",
                titleUrl="https://www.youtube.com/watch?v=abc123",
                subtitles=[{"name": "Chan", "url": "https://www.youtube.com/channel/UCx"}],
            ),
            row(
                header="YouTube",
                products=["YouTube"],
                time="2024-01-02T12:00:00Z",
                titleUrl="https://youtu.be/xyz789?t=5",
                subtitles=[{"name": "Chan Two"}],
            ),
        ],
    )
    a, b = df.iloc[0], df.iloc[1]
    assert (a["source"], a["kind"], a["item_id"]) == ("youtube", "watch", "abc123")
    assert a["creator_id"] == "https://www.youtube.com/channel/UCx"
    assert a["text"] == "Watched Some Video"
    assert a["url"] == "https://www.youtube.com/watch?v=abc123"
    assert (b["kind"], b["item_id"], b["creator_id"]) == ("watch", "xyz789", "Chan Two")


def test_kind_comes_from_url_not_title(tmp_path):
    df = load(
        tmp_path,
        [
            row(
                header="YouTube",
                products=["YouTube"],
                title="Angesehen: Ein Video",
                titleUrl="https://www.youtube.com/watch?v=de1",
            )
        ],
    )
    assert df.iloc[0]["kind"] == "watch"


def test_youtube_search(tmp_path):
    df = load(
        tmp_path,
        [
            row(
                header="YouTube",
                products=["YouTube"],
                titleUrl="https://www.youtube.com/results?search_query=lo+fi%20beats",
            )
        ],
    )
    r = df.iloc[0]
    assert (r["source"], r["kind"], r["item_id"]) == ("youtube", "search", "lo fi beats")


def test_google_search_decodes_q(tmp_path):
    df = load(
        tmp_path,
        [
            row(titleUrl="https://www.google.com/search?q=red+panda%21"),
            row(time="2024-01-03T00:00:00Z", titleUrl="https://www.google.de/search?q=x&hl=de"),
        ],
    )
    assert (df.iloc[0]["source"], df.iloc[0]["kind"], df.iloc[0]["item_id"]) == (
        "google",
        "search",
        "red panda!",
    )
    assert df.iloc[1]["item_id"] == "x"


def test_other_search_url_is_visit(tmp_path):
    df = load(tmp_path, [row(titleUrl="https://example.com/page?a=1")])
    r = df.iloc[0]
    assert (r["source"], r["kind"], r["item_id"]) == ("google", "visit", "https://example.com/page?a=1")


def test_google_redirect_url_is_unwrapped_for_visit(tmp_path):
    df = load(tmp_path, [row(titleUrl="https://www.google.com/url?q=https://example.com/a&usg=zz")])
    r = df.iloc[0]
    assert (r["kind"], r["item_id"]) == ("visit", "https://example.com/a")


def test_discover_and_other_products(tmp_path):
    df = load(
        tmp_path,
        [
            row(header="Discover", products=["Discover"], titleUrl="https://example.com/news"),
            row(header="Maps", products=["Maps"], time="2024-01-02T00:00:00Z", title="Searched"),
            row(header="Shopping", products=["Shopping"], time="2024-01-03T00:00:00Z"),
        ],
    )
    assert list(df["kind"]) == ["discover", "activity", "activity"]
    assert list(df["source"]) == ["google"] * 3


def test_product_from_products_when_header_missing(tmp_path):
    r = row(products=["Maps"])
    del r["header"]
    df = load(tmp_path, [r])
    assert len(df) == 1 and df.iloc[0]["kind"] == "activity"


def test_removed_video_kept_as_watch_with_null_id(tmp_path):
    df = load(
        tmp_path,
        [row(header="YouTube", products=["YouTube"], title="Watched a video that has been removed")],
    )
    r = df.iloc[0]
    assert (r["source"], r["kind"]) == ("youtube", "watch")
    assert pd.isna(r["item_id"])


def test_allow_list_skips_gemini_and_counts(tmp_path):
    df = load(
        tmp_path,
        [
            row(header="Gemini Apps", products=["Gemini Apps"], title="SECRET PROMPT"),
            row(header="Chrome", products=["Chrome"], time="2024-01-02T00:00:00Z"),
            row(header="AI Mode", products=["AI Mode"], time="2024-01-03T00:00:00Z"),
        ],
    )
    assert len(df) == 1
    assert df.iloc[0]["kind"] == "activity"
    assert df.attrs["n_skipped_rows"] == 2
    assert "SECRET PROMPT" not in df.to_csv()


def test_from_ads(tmp_path):
    df = load(
        tmp_path,
        [
            row(details=[{"name": "From Google Ads"}], titleUrl="https://www.google.com/search?q=a"),
            row(time="2024-01-02T00:00:00Z", details=[{"name": "Something else"}]),
            row(time="2024-01-03T00:00:00Z"),
        ],
    )
    assert list(df["from_ads"]) == [True, False, False]
    assert df["from_ads"].dtype == bool


def test_files_found_recursively_by_name_only(tmp_path):
    watch = [row(header="YouTube", products=["YouTube"], titleUrl="https://youtu.be/aaa")]
    search = [
        row(
            header="YouTube",
            products=["YouTube"],
            time="2024-02-01T00:00:00Z",
            titleUrl="https://www.youtube.com/results?search_query=q",
        )
    ]
    write(tmp_path / "Takeout" / "YouTube und YouTube Music" / "Verlauf" / "watch-history.json", watch)
    write(tmp_path / "x" / "y" / "search-history.json", search)
    write(tmp_path / "Takeout" / "Meine Aktivitäten" / "Suche" / "MyActivity.json", [row()])
    write(tmp_path / "Takeout" / "other.json", [row()])
    df = load_google(tmp_path)
    assert df.attrs["n_files"] == 3
    assert len(df) == 3


def test_dedup_across_watch_history_and_my_activity(tmp_path):
    r = row(header="YouTube", products=["YouTube"], titleUrl="https://www.youtube.com/watch?v=dup")
    write(tmp_path / "a" / "watch-history.json", [r])
    write(tmp_path / "b" / "My Activity" / "YouTube" / "MyActivity.json", [r])
    df = load_google(tmp_path)
    assert len(df) == 1


def test_sorted_by_ts(tmp_path):
    df = load(
        tmp_path,
        [row(time="2024-03-01T00:00:00Z"), row(time="2024-01-01T00:00:00Z"), row(time=Z)],
    )
    assert df["ts"].is_monotonic_increasing


def test_bad_json_counted_not_raised(tmp_path):
    write(tmp_path / "a" / "MyActivity.json", "{not json")
    write(tmp_path / "b" / "watch-history.json", {"not": "a list"})
    write(tmp_path / "c" / "search-history.json", [row(header="YouTube", products=["YouTube"])])
    df = load_google(tmp_path)
    assert df.attrs["n_skipped_files"] == 2
    assert df.attrs["n_files"] == 1
    assert len(df) == 1


def test_bad_rows_counted(tmp_path):
    df = load(
        tmp_path,
        [row(time="garbage"), row(time=None), "not a dict", row(), 5],
    )
    assert len(df) == 1
    assert df.attrs["n_bad_rows"] == 4


def test_empty_folder_gives_empty_frame(tmp_path):
    df = load_google(tmp_path)
    assert len(df) == 0 and list(df.columns) == EVENT_COLUMNS
    assert df.attrs["n_files"] == 0


def test_module_has_no_network_imports():
    src = open(google_takeout.__file__).read()
    assert "requests" not in src and "urlopen" not in src


@pytest.mark.parametrize("time", ["2024-01-01T12:00:00Z", "2024-01-01T12:00:00.5Z"])
def test_time_variants(tmp_path, time):
    df = load(tmp_path, [row(time=time)])
    assert len(df) == 1
