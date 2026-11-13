import json

import pytest

from amber.decide import instagram
from amber.decide.events import EVENT_COLUMNS
from amber.decide.instagram import fix_mojibake, load_instagram

T1, T2, T3 = 1_700_000_000, 1_700_100_000, 1_700_200_000


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data if isinstance(data, str) else json.dumps(data), encoding="utf-8")
    return path


def list_rec(title, href, ts, value=""):
    return {
        "title": title,
        "string_list_data": [{"href": href, "value": value, "timestamp": ts}],
    }


def map_rec(ts, **fields):
    data = {k: {"value": v} for k, v in fields.items()}
    data["Time"] = {"timestamp": ts}
    return {"string_map_data": data}


def test_mojibake_repair():
    assert fix_mojibake("â\u0080\u0099") == "’"
    assert fix_mojibake("plain ascii") == "plain ascii"
    assert fix_mojibake("already fine ’") == "already fine ’"  # not latin-1 encodable
    assert fix_mojibake("café") == "café"  # latin-1 but not valid utf-8
    assert fix_mojibake(None) is None


def test_liked_posts_string_list_data(tmp_path):
    write(
        tmp_path / "your_instagram_activity" / "likes" / "liked_posts.json",
        {"likes_media_likes": [list_rec("someauthor", "https://www.instagram.com/p/AAA/", T1)]},
    )
    df = load_instagram(tmp_path)
    assert list(df.columns) == EVENT_COLUMNS
    r = df.iloc[0]
    assert (r["source"], r["kind"]) == ("instagram", "like")
    assert r["item_id"] == "https://www.instagram.com/p/AAA/"
    assert r["creator_id"] == "someauthor"
    assert r["ts"].timestamp() == T1
    assert str(df["ts"].dt.tz) == "UTC"
    assert not r["from_ads"]
    assert df.attrs["n_files"] == 1 and df.attrs["n_bad_rows"] == 0


def test_posts_viewed_string_map_data(tmp_path):
    write(
        tmp_path / "ads_and_content" / "posts_viewed.json",
        {
            "impressions_history_posts_seen": [
                map_rec(T1, Author="seen_author"),
                {
                    "string_map_data": {
                        "Author": {"value": "other", "href": "https://www.instagram.com/p/BBB/"},
                        "Time": {"timestamp": T2},
                    }
                },
            ]
        },
    )
    df = load_instagram(tmp_path)
    assert list(df["kind"]) == ["view_post", "view_post"]
    assert list(df["creator_id"]) == ["seen_author", "other"]
    assert df.iloc[1]["item_id"] == "https://www.instagram.com/p/BBB/"


def test_bare_list_top_level(tmp_path):
    write(tmp_path / "videos_watched.json", [map_rec(T1, Author="v")])
    df = load_instagram(tmp_path)
    assert (df.iloc[0]["kind"], df.iloc[0]["creator_id"]) == ("view_video", "v")


@pytest.mark.parametrize(
    "stem,kind",
    [
        ("liked_posts", "like"),
        ("likes_media_likes", "like"),
        ("saved_posts", "save"),
        ("post_comments", "comment"),
        ("post_comments_1", "comment"),
        ("comments", "comment"),
        ("comments_reels", "comment"),
        ("story_likes", "story_interaction"),
        ("polls", "story_interaction"),
        ("quizzes", "story_interaction"),
        ("emoji_sliders", "story_interaction"),
        ("questions", "story_interaction"),
        ("posts_viewed", "view_post"),
        ("videos_watched", "view_video"),
        ("ads_viewed", "ad_view"),
        ("ads_clicked", "ad_click"),
        ("following", "follow"),
        ("account_searches", "ig_search"),
        ("word_or_phrase_searches", "ig_search"),
        ("tag_searches", "ig_search"),
        ("link_history", "link_open"),
    ],
)
def test_stem_to_kind(tmp_path, stem, kind):
    write(tmp_path / "a" / f"{stem}.json", {"k": [list_rec("who", "https://x/1", T1)]})
    df = load_instagram(tmp_path)
    assert list(df["kind"]) == [kind]


def test_ads_files_set_from_ads(tmp_path):
    write(tmp_path / "ads_viewed.json", {"k": [map_rec(T1, Author="adv")]})
    write(tmp_path / "ads_clicked.json", {"k": [map_rec(T2, Author="adc")]})
    write(tmp_path / "liked_posts.json", {"k": [list_rec("a", "h", T3)]})
    df = load_instagram(tmp_path)
    assert list(df["from_ads"]) == [True, True, False]


def test_creator_fallbacks_and_text(tmp_path):
    write(
        tmp_path / "a" / "following.json",
        {"k": [{"string_list_data": [{"value": "valuename", "href": "https://i/u", "timestamp": T1}]}]},
    )
    write(
        tmp_path / "b" / "post_comments.json",
        {"k": [map_rec(T2, Comment="nice one", **{"Media Owner": "owner"})]},
    )
    write(tmp_path / "c" / "account_searches.json", {"k": [map_rec(T3, Search="query text")]})
    df = load_instagram(tmp_path).set_index("kind")
    assert df.loc["follow", "creator_id"] == "valuename"
    assert df.loc["comment", "text"] == "nice one"
    assert df.loc["ig_search", "text"] == "query text"
    assert df.loc["ig_search", "item_id"] == "query text"


def test_mojibake_repaired_in_rows(tmp_path):
    bad = "itâ\u0080\u0099s".encode("latin-1", "ignore").decode("latin-1")
    write(tmp_path / "liked_posts.json", {"k": [list_rec("itâ\u0080\u0099s", "h", T1)]})
    df = load_instagram(tmp_path)
    assert bad  # sanity: fixture is a str
    assert df.iloc[0]["creator_id"] == "it’s"


def test_record_without_timestamp_is_bad_row(tmp_path):
    write(
        tmp_path / "liked_posts.json",
        {"k": [{"title": "x", "string_list_data": [{"href": "h"}]}, list_rec("y", "h2", T1), 3]},
    )
    df = load_instagram(tmp_path)
    assert len(df) == 1
    assert df.attrs["n_bad_rows"] == 2


def test_unknown_files_skipped_and_counted(tmp_path):
    write(tmp_path / "profile" / "personal_information.json", {"k": [list_rec("a", "h", T1)]})
    write(tmp_path / "liked_posts.json", {"k": [list_rec("a", "h", T1)]})
    df = load_instagram(tmp_path)
    assert len(df) == 1
    assert df.attrs["n_files"] == 1
    assert df.attrs["n_skipped_files"] == 1


def test_messages_are_never_opened(tmp_path, monkeypatch):
    msg = write(
        tmp_path / "messages" / "inbox" / "x_123" / "message_1.json",
        {"messages": [{"sender_name": "SECRET", "content": "hi", "timestamp_ms": 1}]},
    )
    write(tmp_path / "media" / "posts" / "post_1.json", {"k": []})
    write(tmp_path / "liked_posts.json", {"k": [list_rec("a", "h", T1)]})
    opened = []
    real = instagram._read_json

    def spy(f):
        opened.append(f.name)
        return real(f)

    monkeypatch.setattr(instagram, "_read_json", spy)
    df = load_instagram(tmp_path)
    assert opened == ["liked_posts.json"]
    assert msg.exists()
    assert "SECRET" not in df.to_csv()
    assert df.attrs["n_skipped_files"] == 2


def test_bad_json_counted_not_raised(tmp_path):
    write(tmp_path / "a" / "liked_posts.json", "{broken")
    write(tmp_path / "b" / "saved_posts.json", "[]")
    write(tmp_path / "c" / "following.json", '"a string"')
    write(tmp_path / "d" / "polls.json", {"k": [list_rec("a", "h", T1)]})
    df = load_instagram(tmp_path)
    assert len(df) == 1
    assert df.attrs["n_skipped_files"] == 2  # broken and non-container
    assert df.attrs["n_files"] == 2  # polls and the empty list


def test_dedup_and_sort(tmp_path):
    rec = list_rec("a", "https://p/1", T2)
    write(tmp_path / "x" / "liked_posts.json", {"k": [rec, list_rec("b", "https://p/2", T1)]})
    write(tmp_path / "y" / "liked_posts.json", {"k": [rec]})
    df = load_instagram(tmp_path)
    assert len(df) == 2
    assert df["ts"].is_monotonic_increasing


def test_empty_folder(tmp_path):
    df = load_instagram(tmp_path)
    assert len(df) == 0 and list(df.columns) == EVENT_COLUMNS
