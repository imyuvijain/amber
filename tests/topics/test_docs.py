import pandas as pd

from amber.topics.docs import build_docs, clean_text


def _ev(**kw):
    base = {
        "ts": pd.Timestamp("2024-01-01", tz="UTC"),
        "source": "youtube",
        "kind": "watch",
        "item_id": None,
        "creator_id": None,
        "text": "a title",
        "url": None,
        "from_ads": False,
    }
    return {**base, **kw}


def test_clean_text_strips_verbs_urls_and_emoji_only():
    assert clean_text("Watched How to bake bread") == "How to bake bread"
    assert clean_text("Searched for quantum tunnelling") == "quantum tunnelling"
    assert clean_text("https://example.com/x") is None
    assert clean_text("🔥🔥") is None
    assert clean_text("  ") is None
    assert clean_text(None) is None


def test_build_docs_weights_deliberate_actions_and_drops_ads():
    ev = pd.DataFrame(
        [
            _ev(kind="watch", text="Watched bread"),
            _ev(source="google", kind="search", text="sourdough starter"),
            _ev(source="instagram", kind="like", text="crumb shot"),
            _ev(source="instagram", kind="save", text="recipe card"),
            _ev(source="instagram", kind="ad_view", text="buy this", from_ads=True),
            _ev(source="instagram", kind="see_more", text=None),
            _ev(source="instagram", kind="like_comment", text="😂"),
        ]
    )
    docs = build_docs(ev)
    assert list(docs["kind"]) == ["watch", "search", "like", "save"]
    assert list(docs["deliberate"]) == [False, True, True, True]
    w = dict(zip(docs["kind"], docs["weight"]))
    assert w["watch"] < w["like"] <= w["search"] <= w["save"]
    assert docs.loc[0, "text"] == "bread"
