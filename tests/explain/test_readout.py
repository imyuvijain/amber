import json
from datetime import UTC, datetime, timedelta

import pandas as pd
from fakes import FakeEncoder

from amber.explain.readout import (
    Readout,
    build_readout,
    concept_alignment,
    label_lift,
    load_concepts,
)
from amber.store.db import Choice
from amber.yous.base import You, text_direction
from amber.yous.stated import load_style_words

BRANDS = {
    "nike",
    "adidas",
    "gucci",
    "zara",
    "levis",
    "puma",
    "prada",
    "chanel",
    "reebok",
    "h&m",
}
T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _red_you(name="revealed", words=None):
    w = text_direction(FakeEncoder(), ["a photo of red clothing"])
    return You(name, w, {"words": words} if words is not None else {})


def _abstain(name):
    return You(name, None, {"reason": "test"})


def _choice(i, a, b, chosen):
    return Choice(
        pair_id=f"p{i}",
        user_id="u",
        domain="clothes",
        phase="showdown",
        a_id=a,
        b_id=b,
        chosen_id=chosen,
        shown_at=T0 + timedelta(seconds=i),
        answered_at=None
        if chosen is None
        else T0 + timedelta(seconds=i, milliseconds=500),
        retest_of=None,
    )


def _items(colours):
    return pd.DataFrame(
        {
            "id": [f"c:{i}" for i in range(len(colours))],
            "base_colour": colours,
            "article_type": ["Tshirts"] * len(colours),
        }
    )


# --- load_concepts --------------------------------------------------------------------


def test_concepts_are_about_120_unique_lowercase_and_cover_the_style_words():
    concepts = load_concepts()
    assert 110 <= len(concepts) <= 130
    assert len(set(concepts)) == len(concepts)
    assert set(load_style_words()) <= set(concepts)
    assert all(c == c.lower() and c.strip() == c for c in concepts)


def test_concepts_have_no_brand_names_or_junk():
    for concept in load_concepts():
        assert not (set(concept.replace("-", " ").split()) & BRANDS)
        assert all(ch.isalpha() or ch in " -" for ch in concept)
        assert 1 <= len(concept.split()) <= 3


# --- concept_alignment ----------------------------------------------------------------


def test_alignment_is_sorted_descending_and_ranks_red_above_blue():
    ranked = concept_alignment(_red_you(), FakeEncoder(), ["blue", "red", "minimal"])
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    names = [c for c, _ in ranked]
    assert names[0] == "red" and names.index("red") < names.index("blue")
    assert all(-1.0001 <= s <= 1.0001 for s in scores)


def test_alignment_is_empty_when_the_you_abstains():
    assert concept_alignment(_abstain("revealed"), FakeEncoder(), ["red", "blue"]) == []


def test_alignment_with_no_concepts_is_empty():
    assert concept_alignment(_red_you(), FakeEncoder(), []) == []


# --- label_lift -----------------------------------------------------------------------


def test_label_lift_is_chosen_over_shown_sorted_by_rate_then_label():
    items = _items(["Red", "Blue", "Red", "Blue", "Red", "Blue", "Black", "Black"])
    # Red always beats Blue in 3 pairs; Black (c:6, c:7) is shown 3 times and loses twice.
    choices = [
        _choice(0, "c:0", "c:1", "c:0"),
        _choice(1, "c:2", "c:3", "c:2"),
        _choice(2, "c:4", "c:5", "c:4"),
        _choice(3, "c:6", "c:0", "c:0"),
        _choice(4, "c:7", "c:2", "c:2"),
        _choice(5, "c:6", "c:3", "c:6"),
    ]
    lifts = dict(label_lift(choices, items, "base_colour", min_count=3))
    assert lifts == {"Red": 5 / 5, "Blue": 0.0, "Black": 1 / 3}
    order = [
        label for label, _ in label_lift(choices, items, "base_colour", min_count=3)
    ]
    assert order == ["Red", "Black", "Blue"]


def test_label_lift_drops_labels_shown_fewer_than_min_count_times():
    items = _items(["Red", "Blue", "Red", "Blue", "Green", "Blue"])
    choices = [
        _choice(0, "c:0", "c:1", "c:0"),
        _choice(1, "c:2", "c:3", "c:2"),
        _choice(2, "c:4", "c:5", "c:5"),
    ]
    labels = [
        label for label, _ in label_lift(choices, items, "base_colour", min_count=3)
    ]
    assert labels == ["Blue"]  # Red shown 2x, Green 1x, Blue 3x


def test_label_lift_ignores_unanswered_choices_and_missing_labels():
    items = _items(["Red", "Blue", None, "Blue"])
    items.loc[3, "base_colour"] = pd.NA
    choices = [
        _choice(0, "c:0", "c:1", "c:0"),
        _choice(1, "c:0", "c:1", "c:0"),
        _choice(2, "c:0", "c:1", None),  # unanswered: not counted
        _choice(3, "c:2", "c:0", "c:0"),  # c:2 has no label
    ]
    lifts = dict(label_lift(choices, items, "base_colour", min_count=1))
    assert lifts == {"Red": 1.0, "Blue": 0.0}


# --- build_readout --------------------------------------------------------------------


def _clothes(n=12):
    colours = ["Red", "Blue"] * (n // 2)
    return _items(colours)


def _red_wins(items, n=6):
    reds = [i for i, c in enumerate(items["base_colour"]) if c == "Red"]
    blues = [i for i, c in enumerate(items["base_colour"]) if c == "Blue"]
    return [
        _choice(k, f"c:{reds[k]}", f"c:{blues[k]}", f"c:{reds[k]}") for k in range(n)
    ]


def test_readout_with_abstaining_revealed_has_no_picks_or_surprises():
    yous = {
        "stated": _red_you("stated", words=["red", "minimal"]),
        "bigfive": _abstain("bigfive"),
        "art": _abstain("art"),
        "revealed": _abstain("revealed"),
    }
    out = build_readout(yous, [], _clothes(), FakeEncoder())
    assert isinstance(out, Readout)
    assert out.said == ["red", "minimal"]
    assert out.picks == [] and out.surprises == [] and out.art_says == []
    assert out.sentences == ["You said: red, minimal."]


def test_readout_with_nothing_at_all_is_empty():
    yous = {n: _abstain(n) for n in ("stated", "bigfive", "art", "revealed")}
    out = build_readout(yous, [], _clothes(), FakeEncoder())
    assert out == Readout(said=[], picks=[], art_says=[], surprises=[], sentences=[])


def test_readout_fields_for_a_red_picker():
    items = _clothes()
    yous = {
        "stated": _red_you("stated", words=["minimal"]),
        "bigfive": _abstain("bigfive"),
        "art": _red_you("art"),
        "revealed": _red_you("revealed"),
    }
    out = build_readout(yous, _red_wins(items), items, FakeEncoder())
    assert out.said == ["minimal"]
    assert "red" in out.picks[:5]
    assert "red" in out.picks  # label lift: Red won 6 of 6
    assert len(out.picks) == len(set(out.picks))
    assert all(p == p.lower() for p in out.picks)
    assert len(out.art_says) == 5 and "red" in out.art_says
    assert 1 <= len(out.surprises) <= 3
    assert not set(out.surprises) & set(out.said)
    assert out.sentences[0] == "You said: minimal."
    assert any(s.startswith("You pick: ") for s in out.sentences)
    assert any(s.startswith("Your art taste says: ") for s in out.sentences)
    assert any(
        s.startswith("Surprise: you pick ") and s.endswith(" but never said it.")
        for s in out.sentences
    )


def test_surprises_never_repeat_a_said_word():
    items = _clothes()
    yous = {
        "stated": _red_you("stated", words=["red"]),
        "bigfive": _abstain("bigfive"),
        "art": _abstain("art"),
        "revealed": _red_you("revealed"),
    }
    out = build_readout(yous, _red_wins(items), items, FakeEncoder())
    assert "red" not in out.surprises
    assert not set(out.surprises) & set(out.said)
    assert not any("art taste" in s for s in out.sentences)  # art abstains: omitted


def test_readout_serialises_to_json():
    from dataclasses import asdict

    out = build_readout(
        {n: _abstain(n) for n in ("stated", "bigfive", "art", "revealed")},
        [],
        _clothes(),
        FakeEncoder(),
    )
    json.dumps(asdict(out))
