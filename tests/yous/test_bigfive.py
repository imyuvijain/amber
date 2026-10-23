import numpy as np
import pytest
from fakes import play_art, play_clothes

from amber.store.db import save_answer
from amber.yous.base import load_banned_words, text_direction
from amber.yous.bigfive import (
    bridge_prompts,
    build_bigfive,
    load_bridge,
    load_ipip,
    score_ipip,
)
from amber.yous.registry import BUILDERS
from amber.yous.stated import load_style_words

TRAITS = {"O", "C", "E", "A", "N"}

# Published keying (Donnellan, Oswald, Baird & Lucas, 2006): text -> (trait, keyed).
PUBLISHED = {
    "Am the life of the party": ("E", 1),
    "Don't talk a lot": ("E", -1),
    "Talk to a lot of different people at parties": ("E", 1),
    "Keep in the background": ("E", -1),
    "Sympathize with others' feelings": ("A", 1),
    "Am not interested in other people's problems": ("A", -1),
    "Feel others' emotions": ("A", 1),
    "Am not really interested in others": ("A", -1),
    "Get chores done right away": ("C", 1),
    "Often forget to put things back in their proper place": ("C", -1),
    "Like order": ("C", 1),
    "Make a mess of things": ("C", -1),
    "Have frequent mood swings": ("N", 1),
    "Am relaxed most of the time": ("N", -1),
    "Get upset easily": ("N", 1),
    "Seldom feel blue": ("N", -1),
    "Have a vivid imagination": ("O", 1),
    "Am not interested in abstract ideas": ("O", -1),
    "Have difficulty understanding abstract ideas": ("O", -1),
    "Do not have a good imagination": ("O", -1),
}


def _answer(ctx, user, answers: dict[str, int]) -> None:
    for item_id, value in answers.items():
        save_answer(ctx.conn, user, "ipip", item_id, str(value))


def _extremes(trait: str, high: bool) -> dict[str, int]:
    """Answers that put `trait` at +1 (high) or -1 (low)."""
    top = 5 if high else 1
    return {
        i.id: (top if i.keyed == 1 else 6 - top)
        for i in load_ipip()
        if i.trait == trait
    }


def _neutral_except(trait: str, high: bool = True) -> dict[str, int]:
    """Every item answered: `trait` at an extreme, the rest at 3 (score 0)."""
    answers = {i.id: 3 for i in load_ipip()}
    answers.update(_extremes(trait, high))
    return answers


# --- items and scoring -----------------------------------------------------------


def test_ipip_has_the_twenty_published_items_and_keys():
    items = load_ipip()
    assert len(items) == 20 and len({i.id for i in items}) == 20
    assert {i.text: (i.trait, i.keyed) for i in items} == PUBLISHED


def test_ipip_items_are_interleaved_by_trait():
    traits = [i.trait for i in load_ipip()]
    assert traits == ["E", "A", "C", "N", "O"] * 4


def test_ipip_returns_a_copy():
    load_ipip().clear()
    assert len(load_ipip()) == 20


def test_score_ipip_keyed_extremes_hit_plus_and_minus_one():
    for trait in TRAITS:
        assert score_ipip(_extremes(trait, high=True)) == {trait: 1.0}
        assert score_ipip(_extremes(trait, high=False)) == {trait: -1.0}


def test_score_ipip_reverse_scores_minus_keyed_items():
    # Answering 5 to everything: E has two + and two - items, so it nets to the midpoint.
    all_fives = {i.id: 5 for i in load_ipip()}
    scores = score_ipip(all_fives)
    assert scores["E"] == 0.0
    # O has one + and three - items: mean of (5, 1, 1, 1) = 2 -> (2 - 3) / 2 = -0.5.
    assert scores["O"] == -0.5


def test_score_ipip_scores_stay_in_range_and_omit_unanswered_traits():
    rng = np.random.default_rng(7)
    e_and_a = [i.id for i in load_ipip() if i.trait in {"E", "A"}]
    for _ in range(50):
        scores = score_ipip({i: int(rng.integers(1, 6)) for i in e_and_a})
        assert set(scores) == {"E", "A"}
        assert all(-1.0 <= s <= 1.0 for s in scores.values())


def test_score_ipip_rejects_unknown_items_and_out_of_range_values():
    first = load_ipip()[0].id
    for bad in ({"nope": 3}, {first: 0}, {first: 6}):
        with pytest.raises(ValueError):
            score_ipip(bad)


# --- the bridge file and the banned words ----------------------------------------


def test_bridge_has_four_adjectives_per_pole():
    bridge = load_bridge()
    assert set(bridge) == TRAITS
    for poles in bridge.values():
        assert set(poles) == {"high", "low"}
        for words in poles.values():
            assert len(words) == 4 and len(set(words)) == 4


def test_no_bridge_adjective_is_banned():
    banned = load_banned_words()
    used = {w for poles in load_bridge().values() for ws in poles.values() for w in ws}
    assert used and not used & banned


def test_banned_words_cover_style_words_and_are_lowercase():
    banned = load_banned_words()
    assert len(banned) >= 80
    assert all(w == w.lower().strip() and w for w in banned)
    assert set(load_style_words()) <= banned
    assert {"shirt", "red", "warm", "cold", "denim", "striped"} <= banned


def test_bridge_prompts_are_three_rotations():
    prompts = bridge_prompts(["a", "b", "c", "d"])
    assert prompts == [
        "clothing worn by someone who is a, b and c",
        "clothing worn by someone who is b, c and d",
        "clothing worn by someone who is c, d and a",
    ]


# --- the You ---------------------------------------------------------------------


def test_bigfive_is_registered():
    assert BUILDERS["bigfive"] is build_bigfive


def test_bigfive_abstains_when_skipped(ctx, user):
    you = build_bigfive(ctx.conn, user, ctx)
    assert you.name == "bigfive" and you.w is None
    assert you.meta == {"reason": "skipped"}


def test_bigfive_abstains_under_ten_answers(ctx, user):
    nine = dict(list(_neutral_except("E").items())[:9])
    _answer(ctx, user, nine)
    assert build_bigfive(ctx.conn, user, ctx).meta == {"reason": "skipped"}
    _answer(ctx, user, dict(list(_neutral_except("E").items())[9:10]))
    assert build_bigfive(ctx.conn, user, ctx).w is not None


def test_high_e_person_points_toward_e_high_text(ctx, user):
    _answer(ctx, user, _neutral_except("E", high=True))
    you = build_bigfive(ctx.conn, user, ctx)
    assert np.isclose(np.linalg.norm(you.w), 1.0, atol=1e-5)
    e_high = text_direction(ctx.encoder, bridge_prompts(load_bridge()["E"]["high"]))
    assert float(you.w @ e_high) > 0
    assert you.meta == {"traits": {"E": 1.0, "A": 0.0, "C": 0.0, "N": 0.0, "O": 0.0}}


def test_low_e_person_points_away_from_e_high_text(ctx, user):
    _answer(ctx, user, _neutral_except("E", high=False))
    you = build_bigfive(ctx.conn, user, ctx)
    e_high = text_direction(ctx.encoder, bridge_prompts(load_bridge()["E"]["high"]))
    assert float(you.w @ e_high) < 0


def test_bigfive_abstains_when_every_trait_is_neutral(ctx, user):
    _answer(ctx, user, {i.id: 3 for i in load_ipip()})
    you = build_bigfive(ctx.conn, user, ctx)
    assert you.w is None and you.meta["reason"] == "no direction"


def test_bigfive_ignores_clothing_picks(ctx, user):
    # Leak rule. build_bigfive takes no choices argument to reject, so (as for stated)
    # it is checked by invariance: art and clothes picks leave the result unchanged.
    _answer(ctx, user, _neutral_except("C"))
    before = build_bigfive(ctx.conn, user, ctx)
    play_art(ctx, user, prefer="red", n=ctx.settings.art_train_pairs)
    play_clothes(ctx, user, prefer="blue", n=8)
    after = build_bigfive(ctx.conn, user, ctx)
    assert after.meta == before.meta
    assert np.array_equal(after.w, before.w)
