import json
from dataclasses import replace

import numpy as np
import pytest
from fakes import (
    play_art,
    play_clothes,
    preds,
    revealed_p_a_from,
)

from amber.eval.metrics import score_showdown
from amber.flow import (
    CatalogExhausted,
    allowed_ids,
    next_pair,
    pair_rng,
    progress,
)
from amber.store.db import (
    answer_pair,
    get_predictions,
    list_choices,
    load_yous,
    save_answer,
    shown_ids,
)
from amber.yous.base import You, predict
from amber.yous.registry import BUILDERS, FROZEN_YOUS


def _filters(ctx, user, genders=None, sections=None):
    if genders is not None:
        save_answer(ctx.conn, user, "filters", "genders", json.dumps(genders))
    if sections is not None:
        save_answer(ctx.conn, user, "filters", "sections", json.dumps(sections))


# --- resuming, limits, no repeats -----------------------------------------------


def test_next_pair_resumes_open_pair(ctx, user):
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert next_pair(ctx.conn, user, "clothes", "showdown", ctx) == p


def test_clothes_phase_ends_at_setting(ctx, user):
    play_clothes(ctx, user, n=ctx.settings.clothes_pairs)
    assert next_pair(ctx.conn, user, "clothes", "showdown", ctx) is None


def test_art_train_phase_ends_at_setting(ctx, user):
    play_art(ctx, user, n=ctx.settings.art_train_pairs)
    assert next_pair(ctx.conn, user, "art", "train", ctx) is None


def test_pairs_never_repeat_items(ctx, user):
    seen = [i for c in play_clothes(ctx, user, n=12) for i in (c.a_id, c.b_id)]
    assert len(seen) == len(set(seen))


def test_art_pairs_are_art_and_do_not_repeat(ctx, user):
    chosen = play_art(ctx, user, n=6)
    seen = [i for c in chosen for i in (c.a_id, c.b_id)]
    assert all(i.startswith("a:") for i in seen) and len(seen) == len(set(seen))
    assert shown_ids(ctx.conn, user, "clothes") == set()


def test_art_pairs_ignore_clothes_filters(ctx, user):
    _filters(ctx, user, genders=["Men"], sections=[])  # nothing allowed in clothes
    chosen = play_art(ctx, user, prefer="red", n=ctx.settings.art_train_pairs)
    assert len(chosen) == ctx.settings.art_train_pairs
    assert all(c.domain == "art" and c.phase == "train" for c in chosen)


def test_art_active_pairs_follow_the_art_basis(ctx, user):
    # after a few red picks the active chooser should stay on art ids and keep going
    play_art(ctx, user, prefer="red", n=3)
    pair = next_pair(ctx.conn, user, "art", "train", ctx)
    assert pair.a_id.startswith("a:") and pair.b_id.startswith("a:")


def test_clothes_pairs_reproducible_per_user(ctx, user):
    first = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    # same user/domain/phase/n_done on a fresh pair: forget it and ask again
    ctx.conn.execute("DELETE FROM predictions WHERE pair_id = ?", (first.pair_id,))
    ctx.conn.execute("DELETE FROM pairs WHERE pair_id = ?", (first.pair_id,))
    again = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert (again.a_id, again.b_id) == (first.a_id, first.b_id)


def test_pair_rng_is_stable_and_keyed():
    a = pair_rng("u", "clothes", "train", 0).integers(0, 2**32, 4)
    assert np.array_equal(a, pair_rng("u", "clothes", "train", 0).integers(0, 2**32, 4))
    for other in (
        pair_rng("v", "clothes", "train", 0),
        pair_rng("u", "art", "train", 0),
        pair_rng("u", "clothes", "showdown", 0),
        pair_rng("u", "clothes", "train", 1),
    ):
        assert not np.array_equal(a, other.integers(0, 2**32, 4))


def test_pair_rng_value_is_pinned():
    # sha256-derived, so this cannot drift with PYTHONHASHSEED or between processes
    assert int(pair_rng("u", "clothes", "train", 0).integers(0, 10**9)) == 751466491


def test_explicit_rng_is_used(ctx, user):
    p = next_pair(
        ctx.conn, user, "clothes", "showdown", ctx, rng=np.random.default_rng(123)
    )
    assert p is not None and p.phase == "showdown" and p.chosen_id is None


# --- filters ---------------------------------------------------------------------


def test_allowed_ids_default_is_all_genders_default_sections(ctx, user):
    items = ctx.items["clothes"]
    items.loc[items["id"] == "c:1", "master_category"] = "Accessories"
    allowed = allowed_ids(ctx.conn, user, ctx)
    assert "c:1" not in allowed and len(allowed) == len(items) - 1


def test_allowed_ids_respects_filters(ctx, user):
    _filters(ctx, user, genders=["Women"], sections=["Footwear"])
    allowed = allowed_ids(ctx.conn, user, ctx)
    items = ctx.items["clothes"].set_index("id")
    assert allowed and all(
        items.loc[i, "gender"] == "Women"
        and items.loc[i, "master_category"] == "Footwear"
        for i in allowed
    )


def test_allowed_ids_genders_only_keeps_default_sections(ctx, user):
    _filters(ctx, user, genders=["Men"])
    items = ctx.items["clothes"].set_index("id")
    allowed = allowed_ids(ctx.conn, user, ctx)
    assert {items.loc[i, "gender"] for i in allowed} == {"Men"}
    assert {items.loc[i, "master_category"] for i in allowed} == {"Apparel", "Footwear"}


def test_pairs_respect_filters(ctx, user):
    _filters(ctx, user, genders=["Men"], sections=["Footwear"])
    items = ctx.items["clothes"].set_index("id")
    chosen = play_clothes(ctx, user, n=9)
    assert len(chosen) == 9
    for c in chosen:
        for i in (c.a_id, c.b_id):
            assert items.loc[i, "gender"] == "Men"
            assert items.loc[i, "master_category"] == "Footwear"


def test_empty_filter_selection_exhausts_the_catalog(ctx, user):
    _filters(ctx, user, sections=[])
    with pytest.raises(CatalogExhausted):
        next_pair(ctx.conn, user, "clothes", "showdown", ctx)


def test_catalog_exhausted_when_eligible_items_run_out(ctx, user):
    ctx.settings = replace(ctx.settings, clothes_pairs=100)
    _filters(ctx, user, genders=["Men"], sections=["Footwear"])  # 24 items
    answered = 0
    with pytest.raises(CatalogExhausted):
        for _ in range(30):
            pair = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
            answer_pair(ctx.conn, pair.pair_id, pair.a_id)
            answered += 1
    assert 1 <= answered <= 12


# --- the scored clothing phase ---------------------------------------------------


def test_clothes_pair_can_be_answered_after_prediction(ctx, user):
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert p.chosen_id is None
    assert answer_pair(ctx.conn, p.pair_id, p.a_id).chosen_id == p.a_id


def test_frozen_predictions_come_from_the_frozen_w(ctx, user):
    play_art(ctx, user, prefer="red", n=ctx.settings.art_train_pairs)
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    a, b = ctx.index["clothes"].centered([p.a_id, p.b_id])
    got = get_predictions(ctx.conn, p.pair_id)
    for name, (w, meta) in load_yous(ctx.conn, user).items():
        assert got[name] == predict(You(name, w, meta), a, b)


def test_yous_stay_frozen_through_the_clothes_phase(ctx, user):
    play_art(ctx, user, prefer="red", n=ctx.settings.art_train_pairs)
    play_clothes(ctx, user, prefer="blue", n=1)
    frozen = load_yous(ctx.conn, user)
    save_answer(ctx.conn, user, "describe", "text", "I love blue things.")
    play_clothes(ctx, user, prefer="blue", n=8)
    after = load_yous(ctx.conn, user)
    assert set(after) == set(frozen) == set(FROZEN_YOUS) & set(BUILDERS)
    for name in frozen:
        w0, m0 = frozen[name]
        w1, m1 = after[name]
        assert m0 == m1 and (w0 is None) == (w1 is None)
        if w0 is not None:
            assert np.array_equal(w0, w1)


def test_clothes_picks_never_leak_into_the_frozen_art_you(ctx, user):
    play_clothes(ctx, user, prefer="red", n=8)
    art_w, art_meta = load_yous(ctx.conn, user)["art"]
    assert art_w is None and "reason" in art_meta


def test_clothes_freezes_once(ctx, user):
    next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    before = load_yous(ctx.conn, user)
    next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    after = load_yous(ctx.conn, user)
    assert {
        k: (None if v[0] is None else v[0].tobytes(), v[1]) for k, v in before.items()
    } == {k: (None if v[0] is None else v[0].tobytes(), v[1]) for k, v in after.items()}


def test_clothes_resumes_open_pair_without_new_pairs(ctx, user):
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert next_pair(ctx.conn, user, "clothes", "showdown", ctx) == p
    assert (
        len(list_choices(ctx.conn, user, "clothes", "showdown", answered_only=False))
        == 1
    )


def test_clothes_resume_repairs_missing_predictions(ctx, user):
    play_clothes(ctx, user, prefer="red", n=7)
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    original = get_predictions(ctx.conn, p.pair_id)
    ctx.conn.execute("DELETE FROM predictions WHERE pair_id = ?", (p.pair_id,))
    assert next_pair(ctx.conn, user, "clothes", "showdown", ctx) == p
    assert get_predictions(ctx.conn, p.pair_id) == original
    assert "revealed" in original


def test_showdown_is_clothes_only(ctx, user):
    with pytest.raises(ValueError):
        next_pair(ctx.conn, user, "art", "showdown", ctx)
    assert load_yous(ctx.conn, user) == {}


def test_retest_is_not_wired_yet(ctx, user):
    with pytest.raises(NotImplementedError):
        next_pair(ctx.conn, user, "clothes", "retest", ctx)


def test_clothes_pairs_are_random_not_model_chosen(ctx, user, monkeypatch):
    from amber import flow

    def boom(*_a, **_k):
        raise AssertionError("a model chose a clothes pair")

    monkeypatch.setattr(flow, "choose_pair", boom)
    assert len(play_clothes(ctx, user, prefer="red", n=4)) == 4


# --- progress --------------------------------------------------------------------


def test_progress_counts_answered_pairs_per_domain_and_phase(ctx, user):
    s = ctx.settings
    assert progress(ctx.conn, user, "clothes", "showdown", ctx) == {
        "done": 0,
        "total": s.clothes_pairs,
    }
    play_clothes(ctx, user, n=2)
    next_pair(ctx.conn, user, "clothes", "showdown", ctx)  # open, not yet answered
    assert progress(ctx.conn, user, "clothes", "showdown", ctx)["done"] == 2
    play_art(ctx, user, n=1)
    assert progress(ctx.conn, user, "art", "train", ctx) == {
        "done": 1,
        "total": s.art_train_pairs,
    }
    assert progress(ctx.conn, user, "clothes", "retest", ctx) == {
        "done": 0,
        "total": s.retest_pairs,
    }
    with pytest.raises(ValueError):
        progress(ctx.conn, user, "clothes", "train", ctx)


def test_freeze_is_atomic(ctx, user, monkeypatch):
    """A failure while saving the Yous leaves none of them saved (no half-frozen set)."""
    from amber import flow

    good = You("stated", np.ones(4, dtype=np.float32), {"ok": True})
    bad = You("art", None, {"unserializable": object()})
    monkeypatch.setattr(
        flow,
        "build_frozen_yous",
        lambda conn, user_id, ctx: {"stated": good, "art": bad},
    )
    with pytest.raises(TypeError):
        flow._freeze_yous(ctx.conn, user, ctx)
    assert load_yous(ctx.conn, user) == {}


# --- one scored clothing phase --------------------------------------------------


def test_clothes_train_phase_is_gone(ctx, user):
    with pytest.raises(ValueError):
        next_pair(ctx.conn, user, "clothes", "train", ctx)


def test_frozen_before_first_clothes_pair(ctx, user):
    play_art(ctx, user, n=ctx.settings.art_train_pairs)
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert set(load_yous(ctx.conn, user)) == set(FROZEN_YOUS) & set(BUILDERS)
    assert set(get_predictions(ctx.conn, p.pair_id)) == (
        set(FROZEN_YOUS) & set(BUILDERS)
    ) | {"revealed"}


def test_revealed_abstains_for_first_five_pairs(ctx, user):
    choices = play_clothes(ctx, user, prefer="red", n=6)
    assert [get_predictions(ctx.conn, c.pair_id)["revealed"] for c in choices[:5]] == [
        0.5
    ] * 5


def test_revealed_prediction_uses_only_earlier_picks(ctx, user):
    choices = play_clothes(ctx, user, prefer="red", n=8)
    p7 = get_predictions(ctx.conn, choices[7].pair_id)["revealed"]
    assert p7 == pytest.approx(revealed_p_a_from(choices[:7], choices[7], ctx))
    assert p7 != 0.5


def test_revealed_learns_red(ctx):
    # A fixed user id fixes the (sha256-seeded) random pairs, so this threshold is not luck:
    # over many ids the late accuracy ranges from about 0.67 to 0.86 (mean 0.77).
    user = "0" * 32
    ctx.conn.execute(
        "INSERT INTO users (user_id, name, created_at) VALUES (?, 'fixed', 'now')",
        (user,),
    )
    choices = play_clothes(ctx, user, prefer="red", n=ctx.settings.clothes_pairs)
    assert len(choices) == ctx.settings.clothes_pairs
    late = score_showdown(choices, preds(ctx, choices), window=(21, 40))
    assert next(s for s in late if s.you == "revealed").accuracy > 0.7
