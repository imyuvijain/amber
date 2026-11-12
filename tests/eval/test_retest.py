from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fakes import play_clothes, preds

from amber.eval.retest import (
    RetestNotDue,
    next_retest_pair,
    retest_due,
    retest_due_at,
    retest_finished,
    retest_plan,
)
from amber.flow import next_pair, progress
from amber.store.db import (
    answer_pair,
    create_user,
    get_predictions,
    list_choices,
    shown_ids,
)


@pytest.fixture
def settings(ctx):
    """6 showdown pairs, 3 retest pairs, 7 days."""
    ctx.settings = replace(ctx.settings, clothes_pairs=6, retest_pairs=3)
    return ctx.settings


@pytest.fixture
def played(ctx, user, settings):
    return play_clothes(ctx, user, n=6)


def _after(played, days: float):
    return max(c.answered_at for c in played) + timedelta(days=days)


def _play_retest(ctx, user, now, pick="a"):
    out = []
    while (pair := next_retest_pair(ctx.conn, user, ctx.settings, now)) is not None:
        out.append(
            answer_pair(ctx.conn, pair.pair_id, pair.a_id if pick == "a" else pair.b_id)
        )
    return out


# --- the plan: which pairs come back, in which order, which way round ------------


def test_plan_picks_k_originals_in_their_original_order(played):
    plan = retest_plan("u", played, 3)
    assert len(plan) == 3
    ids = [orig.pair_id for orig, _ in plan]
    assert len(set(ids)) == 3
    assert ids == [c.pair_id for c in played if c.pair_id in ids]


def test_plan_is_stable_across_calls_and_differs_between_users(played):
    assert retest_plan("u", played, 3) == retest_plan("u", played, 3)
    chosen = {
        tuple(o.pair_id for o, _ in retest_plan(f"user{i}", played, 3))
        for i in range(12)
    }
    assert len(chosen) > 1


def test_plan_swaps_sides_for_some_users_not_all(played):
    flags = [s for i in range(12) for _, s in retest_plan(f"user{i}", played, 3)]
    assert any(flags) and not all(flags)


def test_plan_never_asks_for_more_than_exist(played):
    assert len(retest_plan("u", played, 99)) == len(played)


# --- when it is due ---------------------------------------------------------------


def test_due_at_is_the_last_pick_plus_min_days(ctx, user, settings, played):
    assert retest_due_at(ctx.conn, user, settings) == _after(played, 7)


def test_not_due_before_min_days_and_due_from_the_day(ctx, user, settings, played):
    assert not retest_due(ctx.conn, user, settings, _after(played, 6.99))
    assert retest_due(ctx.conn, user, settings, _after(played, 7))
    assert retest_due(ctx.conn, user, settings, _after(played, 30))


def test_not_due_while_the_showdown_is_unfinished(ctx, user, settings):
    played = play_clothes(ctx, user, n=4)
    assert retest_due_at(ctx.conn, user, settings) is None
    assert not retest_due(ctx.conn, user, settings, _after(played, 30))


def test_no_picks_at_all_is_not_due(ctx, user, settings):
    assert retest_due_at(ctx.conn, user, settings) is None
    assert not retest_due(ctx.conn, user, settings, datetime.now(UTC))


def test_early_retest_raises(ctx, user, settings, played):
    with pytest.raises(RetestNotDue):
        next_retest_pair(ctx.conn, user, settings, _after(played, 6))
    assert list_choices(ctx.conn, user, "clothes", "retest", answered_only=False) == []


def test_naive_now_counts_as_utc(ctx, user, settings, played):
    now = _after(played, 8).replace(tzinfo=None)
    assert retest_due(ctx.conn, user, settings, now)


# --- serving the retest -----------------------------------------------------------


def test_retest_pairs_reuse_original_pairs_sides_may_swap(ctx, user, settings, played):
    now = _after(played, 8)
    pair = next_retest_pair(ctx.conn, user, settings, now)
    assert pair.phase == "retest" and pair.domain == "clothes"
    orig = next(c for c in played if c.pair_id == pair.retest_of)
    assert {pair.a_id, pair.b_id} == {orig.a_id, orig.b_id}
    assert pair.chosen_id is None


def test_open_retest_pair_resumes(ctx, user, settings, played):
    now = _after(played, 8)
    first = next_retest_pair(ctx.conn, user, settings, now)
    assert next_retest_pair(ctx.conn, user, settings, now) == first


def test_retest_runs_k_pairs_in_plan_order_then_none(ctx, user, settings, played):
    now = _after(played, 8)
    retest = _play_retest(ctx, user, now)
    assert len(retest) == 3
    plan = retest_plan(user, played, 3)
    assert [c.retest_of for c in retest] == [o.pair_id for o, _ in plan]
    for c, (orig, swapped) in zip(retest, plan, strict=True):
        assert (c.a_id, c.b_id) == (
            (orig.b_id, orig.a_id) if swapped else (orig.a_id, orig.b_id)
        )
    assert next_retest_pair(ctx.conn, user, settings, now) is None
    assert not retest_due(ctx.conn, user, settings, now)


def test_finished_retest_returns_none_even_before_the_clock_says_so(
    ctx, user, settings, played
):
    _play_retest(ctx, user, _after(played, 8))
    assert next_retest_pair(ctx.conn, user, settings, _after(played, 0)) is None


def test_retest_leaves_scored_data_alone(ctx, user, settings, played):
    before_shown = shown_ids(ctx.conn, user, "clothes")
    before = (list_choices(ctx.conn, user, "clothes", "showdown"), preds(ctx, played))
    retest = _play_retest(ctx, user, _after(played, 8))
    assert shown_ids(ctx.conn, user, "clothes") == before_shown
    assert (
        list_choices(ctx.conn, user, "clothes", "showdown"),
        preds(ctx, played),
    ) == before
    assert all(get_predictions(ctx.conn, c.pair_id) == {} for c in retest)


def test_other_users_are_unaffected(ctx, user, settings, played):
    other = create_user(ctx.conn, "other")
    assert retest_due_at(ctx.conn, other, settings) is None


# --- through flow.next_pair -------------------------------------------------------


def test_flow_serves_the_retest_and_counts_progress(ctx, user, settings, played):
    now = _after(played, 8)
    pair = next_pair(ctx.conn, user, "clothes", "retest", ctx, now=now)
    assert pair.phase == "retest"
    assert progress(ctx.conn, user, "clothes", "retest", ctx) == {"done": 0, "total": 3}
    answer_pair(ctx.conn, pair.pair_id, pair.a_id)
    assert progress(ctx.conn, user, "clothes", "retest", ctx)["done"] == 1


def test_flow_raises_when_not_due(ctx, user, settings, played):
    with pytest.raises(RetestNotDue):
        next_pair(ctx.conn, user, "clothes", "retest", ctx, now=_after(played, 1))


def test_flow_default_now_is_the_wall_clock(ctx, user, settings, played):
    with pytest.raises(RetestNotDue):  # the fixture picks were made a moment ago
        next_pair(ctx.conn, user, "clothes", "retest", ctx)


def test_retest_finished_only_after_all_pairs_are_answered(ctx, user, settings, played):
    assert not retest_finished(ctx.conn, user, settings)
    now = _after(played, 8)
    pair = next_retest_pair(ctx.conn, user, settings, now)
    answer_pair(ctx.conn, pair.pair_id, pair.a_id)
    assert not retest_finished(ctx.conn, user, settings)
    _play_retest(ctx, user, now)
    assert retest_finished(ctx.conn, user, settings)
