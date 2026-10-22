from datetime import UTC, datetime

import pytest

from amber.eval.metrics import Score, learning_curve, score_showdown
from amber.store.db import Choice
from amber.yous.base import YOU_ORDER


def _choice(i: int, chose_a: bool = True) -> Choice:
    now = datetime(2026, 1, 1, tzinfo=UTC)
    return Choice(
        pair_id=f"p{i}",
        user_id="u",
        domain="clothes",
        phase="showdown",
        a_id=f"c:{2 * i + 1}",
        b_id=f"c:{2 * i + 2}",
        chosen_id=f"c:{2 * i + 1}" if chose_a else f"c:{2 * i + 2}",
        shown_at=now,
        answered_at=now,
        retest_of=None,
    )


def _by_you(scores: list[Score]) -> dict[str, Score]:
    return {s.you: s for s in scores}


def test_score_counts_abstain_as_half():
    choices = [_choice(0), _choice(1), _choice(2), _choice(3, chose_a=False)]
    preds = {
        "p0": {"stated": 1.0, "revealed": 0.5},
        "p1": {"stated": 1.0, "revealed": 0.5},
        "p2": {"stated": 1.0, "revealed": 0.5},
        "p3": {"stated": 1.0, "revealed": 0.5},  # chose b; stated said a: wrong
    }
    s = _by_you(score_showdown(choices, preds))
    assert s["stated"].accuracy == 0.75 and s["stated"].correct == 3.0
    assert s["revealed"].accuracy == 0.5 and s["revealed"].correct == 2.0
    assert s["stated"].n == 4 and s["revealed"].n == 4


def test_wrong_side_scores_one_minus_p():
    s = _by_you(score_showdown([_choice(0, chose_a=False)], {"p0": {"stated": 0.8}}))
    assert s["stated"].correct == pytest.approx(0.2)


def test_results_follow_you_order_and_only_present_yous():
    choices = [_choice(0)]
    preds = {"p0": {"revealed": 1.0, "stated": 0.0, "art": 0.5}}
    got = [s.you for s in score_showdown(choices, preds)]
    assert got == [y for y in YOU_ORDER if y in {"revealed", "stated", "art"}]


def test_unanswered_and_unpredicted_pairs_are_ignored():
    unanswered = Choice(
        **{**_choice(1).__dict__, "chosen_id": None, "answered_at": None}
    )
    choices = [_choice(0), unanswered, _choice(2)]
    s = _by_you(score_showdown(choices, {"p0": {"stated": 1.0}, "p1": {"stated": 1.0}}))
    assert s["stated"].n == 1


def test_no_choices_gives_no_scores():
    assert score_showdown([], {}) == []


def _twenty():
    choices = [_choice(i, chose_a=i % 3 != 0) for i in range(20)]
    preds = {
        f"p{i}": {"stated": 1.0 if i % 4 else 0.0, "revealed": 0.5, "art": 0.3}
        for i in range(20)
    }
    return choices, preds


def test_ci_brackets_accuracy():
    choices, preds = _twenty()
    scores = score_showdown(choices, preds)
    assert len(scores) == 3
    for s in scores:
        assert s.ci_low <= s.accuracy <= s.ci_high
    assert _by_you(scores)["stated"].ci_low < _by_you(scores)["stated"].ci_high


def test_abstainer_has_a_degenerate_ci():
    choices, preds = _twenty()
    s = _by_you(score_showdown(choices, preds))["revealed"]
    assert s.ci_low == s.accuracy == s.ci_high == 0.5


def test_bootstrap_is_seeded():
    choices, preds = _twenty()
    assert score_showdown(choices, preds) == score_showdown(choices, preds)
    a = score_showdown(choices, preds, seed=1)
    b = score_showdown(choices, preds, seed=2)
    assert [(s.ci_low, s.ci_high) for s in a] != [(s.ci_low, s.ci_high) for s in b]


def test_learning_curve_length_and_bounds():
    choices, preds = _twenty()
    curve = learning_curve(choices, preds, "stated", size=5)
    assert len(curve) == 20
    assert all(0.0 <= x <= 1.0 for x in curve)


def test_learning_curve_is_shorter_at_the_start():
    choices = [_choice(0), _choice(1, chose_a=False), _choice(2)]
    preds = {f"p{i}": {"stated": 1.0} for i in range(3)}
    # windows: [p0], [p0, p1], [p1, p2] with size 2
    assert learning_curve(choices, preds, "stated", size=2) == [1.0, 0.5, 0.5]


def test_learning_curve_gives_abstainers_half_credit_and_skips_missing():
    choices = [_choice(0), _choice(1)]
    assert learning_curve(choices, {"p0": {"revealed": 0.5}}, "revealed") == [0.5]
    assert learning_curve(choices, {}, "revealed") == []


def test_score_window_selects_positions():
    choices = [_choice(i) for i in range(6)]
    preds = {f"p{i}": {"stated": 1.0 if i >= 3 else 0.0} for i in range(6)}
    s = _by_you(score_showdown(choices, preds, window=(4, 6)))["stated"]
    assert s.n == 3 and s.accuracy == 1.0
    s = _by_you(score_showdown(choices, preds, window=(1, 3)))["stated"]
    assert s.n == 3 and s.accuracy == 0.0
    assert _by_you(score_showdown(choices, preds, window=None))["stated"].n == 6
