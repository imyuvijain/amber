"""Scoring: how often each You's saved prediction matched the person's clothing pick."""

from dataclasses import dataclass

import numpy as np

from amber.config import SEED
from amber.store.db import Choice
from amber.yous.base import YOU_ORDER


@dataclass(frozen=True)
class Score:
    you: str
    correct: float  # picks called right (hard call, 1 each); an abstention earns 0.5 per pair
    n: int  # pairs scored
    accuracy: float  # correct / n
    ci_low: float  # 95% percentile bootstrap over pairs
    ci_high: float


def _credit(c: Choice, p: float) -> float:
    """Credit for a You's hard call: it predicts `a` when `p > 0.5` and `b` when `p < 0.5`.

    1 if the call matches the pick, 0 if not, 0.5 when `p == 0.5` (an abstention). Stored
    predictions stay soft `P(a)`; only the credit is hard, so a soft Revealed and the 0/1
    frozen Yous are compared on the same footing ("accuracy" = fraction of picks called right).
    """
    if p == 0.5:
        return 0.5
    return 1.0 if (p > 0.5) == (c.chosen_id == c.a_id) else 0.0


def score_showdown(
    choices: list[Choice],
    predictions: dict[str, dict[str, float]],
    n_boot: int = 2000,
    seed: int = SEED,
    window: tuple[int, int] | None = None,
) -> list[Score]:
    """Score every You that has predictions, in `YOU_ORDER`.

    `predictions` maps `pair_id` to `{you: P(a)}`. A pair earns 1 when the You's hard call
    (`P(a) > 0.5` means `a`, `< 0.5` means `b`) matches the pick, 0 when it does not, and 0.5
    for exactly 0.5 (abstain). Unanswered pairs and
    pairs without a prediction from a You are skipped for that You. The interval is a 95%
    percentile bootstrap (`n_boot` resamples of the pairs, `np.random.default_rng(seed)`).

    `window=(first, last)` keeps only pair positions `first..last` (1-based, inclusive, by
    creation order among `choices`), e.g. `(21, 40)` for the late picks.
    """
    if window is not None:
        choices = choices[window[0] - 1 : window[1]]
    answered = [c for c in choices if c.chosen_id is not None]
    rng = np.random.default_rng(seed)
    scores = []
    for you in YOU_ORDER:
        credit = np.array(
            [
                _credit(c, p)
                for c in answered
                if (p := predictions.get(c.pair_id, {}).get(you)) is not None
            ],
            dtype=np.float64,
        )
        if len(credit) == 0:
            continue
        accuracy = float(credit.mean())
        picks = rng.integers(0, len(credit), size=(n_boot, len(credit)))
        low, high = np.percentile(credit[picks].mean(axis=1), [2.5, 97.5])
        scores.append(
            Score(
                you=you,
                correct=float(credit.sum()),
                n=len(credit),
                accuracy=accuracy,
                # summation order can differ by an ulp; the interval must still bracket
                ci_low=min(float(low), accuracy),
                ci_high=max(float(high), accuracy),
            )
        )
    return scores


def learning_curve(
    choices: list[Choice],
    predictions: dict[str, dict[str, float]],
    you: str,
    size: int = 10,
) -> list[float]:
    """Rolling accuracy of one You: entry `i` is the mean credit (as in `score_showdown`) over
    the last `size` credited pairs up to and including pair `i`, so it is shorter at the start.
    Unanswered pairs and pairs without a prediction from `you` are skipped."""
    credit = [
        _credit(c, p)
        for c in choices
        if c.chosen_id is not None
        and (p := predictions.get(c.pair_id, {}).get(you)) is not None
    ]
    return [
        float(np.mean(credit[max(0, i + 1 - size) : i + 1])) for i in range(len(credit))
    ]
