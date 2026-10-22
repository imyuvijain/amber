"""Showdown scoring: how often each You's frozen prediction matched the person's pick."""

from dataclasses import dataclass

import numpy as np

from amber.config import SEED
from amber.store.db import Choice
from amber.yous.base import YOU_ORDER


@dataclass(frozen=True)
class Score:
    you: str
    correct: float  # sum of per-pair credit; an abstention earns 0.5 per pair
    n: int  # pairs scored
    accuracy: float  # correct / n
    ci_low: float  # 95% percentile bootstrap over pairs
    ci_high: float


def score_showdown(
    choices: list[Choice],
    predictions: dict[str, dict[str, float]],
    n_boot: int = 2000,
    seed: int = SEED,
) -> list[Score]:
    """Score every You that has predictions, in `YOU_ORDER`.

    `predictions` maps `pair_id` to `{you: P(a)}`. A pair earns `P(a)` when `a` was chosen
    and `1 - P(a)` otherwise, so a prediction of exactly 0.5 earns 0.5. Unanswered pairs and
    pairs without a prediction from a You are skipped for that You. The interval is a 95%
    percentile bootstrap (`n_boot` resamples of the pairs, `np.random.default_rng(seed)`).
    """
    answered = [c for c in choices if c.chosen_id is not None]
    rng = np.random.default_rng(seed)
    scores = []
    for you in YOU_ORDER:
        credit = np.array(
            [
                p if c.chosen_id == c.a_id else 1.0 - p
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
