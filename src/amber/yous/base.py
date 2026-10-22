"""The `You` type and the two helpers every builder shares.

A `You` is a named unit direction `w` in the full embedding space. `w is None` means the
You abstains (it has nothing to say about this person) and predicts 0.5 everywhere.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from amber.config import NEUTRAL_PROMPT
from amber.space.encoder import Encoder

YouName = Literal["stated", "bigfive", "art", "values", "revealed"]
YOU_ORDER: tuple[YouName, ...] = ("stated", "bigfive", "art", "values", "revealed")

_EPS = 1e-6  # a direction shorter than this is float32 noise, not a direction


@dataclass(frozen=True)
class You:
    """`w` is a unit vector in full-dim space, or `None` to abstain; `meta` explains it."""

    name: YouName
    w: np.ndarray | None
    meta: dict = field(default_factory=dict)


def text_direction(
    encoder: Encoder, texts: Sequence[str], neutral: str = NEUTRAL_PROMPT
) -> np.ndarray:
    """`normalize(mean(embed(texts)) - embed(neutral))`: what `texts` add to a plain clothing prompt.

    Raises `ValueError` if `texts` is empty, or if the result has no direction (the texts
    embed exactly where the neutral prompt does). Callers that want to abstain catch it.
    """
    if len(texts) == 0:
        raise ValueError("text_direction needs at least one text")
    embedded = np.asarray(encoder.embed_texts(list(texts)), dtype=np.float64)
    base = np.asarray(encoder.embed_texts([neutral]), dtype=np.float64)[0]
    direction = embedded.mean(axis=0) - base
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < _EPS:
        raise ValueError("texts have no direction relative to the neutral prompt")
    return (direction / norm).astype(np.float32)


def predict(you: You, a_centered: np.ndarray, b_centered: np.ndarray) -> float:
    """1.0 if `w·a > w·b`, 0.0 if lower, 0.5 on a tie or when the You abstains."""
    if you.w is None:
        return 0.5
    diff = float(you.w @ np.asarray(a_centered) - you.w @ np.asarray(b_centered))
    if diff > 0:
        return 1.0
    if diff < 0:
        return 0.0
    return 0.5
