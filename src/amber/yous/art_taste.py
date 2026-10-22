"""Art you: the direction the person's taste in artworks points in.

Learned only from art *train* choices (this-or-that on paintings and photos). Clothing
choices never reach it: `art_direction` raises `ValueError` if handed one. The direction is
lifted into the full CLIP space, where it is carried over to clothes.
"""

import sqlite3
from typing import TYPE_CHECKING

import numpy as np

from amber.space.basis import Basis
from amber.space.index import VectorIndex
from amber.store.db import Choice, list_choices
from amber.yous.base import You
from amber.yous.revealed import _EPS, MIN_CHOICES, train_preference_model

if TYPE_CHECKING:
    from amber.context import AppContext


def art_direction(
    choices: list[Choice], index: VectorIndex, basis: Basis
) -> np.ndarray | None:
    """`normalize(basis.lift(model.mean))` from art choices; `None` if there are fewer than
    `MIN_CHOICES` or the learned direction is degenerate.

    Raises `ValueError` if any choice is not an art choice (the leak rule).
    """
    for choice in choices:
        if choice.domain != "art":
            raise ValueError(
                f"art you only learns from art choices, got {choice.domain!r}"
            )
    if len(choices) < MIN_CHOICES:
        return None
    model = train_preference_model(choices, index, basis)
    direction = np.asarray(basis.lift(model.mean), dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < _EPS:
        return None
    return (direction / norm).astype(np.float32)


def build_art(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """The art `You` from the answered art *train* choices.

    `meta={"n": <number of those choices>}`. Abstains (`w=None`, `meta={"reason": ...}`) with
    fewer than `MIN_CHOICES` choices, or if the learned direction is degenerate.
    """
    choices = list_choices(conn, user_id, domain="art", phase="train")
    if len(choices) < MIN_CHOICES:
        return You("art", None, {"reason": f"fewer than {MIN_CHOICES} art choices"})
    w = art_direction(choices, ctx.index["art"], ctx.basis["art"])
    if w is None:
        return You("art", None, {"reason": "no direction"})
    return You("art", w, {"n": len(choices)})
