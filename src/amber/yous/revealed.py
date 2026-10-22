"""Revealed you: the direction the person's own clothing picks point in.

Replays their *train* choices (and nothing else: no showdown, retest or art choices) into
a preference model and lifts its posterior mean back to the full embedding space.
"""

import sqlite3
from typing import TYPE_CHECKING

import numpy as np

from amber.config import PRIOR_VAR
from amber.prefs.model import PreferenceModel
from amber.space.basis import Basis
from amber.space.index import VectorIndex
from amber.store.db import Choice, list_choices
from amber.yous.base import You

if TYPE_CHECKING:
    from amber.context import AppContext

MIN_CHOICES = 5  # fewer answered train choices than this and revealed abstains
_EPS = 1e-6  # a direction shorter than this is float32 noise, not a direction


def train_preference_model(
    choices: list[Choice],
    index: VectorIndex,
    basis: Basis,
    prior_var: float = PRIOR_VAR,
) -> PreferenceModel:
    """A fresh `PreferenceModel` over `basis` coordinates, updated with each answered choice
    in order (winner = the chosen item, loser = the other). Unanswered choices are skipped."""
    model = PreferenceModel(basis.components.shape[0], prior_var)
    for choice in choices:
        if choice.chosen_id is None:
            continue
        loser = choice.b_id if choice.chosen_id == choice.a_id else choice.a_id
        winner_x, loser_x = basis.project(index.centered([choice.chosen_id, loser]))
        model.update(winner_x, loser_x)
    return model


def build_revealed(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """`w = normalize(basis.lift(model.mean))` from the answered clothes *train* choices.

    `meta={"n": <number of those choices>}`. Abstains (`w=None`, with a `reason`) with fewer
    than `MIN_CHOICES` choices, or if the learned direction is degenerate.
    """
    choices = list_choices(conn, user_id, domain="clothes", phase="train")
    meta: dict = {"n": len(choices)}
    if len(choices) < MIN_CHOICES:
        return You(
            "revealed", None, {**meta, "reason": f"fewer than {MIN_CHOICES} choices"}
        )
    model = train_preference_model(choices, ctx.index["clothes"], ctx.basis["clothes"])
    direction = np.asarray(ctx.basis["clothes"].lift(model.mean), dtype=np.float64)
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < _EPS:
        return You("revealed", None, {**meta, "reason": "no direction"})
    return You("revealed", (direction / norm).astype(np.float32), meta)
