"""Revealed you: the direction the person's own clothing picks point in.

Learns from the person's own clothes picks (phase `showdown`: the scored phase) and nothing
else: no art, no retest. Two readings of the same model:

* `revealed_p_a`: the prequential prediction for one pair, from the picks answered before
  that pair was created. This is what is scored, so it can never peek.
* `build_revealed`: the direction after all picks, for the readout, the map and search.
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

MIN_CHOICES = 5  # fewer earlier clothes picks than this and revealed abstains
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


def revealed_p_a(
    conn: sqlite3.Connection, user_id: str, pair: Choice, ctx: "AppContext"
) -> float:
    """`P(a)` for `pair` from a model replayed on this person's earlier clothes picks.

    "Earlier" means answered no later than `pair` was created (`pair` itself is excluded),
    replayed in answer order. Exactly `0.5` under `MIN_CHOICES` such picks.
    """
    earlier = sorted(
        (
            c
            for c in list_choices(conn, user_id, domain="clothes", phase="showdown")
            if c.pair_id != pair.pair_id
            and c.answered_at is not None
            and c.answered_at <= pair.shown_at
        ),
        key=lambda c: (c.answered_at, c.shown_at),
    )
    if len(earlier) < MIN_CHOICES:
        return 0.5
    index, basis = ctx.index["clothes"], ctx.basis["clothes"]
    model = train_preference_model(earlier, index, basis)
    a, b = basis.project(index.centered([pair.a_id, pair.b_id]))
    return model.p_prefer(a, b)


def build_revealed(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """`w = normalize(basis.lift(model.mean))` from all answered clothes `showdown` choices.

    `meta={"n": <number of those choices>}`. Abstains (`w=None`, with a `reason`) with fewer
    than `MIN_CHOICES` choices, or if the learned direction is degenerate.
    """
    choices = list_choices(conn, user_id, domain="clothes", phase="showdown")
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
