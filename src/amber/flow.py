"""The pair flow: which pair to show next, per domain and phase.

* `train`: an active pair, chosen by a preference model replayed from that domain's answered
  train choices.
* `showdown` (clothes only): on the first call every You is built and frozen, then pairs are
  drawn at random and every frozen You's prediction is saved *before* the pair is returned,
  so before it can be answered. Showdown choices never train anything.
* `retest` is wired up in a later task.

No item is ever shown twice to a user within a domain, whatever the phase.
"""

import hashlib
import json
import sqlite3
from typing import TYPE_CHECKING, Literal

import numpy as np

from amber.config import DEFAULT_SECTIONS, SEED
from amber.prefs.active import choose_pair, random_pair
from amber.store.db import (
    Choice,
    create_pair,
    get_answers,
    get_predictions,
    list_choices,
    load_yous,
    open_pair,
    save_predictions,
    save_you,
    shown_ids,
)
from amber.yous.base import You, predict
from amber.yous.registry import build_all_yous
from amber.yous.revealed import train_preference_model

if TYPE_CHECKING:
    from amber.context import AppContext

Domain = Literal["clothes", "art"]
Phase = Literal["train", "showdown", "retest"]


class CatalogExhausted(Exception):
    """Not enough unseen, eligible items are left to form another pair."""


def allowed_ids(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> set[str]:
    """Clothes ids the person's filters let through.

    The `filters` answers `genders` and `sections` hold JSON lists. Missing `genders` means
    every gender; missing `sections` means `DEFAULT_SECTIONS`.
    """
    filters = get_answers(conn, user_id, "filters")
    items = ctx.items["clothes"]
    keep = items["master_category"].isin(
        json.loads(filters["sections"]) if "sections" in filters else DEFAULT_SECTIONS
    )
    if "genders" in filters:
        keep &= items["gender"].isin(json.loads(filters["genders"]))
    return set(items.loc[keep, "id"])


def pair_rng(user_id: str, domain: str, phase: str, n_done: int) -> np.random.Generator:
    """A generator seeded from `(SEED, user_id, domain, phase, n_done)`.

    Derived with sha256, never builtin `hash()` (which is salted per process), so a given
    user is shown the same pairs in the same state across runs.
    """
    key = repr((SEED, user_id, domain, phase, n_done)).encode("utf-8")
    seed = int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    return np.random.default_rng(seed)


def _total(domain: str, phase: str, ctx: "AppContext") -> int:
    s = ctx.settings
    totals = {
        ("art", "train"): s.art_train_pairs,
        ("clothes", "train"): s.clothes_train_pairs,
        ("clothes", "showdown"): s.showdown_pairs,
        ("clothes", "retest"): s.retest_pairs,
    }
    try:
        return totals[(domain, phase)]
    except KeyError:
        raise ValueError(f"no {phase!r} phase for domain {domain!r}") from None


def progress(
    conn: sqlite3.Connection,
    user_id: str,
    domain: Domain,
    phase: Phase,
    ctx: "AppContext",
) -> dict:
    """`{"done": answered pairs in this domain/phase, "total": the matching setting}`."""
    total = _total(domain, phase, ctx)
    return {"done": len(list_choices(conn, user_id, domain, phase)), "total": total}


def _freeze_yous(
    conn: sqlite3.Connection, user_id: str, ctx: "AppContext"
) -> dict[str, You]:
    """Build and save every You, once. Later calls return the saved Yous untouched."""
    if not load_yous(conn, user_id):
        for you in build_all_yous(conn, user_id, ctx).values():
            save_you(conn, user_id, you.name, you.w, you.meta)
    return {
        name: You(name, w, meta) for name, (w, meta) in load_yous(conn, user_id).items()
    }


def _save_frozen_predictions(
    conn: sqlite3.Connection,
    pair: Choice,
    frozen: dict[str, You],
    ctx: "AppContext",
) -> None:
    """Every frozen You's `P(a)` for the pair, saved in one call."""
    a, b = ctx.index["clothes"].centered([pair.a_id, pair.b_id])
    save_predictions(
        conn, pair.pair_id, {name: predict(you, a, b) for name, you in frozen.items()}
    )


def next_pair(
    conn: sqlite3.Connection,
    user_id: str,
    domain: Domain,
    phase: Phase,
    ctx: "AppContext",
    rng: np.random.Generator | None = None,
) -> Choice | None:
    """The pair to show next, or `None` once the phase's pair count is reached.

    An unanswered pair is returned as it is, so closing the tab mid-pair resumes it. Raises
    `CatalogExhausted` when no unseen, non-near-duplicate pair remains, `ValueError` for a
    showdown on art, `NotImplementedError` for retest.
    """
    total = _total(domain, phase, ctx)
    if phase == "retest":
        raise NotImplementedError("retest pairs are wired up in a later task")

    frozen: dict[str, You] | None = None
    pending = open_pair(conn, user_id, domain, phase)
    if pending is not None:
        if phase == "showdown":  # repair a pair whose predictions never landed
            frozen = _freeze_yous(conn, user_id, ctx)
            if set(get_predictions(conn, pending.pair_id)) != set(frozen):
                _save_frozen_predictions(conn, pending, frozen, ctx)
        return pending

    n_done = len(list_choices(conn, user_id, domain, phase))
    if n_done >= total:
        return None
    if phase == "showdown":
        frozen = _freeze_yous(conn, user_id, ctx)

    index = ctx.index[domain]
    allowed = allowed_ids(conn, user_id, ctx) if domain == "clothes" else None
    seen = shown_ids(conn, user_id, domain)
    eligible = [
        i for i in index.ids if i not in seen and (allowed is None or i in allowed)
    ]
    if len(eligible) < 2:
        raise CatalogExhausted(f"only {len(eligible)} unseen {domain} items are left")
    if rng is None:
        rng = pair_rng(user_id, domain, phase, n_done)
    dup_vectors = np.stack([index.vec(i) for i in eligible])
    try:
        if phase == "train":
            model = train_preference_model(
                list_choices(conn, user_id, domain, "train"), index, ctx.basis[domain]
            )
            X = ctx.basis[domain].project(index.centered(eligible))
            a_id, b_id = choose_pair(model, X, eligible, dup_vectors, rng)
        else:
            a_id, b_id = random_pair(eligible, dup_vectors, rng)
    except ValueError as exc:
        raise CatalogExhausted(str(exc)) from exc

    pair = create_pair(conn, user_id, domain, phase, a_id, b_id)
    if frozen is not None:
        _save_frozen_predictions(conn, pair, frozen, ctx)
    return pair
