"""The pair flow: which pair to show next, per domain and phase.

* `train` (art only): an active pair, chosen by a preference model replayed from the answered
  art choices.
* `showdown` (clothes only, the one scored clothing phase): on the first call the Yous that
  must not see clothing choices are built and frozen, then pairs are drawn at random. Before
  a pair is returned every frozen You's prediction is saved, plus Revealed's, which comes
  from a model replayed on the person's *earlier* clothing picks only.
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
    save_yous,
    shown_ids,
)
from amber.yous.base import You, predict
from amber.yous.registry import build_frozen_yous
from amber.yous.revealed import revealed_p_a, train_preference_model

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
        ("clothes", "showdown"): s.clothes_pairs,
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
    """Build and save the frozen Yous, once. Later calls return the saved Yous untouched."""
    if not load_yous(conn, user_id):
        built = build_frozen_yous(conn, user_id, ctx)
        save_yous(  # one transaction: a failure leaves no half-frozen set
            conn, user_id, {y.name: (y.w, y.meta) for y in built.values()}
        )
    return {
        name: You(name, w, meta) for name, (w, meta) in load_yous(conn, user_id).items()
    }


def _save_frozen_predictions(
    conn: sqlite3.Connection,
    pair: Choice,
    frozen: dict[str, You],
    ctx: "AppContext",
) -> None:
    """Every frozen You's `P(a)` for the pair, plus Revealed's, saved in one call."""
    a, b = ctx.index["clothes"].centered([pair.a_id, pair.b_id])
    preds = {name: predict(you, a, b) for name, you in frozen.items()}
    preds["revealed"] = revealed_p_a(conn, pair.user_id, pair, ctx)
    save_predictions(conn, pair.pair_id, preds)


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
    showdown on art or a clothes train, `NotImplementedError` for retest.
    """
    total = _total(domain, phase, ctx)
    if phase == "retest":
        raise NotImplementedError("retest pairs are wired up in a later task")

    frozen: dict[str, You] | None = None
    pending = open_pair(conn, user_id, domain, phase)
    if pending is not None:
        if phase == "showdown":  # repair a pair whose predictions never landed
            frozen = _freeze_yous(conn, user_id, ctx)
            if set(get_predictions(conn, pending.pair_id)) != {*frozen, "revealed"}:
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
