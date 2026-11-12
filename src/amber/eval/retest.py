"""The retest a week later: the same person, the same pairs, and how often they agree.

A person's own repeat consistency is the ceiling for any You: nobody can predict a pick
better than the person can repeat it. `retest_plan` picks which showdown pairs come back
(and which come back swapped), `retest_due` says when, `next_retest_pair` serves them.
Retest picks are stored as phase `retest` with `retest_of` set to the original pair; they
never carry predictions and nothing trains on them.
"""

import hashlib
import sqlite3
from datetime import UTC, datetime, timedelta

import numpy as np

from amber.config import SEED, Settings
from amber.store.db import Choice, create_pair, list_choices, open_pair


class RetestNotDue(Exception):
    """The retest cannot start yet: the clothes picks are unfinished or too recent."""


def _plan_rng(user_id: str) -> np.random.Generator:
    # sha256, never builtin `hash()` (salted per process): the plan is stable across runs.
    key = repr((SEED, user_id, "retest")).encode("utf-8")
    return np.random.default_rng(
        int.from_bytes(hashlib.sha256(key).digest()[:4], "big")
    )


def retest_plan(
    user_id: str, originals: list[Choice], k: int
) -> list[tuple[Choice, bool]]:
    """Which `k` of the answered showdown pairs come back, in the original order, each with
    a flag saying whether its sides are swapped. Seeded per user, so every call agrees."""
    rng = _plan_rng(user_id)
    chosen = np.sort(
        rng.choice(len(originals), size=min(k, len(originals)), replace=False)
    )
    swapped = rng.random(len(chosen)) < 0.5
    return [(originals[i], bool(s)) for i, s in zip(chosen, swapped, strict=True)]


def _utc(now: datetime) -> datetime:
    return now.replace(tzinfo=UTC) if now.tzinfo is None else now


def _showdown(
    conn: sqlite3.Connection, user_id: str, settings: Settings
) -> list[Choice]:
    """The answered showdown pairs, or `[]` while the person has not finished all of them."""
    picks = list_choices(conn, user_id, "clothes", "showdown")
    return picks if len(picks) >= settings.clothes_pairs else []


def retest_due_at(
    conn: sqlite3.Connection, user_id: str, settings: Settings
) -> datetime | None:
    """`retest_min_days` after the last clothes pick, or `None` before the showdown is done."""
    picks = _showdown(conn, user_id, settings)
    if not picks:
        return None
    return max(c.answered_at for c in picks) + timedelta(days=settings.retest_min_days)


def _retest_size(originals: list[Choice], settings: Settings) -> int:
    return min(settings.retest_pairs, len(originals))


def _unfinished(
    conn: sqlite3.Connection, user_id: str, originals: list[Choice], settings: Settings
) -> bool:
    done = list_choices(conn, user_id, "clothes", "retest")
    return len(done) < _retest_size(originals, settings)


def retest_due(
    conn: sqlite3.Connection, user_id: str, settings: Settings, now: datetime
) -> bool:
    """True once the last clothes pick is `retest_min_days` old and the retest is unfinished."""
    due_at = retest_due_at(conn, user_id, settings)
    if due_at is None or _utc(now) < due_at:
        return False
    return _unfinished(conn, user_id, _showdown(conn, user_id, settings), settings)


def next_retest_pair(
    conn: sqlite3.Connection, user_id: str, settings: Settings, now: datetime
) -> Choice | None:
    """The next retest pair, or `None` once all of them are answered.

    An unanswered retest pair is returned as it is. Raises `RetestNotDue` while the retest
    cannot start (showdown unfinished, or the last pick is under `retest_min_days` old).
    """
    originals = _showdown(conn, user_id, settings)
    if not originals:
        raise RetestNotDue("finish the clothes picks first")
    if not _unfinished(conn, user_id, originals, settings):
        return None
    due_at = retest_due_at(conn, user_id, settings)
    if _utc(now) < due_at:
        raise RetestNotDue(f"the retest opens on {due_at.date().isoformat()}")
    pending = open_pair(conn, user_id, "clothes", "retest")
    if pending is not None:
        return pending
    started = {
        c.retest_of
        for c in list_choices(conn, user_id, "clothes", "retest", answered_only=False)
    }
    plan = retest_plan(user_id, originals, _retest_size(originals, settings))
    orig, swapped = next(p for p in plan if p[0].pair_id not in started)
    a_id, b_id = (orig.b_id, orig.a_id) if swapped else (orig.a_id, orig.b_id)
    return create_pair(
        conn, user_id, "clothes", "retest", a_id, b_id, retest_of=orig.pair_id
    )
