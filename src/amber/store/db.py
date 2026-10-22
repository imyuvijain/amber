"""SQLite store: users, answers, this-or-that pairs, frozen predictions and frozen yous.

Timestamps are timezone-aware UTC, stored as ISO-8601 strings with fixed
microsecond precision so they round-trip exactly and sort lexically.
"""

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

Domain = Literal["clothes", "art"]
Phase = Literal["train", "showdown", "retest"]
AnswerKind = Literal["describe", "filters", "ipip", "values"]


class StoreError(Exception):
    """Base class for store errors."""


class AlreadyAnswered(StoreError):
    """The pair was already answered with a different choice."""


class MissingPredictions(StoreError):
    """A showdown pair cannot be answered before its predictions are saved."""


class NotFound(StoreError):
    """No such pair."""


@dataclass(frozen=True)
class Choice:
    pair_id: str
    user_id: str
    domain: Domain
    phase: Phase
    a_id: str
    b_id: str
    chosen_id: str | None
    shown_at: datetime
    answered_at: datetime | None
    retest_of: str | None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS answers (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    kind TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    PRIMARY KEY (user_id, kind, key)
);
CREATE TABLE IF NOT EXISTS pairs (
    pair_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    domain TEXT NOT NULL CHECK (domain IN ('clothes', 'art')),
    phase TEXT NOT NULL CHECK (phase IN ('train', 'showdown', 'retest')),
    a_id TEXT NOT NULL,
    b_id TEXT NOT NULL,
    chosen_id TEXT,
    shown_at TEXT NOT NULL,
    answered_at TEXT,
    retest_of TEXT REFERENCES pairs(pair_id)
);
CREATE INDEX IF NOT EXISTS pairs_by_user ON pairs(user_id, domain, phase);
CREATE TABLE IF NOT EXISTS predictions (
    pair_id TEXT NOT NULL REFERENCES pairs(pair_id),
    you TEXT NOT NULL,
    p_a REAL NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (pair_id, you)
);
CREATE TABLE IF NOT EXISTS yous (
    user_id TEXT NOT NULL REFERENCES users(user_id),
    name TEXT NOT NULL,
    w BLOB,
    meta TEXT NOT NULL,
    PRIMARY KEY (user_id, name)
);
"""


def _utc(now: datetime | None) -> datetime:
    if now is None:
        return datetime.now(UTC)
    if now.tzinfo is None:
        return now.replace(tzinfo=UTC)
    return now.astimezone(UTC)


def _iso(dt: datetime) -> str:
    return dt.isoformat(timespec="microseconds")


def connect(path: Path | str) -> sqlite3.Connection:
    """Open (and create if missing) the store. ``":memory:"`` is allowed.

    ``check_same_thread=False`` because the API runs sync handlers in a
    threadpool; callers are responsible for serializing access.
    """
    conn = sqlite3.connect(str(path), check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


# --- users and answers -------------------------------------------------------


def create_user(conn: sqlite3.Connection, name: str) -> str:
    user_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            "INSERT INTO users (user_id, name, created_at) VALUES (?, ?, ?)",
            (user_id, name, _iso(_utc(None))),
        )
    return user_id


def save_answer(
    conn: sqlite3.Connection, user_id: str, kind: AnswerKind, key: str, value: str
) -> None:
    with conn:
        conn.execute(
            "INSERT INTO answers (user_id, kind, key, value) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (user_id, kind, key) DO UPDATE SET value = excluded.value",
            (user_id, kind, key, value),
        )


def get_answers(conn: sqlite3.Connection, user_id: str, kind: str) -> dict[str, str]:
    rows = conn.execute(
        "SELECT key, value FROM answers WHERE user_id = ? AND kind = ?", (user_id, kind)
    )
    return {key: value for key, value in rows}


# --- pairs -------------------------------------------------------------------

_PAIR_COLS = "pair_id, user_id, domain, phase, a_id, b_id, chosen_id, shown_at, answered_at, retest_of"


def _choice(row: tuple) -> Choice:
    (
        pair_id,
        user_id,
        domain,
        phase,
        a_id,
        b_id,
        chosen_id,
        shown_at,
        answered_at,
        retest_of,
    ) = row
    return Choice(
        pair_id=pair_id,
        user_id=user_id,
        domain=domain,
        phase=phase,
        a_id=a_id,
        b_id=b_id,
        chosen_id=chosen_id,
        shown_at=datetime.fromisoformat(shown_at),
        answered_at=None
        if answered_at is None
        else datetime.fromisoformat(answered_at),
        retest_of=retest_of,
    )


def _get_pair(conn: sqlite3.Connection, pair_id: str) -> Choice:
    row = conn.execute(
        f"SELECT {_PAIR_COLS} FROM pairs WHERE pair_id = ?", (pair_id,)
    ).fetchone()
    if row is None:
        raise NotFound(f"no such pair: {pair_id}")
    return _choice(row)


def create_pair(
    conn: sqlite3.Connection,
    user_id: str,
    domain: Domain,
    phase: Phase,
    a_id: str,
    b_id: str,
    retest_of: str | None = None,
    now: datetime | None = None,
) -> Choice:
    pair_id = uuid.uuid4().hex
    with conn:
        conn.execute(
            f"INSERT INTO pairs ({_PAIR_COLS}) VALUES (?, ?, ?, ?, ?, ?, NULL, ?, NULL, ?)",
            (pair_id, user_id, domain, phase, a_id, b_id, _iso(_utc(now)), retest_of),
        )
    return _get_pair(conn, pair_id)


def answer_pair(
    conn: sqlite3.Connection, pair_id: str, chosen_id: str, now: datetime | None = None
) -> Choice:
    """Record the pick. Idempotent for the same ``chosen_id``.

    Order of checks: unknown pair, chosen not in pair, already answered
    (same id returns the stored Choice, a different id raises), then the
    showdown prediction gate.
    """
    with conn:
        pair = _get_pair(conn, pair_id)
        if chosen_id not in (pair.a_id, pair.b_id):
            raise ValueError(f"{chosen_id!r} is not in pair {pair_id}")
        if pair.chosen_id is not None:
            if pair.chosen_id == chosen_id:
                return pair
            raise AlreadyAnswered(
                f"pair {pair_id} already answered with {pair.chosen_id!r}"
            )
        if pair.phase == "showdown":
            n = conn.execute(
                "SELECT COUNT(*) FROM predictions WHERE pair_id = ?", (pair_id,)
            ).fetchone()[0]
            if n == 0:
                raise MissingPredictions(
                    f"showdown pair {pair_id} has no saved predictions"
                )
        conn.execute(
            "UPDATE pairs SET chosen_id = ?, answered_at = ? "
            "WHERE pair_id = ? AND chosen_id IS NULL",
            (chosen_id, _iso(_utc(now)), pair_id),
        )
    return _get_pair(conn, pair_id)


def open_pair(
    conn: sqlite3.Connection, user_id: str, domain: str, phase: str
) -> Choice | None:
    """The most recent unanswered pair for this user/domain/phase, if any."""
    row = conn.execute(
        f"SELECT {_PAIR_COLS} FROM pairs "
        "WHERE user_id = ? AND domain = ? AND phase = ? AND chosen_id IS NULL "
        "ORDER BY shown_at DESC, rowid DESC LIMIT 1",
        (user_id, domain, phase),
    ).fetchone()
    return None if row is None else _choice(row)


def list_choices(
    conn: sqlite3.Connection,
    user_id: str,
    domain: str | None = None,
    phase: str | None = None,
    answered_only: bool = True,
) -> list[Choice]:
    """A user's pairs, oldest first (by shown_at, then insertion order)."""
    where, args = ["user_id = ?"], [user_id]
    if domain is not None:
        where.append("domain = ?")
        args.append(domain)
    if phase is not None:
        where.append("phase = ?")
        args.append(phase)
    if answered_only:
        where.append("chosen_id IS NOT NULL")
    rows = conn.execute(
        f"SELECT {_PAIR_COLS} FROM pairs WHERE {' AND '.join(where)} "
        "ORDER BY shown_at ASC, rowid ASC",
        args,
    )
    return [_choice(r) for r in rows]


def shown_ids(conn: sqlite3.Connection, user_id: str, domain: str) -> set[str]:
    """Every item id this user has been shown in this domain, any phase, answered or not."""
    rows = conn.execute(
        "SELECT a_id, b_id FROM pairs WHERE user_id = ? AND domain = ?",
        (user_id, domain),
    )
    return {item for row in rows for item in row}
