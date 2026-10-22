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
