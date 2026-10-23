"""Stated you: the direction of what the person *says* they like.

Reads the style words the person tapped (1-5 from a frozen list), embeds each as a clothing-photo
caption and takes the mean direction away from the neutral clothing prompt. It never sees
clothing choices (Leak rule).
"""

import json
import sqlite3
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from amber.store.db import get_answers
from amber.yous.base import You, text_direction

if TYPE_CHECKING:
    from amber.context import AppContext

STYLE_WORDS_PATH = Path(__file__).parent / "files" / "style_words.yaml"
_NO_WORDS = {"reason": "no words"}


@cache
def _rows() -> tuple[tuple[str, ...], ...]:
    data = yaml.safe_load(STYLE_WORDS_PATH.read_text())
    return tuple(tuple(row) for row in data["rows"])


def load_style_words_rows() -> list[list[str]]:
    """The style words in display rows, in file order."""
    return [list(row) for row in _rows()]


def load_style_words() -> list[str]:
    """Every style word, flat, in file order."""
    return [w for row in _rows() for w in row]


def build_stated(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """Abstains (`meta={"reason": "no words"}`) when no style words were tapped."""
    raw = get_answers(conn, user_id, "stated").get("words")
    words: list[str] = json.loads(raw) if raw else []
    if not words:
        return You("stated", None, dict(_NO_WORDS))
    try:
        w = text_direction(ctx.encoder, [f"a photo of {word} clothing" for word in words])
    except ValueError:
        return You("stated", None, {"reason": "no direction", "words": words})
    return You("stated", w, {"words": words})
