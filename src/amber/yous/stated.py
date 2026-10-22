"""Stated you: the direction of what the person *says* they like.

Reads the free-text "describe yourself in clothes" answer, embeds each sentence as a
clothing-photo caption and takes the mean direction away from the neutral clothing prompt.
It never sees clothing choices (Leak rule).
"""

import re
import sqlite3
from typing import TYPE_CHECKING

from amber.store.db import get_answers
from amber.yous.base import You, text_direction

if TYPE_CHECKING:
    from amber.context import AppContext

_SENTENCE_BREAK = re.compile(r"[.!?\n]+")
_NO_DESCRIPTION = {"reason": "no description"}


def split_sentences(text: str) -> list[str]:
    """Split on `.`, `!`, `?` and newlines; strip each piece and drop the empty ones."""
    return [s for piece in _SENTENCE_BREAK.split(text) if (s := piece.strip())]


def build_stated(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """Abstains (`meta={"reason": "no description"}`) when the description is missing or blank."""
    text = get_answers(conn, user_id, "describe").get("text", "")
    sentences = split_sentences(text)
    if not sentences:
        return You("stated", None, dict(_NO_DESCRIPTION))
    try:
        w = text_direction(
            ctx.encoder, [f"a photo of clothing: {s}" for s in sentences]
        )
    except ValueError:
        return You("stated", None, {"reason": "no direction", "sentences": sentences})
    return You("stated", w, {"sentences": sentences})
