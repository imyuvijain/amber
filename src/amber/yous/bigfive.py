"""Big Five you: the clothing direction a personality profile points to.

The player answers the 20 Mini-IPIP items (1-5). Each trait's score `s_t` in [-1, 1] weights
that trait's bridge direction, `text_direction(high_t) - text_direction(low_t)`, where the
poles are Mini-Markers adjectives phrased as "clothing worn by someone who is ...". It never
sees clothing choices (Leak rule).
"""

import sqlite3
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import yaml

from amber.store.db import get_answers
from amber.yous.base import You, text_direction

if TYPE_CHECKING:
    from amber.context import AppContext

Trait = Literal["O", "C", "E", "A", "N"]

FILES = Path(__file__).parent / "files"
IPIP_PATH = FILES / "mini_ipip.yaml"
BRIDGE_PATH = FILES / "bigfive_bridge.yaml"
MIN_ANSWERS = 10
_PROMPT = "clothing worn by someone who is {}, {} and {}"
_PROMPTS_PER_POLE = 3
_EPS = 1e-6


@dataclass(frozen=True)
class IpipItem:
    id: str
    text: str
    trait: Trait
    keyed: Literal[1, -1]


@cache
def _items() -> tuple[IpipItem, ...]:
    data = yaml.safe_load(IPIP_PATH.read_text())
    return tuple(
        IpipItem(row["id"], row["text"], row["trait"], row["keyed"])
        for row in data["items"]
    )


@cache
def _bridge() -> dict[str, dict[str, tuple[str, ...]]]:
    data = yaml.safe_load(BRIDGE_PATH.read_text())
    return {
        t: {pole: tuple(ws) for pole, ws in poles.items()} for t, poles in data.items()
    }


def load_ipip() -> list[IpipItem]:
    """The 20 Mini-IPIP items, in display (file) order."""
    return list(_items())


def load_bridge() -> dict[str, dict[str, list[str]]]:
    """`{trait: {"high": [4 adjectives], "low": [4 adjectives]}}`."""
    return {
        t: {p: list(ws) for p, ws in poles.items()} for t, poles in _bridge().items()
    }


def bridge_prompts(words: list[str]) -> list[str]:
    """Three prompts over rotations of `words`: (a, b, c), (b, c, d), (c, d, a)."""
    n = len(words)
    return [
        _PROMPT.format(*(words[(i + k) % n] for k in range(3)))
        for i in range(_PROMPTS_PER_POLE)
    ]


def score_ipip(answers: dict[str, int]) -> dict[str, float]:
    """Per answered trait, `(mean keyed answer - 3) / 2` in [-1, 1]; reverse-keyed answers
    count as `6 - x`. Traits with no answers are omitted. Raises `ValueError` on an unknown
    item id or a value outside 1..5."""
    by_id = {item.id: item for item in _items()}
    keyed: dict[str, list[int]] = {}
    for item_id, value in answers.items():
        if item_id not in by_id:
            raise ValueError(f"unknown Mini-IPIP item: {item_id!r}")
        if isinstance(value, bool) or value not in (1, 2, 3, 4, 5):
            raise ValueError(f"answer for {item_id} must be 1..5, got {value!r}")
        item = by_id[item_id]
        keyed.setdefault(item.trait, []).append(value if item.keyed == 1 else 6 - value)
    return {t: (float(np.mean(xs)) - 3.0) / 2.0 for t, xs in keyed.items()}


def trait_direction(encoder, trait: str) -> np.ndarray:
    """`text_direction(high) - text_direction(low)` for one trait (not normalized)."""
    poles = _bridge()[trait]
    high = text_direction(encoder, bridge_prompts(list(poles["high"])))
    low = text_direction(encoder, bridge_prompts(list(poles["low"])))
    return high.astype(np.float64) - low.astype(np.float64)


def build_bigfive(conn: sqlite3.Connection, user_id: str, ctx: "AppContext") -> You:
    """Abstains (`{"reason": "skipped"}`) with fewer than 10 answered items, or
    (`{"reason": "no direction", ...}`) when every trait scores 0."""
    answers = {k: int(v) for k, v in get_answers(conn, user_id, "ipip").items()}
    if len(answers) < MIN_ANSWERS:
        return You("bigfive", None, {"reason": "skipped"})
    scores = score_ipip(answers)
    terms = [s * trait_direction(ctx.encoder, t) for t, s in scores.items() if s]
    total = np.sum(terms, axis=0) if terms else None
    norm = float(np.linalg.norm(total)) if total is not None else 0.0
    if not np.isfinite(norm) or norm < _EPS:
        return You("bigfive", None, {"reason": "no direction", "traits": scores})
    return You("bigfive", (total / norm).astype(np.float32), {"traits": scores})
