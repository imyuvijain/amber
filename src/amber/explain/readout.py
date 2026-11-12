"""The readout: "you said... you pick..." in plain words.

Describes the Yous with a frozen list of style concepts (`concepts.yaml`). A concept's score
for a You is the cosine between the You's direction `w` and the concept's text direction
(the same `a photo of <concept> clothing` prompt the Stated you uses), so the readout shows
what each direction points at, in words a person recognises.
"""

from collections import Counter
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from amber.config import NEUTRAL_PROMPT
from amber.space.encoder import Encoder
from amber.store.db import Choice
from amber.yous.base import You

CONCEPTS_PATH = Path(__file__).parent / "concepts.yaml"
TOP_PICK_CONCEPTS = 5
TOP_ART_CONCEPTS = 5
TOP_LABELS = 3
LABEL_COLUMNS = ("base_colour", "article_type")
MIN_LABEL_RATE = 0.5  # a label only counts as "picked" when chosen more often than not
SURPRISE_POOL = 10  # revealed's top concepts that surprises are drawn from
MAX_SURPRISES = 3
_EPS = 1e-6  # a direction shorter than this is float32 noise, not a direction


@cache
def _concepts() -> tuple[str, ...]:
    return tuple(yaml.safe_load(CONCEPTS_PATH.read_text())["concepts"])


def load_concepts() -> list[str]:
    """The frozen style concepts, in file order (a superset of the 30 style words)."""
    return list(_concepts())


def concept_alignment(
    you: You, encoder: Encoder, concepts: list[str]
) -> list[tuple[str, float]]:
    """`(concept, cosine(you.w, concept direction))` sorted best first; `[]` if `you` abstains.

    A concept's direction is what `a photo of <concept> clothing` adds to the neutral clothing
    prompt (as in `text_direction`); a concept with no direction is skipped.
    """
    if you.w is None or not concepts:
        return []
    texts = [f"a photo of {c} clothing" for c in concepts] + [NEUTRAL_PROMPT]
    embedded = np.asarray(encoder.embed_texts(texts), dtype=np.float64)
    directions = embedded[:-1] - embedded[-1]
    norms = np.linalg.norm(directions, axis=1)
    w = np.asarray(you.w, dtype=np.float64)
    scored = [
        (concept, float(direction @ w / norm))
        for concept, direction, norm in zip(concepts, directions, norms)
        if np.isfinite(norm) and norm >= _EPS
    ]
    return sorted(scored, key=lambda pair: (-pair[1], pair[0]))


def label_lift(
    choices: list[Choice], items: pd.DataFrame, column: str, min_count: int = 3
) -> list[tuple[str, float]]:
    """`(label, chosen / shown)` for each value of `column`, best first, then by label.

    Counts answered choices only. Both items of a pair are shown once; the chosen one is
    picked once. Labels shown fewer than `min_count` times are dropped; items without a
    label (missing, NA, or not in `items`) are ignored.
    """
    labels = items.set_index("id")[column]
    shown: Counter[str] = Counter()
    picked: Counter[str] = Counter()
    for choice in choices:
        if choice.chosen_id is None:
            continue
        for item_id in (choice.a_id, choice.b_id):
            label = labels.get(item_id)
            if label is None or pd.isna(label):
                continue
            shown[str(label)] += 1
            if item_id == choice.chosen_id:
                picked[str(label)] += 1
    rates = [(label, picked[label] / n) for label, n in shown.items() if n >= min_count]
    return sorted(rates, key=lambda pair: (-pair[1], pair[0]))


@dataclass
class Readout:
    said: list[str]
    picks: list[str]
    art_says: list[str]
    surprises: list[str]
    sentences: list[str]


def _top(you: You | None, encoder: Encoder, concepts: list[str], n: int) -> list[str]:
    if you is None:
        return []
    return [c for c, _ in concept_alignment(you, encoder, concepts)[:n]]


def _join(words: list[str]) -> str:
    return ", ".join(words)


def build_readout(
    yous: dict[str, You], choices: list[Choice], items: pd.DataFrame, encoder: Encoder
) -> Readout:
    """What the person said, what they pick, what their art taste says, and the gaps.

    * `said`: the tapped style words (stated's `meta["words"]`).
    * `picks`: revealed's top concepts plus the best-liked colours and garment types
      (label rate above one half), lowercased and de-duplicated. Empty if revealed abstains.
    * `art_says`: the art you's top concepts.
    * `surprises`: up to three of revealed's top ten concepts that were not said.
    * `sentences`: short plain English; a sentence is omitted when its You is missing or abstains.
    """
    concepts = load_concepts()
    stated = yous.get("stated")
    said = list(stated.meta.get("words", [])) if stated and stated.w is not None else []
    revealed, art = yous.get("revealed"), yous.get("art")

    picks: list[str] = []
    surprises: list[str] = []
    if revealed is not None and revealed.w is not None:
        ranked = [c for c, _ in concept_alignment(revealed, encoder, concepts)]
        labels = [
            label.lower()
            for column in LABEL_COLUMNS
            for label, rate in label_lift(choices, items, column)[:TOP_LABELS]
            if rate > MIN_LABEL_RATE
        ]
        picks = list(dict.fromkeys(ranked[:TOP_PICK_CONCEPTS] + labels))
        said_lower = {w.lower() for w in said}
        surprises = [c for c in ranked[:SURPRISE_POOL] if c not in said_lower][
            :MAX_SURPRISES
        ]
    art_says = _top(art, encoder, concepts, TOP_ART_CONCEPTS)

    sentences = []
    if said:
        sentences.append(f"You said: {_join(said)}.")
    if picks:
        sentences.append(f"You pick: {_join(picks)}.")
    if art_says:
        sentences.append(f"Your art taste says: {_join(art_says)}.")
    sentences += [f"Surprise: you pick {c} but never said it." for c in surprises]
    return Readout(said, picks, art_says, surprises, sentences)
