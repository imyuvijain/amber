"""Choosing which pair to show next.

`choose_pair` is active: of a few thousand random candidate pairs it takes the one the
model is least sure about *and* knows least about (largest `v * p * (1 - p)`, where `v`
is the posterior variance of the score difference and `p` the predicted probability).
`random_pair` is the baseline. Both refuse near-identical photos.
"""

from collections.abc import Collection

import numpy as np
from scipy.special import ndtr

from amber.config import NEAR_DUP_COS
from amber.prefs.model import PreferenceModel

_MAX_ROUNDS = 50


def _sample_pairs(
    rng: np.random.Generator,
    allowed: np.ndarray,
    dup_vectors: np.ndarray,
    draw: int,
    need: int,
    near_dup_cos: float,
) -> np.ndarray:
    """Random pairs `(m, 2)` of row indices from `allowed`: distinct, not near-duplicates.

    Draws `draw` pairs per round until at least `need` valid ones are in hand (at most
    `_MAX_ROUNDS` rounds); returns up to `need`. Raises ValueError if there is none.
    """
    if len(allowed) >= 2:
        found: list[np.ndarray] = []
        count = 0
        for _ in range(_MAX_ROUNDS):
            pairs = allowed[rng.integers(0, len(allowed), size=(draw, 2))]
            pairs = pairs[pairs[:, 0] != pairs[:, 1]]
            u, v = dup_vectors[pairs[:, 0]], dup_vectors[pairs[:, 1]]
            norms = np.linalg.norm(u, axis=1) * np.linalg.norm(v, axis=1)
            cos = np.einsum("ij,ij->i", u, v) / np.maximum(norms, 1e-12)
            pairs = pairs[cos <= near_dup_cos]
            if len(pairs):
                found.append(pairs)
                count += len(pairs)
            if count >= need:
                break
        if count:
            return np.concatenate(found)[:need]
    raise ValueError(
        "No valid pair left to show (too few items, or all are near-duplicates)"
    )


def _allowed(ids: list[str], exclude: Collection[str]) -> np.ndarray:
    if not exclude:
        return np.arange(len(ids))
    return np.flatnonzero([i not in exclude for i in ids])


def _most_informative(model: PreferenceModel, X: np.ndarray, pairs: np.ndarray) -> int:
    """Row of `pairs` with the largest `v * p * (1 - p)`, scored in one vectorised pass."""
    x = X[pairs[:, 0]] - X[pairs[:, 1]]
    v = np.maximum(np.einsum("ij,jk,ik->i", x, model.cov, x), 0.0)
    p = ndtr((x @ model.mean) / np.sqrt(1.0 + v))
    return int(np.argmax(v * p * (1.0 - p)))


def pick_pair_indices(
    model: PreferenceModel,
    X: np.ndarray,
    dup_vectors: np.ndarray,
    allowed: np.ndarray,
    rng: np.random.Generator,
    n_candidates: int = 2000,
    near_dup_cos: float = NEAR_DUP_COS,
) -> tuple[int, int]:
    """`choose_pair` on row indices: `allowed` are the rows that may be shown."""
    pairs = _sample_pairs(rng, allowed, dup_vectors, n_candidates, 1, near_dup_cos)
    i, j = pairs[_most_informative(model, X, pairs)]
    return int(i), int(j)


def choose_pair(
    model: PreferenceModel,
    X: np.ndarray,
    ids: list[str],
    dup_vectors: np.ndarray,
    rng: np.random.Generator,
    exclude: set[str] = frozenset(),
    n_candidates: int = 2000,
    near_dup_cos: float = NEAR_DUP_COS,
) -> tuple[str, str]:
    """The most informative of `n_candidates` random pairs from the non-excluded ids.

    `X` holds the projected rows for `ids`; `dup_vectors` the full normalised
    embeddings, in the same row order. Pairs whose `dup_vectors` have cosine strictly
    above `near_dup_cos` are dropped before scoring.
    """
    i, j = pick_pair_indices(
        model, X, dup_vectors, _allowed(ids, exclude), rng, n_candidates, near_dup_cos
    )
    return ids[i], ids[j]


def random_pair(
    ids: list[str],
    dup_vectors: np.ndarray,
    rng: np.random.Generator,
    exclude: set[str] = frozenset(),
    near_dup_cos: float = NEAR_DUP_COS,
) -> tuple[str, str]:
    """A uniformly random pair of distinct, non-excluded, non-near-duplicate ids."""
    pairs = _sample_pairs(rng, _allowed(ids, exclude), dup_vectors, 16, 1, near_dup_cos)
    i, j = pairs[0]
    return ids[int(i)], ids[int(j)]
