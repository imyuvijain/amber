"""A flat, in-memory nearest-neighbour index over L2-normalised embeddings.

`VectorIndex` holds one unit-norm float32 row per item id (`"c:12"`, `"a:9"`), in a fixed
order. Besides the raw vectors it can serve *centered* rows, `normalize(v - mean)`: CLIP
embeddings share a large common component, and subtracting the mean makes dot products
reflect what distinguishes items from one another.

On disk an index is `vectors.npy` + `ids.json` inside a directory.
"""

import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np

VECTORS_FILE = "vectors.npy"
IDS_FILE = "ids.json"
_EPS = (
    1e-6  # rows with a smaller norm are float32 noise: left as zeros when normalising
)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    """Unit-norm rows; a (near-)zero row stays all zeros instead of becoming NaN."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return np.divide(x, norms, out=np.zeros_like(x), where=norms > _EPS)


class VectorIndex:
    """`ids[i]` names the row `vectors[i]`.

    `vectors` are expected to be L2-normalised already (the encoders guarantee it); they are
    stored as given, as float32. Ids must be unique.
    """

    def __init__(self, ids: list[str], vectors: np.ndarray):
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"vectors must be a 2-D matrix, got shape {vectors.shape}")
        if len(ids) != len(vectors):
            raise ValueError(f"{len(ids)} ids but {len(vectors)} vectors")
        row = {item_id: i for i, item_id in enumerate(ids)}
        if len(row) != len(ids):
            raise ValueError("ids must be unique")
        self.ids: list[str] = list(ids)
        self.vectors = vectors
        self._row = row
        # mean of the raw vectors (accumulated in float64; an empty index has mean 0)
        self.mean: np.ndarray = (
            vectors.mean(axis=0, dtype=np.float64).astype(np.float32)
            if len(vectors)
            else np.zeros(vectors.shape[1], dtype=np.float32)
        )
        self._centered: np.ndarray | None = None  # all rows, computed on first use

    def __len__(self) -> int:
        return len(self.ids)

    # --- persistence -----------------------------------------------------------------

    def save(self, dir: Path) -> None:
        """Write `vectors.npy` and `ids.json` into `dir` (created if missing)."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / VECTORS_FILE, self.vectors)
        (dir / IDS_FILE).write_text(json.dumps(self.ids))

    @classmethod
    def load(cls, dir: Path) -> "VectorIndex":
        dir = Path(dir)
        vectors = np.load(dir / VECTORS_FILE)
        ids = json.loads((dir / IDS_FILE).read_text())
        return cls(ids, vectors)

    # --- lookup ----------------------------------------------------------------------

    def vec(self, item_id: str) -> np.ndarray:
        """The raw (unit-norm) vector of one item, as a copy. KeyError if not indexed."""
        return self.vectors[self._row[item_id]].copy()

    def _centered_all(self) -> np.ndarray:
        if self._centered is None:
            self._centered = _normalize_rows(self.vectors - self.mean)
        return self._centered

    def centered(self, item_ids: Sequence[str]) -> np.ndarray:
        """`normalize(v - mean)` for each id, in the order given, as a new `(n, d)` array.

        A vector equal to the mean has no direction left; its row is all zeros.
        """
        rows = np.fromiter(
            (self._row[i] for i in item_ids), dtype=np.int64, count=len(item_ids)
        )
        return self._centered_all()[rows]

    # --- search ----------------------------------------------------------------------

    def top_k(
        self,
        direction: np.ndarray,
        k: int,
        allowed: set[str] | None = None,
        use_centered: bool = True,
    ) -> list[tuple[str, float]]:
        """The `k` items scoring highest on the dot product with `direction`, best first.

        Rows scored are the centered ones, or the raw vectors when `use_centered` is False.
        With `allowed`, only those ids compete (ids that are not indexed are ignored).
        Ties keep index order.
        """
        direction = np.asarray(direction, dtype=np.float32)
        if direction.shape != (self.vectors.shape[1],):
            raise ValueError(
                f"direction must have shape ({self.vectors.shape[1]},), "
                f"got {direction.shape}"
            )
        if not np.isfinite(direction).all():
            raise ValueError("direction must be finite")
        if k < 0:
            raise ValueError(f"k must be >= 0, got {k}")

        rows = None
        if allowed is not None:
            rows = np.fromiter(
                sorted(self._row[i] for i in allowed if i in self._row), dtype=np.int64
            )
        if k == 0 or (rows is not None and len(rows) == 0):
            return []

        matrix = self._centered_all() if use_centered else self.vectors
        scores = matrix @ direction  # one matvec over the whole index
        if rows is not None:
            scores = scores[rows]
        order = np.argsort(-scores, kind="stable")[:k]
        row_of = order if rows is None else rows[order]
        return [(self.ids[r], float(s)) for r, s in zip(row_of, scores[order])]
