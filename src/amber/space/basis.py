"""A PCA basis for the embedding space: a handful of orthonormal directions that carry most
of the variance, plus the linear maps between the full space and those coordinates.

Preferences are learned in these few coordinates (`project`) and mapped back to a direction
in the full space (`lift`) to rank items. Both maps are purely linear (no mean subtraction):
the mean is only removed while *fitting* the components, so differences of projected vectors
equal projections of differences, and `project(lift(w)) == w`.
"""

from pathlib import Path

import numpy as np

COMPONENTS_FILE = "components.npy"


class Basis:
    """`components` has shape `(k, d)`: k orthonormal rows in the d-dimensional space."""

    def __init__(self, components: np.ndarray):
        components = np.asarray(components, dtype=np.float32)
        if components.ndim != 2:
            raise ValueError(
                f"components must be a 2-D matrix, got shape {components.shape}"
            )
        self.components = components

    @classmethod
    def fit(cls, X: np.ndarray, k: int) -> "Basis":
        """The top-`k` principal directions of the rows of `X` (PCA via SVD).

        `k` is clamped to `min(k, n_rows, n_dims)`. Each component's sign is fixed so that its
        largest-magnitude entry is positive, which keeps a refit on the same data stable.
        """
        X = np.asarray(X)
        if X.ndim != 2 or 0 in X.shape:
            raise ValueError(f"X must be a non-empty 2-D matrix, got shape {X.shape}")
        if k < 1:
            raise ValueError(f"k must be >= 1, got {k}")
        k = min(k, *X.shape)
        centered = X.astype(np.float64) - X.mean(axis=0, dtype=np.float64)
        components = np.linalg.svd(centered, full_matrices=False)[2][:k]
        biggest = np.abs(components).argmax(axis=1)
        components = components * np.sign(components[np.arange(k), biggest])[:, None]
        return cls(components)

    def project(self, X: np.ndarray) -> np.ndarray:
        """Basis coordinates `(n, k)` of the rows of `X` `(n, d)`: `X @ components.T`."""
        return np.asarray(X) @ self.components.T

    def lift(self, w: np.ndarray) -> np.ndarray:
        """Full-space direction `(d,)` for basis coordinates `w` `(k,)`: `w @ components`."""
        return np.asarray(w) @ self.components

    def save(self, dir: Path) -> None:
        """Write `components.npy` into `dir` (created if missing)."""
        dir = Path(dir)
        dir.mkdir(parents=True, exist_ok=True)
        np.save(dir / COMPONENTS_FILE, self.components)

    @classmethod
    def load(cls, dir: Path) -> "Basis":
        return cls(np.load(Path(dir) / COMPONENTS_FILE))
