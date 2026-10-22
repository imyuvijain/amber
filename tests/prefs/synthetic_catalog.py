import functools

import numpy as np

SEED = 7


@functools.cache
def synthetic_catalog(n: int = 500, k: int = 32, seed: int = SEED):
    """`n` points in `k` dims with cluster structure, scaled like the real projected
    clothes (mean row norm ~0.75, variance decaying across components), plus
    unit-length "full embeddings" for the near-duplicate test."""
    rng = np.random.default_rng(seed)
    spread = (1.0 + np.arange(k)) ** -0.4
    centers = rng.normal(size=(8, k)) * spread
    X = centers[rng.integers(0, 8, n)] + rng.normal(size=(n, k)) * spread * 0.4
    X *= 0.75 / np.linalg.norm(X, axis=1).mean()
    D = rng.normal(size=(n, 64))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    ids = [f"c:{i}" for i in range(n)]
    return X, D, ids
