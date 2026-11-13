"""Online logistic learner and the four-rung prequential ladder.

Every play is predicted with weights trained only on plays with a strictly
earlier ``ts``: within a group of equal timestamps all rows are predicted first,
and only then used to update the model.
"""

import numpy as np
import pandas as pd

from amber.decide.features import CONTEXT_COLS, DECAY_COLS, MEMORY_COLS

P_MIN = 1e-4
STD_FLOOR = 0.1
Z_STD_CLIP = 5.0
Z_CLIP = 30.0

RUNGS: dict[str, list[str] | None] = {
    "base": None,
    "context": CONTEXT_COLS,
    "memory": CONTEXT_COLS + MEMORY_COLS,
    "recency": CONTEXT_COLS + MEMORY_COLS + DECAY_COLS,
}


class OnlineLogit:
    """Logistic regression trained one row at a time by plain SGD on log loss.

    Inputs are standardized with a running Welford mean and variance (mean 0 and
    std 1 before any update), with std = max(sqrt(var), 0.1) and each z clipped to
    [-5, 5] so a column that is constant for a long time and then turns on cannot
    blow the weights up. The bias is not L2-penalized. ``update`` takes its
    gradient step using the standardization that ``predict`` just used, then
    folds the row into the running statistics.
    """

    def __init__(self, n_features: int, lr: float = 0.05, l2: float = 1e-4):
        self.lr = lr
        self.l2 = l2
        self.w = np.zeros(n_features)
        self.b = 0.0
        self.count = 0
        self.mean = np.zeros(n_features)
        self.m2 = np.zeros(n_features)

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        if self.count == 0:
            return np.clip(x, -Z_STD_CLIP, Z_STD_CLIP)
        std = np.maximum(np.sqrt(self.m2 / self.count), STD_FLOOR)
        return np.clip((x - self.mean) / std, -Z_STD_CLIP, Z_STD_CLIP)

    def _prob(self, z: np.ndarray) -> float:
        logit = float(np.clip(self.w @ z + self.b, -Z_CLIP, Z_CLIP))
        return 1.0 / (1.0 + np.exp(-logit))

    def predict(self, x: np.ndarray) -> float:
        return self._prob(self._standardize(x))

    def update(self, x: np.ndarray, y: bool) -> None:
        z = self._standardize(x)
        err = self._prob(z) - float(y)
        self.w -= self.lr * (err * z + self.l2 * self.w)
        self.b -= self.lr * err
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)


def _check_lengths(feats: pd.DataFrame, skip: np.ndarray, ts: pd.Series) -> None:
    if not len(feats) == len(skip) == len(ts):
        raise ValueError(
            f"length mismatch: feats={len(feats)}, skip={len(skip)}, ts={len(ts)}"
        )


def holdout_start(ts: pd.Series, train_frac: float = 0.8) -> int:
    """First holdout row: int(n * train_frac), moved back to the start of its ts group.

    No holdout row then shares a timestamp with a training row.
    """
    values = _check_sorted(ts)
    start = int(len(values) * train_frac)
    while 0 < start < len(values) and values[start] == values[start - 1]:
        start -= 1
    return start


def _check_sorted(ts: pd.Series) -> np.ndarray:
    values = ts.to_numpy()
    if len(values) > 1 and (values[1:] < values[:-1]).any():
        raise ValueError("ts must be sorted ascending")
    return values


def _group_ends(values: np.ndarray) -> np.ndarray:
    """Exclusive end index of every run of equal timestamps."""
    n = len(values)
    if n == 0:
        return np.array([], dtype=int)
    return np.append(np.flatnonzero(values[1:] != values[:-1]) + 1, n)


def _clip(p) -> np.ndarray:
    return np.clip(p, P_MIN, 1 - P_MIN)


def _base_rate(skip: np.ndarray, ends: np.ndarray) -> np.ndarray:
    """(skips so far + 1) / (plays so far + 2), counting only earlier groups."""
    cum = np.concatenate([[0], np.cumsum(skip)])
    starts = np.concatenate([[0], ends[:-1]])
    out = np.empty(len(skip))
    for s, e in zip(starts, ends):
        out[s:e] = (cum[s] + 1) / (s + 2)
    return out


def _learner_columns(feats: pd.DataFrame, rung: str) -> np.ndarray:
    cols = RUNGS[rung]
    assert cols is not None
    return feats[cols].to_numpy(dtype=float)


def run_ladder(feats: pd.DataFrame, skip: np.ndarray, ts: pd.Series) -> pd.DataFrame:
    """Prequential predictions of every rung, columns ``p_<rung>``."""
    skip = np.asarray(skip, dtype=bool)
    _check_lengths(feats, skip, ts)
    values = _check_sorted(ts)
    ends = _group_ends(values)
    out = {"p_base": _clip(_base_rate(skip, ends))}
    for rung in ("context", "memory", "recency"):
        X = _learner_columns(feats, rung)
        model = OnlineLogit(X.shape[1])
        p = np.empty(len(skip))
        start = 0
        for end in ends:
            for i in range(start, end):
                p[i] = model.predict(X[i])
            for i in range(start, end):
                model.update(X[i], skip[i])
            start = end
        out[f"p_{rung}"] = _clip(p)
    return pd.DataFrame(out, index=feats.index)


def run_frozen(
    feats: pd.DataFrame,
    skip: np.ndarray,
    ts: pd.Series,
    rung: str,
    train_frac: float = 0.8,
) -> np.ndarray:
    """Train online on rows [0, start), then predict rows [start, n) with no updates.

    ``start`` is ``holdout_start(ts, train_frac)``.
    """
    if rung not in RUNGS:
        raise KeyError(rung)
    skip = np.asarray(skip, dtype=bool)
    _check_lengths(feats, skip, ts)
    n = len(skip)
    split = holdout_start(ts, train_frac)
    if RUNGS[rung] is None:
        p = np.full(n - split, (skip[:split].sum() + 1) / (split + 2))
    else:
        X = _learner_columns(feats, rung)
        model = OnlineLogit(X.shape[1])
        for i in range(split):
            model.update(X[i], skip[i])
        p = np.array([model.predict(X[i]) for i in range(split, n)])
    assert len(p) == n - split
    return _clip(p)
