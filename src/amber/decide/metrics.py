"""Metrics, calibration and day-block bootstrap CIs for the decision ladder.

All CIs resample whole calendar days (with replacement) so that the strong
within-day correlation between plays is respected.
"""

import itertools
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.stats import rankdata

P_CLIP = 1e-15
N_BOOT = 2000
SEED = 7
RUNG_ORDER = ["base", "context", "memory", "recency"]


def _arrays(y, p) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y, dtype=float), np.asarray(p, dtype=float)


def _row_log_loss(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    p = np.clip(p, P_CLIP, 1 - P_CLIP)
    return -(y * np.log(p) + (1 - y) * np.log1p(-p))


def log_loss(y, p) -> float:
    y, p = _arrays(y, p)
    return float(_row_log_loss(y, p).mean())


def brier(y, p) -> float:
    y, p = _arrays(y, p)
    return float(((p - y) ** 2).mean())


def auc(y, p) -> float:
    """Rank-based Mann-Whitney AUC, ties as half; NaN if only one class."""
    y, p = _arrays(y, p)
    pos = y > 0.5
    n_pos = int(pos.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(p)  # average ranks for ties
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def accuracy(y, p) -> float:
    y, p = _arrays(y, p)
    return float(((p > 0.5) == (y > 0.5)).mean())


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    """Equal-width probability bins; empty bins are dropped."""
    y, p = _arrays(y, p)
    idx = np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append(
                {
                    "bin_low": b / bins,
                    "bin_high": (b + 1) / bins,
                    "n": int(m.sum()),
                    "mean_p": float(p[m].mean()),
                    "skip_rate": float(y[m].mean()),
                }
            )
    return pd.DataFrame(rows, columns=["bin_low", "bin_high", "n", "mean_p", "skip_rate"])


def day_block_ci(
    days: np.ndarray,
    stat: Callable[[np.ndarray], float],
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> tuple[float, float]:
    """95% CI of ``stat(row_indices)`` resampling whole days with replacement."""
    days = np.asarray(days)
    _, inverse = np.unique(days, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.cumsum(np.bincount(inverse))
    groups = np.split(order, bounds[:-1])
    rng = np.random.default_rng(seed)
    n_days = len(groups)
    vals = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, n_days, n_days)
        vals[i] = stat(np.concatenate([groups[j] for j in pick]))
    lo, hi = np.quantile(vals, [0.025, 0.975])
    return float(lo), float(hi)


class _DayBootstrap:
    """Vectorized day-block bootstrap of per-row means over a fixed set of rows.

    The resampled day indices are drawn once, so every statistic computed through
    one instance shares them (this is what makes gaps paired).
    """

    def __init__(self, days: np.ndarray, n_boot: int = N_BOOT, seed: int = SEED):
        _, self._inv = np.unique(days, return_inverse=True)
        self._counts = np.bincount(self._inv).astype(float)
        n_days = len(self._counts)
        rng = np.random.default_rng(seed)
        self._pick = rng.integers(0, n_days, size=(n_boot, n_days))
        self._denom = self._counts[self._pick].sum(axis=1)

    def ci(self, values: np.ndarray) -> tuple[float, float]:
        sums = np.bincount(self._inv, weights=values, minlength=len(self._counts))
        means = sums[self._pick].sum(axis=1) / self._denom
        lo, hi = np.quantile(means, [0.025, 0.975])
        return float(lo), float(hi)


def evaluate(
    preds: pd.DataFrame,
    skip: np.ndarray,
    days: np.ndarray,
    frozen_p: np.ndarray,
    holdout_frac: float = 0.2,
    *,
    start: int | None = None,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> dict:
    """Holdout metrics, paired rung gaps, frozen-vs-updating drift and calibration.

    ``start`` is the first holdout row (default ``int(n * (1 - holdout_frac))``);
    ``frozen_p`` holds the frozen memory-rung predictions for rows ``[start, n)``.
    """
    skip = np.asarray(skip, dtype=bool)
    days = np.asarray(days)
    n = len(skip)
    if len(preds) != n or len(days) != n:
        raise ValueError("preds, skip and days must have the same length")
    if start is None:
        start = int(n * (1 - holdout_frac))
    if not 0 <= start < n:
        raise ValueError(f"start {start} leaves an empty holdout for n={n}")
    frozen_p = np.asarray(frozen_p, dtype=float)
    if len(frozen_p) != n - start:
        raise ValueError(f"frozen_p has {len(frozen_p)} rows, expected {n - start}")

    y = skip[start:].astype(float)
    boot = _DayBootstrap(days[start:], n_boot=n_boot, seed=seed)
    cols = {r: preds[f"p_{r}"].to_numpy(dtype=float)[start:] for r in RUNG_ORDER}
    losses = {r: _row_log_loss(y, cols[r]) for r in RUNG_ORDER}

    rungs = {
        r: {
            "log_loss": float(losses[r].mean()),
            "brier": brier(y, cols[r]),
            "auc": auc(y, cols[r]),
            "accuracy": accuracy(y, cols[r]),
            "ll_ci": boot.ci(losses[r]),
        }
        for r in RUNG_ORDER
    }
    gaps = []
    for lower, higher in itertools.pairwise(RUNG_ORDER):
        diff = losses[lower] - losses[higher]
        gaps.append(
            {
                "lower": lower,
                "higher": higher,
                "ll_gain": float(diff.mean()),
                "ci": boot.ci(diff),
            }
        )
    frozen_loss = _row_log_loss(y, frozen_p)
    drift_diff = frozen_loss - losses["memory"]
    drift = {
        "frozen_ll": float(frozen_loss.mean()),
        "updating_ll": rungs["memory"]["log_loss"],
        "gain": float(drift_diff.mean()),
        "ci": boot.ci(drift_diff),
    }
    best = min(RUNG_ORDER, key=lambda r: rungs[r]["log_loss"])
    return {
        "n_total": n,
        "n_holdout": n - start,
        "base_rate_holdout": float(y.mean()),
        "rungs": rungs,
        "gaps": gaps,
        "drift": drift,
        "calibration": calibration_table(y, cols[best]),
    }
