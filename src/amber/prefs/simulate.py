"""Simulated people and the benchmark that compares pair-selection strategies."""

from typing import Literal

import numpy as np
from scipy.special import expit

from amber import config
from amber.prefs.active import pick_pair_indices, sample_pairs
from amber.prefs.model import PreferenceModel


class SimulatedUser:
    """A person with a taste direction `w` who picks `a` over `b` with probability
    `sigmoid((w.a - w.b) / temperature)`. `w=None` is a coin-flip clicker."""

    def __init__(
        self, w: np.ndarray | None, temperature: float, rng: np.random.Generator
    ):
        self.w = None if w is None else np.asarray(w, dtype=float)
        self.temperature = temperature
        self.rng = rng

    def choose(self, a: np.ndarray, b: np.ndarray) -> int:
        """0 if the user picks `a`, 1 if `b`."""
        if self.w is None:
            p_a = 0.5
        else:
            diff = float(
                self.w @ (np.asarray(a, dtype=float) - np.asarray(b, dtype=float))
            )
            p_a = float(expit(diff / self.temperature))
        return 0 if self.rng.random() < p_a else 1


def _one_user(
    X: np.ndarray,
    dup_vectors: np.ndarray,
    n_train: int,
    n_test: int,
    strategy: str,
    temperature: float,
    prior_var: float,
    random_clicker: bool,
    rng: np.random.Generator,
) -> float:
    n = len(X)
    w = None
    if not random_clicker:
        w = X[rng.integers(0, n)]
        w = w / max(float(np.linalg.norm(w)), 1e-12)
    user = SimulatedUser(w, temperature, rng)
    model = PreferenceModel(X.shape[1], prior_var=prior_var)

    seen = np.zeros(n, dtype=bool)
    near_dup = config.NEAR_DUP_COS
    for _ in range(n_train):
        allowed = np.flatnonzero(~seen)
        if strategy == "active":
            i, j = pick_pair_indices(
                model, X, dup_vectors, allowed, rng, near_dup_cos=near_dup
            )
        else:
            i, j = sample_pairs(rng, allowed, dup_vectors, 16, 1, near_dup)[0]
        seen[i] = seen[j] = True
        if user.choose(X[i], X[j]) == 0:
            model.update(X[i], X[j])
        else:
            model.update(X[j], X[i])

    pairs = sample_pairs(
        rng, np.flatnonzero(~seen), dup_vectors, 2 * n_test, n_test, near_dup
    )
    hits = 0
    for i, j in pairs:
        predicted_a = model.p_prefer(X[i], X[j]) > 0.5
        hits += predicted_a == (user.choose(X[i], X[j]) == 0)
    return hits / len(pairs)


def run_benchmark(
    X: np.ndarray,
    dup_vectors: np.ndarray,
    n_users: int,
    n_train: int,
    n_test: int,
    strategy: Literal["active", "random"],
    temperature: float,
    seed: int,
    prior_var: float | None = None,
    random_clicker: bool = False,
) -> float:
    """Mean held-out pairwise accuracy of `n_users` simulated people.

    Each person's taste `w` is the normalised projected vector of a random catalog item.
    They answer `n_train` pairs chosen by `strategy` (no item is shown twice), then the
    model is scored on `n_test` random pairs of items never used in training: it is right
    when its prediction (`p_prefer > 0.5` means `a`) matches what the person actually
    picks. `random_clicker=True` makes every person a coin flip, so accuracy is ~0.5.
    `prior_var` defaults to `config.PRIOR_VAR`, read at call time.
    """
    if strategy not in ("active", "random"):
        raise ValueError(f"unknown strategy {strategy!r}")
    if prior_var is None:
        prior_var = config.PRIOR_VAR
    X = np.asarray(X, dtype=float)
    dup_vectors = np.asarray(dup_vectors)
    accuracies = [
        _one_user(
            X,
            dup_vectors,
            n_train,
            n_test,
            strategy,
            temperature,
            prior_var,
            random_clicker,
            np.random.default_rng([seed, user]),
        )
        for user in range(n_users)
    ]
    return float(np.mean(accuracies))
