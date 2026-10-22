"""Online Bayesian preference model: probit Bradley-Terry with Gaussian ADF."""

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

from amber import config


class PreferenceModel:
    """Gaussian posterior N(mean, cov) over a taste direction w.

    A choice "winner over loser" is modelled as P = Phi(w . (winner - loser)),
    and the posterior is updated by assumed-density filtering (moment matching).
    """

    def __init__(self, dim: int, prior_var: float | None = None):
        if prior_var is None:  # resolved now, so config reloads and tuning take effect
            prior_var = config.PRIOR_VAR
        self.dim = dim
        self.mean = np.zeros(dim)
        self.cov = prior_var * np.eye(dim)
        self.n_updates = 0

    def update(self, winner: np.ndarray, loser: np.ndarray) -> None:
        x = np.asarray(winner, dtype=float) - np.asarray(loser, dtype=float)
        s = self.cov @ x
        m = float(self.mean @ x)
        v = max(float(x @ s), 0.0)
        denom = np.sqrt(1.0 + v)
        z = m / denom
        # r = phi(z) / Phi(z), computed in log space so very negative z is stable.
        r = float(np.exp(norm.logpdf(z) - log_ndtr(z)))
        self.mean = self.mean + s * (r / denom)
        cov = self.cov - np.outer(s, s) * (r / (1.0 + v)) * (z + r)
        self.cov = (cov + cov.T) / 2.0
        self.n_updates += 1

    def p_prefer(self, a: np.ndarray, b: np.ndarray) -> float:
        x = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
        v = max(float(x @ self.cov @ x), 0.0)
        return float(norm.cdf(float(self.mean @ x) / np.sqrt(1.0 + v)))
