import numpy as np
import pytest

from amber.prefs.model import PreferenceModel

SEED = 7


def rand(dim: int, seed: int = SEED) -> np.ndarray:
    return np.random.default_rng(seed).normal(size=dim)


def e(i: int, dim: int = 4) -> np.ndarray:
    v = np.zeros(dim)
    v[i] = 1.0
    return v


def trained_model() -> PreferenceModel:
    m = PreferenceModel(4)
    rng = np.random.default_rng(SEED)
    for _ in range(20):
        a, b = rng.normal(size=4), rng.normal(size=4)
        m.update(a, b)
    return m


def test_prior_is_indifferent():
    m = PreferenceModel(4)
    a, b = rand(4, 1), rand(4, 2)
    assert m.p_prefer(a, b) == pytest.approx(0.5)
    assert m.p_prefer(a, a) == pytest.approx(0.5)


def test_update_moves_toward_winner():
    m = PreferenceModel(4)
    a, b = e(0), e(1)
    m.update(a, b)
    assert m.p_prefer(a, b) > 0.5 and m.mean @ (a - b) > 0 and m.n_updates == 1


def test_symmetry():
    m = trained_model()
    a, b = rand(4, 3), rand(4, 4)
    assert m.p_prefer(a, b) + m.p_prefer(b, a) == pytest.approx(1.0)


def test_variance_shrinks_along_compared_direction():
    m = PreferenceModel(4)
    x = e(0) - e(1)
    before = x @ m.cov @ x
    m.update(e(0), e(1))
    assert x @ m.cov @ x < before


def test_no_nan_after_many_random_updates():
    m = PreferenceModel(32)
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        a, b = rng.normal(size=32), rng.normal(size=32)
        m.update(*((a, b) if rng.random() < 0.5 else (b, a)))
    assert np.isfinite(m.mean).all() and np.isfinite(m.cov).all()
    assert np.all(np.linalg.eigvalsh(m.cov) > -1e-8)
