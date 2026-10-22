import numpy as np
from synthetic_catalog import synthetic_catalog

from amber.prefs.simulate import SimulatedUser, run_benchmark

SEED = 7


def test_simulated_user_follows_taste_and_temperature():
    rng = np.random.default_rng(SEED)
    w = np.array([1.0, 0.0])
    a, b = np.array([1.0, 0.0]), np.array([0.0, 0.0])
    cold = SimulatedUser(w, 0.01, rng)
    assert all(cold.choose(a, b) == 0 for _ in range(50))
    assert all(cold.choose(b, a) == 1 for _ in range(50))
    warm = SimulatedUser(w, 100.0, rng)
    picks = [warm.choose(a, b) for _ in range(2000)]
    assert 0.4 < np.mean(picks) < 0.6


def test_simulated_coin_flip_clicker():
    user = SimulatedUser(None, 0.05, np.random.default_rng(SEED))
    a, b = np.array([5.0, 0.0]), np.array([0.0, 0.0])
    picks = [user.choose(a, b) for _ in range(2000)]
    assert 0.4 < np.mean(picks) < 0.6


def test_active_beats_random():
    X, D, _ = synthetic_catalog()
    kw = {
        "n_users": 20,
        "n_train": 40,
        "n_test": 200,
        "temperature": 0.05,
        "seed": SEED,
    }
    act = run_benchmark(X, D, strategy="active", **kw)
    rnd = run_benchmark(X, D, strategy="random", **kw)
    assert act > rnd and act >= 0.70


def test_random_clicker_is_near_chance():
    X, D, _ = synthetic_catalog()
    acc = run_benchmark(
        X,
        D,
        n_users=20,
        n_train=40,
        n_test=200,
        strategy="active",
        temperature=0.05,
        seed=SEED,
        random_clicker=True,
    )
    assert 0.40 <= acc <= 0.60


def test_benchmark_is_reproducible():
    X, D, _ = synthetic_catalog()
    kw = {
        "n_users": 3,
        "n_train": 10,
        "n_test": 30,
        "strategy": "active",
        "temperature": 0.2,
    }
    assert run_benchmark(X, D, seed=3, **kw) == run_benchmark(X, D, seed=3, **kw)
