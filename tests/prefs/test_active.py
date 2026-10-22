import numpy as np
import pytest
from synthetic_catalog import synthetic_catalog

from amber.prefs.active import choose_pair, random_pair
from amber.prefs.model import PreferenceModel

SEED = 7


def test_choose_pair_distinct_and_not_excluded():
    X, D, IDS = synthetic_catalog()
    rng = np.random.default_rng(SEED)
    a, b = choose_pair(PreferenceModel(32), X, IDS, D, rng, exclude={"c:1"})
    assert a != b and "c:1" not in (a, b)
    assert a in IDS and b in IDS


def test_choose_pair_skips_near_duplicates():
    # c:1 and c:2 are far apart in X (the most informative pair by a mile) but their
    # full embeddings are identical: they are the same photo twice.
    n, k = 20, 8
    rng = np.random.default_rng(SEED)
    Xs = rng.normal(size=(n, k)) * 0.1
    Xs[1], Xs[2] = 50.0, -50.0
    Ds = rng.normal(size=(n, 16))
    Ds[2] = Ds[1]
    Ds /= np.linalg.norm(Ds, axis=1, keepdims=True)
    ids = [f"c:{i}" for i in range(n)]
    for seed in range(10):
        pair = choose_pair(PreferenceModel(k), Xs, ids, Ds, np.random.default_rng(seed))
        assert set(pair) != {"c:1", "c:2"}


def test_choose_pair_prefers_informative_pairs():
    # Once the model is sure of one axis, a pair differing only along it is uninformative.
    model = PreferenceModel(2)
    for _ in range(30):
        model.update(np.array([1.0, 0.0]), np.array([-1.0, 0.0]))
    Xs = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    Ds = np.eye(4)
    ids = ["c:0", "c:1", "c:2", "c:3"]
    pair = choose_pair(model, Xs, ids, Ds, np.random.default_rng(SEED))
    assert set(pair) == {"c:2", "c:3"}


def test_choose_pair_is_vectorised_and_fast():
    X, D, IDS = synthetic_catalog()
    import time

    model = PreferenceModel(32)
    rng = np.random.default_rng(SEED)
    start = time.perf_counter()
    for _ in range(20):
        choose_pair(model, X, IDS, D, rng)
    assert time.perf_counter() - start < 2.0


def test_choose_pair_raises_when_no_valid_pair():
    Ds = np.ones((5, 4)) / 2.0  # every pair is a duplicate
    with pytest.raises(ValueError, match="pair"):
        choose_pair(
            PreferenceModel(2),
            np.zeros((5, 2)),
            [f"c:{i}" for i in range(5)],
            Ds,
            np.random.default_rng(SEED),
        )


def test_random_pair_respects_exclude():
    _, D, IDS = synthetic_catalog()
    rng = np.random.default_rng(SEED)
    exclude = {f"c:{i}" for i in range(490)}
    for _ in range(50):
        a, b = random_pair(IDS, D, rng, exclude=exclude)
        assert a != b and a not in exclude and b not in exclude


def test_random_pair_skips_near_duplicates():
    Ds = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    ids = ["c:0", "c:1", "c:2"]
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        assert set(random_pair(ids, Ds, rng)) != {"c:0", "c:1"}


def test_random_pair_raises_when_nothing_left():
    _, D, IDS = synthetic_catalog()
    rng = np.random.default_rng(SEED)
    with pytest.raises(ValueError, match="pair"):
        random_pair(IDS, D, rng, exclude=set(IDS[:-1]))
