import numpy as np
import pytest

from amber.config import SEED
from amber.space.basis import Basis


def rand(*shape: int, seed: int = SEED) -> np.ndarray:
    """Seeded standard-normal float64 array of the given shape."""
    return np.random.default_rng(seed).standard_normal(shape)


def test_basis_orthonormal_and_lift_inverts_project():
    B = Basis.fit(rand(200, 16), k=4)
    assert np.allclose(B.components @ B.components.T, np.eye(4), atol=1e-5)
    w = rand(4)
    assert np.allclose(B.project(B.lift(w)[None])[0], w, atol=1e-5)


def test_shapes_and_dtype():
    B = Basis.fit(rand(50, 16), k=5)
    assert B.components.shape == (5, 16) and B.components.dtype == np.float32
    assert B.project(rand(7, 16)).shape == (7, 5)
    assert B.lift(rand(5)).shape == (16,)


def test_project_and_lift_are_pure_linear_maps_without_mean_subtraction():
    B = Basis.fit(rand(80, 16) + 3.0, k=4)  # data far from the origin
    X, w = rand(6, 16, seed=1), rand(4, seed=2)
    assert np.allclose(B.project(X), X @ B.components.T, atol=1e-5)
    assert np.allclose(B.lift(w), w @ B.components, atol=1e-5)
    assert np.allclose(B.project(np.zeros((1, 16))), 0)
    # linearity, which is what lets preference differences live in the basis
    a, b = X[0], X[1]
    assert np.allclose(
        B.project((a - b)[None])[0],
        B.project(a[None])[0] - B.project(b[None])[0],
        atol=1e-5,
    )


def test_fit_finds_the_high_variance_axes_in_order():
    rng = np.random.default_rng(SEED)
    scales = np.array([5.0, 2.0, 0.5] + [0.05] * 5)
    X = rng.standard_normal((500, 8)) * scales + 10.0  # offset: mean must be removed
    B = Basis.fit(X, k=3)
    for row, axis in enumerate((0, 1, 2)):
        assert abs(B.components[row, axis]) > 0.99
    assert np.all(
        np.diff(B.project(X - X.mean(0)).var(axis=0)) < 0
    )  # variance descends


def test_fit_removes_the_mean_before_the_svd():
    # all rows share a big offset along e0; the only real variation is along e1
    rng = np.random.default_rng(SEED)
    X = np.zeros((100, 6))
    X[:, 0] = 50.0
    X[:, 1] = rng.standard_normal(100)
    B = Basis.fit(X, k=1)
    assert abs(B.components[0, 1]) > 0.99


def test_signs_are_deterministic_largest_entry_positive():
    X = rand(60, 12)
    B = Basis.fit(X, k=6)
    biggest = B.components[np.arange(6), np.abs(B.components).argmax(axis=1)]
    assert (biggest > 0).all()
    assert np.array_equal(Basis.fit(X, k=6).components, B.components)
    assert np.allclose(Basis.fit(X[::-1], k=6).components, B.components, atol=1e-4)


@pytest.mark.parametrize(
    "n, d, k, expected_k",
    [(24, 16, 32, 16), (3, 16, 32, 3), (50, 16, 20, 16), (50, 16, 5, 5)],
)
def test_k_is_clamped_to_rows_and_dims(n, d, k, expected_k):
    B = Basis.fit(rand(n, d), k=k)
    assert B.components.shape == (expected_k, d)
    assert np.allclose(B.components @ B.components.T, np.eye(expected_k), atol=1e-5)


@pytest.mark.parametrize("k", [0, -3])
def test_fit_rejects_a_non_positive_k(k):
    with pytest.raises(ValueError):
        Basis.fit(rand(10, 4), k=k)


@pytest.mark.parametrize("X", [np.zeros((0, 4)), np.zeros(4)])
def test_fit_rejects_empty_or_non_matrix_input(X):
    with pytest.raises(ValueError):
        Basis.fit(X, k=2)


def test_fit_accepts_float32_input():
    B = Basis.fit(rand(40, 8).astype(np.float32), k=3)
    assert np.allclose(B.components @ B.components.T, np.eye(3), atol=1e-5)


def test_save_and_load_roundtrip(tmp_path):
    B = Basis.fit(rand(60, 16), k=4)
    out = tmp_path / "nested" / "pca"
    B.save(out)
    loaded = Basis.load(out)
    assert np.array_equal(loaded.components, B.components)
    assert loaded.components.dtype == np.float32
    X = rand(5, 16, seed=3)
    assert np.allclose(loaded.project(X), B.project(X))


def test_load_of_a_missing_basis_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        Basis.load(tmp_path / "nope")
