import numpy as np
import pytest

from amber.config import SEED
from amber.space.index import VectorIndex


def unit_rows(n: int, d: int, seed: int = SEED) -> np.ndarray:
    """`n` seeded random unit-norm float32 rows of dimension `d`."""
    x = np.random.default_rng(seed).standard_normal((n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def make_index(n: int = 6, d: int = 8) -> VectorIndex:
    return VectorIndex([f"c:{i}" for i in range(1, n + 1)], unit_rows(n, d))


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


# --- construction, save / load, lookup -------------------------------------------------


def test_index_roundtrip(tmp_path):
    idx = VectorIndex(["c:1", "c:2"], unit_rows(2, 8))
    idx.save(tmp_path)
    assert VectorIndex.load(tmp_path).ids == ["c:1", "c:2"]


def test_save_writes_the_documented_files_and_load_restores_vectors(tmp_path):
    idx = make_index()
    out = tmp_path / "nested" / "emb"  # save creates missing directories
    idx.save(out)
    assert sorted(p.name for p in out.iterdir()) == ["ids.json", "vectors.npy"]
    loaded = VectorIndex.load(out)
    assert loaded.vectors.dtype == np.float32
    assert np.array_equal(loaded.vectors, idx.vectors)
    assert np.array_equal(loaded.vec("c:4"), idx.vec("c:4"))


def test_load_of_a_missing_index_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        VectorIndex.load(tmp_path / "nope")


def test_vec_returns_the_row_and_does_not_alias_the_index():
    idx = make_index()
    v = idx.vec("c:3")
    assert v.shape == (8,) and np.array_equal(v, idx.vectors[2])
    v[:] = 0  # mutating the result must not corrupt the index
    assert np.linalg.norm(idx.vec("c:3")) == pytest.approx(1.0, abs=1e-5)


def test_vec_of_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        make_index().vec("c:99")


def test_vectors_are_coerced_to_float32():
    idx = VectorIndex(["c:1", "c:2"], unit_rows(2, 4).astype(np.float64))
    assert idx.vectors.dtype == np.float32 and idx.mean.dtype == np.float32


@pytest.mark.parametrize(
    "ids, shape",
    [
        (["c:1", "c:2"], (3, 4)),  # more rows than ids
        (["c:1", "c:2", "c:3"], (2, 4)),  # more ids than rows
        (["c:1", "c:1"], (2, 4)),  # duplicate id
        (["c:1", "c:2"], (2,)),  # not a matrix
    ],
)
def test_constructor_rejects_inconsistent_input(ids, shape):
    with pytest.raises(ValueError):
        VectorIndex(ids, np.ones(shape, dtype=np.float32))


def test_mean_is_the_mean_of_the_raw_vectors():
    x = unit_rows(10, 8)
    idx = VectorIndex([f"c:{i}" for i in range(10)], x)
    assert idx.mean.shape == (8,)
    assert np.allclose(idx.mean, x.mean(axis=0), atol=1e-6)


# --- centered ------------------------------------------------------------------------


def test_centered_rows_are_unit_and_mean_removed():
    ids = [f"c:{i}" for i in range(1, 21)]
    x = unit_rows(20, 8)
    idx = VectorIndex(ids, x)
    c = idx.centered(ids)
    assert c.shape == (20, 8) and c.dtype == np.float32
    assert np.allclose(np.linalg.norm(c, axis=1), 1, atol=1e-5)
    # each row is the normalised (v - mean), so removing the mean leaves zero average
    expected = x - x.mean(axis=0)
    assert np.allclose(expected.mean(axis=0), 0, atol=1e-6)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    assert np.allclose(c, expected, atol=1e-5)


def test_centered_follows_the_order_and_subset_asked_for():
    idx = make_index()
    everything = idx.centered(idx.ids)
    picked = idx.centered(["c:4", "c:1", "c:4"])
    assert np.allclose(picked, everything[[3, 0, 3]])
    assert idx.centered([]).shape == (0, 8)


def test_centered_of_unknown_id_raises_keyerror():
    with pytest.raises(KeyError):
        make_index().centered(["c:1", "c:99"])


def test_centered_returns_a_copy():
    idx = make_index()
    idx.centered(["c:1"])[:] = 7.0
    assert np.allclose(np.linalg.norm(idx.centered(["c:1"]), axis=1), 1, atol=1e-5)


def test_centered_row_equal_to_the_mean_is_zero_not_nan():
    same = np.tile(unit_rows(1, 8), (3, 1))  # every vector == the mean
    c = VectorIndex(["a:1", "a:2", "a:3"], same).centered(["a:1", "a:2", "a:3"])
    assert np.isfinite(c).all() and not c.any()


def test_centering_a_shared_offset_makes_clusters_visible():
    # two clusters far along +e0: raw cosines are all ~1, centred ones separate the clusters
    rng = np.random.default_rng(SEED)

    def cluster(sign: int) -> np.ndarray:
        x = np.zeros((5, 8), dtype=np.float32)
        x[:, 0], x[:, 1] = 5.0, sign * 0.5
        return x + 0.05 * rng.standard_normal((5, 8)).astype(np.float32)

    x = np.vstack([cluster(+1), cluster(-1)])
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    ids = [f"c:{i}" for i in range(10)]
    idx = VectorIndex(ids, x)
    raw = idx.vectors @ idx.vectors.T
    c = idx.centered(ids)
    cen = c @ c.T
    assert raw.min() > 0.9  # everything looks alike before centering
    assert (
        cen[:5, :5].mean() > 0.8 and cen[:5, 5:].mean() < -0.8
    )  # clusters split after


# --- top_k ---------------------------------------------------------------------------


def test_top_k_respects_allowed_and_sorts():
    idx = make_index()
    direction = unit_rows(1, 8, seed=3)[0]
    res = idx.top_k(direction, k=3, allowed={"c:1", "c:3"})
    assert {i for i, _ in res} <= {"c:1", "c:3"} and len(res) == 2
    assert [s for _, s in res] == sorted([s for _, s in res], reverse=True)


def test_top_k_matches_a_brute_force_ranking_on_centered_rows():
    idx = make_index(n=30)
    direction = unit_rows(1, 8, seed=3)[0]
    scores = idx.centered(idx.ids) @ direction
    want = sorted(zip(idx.ids, scores.tolist()), key=lambda t: -t[1])[:5]
    got = idx.top_k(direction, k=5)
    assert [i for i, _ in got] == [i for i, _ in want]
    assert np.allclose([s for _, s in got], [s for _, s in want], atol=1e-5)
    assert all(isinstance(s, float) for _, s in got)


def test_top_k_can_score_raw_vectors_instead_of_centered_ones():
    idx = make_index(n=30)
    direction = unit_rows(1, 8, seed=3)[0]
    scores = idx.vectors @ direction
    want = [idx.ids[i] for i in np.argsort(-scores)[:5]]
    got = idx.top_k(direction, k=5, use_centered=False)
    assert [i for i, _ in got] == want
    assert np.allclose([s for _, s in got], np.sort(scores)[::-1][:5], atol=1e-5)


def test_top_k_of_a_stored_vector_finds_itself_first():
    idx = make_index(n=30)
    (best, score), *_ = idx.top_k(idx.vec("c:7"), k=3, use_centered=False)
    assert best == "c:7" and score == pytest.approx(1.0, abs=1e-5)


def test_top_k_larger_than_the_index_returns_everything():
    idx = make_index(n=4)
    assert len(idx.top_k(unit_rows(1, 8, seed=3)[0], k=100)) == 4


def test_top_k_zero_and_empty_allowed_return_nothing():
    idx = make_index()
    direction = unit_rows(1, 8, seed=3)[0]
    assert idx.top_k(direction, k=0) == []
    assert idx.top_k(direction, k=3, allowed=set()) == []


def test_top_k_ignores_allowed_ids_that_are_not_indexed():
    idx = make_index()
    res = idx.top_k(unit_rows(1, 8, seed=3)[0], k=5, allowed={"c:2", "a:404"})
    assert [i for i, _ in res] == ["c:2"]


def test_top_k_breaks_ties_by_index_order():
    v = unit_rows(1, 8)
    idx = VectorIndex(["c:1", "c:2", "c:3"], np.vstack([v, v, v]))
    res = idx.top_k(v[0], k=3, use_centered=False)
    assert [i for i, _ in res] == ["c:1", "c:2", "c:3"]


def test_top_k_accepts_a_float64_direction():
    idx = make_index()
    direction = unit_rows(1, 8, seed=3)[0]
    assert idx.top_k(direction.astype(np.float64), k=3) == idx.top_k(direction, k=3)


@pytest.mark.parametrize("bad", [np.ones(5), np.ones((1, 8)), np.full(8, np.nan)])
def test_top_k_rejects_a_bad_direction(bad):
    with pytest.raises(ValueError):
        make_index().top_k(bad, k=3)


def test_top_k_rejects_a_negative_k():
    with pytest.raises(ValueError):
        make_index().top_k(np.ones(8), k=-1)
