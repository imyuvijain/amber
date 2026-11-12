import numpy as np
import pytest

from amber.explain.geometry import fit_map_basis, map_points, sample_ids, you_cosines
from amber.space.basis import Basis
from amber.space.index import VectorIndex
from amber.yous.base import You


def _you(name, w):
    return You(name, w, {})


def _unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_you_cosines_is_symmetric_with_unit_diagonal():
    yous = {
        "stated": _you("stated", _unit(1, 0, 0)),
        "art": _you("art", _unit(1, 1, 0)),
        "revealed": _you("revealed", _unit(0, 0, 1)),
    }
    cos = you_cosines(yous)
    assert set(cos) == {"stated", "art", "revealed"}
    for a in cos:
        assert cos[a][a] == pytest.approx(1.0)
        for b in cos:
            assert cos[a][b] == pytest.approx(cos[b][a])
    assert cos["stated"]["art"] == pytest.approx(np.sqrt(0.5))
    assert cos["stated"]["revealed"] == pytest.approx(0.0)


def test_you_cosines_leaves_out_abstaining_yous():
    yous = {"stated": _you("stated", _unit(1, 0)), "bigfive": _you("bigfive", None)}
    cos = you_cosines(yous)
    assert cos == {"stated": {"stated": pytest.approx(1.0)}}


def test_you_cosines_skips_a_zero_direction_instead_of_returning_nan():
    yous = {
        "stated": _you("stated", _unit(1, 0)),
        "art": _you("art", np.zeros(2, dtype=np.float32)),
    }
    cos = you_cosines(yous)
    assert set(cos) == {"stated"}
    assert all(np.isfinite(v) for row in cos.values() for v in row.values())


def test_sample_ids_is_seeded_sorted_and_capped():
    ids = [f"c:{i}" for i in range(50)]
    a = sample_ids(ids, 10)
    assert a == sample_ids(
        list(reversed(ids)), 10
    )  # same sample whatever the input order
    assert len(a) == 10 and len(set(a)) == 10 and set(a) <= set(ids)
    assert sample_ids(ids[:4], 10) == sorted(ids[:4])


@pytest.fixture
def spaces():
    rng = np.random.default_rng(0)

    def make(prefix, n):
        v = rng.standard_normal((n, 8)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return VectorIndex([f"{prefix}:{i}" for i in range(n)], v)

    return make("c", 60), make("a", 30)


def test_map_points_shape(spaces):
    clothes, art = spaces
    ids = sample_ids(clothes.ids, 40)
    basis2d = fit_map_basis(clothes, ids)
    yous = {
        "stated": _you("stated", _unit(*range(1, 9))),
        "bigfive": _you("bigfive", None),
        "revealed": _you("revealed", _unit(1, -1, 2, 0, 0, 1, 0, 3)),
    }
    out = map_points(clothes, basis2d, yous, ids, top=5, art_index=art, n_art=12)
    clothes_pts = [p for p in out["points"] if p["domain"] == "clothes"]
    art_pts = [p for p in out["points"] if p["domain"] == "art"]
    assert [p["id"] for p in clothes_pts] == ids
    assert len(art_pts) == 12
    assert all(set(p) == {"id", "domain", "x", "y"} for p in out["points"])
    assert all(np.isfinite([p["x"], p["y"]]).all() for p in out["points"])
    assert [m["you"] for m in out["yous"]] == [
        "stated",
        "revealed",
    ]  # abstainer: no marker
    assert all(len(m["top"]) == 5 and set(m["top"]) <= set(ids) for m in out["yous"])


def test_you_markers_sit_at_the_edge_of_the_dot_cloud(spaces):
    clothes, art = spaces
    ids = sample_ids(clothes.ids, 40)
    out = map_points(
        clothes,
        fit_map_basis(clothes, ids),
        {"stated": _you("stated", _unit(*range(1, 9)))},
        ids,
        art_index=art,
        n_art=10,
    )
    reach = max(
        np.hypot(p["x"], p["y"]) for p in out["points"] if p["domain"] == "clothes"
    )
    marker = out["yous"][0]
    assert np.hypot(marker["x"], marker["y"]) == pytest.approx(reach)


def test_marker_points_the_way_of_w_in_the_plane(spaces):
    clothes, art = spaces
    ids = sample_ids(clothes.ids, 40)
    basis2d = fit_map_basis(clothes, ids)
    w = _unit(*range(1, 9))
    out = map_points(
        clothes, basis2d, {"art": _you("art", w)}, ids, art_index=art, n_art=5
    )
    want = basis2d.project(w)
    got = np.array([out["yous"][0]["x"], out["yous"][0]["y"]])
    assert got @ want / (np.linalg.norm(got) * np.linalg.norm(want)) == pytest.approx(
        1.0
    )


def test_art_is_centered_by_the_art_mean(spaces):
    clothes, art = spaces
    ids = sample_ids(clothes.ids, 20)
    basis2d = fit_map_basis(clothes, ids)
    out = map_points(clothes, basis2d, {}, ids, art_index=art, n_art=6)
    pt = next(p for p in out["points"] if p["domain"] == "art")
    want = basis2d.project(art.centered([pt["id"]]))[0]
    assert (pt["x"], pt["y"]) == pytest.approx((want[0], want[1]), abs=1e-6)


def test_map_points_without_art_or_markers(spaces):
    clothes, _ = spaces
    ids = sample_ids(clothes.ids, 10)
    out = map_points(clothes, fit_map_basis(clothes, ids), {}, ids)
    assert len(out["points"]) == 10 and out["yous"] == []


def test_fit_map_basis_is_two_dimensional(spaces):
    clothes, _ = spaces
    basis = fit_map_basis(clothes, sample_ids(clothes.ids, 30))
    assert isinstance(basis, Basis) and basis.components.shape == (2, 8)
