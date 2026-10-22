import numpy as np
import pandas as pd
import pytest

from amber.config import PCA_DIM
from amber.space.basis import Basis
from amber.space.evaluate import label_retrieval_score
from amber.space.index import VectorIndex, embed_domain

D = 8


def _unit(v: np.ndarray) -> np.ndarray:
    return (v / np.linalg.norm(v)).astype(np.float32)


class OracleEncoder:
    """Maps each group's query text to the direction that group's items point along."""

    model_name = "oracle"
    pretrained = "test"
    dim = D

    def __init__(self, groups: dict[tuple[str, str], int]):
        self._by_text = {
            f"a photo of a {colour.lower()} {kind.lower()}": axis
            for (colour, kind), axis in groups.items()
        }

    def embed_texts(self, texts):
        out = np.zeros((len(texts), D), dtype=np.float32)
        for i, text in enumerate(texts):
            out[i, self._by_text[text]] = 1.0
        return out

    def embed_images(self, paths):
        raise AssertionError("label retrieval must not embed images")


def _catalog(sizes: dict[tuple[str, str], int], seed: int = 0):
    """Items + an index whose vectors cluster along one axis per group."""
    rng = np.random.default_rng(seed)
    rows, vectors = [], []
    groups = {}
    n = 0
    for axis, ((colour, kind), size) in enumerate(sizes.items()):
        groups[(colour, kind)] = axis
        for _ in range(size):
            n += 1
            base = np.zeros(D)
            base[axis] = 1.0
            vectors.append(_unit(base + rng.normal(scale=0.05, size=D)))
            rows.append({"id": f"c:{n}", "base_colour": colour, "article_type": kind})
    ids = [r["id"] for r in rows]
    return VectorIndex(ids, np.stack(vectors)), pd.DataFrame(rows), groups


def test_label_retrieval_perfect_encoder():
    idx, items, groups = _catalog(
        {("Red", "Dresses"): 8, ("Blue", "Jeans"): 8, ("Black", "Shoes"): 8}
    )
    assert (
        label_retrieval_score(idx, items, OracleEncoder(groups), k=5, min_group=5)
        == 1.0
    )


def test_label_retrieval_ignores_small_groups():
    idx, items, groups = _catalog({("Red", "Dresses"): 8, ("Blue", "Jeans"): 3})
    enc = OracleEncoder(groups)
    only_big = label_retrieval_score(idx, items, enc, k=5, min_group=5)
    assert only_big == 1.0
    # the small group is dropped, so a query encoder that knows only the big group works
    assert (
        label_retrieval_score(
            idx, items, OracleEncoder({("Red", "Dresses"): 0}), k=5, min_group=5
        )
        == 1.0
    )


def test_label_retrieval_counts_wrong_neighbours():
    idx, items, groups = _catalog({("Red", "Dresses"): 6, ("Blue", "Jeans"): 6})
    wrong = OracleEncoder({("Red", "Dresses"): 1, ("Blue", "Jeans"): 0})  # swapped axes
    assert label_retrieval_score(idx, items, wrong, k=5, min_group=5) == 0.0


def test_label_retrieval_precision_is_fractional_when_group_is_smaller_than_k():
    idx, items, groups = _catalog({("Red", "Dresses"): 5, ("Blue", "Jeans"): 10})
    score = label_retrieval_score(idx, items, OracleEncoder(groups), k=10, min_group=5)
    assert score == pytest.approx((0.5 + 1.0) / 2)


def test_label_retrieval_ranks_raw_vectors_not_centered(monkeypatch):
    idx, items, groups = _catalog({("Red", "Dresses"): 6, ("Blue", "Jeans"): 6})
    seen = []
    real = VectorIndex.top_k

    def spy(self, direction, k, allowed=None, use_centered=True):
        seen.append(use_centered)
        return real(self, direction, k, allowed, use_centered)

    monkeypatch.setattr(VectorIndex, "top_k", spy)
    label_retrieval_score(idx, items, OracleEncoder(groups), k=3, min_group=5)
    assert seen and not any(seen)


def test_label_retrieval_no_eligible_group_raises():
    idx, items, groups = _catalog({("Red", "Dresses"): 3})
    with pytest.raises(ValueError, match="min_group"):
        label_retrieval_score(idx, items, OracleEncoder(groups), k=2, min_group=5)


def test_embed_domain_writes_index_and_bases(fixture_data, fake_encoder):
    idx = embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    assert len(idx.ids) == 96 and (fixture_data / "clothes" / "emb").exists()
    emb = fixture_data / "clothes" / "emb" / "fake__fixture"
    loaded = VectorIndex.load(emb)
    assert loaded.ids == idx.ids and np.array_equal(loaded.vectors, idx.vectors)
    assert Basis.load(emb / "basis").components.shape == (min(PCA_DIM, 96, 16), 16)
    assert Basis.load(emb / "basis2d").components.shape == (2, 16)


def test_embed_domain_follows_catalog_order_and_embeds_the_thumbnails(
    fixture_data, fake_encoder
):
    idx = embed_domain("art", fake_encoder, data_dir=fixture_data)
    assert idx.ids == [f"a:{n}" for n in range(1, 25)]
    expected = fake_encoder.embed_images([fixture_data / "art" / "thumbs" / "1.jpg"])[0]
    assert np.allclose(idx.vec("a:1"), expected)


def test_embed_domain_basis_is_fit_on_centered_vectors(fixture_data, fake_encoder):
    idx = embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    emb = fixture_data / "clothes" / "emb" / "fake__fixture"
    want = Basis.fit(idx.centered(idx.ids), k=2)
    assert np.allclose(
        Basis.load(emb / "basis2d").components, want.components, atol=1e-5
    )


def test_embed_domain_leaves_no_temp_dirs(fixture_data, fake_encoder):
    embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    assert [p.name for p in (fixture_data / "clothes" / "emb").iterdir()] == [
        "fake__fixture"
    ]


def test_embed_domain_crash_leaves_no_half_written_dir(fixture_data, fake_encoder):
    class Exploding:
        model_name, pretrained, dim = "fake", "fixture", 16

        def embed_images(self, paths):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        embed_domain("clothes", Exploding(), data_dir=fixture_data)
    emb = fixture_data / "clothes" / "emb"
    assert not emb.exists() or list(emb.iterdir()) == []


def test_embed_domain_replaces_an_earlier_run(fixture_data, fake_encoder):
    embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    assert [p.name for p in (fixture_data / "clothes" / "emb").iterdir()] == [
        "fake__fixture"
    ]


def test_embed_domain_rejects_unknown_domain(fixture_data, fake_encoder):
    with pytest.raises(ValueError):
        embed_domain("shoes", fake_encoder, data_dir=fixture_data)
