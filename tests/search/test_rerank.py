import numpy as np
import pytest

from amber.search.rerank import SearchResult, search
from amber.space.index import VectorIndex
from amber.yous.base import You

QUERY = "red jacket"


@pytest.fixture
def index(ctx) -> VectorIndex:
    return ctx.index["clothes"]


@pytest.fixture
def everything(ctx) -> set[str]:
    return set(ctx.items["clothes"]["id"])


def _you(index: VectorIndex, seed: int, name: str = "art") -> You:
    w = np.random.default_rng(seed).standard_normal(index.vectors.shape[1])
    return You(name, (w / np.linalg.norm(w)).astype(np.float32))


def test_search_respects_allowed(ctx, index):
    allowed = {i for i in ctx.items["clothes"]["id"] if int(i.split(":")[1]) <= 24}
    result = search(QUERY, _you(index, 1), ctx.encoder, index, allowed, 30, 10)
    assert result.generic and result.personal
    assert set(result.generic) <= allowed
    assert set(result.personal) <= allowed


def test_generic_is_the_best_query_matches_in_order(ctx, index, everything):
    result = search(QUERY, None, ctx.encoder, index, everything, 20, 8)
    q = ctx.encoder.embed_texts([QUERY])[0]
    expected = [i for i, _ in index.top_k(q, 8, allowed=everything, use_centered=False)]
    assert result.generic == expected
    assert len(result.generic) == 8


def test_personal_is_reranked_from_the_candidates(ctx, index, everything):
    result = search(QUERY, _you(index, 1), ctx.encoder, index, everything, 20, 8)
    q = ctx.encoder.embed_texts([QUERY])[0]
    candidates = {i for i, _ in index.top_k(q, 20, allowed=everything, use_centered=False)}
    assert len(result.personal) == 8
    assert len(set(result.personal)) == 8
    assert set(result.personal) <= candidates


def test_ranking_changes_with_you(ctx, index, everything):
    a = search(QUERY, _you(index, 1), ctx.encoder, index, everything, 40, 10)
    b = search(QUERY, _you(index, 2), ctx.encoder, index, everything, 40, 10)
    assert a.generic == b.generic
    assert a.personal != b.personal
    assert a.personal != a.generic


def test_opposite_yous_rank_the_candidates_in_opposite_ends(ctx, index, everything):
    you = _you(index, 3)
    flipped = You("art", -you.w)
    n = 12
    up = search(QUERY, you, ctx.encoder, index, everything, n, n, lam=50.0)
    down = search(QUERY, flipped, ctx.encoder, index, everything, n, n, lam=50.0)
    assert set(up.personal) == set(down.personal)  # all n candidates are shown
    assert up.personal != down.personal
    assert up.personal[0] == down.personal[-1]  # lam dominates, so the order inverts


def test_lambda_zero_keeps_the_query_order(ctx, index, everything):
    result = search(QUERY, _you(index, 1), ctx.encoder, index, everything, 20, 8, lam=0.0)
    assert result.personal == result.generic


def test_no_you_means_generic_only(ctx, index, everything):
    result = search(QUERY, None, ctx.encoder, index, everything)
    assert isinstance(result, SearchResult)
    assert result.generic
    assert result.personal == []


def test_abstaining_you_means_generic_only(ctx, index, everything):
    result = search(QUERY, You("art", None), ctx.encoder, index, everything)
    assert result.generic
    assert result.personal == []


def test_empty_allowed_returns_nothing(ctx, index):
    result = search(QUERY, _you(index, 1), ctx.encoder, index, set())
    assert result == SearchResult([], [])


def test_fewer_candidates_than_results(ctx, index):
    allowed = {"c:1", "c:2", "c:3"}
    result = search(QUERY, _you(index, 1), ctx.encoder, index, allowed)
    assert len(result.generic) == 3
    assert sorted(result.personal) == ["c:1", "c:2", "c:3"]


def test_you_who_cannot_tell_candidates_apart_keeps_query_order(ctx, index, everything):
    zero = You("art", np.zeros(index.vectors.shape[1], dtype=np.float32))
    result = search(QUERY, zero, ctx.encoder, index, everything, 20, 8)
    assert result.personal == result.generic
