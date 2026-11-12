"""Text search over clothes, with and without a You.

The query is embedded as text and matched to item embeddings by cosine: the best
`n_candidates` allowed items are the candidates, and the first `n_results` of them are the
generic results. The personal results rerank those same candidates by
`z(query_sim) + lam * z(you . centered(item))`, with both z-scores taken within the
candidate set, so the You can reorder what the query found but never add items to it.
"""

from dataclasses import dataclass

import numpy as np

from amber.config import SEARCH_CANDIDATES, SEARCH_LAMBDA, SEARCH_RESULTS
from amber.space.encoder import Encoder
from amber.space.index import VectorIndex
from amber.yous.base import You


@dataclass
class SearchResult:
    generic: list[str]
    personal: list[str]


def _zscore(x: np.ndarray) -> np.ndarray:
    """`x` standardised; all zeros when it has no spread (so nothing divides by zero)."""
    spread = x.std()
    return (x - x.mean()) / spread if spread > 1e-9 else np.zeros_like(x)


def search(
    query: str,
    you: You | None,
    encoder: Encoder,
    index: VectorIndex,
    allowed: set[str],
    n_candidates: int = SEARCH_CANDIDATES,
    n_results: int = SEARCH_RESULTS,
    lam: float = SEARCH_LAMBDA,
) -> SearchResult:
    """Items matching `query` among `allowed`, plainly and ranked by `you`.

    With `you` None or abstaining (`w` None), `personal` is empty.
    """
    q = np.asarray(encoder.embed_texts([query]), dtype=np.float32)[0]
    found = index.top_k(q, n_candidates, allowed=allowed, use_centered=False)
    ids = [item_id for item_id, _ in found]
    generic = ids[:n_results]
    if you is None or you.w is None or not ids:
        return SearchResult(generic, [])
    query_sim = np.array([s for _, s in found], dtype=np.float64)
    taste = index.centered(ids).astype(np.float64) @ np.asarray(you.w, dtype=np.float64)
    score = _zscore(query_sim) + lam * _zscore(taste)
    order = np.argsort(-score, kind="stable")[:n_results]
    return SearchResult(generic, [ids[i] for i in order])
