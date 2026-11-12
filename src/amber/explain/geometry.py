"""Where the Yous sit: their pairwise cosines and a 2-D map of clothes, art and the Yous.

The map is drawn in a 2-D PCA basis fit on clothes only. Art items are centered by the art
domain's own mean (as everywhere else) and then projected into that clothes basis, so both
kinds of dots share one plane. A You is a direction, so its marker is `w` projected into the
plane and rescaled to the edge of the dot cloud: only the direction on the map means anything.
"""

import numpy as np

from amber.config import SEED
from amber.space.basis import Basis
from amber.space.index import VectorIndex
from amber.yous.base import You

MAP_CLOTHES = 300
MAP_ART = 100
_EPS = 1e-9


def you_cosines(yous: dict[str, You]) -> dict[str, dict[str, float]]:
    """`{a: {b: cosine(w_a, w_b)}}` over the Yous that have a direction (abstainers are left out)."""
    live = {
        name: np.asarray(y.w, dtype=np.float64)
        for name, y in yous.items()
        if y.w is not None
    }
    unit = {name: w / np.linalg.norm(w) for name, w in live.items()}
    return {a: {b: float(unit[a] @ unit[b]) for b in unit} for a in unit}


def sample_ids(ids: list[str], n: int, seed: int = SEED) -> list[str]:
    """Up to `n` of `ids`, sorted, drawn with a seeded generator (input order does not matter)."""
    ordered = sorted(ids)
    if len(ordered) <= n:
        return ordered
    picked = np.random.default_rng(seed).choice(len(ordered), size=n, replace=False)
    return [ordered[i] for i in sorted(picked)]


_draw = sample_ids  # `map_points` has a parameter of the same name


def fit_map_basis(index: VectorIndex, ids: list[str]) -> Basis:
    """The 2-D PCA basis of the centered vectors of `ids` (the plane the map is drawn in)."""
    return Basis.fit(index.centered(ids), k=2)


def map_points(
    index: VectorIndex,
    basis2d: Basis,
    yous: dict[str, You],
    sample_ids: list[str],
    top: int = 20,
    art_index: VectorIndex | None = None,
    n_art: int = MAP_ART,
) -> dict:
    """The dots and markers of the map.

    `index` holds the clothes; `sample_ids` are the clothing dots. With `art_index`, `n_art`
    seeded art items are projected into the same plane (centered by the art mean).
    Returns `{"points": [{id, domain, x, y}], "yous": [{you, x, y, top}]}`, where each
    non-abstaining You has one marker at the edge of the clothing dots, in the direction of
    its `w` in the plane, and `top` is the ids among `sample_ids` it scores highest.
    """
    clothes_xy = basis2d.project(index.centered(sample_ids))
    points = [
        {"id": i, "domain": "clothes", "x": float(x), "y": float(y)}
        for i, (x, y) in zip(sample_ids, clothes_xy)
    ]
    if art_index is not None:
        art_ids = _draw(art_index.ids, n_art)
        art_xy = basis2d.project(art_index.centered(art_ids))
        points += [
            {"id": i, "domain": "art", "x": float(x), "y": float(y)}
            for i, (x, y) in zip(art_ids, art_xy)
        ]
    reach = float(np.hypot(*clothes_xy.T).max()) if len(clothes_xy) else 1.0
    allowed = set(sample_ids)
    markers = []
    for name, you in yous.items():
        if you.w is None:
            continue
        xy = basis2d.project(np.asarray(you.w))
        length = float(np.hypot(*xy))
        if length < _EPS:
            continue  # w is orthogonal to the plane: no direction to draw
        x, y = xy / length * reach
        markers.append(
            {
                "you": name,
                "x": float(x),
                "y": float(y),
                "top": [i for i, _ in index.top_k(you.w, top, allowed=allowed)],
            }
        )
    return {"points": points, "yous": markers}
