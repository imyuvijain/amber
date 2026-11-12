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
    live = {name: np.asarray(y.w, dtype=np.float64) for name, y in yous.items() if y.w is not None}
    unit = {name: w / np.linalg.norm(w) for name, w in live.items()}
    return {a: {b: float(unit[a] @ unit[b]) for b in unit} for a in unit}
