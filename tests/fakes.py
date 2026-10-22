"""Test fakes: a deterministic fake encoder and a tiny generated catalog builder.

Plain module (not a conftest) so tests can `from fakes import ...` without nested
conftest.py files shadowing it. The pytest fixtures live in conftest.py.

`FakeEncoder` implements the `Encoder` protocol (Task 4) without any model:

* image vector: dims 0-2 are the image's mean R, G, B / 255; the remaining dims are
  deterministic noise x 0.1; then L2-normalised. A red photo therefore points along dim 0.
  The noise is seeded from the domain-relative path (`clothes/thumbs/7.jpg`), not the
  absolute one, so vectors do not change with where the data dir lives (e.g. tmp_path).
* text vector: deterministic noise x 0.1 on every dim; if the lowercased text contains
  "red" / "green" / "blue", dim 0 / 1 / 2 is set to 1; then L2-normalised.

Noise is seeded from a hashlib digest (never builtin `hash()`, which is salted per
process), so vectors are identical across runs and processes.

`build_fixture_data` builds a small data dir (see `build_fixture_data`) that matches what the
real `amber ingest` commands write, so everything downstream can run against it fast.
"""

import hashlib
import itertools
import random
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from amber.config import SEED

_COLOR_DIMS = {"red": 0, "green": 1, "blue": 2}


def _stable_seed(key: str) -> int:
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big")


def _noise(key: str, n: int) -> np.ndarray:
    return np.random.default_rng(_stable_seed(key)).standard_normal(n) * 0.1


def _unit(vec: np.ndarray) -> np.ndarray:
    return (vec / np.linalg.norm(vec)).astype(np.float32)


class FakeEncoder:
    """Model-free stand-in for `ClipEncoder`; see the module docstring."""

    model_name = "fake"
    pretrained = "fixture"

    def __init__(self, dim: int = 16):
        if dim < 4:
            raise ValueError("FakeEncoder needs dim >= 4 (3 colour dims plus noise)")
        self.dim = dim

    def embed_images(self, paths: Sequence[Path]) -> np.ndarray:
        out = np.zeros((len(paths), self.dim), dtype=np.float32)
        for i, path in enumerate(paths):
            with Image.open(path) as im:
                pixels = np.asarray(im.convert("RGB"), dtype=np.float32)
            vec = np.empty(self.dim, dtype=np.float64)
            vec[:3] = pixels.reshape(-1, 3).mean(axis=0) / 255.0
            vec[3:] = _noise("/".join(Path(path).parts[-3:]), self.dim - 3)
            out[i] = _unit(vec)
        return out

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            vec = _noise(text, self.dim)
            lowered = text.lower()
            for word, dim in _COLOR_DIMS.items():
                if word in lowered:
                    vec[dim] = 1.0
            out[i] = _unit(vec)
        return out


# --- generated fixture catalog -------------------------------------------------------

CLOTHES_PER_GROUP = 24
ART_COUNT = 24
_CLOTHES_THUMB_SIZE = (48, 64)
_ART_THUMB_SIZE = (64, 48)

# Names for the seeded-random colours. "Red" and "Blue" are deliberately absent: only the
# explicit clear reds and blues carry those names.
_NAMED_COLOURS = {
    "Black": (0, 0, 0),
    "White": (255, 255, 255),
    "Grey": (128, 128, 128),
    "Green": (0, 160, 0),
    "Yellow": (240, 220, 30),
    "Orange": (240, 140, 20),
    "Pink": (240, 130, 180),
    "Purple": (130, 40, 160),
    "Brown": (120, 70, 30),
    "Teal": (0, 128, 128),
    "Navy Blue": (0, 0, 128),
    "Maroon": (128, 0, 0),
}

# (article_type, sub_category) per gender x section, cycled through the 24 items of a group.
_ARTICLES = {
    ("Men", "Apparel"): [
        ("Tshirts", "Topwear"),
        ("Shirts", "Topwear"),
        ("Jeans", "Bottomwear"),
        ("Shorts", "Bottomwear"),
        ("Jackets", "Topwear"),
        ("Trousers", "Bottomwear"),
    ],
    ("Women", "Apparel"): [
        ("Tops", "Topwear"),
        ("Kurtas", "Topwear"),
        ("Dresses", "Dress"),
        ("Jeans", "Bottomwear"),
        ("Skirts", "Bottomwear"),
        ("Leggings", "Bottomwear"),
    ],
    ("Men", "Footwear"): [
        ("Casual Shoes", "Shoes"),
        ("Sports Shoes", "Shoes"),
        ("Sandals", "Sandal"),
        ("Flip Flops", "Flip Flops"),
        ("Formal Shoes", "Shoes"),
    ],
    ("Women", "Footwear"): [
        ("Heels", "Shoes"),
        ("Flats", "Shoes"),
        ("Sandals", "Sandal"),
        ("Casual Shoes", "Shoes"),
        ("Sports Shoes", "Shoes"),
    ],
}
_BRANDS = ("Nova", "Atlas", "Juno", "Kestrel")
_SEASONS = ("Summer", "Fall", "Winter", "Spring")
_USAGES = ("Casual", "Sports", "Formal")
_ART_TYPES = ("Painting", "Photograph", "Print", "Drawing and Watercolor")
_ARTISTS = ("Anna Veldt", "Marek Sorel", "Ines Calder", "Tomas Wren", "Lia Hartmann")


def _colour_name(rgb: tuple[int, int, int]) -> str:
    def dist(named: tuple[int, int, int]) -> int:
        return sum((a - b) ** 2 for a, b in zip(rgb, named))

    return min(_NAMED_COLOURS, key=lambda name: dist(_NAMED_COLOURS[name]))


class _ColourBank:
    """Hands out distinct colours: seeded-random, plus clear reds and blues on request."""

    def __init__(self) -> None:
        self._rng = random.Random(SEED)
        self._used: set[tuple[int, int, int]] = set()
        self._specials = 0

    def _claim(self, rgb: tuple[int, int, int]) -> tuple[int, int, int]:
        self._used.add(rgb)
        return rgb

    def red(self) -> tuple[str, tuple[int, int, int]]:
        self._specials += 1  # nudge the green channel so every red stays distinct
        return "Red", self._claim((220, 20 + self._specials, 20))

    def blue(self) -> tuple[str, tuple[int, int, int]]:
        self._specials += 1
        return "Blue", self._claim((20, 20 + self._specials, 220))

    def random(self) -> tuple[str, tuple[int, int, int]]:
        while True:
            rgb = tuple(self._rng.randrange(256) for _ in range(3))
            if rgb not in self._used:
                return _colour_name(rgb), self._claim(rgb)  # type: ignore[arg-type]


def _colours(bank: _ColourBank, count: int) -> list[tuple[str, tuple[int, int, int]]]:
    """A clear red, a clear blue, then seeded-random colours."""
    return [bank.red(), bank.blue()] + [bank.random() for _ in range(count - 2)]


def _save_solid(path: Path, rgb: tuple[int, int, int], size: tuple[int, int]) -> None:
    Image.new("RGB", size, rgb).save(path, "JPEG", quality=95)


def build_fixture_data(root: Path) -> Path:
    """Write a deterministic mini catalog under `root` and return `root`.

    * `clothes/`: 96 items `c:1`..`c:96` in four contiguous blocks of 24 (Men Apparel,
      Men Footwear, Women Apparel, Women Footwear); each block starts with a clear red
      and a clear blue. 48x64 solid-colour JPEGs at `clothes/thumbs/<n>.jpg`, plus
      `clothes/items.parquet` with the Task 2 columns.
    * `art/`: 24 items `a:1`..`a:24` (a:1 red, a:2 blue), 64x48 JPEGs at
      `art/thumbs/<n>.jpg`, plus `art/items.parquet` with the Task 3 columns.
    """
    bank = _ColourBank()
    (root / "clothes" / "thumbs").mkdir(parents=True, exist_ok=True)
    (root / "art" / "thumbs").mkdir(parents=True, exist_ok=True)

    rows = []
    n = 0
    for gender, section in itertools.product(("Men", "Women"), ("Apparel", "Footwear")):
        articles = _ARTICLES[(gender, section)]
        for i, (colour, rgb) in enumerate(_colours(bank, CLOTHES_PER_GROUP)):
            n += 1
            article_type, sub_category = articles[i % len(articles)]
            _save_solid(
                root / "clothes" / "thumbs" / f"{n}.jpg", rgb, _CLOTHES_THUMB_SIZE
            )
            rows.append(
                {
                    "id": f"c:{n}",
                    "title": f"{_BRANDS[n % len(_BRANDS)]} {gender} {colour} {article_type}",
                    "gender": gender,
                    "master_category": section,
                    "sub_category": sub_category,
                    "article_type": article_type,
                    "base_colour": colour,
                    "season": _SEASONS[n % len(_SEASONS)],
                    "usage": _USAGES[n % len(_USAGES)],
                    "year": 2011 + n % 7,
                }
            )
    pd.DataFrame(rows).to_parquet(root / "clothes" / "items.parquet", index=False)

    rows = []
    for n, (colour, rgb) in enumerate(_colours(bank, ART_COUNT), start=1):
        _save_solid(root / "art" / "thumbs" / f"{n}.jpg", rgb, _ART_THUMB_SIZE)
        rows.append(
            {
                "id": f"a:{n}",
                "title": f"Study in {colour}, No. {n}",
                "artist": _ARTISTS[n % len(_ARTISTS)],
                "date": str(1840 + 7 * n),
                "artwork_type": _ART_TYPES[n % len(_ART_TYPES)],
            }
        )
    pd.DataFrame(rows).to_parquet(root / "art" / "items.parquet", index=False)
    return root


def red_and_blue_centered(ctx) -> tuple[np.ndarray, np.ndarray]:
    """Centered vectors `(red, blue)` of one clear-Red and one clear-Blue clothes item.

    Picked by `base_colour` from `ctx.items["clothes"]` (the first of each), then looked up
    in `ctx.index["clothes"]`. `ctx` is an `AppContext`; it is not imported here so this
    module stays free of app imports beyond `amber.config`.
    """
    items = ctx.items["clothes"]
    red_id = items.loc[items["base_colour"] == "Red", "id"].iloc[0]
    blue_id = items.loc[items["base_colour"] == "Blue", "id"].iloc[0]
    red, blue = ctx.index["clothes"].centered([red_id, blue_id])
    return red, blue
