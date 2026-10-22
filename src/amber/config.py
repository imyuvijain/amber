"""Central constants for Amber. Everything tunable lives here."""

import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Where catalogs, embeddings and the SQLite database live (override with AMBER_DATA_DIR).
DATA_DIR: Path = Path(
    os.environ.get("AMBER_DATA_DIR") or REPO_ROOT / "data"
).expanduser()
DB_PATH: Path = DATA_DIR / "amber.db"

SEED = 7  # used wherever randomness affects results

# Vector space
PCA_DIM = 32
NEAR_DUP_COS = 0.95  # pairs at or above this cosine similarity are never shown together
MODEL_NAME: str = "ViT-B-32"  # chosen in Task 5; see docs/results/space.md
MODEL_PRETRAINED: str = "laion2b_s34b_b79k"

# Preference model
PRIOR_VAR = 1.0  # prior variance of each taste coordinate; tuned on the benchmark (docs/results/bench.md)

# Clothing catalog
KEEP_MASTER_CATEGORIES = ("Apparel", "Footwear", "Accessories")
DEFAULT_SECTIONS = ("Apparel", "Footwear")
THUMB_SIZE = (216, 288)  # (width, height) in pixels
NEUTRAL_PROMPT = "a photo of a piece of clothing"

# Text search
SEARCH_CANDIDATES = 200
SEARCH_RESULTS = 24
SEARCH_LAMBDA = 1.0


@dataclass(frozen=True)
class Settings:
    art_train_pairs: int
    clothes_train_pairs: int
    showdown_pairs: int
    retest_pairs: int
    retest_min_days: int


DEFAULT_SETTINGS = Settings(30, 40, 20, 10, 7)
