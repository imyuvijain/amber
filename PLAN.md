# Amber Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local web app that answers one question: **can your taste in art predict what you'd wear?** Four versions of you live in one vector space. Three of them never see a clothing choice. Every clothing pick you make is a test they all predict in advance. About 4 minutes, tap-only, no typing.

**Architecture:** A pretrained image-and-text model (CLIP ViT-B-32 via `open_clip`) embeds ~41k clothing photos, ~2k public-domain artworks and short text into one shared space. Each "You" is a unit direction in that space and predicts a this-or-that pair by which item points further along it. An online Bayesian probit model learns from choices. FastAPI + SQLite serve a Vite + React frontend. No LLM anywhere.

**Tech Stack:**
- **Python 3.12 via `uv`:** PyTorch (MPS), `open_clip_torch`, Hugging Face `datasets` (streaming), numpy, pandas, scipy, FastAPI, SQLite (stdlib), Typer, httpx, pytest, ruff
- **Web:** Vite + React + TypeScript

**Spec:** [NOTES.md](NOTES.md) (decisions, the research behind them, references)

---

## Redesign of 2026-10-22 (read this first)

Tasks 1–13 were built against the first design. The rest of the plan was rewritten after a research pass (sources in NOTES.md). What changed:

| Before | Now | Why |
|---|---|---|
| Five yous | Four: `stated`, `bigfive`, `art`, `revealed` | Values needed written answers and an LLM; cut for zero friction and a sharper story |
| Free-text "describe your style" | Tap 3–5 style words from a fixed list | No typing, anywhere |
| IPIP-50 (50 items) | Mini-IPIP (20 items, Donnellan et al. 2006, public domain) | α ≈ .73–.82 at 40% of the length; Big Five barely predicts clothes, so precision beyond that buys nothing |
| Hand-written Big Five bridge adjectives | Saucier (1994) Mini-Markers adjectives per trait pole | Published, validated trait adjectives instead of invented ones |
| 30 art pairs | 20 art pairs | Session budget |
| 40 active training pairs, then 20 random showdown pairs | **40 random clothing pairs, every one scored** | The non-clothing yous are frozen before any clothes are shown, so every clothing pick is a fair test of them. Revealed is scored *prequentially*: before each pair it predicts from the picks so far, then learns. Twice the scoring data in less time, plus a learning curve. Active selection only bought ≈ +0.02 (docs/results/bench.md) |
| `accuracy / ceiling` | `(accuracy − 0.5) / (ceiling − 0.5)` | Chance-corrected: a coin flip is 0%, your own retest consistency is 100% |
| Retest of 10 pairs | Retest of 20 of the 40 pairs | 10 pairs is too noisy for a ceiling |

**Session (≈ 4 min):** sections (5 s) → style words (15 s) → Mini-IPIP (≈ 60 s) → art × 20 (≈ 50 s) → clothes × 40 (≈ 100 s) → results.

## Roadmap at a glance

| Phase | Tasks | When it's done, you can… |
|---|---|---|
| 1. The vector space | 1–5 ✅ | type "red floral dress" in the terminal and get the right clothes back |
| 2. Revealed you | 6–8 ✅ | watch a simulated person get learned, and see active vs. random pairs |
| 3. First playable | 9–12 ✅ | play this-or-that in the browser |
| 4. The redesign | 13 ✅, 14–16 | play the 4-minute tap-only flow with all four yous scored on every clothing pick |
| 5. Results | 17–19 | see the learning curve, the map of your yous, the readout, and a retest a week later |
| 6. Search | 20 | search "black jacket" ranked by your art taste |
| 7. Ship | 21 | have a README with your real results and a demo GIF |

## The four yous

| You | Built from | Sees clothing choices? | Role in the story |
|---|---|---|---|
| `stated` | 3–5 style words you tap | no | what you *say* you like |
| `bigfive` | Mini-IPIP (20 taps), bridged to clothes through Mini-Markers adjectives | no | what personality alone predicts |
| `art` | 20 this-or-that picks on paintings and photographs, carried over to clothes | no | **the headline** |
| `revealed` | your clothing picks so far (prequential) | yes, only picks *before* the one it predicts | the in-domain reference |

## Global Constraints

- **Disk:** everything under `data/` stays ≤ 1 GB. No dataset is downloaded whole.
- **Runtime:** Python 3.12 pinned with `uv`. Runs locally on Apple Silicon with MPS and falls back to CPU.
- **No typing for the player:** every input is a tap. No free-text fields anywhere in the web flow.
- **No LLM, no paid API, no secrets.**
- **Leak rule:** the `stated`, `bigfive` and `art` builders never receive clothing choices. Each raises `ValueError` if handed one.
- **Scoring rules:**
  - Clothing pairs are drawn at random (never chosen by a model), from the person's allowed sections, never repeating an item.
  - `stated`, `bigfive` and `art` are frozen right before the first clothing pair is created.
  - Before each clothing pair is returned, every You's `P(a)` is saved. Revealed's comes from a model replayed on that person's *earlier* clothing picks only (it abstains, `P(a) = 0.5`, under 5 picks).
  - An answer is rejected if any prediction is missing (already enforced by `MissingPredictions`).
  - Retest choices never train anything.
- **Frozen inputs:** the style-word list, the Mini-IPIP items, the Mini-Markers bridge file, the banned-word list and the readout concept list are versioned files, frozen with git tag `prereg-v1` before the first real run.
- **Seeds:** `SEED = 7` wherever randomness affects results.
- **Fast tests:** the default `pytest` run uses `FakeEncoder` and a generated fixture catalog. Real-model tests are `@pytest.mark.slow` and excluded by default.
- **AIC etiquette:** every AIC request sends `AIC-User-Agent: amber (https://github.com/imyuvijain/amber)` and stays at or under 1 request per second.
- **Small commits (the user wants many):** split every task's "Commit" step into several commits, in this order:
  - `build:` or `chore:` for dependency and tooling changes
  - `test(<area>):` for the failing tests
  - `feat(<area>):` / `refactor(<area>):` for each function or module that turns them green, one per function or module
  - `feat(cli):` or `feat(api):` for the wiring
  - `docs(results):` for measured results
  - one `feat(web):` commit per screen or component

  Messages describe what changed, and each one ends with the `Co-Authored-By` line. Run `git push origin main` after every finished task.
- **Item ids:** clothes are `"c:<n>"` and art is `"a:<n>"`. Thumbnails live at `data/<domain>/thumbs/<n>.jpg`.
- **Licenses:** AIC public-domain images are CC0. IPIP items are public domain. The clothing dataset's license is unverified (Kaggle *Fashion Product Images*, Param Aggarwal); check it before any public hosting.

## Review Focus

1. **Random or inconsistent clicking:** the model never produces NaN or crashes, and a random clicker sits near 50%. *(Tasks 6–7.)*
2. **Near-identical pairs:** two near-identical photos are never shown together (cosine > 0.95). *(Task 7.)*
3. **Closing the tab mid-flow or double-clicking:** the open pair resumes, a repeated identical answer is a no-op, a conflicting second answer is rejected. *(Task 8; Task 12 disables buttons in flight.)*
4. **Broken or missing images:** skipped at ingest, never served. *(Tasks 2, 11.)*
5. **Items outside your chosen sections:** never shown, in pairs or in search. *(Tasks 11, 20.)*
6. **No peeking:** a prediction for clothing pair *k* never depends on pick *k* or later. *(Task 14: `test_revealed_prediction_uses_only_earlier_picks`.)*

---

## Architecture

### Vector space

- **Encoder:** `("ViT-B-32", "laion2b_s34b_b79k")`, chosen in Task 5 (docs/results/space.md). All vectors L2-normalized `float32`.
- **Centered item vectors:** `centered(x) = normalize(x − domain_mean)`. Centering removes "clothing-ness" and "art-ness", so a taste direction learned on art can point somewhere meaningful among clothes.
- **Learning basis:** each domain gets a PCA basis (`PCA_DIM = 32`) fit on its centered vectors. Preference models learn in 32 dimensions.
- **Every You is a unit vector `w`** in full-dimensional space. Its score for an item is `w · centered(item)`; it predicts a pair by which score is higher.
  - Learned yous lift their 32-dimensional weights back with `basis.lift`.
  - Text yous use `text_direction(texts) = normalize(mean(embed_texts(texts)) − embed_texts([NEUTRAL_PROMPT]))`.

### Flow in the app

Welcome → sections → style words → Mini-IPIP (skippable) → art this-or-that (20, active) → clothes this-or-that (40, random, all scored) → results → search. The retest of 20 of the clothing pairs unlocks 7 days later.

Personality and art inputs come *before* any clothing is shown, so clothes can't color them.

### Package layout

```
src/amber/
  config.py            paths, constants, Settings
  cli.py               `amber` command (Typer)
  context.py           AppContext: everything the app needs loaded once
  flow.py              next_pair(): train / showdown / retest pair logic
  data/clothes.py      stream + filter + thumbnail the clothing catalog
  data/art.py          fetch public-domain artworks from AIC
  data/catalog.py      load_items, thumb_path, domain_of
  space/encoder.py     Encoder protocol, ClipEncoder, pick_device
  space/index.py       VectorIndex, embed_domain
  space/basis.py       Basis (PCA)
  space/evaluate.py    label_retrieval_score
  prefs/model.py       PreferenceModel (online Bayesian probit)
  prefs/active.py      choose_pair, random_pair
  prefs/simulate.py    SimulatedUser, run_benchmark
  yous/base.py         You, YouName, text_direction, predict
  yous/registry.py     BUILDERS, build_all_yous
  yous/stated.py  yous/revealed.py  yous/art_taste.py  yous/bigfive.py
  yous/files/          style_words.yaml, mini_ipip.yaml, bigfive_bridge.yaml, banned_words.txt
  eval/metrics.py      Score, score_showdown, self_consistency, normalized_accuracy
  eval/retest.py       retest_due, next_retest_pair
  explain/readout.py   concepts.yaml, concept_alignment, label_lift, build_readout
  explain/geometry.py  you_cosines, map_points
  search/rerank.py     search
  store/db.py          SQLite schema + queries
  api/app.py           create_app (FastAPI)
web/                   Vite + React + TS frontend
tests/                 mirrors src/amber/
docs/results/          measured results that get committed (space.md, bench.md)
```

---

## Phase 1: The vector space

### Task 1: Project skeleton and tooling

**Files:**
- Create: `pyproject.toml`, `.python-version`, `.gitignore`, `src/amber/__init__.py`, `src/amber/config.py`, `src/amber/cli.py`, `tests/conftest.py`, `tests/test_config.py`, `tests/test_fake_encoder.py`

**Interfaces:**
- Produces, from `amber.config`:
  - `DATA_DIR: Path`, read from env `AMBER_DATA_DIR` (default `<repo>/data`)
  - `DB_PATH = DATA_DIR / "amber.db"`
  - `SEED = 7`
  - `PCA_DIM = 32`
  - `NEAR_DUP_COS = 0.95`
  - `KEEP_MASTER_CATEGORIES = ("Apparel", "Footwear", "Accessories")`
  - `DEFAULT_SECTIONS = ("Apparel", "Footwear")`
  - `THUMB_SIZE = (288, 384)`
  - `NEUTRAL_PROMPT = "a photo of a piece of clothing"`
  - `SEARCH_CANDIDATES = 200`, `SEARCH_RESULTS = 24`, `SEARCH_LAMBDA = 1.0`
  - `MODEL_NAME: str`, `MODEL_PRETRAINED: str` (placeholders until Task 5 sets them)
  - `@dataclass(frozen=True) class Settings: art_train_pairs: int; clothes_train_pairs: int; showdown_pairs: int; retest_pairs: int; retest_min_days: int`
  - `DEFAULT_SETTINGS = Settings(30, 40, 20, 10, 7)`
- Produces, from `tests/conftest.py`:
  - `FakeEncoder(dim=16)`, which implements the `Encoder` protocol from Task 4:
    - image vector: dims 0–2 are the image's mean RGB / 255, the remaining dims are hash noise × 0.1, then normalized
    - text vector: hash noise, unless the text contains a color word (`red`, `green`, `blue`), which sets the matching dim to 1
  - fixture `fixture_data(tmp_path) -> Path`: returns a data dir holding:
    - `clothes/`: 96 solid-color JPEGs plus `items.parquet` (same columns as Task 2), 24 for each combination of gender (`Men`, `Women`) × section (`Apparel`, `Footwear`)
    - `art/`: 24 solid-color JPEGs plus `items.parquet` (same columns as Task 3)

- [x] **Step 1: Install uv and create the project**

Run: `brew install uv && uv python install 3.12`
Expected: `uv --version` prints a version.

Write `pyproject.toml`:
- `name = "amber"`, `requires-python = ">=3.12,<3.13"`, src layout
- `[project.scripts] amber = "amber.cli:app"`
- dependencies: `torch`, `open_clip_torch`, `datasets`, `pillow`, `numpy`, `pandas`, `pyarrow`, `fastapi`, `uvicorn[standard]`, `httpx`, `typer`, `pyyaml`
- optional extra `llm = ["anthropic"]`
- dev group: `pytest`, `ruff`
- pytest config: `markers = ["slow: needs real model or network"]`, `addopts = "-m 'not slow'"`

`.gitignore` covers: `data/`, `.venv/`, `node_modules/`, `web/dist/`, `.env`, `.claude/settings.local.json`, `__pycache__/`.

- [x] **Step 2: Write the failing tests**

```python
# tests/test_config.py
import importlib
from amber import config

def test_constants():
    assert config.SEED == 7 and config.PCA_DIM == 32 and config.NEAR_DUP_COS == 0.95
    assert config.KEEP_MASTER_CATEGORIES == ("Apparel", "Footwear", "Accessories")
    assert config.THUMB_SIZE == (288, 384)
    assert config.DEFAULT_SETTINGS == config.Settings(30, 40, 20, 10, 7)

def test_data_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("AMBER_DATA_DIR", str(tmp_path))
    importlib.reload(config)
    assert config.DATA_DIR == tmp_path and config.DB_PATH == tmp_path / "amber.db"
```

```python
# tests/test_fake_encoder.py
def test_fake_encoder_is_deterministic_and_unit(fake_encoder):
    a, b = fake_encoder.embed_texts(["hello", "hello"])
    assert np.allclose(a, b) and np.isclose(np.linalg.norm(a), 1.0)

def test_fake_encoder_colors(fake_encoder, fixture_data):
    # a red image is closer to "red" text than a blue image is
    ...  # pick a red and a blue thumbnail from fixture_data/"clothes"; assert dot(red_img, t_red) > dot(blue_img, t_red)
```

- [x] **Step 3: Run the tests and confirm they fail**

Run: `uv run pytest -q`
Expected: FAIL (`amber.config` missing).

- [x] **Step 4: Implement `config.py`, `cli.py` (an empty Typer `app`), and the `FakeEncoder` and `fixture_data` fixtures in `tests/conftest.py`**

- [x] **Step 5: Run the tests and lint**

Run: `uv run pytest -q && uv run ruff check .`
Expected: PASS, no lint errors.

- [x] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore src tests uv.lock
git commit -m "chore: project skeleton, config, fake encoder fixtures"
```

### Task 2: Stream the clothing catalog

**Files:**
- Create: `src/amber/data/__init__.py`, `src/amber/data/clothes.py`, `src/amber/data/catalog.py`, `tests/data/test_clothes.py`
- Modify: `src/amber/cli.py` (add `amber ingest clothes [--limit N]`)

**Interfaces:**
- Produces:
  - `CLOTHES_DATASET = "benitomartin/fashion-product-images-small-384x512"`
  - `ingest_clothes(out_dir: Path, rows: Iterable[dict] | None = None, limit: int | None = None) -> pd.DataFrame`
    - `rows=None` streams the dataset via `datasets.load_dataset(CLOTHES_DATASET, split="train", streaming=True)`
    - writes `out_dir/items.parquet` and `out_dir/thumbs/<n>.jpg` (resized to `THUMB_SIZE`, JPEG quality 85)
  - `items.parquet` columns: `id` (`"c:<n>"`), `title`, `gender`, `master_category`, `sub_category`, `article_type`, `base_colour`, `season`, `usage`, `year`. Missing strings become `""`.
  - `load_items(domain: Literal["clothes", "art"], data_dir: Path = DATA_DIR) -> pd.DataFrame`
  - `thumb_path(item_id: str, data_dir: Path = DATA_DIR) -> Path`
  - `domain_of(item_id: str) -> Literal["clothes", "art"]`

- [x] **Step 1: Write the failing tests**

Build `fake_rows` in the test: 5 dicts shaped like the Hugging Face rows, using the field names `id, gender, masterCategory, subCategory, articleType, baseColour, season, year, usage, productDisplayName, image`. Three have kept categories, one is `"Personal Care"`, and one has an image whose `.convert()` raises.

```python
def test_ingest_filters_and_writes(tmp_path, fake_rows):
    df = ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert len(df) == 3 and df.id.str.startswith("c:").all()
    assert set(df.master_category) <= {"Apparel", "Footwear", "Accessories"}
    for item_id in df.id:
        with Image.open(thumb_path(item_id, data_dir=tmp_path)) as im:
            assert im.size == (288, 384)

def test_ingest_skips_corrupt_image(tmp_path, fake_rows):
    df = ingest_clothes(tmp_path / "clothes", rows=fake_rows)
    assert "c:5" not in set(df.id)   # the corrupt row's id

def test_ingest_resumes(tmp_path, fake_rows):
    out = tmp_path / "clothes"
    ingest_clothes(out, rows=fake_rows)
    mtimes = {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()}
    df2 = ingest_clothes(out, rows=fake_rows)
    assert {p: p.stat().st_mtime for p in (out / "thumbs").iterdir()} == mtimes and len(df2) == 3

def test_domain_of_and_thumb_path(tmp_path):
    assert domain_of("c:12") == "clothes" and domain_of("a:9") == "art"
    assert thumb_path("a:9", data_dir=tmp_path) == tmp_path / "art" / "thumbs" / "9.jpg"
    with pytest.raises(ValueError):
        domain_of("x:1")
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/data/test_clothes.py -q`
Expected: FAIL (module missing).

- [x] **Step 3: Implement `clothes.py` and `catalog.py`**

`ingest_clothes` skips rows whose thumbnail already exists (resume). It catches image decode errors per row, logs them, and moves on. It writes the parquet once at the end, merging with the existing parquet if one is present.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/data/test_clothes.py -q`
Expected: PASS.

- [x] **Step 5: Run the real ingest (about 10–30 min, in the background)**

Run: `uv run amber ingest clothes`
Expected:
- the row count prints between 38,000 and 44,072
- `du -sh data/clothes` is under 700M
- `du -sh ~/.cache/huggingface` shows no multi-GB cache (streaming shouldn't write one; if it did, find out why before going on)

Streaming moves about 2 GB over the network once, but only the thumbnails are kept.

- [x] **Step 6: Commit**

```bash
git add src/amber/data src/amber/cli.py tests/data
git commit -m "feat: stream clothing catalog into thumbnails + parquet"
```

### Task 3: Fetch the art catalog

**Files:**
- Create: `src/amber/data/art.py`, `tests/data/test_art.py`
- Modify: `src/amber/cli.py` (add `amber ingest art [--n 2000]`)

**Interfaces:**
- Produces:
  - `KEEP_ART_TYPES = ("Painting", "Photograph", "Print", "Drawing and Watercolor")`
  - `AIC_HEADERS = {"AIC-User-Agent": "amber (https://github.com/imyuvijain/amber)"}`
  - `ingest_art(out_dir: Path, n: int = 2000, client: httpx.Client | None = None, min_interval: float = 1.0) -> pd.DataFrame`
- Writes `out_dir/items.parquet` with columns `id` (`"a:<aic_id>"`), `title`, `artist`, `date`, `artwork_type`, plus `out_dir/thumbs/<aic_id>.jpg` (longest side 384, aspect kept).
- Source details:
  - search: `POST https://api.artic.edu/api/v1/artworks/search`, filtered on `is_public_domain = true`, `artwork_type_title ∈ KEEP_ART_TYPES`, and `image_id` exists
  - fields: `id,title,artist_title,date_display,artwork_type_title,image_id`, 100 per page
  - images: `https://www.artic.edu/iiif/2/<image_id>/full/843,/0/default.jpg`

- [x] **Step 1: Write the failing tests** using `httpx.MockTransport` (search returns 3 records, one image request returns 404)

```python
def test_ingest_art_filters_and_prefixes(tmp_path, mock_client):
    df = ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert df.id.tolist() == ["a:1", "a:2"]          # a:3's image 404 → skipped
    assert set(df.artwork_type) <= set(KEEP_ART_TYPES)
    assert max(Image.open(tmp_path / "thumbs" / "1.jpg").size) == 384

def test_ingest_art_sends_user_agent(tmp_path, mock_client, seen_requests):
    ingest_art(tmp_path, n=3, client=mock_client, min_interval=0)
    assert all(r.headers["AIC-User-Agent"].startswith("amber") for r in seen_requests)
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/data/test_art.py -q`
Expected: FAIL.

- [x] **Step 3: Implement `ingest_art`.** Throttle with `min_interval` seconds between all requests. Make it resumable the same way as Task 2.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/data/test_art.py -q`
Expected: PASS.

- [x] **Step 5: Run the real fetch (about 35–70 min at 1 request per second, in the background)**

Run: `uv run amber ingest art --n 2000`
Expected: 1,900–2,000 rows, and `du -sh data/art` under 100M.

- [x] **Step 6: Commit**

```bash
git add src/amber/data/art.py src/amber/cli.py tests/data/test_art.py
git commit -m "feat: fetch public-domain art catalog from AIC"
```

### Task 4: Encoder, vector index, PCA basis

**Files:**
- Create: `src/amber/space/__init__.py`, `encoder.py`, `index.py`, `basis.py`, `tests/space/test_index.py`, `tests/space/test_basis.py`, `tests/space/test_encoder.py`

**Interfaces:**
- Produces, in `amber.space.encoder`:
  - `class Encoder(Protocol)` with `dim: int`, `embed_images(paths: Sequence[Path]) -> np.ndarray`, and `embed_texts(texts: Sequence[str]) -> np.ndarray`. Both return `(n, dim)` `float32` arrays with L2-normalized rows.
  - `class ClipEncoder`: `__init__(self, model_name: str, pretrained: str, device: str | None = None, batch_size: int = 64)`
  - `pick_device() -> str`, returning `"mps"`, then `"cuda"`, then `"cpu"`
- Produces, in `amber.space.index`:
  - `class VectorIndex`: `__init__(self, ids: list[str], vectors: np.ndarray)`
  - `save(self, dir: Path) -> None` writes `vectors.npy` + `ids.json`, and `@classmethod load(cls, dir: Path) -> VectorIndex`
  - `vec(self, item_id: str) -> np.ndarray`
  - `mean: np.ndarray` (mean of raw vectors)
  - `centered(self, item_ids: Sequence[str]) -> np.ndarray`: rows are `normalize(v − mean)`
  - `top_k(self, direction: np.ndarray, k: int, allowed: set[str] | None = None, use_centered: bool = True) -> list[tuple[str, float]]`, sorted descending
- Produces, in `amber.space.basis`:
  - `class Basis`: `@classmethod fit(cls, X: np.ndarray, k: int) -> Basis` (PCA via SVD on mean-removed `X`)
  - `components: np.ndarray` of shape `(k, d)`, orthonormal rows
  - `project(self, X: np.ndarray) -> np.ndarray` of shape `(n, k)`, and `lift(self, w: np.ndarray) -> np.ndarray` of shape `(d,)`
  - `save(dir)` and `load(dir)`

- [x] **Step 1: Write the failing tests**

```python
def test_index_roundtrip(tmp_path):
    idx = VectorIndex(["c:1", "c:2"], unit_rows(2, 8)); idx.save(tmp_path)
    assert VectorIndex.load(tmp_path).ids == ["c:1", "c:2"]

def test_centered_rows_are_unit_and_mean_removed():
    idx = VectorIndex(ids, X); C = idx.centered(ids)
    assert np.allclose(np.linalg.norm(C, axis=1), 1) and np.allclose((X - X.mean(0)).mean(0), 0, atol=1e-6)

def test_top_k_respects_allowed_and_sorts():
    res = idx.top_k(direction, k=3, allowed={"c:1", "c:3"})
    assert {i for i, _ in res} <= {"c:1", "c:3"} and [s for _, s in res] == sorted([s for _, s in res], reverse=True)

def test_basis_orthonormal_and_lift_inverts_project():
    B = Basis.fit(rand(200, 16), k=4)
    assert np.allclose(B.components @ B.components.T, np.eye(4), atol=1e-5)
    w = rand(4); assert np.allclose(B.project(B.lift(w)[None])[0], w, atol=1e-5)

def test_pick_device_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert pick_device() == "cpu"

@pytest.mark.slow
def test_clip_encoder_color_sanity(tmp_path):
    enc = ClipEncoder("ViT-B-32", "laion2b_s34b_b79k")
    red, blue = enc.embed_images([solid(tmp_path, "red"), solid(tmp_path, "blue")])
    t = enc.embed_texts(["a red shirt"])[0]
    assert red.shape == (enc.dim,) and red @ t > blue @ t
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/space -q`
Expected: FAIL.

- [x] **Step 3: Implement the three modules.** `ClipEncoder` uses `open_clip.create_model_and_transforms` and `get_tokenizer`. Run it under `torch.inference_mode()` and normalize the outputs.

- [x] **Step 4: Run all tests, including the slow ones (the slow ones download a model once, about 600 MB)**

Run: `uv run pytest tests/space -q && uv run pytest tests/space -q -m slow`
Expected: PASS both.

- [x] **Step 5: Commit**

```bash
git add src/amber/space tests/space
git commit -m "feat: encoder, vector index, PCA basis"
```

### Task 5: Pick the model by measuring, then embed everything

**Files:**
- Create: `src/amber/space/evaluate.py`, `tests/space/test_evaluate.py`, `docs/results/space.md`
- Modify: `src/amber/space/index.py` (add `embed_domain`), `src/amber/config.py` (set `MODEL_NAME` and `MODEL_PRETRAINED`), `src/amber/cli.py` (add `embed`, `eval-space`, `search-text`)

**Interfaces:**
- Consumes: Task 2's `load_items` and `thumb_path`. Task 4's `Encoder`, `VectorIndex` and `Basis`.
- Produces:
  - `embed_domain(domain: str, encoder: Encoder, data_dir: Path = DATA_DIR) -> VectorIndex`
    - writes `data/<domain>/emb/<model_slug>/`, where `model_slug = f"{name}__{pretrained}"`
    - also fits and saves a `Basis(k=PCA_DIM)` on that domain's centered vectors, plus a 2-component `Basis` saved as `basis2d`
  - `label_retrieval_score(index: VectorIndex, items: pd.DataFrame, encoder: Encoder, k: int = 20, min_group: int = 20) -> float`
    - takes the mean precision@k over `(base_colour, article_type)` groups with ≥ `min_group` items
    - query text: `f"a photo of a {base_colour.lower()} {article_type.lower()}"`
    - ranks raw (uncentered) vectors
  - CLI:
    - `amber embed --model NAME --pretrained TAG [--domain clothes|art|all]`
    - `amber eval-space --model NAME --pretrained TAG`
    - `amber search-text "query"` writes `data/contact_sheet.jpg` showing the top 12 thumbnails

- [x] **Step 1: Write the failing tests**

```python
def test_label_retrieval_perfect_encoder():
    # index vectors == the query vector of their own group → every group's top-k is pure
    assert label_retrieval_score(idx, items, OracleEncoder(groups), k=5, min_group=5) == 1.0

def test_label_retrieval_ignores_small_groups():
    assert label_retrieval_score(idx, items_with_one_big_group, enc, k=5, min_group=5) == score_of_big_group_only

def test_embed_domain_writes_index_and_bases(fixture_data, fake_encoder):
    idx = embed_domain("clothes", fake_encoder, data_dir=fixture_data)
    assert len(idx.ids) == 96 and (fixture_data / "clothes" / "emb").exists()
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/space/test_evaluate.py -q`
Expected: FAIL.

- [x] **Step 3: Implement `label_retrieval_score`, `embed_domain`, and the CLI commands.**

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/space -q`
Expected: PASS.

- [x] **Step 5: Confirm the candidate model names exist**

Run: `uv run python -c "import open_clip; print([p for p in open_clip.list_pretrained() if 'SigLIP2' in p[0] or p[0]=='ViT-B-32'])"`
Expected: both candidates appear. If `ViT-B-16-SigLIP2` is missing, use the closest SigLIP listed and note that in `space.md`.

- [x] **Step 6: Embed the clothes with both candidates and score them**

Run: `uv run amber embed --model <cand> --pretrained <tag> --domain clothes && uv run amber eval-space --model <cand> --pretrained <tag>`, once per candidate.

Record in `docs/results/space.md`: model, P@20, embedding time, and vector dimension. The winner is the one with the higher P@20, with an art spot-check as the tie-breaker (`amber search-text "a stormy seascape"` against art).

- [x] **Step 7: Set the winner and embed art**

Set the winner in `config.py`, then run `uv run amber embed --domain art`. Check `uv run amber search-text "red floral dress"` and look at `data/contact_sheet.jpg`.
Expected: mostly red floral dresses.

- [x] **Step 8: Commit**

```bash
git add src/amber tests docs/results/space.md
git commit -m "feat: pick embedding model by label retrieval; embed clothes and art"
```

---

## Phase 2: Revealed you

### Task 6: Online Bayesian preference model

**Files:**
- Create: `src/amber/prefs/__init__.py`, `src/amber/prefs/model.py`, `tests/prefs/test_model.py`

**Interfaces:**
- Produces: `class PreferenceModel`:
  - `__init__(self, dim: int, prior_var: float = 1.0)`
  - `update(self, winner: np.ndarray, loser: np.ndarray) -> None`
  - `p_prefer(self, a: np.ndarray, b: np.ndarray) -> float`
  - `mean: np.ndarray` of shape `(dim,)`, `cov: np.ndarray` of shape `(dim, dim)`
  - `n_updates: int`
  - Inputs are 32-dimensional projected vectors from `Basis.project`.

**Algorithm:** probit Bradley–Terry with Gaussian assumed-density filtering. This isn't implied by the signature, so it's pinned here.
- Prior: `w ~ N(0, prior_var·I)`.
- For each update, set `x = winner − loser` and `s = Σx`, then compute:
  - `m = μᵀx`, `v = xᵀΣx`, `z = m / √(1+v)`
  - `r = φ(z) / Φ(z)` (use `scipy.special.log_ndtr`, or a stable ratio, for very negative `z`)
- Then update:
  - `μ ← μ + s · r / √(1+v)`
  - `Σ ← Σ − (s sᵀ) · (r / (1+v)) · (z + r)`
  - symmetrize `Σ` after each update
- `p_prefer(a, b) = Φ(μᵀx / √(1 + xᵀΣx))` with `x = a − b`.
- Add `scipy` to dependencies.

- [x] **Step 1: Write the failing tests**

```python
def test_prior_is_indifferent():
    m = PreferenceModel(4); a, b = rand(4), rand(4)
    assert m.p_prefer(a, b) == pytest.approx(0.5) and m.p_prefer(a, a) == pytest.approx(0.5)

def test_update_moves_toward_winner():
    m = PreferenceModel(4); a, b = e(0), e(1)
    m.update(a, b)
    assert m.p_prefer(a, b) > 0.5 and m.mean @ (a - b) > 0 and m.n_updates == 1

def test_symmetry():
    m = trained_model(); a, b = rand(4), rand(4)
    assert m.p_prefer(a, b) + m.p_prefer(b, a) == pytest.approx(1.0)

def test_variance_shrinks_along_compared_direction():
    m = PreferenceModel(4); x = e(0) - e(1); before = x @ m.cov @ x
    m.update(e(0), e(1)); assert x @ m.cov @ x < before

def test_no_nan_after_many_random_updates():
    m = PreferenceModel(32); rng = np.random.default_rng(7)
    for _ in range(500):
        a, b = rng.normal(size=32), rng.normal(size=32); m.update(*((a, b) if rng.random() < .5 else (b, a)))
    assert np.isfinite(m.mean).all() and np.isfinite(m.cov).all()
    assert np.all(np.linalg.eigvalsh(m.cov) > -1e-8)
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/prefs/test_model.py -q`
Expected: FAIL.

- [x] **Step 3: Implement `PreferenceModel`** per the algorithm above.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/prefs/test_model.py -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber/prefs tests/prefs pyproject.toml uv.lock
git commit -m "feat: online Bayesian probit preference model"
```

### Task 7: Active pair selection and the simulated-person benchmark

**Files:**
- Create: `src/amber/prefs/active.py`, `src/amber/prefs/simulate.py`, `tests/prefs/test_active.py`, `tests/prefs/test_simulate.py`, `docs/results/bench.md`
- Modify: `src/amber/cli.py` (add `amber bench-active`)

**Interfaces:**
- Consumes: `PreferenceModel` (Task 6), `NEAR_DUP_COS` (Task 1).
- Produces:
  - `choose_pair(model: PreferenceModel, X: np.ndarray, ids: list[str], dup_vectors: np.ndarray, rng: np.random.Generator, exclude: set[str] = frozenset(), n_candidates: int = 2000, near_dup_cos: float = NEAR_DUP_COS) -> tuple[str, str]`
    - `X` holds the projected rows for `ids`, and `dup_vectors` holds the full normalized embeddings in the same row order
    - it samples `n_candidates` random pairs from non-excluded ids, drops pairs with cosine > `near_dup_cos`, and returns the pair that maximizes `v · p(1−p)`, where `v = xᵀΣx` and `p = model.p_prefer`
  - `random_pair(ids: list[str], dup_vectors: np.ndarray, rng: np.random.Generator, exclude: set[str] = frozenset(), near_dup_cos: float = NEAR_DUP_COS) -> tuple[str, str]`
  - `class SimulatedUser`: `__init__(self, w: np.ndarray | None, temperature: float, rng: np.random.Generator)` and `choose(self, a: np.ndarray, b: np.ndarray) -> int`
    - `choose` returns `0` if it picks `a`, else `1`
    - it picks `a` with probability `sigmoid((w·a − w·b) / temperature)`
    - `w=None` makes it a coin-flip clicker
  - `run_benchmark(X: np.ndarray, dup_vectors: np.ndarray, n_users: int, n_train: int, n_test: int, strategy: Literal["active", "random"], temperature: float, seed: int, prior_var: float = PRIOR_VAR, random_clicker: bool = False) -> float`
    - returns mean held-out pairwise accuracy
    - each simulated user's `w` is the normalized projected vector of a random catalog item
    - `random_clicker=True` uses `w=None` (a coin flip) for every simulated user
  - `PRIOR_VAR` is added to `config.py` (starting at `1.0`, tuned in Step 5). It becomes the default `prior_var` of `PreferenceModel`.
  - CLI `amber bench-active` runs on the real clothes basis and prints a table for `strategy × temperature ∈ {0.05, 0.2} × prior_var ∈ {0.1, 1, 10}`.

- [x] **Step 1: Write the failing tests**

```python
def test_choose_pair_distinct_and_not_excluded():
    a, b = choose_pair(PreferenceModel(8), X, ids, D, rng, exclude={"c:1"})
    assert a != b and "c:1" not in (a, b)

def test_choose_pair_skips_near_duplicates():
    # ids c:1 and c:2 have identical dup_vectors and are the only "informative" pair
    assert set(choose_pair(model, X, ids, D_with_dupes, rng)) != {"c:1", "c:2"}

def test_random_pair_respects_exclude(): ...

def test_active_beats_random():
    act = run_benchmark(X, D, n_users=20, n_train=40, n_test=200, strategy="active", temperature=0.05, seed=7)
    rnd = run_benchmark(X, D, n_users=20, n_train=40, n_test=200, strategy="random", temperature=0.05, seed=7)
    assert act > rnd and act >= 0.70

def test_random_clicker_is_near_chance():
    acc = run_benchmark(X, D, n_users=20, n_train=40, n_test=200, strategy="active", temperature=0.05, seed=7, random_clicker=True)
    assert 0.40 <= acc <= 0.60
```

`X` here is 500 synthetic 32-dimensional points with cluster structure, generated with seed 7.

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/prefs -q`
Expected: FAIL.

- [x] **Step 3: Implement `active.py` and `simulate.py`, and add `PRIOR_VAR = 1.0` to `config.py`.**

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/prefs -q`
Expected: PASS. If `act >= 0.70` fails, the culprit is usually `prior_var` scale versus the size of the projected vectors. Tune `prior_var`; don't loosen the assertion.

- [x] **Step 5: Run the real benchmark and record it**

Run: `uv run amber bench-active`
Record the table in `docs/results/bench.md`, and set `config.PRIOR_VAR` to the best `prior_var`.

- [x] **Step 6: Commit**

```bash
git add src/amber/prefs src/amber/cli.py src/amber/config.py tests/prefs docs/results/bench.md
git commit -m "feat: active pair selection + simulated-person benchmark"
```

### Task 8: SQLite store

**Files:**
- Create: `src/amber/store/__init__.py`, `src/amber/store/db.py`, `tests/store/test_db.py`

**Interfaces:**
- Produces:
  - `connect(path: Path | str) -> sqlite3.Connection`: creates the schema if missing; `":memory:"` is allowed
  - `@dataclass(frozen=True) class Choice` with fields:
    - `pair_id: str`, `user_id: str`
    - `domain: Literal["clothes", "art"]`, `phase: Literal["train", "showdown", "retest"]`
    - `a_id: str`, `b_id: str`, `chosen_id: str | None`
    - `shown_at: datetime`, `answered_at: datetime | None`, `retest_of: str | None`
  - Exceptions: `AlreadyAnswered`, `MissingPredictions`, `NotFound`, all subclasses of `StoreError`
  - Users and answers:
    - `create_user(conn, name: str) -> str` (uuid4 hex)
    - `save_answer(conn, user_id: str, kind: Literal["describe", "filters", "ipip", "values"], key: str, value: str) -> None` (upsert)
    - `get_answers(conn, user_id: str, kind: str) -> dict[str, str]`
  - Pairs:
    - `create_pair(conn, user_id, domain, phase, a_id, b_id, retest_of: str | None = None, now: datetime | None = None) -> Choice`
    - `answer_pair(conn, pair_id: str, chosen_id: str, now: datetime | None = None) -> Choice`
    - `open_pair(conn, user_id, domain, phase) -> Choice | None`
    - `list_choices(conn, user_id, domain: str | None = None, phase: str | None = None, answered_only: bool = True) -> list[Choice]`, oldest first
    - `shown_ids(conn, user_id, domain) -> set[str]`
  - Predictions:
    - `save_predictions(conn, pair_id: str, preds: dict[str, float], now: datetime | None = None) -> None`
    - `get_predictions(conn, pair_id: str) -> dict[str, float]`
  - Frozen yous:
    - `save_you(conn, user_id: str, name: str, w: np.ndarray | None, meta: dict) -> None` (stores `w` as float32 bytes)
    - `load_yous(conn, user_id: str) -> dict[str, tuple[np.ndarray | None, dict]]`
- Rules enforced by `answer_pair`:
  - the same `chosen_id` twice returns the stored `Choice` unchanged
  - a different `chosen_id` after answering raises `AlreadyAnswered`
  - a `chosen_id` that's not in the pair raises `ValueError`
  - a showdown pair without predictions raises `MissingPredictions`

- [x] **Step 1: Write the failing tests**

```python
def test_answer_is_idempotent(conn):
    u = create_user(conn, "y"); p = create_pair(conn, u, "clothes", "train", "c:1", "c:2")
    first = answer_pair(conn, p.pair_id, "c:1"); again = answer_pair(conn, p.pair_id, "c:1")
    assert first == again and len(list_choices(conn, u)) == 1

def test_conflicting_answer_rejected(conn): ... pytest.raises(AlreadyAnswered)
def test_chosen_must_be_in_pair(conn): ... pytest.raises(ValueError)
def test_open_pair_resumes(conn):
    p = create_pair(conn, u, "art", "train", "a:1", "a:2"); assert open_pair(conn, u, "art", "train") == p

def test_showdown_requires_predictions(conn):
    p = create_pair(conn, u, "clothes", "showdown", "c:1", "c:2")
    with pytest.raises(MissingPredictions): answer_pair(conn, p.pair_id, "c:1")
    save_predictions(conn, p.pair_id, {"stated": 0.6}); assert answer_pair(conn, p.pair_id, "c:1").chosen_id == "c:1"

def test_yous_roundtrip(conn):
    save_you(conn, u, "revealed", np.ones(4, np.float32) / 2, {"n": 40}); save_you(conn, u, "values", None, {"reason": "no key"})
    w, meta = load_yous(conn, u)["revealed"]; assert np.allclose(w, 0.5) and meta == {"n": 40}
    assert load_yous(conn, u)["values"][0] is None
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/store -q`
Expected: FAIL.

- [x] **Step 3: Implement `db.py`** with the tables `users`, `answers(user_id, kind, key, value, PRIMARY KEY(user_id, kind, key))`, `pairs`, `predictions`, `yous`, and timestamps in UTC ISO format.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/store -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber/store tests/store
git commit -m "feat: SQLite store with idempotent answers and prediction gate"
```

---

## Phase 3: First playable

### Task 9: The `You` type, Stated you, and the app context

**Files:**
- Create: `src/amber/yous/__init__.py`, `yous/base.py`, `yous/stated.py`, `yous/registry.py`, `src/amber/context.py`, `tests/yous/test_base.py`, `tests/yous/test_stated.py`

**Interfaces:**
- Consumes: `Encoder`, `VectorIndex` and `Basis` (Task 4), `connect` and `get_answers` (Task 8).
- Produces, in `amber.yous.base`:
  - `YouName = Literal["stated", "bigfive", "art", "values", "revealed"]`
  - `YOU_ORDER: tuple[YouName, ...] = ("stated", "bigfive", "art", "values", "revealed")`
  - `@dataclass(frozen=True) class You: name: YouName; w: np.ndarray | None; meta: dict`, where `w` is a unit vector in full-dim space and `None` means abstain
  - `text_direction(encoder: Encoder, texts: Sequence[str], neutral: str = NEUTRAL_PROMPT) -> np.ndarray`
  - `predict(you: You, a_centered: np.ndarray, b_centered: np.ndarray) -> float`: returns `1.0` if `w·a > w·b`, `0.0` if lower, and `0.5` on a tie or abstain
- Produces, in `amber.context`:
  - `@dataclass class AppContext` with fields:
    - `conn: sqlite3.Connection`, `encoder: Encoder`
    - `items: dict[str, pd.DataFrame]`, `index: dict[str, VectorIndex]`, `basis: dict[str, Basis]`, `basis2d: dict[str, Basis]`
    - `settings: Settings`, `llm: Any | None`, `data_dir: Path`
  - `load_context(data_dir: Path = DATA_DIR, settings: Settings = DEFAULT_SETTINGS, encoder: Encoder | None = None) -> AppContext`
- Produces, in `amber.yous.stated`: `build_stated(conn, user_id: str, ctx: AppContext) -> You`
  - reads `get_answers(conn, user_id, "describe")["text"]`
  - splits the text into sentences, embeds each as `f"a photo of clothing: {sentence}"`, and runs `text_direction` over them
  - empty or missing text means abstain, with `meta={"reason": "no description"}`
- Produces, in `amber.yous.registry`:
  - `BUILDERS: dict[YouName, Callable[[sqlite3.Connection, str, AppContext], You]]` (`stated` registered here)
  - `build_all_yous(conn, user_id, ctx) -> dict[YouName, You]`: runs every registered builder, and one builder raising becomes an abstaining `You` whose meta records the error

- [x] **Step 1: Write the failing tests**

```python
def test_predict_sign_and_abstain():
    you = You("stated", np.array([1., 0.]), {})
    assert predict(you, np.array([1., 0.]), np.array([0., 1.])) == 1.0
    assert predict(You("stated", None, {}), np.ones(2), np.zeros(2)) == 0.5

def test_text_direction_is_unit_and_differs_from_neutral(fake_encoder):
    w = text_direction(fake_encoder, ["red things"]); assert np.isclose(np.linalg.norm(w), 1)

def test_stated_abstains_without_description(ctx, user): assert build_stated(ctx.conn, user, ctx).w is None
def test_stated_prefers_red_items_when_asked_for_red(ctx, user):
    save_answer(ctx.conn, user, "describe", "text", "I love red clothes.")
    you = build_stated(ctx.conn, user, ctx); a, b = red_and_blue_centered(ctx)
    assert predict(you, a, b) == 1.0

def test_build_all_yous_isolates_failures(ctx, user, monkeypatch):
    monkeypatch.setitem(BUILDERS, "bigfive", lambda *_: 1 / 0)
    yous = build_all_yous(ctx.conn, user, ctx); assert yous["bigfive"].w is None and "error" in yous["bigfive"].meta
```

`ctx` is a test fixture: `load_context(fixture_data, Settings(6, 6, 3, 2, 7), FakeEncoder())` with a `":memory:"` DB, after `embed_domain` has run on both fixture domains. `user` is a fixture that runs `create_user(ctx.conn, "test")`.

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/yous -q`
Expected: FAIL.

- [x] **Step 3: Implement `base.py`, `stated.py`, `registry.py` and `context.py`.** Add the `ctx` and `user` fixtures to `tests/conftest.py`.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest tests/yous -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber/yous src/amber/context.py tests
git commit -m "feat: You type, stated-you, builder registry, app context"
```

### Task 10: Revealed you and the pair flow (train → showdown)

**Files:**
- Create: `src/amber/yous/revealed.py`, `src/amber/flow.py`, `tests/yous/test_revealed.py`, `tests/test_flow.py`
- Modify: `src/amber/yous/registry.py` (register `revealed`)

**Interfaces:**
- Consumes: `PreferenceModel` (6), `choose_pair` and `random_pair` (7), the store functions (8), `AppContext`, `build_all_yous` and `predict` (9).
- Produces:
  - `train_preference_model(choices: list[Choice], index: VectorIndex, basis: Basis, prior_var: float = PRIOR_VAR) -> PreferenceModel`: replays answered choices in order
  - `build_revealed(conn, user_id, ctx) -> You`
    - uses only `list_choices(..., domain="clothes", phase="train")`
    - computes `w = normalize(basis.lift(model.mean))`, with `meta={"n": len(choices)}`
    - fewer than 5 choices means abstain
  - `allowed_ids(conn, user_id, ctx) -> set[str]`
    - reads `get_answers(..., "filters")`, where the keys `genders` and `sections` hold JSON lists
    - the default is all genders, with `master_category ∈ DEFAULT_SECTIONS`
  - `class CatalogExhausted(Exception)`
  - `next_pair(conn, user_id: str, domain: Literal["clothes", "art"], phase: Literal["train", "showdown", "retest"], ctx: AppContext, rng: np.random.Generator | None = None) -> Choice | None`
    - returns the open pair if there is one
    - returns `None` when that phase's count from `ctx.settings` is reached
    - `train` builds an active pair from a model replayed on that domain's train choices
    - `showdown` works like this:
      - it's clothes only
      - on the first call, it runs `build_all_yous` and `save_you` for each You (the freeze)
      - it draws a `random_pair` and saves `predict` for every frozen You via `save_predictions`, then returns the pair
    - `retest` is wired up in Task 18
    - excludes `shown_ids` for that domain
  - `progress(conn, user_id, domain, phase, ctx) -> dict`, shaped `{"done": int, "total": int}`

- [x] **Step 1: Write the failing tests**

```python
def test_build_revealed_ignores_showdown_choices(ctx, user):
    play_train(ctx, user, prefer="red", n=6); play_showdown(ctx, user, prefer="blue", n=3)
    you = build_revealed(ctx.conn, user, ctx); assert you.meta["n"] == 6

def test_revealed_abstains_under_5(ctx, user):
    play_train(ctx, user, n=4); assert build_revealed(ctx.conn, user, ctx).w is None

def test_revealed_learns_simulated_preference(ctx, user):
    play_train(ctx, user, prefer="red", n=6)   # FakeEncoder: dim 0 is red
    assert predict(build_revealed(ctx.conn, user, ctx), *red_and_blue_centered(ctx)) == 1.0

def test_next_pair_resumes_open_pair(ctx, user):
    p = next_pair(ctx.conn, user, "clothes", "train", ctx); assert next_pair(ctx.conn, user, "clothes", "train", ctx) == p

def test_train_phase_ends_at_setting(ctx, user):
    play_train(ctx, user, n=ctx.settings.clothes_train_pairs)
    assert next_pair(ctx.conn, user, "clothes", "train", ctx) is None

def test_showdown_freezes_yous_and_predicts_before_answer(ctx, user):
    play_train(ctx, user, n=6); p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert set(load_yous(ctx.conn, user)) == set(BUILDERS)
    assert set(get_predictions(ctx.conn, p.pair_id)) == set(BUILDERS)

def test_pairs_never_repeat_items(ctx, user):
    seen = [i for c in play_train(ctx, user, n=6) for i in (c.a_id, c.b_id)]; assert len(seen) == len(set(seen))
```

`play_train`, `play_showdown` and `play_art` live in `tests/conftest.py`. They're test helpers that loop `next_pair` and `answer_pair` (for clothes train, clothes showdown, or art train), choosing with a `SimulatedUser` whose `w` points at the `prefer` color dim (a coin flip when `prefer` is omitted). Each returns the answered `Choice` list.

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/yous/test_revealed.py tests/test_flow.py -q`
Expected: FAIL.

- [x] **Step 3: Implement `revealed.py` and `flow.py`.** Use `rng = np.random.default_rng(hash((SEED, user_id, domain, phase, n_done)) & 0xFFFFFFFF)` so pairs are reproducible per user.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber tests
git commit -m "feat: revealed-you and train/showdown pair flow with frozen predictions"
```

### Task 11: Scoring and the API (v1)

**Files:**
- Create: `src/amber/eval/__init__.py`, `src/amber/eval/metrics.py`, `src/amber/api/__init__.py`, `src/amber/api/app.py`, `tests/eval/test_metrics.py`, `tests/api/test_flow_api.py`
- Modify: `src/amber/cli.py` (add `amber serve [--port 8000]`)

**Interfaces:**
- Consumes: everything from Task 10.
- Produces, in `amber.eval.metrics`:
  - `@dataclass(frozen=True) class Score: you: str; correct: float; n: int; accuracy: float; ci_low: float; ci_high: float`
  - `score_showdown(choices: list[Choice], predictions: dict[str, dict[str, float]], n_boot: int = 2000, seed: int = SEED) -> list[Score]`
    - `predictions` maps `pair_id` to a `{you: p_a}` dict
    - a pair counts as correct with value `p_a` if `a` was chosen, else `1 − p_a`, so an abstain scores 0.5
    - the CI is a 95% percentile bootstrap over pairs
    - results come back in `YOU_ORDER`
- Produces: `create_app(ctx: AppContext) -> FastAPI`, with these routes:

| Method + path | Body | Returns |
|---|---|---|
| `POST /api/users` | `{name}` | `{user_id}` |
| `PUT /api/users/{uid}/filters` | `{genders: [str], sections: [str]}` | `{allowed: int}` (422 if 0) |
| `PUT /api/users/{uid}/describe` | `{text}` | `{}` |
| `POST /api/users/{uid}/next` | `{domain, phase}` | `{pair_id, a: Item, b: Item, progress}` or `{done: true, progress}` |
| `POST /api/pairs/{pair_id}/choose` | `{chosen_id}` | `{progress, reveal}`, where `reveal` is `{you: bool}` for showdown pairs, else `null` (409 on `AlreadyAnswered`, 422 on a bad id) |
| `GET /api/users/{uid}/results` | | `{scores: [Score], n_showdown: int}` |
| `GET /thumbs/{domain}/{n}.jpg` | | the image file (404 if missing) |

`Item` is `{id, domain, title, thumb_url, labels: {…non-empty catalog columns}}`. In `reveal`, `true` means that You's prediction was on the side you chose.

- [x] **Step 1: Write the failing tests**

```python
def test_score_counts_abstain_as_half():
    s = {x.you: x for x in score_showdown(choices_4, {"p1": {"stated": 1., "revealed": .5}, ...})}
    assert s["stated"].accuracy == 0.75 and s["revealed"].accuracy == 0.5 and s["stated"].n == 4

def test_ci_brackets_accuracy():
    for s in score_showdown(choices_20, preds_20): assert s.ci_low <= s.accuracy <= s.ci_high

def test_full_flow_api(client):   # TestClient(create_app(ctx)) with ctx fixture (small Settings)
    uid = client.post("/api/users", json={"name": "y"}).json()["user_id"]
    client.put(f"/api/users/{uid}/filters", json={"genders": ["Men"], "sections": ["Apparel"]})
    client.put(f"/api/users/{uid}/describe", json={"text": "I like red."})
    for phase in ("train", "showdown"):
        while not (r := client.post(f"/api/users/{uid}/next", json={"domain": "clothes", "phase": phase}).json()).get("done"):
            client.post(f"/api/pairs/{r['pair_id']}/choose", json={"chosen_id": r["a"]["id"]})
    res = client.get(f"/api/users/{uid}/results").json()
    assert res["n_showdown"] == 3 and {s["you"] for s in res["scores"]} >= {"stated", "revealed"}

def test_pairs_respect_filters(client): ...   # every served item has gender Men and master_category Apparel
def test_every_served_item_has_thumbnail(client): ...   # GET thumb_url returns 200 for every served item
def test_double_submit_is_safe(client): ...   # same choose twice → 200 both; different → 409
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/eval tests/api -q`
Expected: FAIL.

- [x] **Step 3: Implement `metrics.py`, `app.py`, and `amber serve`.** Run `amber serve` with uvicorn, and serve `web/dist` at `/` if it exists. Add CORS for `http://localhost:5173`.

- [x] **Step 4: Run the tests and confirm they pass**

Run: `uv run pytest -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber tests
git commit -m "feat: showdown scoring with bootstrap CIs; FastAPI v1"
```

### Task 12: Web app v1 (describe → sections → this-or-that → showdown → scoreboard)

**Files:**
- Create: `web/` (Vite React-TS scaffold), `web/src/api.ts`, `web/src/App.tsx`, and `web/src/screens/Welcome.tsx`, `Describe.tsx`, `Sections.tsx`, `ThisOrThat.tsx`, `Showdown.tsx`, `Scoreboard.tsx`; `.claude/launch.json`
- Modify: `web/vite.config.ts` (proxy `/api` and `/thumbs` to `http://localhost:8000`)

**Interfaces:**
- Consumes: the API routes from Task 11, exactly as tabled.
- Produces:
  - `web/src/api.ts` exports typed functions `createUser`, `setFilters`, `setDescribe`, `nextPair`, `choose` and `getResults`, mirroring the route table
  - the current `user_id` is kept in `localStorage` (wrapped in try/catch) so a reload resumes

- [x] **Step 1: Invoke the `frontend-design` skill** to set the visual direction. Then scaffold with `npm create vite@latest web -- --template react-ts` and `npm i react-router-dom`.

- [x] **Step 2: Build the screens.** Each one has a single job:
  - **Describe:** a textarea with the prompt "Describe your style in a sentence or two."
  - **Sections:** gender and section toggles, which call `setFilters`
  - **ThisOrThat** (reused for art in Task 13):
    - two large images with titles hidden (pictures only), plus a progress bar
    - ← and → keys choose
    - both buttons are disabled while a `choose` request is in flight
    - the next pair's images are prefetched
  - **Showdown:** the same layout. After each pick it briefly shows which yous called it, along with a running tally.
  - **Scoreboard:** one row per You with an accuracy bar, the 95% CI as a whisker, and a dashed line at 50% labeled "coin flip"

- [x] **Step 3: Add `.claude/launch.json`** with two configs: `api` (`uv run amber serve`, port 8000) and `web` (`npm --prefix web run dev`, port 5173).

- [x] **Step 4: Verify in the browser.** Start both servers with `preview_start`, then play the whole flow at real settings (40 + 20). Confirm:
  - (a) a reload mid-pair resumes the same pair
  - (b) a fast double-click records one answer
  - (c) the scoreboard shows stated and revealed

  Take a screenshot of the scoreboard.

- [x] **Step 5: Commit**

```bash
git add web .claude/launch.json
git commit -m "feat: web app v1 — this-or-that, showdown, scoreboard"
```

**Milestone, first playable:** you can play it and see Revealed vs. Stated.

---

## Phase 4: Personality yous

### Task 13: Art you (taste outside shopping)

**Files:**
- Create: `src/amber/yous/art_taste.py`, `tests/yous/test_art_taste.py`
- Modify: `src/amber/yous/registry.py` (register `art`), `src/amber/flow.py` (art train pairs use `ctx.basis["art"]`, with all art allowed), `web/src/App.tsx` (add an art step before the clothes step, reusing `ThisOrThat`)

**Interfaces:**
- Consumes: `train_preference_model` (10), `list_choices` (8).
- Produces: `build_art(conn, user_id, ctx) -> You`
  - trains on `list_choices(..., domain="art", phase="train")` with `ctx.index["art"]` and `ctx.basis["art"]`
  - computes `w = normalize(ctx.basis["art"].lift(model.mean))`
  - fewer than 5 choices means abstain
- Produces: `art_direction(choices: list[Choice], index: VectorIndex, basis: Basis) -> np.ndarray | None`, a pure helper that raises `ValueError` if any choice has `domain != "art"`

- [x] **Step 1: Write the failing tests**

```python
def test_art_direction_rejects_clothes_choices(ctx, user):
    with pytest.raises(ValueError): art_direction([clothes_choice], ctx.index["art"], ctx.basis["art"])

def test_art_you_is_unit_and_abstains_when_short(ctx, user):
    play_art(ctx, user, prefer="red", n=4); assert build_art(ctx.conn, user, ctx).w is None
    play_art(ctx, user, prefer="red", n=2)   # now 6 total
    w = build_art(ctx.conn, user, ctx).w; assert np.isclose(np.linalg.norm(w), 1)

def test_red_art_lover_predicts_red_clothes(ctx, user):   # FakeEncoder shares the RGB dims across domains
    play_art(ctx, user, prefer="red", n=6); assert predict(build_art(ctx.conn, user, ctx), *red_and_blue_centered(ctx)) == 1.0
```

- [x] **Step 2: Run the tests and confirm they fail**

Run: `uv run pytest tests/yous/test_art_taste.py -q`
Expected: FAIL.

- [x] **Step 3: Implement it, then wire the flow and UI.**

- [x] **Step 4: Run the tests and confirm they pass, then verify in the browser** that the art step appears before clothes and that the scoreboard shows an `art` row.

Run: `uv run pytest -q`
Expected: PASS.

- [x] **Step 5: Commit**

```bash
git add src/amber tests web
git commit -m "feat: art-you — taste learned on art, carried over to clothes"
```
---

## Phase 4: The redesign

Tasks 14–16 turn the first-design app (train → showdown, free-text description) into the tap-only, every-pick-scored flow. Existing tests that encode the old flow are **updated or deleted in the same commit as the change that obsoletes them**, never left failing or skipped. Task 13's pending browser check is folded into Task 14 step 5.

### Task 14: One scored clothing phase with a prequential revealed you

**Files:**
- Modify: `src/amber/config.py`, `src/amber/context.py`, `src/amber/yous/base.py`, `tests/store/`, `src/amber/flow.py`, `src/amber/yous/revealed.py`, `src/amber/yous/registry.py`, `src/amber/eval/metrics.py`, `src/amber/api/app.py`, `tests/test_flow.py`, `tests/yous/test_revealed.py`, `tests/eval/test_metrics.py`, `tests/api/test_flow_api.py`, `tests/conftest.py` (`play_train` → `play_clothes`), `web/src/steps.ts`, `web/src/App.tsx`, `web/src/screens/Showdown.tsx` (becomes the only clothes screen), `web/src/screens/Scoreboard.tsx`
- Delete: the clothes `train` step from the web flow

**Interfaces:**
- `Settings` becomes `Settings(art_train_pairs: int, clothes_pairs: int, retest_pairs: int, retest_min_days: int)`, `DEFAULT_SETTINGS = Settings(20, 40, 20, 7)`.
- Phases: art uses `train` only; clothes uses `showdown` (the 40 scored picks) and, later, `retest`. Clothes `train` is gone: `next_pair(..., "clothes", "train", ...)` raises `ValueError`.
- Values is cut: drop `"values"` from `YouName` and `YOU_ORDER` in `yous/base.py` (now `("stated", "bigfive", "art", "revealed")`) and the `llm` field from `AppContext`; fix the store tests that use `"values"` as an example name.
- `FROZEN_YOUS: tuple[YouName, ...] = ("stated", "bigfive", "art")` in `registry.py`; `build_frozen_yous(conn, user_id, ctx) -> dict[str, You]` builds only those registered in `BUILDERS` (so it works before `bigfive` lands). `build_all_yous` stays for the readout.
- `revealed.py`:
  - `revealed_p_a(conn, user_id, pair: Choice, ctx) -> float`: `P(a)` from a `PreferenceModel` replayed on the user's clothes `showdown` choices **answered before `pair` was created** (ordered by answer time, excluding `pair` itself). Returns exactly `0.5` under `MIN_CHOICES` (5) such choices. Uses `model.p_prefer` on the basis projections of `centered(a)` and `centered(b)`.
  - `build_revealed` learns from all answered clothes `showdown` choices (used by readout, map and search after the run).
- `flow.next_pair` for clothes `showdown`:
  - first call: `_freeze_yous` with `build_frozen_yous`
  - draws `random_pair` from allowed, unseen items
  - saves predictions for every frozen You **plus** `revealed` (from `revealed_p_a`) before returning the pair
  - resuming an open pair repairs missing predictions as today
- `eval/metrics.py`:
  - `score_showdown` unchanged in contract; add `window: tuple[int, int] | None = None` (1-based inclusive pair positions by creation order) so results can show "all 40" and "picks 21–40"
  - `learning_curve(choices, predictions, you: str, size: int = 10) -> list[float]`: rolling-window accuracy per pair position (credit as in `score_showdown`), shorter at the start
- API:
  - `POST /next` no longer requires a clothes train phase; clothes `showdown` requires the art train phase to be done (409 otherwise).
  - `/choose` `reveal` covers all four yous (abstainers omitted, as now).
  - `GET /results` returns `{scores_all, scores_late, curves: {you: [float]}, n_clothes}`.
- Web: steps become `welcome → sections → describe → art → clothes → results` (describe is replaced in Task 15, the quiz inserted in Task 16). The clothes screen shows a running tally per You after each pick. The scoreboard shows `scores_all` with the late-window number beside it.

- [x] **Step 1: Write the failing tests**

```python
def test_clothes_train_phase_is_gone(ctx, user):
    with pytest.raises(ValueError): next_pair(ctx.conn, user, "clothes", "train", ctx)

def test_frozen_before_first_clothes_pair(ctx, user):
    play_art(ctx, user, n=ctx.settings.art_train_pairs)
    p = next_pair(ctx.conn, user, "clothes", "showdown", ctx)
    assert set(load_yous(ctx.conn, user)) == set(FROZEN_YOUS) & set(BUILDERS)
    assert set(get_predictions(ctx.conn, p.pair_id)) == (set(FROZEN_YOUS) & set(BUILDERS)) | {"revealed"}

def test_revealed_abstains_for_first_five_pairs(ctx, user):
    choices = play_clothes(ctx, user, prefer="red", n=6)
    assert [get_predictions(ctx.conn, c.pair_id)["revealed"] for c in choices[:5]] == [0.5] * 5

def test_revealed_prediction_uses_only_earlier_picks(ctx, user):
    choices = play_clothes(ctx, user, prefer="red", n=8)
    p7 = get_predictions(ctx.conn, choices[7].pair_id)["revealed"]
    assert p7 == pytest.approx(revealed_p_a_from(choices[:7], choices[7], ctx))  # replay helper in tests/fakes.py

def test_revealed_learns_red(ctx, user):
    choices = play_clothes(ctx, user, prefer="red", n=ctx.settings.clothes_pairs)
    late = score_showdown(choices, preds(ctx, choices), window=(21, 40))
    assert next(s for s in late if s.you == "revealed").accuracy > 0.7

def test_learning_curve_length_and_bounds(): ...
def test_score_window_selects_positions(): ...
def test_results_route_shape(client, finished_user): ...
```

- [x] **Step 2: Run the tests and confirm they fail.** `uv run pytest -q` — new tests FAIL; then update or delete each old-flow test the change obsoletes (list them in the commit message).
- [x] **Step 3: Implement** config, registry, revealed, flow, metrics, API, then the web changes.
- [x] **Step 4: Run the tests and confirm they pass.** `uv run pytest -q && npm --prefix web run build`
- [x] **Step 5: Browser check** (start both servers with `preview_start`): sections → describe → 20 art picks → 40 clothes picks with a live tally → results with four rows possible (stated, art, revealed now; bigfive after Task 16). Reload mid-pair resumes; a fast double-click records one answer. Screenshot the results.
- [x] **Step 6: Commit** in small commits (see Global Constraints), push.

### Task 15: Stated you from tapped style words

**Files:**
- Create: `src/amber/yous/files/style_words.yaml`, `web/src/screens/StyleWords.tsx`
- Modify: `src/amber/yous/stated.py`, `src/amber/api/app.py`, `tests/yous/test_stated.py`, `tests/api/test_flow_api.py`, `web/src/steps.ts`, `web/src/App.tsx`
- Delete: `web/src/screens/Describe.tsx`, `Describe.css`, and the `PUT /describe` route with its tests

**Interfaces:**
- `style_words.yaml`: 30 single words or two-word phrases a person would use for their own style, grouped for display into 5 rows of 6 (e.g. *minimal, classic, edgy, sporty, bohemian, preppy, streetwear, vintage, elegant, relaxed, bold, earthy, monochrome, colorful, romantic, utilitarian, …*). No brand names. Frozen with `prereg-v1`.
- `load_style_words() -> list[str]`
- `PUT /api/users/{uid}/style {words: [str]}`: 1–5 words, each in the list (422 otherwise). Stored as `save_answer(conn, uid, "stated", "words", json.dumps(words))`.
- `GET /api/style-words` → `{rows: [[str]]}`
- `build_stated`: abstains with `{"reason": "no words"}` when none; else `w = text_direction(encoder, [f"a photo of {word} clothing" for word in words])`, `meta={"words": words}`. Still never receives clothing choices.
- StyleWords screen: chips, tap to toggle, counter "3 of up to 5", Continue enabled at ≥ 1.

- [ ] **Step 1: Write the failing tests** (abstains with no words; red-word direction prefers the red item via `red_and_blue_centered`; route rejects 0, 6, and unknown words; leak rule still raises).
- [ ] **Step 2: Confirm they fail.** `uv run pytest tests/yous/test_stated.py tests/api -q`
- [ ] **Step 3: Implement**, replace the Describe step with StyleWords in `steps.ts`, delete Describe and its route.
- [ ] **Step 4: Confirm pass.** `uv run pytest -q && npm --prefix web run build`; browser-check the chips screen.
- [ ] **Step 5: Commit** in small commits, push.

### Task 16: Big Five you (Mini-IPIP + Mini-Markers bridge), then freeze the inputs

**Files:**
- Create: `src/amber/yous/bigfive.py`, `src/amber/yous/files/mini_ipip.yaml`, `src/amber/yous/files/bigfive_bridge.yaml`, `src/amber/yous/files/banned_words.txt`, `tests/yous/test_bigfive.py`, `web/src/screens/Quiz.tsx`
- Modify: `src/amber/yous/base.py` (`load_banned_words() -> set[str]`), `src/amber/yous/registry.py` (register `bigfive`), `src/amber/api/app.py`, `web/src/steps.ts`, `web/src/App.tsx`

**Interfaces:**
- `mini_ipip.yaml`: the 20 Mini-IPIP items (Donnellan, Oswald, Baird & Lucas, 2006), 4 per trait, with trait and keying. IPIP items are public domain; cite the source in the file header.
- `@dataclass(frozen=True) class IpipItem: id: str; text: str; trait: Literal["O","C","E","A","N"]; keyed: Literal[1, -1]`
- `load_ipip() -> list[IpipItem]`; `score_ipip(answers: dict[str, int]) -> dict[str, float]`: per trait, mean of keyed answers (1–5, reverse = 6 − x) mapped to `[-1, 1]` by `(mean − 3) / 2`; a trait with no answers is omitted.
- `bigfive_bridge.yaml`: per trait, `high` and `low` lists of 4 adjectives taken from Saucier's (1994) Mini-Markers for that pole (e.g. E high: talkative, energetic, bold, extraverted; E low: shy, quiet, bashful, withdrawn). Three prompts per pole: `"clothing worn by someone who is {a}, {b} and {c}"` over rotations of the list. A test asserts no adjective is in `banned_words.txt` (~80 garment, color, fabric and style words).
- `build_bigfive`: `w = normalize(Σ_t s_t · (text_direction(high_t) − text_direction(low_t)))` over answered traits; abstains with fewer than 10 answered items (`{"reason": "skipped"}`). `meta={"traits": {t: s_t}}`.
- Routes: `GET /api/ipip` → items; `PUT /api/users/{uid}/ipip {answers: {item_id: 1..5}}`.
- Quiz screen: one statement at a time, five big tap targets ("Very inaccurate" … "Very accurate"), auto-advance on tap, back button, progress "7 / 20", and "Skip quiz". Inserted after style words.

- [ ] **Step 1: Write the failing tests** (keying and reverse scoring; trait range; skip → abstain; bridge words never banned; a high-E fake person's `w` has positive cosine with E-high text direction; leak rule).
- [ ] **Step 2: Confirm they fail.** `uv run pytest tests/yous/test_bigfive.py -q`
- [ ] **Step 3: Write the YAML files and `banned_words.txt`, then implement** `bigfive.py`, routes, the Quiz screen.
- [ ] **Step 4: Confirm pass**, then do a timed browser run of the whole flow and check it fits ≈ 4 minutes of tapping (excluding image load); record the time in the commit message.
- [ ] **Step 5: Freeze the inputs.** Show the user `style_words.yaml`, `mini_ipip.yaml`, `bigfive_bridge.yaml` and `banned_words.txt` for one last review. Then commit and `git tag prereg-v1` (the readout concept list in Task 17 joins as `prereg-v1.1`, noted in its commit).

**Milestone, the real experiment:** play the full 4-minute flow for real for the first time.

---

## Phase 5: Results

### Task 17: The readout ("you said… you pick…")

**Files:**
- Create: `src/amber/explain/__init__.py`, `src/amber/explain/readout.py`, `src/amber/explain/concepts.yaml`, `tests/explain/test_readout.py`

**Interfaces:**
- `load_concepts() -> list[str]`: about 120 style concepts (superset of the style words)
- `concept_alignment(you: You, encoder: Encoder, concepts: list[str]) -> list[tuple[str, float]]`
- `label_lift(choices: list[Choice], items: pd.DataFrame, column: str, min_count: int = 3) -> list[tuple[str, float]]`
- `@dataclass class Readout: said: list[str]; picks: list[str]; art_says: list[str]; surprises: list[str]; sentences: list[str]`
- `build_readout(yous: dict[str, You], choices: list[Choice], items: pd.DataFrame, encoder: Encoder) -> Readout`. `said` = the tapped words; `picks` = top concepts of revealed plus top label lifts; `art_says` = top concepts of the art you ("your art taste says: earthy, textured, muted"); `surprises` = concepts high for revealed but not in `said`.

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Confirm they fail.** `uv run pytest tests/explain/test_readout.py -q`
- [ ] **Step 3: Implement and write `concepts.yaml`**, tag `prereg-v1.1`.
- [ ] **Step 4: Confirm pass.**
- [ ] **Step 5: Commit** in small commits, push.

### Task 18: Results page — learning curve, map of your yous, readout

**Files:**
- Create: `src/amber/explain/geometry.py`, `tests/explain/test_geometry.py`, `web/src/screens/Results.tsx`, `web/src/components/YouMap.tsx`, `web/src/components/LearningCurve.tsx`
- Modify: `src/amber/api/app.py` (`/results` adds `cosines`, `map`, `readout`), `web/src/App.tsx` (Results replaces Scoreboard and contains it)

**Interfaces:**
- `you_cosines(yous: dict[str, You]) -> dict[str, dict[str, float]]`: symmetric, non-abstaining only
- `map_points(index: VectorIndex, basis2d: Basis, yous: dict[str, You], sample_ids: list[str], top: int = 20) -> dict`: 300 allowed clothing items (sampled with `SEED`) plus 100 art items projected into the **clothes** 2-D basis, and one marker per You
- Results page order (run the `dataviz` skill before writing charts):
  1. **Headline sentence:** "Your art taste predicted X of your 40 picks" with its CI, next to coin flip and the other yous
  2. **Scoreboard:** one row per You, accuracy bar, 95% CI whisker, dashed 50% line
  3. **Learning curve:** rolling accuracy per pick for all four yous; revealed starts at 50% and climbs, the frozen yous are flat-ish lines
  4. **Readout sentences**
  5. **YouMap:** SVG scatter, clothes and art as small dots (hover shows the thumbnail), four labeled You markers, colors consistent with the scoreboard
  6. **Cosine table**

- [ ] **Step 1: Write the failing tests** (geometry; `/results` shape)
- [ ] **Step 2: Confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm pass**, browser check at desktop and phone widths, screenshot.
- [ ] **Step 5: Commit** in small commits, push.

### Task 19: The retest a week later (the ceiling)

**Files:**
- Create: `src/amber/eval/retest.py`, `tests/eval/test_retest.py`
- Modify: `src/amber/flow.py` (`phase="retest"`), `src/amber/eval/metrics.py`, `src/amber/api/app.py` (`/results` adds `ceiling`, `normalized`, `retest_due_at`), `web/src/screens/Results.tsx`

**Interfaces:**
- `retest_due(conn, user_id, settings, now) -> bool`: last clothes pick ≥ `retest_min_days` old and retest unfinished
- `next_retest_pair(conn, user_id, settings, now) -> Choice | None`: re-shows 20 of the 40 clothing pairs (chosen with `SEED`), sides swapped at random
- `self_consistency(original, retest) -> float | None`
- `normalized_accuracy(accuracy, ceiling) -> float | None`: `(accuracy − 0.5) / (ceiling − 0.5)`; `None` when `ceiling ≤ 0.5`
- Results: a "Come back on <date>" banner, a retest button when due, then a ceiling line on the scoreboard and normalized numbers.

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Confirm they fail.**
- [ ] **Step 3: Implement.** `now` is injectable; the API uses `datetime.now(UTC)`, overridable by a dev-only `AMBER_NOW` env var.
- [ ] **Step 4: Confirm pass**, browser check by shifting `AMBER_NOW`.
- [ ] **Step 5: Commit** in small commits, push.

---

## Phase 6: Search

### Task 20: Search ranked by any of your yous

**Files:**
- Create: `src/amber/search/__init__.py`, `src/amber/search/rerank.py`, `tests/search/test_rerank.py`, `web/src/screens/Search.tsx`
- Modify: `src/amber/api/app.py` (`GET /api/users/{uid}/search?q=…&you=art`), `web/src/App.tsx`

**Interfaces:**
- `@dataclass class SearchResult: generic: list[str]; personal: list[str]`
- `search(query, you, encoder, index, allowed, n_candidates=SEARCH_CANDIDATES, n_results=SEARCH_RESULTS, lam=SEARCH_LAMBDA) -> SearchResult`
- `you` defaults to `art` (the headline) and can be any You. The search box is the one optional text input in the app; suggested query chips ("black jacket", "summer dress", "sneakers", …) make it tap-only by default.
- UI: query chips + box, two columns ("Everyone" / "Ranked by your art taste"), You picker.

- [ ] **Step 1: Write the failing tests** (`test_search_respects_allowed`, ranking changes with `you`, `you=None` → generic only)
- [ ] **Step 2: Confirm they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Confirm pass**, browser check that "black jacket" gives visibly different columns.
- [ ] **Step 5: Commit** in small commits, push.

---

## Phase 7: Ship

### Task 21: README, real results, demo

**Files:**
- Create: `docs/results/first-run.md`, `docs/demo.gif`
- Modify: `README.md`, `NOTES.md`

- [ ] **Step 1: Check the whole suite.** `uv run pytest -q && uv run pytest -q -m slow && uv run ruff check . && npm --prefix web run build`
- [ ] **Step 2: The user does a full run.** Record scoreboard, curves, cosines and readout in `docs/results/first-run.md`; fill in the retest a week later.
- [ ] **Step 3: Write `README.md`:** the one-line question ("Can your taste in art predict what you'd wear?"), the headline number, the four yous, how it works (vector space, centering, PCA, probit, prequential scoring, freeze rules), results with CIs, how to run it in 4 commands, honest caveats (n = 1; CLIP has seen product photos; Mini-IPIP isn't clinical), credits and licenses, and the research it builds on (from NOTES.md).
- [ ] **Step 4: Record a demo GIF** (< 10 MB) of art picks, clothes picks with the live tally, and the results page.
- [ ] **Step 5: Commit and push** (pushing to the public repo is approved).

---

## Later (each one is its own plan)

- **Host it at `amber.yuvijain.com` for $0:** Cloudflare Pages (static site) + R2 (thumbnails; trim the catalog to fit) + D1 (save everyone's results). Precompute every vector and text direction offline, port the 32-dim probit model and pair flow to TypeScript so nothing runs server-side but D1 writes. Verify the clothing dataset's license first, or switch catalogs.
- **Friends / population results:** with D1 storing runs, compare art → clothes transfer across people and add an "average person" baseline.
- **Image-based Big Five:** replace Mini-IPIP with forced-choice image pairs (Hilliard et al. 2022 got r = .60–.78 vs. IPIP-NEO-120 in 5 min), making the whole app pictures-only. Needs its own validation.
- **More domains:** rooms, architecture (Vessel et al. 2018: highly individual, like art).
- **The paper:** pre-register, recruit 30–50 people, report chance-corrected normalized accuracy.
