# Amber

> A tool that knows you better than you know yourself.

**Status: work in progress.** This is a rough first draft. The vector space is built; the playable app, the personality yous and the real results are not. Nothing here is a result yet.

## The idea

What people *say* they like and what they *pick* are often different. Amber tests that. It builds five versions of you, four of which never see a single clothing choice of yours, and then makes them compete to predict which clothes you'll choose.

- **Stated preference:** you describe your style in your own words.
- **Revealed preference:** you answer 40 this-or-that clothing pairs and the model learns from your picks.
- **The showdown:** 20 more random clothing pairs. Before each one, every you predicts your pick, and the prediction is saved before your answer is accepted. Scores are compared against a coin flip (50%) and against your own retest a week later, since nobody agrees with themselves 100% of the time.

The core question: can a model that never saw anyone's shopping choices predict yours, just from who you are?

## The five yous

| You | Built from | Sees clothing choices? |
|---|---|---|
| `stated` | your own words describing your style | no |
| `bigfive` | 50-question IPIP personality test, bridged to clothes through fixed text prompts | no |
| `art` | this-or-that on paintings and photographs, learned in art space, carried over to clothes | no |
| `values` | open answers about how you live, condensed to personality adjectives by Claude (no clothing words allowed) | no |
| `revealed` | 40 this-or-that clothing choices | yes, training phase only |

## How it works

- **One shared vector space.** A pretrained CLIP model (`open_clip`, ViT-B-32 laion2b) embeds clothing photos, artworks and free text into the same space. No model is trained on consumer preference data. (The model has seen product photos online, so the fair claim is "no consumer preference data", not "never saw a product".)
- **Centering.** Each domain's mean vector is subtracted from its items and the result re-normalized. This removes "clothing-ness" and "art-ness", so a taste learned on art can point somewhere meaningful among clothes.
- **PCA to 32 dimensions.** Each domain gets a PCA basis fit on its centered vectors. Preferences are learned in 32 dimensions, which is what makes 30-40 choices enough.
- **Bayesian preference model.** An online probit model updates after every this-or-that answer. Active pair selection picks the next pair that should teach it the most, skipping near-identical photos.
- **Every you is a direction.** Each you is a unit vector in the space and predicts a pair by which item scores higher along it.
- **Showdown rules.**
  - Pairs are random, not chosen by a model.
  - Every prediction is saved before the answer is accepted.
  - All yous are frozen when the showdown starts.
  - Showdown and retest choices never train anything.
  - The `stated`, `bigfive`, `art` and `values` builders raise an error if handed clothing choices.

## Data

No big download is needed. Everything under `data/` is capped at about 1 GB.

- **Clothes:** about 41,800 products streamed from the Hugging Face dataset `benitomartin/fashion-product-images-small-384x512` and stored as small thumbnails (216x288). The streamed catalog is never downloaded whole. (Its license is unstated on Hugging Face; it originates from Kaggle's *Fashion Product Images*.)
- **Art:** 2,000 public-domain (CC0) paintings, photos and prints from the Art Institute of Chicago API, fetched politely at 1 request per second.
- **On disk today:** about 512 MB for `data/`, including the embeddings (see [docs/results/space.md](docs/results/space.md)).

## Tech stack

- Python 3.12 via `uv`
- PyTorch (Apple GPU via MPS, CPU fallback) and `open_clip_torch`
- Hugging Face `datasets` (streaming), numpy, pandas, pyarrow, scipy, pillow
- Typer (CLI), httpx
- SQLite (stdlib) for the store
- pytest and ruff
- Planned: FastAPI backend and a Vite + React + TypeScript frontend; optional Claude API for the `values` you only

## Status

Tracked in [PLAN.md](PLAN.md) (20 tasks in 7 phases).

- [x] **Phase 1: The vector space** (tasks 1-5): ingest clothes and art, encoder, vector index, PCA basis, embeddings, text search
- [ ] **Phase 2: Revealed you** (tasks 6-8): the Bayesian preference model and the SQLite store are built; active pair selection and the simulated-person benchmark are in progress
- [ ] **Phase 3: First playable** (tasks 9-12): API and web app for describe, this-or-that, showdown, scoreboard
- [ ] **Phase 4: Personality yous** (tasks 13-15): art, Big Five and values yous
- [ ] **Phase 5: Results** (tasks 16-18): readout, map of your yous, retest a week later
- [ ] **Phase 6: Search** (task 19): generic vs. for-you search results
- [ ] **Phase 7: Ship** (task 20): real results, demo, final README

## Setup (what works today)

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
```

Build the data and the space:

```bash
uv run amber ingest clothes          # stream the catalog and save thumbnails (--limit N for a quick trial)
uv run amber ingest art              # fetch public-domain artworks from the AIC API
uv run amber shrink-thumbs           # optional: shrink clothing thumbnails in place to save disk
uv run amber embed                   # embed everything and fit the PCA bases
uv run amber eval-space              # measure retrieval quality (P@20)
```

Try it:

```bash
uv run amber search-text "red floral dress"
uv run amber search-text "a stormy seascape" --domain art
```

There is no web app yet.
