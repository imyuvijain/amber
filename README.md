# Amber

A personal decision engine, trained on your own data exports.

The goal is a model that makes the same decisions you would. Platforms each see one slice of you, but you can legally download every slice. The goal: a model you own that eventually sees more than any single platform does. Nothing is asked of you except the files you already have the right to download.

## Status

- **v0 (Amber): closed, negative result.** See below.
- **v1: Spotify skips, built and tested on fake data.** The real run is pending the owner's export. There is no real-person result yet.
- **Next:** friends' exports, YouTube Takeout (v2), then a "veto, not quiz" screen once predictions are good enough to act on.

## v0 (Amber): a quiz that lost

Amber tested whether art taste predicts clothing picks: four "yous" (stated, Big Five, art, and one learned from clothing picks) in one CLIP vector space, each predicting 40 random this-or-that clothing pairs. The owner played the real run on 2026-11-13.

- The art you scored **52.5%** over 40 pairs (95% CI 37.5 to 67.5). That is a coin flip.
- The you that learns from the owner's own clothing picks scored 66.25% (CI 52.5 to 80).
- The yous pointed in unrelated directions (all cosines within 0.06 of zero). Art's readout said "animal print, grunge, tie-dye"; the owner picked business, formal, elegant.
- The owner's verdict: "low quality photos, way too long onboarding, the choices were bs ... quizzing the user is a very big waste of time."

So v1 has no quiz. Full numbers and the caveats: [docs/results/first-run.md](docs/results/first-run.md). The v0 code (`amber ingest`, `embed`, `serve`, the web app) is still in the repo, frozen as it ran. Its design notes and research are in [NOTES.md](NOTES.md) and [PLAN.md](PLAN.md).

## v1: predict your own Spotify skips

For every play in your history, predict whether you skip it, using only plays that happened earlier. Models are scored prequentially (predict, record, then learn) on the last 20% of your timeline.

The ladder. Each rung adds one kind of knowledge, so the gaps between rungs are the result:

| Rung | Knows |
|---|---|
| base rate | your running skip rate |
| context | when and how you listen (hour, weekday, platform, shuffle, how the play started, position in session, gap since the previous play, whether that one was skipped) |
| memory | context plus your past skip rate for this artist and track, and play counts |
| recency | memory with exponentially decayed counts |

Label rule (frozen with git tag `decide-v1` before any real data): extended format, skip = `reason_end == "fwdbtn"` or `skipped` is true; account format, skip = played under 30 seconds. Metrics are log loss (primary), Brier, AUC and accuracy, always next to the base-rate baseline. 95% CIs come from a block bootstrap by calendar day.

### Results so far: fake users only

These are synthetic sanity checks, not a result about any real person. 5,000 fake plays, about 972 held out. Holdout log loss, lower is better (reproduce with `amber decide fake <folder> --kind personal --n-plays 5000`, or `--kind null`):

| User | base | context | memory | recency |
|---|---|---|---|---|
| personal (planted skip structure) | 0.670 | 0.650 | 0.562 | 0.526 |
| null (skips are random) | 0.602 | 0.602 | 0.601 | 0.602 |

Holdout accuracy: the personal user goes from 60.9% (base rate) to 73.9% (recency); every rung on the null user is 71.1%, the base rate.

On the personal user, memory beats context by 0.088 log loss (95% CI 0.067 to 0.109). On the null user every gap's CI contains 0, as it should.

**Two leaks the null user caught during the build, both fixed before the freeze.** Spotify's `ts` is the play's end time, so a "gap since last play" feature secretly encoded the play's own duration; the null user, who has no signal, scored AUC 0.957 before the fix in commit 9f84dfe ("decisions happen at play start"). Fix: decisions are timestamped at play start (`ts` minus `ms_played`). Then `ms_played` turned out to exclude pauses, so the current play's gap is replaced by the gap before the previous play. A null control that fails loudly is useful evidence the evaluation catches leaks. The real run is pending.

### Run it

Needs [uv](https://docs.astral.sh/uv/).

```bash
uv sync
# try it on a fake export (kept out of the real export folder)
uv run amber decide fake data/exports/fake --kind personal --n-plays 5000   # or --kind null, --fmt account
uv run amber decide ingest data/exports/fake
uv run amber decide eval data/exports/fake

# the real thing, from data/exports/spotify/ (see below)
uv run amber decide ingest
uv run amber decide eval --tz Asia/Kolkata                      # your timezone; add --publish for the aggregates-only report
```

`ingest` and `eval` write to `data/decide/`, so a fake run and a real run overwrite the same private files. `eval` writes the full private report there. With `--publish` it also copies an aggregates-only report to `docs/results/`; don't publish a fake run.

### What export to request

1. Go to [spotify.com/account/privacy](https://www.spotify.com/account/privacy).
2. Request **Extended streaming history** (it can take up to 30 days). Optionally also request **Account data** (about 5 days); it is coarser and is ignored when the extended files are present.
3. Put the files in `data/exports/spotify/`.

### Privacy

- `ip_addr` and `conn_country` are dropped at ingest and never stored.
- Podcasts, audiobooks, rows with no track and incognito plays are dropped.
- `data/` is git-ignored: raw exports and private reports never leave your machine.
- Committed reports carry no track, artist or album names and no IDs.
- No LLM, no paid API, no Spotify Web API, and no downloads beyond your own export.

## Caveats

- **n = 1.** The first real result will be one person. Friends' exports come next, comparing your own profile, someone else's profile and a pooled one.
- **Skip is not dislike.** People skip because they are busy, in the wrong mood, or have heard it too often. v1 predicts the behavior, not the opinion.
- **Beating the base rate is the bar.** A skip rate of 30% means "never skip" is already 70% accurate, which is why every metric is shown next to it.
- Amber v0's own reporting had known problems (retest "ceiling" is really retest agreement, bootstrap CIs degenerate at tiny n); see [first-run.md](docs/results/first-run.md).

## Tests

```bash
uv run pytest -q
uv run ruff check .
```

## Credits

- **Vector space (v0):** [open_clip](https://github.com/mlfoundations/open_clip), ViT-B-32 laion2b.
- **Clothes (v0):** Hugging Face dataset `benitomartin/fashion-product-images-small-384x512`, streamed and stored as small thumbnails. Its license is unstated on Hugging Face; it originates from Kaggle's *Fashion Product Images*.
- **Art (v0):** 2,000 public-domain (CC0) works from the [Art Institute of Chicago API](https://api.artic.edu/docs/).
- **Research behind v0:** Vessel et al. 2018 (taste in art is individual), the XPASS-Vis preprint (cross-domain transfer), Palmer and Griscom (harmony), Chamorro-Premuzic et al. (openness and art taste). Links are in [NOTES.md](NOTES.md).
- **Stack:** Python 3.12 with uv, PyTorch, numpy, pandas, pyarrow, scipy, Typer, FastAPI, SQLite, pytest, ruff.
