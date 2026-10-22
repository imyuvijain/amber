# Amber: project notes

Everything decided so far. The build plan is in [PLAN.md](PLAN.md).

## Status

- **2026-10-22:** brainstorm on *what* to build (below).
- **2026-10-23:** `PLAN.md` written (20 tasks in 7 phases). The hardware check is done: Apple M5, 24 GB RAM, 311 GB free, Python 3.14 (the project pins 3.12 via uv).
- **GitHub:** `imyuvijain/amber` is **public** and empty. The local repo is set up with that remote, but nothing is pushed yet. Waiting on your OK.

## Redesign on 2026-10-22: art → clothing, tap-only

**The question:** can your taste in art predict what you'd wear? Chosen over "art-only" (pretty but no testable claim) and "clothing-only" (a generic recommender) because it gives one headline number that nobody else has.

**User constraints:** no written answers anywhere; onboarding 2–5 minutes; no paid APIs (no LLM needed now); local demo first, hosting at `amber.yuvijain.com` on Cloudflare later for $0, saving results.

**What the research says, and what it changed:**
- *Personality predicts clothes weakly.* Openness and extraversion are the only consistent links, with small effects (e.g. openness → "urban, sophisticated" dress, OR 1.8; [PMC8455911](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8455911/)). Big Five stays as the comparison arm, measured cheaply.
- *Art taste is individual, and taste partly transfers.* Shared taste is high for faces and landscapes but low for art and architecture, so art separates people ([Vessel et al. 2018](https://www.aesthetics.mpg.de/en/research/former-departments-and-research-groups/department-of-neuroscience/aesthetic-experience/multiple-visual-domains.html)). A model of one rater trained on one domain recovers ~60% of in-domain accuracy on another, art ↔ fashion included ([XPASS-Vis preprint](https://arxiv.org/pdf/2606.15629)). Preference for harmony forms one factor across colour, shape and music ([Palmer & Griscom](https://palmerlab.berkeley.edu/pdf/PalmerGriscom2013.pdf)). → Art is the headline you.
- *Openness drives art taste* ([Chamorro-Premuzic et al.](https://research.gold.ac.uk/id/eprint/5046)), so art and Big Five aren't independent; the cosine table shows how much they overlap.
- *Choices beat self-report.* A model beat a friend's personality judgment after 70 Facebook likes ([Youyou et al. 2015](https://news.stanford.edu/2015/01/12/personality-computer-knows-011215/)); favourite pictures predict personality ([Segalin et al.](https://eprints.gla.ac.uk/124063)). → Taste picks are the core input.
- *Short tests:* Mini-IPIP, 20 items, α ≈ .73–.82 ([validation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12437056/)), public domain. Values (TIVI, 10 items, [free](https://gosling.psy.utexas.edu/two-short-measures-of-values-tivi-and-twivi/)) considered and cut to keep the story to four yous.
- *Pictures-only personality is possible* (forced-choice image pairs, r = .60–.78 vs. IPIP-NEO-120, [Hilliard et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC8883940/)) but the item bank isn't public. Parked under Later.
- *Adaptive testing* saves ~27–67% of items in simulations ([Nieto 2019](https://produccioncientifica.ucm.es/documentos/6198997349d6133331f3ecca)); not worth it at 20 items.

**Design consequences:** four yous (stated via tapped style words, Big Five via Mini-IPIP + Mini-Markers adjectives, art, revealed); 20 art picks; 40 random clothing picks, every one scored, with revealed scored prequentially; chance-corrected ceiling; ≈ 4-minute session. Details in PLAN.md, "Redesign of 2026-10-22".

**Free LLM, if ever needed:** Cloudflare Workers AI gives 10,000 neurons/day free on open models (Llama, Qwen; not GPT). Not used now.

## Changed on 2026-10-23

- **No 30 GB download.** The purchase history was only needed for the old "train on what people buy" idea.
  - Clothes now come from `benitomartin/fashion-product-images-small-384x512` on Hugging Face: 44k products at 384×512, with gender, category, type, color, season and usage labels. They're streamed and stored as thumbnails, about 0.5 GB on disk.
  - Art comes from the Art Institute of Chicago's free API: about 2,000 public-domain paintings, photos and prints, under 100 MB.
- **Three yous became five:** stated, Big Five, art, values, revealed. Personality You splits into its three signals so they can be compared.
- **No popularity baseline,** since we have no purchase data. The baselines are a coin flip (50%) and your own retest consistency as the ceiling.

## The idea

In your words: "a tool which knows you better than you know yourself." It's for fun, your resume, and curiosity. You want "an actual vector space."

**Core question:** can a model that never saw anyone's shopping choices predict yours, just from who you are?

- **Input:** a general personality engine. It's never trained on consumer preference data. *(your idea)*
- **Test:** clothing choices. *(agreed)*

### Three versions of you, in one shared vector space

| Point | Built from |
|---|---|
| **Stated you** | your own description of your style |
| **Personality you** | non-shopping signals only |
| **Revealed you** | your this-or-that clothing choices |

We measure which point sits closest to the clothes you actually pick, and how far apart the three are.

### The zero-shot bridge

A pretrained image-and-text model (CLIP or SigLIP) puts words and pictures in one space. Personality descriptions and product photos land in the same space, so there's no consumer-preference training.

### User flow (first proposal)

1. **Describe your style** in a sentence or two (gives Stated you).
2. **This or that:** about 40 pairs of real clothing items. Each pair is picked to teach the model the most (active learning).
3. **Showdown:** 20 more pairs. Before each, every version of you predicts the pick. A scoreboard shows the results at the end.
4. **Readout,** for example: "You said minimal and neutral. You keep picking warm colors, texture, loose fits."
5. **Search:** "black jacket", with normal results next to results ranked for you.

### What goes into "Personality you" (undecided)

1. **Big Five personality test.** Well studied, but it's only 5 numbers, and its links to taste are real but small. My guess is that it's the weakest of the three.
2. **Values and life questions** answered in your own words, interview-style. Park et al. 2024 did this with 1,000 people.
3. **Non-shopping visual taste:** paintings, landscapes, buildings, colors. It lives in the same image space. My guess is that it's the most promising.

### Scoring

- **Baselines:** random and popularity.
- **Ceiling:** test-retest. Redo some choices a week later; nobody agrees with themselves 100%.
- **Caveat:** pretrained models have seen product photos online. The fair claim is "no consumer preference data," not "never saw a product."
- **My bet:** Revealed you wins, and personality closes part of the gap. That's a real finding either way.

## Data

See "Changed on 2026-10-23" above. H&M (30 GB) was dropped because you didn't want the download.

## What we cut

- **Perfect size:** needs body measurements plus every brand's size chart. It's unsolved even for big retailers.
- **Search anything:** no free catalog covers everything. Start with fashion.
- **Pure personality-from-Spotify/Reddit:** hard to verify, and it ends up looking like a chatbot wrapper.

## Decisions

**Yours:**
- general personality engine as the input, clothing as the test
- a real vector space
- research paper is far in the future, so don't design around it now
- plan goes in `PLAN.md`
- use the `amber` GitHub repo

- no 30 GB download

**Mine (you can veto):**
- Python 3.12 + uv, FastAPI, SQLite, Vite + React
- CLIP-family embeddings, picked by measurement in PLAN Task 5
- streamed clothing catalog plus AIC art
- test-retest ceiling
- keep Revealed you in v1 as the comparison
- I build and you steer, since you didn't answer that one
- a web app as v1, not notebooks
- all three personality signals, in this order: art, Big Five, values (values needs your own Claude API key and abstains without it)

## Open questions

- Your coding and ML experience. This decides whether you'll write any parts yourself.
- OK to push to the public repo?

## References

- Youyou, Kosinski & Stillwell (2015, PNAS): computer personality judgments from Facebook likes beat friends'.
- Park et al. (2024), "Generative Agent Simulations of 1,000 People": interview-based agents matched survey answers at about 85% of people's own two-week consistency.
- Rentfrow & Gosling (2003): personality and music preference.
- Personality-aware recommender systems literature (e.g., Tkalčič & Chen).
