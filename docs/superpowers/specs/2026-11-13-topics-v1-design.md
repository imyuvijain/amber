# Topics v1: what the engine thinks matters to you

The engine works on **topics** (gaming, music production, physics, cooking...), never on
creators. Which creator someone engaged with, especially on Instagram, says little about
who they are. The creator-return tasks (`decide-v2`) stay as plumbing only.

## 1. Topic extraction (local only)

- **Documents.** One document per event that carries text: YouTube watch, like and
  activity titles; Google searches, visits and activity; Instagram captions of liked,
  saved and viewed posts, stories, videos, search terms and comments. Ads, emoji-only
  comment likes and rows without text are dropped. Verb prefixes ("Watched ",
  "Searched for ") are stripped.
- **Embedding.** `all-MiniLM-L6-v2` from sentence-transformers (about 90 MB, local,
  free). Vectors are cached under `data/topics/`, keyed by a hash of the text.
- **Clustering.** Spherical k-means on unique texts (weighted by how often each text
  occurs), `k = clamp(round(sqrt(n_unique / 10)), 20, 80)`, seed 7. Centroids closer
  than cosine 0.75 are merged. A document further than cosine 0.3 from its centroid,
  or in a topic with fewer than 30 documents, is left unassigned.
- **Names.** Each centroid is matched to a fixed, generic vocabulary of topic labels
  (`amber.topics.labels`). If the best label scores at least 0.3 cosine it becomes
  the name; otherwise the name is the top three c-TF-IDF keywords. Two clusters given
  the same label merge. Labels marked *utility* (weather, maps, translation...) flag
  chores, not interests.
- **Privacy.** Texts, keywords and topic names are written only under `data/`. The
  CLI prints counts, never names. Commits, the README and published reports carry
  aggregates only.

## 2. Relevance: "how important is this to me"

Every feature compares the topic to what it would look like if it simply followed the
person's overall activity (same sources, same times). A topic that behaves like
everything else scores 0 on every feature, so a person whose topics are random over
time gets **flat** relevance (the null control).

| feature | meaning | sign |
|---|---|---|
| `steady` | log2(active months / months expected if the topic's events were spread like all activity on its sources) | + |
| `phases` | log2(life phases touched / phases expected) | + |
| `deliberate` | log2(share of searches, likes, saves, comments, follows vs expected share) | + |
| `cross` | share of the person's platforms where the topic is present (lift ≥ 0.2), minus 1 | + |
| `now` | log2(share in the last 180 days vs expected) | + |
| `exam` | log2(share inside exam windows vs expected), positive part | − |
| `trip` | log2(share while abroad or on a listed trip vs expected), positive part | − |
| `utility` | named by a utility label | − |

Ratios are smoothed with 20 pseudo-events at the expected value and clipped.
`relevance = sigmoid(b + w · x)`. The frozen weights are in `amber.topics.relevance`.
Gaming should come out low; that is a sanity check, not a rule (nothing names gaming).

**Drivers** are short readable reasons built from the same numbers: "steady for 3
years", "mostly during exams", "one summer", "one short burst", "mostly on a trip",
"quiet since 2022", "active now", "mostly searched or liked", "mostly passive viewing",
"on YouTube and Instagram".

## 3. The veto screen

Ranked topics, most relevant first, each with its drivers. Tap ✕ on a wrong one (it is
muted and stored as a label `0`), tap ↑ to boost (label `1`), undo either. No typing.

**Tuning.** Labels keep the feature vector the topic had when it was tapped. Topics that
were shown above a vetoed topic and left alone count as weak positives (weight 0.3).
Weights are refit as a MAP logistic regression with a Gaussian prior (λ = 5) centred on
the frozen weights, so a few taps nudge the score and many taps can reshape it.

## 4. Rules

Tests first, with a null control (random topics over time give flat relevance) and a
planted user (a steady, deliberate, cross-platform topic outranks exam, trip, burst and
long-ago topics). The scoring rules are frozen with git tag `topics-v1` before the first
run on real data.
