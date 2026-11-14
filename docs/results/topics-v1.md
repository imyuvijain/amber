# Topics v1: results

Spec: [`docs/superpowers/specs/2026-11-13-topics-v1-design.md`](../superpowers/specs/2026-11-13-topics-v1-design.md).
Scoring rules frozen as git tag `topics-v1` before the first real run. Two later tags
change explanations only, never the score:

- `topics-v1.1`: the "one short burst" driver read timestamps as nanoseconds. Real
  exports are stored in microseconds, so every topic was labelled a burst. Now fixed,
  with a test for every timestamp resolution.
- `topics-v1.2`: driver phrases fire only when a topic differs from the person's own
  baseline. Before this, 39 of 41 real topics read "active now".

Aggregates only. Topic names, keywords, titles and queries stay in `data/topics/`.

## Synthetic users

| User | Topics | Relevance spread (max − min) | Top topic |
|---|---|---|---|
| Null: every topic random over time, source and action (seeds 7, 8, 9) | 12 | 0.083, 0.099, 0.056 | n/a |
| Planted: a steady, deliberate, cross-platform topic plus five traps (seeds 7, 8, 9) | 14 | 0.725, 0.737, 0.702 | the planted topic, all seeds |

On the planted user, all five traps score below the background median: exam-only,
one-trip, three-week burst, first-year-only, and a passive exam-and-summer topic (the
gaming stand-in). The passive trap scores below every background topic. Driver
phrases name each trap correctly ("mostly during exams", "mostly on a trip", "one
short burst", "quiet since 2022").

## One person's real exports

| | |
|---|---|
| Documents with usable text (YouTube, Google, Instagram) | 49,912 (41,139 unique) |
| Topics (MiniLM embeddings, spherical k-means, merged by label) | 41 |
| Documents assigned to a topic | 91% |
| Topics named from the generic vocabulary / by keywords | 39 / 2 |
| Utility topics (chores such as maps or weather) | 6, all in the bottom half |
| Relevance, min / median / max | 0.01 / 0.35 / 0.61 |
| Build time on a laptop (embedding included, first run) | about 90 s |

- **Sanity check from the spec passes.** The topic the spec expects to score low ranks
  27th of 41 (relevance 0.17, median 0.35). It's driven by passive viewing, not by a
  rule: nothing in the code names it.
- **What separates topics.** Most of the spread comes from deliberate actions
  (feature sd 0.53), cross-platform presence (0.27) and steadiness (0.26). Exam and
  trip windows rarely dominate a topic at this granularity. One topic is mostly on a
  trip, and two lean on exam weeks.
- **Activity is recent.** 62% of the documents are from 2026, because the Instagram
  and Google exports start in late 2023 to 2024. Every feature compares a topic with
  the person's own activity on the same sources, so this skew doesn't make old topics
  look steady or new ones look like bursts.

## Not yet measured

Whether the ranking matches the person. That's the veto screen's job: each ✕ or ↑
becomes a label, and the weights are refit around the frozen ones.
