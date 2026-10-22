# Active-learning benchmark

Does choosing the next pair actively beat showing random pairs, and which prior variance
should the preference model start with?

## Result

Held-out pairwise accuracy of simulated people on the real clothes basis
(`uv run amber bench-active --n-users 100`, 222 s on an Apple Silicon laptop; exactly 200 held-out pairs scored per person):

| temperature | prior_var | active | random | active - random |
| --- | --- | --- | --- | --- |
| 0.05 | 0.1 | 0.805 | 0.788 | +0.017 |
| 0.05 | 1 | 0.818 | 0.797 | +0.022 |
| 0.05 | 10 | 0.812 | 0.787 | +0.025 |
| 0.2 | 0.1 | 0.654 | 0.646 | +0.008 |
| 0.2 | 1 | 0.643 | 0.637 | +0.006 |
| 0.2 | 10 | 0.640 | 0.620 | +0.021 |

A coin-flip clicker (`random_clicker=True`, active selection, `prior_var` 1) scores 0.489,
so the model neither crashes nor invents signal for someone who clicks at random.

**Chosen `config.PRIOR_VAR = 1.0`.** Criterion: the best active accuracy averaged over the
two temperatures. That is 0.730 for `prior_var` 0.1, **0.731 for 1** and 0.726 for 10. The
spread is small (about 0.005); 1.0 also keeps the plain unit-Gaussian prior, so it was
already the starting value and did not change.

## Reading it

* Active selection is never worse than random in this table and is better by about 0.02
  where there is signal to learn (low temperature). At temperature 0.2 (a noisy person)
  both strategies sit near 0.65 and the gap is small (+0.006 to +0.008 at `prior_var` 0.1
  and 1; +0.021 at 10).
* The gain is modest on the real catalog and each cell is a single run of 100 users (the
  standard error of a cell is roughly 0.005 to 0.01), so treat gaps under about 0.01 as
  within noise. 40 pairs for a 32-dimensional taste is not many; the benchmark did not
  vary the number of pairs.
* Accuracy is capped by the person's own noise, not only by the model: at temperature 0.2 a
  person contradicts their own taste on many close pairs.

## How it is measured

* **Space:** `clothes/emb/ViT-B-32__laion2b_s34b_b79k`, 41,802 items. `X` is the 32-component
  PCA basis applied to the centered embeddings; near-duplicates are judged on the raw unit
  embeddings (cosine strictly above `NEAR_DUP_COS` = 0.95 is never shown as a pair).
* **Simulated person:** taste `w` is the normalized projected vector of a random catalog item.
  They pick `a` over `b` with probability `sigmoid((w.a - w.b) / temperature)`. Temperature
  0.05 is a decisive person, 0.2 a noisy one.
* **Training:** 40 pairs per person, none of whose items repeats. `active` takes the best of
  2000 random candidate pairs by `v * p * (1 - p)` (`v` is the posterior variance of the
  score difference, `p` the predicted probability); `random` takes a random pair.
* **Scoring:** 200 random pairs from items not used in training. The model is right when
  `p_prefer > 0.5` agrees with what the person actually picks (a noisy pick counts, which is
  why a coin-flip clicker lands near 0.5).
* **Settings:** 100 users per cell, 40 training pairs, 200 test pairs, seed 7. Both
  strategies see the same users (same taste vectors).
