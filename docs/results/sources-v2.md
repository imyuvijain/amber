# Sources v2: decisions from Google and Instagram exports

One person's real exports. Every decision is predicted from strictly earlier data;
a 7-day return label is used for training only after its 7 days have passed.
Holdout = the last 20% of scored decisions; 95% CIs resample whole days.
Protocol frozen as git tag `decide-v2` before this run.

| Decision | Holdout | Base rate | AUC base | AUC context | AUC life | AUC memory | AUC recency | Works? |
|---|---|---|---|---|---|---|---|---|
| Watch the same YouTube channel again within 7 days | 2,878 (2026-06-10 to 2026-11-06) | 31% | 0.464 | 0.513 | 0.537 | 0.850 | 0.860 | yes |
| Like the same Instagram creator again within 7 days | 1,949 (2026-06-11 to 2026-11-06) | 29% | 0.421 | 0.580 | 0.568 | 0.938 | 0.941 | yes |
| Reels feedback: "see more" vs "not interested" | 381 (2025-03-31 to 2026-11-08) | 97% | 0.567 | 0.596 | 0.724 | 0.760 | 0.757 | no |

## Log-loss gain of each rung over the one below (higher is better)

| Decision | context over base | life over context | memory over life | recency over memory |
|---|---|---|---|---|
| Watch the same YouTube channel again within 7 days | -0.002 (-0.011 to +0.006) | -0.004 (-0.012 to +0.004) | +0.193 (+0.158 to +0.222) | +0.015 (+0.007 to +0.024) |
| Like the same Instagram creator again within 7 days | +0.013 (-0.008 to +0.030) | -0.010 (-0.022 to +0.001) | +0.367 (+0.311 to +0.437) | +0.012 (+0.007 to +0.017) |
| Reels feedback: "see more" vs "not interested" | -0.101 (-0.119 to -0.078) | +0.060 (+0.014 to +0.094) | +0.037 (+0.022 to +0.056) | +0.001 (+0.000 to +0.002) |

## Holdout log loss per rung (lower is better)

| Decision | base | context | life | memory | recency |
|---|---|---|---|---|---|
| Watch the same YouTube channel again within 7 days | 0.622 | 0.624 | 0.628 | 0.435 | 0.420 |
| Like the same Instagram creator again within 7 days | 0.610 | 0.597 | 0.607 | 0.240 | 0.228 |
| Reels feedback: "see more" vs "not interested" | 0.181 | 0.282 | 0.221 | 0.184 | 0.183 |

"Works" means memory beats both base and context with a CI lower bound above 0.
