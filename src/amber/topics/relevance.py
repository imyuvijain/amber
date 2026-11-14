"""Relevance per topic: "how important is this to me" (topics v1 spec, section 2).

Frozen protocol (git tag ``topics-v1``, fixed before the first real run):

Every feature compares a topic with what it would look like if it simply followed the
person's overall activity on the same sources. A topic that behaves like everything
else scores 0 on every feature, so random topics give flat relevance (null control).

- ``steady``: log2((active months + 1) / (expected active months + 1)). Expected months
  come from placing the topic's documents like all documents on its sources.
- ``phases``: the same over life phases (0 if the timeline has fewer than two).
- ``deliberate``: log2 of the topic's deliberate share over the expected share.
- ``cross``: share of the person's platforms where the topic shows up (lift >= 0.2),
  minus 1.
- ``now``: log2 of the topic's share in the last ``NOW_DAYS`` over the expected share.
- ``exam`` / ``trip``: positive part of log2 of the share inside exam windows / while
  abroad over the expected share.
- ``utility``: 1 if the topic was named by a utility label.

Shares are smoothed with pseudo-documents at the expected share: ``ALPHA``, or more
for rare events so the prior holds at least ``MIN_PSEUDO_HITS`` expected hits.
``exam`` and ``trip`` ignore the first ``DEADZONE`` (log2) above 1.
``relevance = sigmoid(bias + sum(w * x))``.
"""

import math

import numpy as np
import pandas as pd

FEATURES = ["steady", "phases", "deliberate", "cross", "now", "exam", "trip", "utility"]
FROZEN: dict = {
    "bias": 0.0,
    "weights": {
        "steady": 1.5,
        "phases": 0.5,
        "deliberate": 1.0,
        "cross": 1.5,
        "now": 1.0,
        "exam": -1.5,
        "trip": -1.5,
        "utility": -2.0,
    },
}
ALPHA = 200.0
MIN_PSEUDO_HITS = 20.0  # rare events (a short trip) get a stronger prior
DEADZONE = 0.25  # exam/trip penalties ignore ratios this close to 1 (log2)
NOW_DAYS = 365
LIFT_PRESENT = 0.2
MIN_SOURCE_SHARE = 0.01
BURST_DAYS = 60
CLIPS = {
    "steady": (-3.0, 0.5),
    "phases": (-3.0, 0.5),
    "deliberate": (-2.0, 2.0),
    "now": (-3.0, 0.5),
    "exam": (0.0, 3.0),
    "trip": (0.0, 3.0),
}
FACT_DEFAULTS = {
    "span_years": 0.0, "exam_share": 0.0, "trip_share": 0.0, "burst_share": 0.0,
    "summer_share": 0.0, "deliberate_share": 0.0, "recent_share": 1.0, "last_year": 0,
    "sources": "",
}
SOURCE_NAMES = {"youtube": "YouTube", "google": "Google", "instagram": "Instagram"}


def _log2_ratio(hit: float, n: float, expected: float) -> float:
    if expected <= 0 or expected >= 1:
        return 0.0
    alpha = max(ALPHA, MIN_PSEUDO_HITS / expected, MIN_PSEUDO_HITS / (1 - expected))
    share = (hit + alpha * expected) / (n + alpha)
    return math.log2(share / expected)


def _expected_cover(n: int, probs: np.ndarray) -> float:
    return float(np.sum(1.0 - (1.0 - probs) ** n))


def _max_window_share(ts: pd.Series, days: int) -> float:
    t = np.sort(ts.astype("int64").to_numpy())
    if len(t) == 0:
        return 0.0
    j = np.searchsorted(t, t + days * 86_400 * 10**9, side="right")
    return float((j - np.arange(len(t))).max() / len(t))


def topic_features(docs: pd.DataFrame, now: pd.Timestamp | None = None, utility=None) -> pd.DataFrame:
    """Features (``FEATURES``) and plain facts for every topic in ``docs["topic"]``.

    ``docs`` needs ts, source, deliberate, topic, and optionally in_exam, abroad, phase.
    Rows whose topic is None or -1 count as activity but are not topics.
    """
    d = docs.copy()
    now = now or d["ts"].max()
    d["month"] = d["ts"].dt.strftime("%Y-%m")
    d["recent"] = d["ts"] > now - pd.Timedelta(days=NOW_DAYS)
    for col in ("in_exam", "abroad"):
        d[col] = d[col].astype(bool) if col in d else False
    has_phase = "phase" in d and d["phase"].nunique(dropna=False) >= 2
    d["phase_key"] = d["phase"].fillna("-").astype(str) if "phase" in d else "-"

    n_all = len(d)
    src_n = d.groupby("source").size()
    sources = [s for s, k in src_n.items() if k / n_all >= MIN_SOURCE_SHARE]
    month_given_src = pd.crosstab(d["month"], d["source"], normalize="columns")
    phase_given_src = pd.crosstab(d["phase_key"], d["source"], normalize="columns")
    rate_given_src = d.groupby("source")[["deliberate", "recent", "in_exam", "abroad"]].mean()

    topic = d["topic"]
    is_topic = topic.notna() & (topic.astype(str) != "-1")
    rows = {}
    for t, g in d[is_topic].groupby("topic", sort=True):
        n = len(g)
        mix = g["source"].value_counts(normalize=True).reindex(src_n.index, fill_value=0.0)
        exp_month = (month_given_src * mix).sum(axis=1).to_numpy()
        exp_phase = (phase_given_src * mix).sum(axis=1).to_numpy()
        expected = (rate_given_src.T * mix).sum(axis=1)
        x = {
            "steady": math.log2((g["month"].nunique() + 1) / (_expected_cover(n, exp_month) + 1)),
            "phases": (
                math.log2((g["phase_key"].nunique() + 1) / (_expected_cover(n, exp_phase) + 1))
                if has_phase else 0.0
            ),
            "deliberate": _log2_ratio(g["deliberate"].sum(), n, expected["deliberate"]),
            "now": _log2_ratio(g["recent"].sum(), n, expected["recent"]),
            "exam": _log2_ratio(g["in_exam"].sum(), n, expected["in_exam"]),
            "trip": _log2_ratio(g["abroad"].sum(), n, expected["abroad"]),
        }
        x["exam"] -= DEADZONE
        x["trip"] -= DEADZONE
        for f, (lo, hi) in CLIPS.items():
            x[f] = float(min(hi, max(lo, x[f])))
        base = src_n / n_all
        present = [s for s in sources if mix[s] / base[s] >= LIFT_PRESENT]
        x["cross"] = len(present) / max(1, len(sources)) - 1.0
        x["utility"] = float(bool(utility.get(t, False))) if utility is not None else 0.0
        months = g["ts"].dt.year * 12 + g["ts"].dt.month
        summer = g[g["ts"].dt.month.isin([6, 7, 8])]
        rows[t] = {
            **x,
            "n_docs": n,
            "span_years": (months.max() - months.min() + 1) / 12.0,
            "exam_share": float(g["in_exam"].mean()),
            "trip_share": float(g["abroad"].mean()),
            "burst_share": _max_window_share(g["ts"], BURST_DAYS),
            "summer_share": float(summer["ts"].dt.year.value_counts().max() / n) if len(summer) else 0.0,
            "deliberate_share": float(g["deliberate"].mean()),
            "recent_share": float(g["recent"].mean()),
            "last_year": int(g["ts"].max().year),
            "sources": ",".join(present),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def score(features: pd.DataFrame, weights: dict | None = None) -> pd.Series:
    """``sigmoid(bias + w . x)`` per topic."""
    w = weights or FROZEN
    z = w["bias"] + sum(features[f] * w["weights"][f] for f in FEATURES)
    return 1.0 / (1.0 + np.exp(-z.astype(float)))


def _sources_phrase(present: list[str]) -> str:
    names = [SOURCE_NAMES.get(s, s) for s in present]
    if len(names) == 1:
        return f"only on {names[0]}"
    return "on " + (", ".join(names[:-1]) + " and " + names[-1])


def drivers(row: pd.Series, weights: dict | None = None) -> list[str]:
    """Short readable reasons for one topic's score, strongest first."""
    w = (weights or FROZEN)["weights"]
    row = pd.Series({**FACT_DEFAULTS, **row.to_dict()})
    c = {f: float(row[f]) * w[f] for f in FEATURES}
    out: list[tuple[float, str]] = []
    if row["burst_share"] >= 0.6:
        out.append((abs(c["steady"]) + 0.5, "one short burst"))
    elif row["summer_share"] >= 0.6:
        out.append((abs(c["steady"]) + 0.4, "one summer"))
    elif row["steady"] >= -0.5 and row["span_years"] >= 1.5:
        out.append((abs(c["steady"]) + 0.3, f"steady for {round(row['span_years'])} years"))
    elif row["steady"] < -1.0:
        out.append((abs(c["steady"]), "on and off"))
    if row["exam_share"] >= 0.5:
        out.append((abs(c["exam"]) + 0.2, "mostly during exams"))
    elif row["exam"] >= 1.0:
        out.append((abs(c["exam"]), "often during exams"))
    if row["trip_share"] >= 0.5:
        out.append((abs(c["trip"]) + 0.2, "mostly on a trip"))
    if row["recent_share"] < 0.05:
        out.append((abs(c["now"]), f"quiet since {row['last_year']}"))
    elif row["now"] >= -0.3:
        out.append((abs(c["now"]) + 0.1, "active now"))
    if row["deliberate"] >= 0.5:
        lead = "mostly" if row["deliberate_share"] >= 0.5 else "often"
        out.append((abs(c["deliberate"]), f"{lead} searched or liked"))
    elif row["deliberate"] <= -0.5:
        out.append((abs(c["deliberate"]), "mostly passive viewing"))
    present = [s for s in str(row["sources"]).split(",") if s]
    if present:
        out.append((abs(c["cross"]) + 0.05, _sources_phrase(present)))
    if row["utility"]:
        out.append((abs(c["utility"]), "a chore, not an interest"))
    return [p for _, p in sorted(out, key=lambda s: -s[0])]
