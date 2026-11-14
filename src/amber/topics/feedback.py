"""Veto and boost taps on topics: stored as labels, used to tune the relevance weights.

Each tap is appended to a JSON log under ``data/topics/`` with the feature vector the
topic had when it was tapped, so labels survive a rebuild of the topics. The latest
action per topic wins; ``clear`` undoes it.

Labels: veto -> y=0, boost -> y=1 (weight 1). Topics that were shown above a vetoed
topic and left alone are weak positives (y=1, weight ``IMPLICIT_W``).
Tuning is a MAP logistic regression with a Gaussian prior (``LAMBDA``) centred on the
frozen weights.
"""

import copy
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from amber.topics.relevance import FEATURES, FROZEN, drivers, score

ACTIONS = ("veto", "boost", "clear")
IMPLICIT_W = 0.3
LAMBDA = 5.0


class FeedbackStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.events: list[dict] = (
            json.loads(self.path.read_text(encoding="utf-8"))["events"] if self.path.exists() else []
        )

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"events": self.events}, indent=1), encoding="utf-8")
        tmp.replace(self.path)

    def record(self, topic_id: str, action: str, features: dict, shown: list[str] | None = None,
               features_of: pd.DataFrame | None = None) -> None:
        if action not in ACTIONS:
            raise ValueError(f"unknown action {action!r}")
        shown = list(shown or [])
        above = shown[: shown.index(topic_id)] if topic_id in shown else []
        snap = (
            {t: {f: float(features_of.loc[t, f]) for f in FEATURES} for t in above if t in features_of.index}
            if features_of is not None else {}
        )
        self.events.append(
            {
                "ts": datetime.now(UTC).isoformat(timespec="seconds"),
                "topic_id": topic_id,
                "action": action,
                "features": {f: float(features.get(f, 0.0)) for f in FEATURES},
                "above": snap,
            }
        )
        self._save()

    def clear(self, topic_id: str) -> None:
        self.record(topic_id, "clear", {})

    def _latest(self) -> dict[str, dict]:
        latest: dict[str, dict] = {}
        for e in self.events:
            latest[e["topic_id"]] = e
        return {t: e for t, e in latest.items() if e["action"] != "clear"}

    def overrides(self) -> dict[str, str]:
        return {t: e["action"] for t, e in self._latest().items()}

    def labels(self) -> list[dict]:
        latest = self._latest()
        out = [
            {"topic_id": t, "y": int(e["action"] == "boost"), "w": 1.0, "x": e["features"]}
            for t, e in latest.items()
        ]
        seen = set(latest)
        for e in latest.values():
            if e["action"] != "veto":
                continue
            for t, x in e["above"].items():
                if t not in seen:
                    seen.add(t)
                    out.append({"topic_id": t, "y": 1, "w": IMPLICIT_W, "x": x})
        return out


def tune(labels: list[dict], prior: dict = FROZEN, lam: float = LAMBDA) -> dict:
    """MAP weights given labels; the frozen weights when there are none."""
    if not labels:
        return copy.deepcopy(prior)
    X = np.array([[1.0] + [lab["x"][f] for f in FEATURES] for lab in labels])
    y = np.array([lab["y"] for lab in labels], dtype=float)
    c = np.array([lab["w"] for lab in labels], dtype=float)
    th0 = np.array([prior["bias"]] + [prior["weights"][f] for f in FEATURES])

    def loss(th):
        z = X @ th
        p = 1 / (1 + np.exp(-z))
        ll = np.sum(c * (np.logaddexp(0, z) - y * z))
        g = X.T @ (c * (p - y))
        d = th - th0
        return ll + lam / 2 * d @ d, g + lam * d

    th = minimize(loss, th0, jac=True, method="L-BFGS-B").x
    return {"bias": float(th[0]), "weights": {f: float(v) for f, v in zip(FEATURES, th[1:])}}


def ranked_topics(meta: pd.DataFrame, feats: pd.DataFrame, store: FeedbackStore) -> dict:
    """The veto screen's list: boosts first, then by tuned relevance; vetoes muted."""
    weights = tune(store.labels())
    rel = score(feats, weights)
    over = store.overrides()
    order = sorted(feats.index, key=lambda t: (over.get(t) != "boost", -rel[t], str(t)))
    topics, muted = [], []
    for t in order:
        item = {
            "id": str(t),
            "name": str(meta.loc[t, "name"]),
            "relevance": round(float(rel[t]), 3),
            "drivers": drivers(feats.loc[t], weights),
            "state": over.get(t),
            "n_docs": int(meta.loc[t, "n_docs"]),
        }
        (muted if over.get(t) == "veto" else topics).append(item)
    return {"topics": topics, "muted": muted, "n_labels": len(store.labels()), "weights": weights}
