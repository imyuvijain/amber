"""Fake users for the relevance score, with topics already assigned (no embedding).

- ``null``: every topic is spread like all activity (random times, sources and
  actions), so relevance should come out flat.
- ``planted``: background topics like ``null`` plus a steady, deliberate,
  cross-platform ``core`` topic and five traps (``exam``, ``trip``, ``burst``,
  ``old``, and a passive, exam-and-summer ``gaming``).
"""

import numpy as np
import pandas as pd

SEED = 7
START, END = pd.Timestamp("2022-01-01", tz="UTC"), pd.Timestamp("2025-12-31", tz="UTC")
SOURCES = {"youtube": 0.5, "google": 0.3, "instagram": 0.2}
DELIBERATE_P = {"youtube": 0.1, "google": 0.7, "instagram": 0.6}
KINDS = {  # source -> (deliberate kind, passive kind)
    "youtube": ("like", "watch"),
    "google": ("search", "visit"),
    "instagram": ("like", "view_post"),
}
EXAMS = [((3, 1), (3, 21)), ((6, 1), (6, 14)), ((11, 15), (12, 5))]
TRIP = (pd.Timestamp("2024-07-10", tz="UTC"), pd.Timestamp("2024-07-28", tz="UTC"))


def exam_windows() -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    return [
        (pd.Timestamp(y, a[0], a[1], tz="UTC"), pd.Timestamp(y, b[0], b[1], 23, tz="UTC"))
        for y in range(START.year, END.year + 1)
        for a, b in EXAMS
    ]


def _uniform(rng, n, lo=START, hi=END):
    span = (hi - lo).total_seconds()
    return [lo + pd.Timedelta(seconds=float(s)) for s in rng.uniform(0, span, n)]


def _in_windows(rng, n, windows):
    lengths = np.array([(b - a).total_seconds() for a, b in windows])
    pick = rng.choice(len(windows), n, p=lengths / lengths.sum())
    return [windows[i][0] + pd.Timedelta(seconds=float(rng.uniform(0, lengths[i]))) for i in pick]


def _events(rng, topic, ts, sources=None, deliberate_p=None):
    n = len(ts)
    src = rng.choice(list(SOURCES), n, p=list(SOURCES.values())) if sources is None else np.array(
        [sources] * n if isinstance(sources, str) else rng.choice(sources, n)
    )
    p = np.array([DELIBERATE_P[s] if deliberate_p is None else deliberate_p for s in src])
    dl = rng.random(n) < p
    return pd.DataFrame(
        {
            "ts": ts,
            "source": src,
            "kind": [KINDS[s][0 if d else 1] for s, d in zip(src, dl)],
            "weight": np.where(dl, 3.0, 1.0),
            "deliberate": dl,
            "topic": topic,
        }
    )


def _life(df: pd.DataFrame) -> pd.DataFrame:
    wins = exam_windows()
    df["in_exam"] = [any(a <= t <= b for a, b in wins) for t in df["ts"]]
    df["abroad"] = [TRIP[0] <= t <= TRIP[1] for t in df["ts"]]
    df["phase"] = [f"y{t.year - START.year + 1}" for t in df["ts"]]
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def fake_topic_docs(kind: str = "planted", seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_bg = 12 if kind == "null" else 8
    parts = [_events(rng, f"bg{i:02d}", _uniform(rng, int(rng.integers(150, 600)))) for i in range(n_bg)]
    if kind == "planted":
        y = lambda *a: pd.Timestamp(*a, tz="UTC")
        summer = [(y(2023, 6, 1), y(2023, 8, 31))]
        parts += [
            _events(rng, "core", _uniform(rng, 600), deliberate_p=0.8),
            _events(rng, "exam", _in_windows(rng, 300, exam_windows()), sources="youtube"),
            _events(rng, "trip", _in_windows(rng, 200, [TRIP])),
            _events(rng, "burst", _in_windows(rng, 200, [(y(2023, 2, 1), y(2023, 2, 21))])),
            _events(rng, "old", _uniform(rng, 300, y(2022, 1, 1), y(2022, 12, 31))),
            _events(
                rng, "gaming",
                _in_windows(rng, 240, exam_windows()) + _in_windows(rng, 160, summer),
                sources="youtube", deliberate_p=0.02,
            ),
        ]
    elif kind != "null":
        raise ValueError(f"unknown fake user {kind!r}")
    return _life(pd.concat(parts, ignore_index=True))
