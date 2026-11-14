"""Build topics from the event log and write them under ``<data dir>/topics/``.

Outputs (all private, git-ignored with the rest of ``data/``):

- ``topics.parquet``: one row per topic: id, name, keywords, features, facts,
  frozen relevance and drivers.
- ``doc_topics.parquet``: ts, source, kind, topic_id per document (no text).
- ``centroids.npy`` and ``embeddings.npz`` (cache).

``summary`` holds counts only and is safe to print.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from amber.decide.life import _d, _home, annotate, load_timeline
from amber.topics.docs import build_docs
from amber.topics.extract import MIN_DOCS, extract_topics
from amber.topics.relevance import drivers, score, topic_features


def load_trips(path: Path) -> list[tuple]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return [(_d(t["from"]), _d(t.get("to", t["from"]))) for t in raw.get("trips") or []]


def life_columns(events: pd.DataFrame, life_path: Path | None) -> pd.DataFrame:
    """Add in_exam, abroad and phase (all off without a life file)."""
    if life_path is None or not Path(life_path).exists():
        return events.assign(in_exam=False, abroad=False, phase=None)
    tl = load_timeline(life_path)
    out = annotate(events, tl)
    trips = load_trips(life_path)
    home = [_home(d, tl)["country"] for d in out["local_date"]]
    on_trip = [any(a <= d <= b for a, b in trips) for d in out["local_date"]]
    out["abroad"] = [(c != h) or t for c, h, t in zip(out["country"], home, on_trip)]
    return out


def build_topics(events: pd.DataFrame, encoder, out_dir: Path, life_path: Path | None = None,
                 min_docs: int = MIN_DOCS) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    docs = build_docs(life_columns(events, life_path))
    if docs.empty:
        raise ValueError("no events with usable text")
    model = extract_topics(docs, encoder, min_docs=min_docs)
    ids = model.topics["topic_id"].to_numpy()
    docs["topic"] = [ids[t] if t >= 0 else None for t in model.doc_topic]
    meta = model.topics.set_index("topic_id")
    feats = topic_features(docs, utility=meta["utility"].to_dict())
    table = meta.join(feats.drop(columns=["utility", "n_docs"]), how="inner")
    table["relevance"] = score(table)
    table["drivers"] = [drivers(table.loc[t]) for t in table.index]
    table = table.sort_values("relevance", ascending=False)
    table.to_parquet(out_dir / "topics.parquet")
    docs[["ts", "source", "kind", "topic"]].rename(columns={"topic": "topic_id"}).to_parquet(
        out_dir / "doc_topics.parquet"
    )
    np.save(out_dir / "centroids.npy", model.centroids)
    if hasattr(encoder, "save"):
        encoder.save()
    return {
        "documents": len(docs),
        "unique texts": int(docs["text"].nunique()),
        "topics": len(table),
        "assigned": f"{(docs['topic'].notna()).mean():.0%}",
        "named by label": int((table["named_by"] == "label").sum()),
        "utility topics": int(table["utility"].sum()),
        "life context": "on" if life_path is not None and Path(life_path).exists() else "off",
    }
