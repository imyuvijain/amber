import numpy as np
import pandas as pd

from amber.topics.feedback import FeedbackStore, ranked_topics, tune
from amber.topics.relevance import FEATURES, FROZEN


def _feats(n=8, seed=7):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(0, 0.5, (n, len(FEATURES))), columns=FEATURES)
    df.index = [f"t{i}" for i in range(n)]
    return df


def test_no_labels_keeps_frozen_weights():
    w = tune([])
    assert w == FROZEN


def test_vetoing_exam_heavy_topics_makes_exam_weight_more_negative(tmp_path):
    store = FeedbackStore(tmp_path / "fb.json")
    feats = _feats(40)
    feats["exam"] = np.linspace(0, 3, 40)
    for t in feats.index[-15:]:
        store.record(t, "veto", feats.loc[t].to_dict(), shown=list(feats.index))
    w = tune(store.labels())
    assert w["weights"]["exam"] < FROZEN["weights"]["exam"]


def test_store_latest_action_wins_and_clear_removes(tmp_path):
    store = FeedbackStore(tmp_path / "fb.json")
    f = _feats().loc["t0"].to_dict()
    store.record("t0", "veto", f, shown=["t0"])
    store.record("t0", "boost", f, shown=["t0"])
    assert store.overrides() == {"t0": "boost"}
    store.record("t1", "veto", f, shown=["t1"])
    store.clear("t1")
    assert store.overrides() == {"t0": "boost"}
    # persisted
    assert FeedbackStore(tmp_path / "fb.json").overrides() == {"t0": "boost"}


def test_implicit_keeps_are_weak_positives(tmp_path):
    store = FeedbackStore(tmp_path / "fb.json")
    feats = _feats(4)
    store.record("t2", "veto", feats.loc["t2"].to_dict(), shown=["t0", "t1", "t2", "t3"], features_of=feats)
    labels = store.labels()
    by = {(lab["topic_id"], lab["y"]): lab["w"] for lab in labels}
    assert by[("t2", 0)] == 1.0
    assert by[("t0", 1)] == 0.3 and by[("t1", 1)] == 0.3
    assert ("t3", 1) not in by


def test_ranked_topics_puts_boosts_first_and_mutes_vetoes(tmp_path):
    store = FeedbackStore(tmp_path / "fb.json")
    feats = _feats(5)
    meta = pd.DataFrame({"name": [f"n{i}" for i in range(5)], "n_docs": 50}, index=feats.index)
    store.record("t4", "boost", feats.loc["t4"].to_dict(), shown=[])
    store.record("t0", "veto", feats.loc["t0"].to_dict(), shown=[])
    out = ranked_topics(meta, feats, store)
    ids = [t["id"] for t in out["topics"]]
    assert ids[0] == "t4"
    assert "t0" not in ids and [m["id"] for m in out["muted"]] == ["t0"]
    assert all({"id", "name", "relevance", "drivers", "state"} <= set(t) for t in out["topics"])
