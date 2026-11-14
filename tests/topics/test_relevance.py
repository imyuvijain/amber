import numpy as np

from amber.topics.fake import fake_topic_docs
from amber.topics.relevance import FEATURES, FROZEN, drivers, score, topic_features


def _ranked(kind):
    docs = fake_topic_docs(kind)
    feats = topic_features(docs)
    return feats, score(feats)


def test_null_user_gets_flat_relevance():
    feats, rel = _ranked("null")
    assert len(rel) >= 10
    assert rel.max() - rel.min() < 0.1
    for f in ("steady", "deliberate", "now", "exam", "trip"):
        assert feats[f].abs().max() < 0.5, f


def test_planted_core_topic_wins_and_traps_rank_below_background():
    feats, rel = _ranked("planted")
    assert rel.idxmax() == "core"
    background = rel[[t for t in rel.index if t.startswith("bg")]]
    for trap in ("exam", "trip", "burst", "old", "gaming"):
        assert rel[trap] < background.median(), trap
    assert rel["gaming"] < background.min()


def test_drivers_explain_each_planted_topic():
    feats, _ = _ranked("planted")
    d = {t: " | ".join(drivers(feats.loc[t])) for t in feats.index}
    assert "steady for" in d["core"]
    assert "mostly searched or liked" in d["core"]
    assert "mostly during exams" in d["exam"]
    assert "mostly on a trip" in d["trip"]
    assert "one short burst" in d["burst"]
    assert "quiet since" in d["old"]
    assert "one summer" in d["gaming"] or "mostly during exams" in d["gaming"]


def test_features_and_weights_line_up():
    assert set(FROZEN["weights"]) == set(FEATURES)
    feats, _ = _ranked("planted")
    assert np.isfinite(feats[FEATURES].to_numpy()).all()
