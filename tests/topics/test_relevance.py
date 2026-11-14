import numpy as np
import pytest

from amber.topics.fake import fake_topic_docs
from amber.topics.relevance import FEATURES, FROZEN, drivers, score, topic_features


def _ranked(kind):
    docs = fake_topic_docs(kind)
    feats = topic_features(docs)
    return feats, score(feats)


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_null_user_gets_flat_relevance(seed):
    feats = topic_features(fake_topic_docs("null", seed=seed))
    rel = score(feats)
    assert len(rel) >= 10
    assert rel.max() - rel.min() < 0.12
    for f in ("steady", "deliberate", "now", "exam", "trip"):
        assert feats[f].abs().max() < 0.5, f
    planted = score(topic_features(fake_topic_docs("planted", seed=seed)))
    assert planted.max() - planted.min() > 0.5


def test_planted_core_topic_wins_and_traps_rank_below_background():
    _feats, rel = _ranked("planted")
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


@pytest.mark.parametrize("unit", ["ns", "us", "ms", "s"])
def test_burst_window_ignores_timestamp_resolution(unit):
    import pandas as pd

    from amber.topics.relevance import _max_window_share

    ts = pd.Series(pd.date_range("2022-01-01", periods=48, freq="MS", tz="UTC")).astype(f"datetime64[{unit}, UTC]")
    assert _max_window_share(ts, 60) < 0.1  # one event a month: no 60-day window holds 10%


def test_drivers_skip_baseline_phrases():
    """A topic that looks like everything else gets no 'active now' or deliberate phrase."""
    feats = topic_features(fake_topic_docs("null"))
    for t in feats.index:
        d = drivers(feats.loc[t])
        assert "active now" not in d
        assert not any("searched or liked" in x or "passive" in x for x in d)
