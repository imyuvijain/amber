import time

import numpy as np
import pandas as pd
import pytest

from amber.decide.features import CONTEXT_COLS, DECAY_COLS, MEMORY_COLS
from amber.decide.model import RUNGS, OnlineLogit, run_frozen, run_ladder

ALL_COLS = CONTEXT_COLS + MEMORY_COLS + DECAY_COLS
LADDER_COLS = ["p_base", "p_context", "p_memory", "p_recency"]


def _synthetic(n, seed=0):
    rng = np.random.default_rng(seed)
    feats = pd.DataFrame(rng.normal(size=(n, len(ALL_COLS))), columns=ALL_COLS)
    skip = (feats[CONTEXT_COLS[0]].to_numpy() > 0).astype(bool)
    ts = pd.Series(pd.date_range("2024-01-01", periods=n, freq="min", tz="UTC"))
    return feats, skip, ts


def test_logit_learns_separable():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(2000, 3))
    y = X[:, 0] > 0
    model = OnlineLogit(3)
    p = np.empty(len(y))
    for i in range(len(y)):
        p[i] = model.predict(X[i])
        model.update(X[i], bool(y[i]))
    tail = slice(1500, None)
    assert p[tail][y[tail]].mean() > 0.9
    assert p[tail][~y[tail]].mean() < 0.1


def test_logit_untrained_predicts_half():
    assert OnlineLogit(4).predict(np.array([3.0, -1.0, 0.0, 2.0])) == pytest.approx(0.5)


def test_ladder_same_ts_no_update_between():
    feats, skip, _ = _synthetic(6)
    skip[:] = False
    skip[2] = True
    ts = pd.Series(
        pd.to_datetime(
            ["2024-01-01 00:00", "2024-01-01 00:01", "2024-01-01 00:02",
             "2024-01-01 00:02", "2024-01-01 00:03", "2024-01-01 00:04"],
            utc=True,
        )
    )
    # rows 2 and 3 share a ts and identical features, row 2 has the extreme label
    feats.iloc[3] = feats.iloc[2]
    out = run_ladder(feats, skip, ts)
    for col in LADDER_COLS:
        assert out[col].iloc[2] == out[col].iloc[3]
    # the update did land afterwards
    assert out["p_context"].iloc[4] != pytest.approx(out["p_context"].iloc[3])


def test_ladder_shapes_and_bounds():
    feats, skip, ts = _synthetic(300)
    out = run_ladder(feats, skip, ts)
    assert len(out) == 300
    assert list(out.columns) == LADDER_COLS
    assert out.index.equals(feats.index)
    assert np.isfinite(out.to_numpy()).all()
    assert out.to_numpy().min() >= 1e-4
    assert out.to_numpy().max() <= 1 - 1e-4


def test_ladder_clips_extreme_base_rate():
    feats, _, ts = _synthetic(3)
    out = run_ladder(feats, np.zeros(3, dtype=bool), ts)
    assert out["p_base"].min() >= 1e-4


def test_base_rate_rung():
    feats, _, ts = _synthetic(3)
    out = run_ladder(feats, np.array([True, True, False]), ts)
    np.testing.assert_allclose(out["p_base"], [0.5, 2 / 3, 0.75])


def test_base_rate_same_ts_group_sees_no_labels():
    feats, _, _ = _synthetic(3)
    ts = pd.Series(pd.to_datetime(["2024-01-01"] * 2 + ["2024-01-02"], utc=True))
    out = run_ladder(feats, np.array([True, True, False]), ts)
    np.testing.assert_allclose(out["p_base"], [0.5, 0.5, 0.75])


def test_rungs_definition():
    assert RUNGS["base"] is None
    assert RUNGS["context"] == CONTEXT_COLS
    assert RUNGS["memory"] == CONTEXT_COLS + MEMORY_COLS
    assert RUNGS["recency"] == CONTEXT_COLS + MEMORY_COLS + DECAY_COLS


def test_ladder_rejects_unsorted_ts():
    feats, skip, ts = _synthetic(5)
    with pytest.raises(ValueError):
        run_ladder(feats, skip, ts.iloc[::-1].reset_index(drop=True))


def test_ladder_beats_base_on_signal():
    feats, skip, ts = _synthetic(3000)
    out = run_ladder(feats, skip, ts)

    def ll(p):
        p = np.asarray(p)[1000:]
        y = skip[1000:]
        return -np.mean(y * np.log(p) + (~y) * np.log(1 - p))

    assert ll(out["p_context"]) < ll(out["p_base"]) - 0.2


def test_frozen_length_and_no_updates():
    feats, skip, ts = _synthetic(1000)
    p = run_frozen(feats, skip, ts, "context")
    assert p.shape == (200,)
    assert p.min() >= 1e-4 and p.max() <= 1 - 1e-4
    # holdout labels must not matter: flipping them leaves predictions unchanged
    flipped = skip.copy()
    flipped[800:] = ~flipped[800:]
    np.testing.assert_array_equal(p, run_frozen(feats, flipped, ts, "context"))
    # but train labels do
    flipped = skip.copy()
    flipped[:800] = ~flipped[:800]
    assert not np.allclose(p, run_frozen(feats, flipped, ts, "context"))


def test_frozen_split_boundary_and_base():
    feats, skip, ts = _synthetic(101)
    assert len(run_frozen(feats, skip, ts, "context", train_frac=0.8)) == 101 - 80
    p = run_frozen(feats, skip, ts, "base")
    assert p.shape == (101 - 80,)
    np.testing.assert_allclose(p, np.clip((skip[:80].sum() + 1) / 82, 1e-4, 1 - 1e-4))


def test_frozen_unknown_rung():
    feats, skip, ts = _synthetic(20)
    with pytest.raises(KeyError):
        run_frozen(feats, skip, ts, "nope")


def test_ladder_speed_50k():
    feats, skip, ts = _synthetic(50_000)
    t0 = time.time()
    run_ladder(feats, skip, ts)
    assert time.time() - t0 < 10  # about 15 us per row per rung, so 300k rows is roughly 15 s
