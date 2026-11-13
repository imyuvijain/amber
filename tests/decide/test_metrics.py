import time

import numpy as np
import pandas as pd
import pytest

from amber.decide.metrics import (
    auc,
    brier,
    calibration_table,
    day_block_ci,
    evaluate,
    log_loss,
)

LADDER_COLS = ["p_base", "p_context", "p_memory", "p_recency"]


def test_log_loss_known():
    assert log_loss([1, 0], [0.8, 0.2]) == pytest.approx(0.2231, abs=1e-4)


def test_brier_known():
    assert brier([1, 0], [0.8, 0.2]) == pytest.approx(0.04)


def test_auc_perfect_and_degenerate():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auc([0, 0, 0], [0.1, 0.2, 0.3]) != auc([0, 0, 0], [0.1, 0.2, 0.3])  # NaN
    assert auc([1, 1], [0.5, 0.6]) != auc([1, 1], [0.5, 0.6])


def test_auc_ties_count_half():
    assert auc([0, 1], [0.5, 0.5]) == 0.5
    assert auc([1, 0, 1, 0], [0.9, 0.1, 0.3, 0.3]) == pytest.approx(0.875)


def test_log_loss_extreme_p_finite():
    assert np.isfinite(log_loss([1, 0], [0.0, 1.0]))


def test_calibration_table_drops_empty_bins():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.05, 0.06, 0.95, 0.99])
    t = calibration_table(y, p, bins=10)
    assert list(t.columns) == ["bin_low", "bin_high", "n", "mean_p", "skip_rate"]
    assert len(t) == 2
    assert t["n"].sum() == 4
    assert t["skip_rate"].tolist() == [0.0, 1.0]
    # p == 1.0 lands in the last bin
    assert calibration_table([1], [1.0])["n"].sum() == 1


def test_day_block_resamples_days():
    days = np.array([0] + [1] + [2] * 100)
    value = np.array([0.0, 1.0] + [0.5] * 100)

    def stat(idx):
        return float(value[idx].mean())

    lo, hi = day_block_ci(days, stat, n_boot=500, seed=7)
    assert lo < hi
    assert lo <= value.mean() <= hi
    # a row-level bootstrap would be much tighter than the day-block one
    rng = np.random.default_rng(0)
    rows = [value[rng.integers(0, len(value), len(value))].mean() for _ in range(500)]
    row_width = np.quantile(rows, 0.975) - np.quantile(rows, 0.025)
    assert hi - lo > row_width


def _holdout_inputs(n=600, holdout_frac=0.2, seed=0, all_keep=False):
    rng = np.random.default_rng(seed)
    skip = np.zeros(n, dtype=bool) if all_keep else rng.random(n) < 0.4
    days = np.arange(n) // 30
    p = rng.uniform(0.05, 0.95, size=(n, 4))
    preds = pd.DataFrame(p, columns=LADDER_COLS)
    start = int(n * (1 - holdout_frac))
    frozen = rng.uniform(0.05, 0.95, size=n - start)
    return preds, skip, days, frozen, start


def test_evaluate_all_keep_holdout_no_crash():
    preds, skip, days, frozen, start = _holdout_inputs(all_keep=True)
    r = evaluate(preds, skip, days, frozen, start=start)
    for rung in r["rungs"].values():
        assert np.isfinite(rung["log_loss"])
        assert np.isnan(rung["auc"])
    assert r["base_rate_holdout"] == 0.0


def test_evaluate_structure_and_types():
    preds, skip, days, frozen, start = _holdout_inputs()
    r = evaluate(preds, skip, days, frozen, start=start)
    assert r["n_total"] == 600 and r["n_holdout"] == 600 - start
    assert list(r["rungs"]) == ["base", "context", "memory", "recency"]
    assert [(g["lower"], g["higher"]) for g in r["gaps"]] == [
        ("base", "context"),
        ("context", "memory"),
        ("memory", "recency"),
    ]
    ho = slice(start, None)
    for name, rung in r["rungs"].items():
        assert rung["log_loss"] == pytest.approx(log_loss(skip[ho], preds[f"p_{name}"][ho]))
        lo, hi = rung["ll_ci"]
        assert lo <= rung["log_loss"] <= hi
        assert type(rung["log_loss"]) is float and type(lo) is float
    g = r["gaps"][0]
    assert g["ll_gain"] == pytest.approx(
        r["rungs"]["base"]["log_loss"] - r["rungs"]["context"]["log_loss"]
    )
    assert g["ci"][0] <= g["ci"][1]
    d = r["drift"]
    assert d["updating_ll"] == pytest.approx(r["rungs"]["memory"]["log_loss"])
    assert d["gain"] == pytest.approx(d["frozen_ll"] - d["updating_ll"])
    best = min(r["rungs"], key=lambda k: r["rungs"][k]["log_loss"])
    expected = calibration_table(skip[ho], preds[f"p_{best}"][ho])
    pd.testing.assert_frame_equal(r["calibration"], expected)


def test_evaluate_default_start_and_length_check():
    preds, skip, days, frozen, start = _holdout_inputs()
    r = evaluate(preds, skip, days, frozen)  # int(600 * 0.8) == 480
    assert r["n_holdout"] == 120
    with pytest.raises(ValueError):
        evaluate(preds, skip, days, frozen[:-1], start=start)


def test_evaluate_speed_large_holdout():
    n = 75_000
    preds, skip, days, frozen, start = _holdout_inputs(n=n)
    t0 = time.time()
    evaluate(preds, skip, days, frozen, start=start)
    assert time.time() - t0 < 30


def test_evaluate_memory_vs_base_gain():
    preds, skip, days, frozen, start = _holdout_inputs()
    r = evaluate(preds, skip, days, frozen, start=start)
    mvb = r["memory_vs_base"]
    assert mvb["ll_gain"] == pytest.approx(
        r["rungs"]["base"]["log_loss"] - r["rungs"]["memory"]["log_loss"]
    )
    assert mvb["ci"][0] <= mvb["ll_gain"] <= mvb["ci"][1]
    assert type(mvb["ll_gain"]) is float
