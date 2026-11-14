import numpy as np
import pandas as pd
import pytest

from amber.decide.life import Timeline
from amber.decide.tasks import (
    TASKS,
    TaskSpec,
    build_task,
    evaluate_task,
    fake_task_events,
    run_task_ladder,
)

DAY = pd.Timedelta(days=1)
T0 = pd.Timestamp("2025-01-01 12:00", tz="UTC")
RET = TaskSpec("t_return", "youtube", ("watch",), "return", horizon_days=7)


def ev(rows, source="youtube"):
    """rows: (days after T0, kind, creator)."""
    df = pd.DataFrame(
        {
            "ts": [T0 + d * DAY for d, _, _ in rows],
            "source": source,
            "kind": [k for _, k, _ in rows],
            "item_id": None,
            "creator_id": [c for _, _, c in rows],
            "text": None,
            "url": None,
            "from_ads": False,
        }
    )
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def test_frozen_tasks():
    assert set(TASKS) == {"yt_channel_return", "ig_creator_return", "reels_feedback"}
    assert TASKS["yt_channel_return"].horizon_days == 7
    assert TASKS["reels_feedback"].label == "feedback"


def test_return_label_and_censoring():
    df = ev([(0, "watch", "A"), (3, "watch", "A"), (20, "watch", "A"), (40, "watch", "B")])
    t = build_task(df, RET)
    assert t["y"].tolist() == [1, 0, 0, 0]
    # data ends at day 40: the day-40 event's 7-day window is not observed
    assert t["scored"].tolist() == [True, True, True, False]
    assert (t["ready_ts"] == t["ts"] + 7 * DAY).all()


def test_return_ignores_other_kinds_and_missing_creators():
    df = ev([(0, "watch", "A"), (1, "like", "A"), (2, "watch", None), (30, "watch", "C")])
    t = build_task(df, RET)
    assert len(t) == 2  # the like and the creator-less watch are not decisions
    assert t["y"].tolist() == [0, 0]


def test_feedback_label_is_immediate():
    spec = TASKS["reels_feedback"]
    df = ev(
        [(0, "see_more", "A"), (1, "not_interested", "B"), (2, "like", "A")],
        source="instagram",
    )
    t = build_task(df, spec)
    assert t["y"].tolist() == [1, 0]
    assert (t["ready_ts"] == t["ts"]).all()
    assert t["scored"].all()


def test_same_timestamp_rows_do_not_see_each_other():
    df = ev([(0, "watch", "A"), (0, "watch", "A"), (1, "watch", "A")])
    t = build_task(df, RET)
    assert t["c_n_log"].tolist()[:2] == [0.0, 0.0]
    assert t["c_n_log"].iloc[2] == pytest.approx(np.log1p(2))


def test_cross_kind_memory_counts_other_kinds_of_same_source():
    df = ev([(0, "like", "A"), (1, "like", "A"), (2, "watch", "A")])
    t = build_task(df, RET)
    assert t["cross_n_log"].iloc[0] == pytest.approx(np.log1p(2))
    assert t["c_n_log"].iloc[0] == 0.0


def test_base_rate_waits_for_labels_to_mature():
    rows = [(d, "watch", "A") for d in range(30)]
    t = build_task(ev(rows), RET)
    p = run_task_ladder(t)["p_base"].to_numpy()
    # first label (day 0) is known only after day 7: days 0..7 see no labels
    assert np.allclose(p[:8], 0.5)
    assert p[8] == pytest.approx(2 / 3)  # one positive matured


def _random_events(seed=0, n=400):
    rng = np.random.default_rng(seed)
    days = np.sort(rng.uniform(0, 120, n))
    creators = rng.choice(list("ABCDEFG"), n, p=[0.4, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03])
    kinds = rng.choice(["watch", "like"], n, p=[0.8, 0.2])
    return ev(list(zip(days, kinds, creators)))


@pytest.mark.parametrize("cut_day", [30, 61, 95])
def test_prefix_invariance_no_look_ahead(cut_day):
    """Predictions before T are identical whether or not data after T exists."""
    tl = Timeline(None, [{"from": pd.Timestamp("2000-01-01").date(), "tz": "Asia/Tokyo", "country": "JP"}])
    full = _random_events()
    cut = T0 + cut_day * DAY
    short = full[full["ts"] < cut].reset_index(drop=True)
    a = build_task(full, RET, tl)
    b = build_task(short, RET, tl)
    pa = run_task_ladder(a)
    pb = run_task_ladder(b)
    k = len(b)
    assert (a["ts"].iloc[:k].to_numpy() == b["ts"].to_numpy()).all()
    cols = [c for c in a.attrs["features"]]
    np.testing.assert_allclose(a[cols].iloc[:k].to_numpy(), b[cols].to_numpy())
    np.testing.assert_allclose(pa.iloc[:k].to_numpy(), pb.to_numpy())


def test_life_columns_from_timeline():
    tl = Timeline(
        None,
        [{"from": pd.Timestamp("2000-01-01").date(), "tz": "Asia/Tokyo", "country": "JP"}],
        phases=[{"from": pd.Timestamp("2024-01-01").date(), "to": pd.Timestamp("2025-12-31").date(), "name": "x"}],
        exams=[{"from": pd.Timestamp("2025-01-02").date(), "to": pd.Timestamp("2025-01-05").date(), "name": "e"}],
    )
    t = build_task(ev([(0, "watch", "A"), (2, "watch", "A"), (30, "watch", "A")]), RET, tl)
    assert t["in_exam"].tolist() == [0.0, 1.0, 0.0]
    assert t["phase_0"].tolist() == [1.0, 1.0, 1.0]
    # 12:00 UTC is 21:00 in Tokyo
    assert t["hour_sin"].iloc[0] == pytest.approx(np.sin(2 * np.pi * 21 / 24))
    assert "phase_0" in t.attrs["rungs"]["life"]
    assert "phase_0" not in t.attrs["rungs"]["context"]


def test_evaluate_scores_only_scored_holdout_rows():
    t = build_task(fake_task_events("personal", seed=1), RET)
    preds = run_task_ladder(t)
    res = evaluate_task(t, preds, n_boot=200)
    assert 0 < res["n_holdout"] < t["scored"].sum()
    assert res["n_holdout"] == pytest.approx(0.2 * t["scored"].sum(), abs=5)
    assert set(res["rungs"]) == {"base", "context", "life", "memory", "recency"}
    assert [g["higher"] for g in res["gaps"]] == ["context", "life", "memory", "recency"]
    lo, hi = res["rungs"]["base"]["ll_ci"]
    assert lo <= res["rungs"]["base"]["log_loss"] <= hi


def test_personal_user_memory_beats_base():
    t = build_task(fake_task_events("personal", seed=2), RET)
    res = evaluate_task(t, run_task_ladder(t), n_boot=300)
    assert res["memory_vs_base"]["ci"][0] > 0
    assert res["rungs"]["memory"]["auc"] > 0.65


def test_null_user_memory_does_not_help():
    """Every creator is equally likely and draws are independent: the past says nothing."""
    t = build_task(fake_task_events("null", seed=2, n_days=1500), RET)
    res = evaluate_task(t, run_task_ladder(t), n_boot=300)
    assert res["memory_vs_base"]["ci"][0] < 0.005
    assert not res["works"]
    assert abs(res["rungs"]["memory"]["auc"] - 0.5) < 0.05
