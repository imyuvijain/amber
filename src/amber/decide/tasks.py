"""Decision tasks on the Google and Instagram event log (sources v2), scored prequentially.

Frozen protocol (git tag ``decide-v2``, fixed before the first real run):

- **Tasks.** ``yt_channel_return``: after watching a YouTube video, does the person
  watch the same channel again within 7 days? ``ig_creator_return``: after liking an
  Instagram post, do they like the same creator again within 7 days?
  ``reels_feedback``: when they give Reels feedback on a creator, is it "see more"
  (1) or "not interested" (0)? Only events with a creator count as decisions.
- **Delayed labels.** A return label is known only once its window has closed:
  event ``i`` (time ``t_i``) becomes training data, for the model and for every
  label-based feature, only for decisions at times strictly after ``t_i + 7 days``.
  Feedback labels are known immediately. Counts of earlier events are visible at
  once. Rows sharing a timestamp never see each other.
- **Censoring.** A return decision whose window runs past the source's last event
  is not scored (its label is unknown) and never trains anything.
- **Ladder.** base (matured label rate) -> context (local hour, weekend, activity
  burst) -> life (exam window, abroad, life phase) -> memory (this creator's count,
  days since, distinct days, matured label rate, same-source other-kind count) ->
  recency (decayed count, last-24h count). Each rung is the v1 online logistic
  learner (``OnlineLogit``) on its columns.
- **Evaluation.** Holdout = the last 20% of scored decisions in time. Holdout rows
  are predicted prequentially like all others. Log loss, Brier, AUC, accuracy;
  95% CIs from a block bootstrap over local calendar days (2,000 resamples, seed 7);
  paired gaps between consecutive rungs and memory vs base.
- **Headline rule.** A task "works" only if memory beats both base and context with
  a CI lower bound above 0.
"""

import heapq
import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from amber.decide.life import Timeline, _home, annotate
from amber.decide.metrics import (
    N_BOOT,
    SEED,
    _DayBootstrap,
    _row_log_loss,
    accuracy,
    auc,
    brier,
)
from amber.decide.model import OnlineLogit, holdout_start

TRAIN_FRAC = 0.8
K_SMOOTH = 3.0
NEVER_DAYS = 365.0
GAP_CAP_MIN = 1440.0
BURST_S = 3600.0
DECAY_HALF_LIFE_DAYS = 7.0
RUNG_ORDER = ["base", "context", "life", "memory", "recency"]

CONTEXT_COLS = ["hour_sin", "hour_cos", "weekend", "gap_prev_log", "burst_log"]
MEMORY_COLS = ["c_n_log", "c_days_log", "c_ndays_log", "c_rate_s", "cross_n_log"]
RECENCY_COLS = ["c_dn_log", "c_24h_log"]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    source: str
    kinds: tuple[str, ...]
    label: str  # "return" or "feedback"
    horizon_days: float = 7.0
    positive_kind: str | None = None  # feedback: the kind that counts as 1


TASKS: dict[str, TaskSpec] = {
    "yt_channel_return": TaskSpec("yt_channel_return", "youtube", ("watch",), "return"),
    "ig_creator_return": TaskSpec("ig_creator_return", "instagram", ("like",), "return"),
    "reels_feedback": TaskSpec(
        "reels_feedback", "instagram", ("see_more", "not_interested"), "feedback",
        horizon_days=0.0, positive_kind="see_more",
    ),
}

QUESTIONS = {
    "yt_channel_return": "Watch the same YouTube channel again within 7 days",
    "ig_creator_return": "Like the same Instagram creator again within 7 days",
    "reels_feedback": 'Reels feedback: "see more" vs "not interested"',
}


def _seconds(ts: pd.Series) -> np.ndarray:
    return ((ts - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype=float)


def _labels(dec: pd.DataFrame, spec: TaskSpec, t: np.ndarray) -> np.ndarray:
    if spec.label == "feedback":
        return (dec["kind"] == spec.positive_kind).to_numpy(dtype=int)
    h = spec.horizon_days * 86400
    y = np.zeros(len(dec), dtype=int)
    for idx in dec.groupby("creator_id", sort=False).indices.values():
        times = t[idx]  # already ascending
        nxt = np.searchsorted(times, times, side="right")
        has = nxt < len(times)
        gap = np.where(has, times[np.minimum(nxt, len(times) - 1)] - times, np.inf)
        y[idx] = (gap <= h).astype(int)
    return y


def _life(src: pd.DataFrame, tl: Timeline | None) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Local-time and life columns, life column names and local date per event."""
    out = pd.DataFrame(index=src.index)
    if tl is None:
        local = src["ts"].dt.tz_convert("UTC")
        hour = local.dt.hour + local.dt.minute / 60
        weekday = local.dt.day_of_week
        day = local.dt.date
        out["in_exam"] = 0.0
        out["exam_near"] = 0.0
        out["abroad"] = 0.0
        phase_cols: list[str] = []
    else:
        ann = annotate(src, tl)
        hour = ann["local_hour"].astype(float) + src["ts"].dt.minute / 60
        weekday = ann["local_weekday"]
        day = ann["local_date"]
        out["in_exam"] = ann["in_exam"].astype(float)
        out["exam_near"] = 1.0 - ann["days_to_exam"].astype(float) / 60.0
        home_cc = [_home(d, tl)["country"] for d in ann["local_date"]]
        out["abroad"] = (ann["country"].to_numpy() != np.array(home_cc)).astype(float)
        names = sorted({p["name"] for p in tl.phases})
        phase_cols = [f"phase_{i}" for i in range(len(names))]
        for i, name in enumerate(names):
            out[f"phase_{i}"] = (ann["phase"] == name).astype(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["weekend"] = (weekday >= 5).astype(float)
    return out, ["in_exam", "exam_near", "abroad"] + phase_cols, pd.Series(list(day), index=src.index)


def build_task(events: pd.DataFrame, spec: TaskSpec, tl: Timeline | None = None) -> pd.DataFrame:
    """One row per decision: ``ts``, ``ready_ts``, ``y``, ``scored``, ``day`` and features.

    ``attrs["features"]`` lists every feature column and ``attrs["rungs"]`` maps each
    learner rung to its columns. All features use only strictly earlier events and
    only labels that have matured (see the module docstring).
    """
    src = events[events["source"] == spec.source].sort_values("ts", kind="stable")
    src = src.reset_index(drop=True)
    life, life_cols, local_day = _life(src, tl)
    is_dec = src["kind"].isin(spec.kinds) & src["creator_id"].notna()
    dec = src[is_dec]
    t_all = _seconds(src["ts"])
    t = t_all[is_dec.to_numpy()]
    y = _labels(dec, spec, t)
    h = spec.horizon_days * 86400
    ready = t + h
    end = t_all.max() if len(t_all) else 0.0
    scored = ready <= end if spec.label == "return" else np.ones(len(dec), dtype=bool)

    feats = _features(src, is_dec.to_numpy(), t_all, y, ready, scored, life)
    feats = feats[is_dec.to_numpy()].reset_index(drop=True)
    out = pd.DataFrame(
        {
            "ts": dec["ts"].to_numpy(),
            "ready_ts": pd.to_datetime(ready, unit="s", utc=True),
            "y": y,
            "scored": scored,
        }
    )
    out["ts"] = pd.to_datetime(out["ts"], utc=True)
    out["day"] = local_day[is_dec.to_numpy()].to_numpy()
    out = pd.concat([out, feats], axis=1)
    context = CONTEXT_COLS
    life_r = context + life_cols
    memory = life_r + MEMORY_COLS
    recency = memory + RECENCY_COLS
    out.attrs["features"] = recency
    out.attrs["rungs"] = {"context": context, "life": life_r, "memory": memory, "recency": recency}
    out.attrs["task"] = spec.name
    return out


def _features(src, is_dec, t, y_dec, ready_dec, scored_dec, life) -> pd.DataFrame:
    """One pass over every event of the source, in ts groups (see build_task)."""
    n = len(src)
    creators = src["creator_id"].astype(object).tolist()
    days = (t // 86400).astype(int)
    dec_idx = np.flatnonzero(is_dec)
    dec_pos = {int(i): j for j, i in enumerate(dec_idx)}
    decay = math.log(2) / (DECAY_HALF_LIFE_DAYS * 86400)
    log1p, exp = math.log1p, math.exp

    cols = CONTEXT_COLS[3:] + MEMORY_COLS + RECENCY_COLS
    out = np.zeros((n, len(cols)))
    # task-kind state per creator: [n, last_t, n_days, last_day, dn, dn_t]
    st: dict = {}
    recent: dict = {}  # creator -> list of task-event times (for 24h counts)
    cross: dict = {}  # creator -> count of other-kind events
    mat: dict = {}  # creator -> [matured n, matured positives]
    mat_n = mat_pos = 0
    pending: list = []  # heap of (ready_t, dec position)
    burst: list = []  # times of earlier task events, for the 1-hour count
    b_lo = 0
    last_task_t = None

    start = 0
    while start < n:
        now = t[start]
        stop = start + 1
        while stop < n and t[stop] == now:
            stop += 1
        while pending and pending[0][0] < now:
            _, j = heapq.heappop(pending)
            c = creators[dec_idx[j]]
            m = mat.setdefault(c, [0, 0])
            m[0] += 1
            m[1] += y_dec[j]
            mat_n += 1
            mat_pos += y_dec[j]
        g = (mat_pos + 1) / (mat_n + 2)
        while b_lo < len(burst) and burst[b_lo] < now - BURST_S:
            b_lo += 1
        n_burst = len(burst) - b_lo
        gap = GAP_CAP_MIN if last_task_t is None else min((now - last_task_t) / 60, GAP_CAP_MIN)

        for i in range(start, stop):
            if not is_dec[i]:
                continue
            c = creators[i]
            s = st.get(c)
            m = mat.get(c, [0, 0])
            rate = (m[1] + K_SMOOTH * g) / (m[0] + K_SMOOTH)
            if s is None:
                mem = (0.0, log1p(NEVER_DAYS), 0.0, rate)
                rec = (0.0, 0.0)
            else:
                cnt, last, nd, _, dn, dn_t = s
                mem = (log1p(cnt), log1p((now - last) / 86400), log1p(nd), rate)
                times = recent[c]
                n24 = len(times) - int(np.searchsorted(times, now - 86400, side="left"))
                rec = (log1p(dn * exp(-decay * (now - dn_t))), log1p(n24))
            out[i] = (log1p(gap), log1p(n_burst), *mem, log1p(cross.get(c, 0)), *rec)

        for i in range(start, stop):
            c = creators[i]
            if c is None or (isinstance(c, float) and math.isnan(c)):
                continue
            if not is_dec[i]:
                cross[c] = cross.get(c, 0) + 1
                continue
            s = st.get(c)
            if s is None:
                st[c] = [1, now, 1, days[i], 1.0, now]
                recent[c] = [now]
            else:
                s[0] += 1
                s[1] = now
                if days[i] != s[3]:
                    s[2] += 1
                    s[3] = days[i]
                s[4] = s[4] * exp(-decay * (now - s[5])) + 1
                s[5] = now
                recent[c].append(now)
            j = dec_pos[i]
            if scored_dec[j]:
                heapq.heappush(pending, (ready_dec[j], j))
        task_here = [i for i in range(start, stop) if is_dec[i]]
        if task_here:
            burst.extend([now] * len(task_here))
            last_task_t = now
        start = stop

    res = pd.DataFrame(out, columns=cols, index=src.index)
    res = pd.concat([life, res], axis=1)
    return res


def run_task_ladder(task: pd.DataFrame) -> pd.DataFrame:
    """Prequential predictions ``p_<rung>``; a row trains only after its label matures."""
    y = task["y"].to_numpy(dtype=bool)
    t = _seconds(task["ts"])
    ready = _seconds(task["ready_ts"])
    scored = task["scored"].to_numpy(dtype=bool)
    n = len(task)
    rungs = task.attrs["rungs"]
    models = {r: OnlineLogit(len(rungs[r])) for r in rungs}
    X = {r: task[rungs[r]].to_numpy(dtype=float) for r in rungs}
    p = {r: np.empty(n) for r in RUNG_ORDER}
    pending: list = []
    mat_n = mat_pos = 0
    start = 0
    while start < n:
        now = t[start]
        stop = start + 1
        while stop < n and t[stop] == now:
            stop += 1
        while pending and pending[0][0] < now:
            _, j = heapq.heappop(pending)
            for r, model in models.items():
                model.update(X[r][j], y[j])
            mat_n += 1
            mat_pos += y[j]
        for i in range(start, stop):
            p["base"][i] = (mat_pos + 1) / (mat_n + 2)
            for r, model in models.items():
                p[r][i] = model.predict(X[r][i])
        for i in range(start, stop):
            if scored[i]:
                heapq.heappush(pending, (ready[i], i))
        start = stop
    return pd.DataFrame(
        {f"p_{r}": np.clip(p[r], 1e-4, 1 - 1e-4) for r in RUNG_ORDER}, index=task.index
    )


def evaluate_task(
    task: pd.DataFrame, preds: pd.DataFrame, *, n_boot: int = N_BOOT, seed: int = SEED
) -> dict:
    """Holdout metrics on the last 20% of scored decisions, with day-block CIs."""
    keep = task["scored"].to_numpy(dtype=bool)
    sc = task[keep].reset_index(drop=True)
    pr = preds[keep].reset_index(drop=True)
    start = holdout_start(sc["ts"], TRAIN_FRAC)
    y = sc["y"].to_numpy(dtype=float)[start:]
    days = pd.factorize(sc["day"].iloc[start:])[0]
    boot = _DayBootstrap(days, n_boot=n_boot, seed=seed)
    cols = {r: pr[f"p_{r}"].to_numpy(dtype=float)[start:] for r in RUNG_ORDER}
    losses = {r: _row_log_loss(y, cols[r]) for r in RUNG_ORDER}
    rungs = {
        r: {
            "log_loss": float(losses[r].mean()),
            "brier": brier(y, cols[r]),
            "auc": auc(y, cols[r]),
            "accuracy": accuracy(y, cols[r]),
            "ll_ci": boot.ci(losses[r]),
        }
        for r in RUNG_ORDER
    }

    def gain(lo, hi):
        d = losses[lo] - losses[hi]
        return {"lower": lo, "higher": hi, "ll_gain": float(d.mean()), "ci": boot.ci(d)}

    gaps = [gain(a, b) for a, b in itertools.pairwise(RUNG_ORDER)]
    mvb = gain("base", "memory")
    mvc = gain("context", "memory")
    best = min(RUNG_ORDER, key=lambda r: rungs[r]["log_loss"])
    return {
        "task": task.attrs.get("task"),
        "n_decisions": len(task),
        "n_scored": len(sc),
        "n_holdout": len(y),
        "n_holdout_days": int(days.max() + 1) if len(days) else 0,
        "base_rate_holdout": float(y.mean()),
        "first": str(sc["ts"].iloc[0].date()),
        "holdout_from": str(sc["ts"].iloc[start].date()),
        "last": str(sc["ts"].iloc[-1].date()),
        "rungs": rungs,
        "gaps": gaps,
        "memory_vs_base": mvb,
        "memory_vs_context": mvc,
        "works": mvb["ci"][0] > 0 and mvc["ci"][0] > 0,
        "best": best,
    }


def fake_task_events(kind: str, seed: int = 0, n_days: int = 300) -> pd.DataFrame:
    """Synthetic YouTube watches. ``personal``: loyal, bingey; ``null``: iid uniform."""
    if kind not in ("personal", "null"):
        raise ValueError(kind)
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    n = int(n_days * 3)
    secs = np.sort(rng.uniform(0, n_days * 86400, n))
    if kind == "null":
        creators = rng.integers(0, 100, n)
    else:
        weights = 1.0 / np.arange(1, 301) ** 1.1
        weights /= weights.sum()
        creators = np.empty(n, dtype=int)
        for i in range(n):
            if i and rng.random() < 0.35:
                creators[i] = creators[i - 1]
            else:
                creators[i] = rng.choice(300, p=weights)
    return pd.DataFrame(
        {
            "ts": t0 + pd.to_timedelta(secs, unit="s"),
            "source": "youtube",
            "kind": "watch",
            "item_id": None,
            "creator_id": [f"c{c}" for c in creators],
            "text": None,
            "url": None,
            "from_ads": False,
        }
    )


def is_synthetic_events(events: pd.DataFrame) -> bool:
    """True when most creator ids or URLs come from ``fake_sources`` (they say "fake")."""
    text = pd.concat([events["creator_id"], events["url"]]).dropna().astype(str)
    return bool(len(text)) and text.str.lower().str.contains("fake").mean() > 0.5


def run_tasks(events: pd.DataFrame, tl: Timeline | None, *, n_boot: int = N_BOOT) -> list[dict]:
    """Evaluate every frozen task that has enough scored decisions (>= 200)."""
    results = []
    for spec in TASKS.values():
        task = build_task(events, spec, tl)
        if task["scored"].sum() < 200 or task.loc[task["scored"], "y"].nunique() < 2:
            results.append({"task": spec.name, "skipped": f"{int(task['scored'].sum())} scored decisions"})
            continue
        results.append(evaluate_task(task, run_task_ladder(task), n_boot=n_boot))
    return results


def _ci(ci) -> str:
    return f"{ci[0]:+.3f} to {ci[1]:+.3f}"


def render_tasks_md(results: list[dict]) -> str:
    """Aggregates-only Markdown: counts, dates, metrics. No creators, titles or URLs."""
    lines = [
        "# Sources v2: decisions from Google and Instagram exports",
        "",
        "One person's real exports. Every decision is predicted from strictly earlier data;",
        "a 7-day return label is used for training only after its 7 days have passed.",
        "Holdout = the last 20% of scored decisions; 95% CIs resample whole days.",
        "Protocol frozen as git tag `decide-v2` before this run.",
        "",
        "| Decision | Holdout | Base rate | AUC base | AUC context | AUC life | AUC memory | AUC recency | Works? |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "skipped" in r:
            lines.append(f"| {QUESTIONS[r['task']]} | skipped ({r['skipped']}) | | | | | | | |")
            continue
        a = [f"{r['rungs'][k]['auc']:.3f}" for k in RUNG_ORDER]
        lines.append(
            f"| {QUESTIONS[r['task']]} | {r['n_holdout']:,} ({r['holdout_from']} to {r['last']}) "
            f"| {r['base_rate_holdout']:.0%} | " + " | ".join(a) + f" | {'yes' if r['works'] else 'no'} |"
        )
    lines += ["", "## Log-loss gain of each rung over the one below (higher is better)", ""]
    lines += ["| Decision | " + " | ".join(f"{g} over {lo}" for lo, g in itertools.pairwise(RUNG_ORDER)) + " |"]
    lines += ["|---|" + "---|" * (len(RUNG_ORDER) - 1)]
    for r in results:
        if "skipped" in r:
            continue
        cells = [f"{g['ll_gain']:+.3f} ({_ci(g['ci'])})" for g in r["gaps"]]
        lines.append(f"| {QUESTIONS[r['task']]} | " + " | ".join(cells) + " |")
    lines += ["", "## Holdout log loss per rung (lower is better)", ""]
    lines += ["| Decision | " + " | ".join(RUNG_ORDER) + " |", "|---|" + "---|" * len(RUNG_ORDER)]
    for r in results:
        if "skipped" in r:
            continue
        lines.append(
            f"| {QUESTIONS[r['task']]} | "
            + " | ".join(f"{r['rungs'][k]['log_loss']:.3f}" for k in RUNG_ORDER) + " |"
        )
    lines += [
        "",
        '"Works" means memory beats both base and context with a CI lower bound above 0.',
        "",
    ]
    return "\n".join(lines)
