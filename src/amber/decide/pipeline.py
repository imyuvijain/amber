"""Spotify export in, decision metrics and reports out."""

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from amber.decide.fake import is_synthetic
from amber.decide.features import build_features
from amber.decide.metrics import evaluate
from amber.decide.model import holdout_start, run_frozen, run_ladder
from amber.decide.report import period, render_private, render_public
from amber.decide.spotify import load_export

MIN_PLAYS = 50
TRAIN_FRAC = 0.8


def _check_tz(tz: str) -> None:
    try:
        ZoneInfo(tz)
    except (KeyError, ValueError, OSError) as err:
        raise ValueError(f"unknown timezone: {tz!r}") from err


def load_plays(export_dir: Path) -> pd.DataFrame:
    """``load_export`` plus a clear error when there is too little to evaluate."""
    df = load_export(export_dir)
    if df.empty:
        raise ValueError(f"no plays found under {export_dir}")
    if len(df) < MIN_PLAYS:
        raise ValueError(f"need at least {MIN_PLAYS} plays to evaluate, found {len(df)}")
    return df


def run_pipeline(export_dir: Path, out_dir: Path, tz: str = "UTC") -> dict:
    """Evaluate the four-rung ladder on one export and write the reports.

    Writes ``decisions.parquet`` (plays plus predictions), ``report.md`` (private,
    names artists) and ``public.md`` (aggregates only) to ``out_dir`` and returns
    the ``evaluate`` result plus ``source``, ``synthetic`` and the loader's
    ``n_near_duplicates`` and ``n_overlap`` counts.
    """
    _check_tz(tz)
    df = load_plays(Path(export_dir))
    feats = build_features(df, tz=tz)
    skip = df["skip"].to_numpy(dtype=bool)
    preds = run_ladder(feats, skip, df["ts"])
    frozen = run_frozen(feats, skip, df["ts"], "memory", TRAIN_FRAC)
    local_day = df["ts"].dt.tz_convert(tz).dt.date
    days = pd.factorize(local_day)[0]
    start = holdout_start(df["ts"], TRAIN_FRAC)
    result = evaluate(preds, skip, np.asarray(days), frozen, start=start)
    result["source"] = str(df["source"].iloc[0])
    result["synthetic"] = is_synthetic(df)
    result["n_near_duplicates"] = int(df.attrs.get("n_near_duplicates", 0))
    result["n_overlap"] = int(df.attrs.get("n_overlap", 0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.concat([df, preds], axis=1).to_parquet(out_dir / "decisions.parquet")
    (out_dir / "report.md").write_text(render_private(result, df, tz), encoding="utf-8")
    (out_dir / "public.md").write_text(
        render_public(result, period(df, tz)), encoding="utf-8"
    )
    return result
