"""Detection/FPR per window N, split by series cadence (points per day),
since N counts points: N=14 is two weeks for a daily series but half a day
for an hourly one.

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/window_sweep_by_cadence.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

pts = pd.read_parquet(HERE / "window_sweep_points.parquet")

src = pd.read_parquet(HERE / "all_labeled_points.parquet")
src = src[~src.duplicated(subset=["metric_id", "pit"], keep="first")]
span = src.groupby("metric_id")["pit"].agg(["min", "max", "size"])
days = ((span["max"] - span["min"]).dt.total_seconds() / 86400).clip(lower=1)
freq = span["size"] / days
cadence = pd.cut(
    freq,
    bins=[0, 0.5, 4, float("inf")],
    labels=["sparse (<0.5/d)", "daily-ish (0.5-4/d)", "hf (>4/d)"],
)
pts["cadence"] = pts["metric_id"].map(cadence)

for name, sub in pts.groupby("cadence", observed=True):
    n_series = sub["metric_id"].nunique()
    anom = sub[sub["is_anomaly"]]
    norm = sub[~sub["is_anomaly"]]
    det = anom.groupby("window")["flagged"].agg(["sum", "size", "mean"])
    fpr = norm.groupby("window")["flagged"].mean().mul(100).round(2)
    tbl = pd.DataFrame(
        {
            "detected": det["sum"].astype(int),
            "anomalies": det["size"].astype(int),
            "detection_pct": (100 * det["mean"]).round(1),
            "fpr_pct": fpr,
        }
    )
    print(f"\n== {name}: {n_series} series ==")
    print(tbl.to_string())
