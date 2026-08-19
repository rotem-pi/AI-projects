"""Classify every labeled point in the anomaly datasets by GuardedBand
qualification, mirroring app.brain.anomaly.guarded_band_calibration:

- const_carveout: >=40 prior points and the entire prior history is constant
- gb_qualified:   >=10 prior points with a valid prediction (index >= 1)
                  inside the 56 days before the point
- insufficient_history: everything else

Writes the qualified samples to guarded_band_qualified_samples.csv and
prints the distribution.

Run from backend/:  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/gb_qualified.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

from app.brain.anomaly.guarded_band_calibration import (
    CAL_WINDOW_DAYS,
    MIN_CAL_POINTS,
    MIN_CONST_LOOKBACK,
)

HERE = Path(__file__).resolve().parent

df = pd.read_parquet(HERE / "all_labeled_points.parquet")
df = df[df["is_anomaly"].notna()].copy()
df["is_anomaly"] = df["is_anomaly"].astype(bool)

parts = []
for (ds, mid), g in df.groupby(["dataset", "metric_id"]):
    g = g.sort_values("pit").reset_index(drop=True)
    pits = g["pit"].values
    vals = g["value"].to_numpy(dtype=float)
    n = len(g)
    idx = np.arange(n)

    # calibration points: strictly before the sample, inside the 56d window,
    # with a defined trailing-median prediction (i.e. not the series' first point)
    lo = np.searchsorted(pits, pits - np.timedelta64(CAL_WINDOW_DAYS, "D"), side="left")
    cal_pts = idx - np.maximum(lo, 1)  # index 0 has no prediction

    # const carve-out over the entire causal (prior) history
    run_min = np.concatenate(([np.inf], np.minimum.accumulate(vals)[:-1]))
    run_max = np.concatenate(([-np.inf], np.maximum.accumulate(vals)[:-1]))
    is_const_prior = (idx >= MIN_CONST_LOOKBACK) & (run_min == run_max)

    g["cal_points_56d"] = cal_pts
    g["prior_points"] = idx
    g["qualification"] = np.select(
        [is_const_prior, cal_pts >= MIN_CAL_POINTS],
        ["const_carveout", "gb_qualified"],
        default="insufficient_history",
    )
    parts.append(g)

out = pd.concat(parts, ignore_index=True)

# exact duplicate (metric_id, pit) pairs across datasets (agoda overlap)
dups = out.duplicated(subset=["metric_id", "pit"], keep="first")
print(f"cross-dataset duplicate (metric_id, pit) pairs dropped: {int(dups.sum())}")
out = out[~dups]

qualified = out[out["qualification"] == "gb_qualified"].drop(
    columns=["has_bounds", "has_test"]
)
qualified.to_csv(HERE / "guarded_band_qualified_samples.csv", index=False)

pd.set_option("display.width", 200)
print(f"\ntotal labeled points: {len(out)}")
print(out["qualification"].value_counts().to_string())

print("\n== qualification x label ==")
print(pd.crosstab(out["qualification"], out["is_anomaly"]).to_string())

q = qualified
print(f"\n== gb_qualified: {len(q)} points, {q['metric_id'].nunique()} metrics ==")
print("\nby dataset:")
print(
    q.groupby("dataset")
    .agg(
        points=("pit", "size"),
        metrics=("metric_id", "nunique"),
        anomalies=("is_anomaly", "sum"),
    )
    .assign(pct_anomaly=lambda d: (100 * d["anomalies"] / d["points"]).round(2))
    .to_string()
)
print("\nby metric_group_type:")
print(
    q.groupby("metric_group_type")
    .agg(points=("pit", "size"), metrics=("metric_id", "nunique"), anomalies=("is_anomaly", "sum"))
    .sort_values("points", ascending=False)
    .to_string()
)
print("\nby metric_type (top 15 by points):")
print(
    q.groupby("metric_type")
    .agg(points=("pit", "size"), metrics=("metric_id", "nunique"), anomalies=("is_anomaly", "sum"))
    .sort_values("points", ascending=False)
    .head(15)
    .to_string()
)
print("\ncal_points_56d percentiles (qualified points):")
print(q["cal_points_56d"].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).to_string())
