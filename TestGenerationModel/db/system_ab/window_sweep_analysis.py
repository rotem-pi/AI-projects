"""Consistency checks on the window sweep: per-dataset scores per N, and
paired flip analysis (which anomalies/normals change verdict) for N=3 vs
larger windows, with a McNemar-style significance read.

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/window_sweep_analysis.py
"""

from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent

pts = pd.read_parquet(HERE / "window_sweep_points.parquet")

print("== per-dataset detection% (rows: dataset, cols: window) ==")
det = (
    pts[pts["is_anomaly"]]
    .pivot_table(index="dataset", columns="window", values="flagged", aggfunc="mean")
    .mul(100)
    .round(1)
)
counts = pts[pts["is_anomaly"]].groupby("dataset").size() // pts["window"].nunique()
det.insert(0, "n_anom", counts)
print(det.to_string())

print("\n== per-dataset FPR% ==")
fpr = (
    pts[~pts["is_anomaly"]]
    .pivot_table(index="dataset", columns="window", values="flagged", aggfunc="mean")
    .mul(100)
    .round(2)
)
print(fpr.to_string())

base = pts[pts["window"] == 3].set_index(["metric_id", "pit"])["flagged"]
for n in [5, 7, 9, 14]:
    cand = pts[pts["window"] == n].set_index(["metric_id", "pit"])["flagged"]
    lab = pts[pts["window"] == n].set_index(["metric_id", "pit"])["is_anomaly"]
    both = pd.DataFrame({"base": base, "cand": cand, "is_anomaly": lab}).dropna()
    for is_anom, name in [(True, "anomalies"), (False, "normals")]:
        sub = both[both["is_anomaly"] == is_anom]
        gained = int((~sub["base"] & sub["cand"]).sum())
        lost = int((sub["base"] & ~sub["cand"]).sum())
        p = binomtest(gained, gained + lost).pvalue if gained + lost else float("nan")
        print(
            f"N=3 vs N={n} on {name}: +{gained} newly flagged, "
            f"-{lost} no longer flagged (McNemar p={p:.3f})"
        )

print("\n== false-flag concentration: series driving FPR (N=14) ==")
n14 = pts[(pts["window"] == 14) & ~pts["is_anomaly"]]
per_series = n14.groupby("metric_id")["flagged"].agg(["sum", "size"])
top = per_series.sort_values("sum", ascending=False).head(10)
print(top.to_string())
print(
    f"share of false flags from top-10 series: "
    f"{100 * top['sum'].sum() / n14['flagged'].sum():.0f}%"
)
