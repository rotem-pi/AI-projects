"""Finish the crashed tail of auto_tests_hour_seasonality.py from saved CSVs:
alert-rate comparison and configured-Range-width vs hour-bucket-width."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent

merged = pd.read_csv(HERE / "auto_tests_hour_seasonality.csv")
hourly = pd.read_csv(HERE / "auto_tests_hourly.csv", parse_dates=["day"])
merged["asset_value"] = merged["asset_value"].fillna("")
hourly["asset_value"] = hourly["asset_value"].fillna("")

testable = merged[merged["p"].notna()].copy()
testable["robust"] = testable["robust"].fillna(False).astype(bool)
testable["alert_rate"] = (testable["total_alerts"].astype(float)
                          / testable["total_metrics"].astype(float))

a = testable[testable["robust"]]["alert_rate"].values
b = testable[~testable["robust"]]["alert_rate"].values
mw = stats.mannwhitneyu(a, b)
print("alert rate (total_alerts / total_metrics):")
print(f"  hour-seasonal ({len(a)}):    median {np.median(a):.4f}  mean {a.mean():.4f}  "
      f"p90 {np.percentile(a, 90):.4f}  share with any alert {np.mean(a > 0):.1%}")
print(f"  non-seasonal ({len(b)}):     median {np.median(b):.4f}  mean {b.mean():.4f}  "
      f"p90 {np.percentile(b, 90):.4f}  share with any alert {np.mean(b > 0):.1%}")
print(f"  Mann-Whitney p = {mw.pvalue:.4g}")

tot_a = testable[testable["robust"]]["total_alerts"].sum()
tot_all = testable["total_alerts"].sum()
print(f"  share of ALL alerts in sample carried by seasonal metrics: "
      f"{tot_a / tot_all:.1%} (they are {testable['robust'].mean():.1%} of tests)")

rng_tests = testable[(testable["test_type"] == "Range") & testable["robust"]]
ratios = []
for _, t in rng_tests.iterrows():
    sdf = hourly[(hourly["metric_id"] == t["metric_id"])
                 & (hourly["asset_value"] == t["asset_value"])]
    if sdf.empty or pd.isna(t["var1"]) or pd.isna(t["var2"]):
        continue
    width_cfg = float(t["var2"]) - float(t["var1"])
    buckets = sdf.assign(bucket=sdf["hr"] // 6).groupby("bucket")["avg_val"]
    widths = [np.percentile(g, 95) - np.percentile(g, 5)
              for _, g in buckets if len(g) >= 10]
    if widths and np.mean(widths) > 0 and width_cfg > 0:
        ratios.append(width_cfg / np.mean(widths))

if ratios:
    r = np.array(ratios)
    print(f"\nconfigured Range width vs mean per-hour-bucket p5-p95 width "
          f"(robust seasonal Range tests, n={len(r)}):")
    print(f"  median {np.median(r):.2f}x  p25 {np.percentile(r, 25):.2f}x  "
          f"p75 {np.percentile(r, 75):.2f}x  max {r.max():.1f}x")
    print(f"  tests with configured band > 2x the hour-aware need: {np.mean(r > 2):.1%}")
