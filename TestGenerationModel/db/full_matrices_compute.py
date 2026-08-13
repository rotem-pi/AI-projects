"""Compute the B (metric-level) and C (run-level) conversion matrices on the
FULL tested fleet: every enabled-auto-tested series, last 30 days, both
methods replayed on identical deduped runs. Guardrail as deployed
(fallback -> old behavior)."""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from stage4_final_eval import (  # noqa: E402
    FALLBACK_MIN_FLAGS, FALLBACK_RATE, final_band,
)
from visual_side_by_side import EVAL_DAYS, incumbent_bounds  # noqa: E402

START = time.time()

tests = pd.read_csv(HERE / "full_tests.csv")
tests["asset_value"] = tests["asset_value"].fillna("")
tests = tests.drop_duplicates(["metric_id", "asset_value"]).set_index(
    ["metric_id", "asset_value"])
print(f"tests: {len(tests)} ({time.time() - START:.0f}s)", flush=True)

vals = pd.read_csv(HERE / "full_values.csv", parse_dates=["app_pit"])
vals["asset_value"] = vals["asset_value"].fillna("")
print(f"values: {len(vals):,} rows ({time.time() - START:.0f}s)", flush=True)

run_matrix = np.zeros((2, 2), dtype=np.int64)
metric_matrix = np.zeros((2, 2), dtype=np.int64)
n_series = n_skipped = 0

for key, s in vals.groupby(["metric_id", "asset_value"], sort=False):
    if key not in tests.index:
        n_skipped += 1
        continue
    t_row = tests.loc[key]
    ts = pd.DatetimeIndex(s["app_pit"])
    y = s["metric_value"].astype(float).values
    if len(y) < 20:
        n_skipped += 1
        continue
    eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
    t_dict = {"metric_id": key[0], "asset_value": key[1],
              "test_type": t_row["test_type"], "var1": t_row["var1"],
              "var2": t_row["var2"], "var3": t_row["var3"],
              "metric_type": t_row["metric_type"]}
    try:
        ilo, ihi = incumbent_bounds(pd.Series(t_dict), list(ts), y)
    except Exception:
        n_skipped += 1
        continue
    nlo, nhi = final_band(ts, y)
    ok = eval_mask & ~np.isnan(ilo) & ~np.isnan(nlo)
    if ok.sum() < 5:
        n_skipped += 1
        continue
    old_flags = ((y < ilo) | (y > ihi))[ok]
    new_flags = ((y < nlo) | (y > nhi))[ok]
    if new_flags.mean() > FALLBACK_RATE and new_flags.sum() >= FALLBACK_MIN_FLAGS:
        new_flags = old_flags
    n_series += 1
    o, n = old_flags.astype(int), new_flags.astype(int)
    run_matrix[0][0] += int(((o == 0) & (n == 0)).sum())
    run_matrix[0][1] += int(((o == 0) & (n == 1)).sum())
    run_matrix[1][0] += int(((o == 1) & (n == 0)).sum())
    run_matrix[1][1] += int(((o == 1) & (n == 1)).sum())
    metric_matrix[int(o.any())][int(n.any())] += 1
    if n_series % 10000 == 0:
        print(f"{n_series} series ({time.time() - START:.0f}s)", flush=True)

print(f"\nseries evaluated: {n_series:,}; skipped: {n_skipped:,}")

print(f"\n=== B. METRICS WITH >=1 ANOMALY LAST MONTH (FULL DATA) ===")
print(f"{'':22} {'new: quiet':>14} {'new: >=1 anomaly':>17}")
print(f"{'old: quiet':22} {metric_matrix[0][0]:>14,} {metric_matrix[0][1]:>17,}")
print(f"{'old: >=1 anomaly':22} {metric_matrix[1][0]:>14,} {metric_matrix[1][1]:>17,}")

total = run_matrix.sum()
print(f"\n=== C. RUN LEVEL (FULL DATA, {total:,} runs) ===")
print(f"{'':22} {'new: not flagged':>20} {'new: flagged':>16}")
print(f"{'old: not flagged':22} {run_matrix[0][0]:>12,} "
      f"({run_matrix[0][0] / total:.2%}) {run_matrix[0][1]:>9,} "
      f"({run_matrix[0][1] / total:.2%})")
print(f"{'old: flagged':22} {run_matrix[1][0]:>12,} "
      f"({run_matrix[1][0] / total:.2%}) {run_matrix[1][1]:>9,} "
      f"({run_matrix[1][1] / total:.2%})")

np.save(HERE / "full_run_matrix.npy", run_matrix)
np.save(HERE / "full_metric_matrix.npy", metric_matrix)
print(f"\ntotal time: {time.time() - START:.0f}s")
