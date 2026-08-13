"""Compare the EXISTING configured auto Range tests against PctDiff variants
on hour-seasonal series: empirical FPR and injected-anomaly detection."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.brain.metric_tests.hourly_pct_diff_test import HourlyPctDiffTest  # noqa: E402
from app.brain.metric_tests.pct_diff_test import PctDiffTest  # noqa: E402
from backtest_hourly_pctdiff import (  # noqa: E402
    FPR_TARGET, INJECT_DELTAS, INJECT_POINTS, WARMUP, WINDOW, BUCKET_HOURS,
    build_history, breach_flags, calibrate,
)

SEED = 1337


def range_flags_and_detection(var1: float, var2: float, mh, rng) -> dict:
    lo, hi = float(var1), float(var2)
    vals = np.asarray(mh.metric_values[WARMUP:])
    fpr = float(np.mean((vals < lo) | (vals > hi)))
    out = {"range_fpr": fpr}
    n = len(mh.metric_values)
    idx = rng.choice(np.arange(WARMUP, n), size=min(INJECT_POINTS, n - WARMUP),
                     replace=False)
    for delta in INJECT_DELTAS:
        injected = np.asarray([mh.metric_values[i] * (1 + delta) for i in idx])
        out[f"range_det_{delta}"] = float(np.mean((injected < lo) | (injected > hi)))
    return out


def pct_detection(make_test, tol: int, mh, delta: float, rng) -> float:
    test = make_test(tol)
    preds = test.predict_all_pits(mh)
    n = len(mh.metric_values)
    idx = rng.choice(np.arange(WARMUP, n), size=min(INJECT_POINTS, n - WARMUP),
                     replace=False)
    hits = sum(
        int(preds[i] is not None
            and not preds[i].is_passed(mh.metric_values[i] * (1 + delta)))
        for i in idx)
    return hits / len(idx)


def main() -> None:
    hourly = pd.read_csv(HERE / "auto_tests_hourly.csv", parse_dates=["day"])
    hourly["asset_value"] = hourly["asset_value"].fillna("")
    res = pd.read_csv(HERE / "auto_tests_hour_seasonality.csv")
    res["asset_value"] = res["asset_value"].fillna("")
    res = res[res["p"].notna()].copy()
    res["robust"] = res["robust"].fillna(False).astype(bool)
    seasonal_range = res[res["robust"] & (res["test_type"] == "Range")]

    rng = np.random.default_rng(SEED)
    make_flat = lambda tol: PctDiffTest(var1=tol, var2=WINDOW)  # noqa: E731
    make_hod = lambda tol: HourlyPctDiffTest(  # noqa: E731
        var1=tol, var2=WINDOW, var3=BUCKET_HOURS)

    rows = []
    for _, r in seasonal_range.iterrows():
        sdf = hourly[(hourly["metric_id"] == r["metric_id"])
                     & (hourly["asset_value"] == r["asset_value"])]
        if len(sdf) < WARMUP * 2 or pd.isna(r["var1"]) or pd.isna(r["var2"]):
            continue
        mh = build_history(sdf)
        row = {"metric_id": r["metric_id"]}
        row.update(range_flags_and_detection(r["var1"], r["var2"], mh, rng))
        for name, mk in [("flat", make_flat), ("hod", make_hod)]:
            tol = calibrate(mk, mh)
            row[f"{name}_tol"] = tol
            if tol is not None:
                for delta in INJECT_DELTAS:
                    row[f"{name}_det_{delta}"] = pct_detection(mk, tol, mh, delta, rng)
        rows.append(row)

    bt = pd.DataFrame(rows)
    bt.to_csv(HERE / "backtest_vs_range.csv", index=False)
    print(f"seasonal series with existing auto Range test: {len(bt)}")
    print(f"existing Range empirical FPR: median {bt['range_fpr'].median():.2%}, "
          f"mean {bt['range_fpr'].mean():.2%}")
    for delta in INJECT_DELTAS:
        print(f"\ndetection of +{delta:.0%} single-point anomalies:")
        print(f"  existing Range test:        {bt[f'range_det_{delta}'].mean():.0%}")
        flat_ok = bt["flat_tol"].notna()
        hod_ok = bt["hod_tol"].notna()
        print(f"  flat PctDiff @1% FPR:       "
              f"{bt.loc[flat_ok, f'flat_det_{delta}'].mean():.0%} "
              f"(calibratable on {flat_ok.sum()}/{len(bt)})")
        print(f"  hourly PctDiff @1% FPR:     "
              f"{bt.loc[hod_ok, f'hod_det_{delta}'].mean():.0%} "
              f"(calibratable on {hod_ok.sum()}/{len(bt)})")


if __name__ == "__main__":
    main()
