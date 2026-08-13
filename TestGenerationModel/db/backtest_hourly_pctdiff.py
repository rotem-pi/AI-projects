"""Backtest HourlyPctDiffTest vs plain PctDiffTest on real prod series.

Protocol per series (day-hour averages, last 90 days, from auto_tests_hourly.csv):
  1. Calibrate: for each test type, find the smallest tolerance (var1) whose
     breach rate on the real series is <= FPR_TARGET. That tolerance IS the
     test's sensitivity floor: deviations smaller than it are invisible.
  2. Inject: multiply single points by (1 + delta) and check detection with
     the calibrated tolerance, using bounds computed from clean history.

Both tests use the same window (var2) and the hourly test uses 6h buckets.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.brain.metric_tests.hourly_pct_diff_test import HourlyPctDiffTest  # noqa: E402
from app.brain.metric_tests.pct_diff_test import PctDiffTest  # noqa: E402
from app.models.metric_history_model import MetricHistory  # noqa: E402

FPR_TARGET = 0.01
TOLERANCE_GRID = list(range(1, 51))
WINDOW = 3
BUCKET_HOURS = 6
INJECT_DELTAS = [0.3, 0.5, 1.0]
INJECT_POINTS = 30
WARMUP = 30
SEED = 1337


def build_history(sdf: pd.DataFrame) -> MetricHistory:
    sdf = sdf.sort_values(["day", "hr"])
    pits = [d + pd.Timedelta(hours=int(h)) for d, h in zip(sdf["day"], sdf["hr"])]
    return MetricHistory(
        metric_id=int(sdf["metric_id"].iloc[0]),
        metric_type="cnt",
        metric_values=[float(v) for v in sdf["avg_val"]],
        app_pits=pits,
    )


def breach_flags(test, mh: MetricHistory) -> np.ndarray:
    preds = test.predict_all_pits(mh)
    flags = []
    for pred, val in zip(preds[WARMUP:], mh.metric_values[WARMUP:]):
        flags.append(pred is not None and not pred.is_passed(val))
    return np.asarray(flags)


def calibrate(make_test, mh: MetricHistory) -> int | None:
    for tol in TOLERANCE_GRID:
        if breach_flags(make_test(tol), mh).mean() <= FPR_TARGET:
            return tol
    return None


def detection_rate(make_test, tol: int, mh: MetricHistory, delta: float,
                   rng: np.random.Generator) -> float:
    test = make_test(tol)
    preds = test.predict_all_pits(mh)
    n = len(mh.metric_values)
    idx = rng.choice(np.arange(WARMUP, n), size=min(INJECT_POINTS, n - WARMUP),
                     replace=False)
    hits = 0
    for i in idx:
        injected = mh.metric_values[i] * (1 + delta)
        pred = preds[i]
        hits += int(pred is not None and not pred.is_passed(injected))
    return hits / len(idx)


def main() -> None:
    hourly = pd.read_csv(HERE / "auto_tests_hourly.csv", parse_dates=["day"])
    hourly["asset_value"] = hourly["asset_value"].fillna("")
    res = pd.read_csv(HERE / "auto_tests_hour_seasonality.csv")
    res["asset_value"] = res["asset_value"].fillna("")
    res = res[res["p"].notna()].copy()
    res["robust"] = res["robust"].fillna(False).astype(bool)

    rng = np.random.default_rng(SEED)
    make_flat = lambda tol: PctDiffTest(var1=tol, var2=WINDOW)  # noqa: E731
    make_hod = lambda tol: HourlyPctDiffTest(  # noqa: E731
        var1=tol, var2=WINDOW, var3=BUCKET_HOURS)

    rows = []
    for _, r in res.iterrows():
        sdf = hourly[(hourly["metric_id"] == r["metric_id"])
                     & (hourly["asset_value"] == r["asset_value"])]
        if len(sdf) < WARMUP * 2:
            continue
        mh = build_history(sdf)
        tol_flat = calibrate(make_flat, mh)
        tol_hod = calibrate(make_hod, mh)
        row = {"metric_id": r["metric_id"], "robust": r["robust"],
               "tol_flat": tol_flat, "tol_hod": tol_hod}
        if tol_flat is not None:
            for delta in INJECT_DELTAS:
                row[f"det_flat_{delta}"] = detection_rate(
                    make_flat, tol_flat, mh, delta, rng)
        if tol_hod is not None:
            for delta in INJECT_DELTAS:
                row[f"det_hod_{delta}"] = detection_rate(
                    make_hod, tol_hod, mh, delta, rng)
        rows.append(row)

    bt = pd.DataFrame(rows)
    bt.to_csv(HERE / "backtest_results.csv", index=False)

    for label, grp in [("hour-seasonal", bt[bt["robust"]]),
                       ("non-seasonal", bt[~bt["robust"]])]:
        if grp.empty:
            continue
        flat_ok = grp["tol_flat"].notna()
        hod_ok = grp["tol_hod"].notna()
        print(f"\n=== {label} series (n={len(grp)}) ===")
        print(f"calibratable at {FPR_TARGET:.0%} FPR within var1<=50: "
              f"flat {flat_ok.sum()}, hourly {hod_ok.sum()}, "
              f"ONLY hourly {(hod_ok & ~flat_ok).sum()}, "
              f"ONLY flat {(flat_ok & ~hod_ok).sum()}, "
              f"neither {(~flat_ok & ~hod_ok).sum()}")

        only_hod = grp[hod_ok & ~flat_ok]
        if len(only_hod):
            print(f"ONLY-hourly series: median tolerance "
                  f"{only_hod['tol_hod'].median():.0f}%, detection of +50%: "
                  f"{only_hod['det_hod_0.5'].mean():.0%}")

        both = grp[flat_ok & hod_ok]
        if len(both):
            print(f"both calibratable (n={len(both)}): "
                  f"flat median tol {both['tol_flat'].median():.0f}%, "
                  f"hourly {both['tol_hod'].median():.0f}%; hourly tighter on "
                  f"{(both['tol_hod'] < both['tol_flat']).mean():.0%}")
            for delta in INJECT_DELTAS:
                f = both[f"det_flat_{delta}"].mean()
                h = both[f"det_hod_{delta}"].mean()
                print(f"  detection +{delta:.0%}: flat {f:.0%} vs hourly {h:.0%}")


if __name__ == "__main__":
    main()
