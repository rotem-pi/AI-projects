"""Sweep the GuardedBand trailing-median window N on the labeled datasets.

For each N, patches TRAILING_WINDOW in both the calibration module and the
test module, then replays the exact shadow flow from shadow_replay.py:
weekly Monday-00:00 recalibration (real calibrate_guarded_band, so the
tolerances re-fit themselves to the forecaster N produces), honest forward
scoring of the following week through the real GuardedBandTest.

This answers "if we shipped N=k instead of 3, what would the full solution
do" - not "which N forecasts best with today's tolerances".

Scores per N, on the GuardedBand-qualified labeled points:
detection %, FPR %, tolerance quantiles / share at the 10% floor, and the
per-calibration rate guardrail (>=3 flags and >3% of runs).

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/window_sweep.py
"""

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

import app.brain.anomaly.guarded_band_calibration as gbc
import app.brain.metric_tests.guarded_band_test as gbt
from app.brain.anomaly.guarded_band_calibration import LifetimeStats
from app.brain.anomaly.guarded_band_shadow import (
    FALLBACK_MIN_FLAGS,
    FALLBACK_RATE,
    FETCH_DAYS,
    LIFETIME_DAYS,
)
from app.brain.metric_tests.test_object_factory import create_test_instance
from app.models.metric_history_model import MetricHistory

HERE = Path(__file__).resolve().parent
EVAL_DAYS = 7
WINDOWS = [1, 2, 3, 4, 5, 7, 9, 14, 21]


def set_window(n: int) -> None:
    gbc.TRAILING_WINDOW = n
    gbt.TRAILING_WINDOW = n


def week_start(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def replay_series(g: pd.DataFrame, window: int) -> tuple[list[dict], list[dict]]:
    g = g.sort_values("pit").reset_index(drop=True)
    pits = [p.to_pydatetime() for p in g["pit"]]
    vals = g["value"].astype(float).to_numpy()
    pits_np = g["pit"].values

    point_rows: list[dict] = []
    cal_rows: list[dict] = []
    first, last = g["pit"].iloc[0], g["pit"].iloc[-1]
    as_of = week_start(first) + pd.Timedelta(days=7)
    while as_of <= last:
        as_of_dt = as_of.to_pydatetime()
        hist_lo = np.searchsorted(
            pits_np, np.datetime64(as_of_dt - timedelta(days=FETCH_DAYS)), side="left"
        )
        hist_hi = np.searchsorted(pits_np, np.datetime64(as_of_dt), side="right")
        life_lo = np.searchsorted(
            pits_np,
            np.datetime64(as_of_dt - timedelta(days=LIFETIME_DAYS)),
            side="left",
        )
        eval_hi = np.searchsorted(
            pits_np, np.datetime64(as_of_dt + timedelta(days=EVAL_DAYS)), side="right"
        )
        if eval_hi == hist_hi:
            as_of += pd.Timedelta(days=7)
            continue

        calibrated = None
        if hist_hi - life_lo > 0:
            life_vals = vals[life_lo:hist_hi]
            calibrated = gbc.calibrate_guarded_band(
                app_pits=pits[hist_lo:hist_hi],
                values=list(vals[hist_lo:hist_hi]),
                lifetime=LifetimeStats(
                    count=int(hist_hi - life_lo),
                    min_value=float(life_vals.min()),
                    max_value=float(life_vals.max()),
                ),
                as_of=as_of_dt,
            )
        if calibrated is not None:
            test = create_test_instance(
                {
                    "test_type": calibrated.test_type,
                    "var1": calibrated.var1,
                    "var2": calibrated.var2,
                    "var3": calibrated.var3,
                }
            )
            ctx_lo = max(0, hist_hi - window)
            mh = MetricHistory(
                metric_id=int(g["metric_id"].iloc[0]),
                metric_type="",
                metric_values=list(vals[ctx_lo:eval_hi]),
                app_pits=pits[ctx_lo:eval_hi],
            )
            predictions = test.predict_all_pits(mh)
            n_runs = n_flagged = 0
            for j in range(hist_hi, eval_hi):
                pred = predictions[j - ctx_lo]
                flagged = not pred.is_passed(float(vals[j]))
                n_runs += 1
                n_flagged += flagged
                point_rows.append(
                    {
                        "dataset": g["dataset"].iloc[0],
                        "metric_id": g["metric_id"].iloc[0],
                        "pit": g["pit"].iloc[j],
                        "is_anomaly": bool(g["is_anomaly"].iloc[j]),
                        "test_type": calibrated.test_type,
                        "flagged": flagged,
                    }
                )
            cal_rows.append(
                {
                    "test_type": calibrated.test_type,
                    "var1": calibrated.var1,
                    "var2": calibrated.var2,
                    "n_runs": n_runs,
                    "n_flagged": n_flagged,
                }
            )
        as_of += pd.Timedelta(days=7)
    return point_rows, cal_rows


def main() -> None:
    df = pd.read_parquet(HERE / "all_labeled_points.parquet")
    df = df[df["is_anomaly"].notna()].copy()
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df = df[~df.duplicated(subset=["metric_id", "pit"], keep="first")]
    groups = [g for _, g in df.groupby(["dataset", "metric_id"])]

    qualified = pd.read_csv(
        HERE / "guarded_band_qualified_samples.csv", parse_dates=["pit"]
    )
    qkeys = set(zip(qualified["metric_id"], qualified["pit"]))

    results = []
    point_frames = []
    for n in WINDOWS:
        t0 = time.monotonic()
        set_window(n)
        points_all: list[dict] = []
        cals_all: list[dict] = []
        for g in groups:
            p, c = replay_series(g, n)
            points_all.extend(p)
            cals_all.extend(c)
        points = pd.DataFrame(points_all)
        cals = pd.DataFrame(cals_all)

        points["qualified"] = [
            (m, p) in qkeys for m, p in zip(points["metric_id"], points["pit"])
        ]
        gb_q = points[points["qualified"] & (points["test_type"] == "GuardedBand")]
        anom, norm = gb_q[gb_q["is_anomaly"]], gb_q[~gb_q["is_anomaly"]]

        gb_cals = cals[cals["test_type"] == "GuardedBand"]
        row = {
            "window": n,
            "points": len(gb_q),
            "anomalies": len(anom),
            "detected": int(anom["flagged"].sum()),
            "detection_pct": round(100 * anom["flagged"].mean(), 1),
            "normals": len(norm),
            "false_flags": int(norm["flagged"].sum()),
            "fpr_pct": round(100 * norm["flagged"].mean(), 3),
            "var1_p50": round(float(gb_cals["var1"].median()), 3),
            "var2_p50": round(float(gb_cals["var2"].median()), 3),
            "var1_p90": round(float(gb_cals["var1"].quantile(0.9)), 3),
            "var2_p90": round(float(gb_cals["var2"].quantile(0.9)), 3),
            "share_at_floor_pct": round(
                100
                * float(((gb_cals["var1"] <= 0.10) & (gb_cals["var2"] <= 0.10)).mean()),
                1,
            ),
            "cals": len(gb_cals),
            "cals_over_budget": int(
                (
                    (cals["n_flagged"] >= FALLBACK_MIN_FLAGS)
                    & (cals["n_flagged"] > FALLBACK_RATE * cals["n_runs"])
                ).sum()
            ),
            "seconds": round(time.monotonic() - t0, 1),
        }
        results.append(row)
        print(json.dumps(row), flush=True)
        point_frames.append(gb_q.drop(columns=["qualified", "test_type"]).assign(window=n))

    out = pd.DataFrame(results)
    out.to_csv(HERE / "window_sweep_results.csv", index=False)
    pd.concat(point_frames, ignore_index=True).to_parquet(
        HERE / "window_sweep_points.parquet", index=False
    )
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
