"""Sweep RECURRENCE_VALUE_TOLERANCE for the recurrence rescue.

Per tolerance (0.0 reproduces the pre-rescue behavior): replay
GuardedBand (N=3) in-process over the 4 labeled harness datasets
(tp/fp vs labels) and over the injection cohorts (injected detection at
both magnitudes, non-injected flag rate).

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/rescue_tolerance_sweep.py
"""

import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import app.brain.anomaly.guarded_band_calibration as gbc  # noqa: E402
from app.tests_gen.models.guarded_band_model import (  # noqa: E402
    GuardedBandSystemTimeSeriesModel,
)
from part2_inject_sim import (  # noqa: E402
    MAGS,
    apply_injections,
    eval_b_series,
    plan_injections,
    set_window,
)

OUTPUTS = HERE.parents[1] / "backend/tests/anomaly/outputs"
DATASETS = ["random-a-200", "random-a-500", "a_self_labeled", "definity_tagging"]
TOLERANCES = [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]


def load_labeled() -> list[dict]:
    series = []
    for ds in DATASETS:
        base = OUTPUTS / "prophet_labeler_grid_search" / ds / "datasets/1/all"
        metrics = pd.read_csv(base / "metrics.csv", parse_dates=["pit"])
        labels = pd.read_csv(base / "labels.csv", parse_dates=["pit"])
        df = metrics.merge(
            labels[["metric_id", "pit", "is_anomaly"]], on=["metric_id", "pit"]
        )
        for _, g in df.groupby("metric_id"):
            g = g.sort_values("pit").drop_duplicates("pit")
            series.append(
                {
                    "sid": len(series),
                    "metric_type": g["metric_type"].iloc[0],
                    "pits": pd.DatetimeIndex(g["pit"]),
                    "vals": g["value"].to_numpy(dtype=float),
                    "gt": dict(zip(g["pit"], g["is_anomaly"].astype(bool))),
                }
            )
    return series


def load_injection() -> list[dict]:
    series = []
    for suffix in ["", "_registry"]:
        v = pd.read_parquet(HERE / f"part2_values{suffix}.parquet")
        v["pit"] = pd.to_datetime(v["pit"], utc=True).dt.tz_localize(None)
        v = v.sort_values("pit")
        for (mid, av), g in v.groupby(["metric_id", "asset_value"], sort=True):
            pits = pd.DatetimeIndex(g["pit"])
            vals = g["value"].to_numpy(dtype=float)
            plan = plan_injections(pits, int(mid), av)
            series.append(
                {
                    "sid": len(series),
                    "metric_type": g["metric_type"].iloc[0] or "",
                    "pits": pits,
                    "arm_vals": {
                        mag: apply_injections(vals, pits, plan, up, down)
                        for mag, (up, down) in MAGS.items()
                    },
                    "inj_idx": {pits.get_loc(i["pit"]): i for i in plan},
                }
            )
    return series


def main() -> None:
    labeled = load_labeled()
    injection = load_injection()
    print(f"{len(labeled)} labeled series, {len(injection)} injection series")
    set_window(3)

    rows = []
    for tol in TOLERANCES:
        t0 = time.monotonic()
        gbc.RECURRENCE_VALUE_TOLERANCE = tol
        tp = fp = 0
        for s in labeled:
            df = pd.DataFrame(
                {
                    "metric_id": s["sid"],
                    "pit": s["pits"],
                    "value": s["vals"],
                    "metric_type": s["metric_type"],
                }
            )
            out = GuardedBandSystemTimeSeriesModel(metric_id=s["sid"]).predict_df(df)
            if out is None:
                continue
            for r in out.reset_index().itertuples():
                if r.is_anomaly:
                    if s["gt"].get(pd.Timestamp(r.pit), False):
                        tp += 1
                    else:
                        fp += 1
        row = {"tolerance": tol, "labeled_tp": tp, "labeled_fp": fp}
        for mag in MAGS:
            det_n = det_d = flags = norm = 0
            for s in injection:
                points, details, _ = eval_b_series(
                    GuardedBandSystemTimeSeriesModel,
                    s["sid"],
                    s["metric_type"],
                    s["pits"],
                    s["arm_vals"][mag],
                    s["inj_idx"],
                )
                for r in points:
                    if r["injected"]:
                        continue
                    norm += 1
                    flags += r["flagged"]
                det_d += len(details)
                det_n += sum(r["flagged"] for r in details)
            row[f"inj_det_{mag}_pct"] = round(100 * det_n / det_d, 1)
            row[f"flag_rate_{mag}_pct"] = round(100 * flags / norm, 2)
        row["seconds"] = round(time.monotonic() - t0, 1)
        rows.append(row)
        print(row, flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "rescue_tolerance_sweep.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
