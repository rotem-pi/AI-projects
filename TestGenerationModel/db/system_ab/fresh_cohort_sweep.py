"""Window sweep on a FRESH cohort pulled from the prod read replica.

Unlabeled complement to window_sweep.py: samples ~600 random currently-active
series from metrics_agg, fetches their raw last-91-days history in chunks
(covering index friendly, read-only session, keepalives so the NAT mapping
does not kill long-silent calls), then replays the weekly shadow flow per
trailing-window N and reports the flag rate - the alert-volume proxy on
today's production data. Anomaly prevalence is unknown but constant across
N, so flag-rate differences reflect band behavior only.

Fetch is cached in fresh_cohort_points.csv; delete it to resample.

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/fresh_cohort_sweep.py
"""

import json
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

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
CACHE = HERE / "fresh_cohort_points.csv"
DB = (
    "postgresql://postgres:postgres"
    "@prod-read-replica.coe3zosbcs5l.eu-north-1.rds.amazonaws.com:5432/app"
    "?keepalives=1&keepalives_idle=30&keepalives_interval=10"
    "&keepalives_count=3&connect_timeout=15"
)
N_SERIES = 600
DAYS_BACK = 91
CHUNK = 40
EVAL_DAYS = 7
WINDOWS = [1, 2, 3, 4, 5, 7, 9, 14, 21]

SAMPLE_SQL = f"""
SELECT ma.metric_id, ma.asset_value, mc.metric_type
FROM metrics_agg ma JOIN metrics_conf mc USING (metric_id)
WHERE ma.max_app_pit >= now() - interval '7 days'
  AND ma.total_metrics >= 60
ORDER BY md5(ma.metric_id::text || ma.asset_value)
LIMIT {N_SERIES}
"""

RAW_SQL = f"""
SELECT m.metric_id, m.asset_value, m.app_pit, m.metric_value
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) s
  USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL
  AND m.app_pit >= now() - interval '{DAYS_BACK} days'
"""


def fetch() -> pd.DataFrame:
    conn = psycopg.connect(DB, autocommit=True)
    conn.read_only = True
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    t0 = time.time()
    cur.execute(SAMPLE_SQL)
    sample = cur.fetchall()
    print(f"sampled {len(sample)} series", flush=True)

    frames = []
    for i in range(0, len(sample), CHUNK):
        chunk = sample[i : i + CHUNK]
        cur.execute(
            RAW_SQL,
            {"mids": [r[0] for r in chunk], "avs": [r[1] for r in chunk]},
        )
        frames.append(
            pd.DataFrame(
                cur.fetchall(),
                columns=["metric_id", "asset_value", "pit", "value"],
            )
        )
        print(
            f"{i + len(chunk)}/{len(sample)} series, "
            f"{sum(len(f) for f in frames)} rows, {time.time() - t0:.0f}s",
            flush=True,
        )
    conn.close()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(CACHE, index=False)
    return df


def week_start(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def replay_series(g: pd.DataFrame, window: int) -> tuple[list[dict], list[dict]]:
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
                    {"test_type": calibrated.test_type, "flagged": flagged}
                )
            cal_rows.append(
                {
                    "test_type": calibrated.test_type,
                    "n_runs": n_runs,
                    "n_flagged": n_flagged,
                }
            )
        as_of += pd.Timedelta(days=7)
    return point_rows, cal_rows


def main() -> None:
    if CACHE.exists():
        df = pd.read_csv(CACHE)
    else:
        df = fetch()
    df["pit"] = pd.to_datetime(df["pit"], format="ISO8601").dt.tz_localize(None)
    df = df[~df.duplicated(subset=["metric_id", "asset_value", "pit"], keep="first")]
    groups = [
        g.sort_values("pit").reset_index(drop=True)
        for _, g in df.groupby(["metric_id", "asset_value"])
    ]
    print(f"{len(groups)} series, {len(df)} points", flush=True)

    results = []
    for n in WINDOWS:
        t0 = time.monotonic()
        gbc.TRAILING_WINDOW = n
        gbt.TRAILING_WINDOW = n
        points_all: list[dict] = []
        cals_all: list[dict] = []
        for g in groups:
            p, c = replay_series(g, n)
            points_all.extend(p)
            cals_all.extend(c)
        points = pd.DataFrame(points_all)
        cals = pd.DataFrame(cals_all)
        gb = points[points["test_type"] == "GuardedBand"]
        row = {
            "window": n,
            "gb_points": len(gb),
            "gb_flag_rate_pct": round(100 * gb["flagged"].mean(), 3),
            "gb_cals": int((cals["test_type"] == "GuardedBand").sum()),
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

    out = pd.DataFrame(results)
    out.to_csv(HERE / "fresh_cohort_sweep_results.csv", index=False)
    print("\n" + out.to_string(index=False))


if __name__ == "__main__":
    main()
