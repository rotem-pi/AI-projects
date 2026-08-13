"""Evaluate Prophet as a direct bounds producer.

Protocol per series (day-hour averages from the prod replica):
  - train on everything except the last 14 days, forecast bounds for those 14
    days (the "refresh weekly, serve precomputed bounds" deployment pattern,
    with a 2x safety margin on staleness)
  - FPR = fraction of real held-out points outside Prophet's band
  - detection = fraction of injected anomalies (+30/50/100%) outside the band
  - also record fit time and serialized model size (ops cost of the idea)
"""

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.WARNING)

from prophet import Prophet  # noqa: E402
from prophet.serialize import model_to_json  # noqa: E402

HERE = Path(__file__).resolve().parent

HOLDOUT_DAYS = 14
MIN_SPAN_DAYS = 60
INTERVAL_WIDTH = 0.995
INJECT_DELTAS = [0.3, 0.5, 1.0]
INJECT_POINTS = 30
NON_SEASONAL_SAMPLE = 60
SEED = 1337


def fit_and_score(sdf: pd.DataFrame, rng: np.random.Generator) -> dict | None:
    sdf = sdf.sort_values(["day", "hr"])
    ds = pd.to_datetime(sdf["day"]) + pd.to_timedelta(sdf["hr"], unit="h")
    df = pd.DataFrame({"ds": ds, "y": sdf["avg_val"].astype(float).values})
    span = (df["ds"].max() - df["ds"].min()).days
    if span < MIN_SPAN_DAYS:
        return None
    cutoff = df["ds"].max() - pd.Timedelta(days=HOLDOUT_DAYS)
    train, test = df[df["ds"] <= cutoff], df[df["ds"] > cutoff]
    if len(train) < 100 or len(test) < 20:
        return None

    mode = "multiplicative" if (train["y"] > 0).all() else "additive"
    model = Prophet(
        interval_width=INTERVAL_WIDTH,
        seasonality_mode=mode,
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    t0 = time.time()
    model.fit(train)
    fit_seconds = time.time() - t0

    fcst = model.predict(test[["ds"]])
    lo = fcst["yhat_lower"].values
    hi = fcst["yhat_upper"].values
    y = test["y"].values
    out = {
        "n_test": len(y),
        "fpr": float(np.mean((y < lo) | (y > hi))),
        "fit_seconds": fit_seconds,
        "model_kb": len(json.dumps(model_to_json(model))) / 1024,
        "rel_band_width": float(np.median((hi - lo) / np.maximum(np.abs(y), 1e-9))),
    }
    idx = rng.choice(np.arange(len(y)), size=min(INJECT_POINTS, len(y)), replace=False)
    for delta in INJECT_DELTAS:
        inj = y[idx] * (1 + delta)
        out[f"det_{delta}"] = float(np.mean((inj < lo[idx]) | (inj > hi[idx])))
    return out


def main() -> None:
    hourly = pd.read_csv(HERE / "auto_tests_hourly.csv", parse_dates=["day"])
    hourly["asset_value"] = hourly["asset_value"].fillna("")
    res = pd.read_csv(HERE / "auto_tests_hour_seasonality.csv")
    res["asset_value"] = res["asset_value"].fillna("")
    res = res[res["p"].notna()].copy()
    res["robust"] = res["robust"].fillna(False).astype(bool)

    rng = np.random.default_rng(SEED)
    non_seasonal = res[~res["robust"]].sample(
        min(NON_SEASONAL_SAMPLE, (~res["robust"]).sum()), random_state=SEED)
    cohort = pd.concat([res[res["robust"]], non_seasonal])

    rows = []
    t0 = time.time()
    for k, (_, r) in enumerate(cohort.iterrows()):
        sdf = hourly[(hourly["metric_id"] == r["metric_id"])
                     & (hourly["asset_value"] == r["asset_value"])]
        try:
            out = fit_and_score(sdf, rng)
        except Exception as e:  # noqa: BLE001 - survey run, log and continue
            print(f"metric {r['metric_id']}: {type(e).__name__}: {e}", flush=True)
            continue
        if out is None:
            continue
        out.update({"metric_id": r["metric_id"], "robust": bool(r["robust"])})
        rows.append(out)
        if k % 20 == 0:
            print(f"{k}/{len(cohort)} done, {time.time() - t0:.0f}s", flush=True)

    bt = pd.DataFrame(rows)
    bt.to_csv(HERE / "prophet_bounds_eval.csv", index=False)

    for label, grp in [("hour-seasonal", bt[bt["robust"]]),
                       ("non-seasonal", bt[~bt["robust"]])]:
        if grp.empty:
            continue
        print(f"\n=== {label} (n={len(grp)}) ===")
        print(f"held-out FPR: median {grp['fpr'].median():.2%}, "
              f"mean {grp['fpr'].mean():.2%}, p90 {grp['fpr'].quantile(0.9):.2%}, "
              f"share of series with FPR > 5%: {(grp['fpr'] > 0.05).mean():.0%}")
        for delta in INJECT_DELTAS:
            print(f"detection +{delta:.0%}: {grp[f'det_{delta}'].mean():.0%}")
        print(f"median relative band width: {grp['rel_band_width'].median():.2f}")

    print(f"\nops: median fit {bt['fit_seconds'].median():.1f}s, "
          f"p90 {bt['fit_seconds'].quantile(0.9):.1f}s; "
          f"median serialized model {bt['model_kb'].median():.0f} KB, "
          f"p90 {bt['model_kb'].quantile(0.9):.0f} KB")
    print(f"fleet estimate for 3363 series: "
          f"{bt['fit_seconds'].median() * 3363 / 60:.0f} CPU-minutes per refresh, "
          f"{bt['model_kb'].median() * 3363 / 1024:.0f} MB storage")


if __name__ == "__main__":
    main()
