"""Bounds-producer bake-off: which architecture best turns metric history
into next-period alert bounds, across all metric types.

Data (prod replica extracts, see fetch_cohorts.py / auto_tests_hour_seasonality.py):
  - HOURLY grain: day x 6h-bucket averages, last 90 days (agg_hourly.csv +
    auto_tests_hourly.csv, deduped) - the high-frequency slice.
  - DAILY grain: daily medians, 120-730 day spans (agg_midspan/oneyear/yearly)
    - random sample across ALL metric types.

Split per series, strictly chronological (no leakage):
  fit [start .. T-cal-test) -> calibration [T-cal-test .. T-test) -> test [T-test .. T]
  hourly: cal 21d, test 14d.  daily: cal 42d, test 28d.
  Models fit on the fit window only; conformal quantiles use calibration
  residuals only; all reported numbers come from the test window only.
  Rolling baselines predict causally (each point uses only earlier points).

Configs: {seasonal-naive, seasonal-window-avg, trailing-mean} x {abs, rel} bands,
Prophet raw 99.5% interval, Prophet + conformal (abs/rel), MSTL + conformal
(daily grain only; hourly cells are irregular). Target FPR 1% (alpha).
"""

import logging
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.INFO)

HERE = Path(__file__).resolve().parent
ALPHA = 0.01
INJECT_DELTAS = [0.3, 0.5, 1.0]
BUCKET_HOURS = 6
ROLL_WINDOW = 3
DEADLINE_SECONDS = 13 * 60
MAX_HOURLY_SERIES = 600
MAX_DAILY_SERIES = 800
SEED = 1337

HOURLY_CAL_DAYS, HOURLY_TEST_DAYS, HOURLY_MIN_DAYS = 21, 14, 60
DAILY_CAL_DAYS, DAILY_TEST_DAYS, DAILY_MIN_DAYS = 42, 28, 130

START = time.time()


def conformal_q(resid: np.ndarray, alpha: float) -> float:
    n = len(resid)
    k = min(n - 1, int(np.ceil((n + 1) * (1 - alpha))) - 1)
    return float(np.sort(resid)[k])


def rolling_preds(y: np.ndarray, buckets: np.ndarray, mode: str) -> np.ndarray:
    """Causal point forecasts: each yhat[t] uses only y[:t]."""
    yhat = np.full(len(y), np.nan)
    last_by_bucket: dict[int, list[float]] = {}
    trail: list[float] = []
    for t in range(len(y)):
        b = int(buckets[t])
        hist = last_by_bucket.get(b, [])
        if mode == "snaive" and hist:
            yhat[t] = hist[-1]
        elif mode == "swa" and hist:
            yhat[t] = float(np.mean(hist[-ROLL_WINDOW:]))
        elif mode == "trail" and trail:
            yhat[t] = float(np.mean(trail[-ROLL_WINDOW:]))
        last_by_bucket.setdefault(b, []).append(float(y[t]))
        trail.append(float(y[t]))
    return yhat


def score_band(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> dict:
    ok = ~(np.isnan(lo) | np.isnan(hi))
    y, lo, hi = y[ok], lo[ok], hi[ok]
    if len(y) < 10:
        return {}
    out = {"fpr": float(np.mean((y < lo) | (y > hi))), "n_test": len(y)}
    for d in INJECT_DELTAS:
        inj = y * (1 + d)
        out[f"det_{d}"] = float(np.mean((inj < lo) | (inj > hi)))
    denom = np.maximum(np.abs(y), 1e-9)
    out["rel_width"] = float(np.median((hi - lo) / denom))
    return out


def apply_conformal(y, yhat, cal_mask, test_mask, kind: str) -> dict:
    resid = y - yhat
    cal_ok = cal_mask & ~np.isnan(yhat)
    if cal_ok.sum() < 20:
        return {}
    if kind == "abs":
        q = conformal_q(np.abs(resid[cal_ok]), ALPHA)
        lo, hi = yhat - q, yhat + q
    else:
        scale = np.maximum(np.abs(yhat), 1e-9)
        q = conformal_q(np.abs(resid[cal_ok]) / scale[cal_ok], ALPHA)
        lo, hi = yhat * 1.0 - q * np.abs(yhat), yhat + q * np.abs(yhat)
    return score_band(y[test_mask], lo[test_mask], hi[test_mask])


def prophet_fit_predict(ts, y, fit_mask, grain: str, predict_ts) -> dict | None:
    from prophet import Prophet

    train = pd.DataFrame({"ds": ts[fit_mask], "y": y[fit_mask]})
    if len(train) < 50:
        return None
    mode = "multiplicative" if (train["y"] > 0).all() else "additive"
    m = Prophet(interval_width=0.995, seasonality_mode=mode,
                daily_seasonality=(grain == "hourly"),
                weekly_seasonality=True, yearly_seasonality=False)
    m.fit(train)
    fc = m.predict(pd.DataFrame({"ds": predict_ts}))
    return {"yhat": fc["yhat"].values, "lo": fc["yhat_lower"].values,
            "hi": fc["yhat_upper"].values}


def mstl_fit_predict(days: pd.DatetimeIndex, y: np.ndarray, fit_mask,
                     horizon: int) -> np.ndarray | None:
    from statsforecast import StatsForecast
    from statsforecast.models import MSTL

    s = pd.Series(y[fit_mask], index=days[fit_mask]).asfreq("D")
    if s.isna().mean() > 0.3:
        return None
    s = s.interpolate(limit=3).dropna()
    if len(s) < 60:
        return None
    df = pd.DataFrame({"unique_id": "s", "ds": s.index, "y": s.values})
    sf = StatsForecast(models=[MSTL(season_length=7)], freq="D", n_jobs=1)
    fc = sf.forecast(df=df, h=horizon)
    return fc["MSTL"].values


def eval_series(ts: pd.DatetimeIndex, y: np.ndarray, buckets: np.ndarray,
                grain: str, run_heavy: bool) -> dict[str, dict]:
    cal_days, test_days = ((HOURLY_CAL_DAYS, HOURLY_TEST_DAYS) if grain == "hourly"
                           else (DAILY_CAL_DAYS, DAILY_TEST_DAYS))
    t_end = ts.max()
    test_start = t_end - pd.Timedelta(days=test_days)
    cal_start = test_start - pd.Timedelta(days=cal_days)
    fit_mask = np.asarray(ts < cal_start)
    cal_mask = np.asarray((ts >= cal_start) & (ts < test_start))
    test_mask = np.asarray(ts >= test_start)
    if fit_mask.sum() < 40 or cal_mask.sum() < 20 or test_mask.sum() < 10:
        return {}

    results: dict[str, dict] = {}
    for mode in ["snaive", "swa", "trail"]:
        yhat = rolling_preds(y, buckets, mode)
        for kind in ["abs", "rel"]:
            r = apply_conformal(y, yhat, cal_mask, test_mask, kind)
            if r:
                results[f"{mode}_{kind}"] = r

    if run_heavy:
        try:
            p = prophet_fit_predict(ts, y, fit_mask, grain, ts)
        except Exception:
            p = None
        if p is not None:
            r = score_band(y[test_mask], p["lo"][test_mask], p["hi"][test_mask])
            if r:
                results["prophet_raw"] = r
            for kind in ["abs", "rel"]:
                r = apply_conformal(y, p["yhat"], cal_mask, test_mask, kind)
                if r:
                    results[f"prophet_conf_{kind}"] = r

        if grain == "daily":
            try:
                horizon = int(cal_mask.sum() + test_mask.sum())
                days = pd.DatetimeIndex(ts)
                fc = mstl_fit_predict(days, y, fit_mask, horizon)
            except Exception:
                fc = None
            if fc is not None:
                yhat = np.full(len(y), np.nan)
                oos_idx = np.where(cal_mask | test_mask)[0]
                grid_start = days[oos_idx[0]]
                for i in oos_idx:
                    offset = (days[i] - grid_start).days
                    if 0 <= offset < len(fc):
                        yhat[i] = fc[offset]
                for kind in ["abs", "rel"]:
                    r = apply_conformal(y, yhat, cal_mask, test_mask, kind)
                    if r:
                        results[f"mstl_conf_{kind}"] = r
    return results


def load_hourly() -> pd.DataFrame:
    frames = []
    for f in ["agg_hourly.csv", "auto_tests_hourly.csv"]:
        df = pd.read_csv(HERE / f, parse_dates=["day"])
        df["asset_value"] = df["asset_value"].fillna("")
        frames.append(df[["metric_id", "asset_value", "day", "hr", "avg_val"]])
    return pd.concat(frames).drop_duplicates(subset=["metric_id", "asset_value",
                                                     "day", "hr"])


def load_daily() -> pd.DataFrame:
    frames = []
    for cohort in ["yearly", "oneyear", "midspan"]:
        agg = pd.read_csv(HERE / f"agg_{cohort}.csv", parse_dates=["day"])
        meta = pd.read_csv(HERE / f"sample_{cohort}.csv")
        for d in (agg, meta):
            d["asset_value"] = d["asset_value"].fillna("")
        agg = agg.merge(meta[["metric_id", "asset_value", "metric_type"]],
                        on=["metric_id", "asset_value"], how="left")
        frames.append(agg)
    return pd.concat(frames).drop_duplicates(subset=["metric_id", "asset_value", "day"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []

    hourly = load_hourly()
    keys = hourly.groupby(["metric_id", "asset_value"]).size()
    keys = keys[keys >= 300].index.tolist()
    rng.shuffle(keys)
    keys = keys[:MAX_HOURLY_SERIES]
    print(f"hourly series: {len(keys)}", flush=True)
    for i, (mid, av) in enumerate(keys):
        s = hourly[(hourly["metric_id"] == mid)
                   & (hourly["asset_value"] == av)].sort_values(["day", "hr"])
        if (s["day"].max() - s["day"].min()).days < HOURLY_MIN_DAYS:
            continue
        ts = pd.DatetimeIndex(s["day"] + pd.to_timedelta(s["hr"], unit="h"))
        run_heavy = time.time() - START < DEADLINE_SECONDS * 0.45
        res = eval_series(ts, s["avg_val"].astype(float).values,
                          (s["hr"] // BUCKET_HOURS).values, "hourly", run_heavy)
        for cfg, r in res.items():
            rows.append({"grain": "hourly", "metric_id": mid, "config": cfg, **r})
        if i % 100 == 0:
            print(f"hourly {i}/{len(keys)} ({time.time() - START:.0f}s)", flush=True)

    daily = load_daily()
    dkeys = daily.groupby(["metric_id", "asset_value"]).size()
    dkeys = dkeys[dkeys >= 100].index.tolist()
    rng.shuffle(dkeys)
    dkeys = dkeys[:MAX_DAILY_SERIES]
    print(f"daily series: {len(dkeys)}", flush=True)
    truncated = 0
    for i, (mid, av) in enumerate(dkeys):
        s = daily[(daily["metric_id"] == mid)
                  & (daily["asset_value"] == av)].sort_values("day")
        if (s["day"].max() - s["day"].min()).days < DAILY_MIN_DAYS:
            continue
        ts = pd.DatetimeIndex(s["day"])
        run_heavy = time.time() - START < DEADLINE_SECONDS * 0.9
        truncated += int(not run_heavy)
        res = eval_series(ts, s["avg_val"].astype(float).values,
                          ts.dayofweek.values, "daily", run_heavy)
        mt = s["metric_type"].iloc[0] if "metric_type" in s else ""
        for cfg, r in res.items():
            rows.append({"grain": "daily", "metric_id": mid, "metric_type": mt,
                         "config": cfg, **r})
        if i % 200 == 0:
            print(f"daily {i}/{len(dkeys)} ({time.time() - START:.0f}s)", flush=True)

    bt = pd.DataFrame(rows)
    bt.to_csv(HERE / "bakeoff_results.csv", index=False)
    if truncated:
        print(f"NOTE: heavy models skipped on {truncated} daily series (deadline)")

    for grain in ["hourly", "daily"]:
        g = bt[bt["grain"] == grain]
        if g.empty:
            continue
        print(f"\n===== {grain} grain =====")
        summary = g.groupby("config").agg(
            n=("metric_id", "nunique"),
            fpr_med=("fpr", "median"), fpr_mean=("fpr", "mean"),
            fpr_p90=("fpr", lambda x: x.quantile(0.9)),
            storm=("fpr", lambda x: (x > 0.05).mean()),
            det30=("det_0.3", "mean"), det50=("det_0.5", "mean"),
            det100=("det_1.0", "mean"), width=("rel_width", "median"),
        ).sort_values("det50", ascending=False)
        print(summary.round(3).to_string())

    print(f"\ntotal wall time: {time.time() - START:.0f}s")


if __name__ == "__main__":
    main()
