"""Round 2: representative-impact benchmark for the trailing-conformal design.

Sample: active series (reported last 30d) carrying an enabled auto test,
stratified proportionally to the auto-test fleet: HF 120, daily 530,
weekly 480, sparse 70. No span filter at sampling time - series too young
for calibration are counted as ineligible, which is itself a result.

Grain and windows per cadence tier:
  hf:     hourly cells, calibration 21d, test 14d
  daily:  daily medians, calibration 56d, test 28d
  weekly: daily medians, calibration 112d, test 56d
  sparse: daily medians, calibration 168d, test 84d

Point forecast is fixed (round-1 winner): mean of last 3 observations, causal.
Band configs: per-series conformal {abs,rel} x alpha {1%,2%,5%} and POOLED
rel-residual conformal (pooling calibration residuals across series of the
same tier+metric_type) x alpha {1%,2%,5%}.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")

STRATA = {"hf": 120, "daily": 530, "weekly": 480, "sparse": 70}
WINDOWS = {"hf": (21, 14), "daily": (56, 28), "weekly": (112, 56),
           "sparse": (168, 84)}
ALPHAS = [0.01, 0.02, 0.05]
ROLL = 3
INJECT_DELTAS = [0.3, 0.5, 1.0]
MIN_CAL_PER_SERIES = 10
MIN_CAL_POOLED = 3
MIN_TEST = 5
POOL_MIN_SERIES = 5
CHUNK = 40
FETCH_DAYS = {"hf": 90, "daily": 420, "weekly": 420, "sparse": 420}
START = time.time()

SAMPLE_SQL = """
WITH s AS (
  SELECT ma.metric_id, ma.asset_value, mc.metric_type, mc.app_id,
         ma.total_metrics,
         EXTRACT(EPOCH FROM (ma.max_app_pit - ma.min_app_pit))/86400.0 AS span_d
  FROM metrics_agg ma
  JOIN metrics_conf mc ON mc.metric_id = ma.metric_id
  JOIN tests t ON t.metric_id = ma.metric_id AND t.asset_value = ma.asset_value
       AND t.is_auto AND t.is_enabled
  WHERE ma.total_metrics >= 5 AND ma.max_app_pit >= now() - interval '30 days'
)
SELECT metric_id, asset_value, metric_type, app_id FROM s
WHERE CASE %(tier)s
      WHEN 'hf' THEN total_metrics/GREATEST(span_d,1) >= 3
      WHEN 'daily' THEN total_metrics/GREATEST(span_d,1) >= 0.8
           AND total_metrics/GREATEST(span_d,1) < 3
      WHEN 'weekly' THEN total_metrics/GREATEST(span_d,1) >= 0.15
           AND total_metrics/GREATEST(span_d,1) < 0.8
      ELSE span_d >= 1 AND total_metrics/GREATEST(span_d,1) < 0.15 END
ORDER BY md5(metric_id::text || asset_value) LIMIT %(n)s
"""

HOURLY_SQL = """
SELECT m.metric_id, m.asset_value, date_trunc('day', m.app_pit)::date AS day,
       EXTRACT(HOUR FROM m.app_pit)::int AS hr, AVG(m.metric_value) AS val
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) u USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL AND m.app_pit >= now() - interval '90 days'
GROUP BY 1,2,3,4
"""

DAILY_SQL = """
SELECT m.metric_id, m.asset_value, date_trunc('day', m.app_pit)::date AS day,
       PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY m.metric_value) AS val
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) u USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL AND m.app_pit >= now() - interval '420 days'
GROUP BY 1,2,3
"""


def fetch(cur, sql: str, series: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    rows = []
    recs = series[["metric_id", "asset_value"]].values
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i:i + CHUNK]
        cur.execute(sql, {"mids": [int(r[0]) for r in chunk],
                          "avs": [str(r[1]) for r in chunk]})
        rows.extend(cur.fetchall())
    df = pd.DataFrame(rows, columns=cols)
    df["val"] = df["val"].astype(float)
    df["day"] = pd.to_datetime(df["day"])
    return df


def trailing_preds(y: np.ndarray) -> np.ndarray:
    yhat = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        yhat[t] = float(np.mean(y[max(0, t - ROLL):t]))
    return yhat


def conformal_q(resid: np.ndarray, alpha: float) -> float:
    k = min(len(resid) - 1, int(np.ceil((len(resid) + 1) * (1 - alpha))) - 1)
    return float(np.sort(resid)[k])


def score(y, yhat, test_mask, q, rel: bool) -> dict:
    scale = np.abs(yhat) if rel else 1.0
    lo, hi = yhat - q * scale, yhat + q * scale
    ok = test_mask & ~np.isnan(yhat)
    yt, lot, hit = y[ok], lo[ok], hi[ok]
    out = {"fpr": float(np.mean((yt < lot) | (yt > hit))), "n_test": int(len(yt))}
    for d in INJECT_DELTAS:
        inj = yt * (1 + d)
        out[f"det_{d}"] = float(np.mean((inj < lot) | (inj > hit)))
    return out


def main() -> None:
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")

    frames = {}
    samples = {}
    for tier, n in STRATA.items():
        cur.execute(SAMPLE_SQL, {"tier": tier, "n": n})
        samples[tier] = pd.DataFrame(
            cur.fetchall(),
            columns=["metric_id", "asset_value", "metric_type", "app_id"])
        samples[tier]["asset_value"] = samples[tier]["asset_value"].fillna("")
        sql = HOURLY_SQL if tier == "hf" else DAILY_SQL
        cols = (["metric_id", "asset_value", "day", "hr", "val"] if tier == "hf"
                else ["metric_id", "asset_value", "day", "val"])
        frames[tier] = fetch(cur, sql, samples[tier], cols)
        frames[tier].to_csv(HERE / f"round2_frame_{tier}.csv", index=False)
        samples[tier].to_csv(HERE / f"round2_sample_{tier}.csv", index=False)
        print(f"{tier}: {len(samples[tier])} series, {len(frames[tier])} rows "
              f"({time.time() - START:.0f}s)", flush=True)
    conn.close()
    run_eval(frames, samples, HERE / "round2_results.csv")


def run_eval(frames: dict, samples: dict, out_path: Path) -> pd.DataFrame:
    # Pass 1: per-series causal predictions, residuals, masks
    series_data = []
    for tier, df in frames.items():
        cal_d, test_d = WINDOWS[tier]
        meta = samples[tier].set_index(["metric_id", "asset_value"])
        for key, s in df.groupby(["metric_id", "asset_value"]):
            s = s.sort_values(["day", "hr"] if tier == "hf" else "day")
            ts = (pd.DatetimeIndex(s["day"] + pd.to_timedelta(s["hr"], unit="h"))
                  if tier == "hf" else pd.DatetimeIndex(s["day"]))
            y = s["val"].values
            t_end = ts.max()
            test_mask = np.asarray(ts >= t_end - pd.Timedelta(days=test_d))
            cal_mask = np.asarray(
                (ts >= t_end - pd.Timedelta(days=cal_d + test_d)) & ~test_mask)
            yhat = trailing_preds(y)
            cal_ok = cal_mask & ~np.isnan(yhat)
            if test_mask.sum() < MIN_TEST or cal_ok.sum() < MIN_CAL_POOLED:
                series_data.append({"tier": tier, "key": key, "eligible": "no",
                                    "mtype": meta.loc[key, "metric_type"]})
                continue
            r = y - yhat
            scale = np.maximum(np.abs(yhat), 1e-9)
            series_data.append({
                "tier": tier, "key": key, "eligible": "yes",
                "mtype": meta.loc[key, "metric_type"], "y": y, "yhat": yhat,
                "test_mask": test_mask, "cal_abs": np.abs(r[cal_ok]),
                "cal_rel": (np.abs(r) / scale)[cal_ok]})

    # Pooled rel-residual quantiles per (tier, metric_type)
    pools: dict[tuple, np.ndarray] = {}
    for sd in series_data:
        if sd["eligible"] == "yes":
            pools.setdefault((sd["tier"], sd["mtype"]), []).append(sd["cal_rel"])
    pooled_q = {}
    for k, residlists in pools.items():
        if len(residlists) >= POOL_MIN_SERIES:
            allr = np.concatenate(residlists)
            pooled_q[k] = {a: conformal_q(allr, a) for a in ALPHAS}

    # Pass 2: score all configs
    rows = []
    for sd in series_data:
        if sd["eligible"] == "no":
            rows.append({"tier": sd["tier"], "metric_id": sd["key"][0],
                         "config": "INELIGIBLE"})
            continue
        for alpha in ALPHAS:
            a_lbl = f"a{int(alpha * 100)}"
            if len(sd["cal_abs"]) >= MIN_CAL_PER_SERIES:
                for kind in ["abs", "rel"]:
                    q = conformal_q(sd[f"cal_{kind}"], alpha)
                    rows.append({"tier": sd["tier"], "metric_id": sd["key"][0],
                                 "config": f"{kind}_{a_lbl}",
                                 **score(sd["y"], sd["yhat"], sd["test_mask"],
                                         q, kind == "rel")})
            pq = pooled_q.get((sd["tier"], sd["mtype"]))
            if pq:
                rows.append({"tier": sd["tier"], "metric_id": sd["key"][0],
                             "config": f"pool_{a_lbl}",
                             **score(sd["y"], sd["yhat"], sd["test_mask"],
                                     pq[alpha], True)})

    bt = pd.DataFrame(rows)
    bt.to_csv(out_path, index=False)

    for tier in STRATA:
        g = bt[bt["tier"] == tier]
        n_sampled = len(samples[tier])
        n_inel = (g["config"] == "INELIGIBLE").sum()
        print(f"\n===== {tier} (sampled {n_sampled}, "
              f"ineligible {n_inel} = {n_inel / n_sampled:.0%}) =====")
        gg = g[g["config"] != "INELIGIBLE"]
        if gg.empty:
            continue
        summary = gg.groupby("config").agg(
            n=("metric_id", "nunique"), fpr_mean=("fpr", "mean"),
            fpr_p90=("fpr", lambda x: x.quantile(0.9)),
            storm=("fpr", lambda x: (x > 0.05).mean()),
            det30=("det_0.3", "mean"), det50=("det_0.5", "mean"),
            det100=("det_1.0", "mean"))
        print(summary.round(3).to_string())

    print(f"\ntotal wall time: {time.time() - START:.0f}s")
    return bt


if __name__ == "__main__":
    main()
