"""Stage 2: rel-band counterfactual on a 2,000-series stratified sample.

Population: the approved 547K-series population, restricted to rel-band
eligible series (>= 10 calibration points in days -88..-30, >= 5 evaluation
points in the last 30 days, both from the full-population window_counts scan).
Stratified by recent cadence (cnt_eval / 30) proportionally to the population;
deterministic md5 ordering.

Simulation per series: values deduped per PIT, trailing-mean(3) causal
forecast, per-series relative conformal band at 1% miss budget, recalibrated
weekly during the evaluation month from the trailing 56 days.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")

SAMPLE_N = 2000
TRAILING_WINDOW = 3
MISS_BUDGET = 0.01
CAL_DAYS = 56
EVAL_DAYS = 30
MIN_CAL, MIN_EVAL = 10, 5
CHUNK = 40
START = time.time()

POP_SQL = """
WITH auto_apps AS (
  SELECT app_id FROM tenant_settings
  LEFT JOIN envs USING (tenant_id) INNER JOIN apps USING (env_id)
  WHERE auto_tests_enabled
)
SELECT ma.metric_id, ma.asset_value, mc.metric_type,
       t.test_id IS NOT NULL AS has_auto_test
FROM metrics_agg ma
JOIN metrics_conf mc ON mc.metric_id = ma.metric_id
JOIN auto_apps a ON a.app_id = mc.app_id
LEFT JOIN tests t ON t.metric_id = ma.metric_id AND t.asset_value = ma.asset_value
     AND t.is_enabled AND t.is_auto
WHERE ma.max_app_pit >= now() - interval '30 days'
  AND ma.total_metrics >= 5
  AND ma.max_app_pit - ma.min_app_pit >= interval '1 day'
  AND mc.is_metric_enabled AND NOT mc.is_temporary
"""

VALUES_SQL = """
SELECT metric_id, asset_value, app_pit, metric_value
FROM (
  SELECT m.metric_id, m.asset_value, m.app_pit, m.metric_value,
         ROW_NUMBER() OVER (PARTITION BY m.metric_id, m.asset_value, m.app_pit
                            ORDER BY m.end_time DESC) AS rn
  FROM metrics m
  JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
               UNNEST(%(avs)s::text[]) AS asset_value) u
    USING (metric_id, asset_value)
  WHERE m.app_pit >= now() - interval '88 days' AND m.metric_value IS NOT NULL
) x WHERE rn = 1
"""


def md5_key(mid: int, av: str) -> str:
    return hashlib.md5(f"{mid}{av}".encode()).hexdigest()


def simulate(series_df: pd.DataFrame, now: pd.Timestamp) -> dict | None:
    series_df = series_df.sort_values("app_pit")
    ts = pd.DatetimeIndex(series_df["app_pit"])
    y = series_df["metric_value"].astype(float).values
    predicted = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        predicted[t] = np.mean(y[max(0, t - TRAILING_WINDOW):t])
    rel_err = np.abs(y - predicted) / np.maximum(np.abs(predicted), 1e-9)

    eval_start = now - pd.Timedelta(days=EVAL_DAYS)
    flag_seq: list[bool] = []
    for week in range(5):
        seg_lo = eval_start + pd.Timedelta(days=7 * week)
        seg_hi = min(seg_lo + pd.Timedelta(days=7), now)
        if seg_lo >= now:
            break
        seg = (ts >= seg_lo) & (ts < seg_hi) & ~np.isnan(predicted)
        cal = ((ts >= seg_lo - pd.Timedelta(days=CAL_DAYS)) & (ts < seg_lo)
               & ~np.isnan(predicted))
        if seg.sum() == 0:
            continue
        if cal.sum() < MIN_CAL:
            continue
        errs = np.sort(rel_err[cal])
        rank = min(len(errs) - 1,
                   int(np.ceil((len(errs) + 1) * (1 - MISS_BUDGET))) - 1)
        tol = errs[rank]
        flag_seq.extend(bool(f) for f in rel_err[seg] > tol)
    if len(flag_seq) < MIN_EVAL:
        return None
    episodes = sum(1 for i, f in enumerate(flag_seq)
                   if f and (i == 0 or not flag_seq[i - 1]))
    return {"n_runs": len(flag_seq), "n_flagged": int(np.sum(flag_seq)),
            "n_episodes": episodes}


def main() -> None:
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 600000")

    cur.execute(POP_SQL)
    pop = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "metric_type", "has_auto_test"])
    pop["asset_value"] = pop["asset_value"].fillna("")
    wc = pd.read_csv(HERE / "window_counts.csv")
    wc["asset_value"] = wc["asset_value"].fillna("")
    pop = pop.merge(wc, on=["metric_id", "asset_value"], how="left")
    pop[["cnt_eval", "cnt_cal"]] = pop[["cnt_eval", "cnt_cal"]].fillna(0)
    pop["rel_eligible"] = (pop["cnt_cal"] >= MIN_CAL) & (pop["cnt_eval"] >= MIN_EVAL)
    print(f"population {len(pop)}, rel_eligible {pop['rel_eligible'].sum()} "
          f"({pop['rel_eligible'].mean():.1%})")
    pop.to_csv(HERE / "stage2_population.csv", index=False)

    elig = pop[pop["rel_eligible"]].copy()
    freq = elig["cnt_eval"] / 30.0
    elig["tier"] = np.select([freq >= 3, freq >= 0.8, freq >= 0.15],
                             ["hf", "daily", "weekly"], "sparse")
    elig["md5"] = [md5_key(m, a) for m, a in
                   zip(elig["metric_id"], elig["asset_value"])]
    shares = elig["tier"].value_counts(normalize=True)
    sample = pd.concat([
        elig[elig["tier"] == t].sort_values("md5").head(
            max(1, int(round(SAMPLE_N * share))))
        for t, share in shares.items()])
    print(f"sample {len(sample)}: {sample['tier'].value_counts().to_dict()}")

    now = pd.Timestamp.now()
    results = []
    recs = sample[["metric_id", "asset_value", "tier", "has_auto_test"]].values
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i:i + CHUNK]
        cur.execute(VALUES_SQL, {"mids": [int(r[0]) for r in chunk],
                                 "avs": [str(r[1]) for r in chunk]})
        vals = pd.DataFrame(cur.fetchall(), columns=[
            "metric_id", "asset_value", "app_pit", "metric_value"])
        vals["asset_value"] = vals["asset_value"].fillna("")
        for r in chunk:
            s = vals[(vals["metric_id"] == r[0]) & (vals["asset_value"] == r[1])]
            if s.empty:
                continue
            out = simulate(s, now)
            if out is not None:
                out.update({"metric_id": r[0], "tier": r[2],
                            "has_auto_test": bool(r[3])})
                results.append(out)
        if (i // CHUNK) % 10 == 0:
            print(f"{i + len(chunk)}/{len(recs)} series, "
                  f"{time.time() - START:.0f}s", flush=True)
    conn.close()

    df = pd.DataFrame(results)
    df["rate"] = df["n_flagged"] / df["n_runs"]
    df.to_csv(HERE / "stage2_rel_sim_results.csv", index=False)

    print(f"\nsimulated series: {len(df)}")
    print(f"overall runs flagged: {df['n_flagged'].sum() / df['n_runs'].sum():.3%} "
          f"({df['n_flagged'].sum()}/{df['n_runs'].sum()})")
    for name, mask in [("all", df["rate"].notna()),
                       ("currently tested", df["has_auto_test"]),
                       ("currently untested", ~df["has_auto_test"])]:
        g = df[mask]
        if g.empty:
            continue
        print(f"\n{name} (n={len(g)}):")
        print(f"  >=3%: {(g['rate'] >= 0.03).mean():.1%}   "
              f"1-3%: {((g['rate'] >= 0.01) & (g['rate'] < 0.03)).mean():.1%}   "
              f"<1%: {(g['rate'] < 0.01).mean():.1%}   "
              f"zero: {(g['rate'] == 0).mean():.1%}")


if __name__ == "__main__":
    main()
