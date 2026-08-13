"""How many existing enabled auto tests sit on hour-of-day-seasonal metrics?

1. Count the population: all enabled auto tests, and those on high-frequency
   series (>= 3 runs/day, >= 21 days span, >= 200 points) where hour-of-day
   seasonality is even observable.
2. Sample 500 of those series, fetch day-hour averages (last 90 days),
   run the same hour-of-day battery (KW on 6h buckets, day-block permutation,
   effect size, band inflation).
3. Extrapolate the robust fraction to the population (Wilson 95% CI).
4. Compare alert rates (total_alerts / total_metrics) between seasonal and
   non-seasonal series, and configured Range width vs per-bucket width.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from db_seasonality_analysis import analyze_hod, bh_fdr  # noqa: E402

DB = "postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l.eu-north-1.rds.amazonaws.com:5432/app"
CHUNK = 40
SAMPLE_N = 500

HF_COND = """
  ma.max_app_pit - ma.min_app_pit >= interval '21 days'
  AND ma.total_metrics >= 200
  AND ma.total_metrics / GREATEST(EXTRACT(EPOCH FROM (ma.max_app_pit - ma.min_app_pit))/86400.0, 1) >= 3
  AND ma.max_app_pit >= now() - interval '30 days'
"""

COUNTS_SQL = f"""
SELECT
  (SELECT COUNT(*) FROM tests WHERE is_auto AND is_enabled) AS auto_tests_total,
  (SELECT COUNT(*)
   FROM tests t JOIN metrics_agg ma USING (metric_id, asset_value)
   WHERE t.is_auto AND t.is_enabled AND {HF_COND}) AS auto_tests_hf
"""

SAMPLE_SQL = f"""
SELECT t.test_id, t.metric_id, t.asset_value, t.test_type, t.var1, t.var2,
       mc.metric_type, ma.total_metrics, ma.total_alerts,
       EXTRACT(EPOCH FROM (ma.max_app_pit - ma.min_app_pit))/86400.0 AS span_days
FROM tests t
JOIN metrics_agg ma USING (metric_id, asset_value)
JOIN metrics_conf mc ON mc.metric_id = t.metric_id
WHERE t.is_auto AND t.is_enabled AND {HF_COND}
ORDER BY md5(t.metric_id::text || t.asset_value)
LIMIT {SAMPLE_N}
"""

HOURLY_SQL = """
SELECT m.metric_id, m.asset_value, date_trunc('day', m.app_pit)::date AS day,
       EXTRACT(HOUR FROM m.app_pit)::int AS hr,
       AVG(m.metric_value) AS avg_val, COUNT(*) AS n
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) s
  USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL AND m.app_pit >= now() - interval '90 days'
GROUP BY 1, 2, 3, 4
"""


def wilson_ci(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return (np.nan, np.nan)
    z = 1.96
    p = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def main() -> None:
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")

    cur.execute(COUNTS_SQL)
    total_auto, total_hf = cur.fetchone()
    print(f"enabled auto tests total: {total_auto}")
    print(f"on high-frequency, recently-active series (hour-seasonality observable): "
          f"{total_hf} ({total_hf / total_auto:.1%})")

    cur.execute(SAMPLE_SQL)
    cols = [d[0] for d in cur.description]
    tests = pd.DataFrame(cur.fetchall(), columns=cols)
    tests["asset_value"] = tests["asset_value"].fillna("")
    print(f"sampled {len(tests)} auto tests")

    rows = []
    t0 = time.time()
    recs = tests[["metric_id", "asset_value"]].values
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i : i + CHUNK]
        cur.execute(HOURLY_SQL, {"mids": [int(r[0]) for r in chunk],
                                 "avs": [r[1] for r in chunk]})
        rows.extend(cur.fetchall())
        if (i // CHUNK) % 4 == 0:
            print(f"fetched {i + len(chunk)}/{len(recs)} series, "
                  f"{len(rows)} rows, {time.time() - t0:.0f}s", flush=True)
    conn.close()

    hourly = pd.DataFrame(rows, columns=["metric_id", "asset_value", "day",
                                         "hr", "avg_val", "n"])
    hourly["asset_value"] = hourly["asset_value"].fillna("")
    hourly["avg_val"] = hourly["avg_val"].astype(float)
    hourly["day"] = pd.to_datetime(hourly["day"])
    hourly.to_csv(HERE / "auto_tests_hourly.csv", index=False)

    results = []
    for (mid, av), sdf in hourly.groupby(["metric_id", "asset_value"]):
        r = analyze_hod(sdf)
        if r is None:
            continue
        r.update({"metric_id": mid, "asset_value": av})
        results.append(r)
    res = pd.DataFrame(results)
    res["sig"] = bh_fdr(res["p"])
    res["robust"] = res["sig"] & (res["effect"] > 0.10) & (res["perm_p"] < 0.05)
    res["material"] = res["robust"] & (res["inflation"] > 1.3)

    merged = tests.merge(res, on=["metric_id", "asset_value"], how="left")
    merged.to_csv(HERE / "auto_tests_hour_seasonality.csv", index=False)

    testable = merged[merged["p"].notna()]
    n_t = len(testable)
    n_robust = int(testable["robust"].sum())
    n_material = int(testable["material"].sum())
    lo, hi = wilson_ci(n_robust, n_t)
    lo_m, hi_m = wilson_ci(n_material, n_t)

    print(f"\ntestable sampled tests: {n_t}")
    print(f"on robust hour-seasonal metrics: {n_robust} ({n_robust / n_t:.1%}, "
          f"95% CI {lo:.1%}-{hi:.1%})")
    print(f"  -> extrapolated to population: {int(lo * total_hf)}-{int(hi * total_hf)} "
          f"of {total_hf} HF auto tests (point est {int(n_robust / n_t * total_hf)})")
    print(f"on material (>1.3x band cost) metrics: {n_material} ({n_material / n_t:.1%}, "
          f"CI {lo_m:.1%}-{hi_m:.1%}) -> {int(lo_m * total_hf)}-{int(hi_m * total_hf)} tests")

    print("\nrobust by test_type of the existing auto test:")
    print(testable.groupby("test_type").agg(
        n=("test_id", "size"), robust=("robust", "sum"),
        robust_pct=("robust", "mean")).to_string())

    print("\nrobust by metric_type (top 10 by count):")
    rb = testable[testable["robust"]]
    if len(rb):
        print(rb.groupby("metric_type").agg(
            n=("test_id", "size"), med_effect=("effect", "median"),
            med_inflation=("inflation", "median"))
            .sort_values("n", ascending=False).head(10).to_string())

    # Alert-rate comparison
    testable = testable.assign(
        alert_rate=testable["total_alerts"].astype(float)
        / testable["total_metrics"].astype(float))
    a = testable[testable["robust"]]["alert_rate"].values
    b = testable[~testable["robust"]]["alert_rate"].values
    if len(a) >= 5 and len(b) >= 5:
        mw = stats.mannwhitneyu(a, b)
        print(f"\nalert rate (total_alerts/total_metrics):")
        print(f"  hour-seasonal metrics:     median {np.median(a):.4f}, mean {a.mean():.4f} (n={len(a)})")
        print(f"  non-seasonal metrics:      median {np.median(b):.4f}, mean {b.mean():.4f} (n={len(b)})")
        print(f"  Mann-Whitney p = {mw.pvalue:.4g}")

    # Range width vs bucket width for robust Range tests
    rng_tests = testable[(testable["test_type"] == "Range") & testable["robust"]]
    ratios = []
    for _, t in rng_tests.iterrows():
        sdf = hourly[(hourly["metric_id"] == t["metric_id"])
                     & (hourly["asset_value"] == t["asset_value"])]
        if sdf.empty or pd.isna(t["var1"]) or pd.isna(t["var2"]):
            continue
        width_cfg = float(t["var2"]) - float(t["var1"])
        buckets = sdf.assign(bucket=sdf["hr"] // 6).groupby("bucket")["avg_val"]
        widths = [np.percentile(g, 95) - np.percentile(g, 5)
                  for _, g in buckets if len(g) >= 10]
        if widths and np.mean(widths) > 0 and width_cfg > 0:
            ratios.append(width_cfg / np.mean(widths))
    if ratios:
        print(f"\nconfigured Range width vs per-hour-bucket p5-p95 width "
              f"(robust seasonal Range tests, n={len(ratios)}):")
        print(f"  median ratio {np.median(ratios):.2f}x, "
              f"p75 {np.percentile(ratios, 75):.2f}x, max {max(ratios):.1f}x")


if __name__ == "__main__":
    main()
