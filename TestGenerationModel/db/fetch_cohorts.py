"""Chunked, read-only extraction of seasonality cohorts from the prod replica.

Samples series from metrics_agg (cheap), then fetches per-series daily/hourly
aggregates in small batches so Postgres uses the covering index
(metric_id, asset_value, app_pit) INCLUDE (metric_value) instead of a seq scan.
"""

import csv
import sys
import time
from pathlib import Path

import psycopg2

DB = "postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l.eu-north-1.rds.amazonaws.com:5432/app"
OUT = Path(__file__).resolve().parent
CHUNK = 40

COHORTS = {
    "yearly": dict(
        where="ma.max_app_pit - ma.min_app_pit >= interval '730 days' AND ma.total_metrics >= 100",
        limit=None, hourly=False, days_back=None),
    "oneyear": dict(
        where="ma.max_app_pit - ma.min_app_pit >= interval '365 days' "
              "AND ma.max_app_pit - ma.min_app_pit < interval '730 days' AND ma.total_metrics >= 100",
        limit=600, hourly=False, days_back=None),
    "midspan": dict(
        where="ma.max_app_pit - ma.min_app_pit >= interval '120 days' "
              "AND ma.max_app_pit - ma.min_app_pit < interval '365 days' AND ma.total_metrics >= 60",
        limit=1200, hourly=False, days_back=None),
    "hourly": dict(
        where="ma.max_app_pit - ma.min_app_pit >= interval '21 days' AND ma.total_metrics >= 200 "
              "AND ma.total_metrics / GREATEST(EXTRACT(EPOCH FROM (ma.max_app_pit - ma.min_app_pit))/86400.0, 1) >= 3",
        limit=400, hourly=True, days_back=90),
}

SAMPLE_SQL = """
SELECT ma.metric_id, ma.asset_value, mc.metric_type, ma.total_metrics,
       ma.min_app_pit, ma.max_app_pit
FROM metrics_agg ma JOIN metrics_conf mc USING (metric_id)
WHERE {where}
ORDER BY md5(ma.metric_id::text || ma.asset_value)
{limit_clause}
"""

DAILY_SQL = """
SELECT m.metric_id, m.asset_value, date_trunc('day', m.app_pit)::date AS day,
       AVG(m.metric_value) AS avg_val,
       MIN(m.metric_value) AS min_val,
       MAX(m.metric_value) AS max_val,
       COUNT(*) AS n
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) s
  USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL
GROUP BY 1, 2, 3
"""

HOURLY_SQL = """
SELECT m.metric_id, m.asset_value, date_trunc('day', m.app_pit)::date AS day,
       EXTRACT(HOUR FROM m.app_pit)::int AS hr,
       AVG(m.metric_value) AS avg_val,
       COUNT(*) AS n
FROM metrics m
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) s
  USING (metric_id, asset_value)
WHERE m.metric_value IS NOT NULL AND m.app_pit >= now() - interval '%(days_back)s days'
GROUP BY 1, 2, 3, 4
"""


def main() -> None:
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")

    for name, cfg in COHORTS.items():
        t0 = time.time()
        limit_clause = f"LIMIT {cfg['limit']}" if cfg["limit"] else ""
        cur.execute(SAMPLE_SQL.format(where=cfg["where"], limit_clause=limit_clause))
        sample = cur.fetchall()
        with open(OUT / f"sample_{name}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["metric_id", "asset_value", "metric_type", "total_metrics",
                        "min_app_pit", "max_app_pit"])
            w.writerows(sample)
        print(f"[{name}] sampled {len(sample)} series", flush=True)

        sql = HOURLY_SQL if cfg["hourly"] else DAILY_SQL
        out_file = OUT / f"agg_{name}.csv"
        header = (["metric_id", "asset_value", "day", "hr", "avg_val", "n"]
                  if cfg["hourly"]
                  else ["metric_id", "asset_value", "day", "avg_val", "min_val", "max_val", "n"])
        n_rows = 0
        with open(out_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for i in range(0, len(sample), CHUNK):
                chunk = sample[i : i + CHUNK]
                params = {
                    "mids": [r[0] for r in chunk],
                    "avs": [r[1] for r in chunk],
                }
                if cfg["hourly"]:
                    q = sql.replace("%(days_back)s", str(cfg["days_back"]))
                else:
                    q = sql
                cur.execute(q, params)
                rows = cur.fetchall()
                w.writerows(rows)
                n_rows += len(rows)
                if (i // CHUNK) % 5 == 0:
                    print(f"[{name}] {i + len(chunk)}/{len(sample)} series, "
                          f"{n_rows} rows, {time.time() - t0:.0f}s", flush=True)
        print(f"[{name}] DONE: {n_rows} agg rows in {time.time() - t0:.0f}s", flush=True)

    conn.close()


if __name__ == "__main__":
    main()
