"""Population stats from the prod read replica, for comparing the labeled
test datasets against the real metric population.

Read-only. Two cheap aggregates:
1. metric_type / group mix of recently-active series (metrics_agg x metrics_conf)
2. lifetime depth percentiles of recently-active series (metrics_agg)

Run from backend/:  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/population_stats.py
"""

import json
from pathlib import Path

import psycopg

DB = (
    "postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
    ".eu-north-1.rds.amazonaws.com:5432/app"
)
OUT = Path(__file__).resolve().parent

TYPE_MIX_SQL = """
SELECT mc.metric_type, COUNT(*) AS n_series
FROM metrics_agg ma
JOIN metrics_conf mc USING (metric_id)
WHERE ma.max_app_pit >= now() - interval '30 days'
GROUP BY 1
ORDER BY 2 DESC
"""

DEPTH_SQL = """
SELECT
  COUNT(*) AS n_series,
  percentile_cont(0.25) WITHIN GROUP (ORDER BY total_metrics) AS p25_points,
  percentile_cont(0.5)  WITHIN GROUP (ORDER BY total_metrics) AS p50_points,
  percentile_cont(0.75) WITHIN GROUP (ORDER BY total_metrics) AS p75_points,
  percentile_cont(0.5) WITHIN GROUP (
    ORDER BY EXTRACT(epoch FROM max_app_pit - min_app_pit) / 86400.0
  ) AS p50_span_days,
  AVG((EXTRACT(epoch FROM max_app_pit - min_app_pit) / 86400.0 >= 63)::int)
    AS share_span_ge_63d
FROM metrics_agg
WHERE max_app_pit >= now() - interval '30 days'
"""

with psycopg.connect(DB, autocommit=True) as conn:
    with conn.cursor() as cur:
        cur.execute("SET default_transaction_read_only = on")
        cur.execute("SET statement_timeout = '600s'")
        cur.execute(TYPE_MIX_SQL)
        type_mix = [
            {"metric_type": r[0], "n_series": int(r[1])} for r in cur.fetchall()
        ]
        cur.execute(DEPTH_SQL)
        cols = [d.name for d in cur.description]
        depth = dict(zip(cols, [float(v) if v is not None else None for v in cur.fetchone()]))

result = {"type_mix": type_mix, "depth": depth}
(OUT / "population_stats.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
