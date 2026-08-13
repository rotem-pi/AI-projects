#!/bin/zsh
# Full-data B/C matrices: export run-level values for ALL enabled-auto-tested
# series (88 days, deduped per PIT), plus test configs; then compute locally.
set -e
cd "$(dirname "$0")"
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=3000000'
DB='postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l.eu-north-1.rds.amazonaws.com:5432/app'

psql "$DB" -c "\copy (
SELECT t.metric_id, t.asset_value, t.test_type, t.var1, t.var2, t.var3,
       mc.metric_type
FROM tests t JOIN metrics_conf mc ON mc.metric_id = t.metric_id
WHERE t.is_auto AND t.is_enabled
) TO 'full_tests.csv' CSV HEADER"
wc -l full_tests.csv

psql "$DB" -c "\copy (
SELECT metric_id, asset_value, app_pit, metric_value FROM (
  SELECT m.metric_id, m.asset_value, m.app_pit, m.metric_value,
         ROW_NUMBER() OVER (PARTITION BY m.metric_id, m.asset_value, m.app_pit
                            ORDER BY m.end_time DESC) AS rn
  FROM metrics m
  JOIN tests t ON t.metric_id = m.metric_id AND t.asset_value = m.asset_value
       AND t.is_auto AND t.is_enabled
  WHERE m.app_pit >= now() - interval '88 days' AND m.metric_value IS NOT NULL
) x WHERE rn = 1
ORDER BY metric_id, asset_value, app_pit
) TO 'full_values.csv' CSV HEADER"
wc -l full_values.csv

cd ../../../backend
uv run python ../analysis_temp/seasonality-check/db/full_matrices_compute.py
