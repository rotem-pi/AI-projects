#!/bin/zsh
# Full-population windowed per-series counts (rel-band eligibility, exact).
# One sequential pass over the last 88 days of the metrics table.
export PGOPTIONS='-c default_transaction_read_only=on -c statement_timeout=1800000'
DB='postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l.eu-north-1.rds.amazonaws.com:5432/app'
cd "$(dirname "$0")"
psql "$DB" -c "\copy (
SELECT metric_id, asset_value,
       COUNT(*) FILTER (WHERE app_pit >= now() - interval '30 days') AS cnt_eval,
       COUNT(*) FILTER (WHERE app_pit <  now() - interval '30 days') AS cnt_cal
FROM (
  SELECT DISTINCT ON (metric_id, asset_value, app_pit) metric_id, asset_value, app_pit
  FROM metrics
  WHERE app_pit >= now() - interval '88 days'
  ORDER BY metric_id, asset_value, app_pit, end_time DESC
) deduped
GROUP BY 1, 2
) TO 'window_counts.csv' CSV HEADER"
wc -l window_counts.csv
