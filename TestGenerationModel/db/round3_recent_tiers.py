"""Round 3: identical benchmark to round 2, but tiers are assigned by RECENT
cadence (reports in the last 45 days / observed days), not lifetime average.

Motivation: 42% of series change tier under recent-window classification;
77% of lifetime-"weekly" series actually report daily now. This rerun asks
whether the round-2 conclusions survive honest tier assignment.

Sampling: a deterministic md5-ordered pool of 6000 active auto-test series is
classified by recent cadence, then per-tier quotas (same as round 2:
hf 120 / daily 530 / weekly 480 / sparse 70) are taken in pool order.
Windows, grains, configs, and scoring are identical to round 2 (imported).
"""

import time
from pathlib import Path

import pandas as pd
import psycopg2

from round2_representative import (
    DAILY_SQL, HOURLY_SQL, STRATA, fetch, run_eval,
)

HERE = Path(__file__).resolve().parent
DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")

POOL_SIZE = 6000
RECENT_DAYS = 45
FREQ_HF, FREQ_DAILY, FREQ_WEEKLY = 3.0, 0.8, 0.15
START = time.time()

POOL_SQL = f"""
WITH pop AS (
  SELECT ma.metric_id, ma.asset_value, mc.metric_type, mc.app_id,
         EXTRACT(EPOCH FROM (now() - ma.min_app_pit))/86400.0 AS age_d
  FROM metrics_agg ma
  JOIN metrics_conf mc ON mc.metric_id = ma.metric_id
  JOIN tests t ON t.metric_id = ma.metric_id AND t.asset_value = ma.asset_value
       AND t.is_auto AND t.is_enabled
  WHERE ma.total_metrics >= 5 AND ma.max_app_pit >= now() - interval '30 days'
    AND ma.max_app_pit - ma.min_app_pit >= interval '1 day'
  ORDER BY md5(ma.metric_id::text || ma.asset_value)
  LIMIT {POOL_SIZE}
)
SELECT p.metric_id, p.asset_value, p.metric_type, p.app_id, p.age_d,
       (SELECT COUNT(*) FROM metrics m
        WHERE m.metric_id = p.metric_id AND m.asset_value = p.asset_value
          AND m.app_pit >= now() - interval '{RECENT_DAYS} days') AS cnt_recent
FROM pop p
"""


def recent_tier(freq: float) -> str:
    if freq >= FREQ_HF:
        return "hf"
    if freq >= FREQ_DAILY:
        return "daily"
    if freq >= FREQ_WEEKLY:
        return "weekly"
    return "sparse"


def main() -> None:
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")

    cur.execute(POOL_SQL)
    pool = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "metric_type", "app_id", "age_d",
        "cnt_recent"])
    pool["asset_value"] = pool["asset_value"].fillna("")
    pool["age_d"] = pool["age_d"].astype(float)
    window_days = pool["age_d"].clip(upper=RECENT_DAYS).clip(lower=1)
    pool["recent_freq"] = pool["cnt_recent"].astype(float) / window_days
    pool["tier"] = pool["recent_freq"].map(recent_tier)
    print(f"pool classified ({time.time() - START:.0f}s):")
    print(pool["tier"].value_counts().to_string())

    frames, samples = {}, {}
    for tier, quota in STRATA.items():
        sample = pool[pool["tier"] == tier].head(quota)
        if len(sample) < quota:
            print(f"NOTE: {tier} quota {quota}, pool only had {len(sample)}")
        samples[tier] = sample[["metric_id", "asset_value", "metric_type",
                                "app_id"]].reset_index(drop=True)
        sql = HOURLY_SQL if tier == "hf" else DAILY_SQL
        cols = (["metric_id", "asset_value", "day", "hr", "val"] if tier == "hf"
                else ["metric_id", "asset_value", "day", "val"])
        frames[tier] = fetch(cur, sql, samples[tier], cols)
        frames[tier].to_csv(HERE / f"round3_frame_{tier}.csv", index=False)
        samples[tier].to_csv(HERE / f"round3_sample_{tier}.csv", index=False)
        print(f"{tier}: {len(samples[tier])} series, {len(frames[tier])} rows "
              f"({time.time() - START:.0f}s)", flush=True)
    conn.close()

    run_eval(frames, samples, HERE / "round3_results.csv")


if __name__ == "__main__":
    main()
