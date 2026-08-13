"""Resilient, checkpointed B/C conversion matrices on a large stratified
sample (10,000 series) of the full enabled-auto-test fleet.

Design for replica instability:
  - Small chunks (25 series/round trip) instead of one giant export.
  - Reconnect-with-backoff around every DB call.
  - Per-series results appended to a checkpoint CSV after each chunk, so a
    crash loses at most one chunk (a few seconds), not the whole run.
  - Re-running the script skips metric_ids already in the checkpoint, so it
    resumes automatically after any interruption.

Sampling frame: full_tests.csv (all 195,876 enabled auto tests, already
exported). Stratified by recent cadence (last 45 days / 45), matching the
tiering used in round3 and stage4, proportional to the full tested fleet.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from stage4_final_eval import (  # noqa: E402
    FALLBACK_MIN_FLAGS, FALLBACK_RATE, band_with_flag,
)
from visual_side_by_side import EVAL_DAYS, incumbent_bounds  # noqa: E402

DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")
SAMPLE_N = 10_000
CHUNK = 25
MAX_RETRIES = 6
CHECKPOINT = HERE / "resilient_checkpoint.csv"
RECENT_DAYS = 45

CADENCE_SQL = """
SELECT ma.metric_id, ma.asset_value,
       EXTRACT(EPOCH FROM (now() - ma.min_app_pit))/86400.0 AS age_d,
       (SELECT COUNT(*) FROM metrics m
        WHERE m.metric_id = ma.metric_id AND m.asset_value = ma.asset_value
          AND m.app_pit >= now() - interval '{RECENT_DAYS} days') AS cnt_recent
FROM metrics_agg ma
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) u
  ON u.metric_id = ma.metric_id AND u.asset_value = ma.asset_value
""".format(RECENT_DAYS=RECENT_DAYS)

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
  WHERE m.app_pit >= now() - interval '180 days' AND m.metric_value IS NOT NULL
) x WHERE rn = 1
"""


def connect():
    conn = psycopg2.connect(DB, connect_timeout=15)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 60000")
    return conn, cur


def with_retry(fn, *args, **kwargs):
    global _conn, _cur
    for attempt in range(MAX_RETRIES):
        try:
            return fn(_cur, *args, **kwargs)
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            wait = min(30, 2 ** attempt)
            print(f"  DB error ({e.__class__.__name__}), reconnecting in "
                  f"{wait}s (attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
            time.sleep(wait)
            try:
                _conn.close()
            except Exception:
                pass
            _conn, _cur = connect()
    raise RuntimeError("exhausted retries")


def fetch_cadence(cur, chunk: pd.DataFrame) -> pd.DataFrame:
    cur.execute(CADENCE_SQL, {"mids": [int(m) for m in chunk["metric_id"]],
                             "avs": [str(a) for a in chunk["asset_value"]]})
    return pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "age_d", "cnt_recent"])


def fetch_values(cur, chunk: np.ndarray) -> pd.DataFrame:
    cur.execute(VALUES_SQL, {"mids": [int(r[0]) for r in chunk],
                            "avs": [str(r[1]) for r in chunk]})
    return pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "app_pit", "metric_value"])


CANDIDATE_POOL = 3 * SAMPLE_N  # oversample before tiering, not the full 196K


def build_sample() -> pd.DataFrame:
    tests = pd.read_csv(HERE / "full_tests.csv")
    tests["asset_value"] = tests["asset_value"].fillna("")
    tests = tests.drop_duplicates(["metric_id", "asset_value"])
    print(f"tests frame: {len(tests)}", flush=True)

    # Random candidate pool FIRST (cheap, local) - cadence lookup only runs
    # against this pool, not the full 196K-row population.
    candidates = tests.sample(min(CANDIDATE_POOL, len(tests)), random_state=1337)
    print(f"candidate pool for cadence lookup: {len(candidates)}", flush=True)

    cadence_rows = []
    recs = candidates[["metric_id", "asset_value"]].values
    for i in range(0, len(recs), 500):
        chunk = pd.DataFrame(recs[i:i + 500], columns=["metric_id", "asset_value"])
        cadence_rows.append(with_retry(fetch_cadence, chunk))
        if i % 5000 == 0:
            print(f"cadence lookup {i}/{len(recs)}", flush=True)
    cadence = pd.concat(cadence_rows, ignore_index=True)
    cadence["asset_value"] = cadence["asset_value"].fillna("")
    cadence["age_d"] = cadence["age_d"].astype(float)
    window = cadence["age_d"].clip(upper=RECENT_DAYS).clip(lower=1)
    cadence["freq"] = cadence["cnt_recent"].astype(float) / window
    cadence["tier"] = np.select(
        [cadence["freq"] >= 3, cadence["freq"] >= 0.8, cadence["freq"] >= 0.15],
        ["hf", "daily", "weekly"], "sparse")

    merged = candidates.merge(cadence[["metric_id", "asset_value", "tier"]],
                              on=["metric_id", "asset_value"], how="inner")
    shares = merged["tier"].value_counts(normalize=True)
    rng = np.random.default_rng(1337)
    sampled = pd.concat([
        merged[merged["tier"] == t].sample(
            min(len(merged[merged["tier"] == t]),
                max(1, int(round(SAMPLE_N * share)))),
            random_state=1337)
        for t, share in shares.items()
    ])
    print(f"sample: {len(sampled)} -> {sampled['tier'].value_counts().to_dict()}",
          flush=True)
    return sampled.reset_index(drop=True)


def load_checkpoint() -> set:
    if not CHECKPOINT.exists():
        return set()
    done = pd.read_csv(CHECKPOINT)
    return set(zip(done["metric_id"], done["asset_value"].fillna("")))


def append_checkpoint(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    df.to_csv(CHECKPOINT, mode="a", header=not CHECKPOINT.exists(), index=False)


def main() -> None:
    global _conn, _cur
    _conn, _cur = connect()

    sample = build_sample()
    done = load_checkpoint()
    print(f"already checkpointed: {len(done)}", flush=True)

    recs = sample[["metric_id", "asset_value", "test_type", "var1", "var2",
                  "var3", "metric_type"]].values
    todo = [r for r in recs if (r[0], r[1]) not in done]
    print(f"remaining: {len(todo)}", flush=True)

    t0 = time.time()
    for i in range(0, len(todo), CHUNK):
        chunk = todo[i:i + CHUNK]
        key_chunk = np.array([[r[0], r[1]] for r in chunk])
        try:
            vals = with_retry(fetch_values, key_chunk)
        except RuntimeError:
            print("giving up on this chunk after retries, continuing",
                  flush=True)
            continue
        vals["asset_value"] = vals["asset_value"].fillna("")
        rows = []
        for r in chunk:
            mid, av, ttype, v1, v2, v3, mtype = r
            s = vals[(vals["metric_id"] == mid)
                     & (vals["asset_value"] == av)].sort_values("app_pit")
            if len(s) < 20:
                rows.append({"metric_id": mid, "asset_value": av, "skip": True})
                continue
            ts = pd.DatetimeIndex(s["app_pit"])
            y = s["metric_value"].astype(float).values
            eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
            t_dict = {"metric_id": mid, "asset_value": av, "test_type": ttype,
                      "var1": v1, "var2": v2, "var3": v3, "metric_type": mtype}
            try:
                ilo, ihi = incumbent_bounds(pd.Series(t_dict), list(ts), y)
            except Exception:
                rows.append({"metric_id": mid, "asset_value": av, "skip": True})
                continue
            nlo, nhi, _ = band_with_flag(ts, y)
            ok = eval_mask & ~np.isnan(ilo) & ~np.isnan(nlo)
            if ok.sum() < 5:
                rows.append({"metric_id": mid, "asset_value": av, "skip": True})
                continue
            old_flags = ((y < ilo) | (y > ihi))[ok]
            new_flags = ((y < nlo) | (y > nhi))[ok]
            rate = new_flags.mean()
            fell_back = rate > FALLBACK_RATE and new_flags.sum() >= FALLBACK_MIN_FLAGS
            if fell_back:
                new_flags = old_flags
            rows.append({
                "metric_id": mid, "asset_value": av, "skip": False,
                "n_runs": int(ok.sum()),
                "old_flagged": int(old_flags.sum()),
                "new_flagged": int(new_flags.sum()),
                "old_any": bool(old_flags.any()), "new_any": bool(new_flags.any()),
                "both_00": int(((~old_flags) & (~new_flags)).sum()),
                "old0_new1": int(((~old_flags) & new_flags).sum()),
                "old1_new0": int((old_flags & (~new_flags)).sum()),
                "both_11": int((old_flags & new_flags).sum()),
                "fell_back": fell_back,
            })
        append_checkpoint(rows)
        n_done = len(done) + i + len(chunk)
        if (i // CHUNK) % 20 == 0:
            elapsed = time.time() - t0
            rate = (i + len(chunk)) / max(elapsed, 1)
            eta = (len(todo) - i - len(chunk)) / max(rate, 0.01)
            print(f"{n_done}/{len(sample)} series checkpointed "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)", flush=True)

    _conn.close()
    print(f"\ndone in {time.time() - t0:.0f}s. Run "
          f"aggregate_resilient.py to compute the matrices.")


if __name__ == "__main__":
    main()
