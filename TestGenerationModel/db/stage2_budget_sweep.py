"""Find the miss budget at which the rel band's alert rate matches the
incumbent, and measure what sensitivity survives there.

On the 274 paired series (same series, same month, both methods measurable):
sweep budgets, and at each report run-flag rate, episodes/month, purged
episodes/month (episodes of >= 2 consecutive runs, mirroring the incumbent's
single-blip purge), and injected +50% / +100% detection. Incumbent reference:
real test_runs flag rate and real incidents opened for these same series.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")
BUDGETS = [0.01, 0.005, 0.003, 0.002, 0.001]
TRAILING_WINDOW = 3
CAL_DAYS = 56
EVAL_DAYS = 30
MIN_CAL, MIN_EVAL = 10, 5
CHUNK = 40

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

INCIDENTS_SQL = """
SELECT COUNT(*) FROM incidents i
JOIN tests t ON t.test_id = i.test_id AND t.is_auto
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) u
  ON u.metric_id = t.metric_id AND u.asset_value = t.asset_value
WHERE i.created_time >= now() - interval '30 days'
"""


def simulate_budgets(series_df: pd.DataFrame, now: pd.Timestamp) -> dict | None:
    series_df = series_df.sort_values("app_pit")
    ts = pd.DatetimeIndex(series_df["app_pit"])
    y = series_df["metric_value"].astype(float).values
    predicted = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        predicted[t] = np.mean(y[max(0, t - TRAILING_WINDOW):t])
    scale = np.maximum(np.abs(predicted), 1e-9)
    rel_err = np.abs(y - predicted) / scale

    eval_start = now - pd.Timedelta(days=EVAL_DAYS)
    out = {b: {"flags": [], "det50": [], "det100": []} for b in BUDGETS}
    n_eval = 0
    for week in range(5):
        seg_lo = eval_start + pd.Timedelta(days=7 * week)
        seg_hi = min(seg_lo + pd.Timedelta(days=7), now)
        if seg_lo >= now:
            break
        seg = np.asarray((ts >= seg_lo) & (ts < seg_hi) & ~np.isnan(predicted))
        cal = np.asarray((ts >= seg_lo - pd.Timedelta(days=CAL_DAYS))
                         & (ts < seg_lo) & ~np.isnan(predicted))
        if seg.sum() == 0 or cal.sum() < MIN_CAL:
            continue
        n_eval += int(seg.sum())
        errs = np.sort(rel_err[cal])
        rel_inj50 = np.abs(y[seg] * 1.5 - predicted[seg]) / scale[seg]
        rel_inj100 = np.abs(y[seg] * 2.0 - predicted[seg]) / scale[seg]
        for b in BUDGETS:
            rank = min(len(errs) - 1,
                       int(np.ceil((len(errs) + 1) * (1 - b))) - 1)
            tol = errs[rank]
            out[b]["flags"].extend(bool(f) for f in rel_err[seg] > tol)
            out[b]["det50"].extend(bool(f) for f in rel_inj50 > tol)
            out[b]["det100"].extend(bool(f) for f in rel_inj100 > tol)
    if n_eval < MIN_EVAL:
        return None

    result = {}
    for b in BUDGETS:
        flags = out[b]["flags"]
        episodes, run = [], 0
        for f in flags + [False]:
            if f:
                run += 1
            elif run:
                episodes.append(run)
                run = 0
        result[b] = {
            "n_runs": len(flags), "n_flagged": int(np.sum(flags)),
            "episodes": len(episodes),
            "purged_episodes": sum(1 for e in episodes if e >= 2),
            "det50": float(np.mean(out[b]["det50"])),
            "det100": float(np.mean(out[b]["det100"]))}
    return result


def main() -> None:
    paired = pd.read_csv(HERE / "stage2_paired.csv")
    paired["asset_value"] = paired["asset_value"].fillna("")
    print(f"paired series: {len(paired)}")
    inc_rate = paired["inc_failed"].sum() / paired["inc_runs"].sum()

    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    cur.execute(INCIDENTS_SQL, {
        "mids": [int(m) for m in paired["metric_id"]],
        "avs": [str(a) for a in paired["asset_value"]]})
    inc_incidents = cur.fetchone()[0]

    now = pd.Timestamp.now()
    agg: dict[float, list] = {b: [] for b in BUDGETS}
    recs = paired[["metric_id", "asset_value"]].values
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
            res = simulate_budgets(s, now)
            if res:
                for b in BUDGETS:
                    agg[b].append(res[b])
    conn.close()

    n = len(agg[BUDGETS[0]])
    print(f"simulated: {n} series")
    print(f"\nINCUMBENT reference (same series): run-flag rate {inc_rate:.3%}, "
          f"incidents opened last 30d: {inc_incidents} "
          f"({inc_incidents / len(paired):.2f}/metric/month)\n")
    print(f"{'budget':>8} {'flag_rate':>10} {'episodes/m':>11} "
          f"{'purged/m':>9} {'det+50%':>8} {'det+100%':>9}")
    for b in BUDGETS:
        rows = agg[b]
        flag_rate = sum(r["n_flagged"] for r in rows) / sum(r["n_runs"] for r in rows)
        eps = sum(r["episodes"] for r in rows) / n
        peps = sum(r["purged_episodes"] for r in rows) / n
        det50 = float(np.mean([r["det50"] for r in rows]))
        det100 = float(np.mean([r["det100"] for r in rows]))
        print(f"{b:>8.1%} {flag_rate:>10.3%} {eps:>11.2f} {peps:>9.2f} "
              f"{det50:>8.1%} {det100:>9.1%}")


if __name__ == "__main__":
    main()
