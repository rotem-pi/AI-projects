"""Answer two follow-ups:
1. Rel-band coverage if the registry allow-list is added to its eligibility.
2. Paired comparison: incumbent (real test_runs) vs rel band (simulated) on
   the SAME series, same last-30-days window.
"""

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.tests_gen.metrics.metrics_registry import MetricsRegistry  # noqa: E402

SAMPLE_N = 2000
MIN_CAL, MIN_EVAL = 10, 5

TEST_RUNS_SQL = """
SELECT t.metric_id, t.asset_value,
       COUNT(*) AS n_runs, COUNT(*) FILTER (WHERE NOT tr.is_passed) AS n_failed
FROM test_runs tr
JOIN tests t ON t.test_id = tr.test_id AND t.is_auto AND t.is_enabled
JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
             UNNEST(%(avs)s::text[]) AS asset_value) u
  ON u.metric_id = t.metric_id AND u.asset_value = t.asset_value
WHERE tr.app_pit >= now() - interval '30 days'
GROUP BY 1, 2
"""


def rebuild_sample(pop: pd.DataFrame) -> pd.DataFrame:
    elig = pop[pop["rel_eligible"]].copy()
    freq = elig["cnt_eval"] / 30.0
    elig["tier"] = np.select([freq >= 3, freq >= 0.8, freq >= 0.15],
                             ["hf", "daily", "weekly"], "sparse")
    elig["md5"] = [hashlib.md5(f"{m}{a}".encode()).hexdigest()
                   for m, a in zip(elig["metric_id"], elig["asset_value"])]
    shares = elig["tier"].value_counts(normalize=True)
    return pd.concat([
        elig[elig["tier"] == t].sort_values("md5").head(
            max(1, int(round(SAMPLE_N * share))))
        for t, share in shares.items()])


def main() -> None:
    pop = pd.read_csv(HERE / "stage2_population.csv")
    pop["asset_value"] = pop["asset_value"].fillna("")

    # --- Q1: rel coverage with the registry allow-list added ---
    registry = set(MetricsRegistry.get_instance().registry.keys())
    registry_ok = (pop["metric_type"].str.lower().isin(registry)
                   | pop["metric_type"].str.startswith("custom."))
    both = pop["rel_eligible"] & registry_ok
    print(f"population: {len(pop)}")
    print(f"rel eligible:                  {pop['rel_eligible'].sum()} "
          f"({pop['rel_eligible'].mean():.1%})")
    print(f"rel eligible AND registry:     {both.sum()} ({both.mean():.1%})")
    print(f"registry-only (any history):   {registry_ok.sum()} "
          f"({registry_ok.mean():.1%})")

    # --- Q2: paired comparison on identical series ---
    sample = rebuild_sample(pop)
    sim = pd.read_csv(HERE / "stage2_rel_sim_results.csv")
    unique_in_sample = sample.drop_duplicates("metric_id", keep=False)
    unique_in_sim = sim.drop_duplicates("metric_id", keep=False)
    merged = unique_in_sim.merge(
        unique_in_sample[["metric_id", "asset_value"]], on="metric_id")
    tested = merged[merged["has_auto_test"]]
    print(f"\npaired candidates (unique-key, currently tested): {len(tested)}")

    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    cur.execute(TEST_RUNS_SQL, {
        "mids": [int(m) for m in tested["metric_id"]],
        "avs": [str(a) for a in tested["asset_value"]]})
    incumbent = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "inc_runs", "inc_failed"])
    conn.close()

    paired = tested.merge(incumbent, on=["metric_id", "asset_value"])
    paired["inc_rate"] = paired["inc_failed"] / paired["inc_runs"]
    paired.to_csv(HERE / "stage2_paired.csv", index=False)
    n = len(paired)
    print(f"paired series with data on both sides: {n}")
    print(f"\nruns flagged, SAME series, last 30 days:")
    print(f"  incumbent: {paired['inc_failed'].sum() / paired['inc_runs'].sum():.3%} "
          f"({int(paired['inc_failed'].sum())}/{int(paired['inc_runs'].sum())})")
    print(f"  rel band:  {paired['n_flagged'].sum() / paired['n_runs'].sum():.3%} "
          f"({int(paired['n_flagged'].sum())}/{int(paired['n_runs'].sum())})")

    def buckets(rates: pd.Series) -> str:
        return (f">=3%: {(rates >= 0.03).mean():.1%}  "
                f"1-3%: {((rates >= 0.01) & (rates < 0.03)).mean():.1%}  "
                f"<1%: {(rates < 0.01).mean():.1%}  "
                f"zero: {(rates == 0).mean():.1%}")

    print(f"\nper-metric buckets, SAME {n} series:")
    print(f"  incumbent: {buckets(paired['inc_rate'])}")
    print(f"  rel band:  {buckets(paired['rate'])}")
    agree_quiet = ((paired["inc_rate"] < 0.01) & (paired["rate"] < 0.01)).mean()
    rel_hot_inc_quiet = ((paired["rate"] >= 0.03)
                         & (paired["inc_rate"] < 0.01)).mean()
    print(f"\n  both quiet (<1%): {agree_quiet:.1%}; "
          f"rel hot (>=3%) while incumbent quiet: {rel_hot_inc_quiet:.1%}")


if __name__ == "__main__":
    main()
