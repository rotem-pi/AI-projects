"""Conversion matrices: existing system -> guarded band (final variant,
guardrail applied as deployed: fallback series keep their existing test).

A. Test coverage (metric level, extrapolated to the 547K population).
B. Metrics with >=1 anomaly in the last month (tested sample, both methods
   on identical runs).
C. Run level: flagged/not-flagged transitions on identical runs.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from stage4_final_eval import (  # noqa: E402
    FALLBACK_MIN_FLAGS, FALLBACK_RATE, final_band,
)
from visual_side_by_side import (  # noqa: E402
    DB, EVAL_DAYS, TESTS_SQL, VALUES_SQL, incumbent_bounds,
)

POP = 547_171
TESTED_POP = 69_933
CHUNK = 40


def main() -> None:
    pop = pd.read_csv(HERE / "stage2_population.csv")
    pop["asset_value"] = pop["asset_value"].fillna("")
    res4 = pd.read_csv(HERE / "stage4_results.csv")

    # ---- A. coverage matrix ----
    tested_elig = pop[pop["has_auto_test"]]["rel_eligible"].mean()
    untested_elig = pop[~pop["has_auto_test"]]["rel_eligible"].mean()
    fb_tested = res4[res4["has_auto_test"]]["fallback"].mean()
    fb_untested = res4[~res4["has_auto_test"]]["fallback"].mean()

    untested_pop = POP - TESTED_POP
    t_to_guarded = TESTED_POP * tested_elig * (1 - fb_tested)
    t_keeps_legacy = TESTED_POP - t_to_guarded
    u_to_guarded = untested_pop * untested_elig * (1 - fb_untested)
    u_stays_none = untested_pop - u_to_guarded

    print("=== A. TEST COVERAGE CONVERSION (population, 547,171 series) ===")
    print(f"{'':32} {'-> guarded band':>16} {'-> keeps existing':>18} "
          f"{'-> no test':>12}")
    print(f"{'had auto test (69,933)':32} {t_to_guarded:>16,.0f} "
          f"{t_keeps_legacy:>18,.0f} {0:>12,}")
    print(f"{'had no test (477,238)':32} {u_to_guarded:>16,.0f} "
          f"{'-':>18} {u_stays_none:>12,.0f}")
    print(f"(eligibility: tested {tested_elig:.1%}, untested "
          f"{untested_elig:.1%}; fallback: tested {fb_tested:.1%}, "
          f"untested {fb_untested:.1%})")

    # ---- B + C: tested sample, both methods on identical runs ----
    tested_keys = res4[res4["has_auto_test"]][["metric_id", "asset_value"]]
    tested_keys["asset_value"] = tested_keys["asset_value"].fillna("")
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    cur.execute(TESTS_SQL, {"mids": [int(m) for m in tested_keys["metric_id"]],
                            "avs": [str(a) for a in tested_keys["asset_value"]]})
    tests = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "test_type", "var1", "var2", "var3",
        "metric_type"]).drop_duplicates(["metric_id", "asset_value"])

    run_matrix = np.zeros((2, 2), dtype=int)   # [old][new]
    metric_matrix = np.zeros((2, 2), dtype=int)
    n_series = 0
    recs = tests[["metric_id", "asset_value"]].values
    for i in range(0, len(recs), CHUNK):
        chunk = recs[i:i + CHUNK]
        cur.execute(VALUES_SQL, {"mids": [int(r[0]) for r in chunk],
                                 "avs": [str(r[1]) for r in chunk]})
        vals = pd.DataFrame(cur.fetchall(), columns=[
            "metric_id", "asset_value", "app_pit", "metric_value"])
        vals["asset_value"] = vals["asset_value"].fillna("")
        for r in chunk:
            s = vals[(vals["metric_id"] == r[0])
                     & (vals["asset_value"] == r[1])].sort_values("app_pit")
            if len(s) < 20:
                continue
            t_row = tests[(tests["metric_id"] == r[0])
                          & (tests["asset_value"] == r[1])].iloc[0]
            ts = pd.DatetimeIndex(s["app_pit"])
            y = s["metric_value"].astype(float).values
            eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
            try:
                ilo, ihi = incumbent_bounds(t_row, list(ts), y)
            except Exception:
                continue
            nlo, nhi = final_band(ts, y)
            ok = eval_mask & ~np.isnan(ilo) & ~np.isnan(nlo)
            if ok.sum() < 5:
                continue
            old_flags = ((y < ilo) | (y > ihi))[ok]
            new_flags = ((y < nlo) | (y > nhi))[ok]
            # guardrail as deployed: fallback -> keep existing behavior
            rate = new_flags.mean()
            if rate > FALLBACK_RATE and new_flags.sum() >= FALLBACK_MIN_FLAGS:
                new_flags = old_flags
            n_series += 1
            for o, n in zip(old_flags, new_flags):
                run_matrix[int(o)][int(n)] += 1
            metric_matrix[int(old_flags.any())][int(new_flags.any())] += 1
    conn.close()

    print(f"\n=== B. METRICS WITH >=1 ANOMALY LAST MONTH "
          f"(sample of {n_series} tested series, identical runs) ===")
    scale = TESTED_POP / n_series
    print(f"{'':28} {'new: quiet':>14} {'new: >=1 anomaly':>17}")
    print(f"{'old: quiet':28} {metric_matrix[0][0]:>7} "
          f"(~{metric_matrix[0][0] * scale:>7,.0f}) "
          f"{metric_matrix[0][1]:>6} (~{metric_matrix[0][1] * scale:>7,.0f})")
    print(f"{'old: >=1 anomaly':28} {metric_matrix[1][0]:>7} "
          f"(~{metric_matrix[1][0] * scale:>7,.0f}) "
          f"{metric_matrix[1][1]:>6} (~{metric_matrix[1][1] * scale:>7,.0f})")
    print("(~ = extrapolated to the 69,933 tested metrics)")

    total_runs = run_matrix.sum()
    print(f"\n=== C. RUN LEVEL ({total_runs:,} identical runs) ===")
    print(f"{'':24} {'new: not flagged':>17} {'new: flagged':>13}")
    print(f"{'old: not flagged':24} {run_matrix[0][0]:>10,} "
          f"({run_matrix[0][0] / total_runs:.2%}) {run_matrix[0][1]:>7,} "
          f"({run_matrix[0][1] / total_runs:.2%})")
    print(f"{'old: flagged':24} {run_matrix[1][0]:>10,} "
          f"({run_matrix[1][0] / total_runs:.2%}) {run_matrix[1][1]:>7,} "
          f"({run_matrix[1][1] / total_runs:.2%})")


if __name__ == "__main__":
    main()
