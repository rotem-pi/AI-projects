"""Guarded band (v2): trailing-MEDIAN(3) baseline, trimmed conformal
tolerance, clamped to [floor, cap]. Fixes case 2 (contaminated-wide) and
case 3 (microscopic-tight) failure modes.

Validates on the 274 paired series: alert rate, episodes, injected detection,
for a small floor/cap grid; then re-renders the three showcase cases with the
chosen variant and rebuilds the HTML report.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psycopg2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from visual_side_by_side import (  # noqa: E402
    EVAL_DAYS, TESTS_SQL, VALUES_SQL, DB, incumbent_bounds,
)

# Historical tuning script: this was the exploratory floor/outlier_k/alpha
# grid search that produced the constants later locked into
# stage4_final_eval.py's band_with_flag. Kept self-contained (rather than
# importing shared band internals) so it remains runnable as a record of
# that search, independent of the current production band definition.
CAL_DAYS = 56
MIN_CAL = 10


def alert_rate(values, lower, upper, eval_mask):
    ok = eval_mask & ~np.isnan(lower)
    if ok.sum() == 0:
        return np.nan, 0, 0
    breach = (values[ok] < lower[ok]) | (values[ok] > upper[ok])
    return float(breach.mean()), int(breach.sum()), int(ok.sum())


def rel_band_bounds(ts, values, miss_budget=0.005, trailing_window=3):
    predicted = np.full(len(values), np.nan)
    for t in range(1, len(values)):
        predicted[t] = np.mean(values[max(0, t - trailing_window):t])
    scale = np.maximum(np.abs(predicted), 1e-9)
    rel_err = np.abs(values - predicted) / scale
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    week_start, end = ts.min(), ts.max()
    while week_start <= end:
        week_end = week_start + pd.Timedelta(days=7)
        seg = np.asarray((ts >= week_start) & (ts < week_end)
                         & ~np.isnan(predicted))
        cal = np.asarray((ts >= week_start - pd.Timedelta(days=CAL_DAYS))
                         & (ts < week_start) & ~np.isnan(predicted))
        if seg.sum() and cal.sum() >= MIN_CAL:
            errs = np.sort(rel_err[cal])
            rank = min(len(errs) - 1,
                       int(np.ceil((len(errs) + 1) * (1 - miss_budget))) - 1)
            tol = errs[rank]
            lower[seg] = predicted[seg] - tol * scale[seg]
            upper[seg] = predicted[seg] + tol * scale[seg]
        week_start = week_end
    return lower, upper

MISS_BUDGET = 0.002  # v2 dial, retuned so guarded flag rate matches incumbent
# Contamination guard: drop calibration errors that are OUTLIER_K times the
# series' own 90th-percentile error. Clean series lose nothing (v1-identical);
# contaminated series (case 2) shed the spikes. Unconditional trimming was
# rejected: it tightened clean series and tripled the fleet alert rate.
# (floor, outlier_k, alpha); alpha=0 -> tolerance = max clean calibration
# error ("never alert on anything smaller than the worst normal error seen")
GRID = [(0.10, 8.0, 0.002), (0.10, 8.0, 0.0), (0.15, 8.0, 0.0)]
CHOSEN = (0.10, 8.0, 0.0)
TRAILING_WINDOW = 3
CHUNK = 40
SHOWCASE = [("Case 1: similar alert rate", 1539617736),
            ("Case 2: existing test noisier", 162746),
            ("Case 3: suggested band noisier", 1178771774)]


def guarded_band_bounds(ts: pd.DatetimeIndex, values: np.ndarray,
                        floor: float, outlier_k: float,
                        alpha: float = MISS_BUDGET):
    predicted = np.full(len(values), np.nan)
    for t in range(1, len(values)):
        # median of last 3: a single anomalous value cannot drag the baseline
        predicted[t] = np.median(values[max(0, t - TRAILING_WINDOW):t])
    scale = np.maximum(np.abs(predicted), 1e-9)
    rel_err = np.abs(values - predicted) / scale
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    week_start, end = ts.min(), ts.max()
    while week_start <= end:
        week_end = week_start + pd.Timedelta(days=7)
        seg = np.asarray((ts >= week_start) & (ts < week_end)
                         & ~np.isnan(predicted))
        cal = np.asarray((ts >= week_start - pd.Timedelta(days=CAL_DAYS))
                         & (ts < week_start) & ~np.isnan(predicted))
        if seg.sum() and cal.sum() >= MIN_CAL:
            errs = np.sort(rel_err[cal])
            # Rare-extreme exclusion: an error > outlier_k x the series' own
            # p90 is contamination ONLY if such points are rare (<= ~3% of
            # calibration). Recurring extremes are legit bursts - keep them.
            extreme = errs > outlier_k * np.quantile(errs, 0.90)
            # an anomaly can contaminate its own error plus a neighbor's
            # (median-of-3 absorbs singles, pairs still leak), so allow
            # twice the anomaly-prior share before calling extremes "recurring"
            max_rare = max(3, int(np.ceil(0.06 * len(errs))))
            clean = errs[~extreme] if 0 < extreme.sum() <= max_rare else errs
            if len(clean):
                if alpha <= 0:
                    tol = max(float(clean[-1]), floor)
                else:
                    rank = min(len(clean) - 1,
                               int(np.ceil((len(clean) + 1)
                                           * (1 - alpha))) - 1)
                    tol = max(float(clean[rank]), floor)
                lower[seg] = predicted[seg] - tol * scale[seg]
                upper[seg] = predicted[seg] + tol * scale[seg]
        week_start = week_end
    return lower, upper


def episode_stats(values, lower, upper, eval_mask):
    ok = eval_mask & ~np.isnan(lower)
    flags = ((values < lower) | (values > upper))[ok]
    episodes, run = 0, 0
    for f in list(flags) + [False]:
        if f:
            run += 1
        elif run:
            episodes += 1
            run = 0
    return int(flags.sum()), int(len(flags)), episodes


def detection(values, predicted_lo, predicted_hi, eval_mask, delta):
    ok = eval_mask & ~np.isnan(predicted_lo)
    inj = values[ok] * (1 + delta)
    return list((inj < predicted_lo[ok]) | (inj > predicted_hi[ok]))


def main() -> None:
    paired = pd.read_csv(HERE / "stage2_paired.csv")
    paired["asset_value"] = paired["asset_value"].fillna("")
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    cur.execute(TESTS_SQL, {"mids": [int(m) for m in paired["metric_id"]],
                            "avs": [str(a) for a in paired["asset_value"]]})
    tests = pd.DataFrame(cur.fetchall(), columns=[
        "metric_id", "asset_value", "test_type", "var1", "var2", "var3",
        "metric_type"]).drop_duplicates(["metric_id", "asset_value"])

    methods = ["incumbent", "v1"] + [
        f"v2_f{int(f*100)}_k{int(k)}_a{alpha:g}" for f, k, alpha in GRID]
    agg = {m: {"flag": 0, "runs": 0, "episodes": 0, "det50": [], "det100": []}
           for m in methods}
    per_series_rows: list[dict] = []
    showcase_data = {}

    recs = tests[["metric_id", "asset_value"]].values
    n_series = 0
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
            if len(s) < 40:
                continue
            t_row = tests[(tests["metric_id"] == r[0])
                          & (tests["asset_value"] == r[1])].iloc[0]
            ts = pd.DatetimeIndex(s["app_pit"])
            y = s["metric_value"].astype(float).values
            eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
            bands = {}
            try:
                bands["incumbent"] = incumbent_bounds(t_row, list(ts), y)
            except Exception:
                continue
            bands["v1"] = rel_band_bounds(ts, y)
            for f, k, alpha in GRID:
                bands[f"v2_f{int(f*100)}_k{int(k)}_a{alpha:g}"] = (
                    guarded_band_bounds(ts, y, f, k, alpha))
            n_series += 1
            for m, (lo, hi) in bands.items():
                nf, nr, eps = episode_stats(y, lo, hi, eval_mask)
                per_series_rows.append({
                    "metric_id": r[0], "asset_value": r[1], "method": m,
                    "n_flagged": nf, "n_runs": nr, "episodes": eps})
                agg[m]["flag"] += nf
                agg[m]["runs"] += nr
                agg[m]["episodes"] += eps
                d50 = detection(y, lo, hi, eval_mask, 0.5)
                d100 = detection(y, lo, hi, eval_mask, 1.0)
                if d50:
                    agg[m]["det50"].append(float(np.mean(d50)))
                if d100:
                    agg[m]["det100"].append(float(np.mean(d100)))
            for label, mid in SHOWCASE:
                if r[0] == mid:
                    showcase_data[label] = {"ts": ts, "y": y, "t_row": t_row,
                                            "bands": bands,
                                            "eval_mask": eval_mask}
    conn.close()

    pd.DataFrame(per_series_rows).to_csv(HERE / "stage3_per_series.csv",
                                         index=False)
    print(f"series: {n_series}")
    print(f"{'method':>16} {'flag_rate':>10} {'episodes/m':>11} "
          f"{'det+50%':>8} {'det+100%':>9}")
    for m in methods:
        a = agg[m]
        print(f"{m:>16} {a['flag'] / max(a['runs'], 1):>10.3%} "
              f"{a['episodes'] / n_series:>11.2f} "
              f"{float(np.mean(a['det50'])):>8.1%} "
              f"{float(np.mean(a['det100'])):>9.1%}")

    # showcase render: existing vs v1 vs chosen v2
    chosen_key = f"v2_f{int(CHOSEN[0]*100)}_k{int(CHOSEN[1])}"
    fig, axes = plt.subplots(3, 3, figsize=(20, 12))
    panel_names = ["EXISTING test", "v1: raw quantile band",
                   f"v2: guarded band (rare-spike exclusion, "
                   f"floor {CHOSEN[0]:.0%})"]
    for row, (label, mid) in enumerate(SHOWCASE):
        d = showcase_data.get(label)
        if d is None:
            continue
        for col, (name, key) in enumerate(zip(
                panel_names, ["incumbent", "v1", chosen_key])):
            ax = axes[row][col]
            lo, hi = d["bands"][key]
            ts, y = d["ts"], d["y"]
            ax.plot(ts, y, ".", ms=3, color="tab:blue")
            ax.fill_between(ts, lo, hi, color="gray", alpha=0.3)
            breach = ((y < lo) | (y > hi)) & ~np.isnan(lo) & d["eval_mask"]
            ax.plot(ts[breach], y[breach], "rx", ms=8, mew=2)
            ax.axvline(ts.max() - pd.Timedelta(days=EVAL_DAYS),
                       color="orange", ls="--", lw=1)
            nf = int(breach.sum())
            ax.set_title(f"{label}\nmetric {mid} - {name}\n"
                         f"alerts last 30d: {nf}", fontsize=9)
            for tick in ax.get_xticklabels():
                tick.set_rotation(25)
    plt.tight_layout()
    out = HERE / "guarded_band_examples.png"
    plt.savefig(out, dpi=110)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
