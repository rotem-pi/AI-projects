"""Research: negative lower bounds in the guarded band.

Questions:
 1. How many series are de-facto non-negative, and how often does the
    symmetric guarded band dip below zero for them (user confusion)?
 2. How many are "drop-blind": a collapse to 0 would NOT breach (dead width)?
 3. Does asymmetric calibration (separate down-side and up-side error
    quantiles) fix it, and what does it cost in flag rate?
 4. What does the registry already declare (lower_bound per metric type)?

Data: round3 representative frames (1,200 series, recent-cadence tiers).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.tests_gen.metrics.metrics_registry import MetricsRegistry  # noqa: E402

FLOOR, K, ALPHA = 0.10, 8.0, 0.002
TRAILING_WINDOW = 3
CAL_DAYS = 56
EVAL_DAYS = 30
MIN_CAL = 10


def clean_quantile(errs: np.ndarray, alpha: float) -> float:
    errs = np.sort(errs)
    extreme = errs > K * np.quantile(errs, 0.90)
    max_rare = max(3, int(np.ceil(0.06 * len(errs))))
    clean = errs[~extreme] if 0 < extreme.sum() <= max_rare else errs
    if not len(clean):
        return np.nan
    rank = min(len(clean) - 1, int(np.ceil((len(clean) + 1) * (1 - alpha))) - 1)
    return float(clean[rank])


def bands(ts, y, symmetric: bool):
    predicted = np.full(len(y), np.nan)
    for t in range(1, len(y)):
        predicted[t] = np.median(y[max(0, t - TRAILING_WINDOW):t])
    scale = np.maximum(np.abs(predicted), 1e-9)
    rel = (y - predicted) / scale
    lo = np.full(len(y), np.nan)
    hi = np.full(len(y), np.nan)
    week_start, end = ts.min(), ts.max()
    while week_start <= end:
        week_end = week_start + pd.Timedelta(days=7)
        seg = np.asarray((ts >= week_start) & (ts < week_end)
                         & ~np.isnan(predicted))
        cal = np.asarray((ts >= week_start - pd.Timedelta(days=CAL_DAYS))
                         & (ts < week_start) & ~np.isnan(predicted))
        if seg.sum() and cal.sum() >= MIN_CAL:
            if symmetric:
                tol = clean_quantile(np.abs(rel[cal]), ALPHA)
                if not np.isnan(tol):
                    tol = max(tol, FLOOR)
                    lo[seg] = predicted[seg] - tol * scale[seg]
                    hi[seg] = predicted[seg] + tol * scale[seg]
            else:
                down = clean_quantile(np.maximum(-rel[cal], 0), ALPHA)
                up = clean_quantile(np.maximum(rel[cal], 0), ALPHA)
                if not (np.isnan(down) or np.isnan(up)):
                    down, up = max(down, FLOOR), max(up, FLOOR)
                    lo[seg] = predicted[seg] - down * scale[seg]
                    hi[seg] = predicted[seg] + up * scale[seg]
        week_start = week_end
    return lo, hi


def main() -> None:
    reg = MetricsRegistry.get_instance().registry
    n_zero_floor = sum(1 for c in reg.values() if c.lower_bound == 0)
    print(f"registry: {n_zero_floor}/{len(reg)} metric types declare "
          f"lower_bound = 0 (non-negative by domain knowledge)")

    frames = []
    for tier in ["hf", "daily", "weekly", "sparse"]:
        f = pd.read_csv(HERE / f"round3_frame_{tier}.csv", parse_dates=["day"])
        f["asset_value"] = f["asset_value"].fillna("")
        f["tier"] = tier
        if "hr" not in f:
            f["hr"] = 0
        frames.append(f)
    data = pd.concat(frames)

    stats = []
    for (mid, av), s in data.groupby(["metric_id", "asset_value"]):
        s = s.sort_values(["day", "hr"])
        ts = pd.DatetimeIndex(s["day"] + pd.to_timedelta(s["hr"], unit="h"))
        y = s["val"].astype(float).values
        if len(y) < 30:
            continue
        eval_mask = np.asarray(ts >= ts.max() - pd.Timedelta(days=EVAL_DAYS))
        s_lo, s_hi = bands(ts, y, symmetric=True)
        a_lo, a_hi = bands(ts, y, symmetric=False)
        ok = eval_mask & ~np.isnan(s_lo) & ~np.isnan(a_lo)
        if ok.sum() < 5:
            continue
        nonneg = bool((y >= 0).all())
        pred_pos = ok & ((s_lo + s_hi) / 2 > 0)
        stats.append({
            "metric_id": mid, "nonneg": nonneg, "n_eval": int(ok.sum()),
            "sym_neg_lo_pts": int((s_lo[ok] < 0).sum()),
            "asym_neg_lo_pts": int((a_lo[ok] < 0).sum()),
            "sym_dropblind_pts": int((s_lo[pred_pos] <= 0).sum()),
            "asym_dropblind_pts": int((a_lo[pred_pos] <= 0).sum()),
            "n_predpos": int(pred_pos.sum()),
            "sym_flags": int((((y < s_lo) | (y > s_hi)) & ok).sum()),
            "asym_flags": int((((y < a_lo) | (y > a_hi)) & ok).sum()),
        })
    df = pd.DataFrame(stats)
    nn = df[df["nonneg"]]
    print(f"\nseries analyzed: {len(df)}; de-facto non-negative: "
          f"{df['nonneg'].mean():.1%}")

    print("\n=== symmetric band (current v2) on non-negative series ===")
    print(f"series with any negative lower bound shown: "
          f"{(nn['sym_neg_lo_pts'] > 0).mean():.1%}")
    print(f"share of eval points with negative lower bound: "
          f"{nn['sym_neg_lo_pts'].sum() / nn['n_eval'].sum():.1%}")
    pp = nn[nn["n_predpos"] > 0]
    print(f"drop-blind points (collapse to 0 would NOT alert): "
          f"{pp['sym_dropblind_pts'].sum() / pp['n_predpos'].sum():.1%}")

    print("\n=== asymmetric band (separate down/up quantiles) ===")
    print(f"series with any negative lower bound: "
          f"{(nn['asym_neg_lo_pts'] > 0).mean():.1%}")
    print(f"eval points with negative lower bound: "
          f"{nn['asym_neg_lo_pts'].sum() / nn['n_eval'].sum():.1%}")
    print(f"drop-blind points: "
          f"{pp['asym_dropblind_pts'].sum() / pp['n_predpos'].sum():.1%}")

    print("\n=== cost: real-data flag rate (all series) ===")
    print(f"symmetric:  {df['sym_flags'].sum() / df['n_eval'].sum():.3%}")
    print(f"asymmetric: {df['asym_flags'].sum() / df['n_eval'].sum():.3%}")

    df.to_csv(HERE / "negative_bounds_research.csv", index=False)


if __name__ == "__main__":
    main()
