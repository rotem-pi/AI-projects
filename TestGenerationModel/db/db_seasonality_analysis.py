"""Full seasonality investigation over prod-replica extracts.

Dimensions tested per series (key = metric_id, asset_value):
  1. Hour-of-day   (hourly cohort): KW over 6h buckets on day-hour cells,
     block permutation over whole days, effect size, band inflation.
  2. Day-of-week   (all daily cohorts): KW over weekdays on detrended daily
     values, lag-7 ACF corroboration, weekend Mann-Whitney.
  3. Day-of-month  (span >= 120d): KW over day-of-month buckets (1-5 ... 26-31)
     on detrended daily values, plus explicit month-edge contrast.
  4. Month-of-year (span >= 540d, >= 8 months seen in >= 2 years): KW over
     calendar months on detrended daily values.

Detrending = value minus centered 15-day rolling median (91-day for
month-of-year so the annual cycle itself is not absorbed).
All p-values BH-FDR corrected within each dimension. Effect sizes are
(max group median - min group median) / |overall median|.
Inflation = flat p5-p95 width / mean per-group p5-p95 width.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(1337)

EFFECT_MIN = 0.10
INFLATION_MATERIAL = 1.3


def bh_fdr(pvals: pd.Series, alpha: float = 0.05) -> pd.Series:
    p = pvals.dropna().sort_values()
    n = len(p)
    if n == 0:
        return pd.Series(False, index=pvals.index)
    crit = alpha * np.arange(1, n + 1) / n
    below = p.values <= crit
    cutoff = p.values[below].max() if below.any() else -1
    return pvals <= cutoff


def load_daily() -> pd.DataFrame:
    frames = []
    for cohort in ["yearly", "oneyear", "midspan"]:
        agg = pd.read_csv(HERE / f"agg_{cohort}.csv", parse_dates=["day"])
        meta = pd.read_csv(HERE / f"sample_{cohort}.csv")
        agg["asset_value"] = agg["asset_value"].fillna("")
        meta["asset_value"] = meta["asset_value"].fillna("")
        agg = agg.merge(meta[["metric_id", "asset_value", "metric_type"]],
                        on=["metric_id", "asset_value"], how="left")
        agg["cohort"] = cohort
        frames.append(agg)
    df = pd.concat(frames, ignore_index=True)
    df["asset_value"] = df["asset_value"].fillna("")
    df = df.drop_duplicates(subset=["metric_id", "asset_value", "day"])
    return df


def detrend(s: pd.Series, window: int) -> pd.Series:
    trend = s.rolling(window, center=True, min_periods=max(window // 3, 3)).median()
    return s - trend


def group_stats(vals_by_group: dict) -> tuple[float, float]:
    meds = {k: np.median(v) for k, v in vals_by_group.items()}
    all_vals = np.concatenate(list(vals_by_group.values()))
    overall = np.median(all_vals)
    denom = abs(overall) if overall != 0 else max(all_vals.std(), 1e-9)
    effect = (max(meds.values()) - min(meds.values())) / denom
    flat = np.percentile(all_vals, 95) - np.percentile(all_vals, 5)
    widths = [np.percentile(v, 95) - np.percentile(v, 5) for v in vals_by_group.values()]
    inflation = flat / np.mean(widths) if np.mean(widths) > 0 else np.nan
    return effect, inflation


def kw_on_groups(g: pd.core.groupby.SeriesGroupBy, min_groups: int, min_per_group: int):
    groups = {k: v.dropna().values for k, v in g if len(v.dropna()) >= min_per_group}
    if len(groups) < min_groups:
        return None
    try:
        kw = stats.kruskal(*groups.values())
    except ValueError:
        return None
    effect, inflation = group_stats(groups)
    return {"p": kw.pvalue, "effect": effect, "inflation": inflation,
            "n_groups": len(groups)}


# ---------------- day of week ----------------
def analyze_dow(sdf: pd.DataFrame) -> dict | None:
    daily = sdf.set_index("day")["avg_val"].sort_index()
    if len(daily) < 28 or daily.std() == 0:
        return None
    resid = detrend(daily, 15).dropna()
    if len(resid) < 28 or resid.std() == 0:
        return None
    frame = resid.to_frame("v")
    frame["dow"] = frame.index.dayofweek
    r = kw_on_groups(frame.groupby("dow")["v"], min_groups=4, min_per_group=4)
    if r is None:
        return None
    # raw-scale effect/inflation (residual scale hides magnitude)
    rawf = daily.to_frame("v")
    rawf["dow"] = rawf.index.dayofweek
    raw_groups = {k: v.values for k, v in rawf.groupby("dow")["v"] if len(v) >= 4}
    if len(raw_groups) >= 4:
        r["effect"], r["inflation"] = group_stats(raw_groups)
    # weekend contrast + lag-7 ACF corroboration
    wk = frame[frame["dow"] < 5]["v"].values
    we = frame[frame["dow"] >= 5]["v"].values
    r["weekend_p"] = (stats.mannwhitneyu(wk, we).pvalue
                      if len(wk) >= 5 and len(we) >= 5 else np.nan)
    full = daily.resample("D").asfreq()
    r["acf7"] = np.nan
    r["acf7_ok"] = False
    if full.isna().mean() <= 0.3 and len(full) >= 35:
        x = detrend(full.interpolate(limit=2), 15).dropna().values
        if len(x) >= 28 and x.std() > 0:
            def acf(lag):
                a, b = x[:-lag], x[lag:]
                return np.corrcoef(a, b)[0, 1] if a.std() > 0 and b.std() > 0 else np.nan
            r["acf7"] = acf(7)
            others = [abs(acf(l)) for l in (2, 3, 4, 5)]
            others = [o for o in others if not np.isnan(o)]
            if not np.isnan(r["acf7"]) and others:
                r["acf7_ok"] = (r["acf7"] > 2 / np.sqrt(len(x))
                                and r["acf7"] > max(others))
    return r


# ---------------- day of month ----------------
def analyze_dom(sdf: pd.DataFrame) -> dict | None:
    daily = sdf.set_index("day")["avg_val"].sort_index()
    span = (daily.index.max() - daily.index.min()).days
    if span < 120 or len(daily) < 60 or daily.std() == 0:
        return None
    resid = detrend(daily, 15).dropna()
    if len(resid) < 60 or resid.std() == 0:
        return None
    frame = resid.to_frame("v")
    frame["bucket"] = np.minimum((frame.index.day - 1) // 5, 5)  # 1-5,...,26-31
    r = kw_on_groups(frame.groupby("bucket")["v"], min_groups=5, min_per_group=6)
    if r is None:
        return None
    edge = frame[(frame.index.day <= 2) | (frame.index.day >= 28)]["v"].values
    mid = frame[(frame.index.day > 2) & (frame.index.day < 28)]["v"].values
    r["edge_p"] = (stats.mannwhitneyu(edge, mid).pvalue
                   if len(edge) >= 6 and len(mid) >= 12 else np.nan)
    r["n_months"] = frame.index.to_period("M").nunique()
    return r


# ---------------- month of year ----------------
def analyze_moy(sdf: pd.DataFrame) -> dict | None:
    daily = sdf.set_index("day")["avg_val"].sort_index()
    span = (daily.index.max() - daily.index.min()).days
    if span < 540 or len(daily) < 100 or daily.std() == 0:
        return None
    resid = detrend(daily, 91).dropna()
    if len(resid) < 100 or resid.std() == 0:
        return None
    frame = resid.to_frame("v")
    frame["month"] = frame.index.month
    frame["year"] = frame.index.year
    months_2y = (frame.groupby("month")["year"].nunique() >= 2).sum()
    if months_2y < 8:
        return None
    r = kw_on_groups(frame.groupby("month")["v"], min_groups=8, min_per_group=6)
    if r is None:
        return None
    r["months_in_2y"] = months_2y
    return r


# ---------------- hour of day ----------------
def analyze_hod(sdf: pd.DataFrame) -> dict | None:
    cells = sdf.dropna(subset=["avg_val"])
    if len(cells) < 100 or cells["avg_val"].std() == 0:
        return None
    if cells["hr"].nunique() < 6:
        return None
    cells = cells.assign(bucket=cells["hr"] // 6)
    groups = {b: g["avg_val"].values for b, g in cells.groupby("bucket")
              if len(g) >= 10}
    if len(groups) < 3:
        return None
    try:
        kw = stats.kruskal(*groups.values())
    except ValueError:
        return None
    effect, inflation = group_stats(groups)
    # block permutation over whole days
    days = cells["day"].unique()
    if len(days) < 10:
        perm_p = np.nan
    else:
        h_obs = kw.statistic
        by_day = {d: g[["bucket", "avg_val"]].values for d, g in cells.groupby("day")}
        h_perm = []
        buckets_order = cells["bucket"].values
        for _ in range(200):
            perm_days = rng.permutation(list(by_day))
            vals = np.concatenate([by_day[d][:, 1] for d in perm_days])
            if len(vals) != len(buckets_order):
                vals = vals[: len(buckets_order)]
            pg = [vals[buckets_order[: len(vals)] == b] for b in groups]
            pg = [g for g in pg if len(g) >= 10]
            if len(pg) >= 3:
                try:
                    h_perm.append(stats.kruskal(*pg).statistic)
                except ValueError:
                    pass
        perm_p = ((np.sum(np.array(h_perm) >= h_obs) + 1) / (len(h_perm) + 1)
                  if h_perm else np.nan)
    return {"p": kw.pvalue, "perm_p": perm_p, "effect": effect,
            "inflation": inflation, "n_cells": len(cells), "n_days": len(days)}


def run_dimension(df, group_cols, fn, name):
    rows = []
    for (mid, av), sdf in df.groupby(group_cols):
        r = fn(sdf)
        if r is None:
            continue
        r.update({"metric_id": mid, "asset_value": av,
                  "metric_type": sdf["metric_type"].iloc[0]})
        rows.append(r)
    out = pd.DataFrame(rows)
    if out.empty:
        print(f"\n=== {name}: no testable series ===")
        return out
    out["sig"] = bh_fdr(out["p"])
    out.to_csv(HERE / f"result_{name}.csv", index=False)
    return out


def summarize(out: pd.DataFrame, name: str, extra_robust: pd.Series | None = None):
    if out.empty:
        return
    robust = out["sig"] & (out["effect"] > EFFECT_MIN)
    if extra_robust is not None:
        robust &= extra_robust
    material = robust & (out["inflation"] > INFLATION_MATERIAL)
    print(f"\n=== {name} (testable series: {len(out)}) ===")
    print(f"significant (FDR 5%): {out['sig'].sum()} ({out['sig'].mean():.1%})")
    print(f"robust (sig + effect > {EFFECT_MIN:.0%}"
          f"{' + corroborated' if extra_robust is not None else ''}): "
          f"{robust.sum()} ({robust.mean():.1%})")
    print(f"material (band inflation > {INFLATION_MATERIAL}x): "
          f"{material.sum()} ({material.mean():.1%})")
    if robust.any():
        top_types = (out[robust].groupby("metric_type")
                     .agg(n=("metric_id", "size"), med_effect=("effect", "median"),
                          med_inflation=("inflation", "median"))
                     .sort_values("n", ascending=False).head(10))
        print(top_types.to_string())
    out.loc[:, "robust"] = robust
    out.loc[:, "material"] = material
    out.to_csv(HERE / f"result_{name}.csv", index=False)


def main() -> None:
    daily = load_daily()
    hourly = pd.read_csv(HERE / "agg_hourly.csv", parse_dates=["day"])
    hmeta = pd.read_csv(HERE / "sample_hourly.csv")
    hourly["asset_value"] = hourly["asset_value"].fillna("")
    hmeta["asset_value"] = hmeta["asset_value"].fillna("")
    hourly = hourly.merge(hmeta[["metric_id", "asset_value", "metric_type"]],
                          on=["metric_id", "asset_value"], how="left")

    n_series = daily.groupby(["metric_id", "asset_value"]).ngroups
    print(f"daily frame: {len(daily)} rows, {n_series} series")
    print(f"hourly frame: {len(hourly)} rows, "
          f"{hourly.groupby(['metric_id', 'asset_value']).ngroups} series")

    key = ["metric_id", "asset_value"]
    hod = run_dimension(hourly, key, analyze_hod, "hour_of_day")
    if not hod.empty:
        summarize(hod, "hour_of_day", extra_robust=(hod["perm_p"] < 0.05))

    dow = run_dimension(daily, key, analyze_dow, "day_of_week")
    if not dow.empty:
        summarize(dow, "day_of_week",
                  extra_robust=(dow["acf7_ok"] | (dow["weekend_p"] < 0.01)))

    dom = run_dimension(daily, key, analyze_dom, "day_of_month")
    if not dom.empty:
        summarize(dom, "day_of_month")

    moy = run_dimension(daily, key, analyze_moy, "month_of_year")
    if not moy.empty:
        summarize(moy, "month_of_year")


if __name__ == "__main__":
    main()
