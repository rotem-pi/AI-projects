"""Seasonality scan over production metric exports (train_data/raw datasets).

For every sufficiently long series, tests:
  1. Day-of-week effect: Kruskal-Wallis over daily medians grouped by weekday
     (BH-FDR corrected), plus a weekday-vs-weekend Mann-Whitney test.
  2. Weekly cycle: lag-7 autocorrelation of the detrended daily series,
     required to peak at 7 relative to lags 2-5.
  3. Hour-of-day effect (intraday series only): Kruskal-Wallis over
     6-hour buckets.

Also quantifies the cost of ignoring weekly seasonality: how much wider a
flat p5-p95 Range band is than the average per-weekday band ("inflation").
"""

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RAW = Path(__file__).resolve().parents[2] / "backend/tests/anomaly/inputs/train_data/raw"
OUT = Path(__file__).resolve().parent

MIN_POINTS = 30
MIN_SPAN_DAYS = 21
MIN_DAILY_POINTS = 15
KW_ALPHA_FDR = 0.05
ACF_PEAK_LAGS = (2, 3, 4, 5)


@dataclass
class Series:
    metric_id: int
    metric_type: str
    metric_group_type: str
    source: str
    values: np.ndarray
    pits: pd.DatetimeIndex
    results: dict = field(default_factory=dict)


def load_json_dataset(path: Path, source: str) -> list[Series]:
    entries = json.loads(path.read_text())
    out = []
    for e in entries:
        out.append(
            Series(
                metric_id=e["metric_id"],
                metric_type=e.get("metric_type") or "unknown",
                metric_group_type=e.get("metric_group_type") or "unknown",
                source=source,
                values=np.asarray(e["metric_values"], dtype=float),
                pits=pd.to_datetime(e["app_pits"]),
            )
        )
    return out


def load_csv_dataset(path: Path, source: str) -> list[Series]:
    df = pd.read_csv(path)
    out = []
    for _, row in df.iterrows():
        try:
            values = ast.literal_eval(row["metric_values"])
            pits = ast.literal_eval(row["app_pits"])
        except (ValueError, SyntaxError):
            continue
        if not isinstance(values, (list, tuple)) or not values:
            continue
        out.append(
            Series(
                metric_id=int(row["metric_id"]),
                metric_type=str(row.get("metric_type") or "unknown"),
                metric_group_type=str(row.get("metric_group_type") or "unknown"),
                source=source,
                values=np.asarray(values, dtype=float),
                pits=pd.to_datetime(list(pits)),
            )
        )
    return out


def load_all() -> list[Series]:
    all_series: list[Series] = []
    for name in ["random-a-500", "random-a-200", "definity_tagging", "definity_sanity"]:
        d = RAW / name / "raw_metrics"
        if not d.exists():
            continue
        for f in d.iterdir():
            if f.suffix == ".json":
                all_series.extend(load_json_dataset(f, name))
            elif f.suffix == ".csv":
                all_series.extend(load_csv_dataset(f, name))
    agoda = RAW / "agoda_self_labeled/raw_metrics"
    if agoda.exists():
        for f in agoda.glob("*.csv"):
            all_series.extend(load_csv_dataset(f, "agoda_self_labeled"))

    # Dedupe by metric_id, keep the longest series.
    best: dict[int, Series] = {}
    for s in all_series:
        cur = best.get(s.metric_id)
        if cur is None or len(s.values) > len(cur.values):
            best[s.metric_id] = s
    return list(best.values())


def eligible(s: Series) -> bool:
    if len(s.values) < MIN_POINTS or len(s.values) != len(s.pits):
        return False
    span = (s.pits.max() - s.pits.min()).days
    if span < MIN_SPAN_DAYS:
        return False
    if np.nanstd(s.values) == 0 or not np.isfinite(s.values).all():
        return False
    return True


def daily_frame(s: Series) -> pd.DataFrame:
    df = pd.DataFrame({"value": s.values}, index=s.pits).sort_index()
    daily = df.resample("D")["value"].median().to_frame("value")
    daily["n"] = df.resample("D")["value"].size()
    daily = daily[daily["n"] > 0]
    daily["dow"] = daily.index.dayofweek
    return daily


def dow_tests(daily: pd.DataFrame) -> dict:
    res = {"kw_p": np.nan, "dow_effect": np.nan, "weekend_p": np.nan, "inflation": np.nan}
    groups = [g["value"].values for _, g in daily.groupby("dow") if len(g) >= 3]
    if len(groups) < 4 or len(daily) < MIN_DAILY_POINTS:
        return res
    try:
        res["kw_p"] = stats.kruskal(*groups).pvalue
    except ValueError:
        return res

    med = daily.groupby("dow")["value"].median()
    overall = daily["value"].median()
    denom = abs(overall) if overall != 0 else max(daily["value"].std(), 1e-9)
    res["dow_effect"] = (med.max() - med.min()) / denom

    wk = daily[daily["dow"] < 5]["value"].values
    we = daily[daily["dow"] >= 5]["value"].values
    if len(wk) >= 5 and len(we) >= 5:
        res["weekend_p"] = stats.mannwhitneyu(wk, we).pvalue

    flat_width = np.percentile(daily["value"], 95) - np.percentile(daily["value"], 5)
    dow_widths = [
        np.percentile(g, 95) - np.percentile(g, 5)
        for _, g in daily.groupby("dow")["value"]
        if len(g) >= 3
    ]
    mean_dow_width = np.mean(dow_widths) if dow_widths else np.nan
    if mean_dow_width and mean_dow_width > 0:
        res["inflation"] = flat_width / mean_dow_width
    return res


def weekly_acf(daily: pd.DataFrame) -> dict:
    res = {"acf7": np.nan, "acf7_significant": False, "acf7_is_peak": False}
    full = daily["value"].resample("D").asfreq()
    if full.isna().mean() > 0.3 or len(full) < 28:
        return res
    filled = full.interpolate(limit=2)
    if filled.isna().any():
        filled = filled.dropna()
        if len(filled) < 28:
            return res
    trend = filled.rolling(15, center=True, min_periods=5).median()
    resid = (filled - trend).dropna()
    if len(resid) < 21 or resid.std() == 0:
        return res

    def acf(x: np.ndarray, lag: int) -> float:
        if len(x) <= lag:
            return np.nan
        a, b = x[:-lag], x[lag:]
        if a.std() == 0 or b.std() == 0:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])

    x = resid.values
    res["acf7"] = acf(x, 7)
    thresh = 2 / np.sqrt(len(x))
    others = [abs(acf(x, l)) for l in ACF_PEAK_LAGS]
    others = [o for o in others if not np.isnan(o)]
    if not np.isnan(res["acf7"]):
        res["acf7_significant"] = res["acf7"] > thresh
        res["acf7_is_peak"] = bool(others) and res["acf7"] > max(others)
    return res


def hourly_test(s: Series) -> dict:
    res = {"hour_p": np.nan, "intraday": False}
    span_days = max((s.pits.max() - s.pits.min()).days, 1)
    if len(s.values) / span_days < 3 or len(s.values) < 100:
        return res
    res["intraday"] = True
    df = pd.DataFrame({"value": s.values, "bucket": s.pits.hour // 6})
    groups = [g["value"].values for _, g in df.groupby("bucket") if len(g) >= 10]
    if len(groups) >= 3:
        try:
            res["hour_p"] = stats.kruskal(*groups).pvalue
        except ValueError:
            pass
    return res


def bh_fdr(pvals: pd.Series, alpha: float) -> pd.Series:
    p = pvals.dropna().sort_values()
    n = len(p)
    if n == 0:
        return pd.Series(False, index=pvals.index)
    crit = alpha * np.arange(1, n + 1) / n
    below = p.values <= crit
    cutoff = p.values[below].max() if below.any() else -1
    return pvals <= cutoff


def main() -> None:
    series = [s for s in load_all() if eligible(s)]
    print(f"eligible series: {len(series)}")

    rows = []
    for s in series:
        daily = daily_frame(s)
        r = {
            "metric_id": s.metric_id,
            "metric_type": s.metric_type,
            "group_type": s.metric_group_type,
            "source": s.source,
            "n_points": len(s.values),
            "n_days": len(daily),
            "span_days": (s.pits.max() - s.pits.min()).days,
        }
        r.update(dow_tests(daily))
        r.update(weekly_acf(daily))
        r.update(hourly_test(s))
        rows.append(r)

    df = pd.DataFrame(rows)
    df["kw_sig"] = bh_fdr(df["kw_p"], KW_ALPHA_FDR)
    df["hour_sig"] = bh_fdr(df["hour_p"], KW_ALPHA_FDR)
    df["weekly_seasonal"] = (
        df["kw_sig"]
        & (df["dow_effect"] > 0.10)
        & (df["acf7_significant"] & df["acf7_is_peak"] | (df["dow_effect"] > 0.25))
    )
    df["material"] = df["weekly_seasonal"] & (df["inflation"] > 1.3)

    df.to_csv(OUT / "seasonality_per_series.csv", index=False)

    n = len(df)
    print(f"\n=== weekly seasonality (n={n}) ===")
    print(f"dow KW significant (FDR 5%): {df['kw_sig'].sum()} ({df['kw_sig'].mean():.1%})")
    print(f"weekly_seasonal (sig + effect>10% + acf7 corroborated): "
          f"{df['weekly_seasonal'].sum()} ({df['weekly_seasonal'].mean():.1%})")
    print(f"material (flat band >1.3x wider than per-dow band): "
          f"{df['material'].sum()} ({df['material'].mean():.1%})")

    print("\n=== by metric group type ===")
    g = df.groupby("group_type").agg(
        n=("metric_id", "size"),
        seasonal=("weekly_seasonal", "sum"),
        seasonal_pct=("weekly_seasonal", "mean"),
        med_inflation_seasonal=("inflation", lambda x: x[df.loc[x.index, "weekly_seasonal"]].median()),
    )
    print(g.to_string())

    print("\n=== by metric type (n>=10) ===")
    g2 = df.groupby("metric_type").agg(
        n=("metric_id", "size"),
        seasonal_pct=("weekly_seasonal", "mean"),
    )
    print(g2[g2["n"] >= 10].sort_values("seasonal_pct", ascending=False).to_string())

    intraday = df[df["intraday"]]
    print(f"\n=== intraday series (>=3 pts/day, n={len(intraday)}) ===")
    print(f"hour-of-day significant (FDR 5%): {intraday['hour_sig'].sum()} "
          f"({intraday['hour_sig'].mean():.1%} of intraday)")

    top = df[df["weekly_seasonal"]].sort_values("inflation", ascending=False).head(12)
    print("\n=== top seasonal examples (by band inflation) ===")
    cols = ["metric_id", "metric_type", "group_type", "n_days", "dow_effect", "acf7", "inflation", "source"]
    print(top[cols].to_string(index=False))


if __name__ == "__main__":
    main()
