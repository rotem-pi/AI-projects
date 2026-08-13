"""Deep dive on the hour-of-day effect: effect sizes, robustness check
(block permutation to respect autocorrelation), and plots of top examples."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from seasonality_analysis import bh_fdr, eligible, load_all

OUT = Path(__file__).resolve().parent
rng = np.random.default_rng(1337)


def hour_effect(s) -> dict | None:
    span_days = max((s.pits.max() - s.pits.min()).days, 1)
    if len(s.values) / span_days < 3 or len(s.values) < 100:
        return None
    df = pd.DataFrame({"value": s.values, "bucket": s.pits.hour // 6}).reset_index(drop=True)
    groups = {b: g["value"].values for b, g in df.groupby("bucket") if len(g) >= 10}
    if len(groups) < 3:
        return None
    try:
        kw = stats.kruskal(*groups.values())
    except ValueError:
        return None

    meds = {b: np.median(v) for b, v in groups.items()}
    overall = np.median(df["value"])
    denom = abs(overall) if overall != 0 else max(df["value"].std(), 1e-9)
    effect = (max(meds.values()) - min(meds.values())) / denom

    # Block permutation: shuffle contiguous blocks of ~1 day of points so
    # autocorrelation is preserved within blocks; recompute KW H-stat.
    block = max(int(len(df) / span_days), 5)
    n_blocks = len(df) // block
    if n_blocks >= 10:
        h_obs = kw.statistic
        h_perm = []
        vals = df["value"].values[: n_blocks * block].reshape(n_blocks, block)
        buckets = df["bucket"].values[: n_blocks * block]
        for _ in range(200):
            perm = vals[rng.permutation(n_blocks)].ravel()
            pg = [perm[buckets == b] for b in groups]
            try:
                h_perm.append(stats.kruskal(*pg).statistic)
            except ValueError:
                pass
        perm_p = (np.sum(np.array(h_perm) >= h_obs) + 1) / (len(h_perm) + 1) if h_perm else np.nan
    else:
        perm_p = np.nan

    # Band inflation: flat p5-p95 vs per-bucket p5-p95
    flat = np.percentile(df["value"], 95) - np.percentile(df["value"], 5)
    widths = [np.percentile(v, 95) - np.percentile(v, 5) for v in groups.values()]
    inflation = flat / np.mean(widths) if np.mean(widths) > 0 else np.nan

    return {
        "metric_id": s.metric_id,
        "metric_type": s.metric_type,
        "group_type": s.metric_group_type,
        "n_points": len(s.values),
        "span_days": span_days,
        "hour_p": kw.pvalue,
        "perm_p": perm_p,
        "hour_effect": effect,
        "inflation": inflation,
        "source": s.source,
    }


def main() -> None:
    series = {s.metric_id: s for s in load_all() if eligible(s)}
    rows = [r for s in series.values() if (r := hour_effect(s))]
    df = pd.DataFrame(rows)
    df["hour_sig"] = bh_fdr(df["hour_p"], 0.05)
    df["robust"] = df["hour_sig"] & (df["perm_p"] < 0.05) & (df["hour_effect"] > 0.10)
    df.to_csv(OUT / "hour_effect_per_series.csv", index=False)

    print(f"intraday series: {len(df)}")
    print(f"KW significant (FDR): {df['hour_sig'].sum()}")
    print(f"robust (survives block permutation + effect>10%): {df['robust'].sum()} "
          f"({df['robust'].mean():.1%})")
    print("\nrobust by metric_type:")
    print(df[df["robust"]].groupby("metric_type").agg(
        n=("metric_id", "size"),
        med_effect=("hour_effect", "median"),
        med_inflation=("inflation", "median"),
    ).sort_values("n", ascending=False).to_string())

    top = df[df["robust"]].sort_values("hour_effect", ascending=False).head(6)
    print("\ntop robust examples:")
    print(top[["metric_id", "metric_type", "hour_effect", "inflation", "perm_p", "span_days"]]
          .to_string(index=False))

    # Plots for top 4
    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    for i, (_, r) in enumerate(top.head(4).iterrows()):
        s = series[r["metric_id"]]
        ax1, ax2 = axes[i]
        ax1.plot(s.pits, s.values, ".", ms=3, alpha=0.6)
        ax1.set_title(f"metric {s.metric_id} ({s.metric_type}) — raw series")
        hours = s.pits.hour
        data = [s.values[hours == h] for h in range(24)]
        ax2.boxplot([d if len(d) else [np.nan] for d in data], positions=range(24), widths=0.6)
        ax2.set_title(f"by hour of day — effect {r['hour_effect']:.0%}, flat band {r['inflation']:.1f}x wider")
        ax2.set_xlabel("hour (UTC)")
    plt.tight_layout()
    plt.savefig(OUT / "hour_effect_examples.png", dpi=110)
    print(f"\nplots -> {OUT / 'hour_effect_examples.png'}")


if __name__ == "__main__":
    main()
