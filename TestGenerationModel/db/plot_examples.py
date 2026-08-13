"""Plot top robust examples for each seasonal dimension found in the DB scan."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent

DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def load_daily() -> pd.DataFrame:
    frames = []
    for cohort in ["yearly", "oneyear", "midspan"]:
        agg = pd.read_csv(HERE / f"agg_{cohort}.csv", parse_dates=["day"])
        agg["asset_value"] = agg["asset_value"].fillna("")
        frames.append(agg)
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["metric_id", "asset_value", "day"])


def top_examples(name: str, k: int, sort_col: str = "inflation") -> pd.DataFrame:
    res = pd.read_csv(HERE / f"result_{name}.csv")
    res["asset_value"] = res["asset_value"].fillna("")
    robust = res[res["robust"] & (res["effect"] < 50)]  # drop near-zero-median artifacts
    return robust.sort_values(sort_col, ascending=False).head(k)


def main() -> None:
    daily = load_daily()
    hourly = pd.read_csv(HERE / "agg_hourly.csv", parse_dates=["day"])
    hourly["asset_value"] = hourly["asset_value"].fillna("")

    fig, axes = plt.subplots(4, 2, figsize=(15, 17))

    # hour of day: 2 examples
    for col, (_, r) in enumerate(top_examples("hour_of_day", 2).iterrows()):
        s = hourly[(hourly["metric_id"] == r["metric_id"])
                   & (hourly["asset_value"] == r["asset_value"])]
        ax = axes[0][col]
        data = [s[s["hr"] == h]["avg_val"].values for h in range(24)]
        ax.boxplot([d if len(d) else [float("nan")] for d in data],
                   positions=range(24), widths=0.6)
        ax.set_title(f"HOUR OF DAY - metric {int(r['metric_id'])} ({r['metric_type']})\n"
                     f"effect {r['effect']:.0%}, flat band {r['inflation']:.2f}x wider")
        ax.set_xlabel("hour of PIT")

    # day of week: 2 examples
    for col, (_, r) in enumerate(top_examples("day_of_week", 2).iterrows()):
        s = daily[(daily["metric_id"] == r["metric_id"])
                  & (daily["asset_value"] == r["asset_value"])].set_index("day")
        ax = axes[1][col]
        dows = s.index.dayofweek
        data = [s[dows == d]["avg_val"].values for d in range(7)]
        ax.boxplot([d if len(d) else [float("nan")] for d in data],
                   positions=range(7), widths=0.5)
        ax.set_xticks(range(7), DOW_NAMES)
        ax.set_title(f"DAY OF WEEK - metric {int(r['metric_id'])} ({r['metric_type']})\n"
                     f"effect {r['effect']:.0%}, flat band {r['inflation']:.2f}x wider")

    # day of month: 2 examples
    for col, (_, r) in enumerate(top_examples("day_of_month", 2).iterrows()):
        s = daily[(daily["metric_id"] == r["metric_id"])
                  & (daily["asset_value"] == r["asset_value"])].set_index("day")
        ax = axes[2][col]
        doms = s.index.day
        data = [s[doms == d]["avg_val"].values for d in range(1, 32)]
        ax.boxplot([d if len(d) else [float("nan")] for d in data],
                   positions=range(1, 32), widths=0.6)
        ax.set_title(f"DAY OF MONTH - metric {int(r['metric_id'])} ({r['metric_type']})\n"
                     f"effect {r['effect']:.0%}, flat band {r['inflation']:.2f}x wider")
        ax.set_xlabel("day of month")

    # month of year: 2 examples (raw series + monthly boxes)
    moy = top_examples("month_of_year", 2)
    for col, (_, r) in enumerate(moy.iterrows()):
        s = daily[(daily["metric_id"] == r["metric_id"])
                  & (daily["asset_value"] == r["asset_value"])].set_index("day").sort_index()
        ax = axes[3][col]
        months = s.index.month
        data = [s[months == m]["avg_val"].values for m in range(1, 13)]
        ax.boxplot([d if len(d) else [float("nan")] for d in data],
                   positions=range(1, 13), widths=0.5)
        ax.set_title(f"MONTH OF YEAR - metric {int(r['metric_id'])} ({r['metric_type']})\n"
                     f"effect {r['effect']:.0%}, flat band {r['inflation']:.2f}x wider")
        ax.set_xlabel("month")

    for row in axes:
        for ax in row:
            if not ax.has_data():
                ax.axis("off")
    plt.tight_layout()
    plt.savefig(HERE / "db_seasonality_examples.png", dpi=110)
    print("saved", HERE / "db_seasonality_examples.png")


if __name__ == "__main__":
    main()
