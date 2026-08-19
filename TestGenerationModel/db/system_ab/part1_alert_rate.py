"""Alert-rate view of the Part 1 harness verdicts: alerts per covered
series per week per arm, pooled and per dataset.

Feeds the "Alert rate on the labeled datasets" section of
system_a_vs_b_report.md. Inputs: part1_verdicts_<arm>_<dataset>.csv
(written by part1_scoring.py).

Run from anywhere:
  uv run python analysis_temp/guarded-band/part1_alert_rate.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
ARMS = ["A", "B5"]
DATASETS = ["random-a-200", "random-a-500", "a_self_labeled", "definity_tagging"]
MIN_WEEKS = 0.1


def summarize(per: pd.DataFrame, label: str) -> dict:
    return {
        "arm": label,
        "alerts_per_series_per_week": round((per["flags"] / per["weeks"]).mean(), 3),
        "false_alerts_per_series_per_week": round(
            (per["fps"] / per["weeks"]).mean(), 3
        ),
        "pct_series_quiet": round(100 * (per["flags"] == 0).mean(), 1),
    }


def main() -> None:
    rows = []
    for arm in ARMS:
        frames = [
            pd.read_csv(
                HERE / f"part1_verdicts_{arm}_{ds}.csv", parse_dates=["pit"]
            ).assign(dataset=ds)
            for ds in DATASETS
        ]
        allp = pd.concat(frames, ignore_index=True)
        per = allp.groupby(["dataset", "metric_id"]).agg(
            flags=("pred_is_anomaly", "sum"), first=("pit", "min"), last=("pit", "max")
        )
        fps = (
            allp[~allp["gt_is_anomaly"]]
            .groupby(["dataset", "metric_id"])["pred_is_anomaly"]
            .sum()
        )
        per["fps"] = fps.reindex(per.index).fillna(0)
        per["weeks"] = ((per["last"] - per["first"]).dt.days / 7).clip(lower=MIN_WEEKS)
        rows.append(summarize(per, arm))
        rows.extend(summarize(per.loc[ds], f"{arm}/{ds}") for ds in DATASETS)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
