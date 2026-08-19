"""Compare the labeled datasets against the production population
(stage2_population.csv: 547K active series from the Aug-11 replica census).

Outputs: metric_type / group mix side by side, cadence-tier mix, and
GuardedBand calibration eligibility on both sides.

Run from backend/:  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/population_compare.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from app.metrics.metric_types import get_metric_group_type

HERE = Path(__file__).resolve().parent
POP_CSV = HERE.parent / "seasonality-check/db/stage2_population.csv"

MIN_CAL, MIN_EVAL = 10, 5
TIER_BINS = [(3.0, "hf (>=3/day)"), (0.8, "daily"), (0.15, "weekly-ish"), (0.0, "sparse")]


def tier(points_per_day: float) -> str:
    for thr, name in TIER_BINS:
        if points_per_day >= thr:
            return name
    return "sparse"


pop = pd.read_csv(POP_CSV)
pop["group"] = pop["metric_type"].map(
    lambda t: get_metric_group_type(str(t)) or "custom/unknown"
)
pop["tier"] = (pop["cnt_eval"] / 30.0).map(tier)

points = pd.read_parquet(HERE / "all_labeled_points.parquet")
labeled = points[points["is_anomaly"].notna()].copy()
per_metric = (
    labeled.groupby(["dataset", "metric_id"])
    .agg(
        metric_type=("metric_type", "first"),
        group=("metric_group_type", "first"),
        n_points=("pit", "size"),
        first_pit=("pit", "min"),
        last_pit=("pit", "max"),
    )
    .reset_index()
)
span_days = (per_metric["last_pit"] - per_metric["first_pit"]).dt.total_seconds() / 86400
per_metric["ppd"] = per_metric["n_points"] / span_days.clip(lower=1)
per_metric["tier"] = per_metric["ppd"].map(tier)


def mix(series: pd.Series, top: int = 15) -> dict:
    vc = series.value_counts(normalize=True).round(4) * 100
    return {k: float(v) for k, v in vc.head(top).items()}


out = {
    "population": {
        "n_series": len(pop),
        "rel_eligible_pct": round(100 * pop["rel_eligible"].mean(), 1),
        "has_auto_test_pct": round(100 * pop["has_auto_test"].mean(), 1),
        "metric_type_mix_pct": mix(pop["metric_type"]),
        "group_mix_pct": mix(pop["group"]),
        "tier_mix_pct": mix(pop["tier"]),
        "tier_mix_eligible_pct": mix(pop.loc[pop["rel_eligible"], "tier"]),
    },
    "datasets_combined": {
        "n_metrics": len(per_metric),
        "metric_type_mix_pct": mix(per_metric["metric_type"]),
        "group_mix_pct": mix(per_metric["group"]),
        "tier_mix_pct": mix(per_metric["tier"]),
    },
}

for ds, g in per_metric.groupby("dataset"):
    out[f"dataset:{ds}"] = {
        "n_metrics": len(g),
        "group_mix_pct": mix(g["group"]),
        "tier_mix_pct": mix(g["tier"]),
    }

# coverage: which population metric_types are absent from the labeled data
pop_types = pop["metric_type"].value_counts(normalize=True)
ds_types = set(per_metric["metric_type"].unique())
missing = pop_types[~pop_types.index.isin(ds_types)]
out["population_types_missing_from_datasets_pct"] = {
    k: round(100 * float(v), 2) for k, v in missing.head(15).items()
}
out["population_share_covered_by_dataset_types_pct"] = round(
    100 * float(pop_types[pop_types.index.isin(ds_types)].sum()), 1
)

(HERE / "population_compare.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
