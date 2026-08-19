"""Census of the labeled anomaly datasets referenced by
tests/anomaly/test_gen_e2e_pipeline.py (plus the unreferenced random-a-500).

For each dataset: unique labeled samples, label distribution, metric_type
mix, and history depth vs GuardedBand calibration needs (>=10 points in a
56-day trailing window; Const carve-out needs >=40 points).

Run from backend/:  uv run python ../analysis_temp/guarded-band/dataset_census.py
"""

import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import pandas as pd

from app.brain.metric_tests.test_object_factory import create_test_instance
from app.models.metric_history_model import MetricHistory
from app.tests_gen.datasets.raw_data_loader import load_metrics_and_tests_entries

BACKEND = Path(__file__).resolve().parents[2] / "backend"
TEST_SETS = BACKEND / "tests/anomaly/inputs/test-sets"
TRAIN_RAW = BACKEND / "tests/anomaly/inputs/train_data/raw"
OUT = Path(__file__).resolve().parent

CAL_WINDOW_DAYS = 56
MIN_CAL_POINTS = 10
MIN_CONST_LOOKBACK = 40
HOLDOUT_DAYS = 7  # shadow design: calibrate on 56d, score the following week

DATASETS = {
    "test-set-single": ("inline", TEST_SETS / "test-set-single.jsonlines"),
    "test-set-1-agoda": ("inline", TEST_SETS / "agoda/test-set-1-agoda.jsonlines"),
    "test-set-2-agoda": ("inline", TEST_SETS / "agoda/test-set-2-agoda.jsonlines"),
    "test-set-tom-2": ("inline", TEST_SETS / "test-set-tom-2.json"),
    "random-a-200": ("manual", TRAIN_RAW / "random-a-200"),
    "random-a-500": ("manual", TRAIN_RAW / "random-a-500"),
}


def inline_rows(path: Path) -> pd.DataFrame:
    rows = []
    for entry in load_metrics_and_tests_entries(path.as_posix()):
        mh = MetricHistory(
            metric_id=entry["metric_id"],
            metric_type=entry["metric_type"],
            metric_values=entry["metric_values"],
            app_pits=entry["app_pits"],
        )
        test = entry.get("test")
        labels = [None] * len(mh.app_pits)
        bounded = [False] * len(mh.app_pits)
        if test:
            t = create_test_instance({**mh.model_dump(), **test})
            preds = t.predict_all_pits(mh)
            labels = [not p.is_passed(v) for p, v in zip(preds, mh.metric_values)]
            bounded = [
                p.lower_bound is not None or p.upper_bound is not None for p in preds
            ]
        for pit, val, lab, b in zip(mh.app_pits, mh.metric_values, labels, bounded):
            rows.append(
                {
                    "metric_id": mh.metric_id,
                    "metric_type": mh.metric_type,
                    "metric_group_type": mh.metric_group_type,
                    "pit": pit.replace(tzinfo=None),
                    "value": val,
                    "is_anomaly": lab,
                    "has_bounds": b,
                    "has_test": test is not None,
                }
            )
    return pd.DataFrame(rows)


def manual_rows(root: Path) -> pd.DataFrame:
    raw_file = next((root / "raw_metrics").glob("*.json"))
    meta, hist = {}, []
    for entry in json.loads(raw_file.read_text()):
        meta[entry["metric_id"]] = (
            entry["metric_type"],
            entry.get("metric_group_type"),
        )
        for pit, val in zip(entry["app_pits"], entry["metric_values"]):
            hist.append(
                {"metric_id": entry["metric_id"], "pit": pit, "value": val}
            )
    hist_df = pd.DataFrame(hist)
    hist_df["pit"] = pd.to_datetime(hist_df["pit"])

    lab = pd.read_csv(root / "manual_labels/labels.csv")
    lab["pit"] = pd.to_datetime(lab["pit"])
    lab["is_anomaly"] = lab["is_anomaly"].astype(bool)
    df = hist_df.merge(
        lab[["metric_id", "pit", "is_anomaly"]], on=["metric_id", "pit"], how="left"
    )
    unmatched = lab.merge(
        hist_df[["metric_id", "pit"]], on=["metric_id", "pit"], how="left", indicator=True
    )
    n_unmatched = int((unmatched["_merge"] == "left_only").sum())
    if n_unmatched:
        print(f"  note: {n_unmatched} label rows have no matching raw point")
    df["metric_type"] = df["metric_id"].map(lambda m: meta[m][0])
    df["metric_group_type"] = df["metric_id"].map(lambda m: meta[m][1])
    df["has_bounds"] = df["is_anomaly"].notna()
    df["has_test"] = df["metric_id"].isin(lab["metric_id"].unique())
    return df


def history_depth(df: pd.DataFrame) -> dict:
    per_metric = []
    for mid, g in df.groupby("metric_id"):
        pits = g["pit"].sort_values()
        last = pits.iloc[-1]
        span = (last - pits.iloc[0]).days
        in_56d = int((pits >= last - timedelta(days=CAL_WINDOW_DAYS)).sum())
        cal_start = last - timedelta(days=CAL_WINDOW_DAYS + HOLDOUT_DAYS)
        cal_end = last - timedelta(days=HOLDOUT_DAYS)
        cal_pts = int(((pits >= cal_start) & (pits <= cal_end)).sum())
        eval_pts = int((pits > cal_end).sum())
        per_metric.append(
            {
                "metric_id": mid,
                "n_points": len(pits),
                "span_days": span,
                "pts_last_56d": in_56d,
                "gb_calibratable_now": in_56d >= MIN_CAL_POINTS,
                "gb_cal_plus_holdout": cal_pts >= MIN_CAL_POINTS and eval_pts >= 1,
                "const_lookback_ok": len(pits) >= MIN_CONST_LOOKBACK,
            }
        )
    pm = pd.DataFrame(per_metric)
    return {
        "n_metrics": len(pm),
        "points_per_metric_median": float(pm["n_points"].median()),
        "points_per_metric_min": int(pm["n_points"].min()),
        "points_per_metric_max": int(pm["n_points"].max()),
        "span_days_median": float(pm["span_days"].median()),
        "span_days_max": int(pm["span_days"].max()),
        "pct_gb_calibratable_now": round(100 * pm["gb_calibratable_now"].mean(), 1),
        "pct_gb_cal_plus_holdout": round(100 * pm["gb_cal_plus_holdout"].mean(), 1),
        "pct_const_lookback_ok": round(100 * pm["const_lookback_ok"].mean(), 1),
    }, pm


def main() -> None:
    summaries = {}
    all_points = []
    for name, (kind, path) in DATASETS.items():
        print(f"== {name} ({kind}) ==")
        df = inline_rows(path) if kind == "inline" else manual_rows(path)
        df = df.drop_duplicates(subset=["metric_id", "pit"])
        labeled = df[df["is_anomaly"].notna()]
        depth, per_metric = history_depth(df)
        summaries[name] = {
            "kind": kind,
            "n_metrics_total": int(df["metric_id"].nunique()),
            "n_metrics_labeled": int(labeled["metric_id"].nunique()),
            "n_points_total": len(df),
            "n_points_labeled": len(labeled),
            "n_anomaly": int(labeled["is_anomaly"].sum()),
            "n_normal": int((~labeled["is_anomaly"].astype(bool)).sum()),
            "pct_anomaly": round(
                100 * labeled["is_anomaly"].astype(bool).mean(), 2
            )
            if len(labeled)
            else None,
            "anomalous_metrics": int(
                labeled.groupby("metric_id")["is_anomaly"].any().sum()
            ),
            "pit_range": [
                str(df["pit"].min().date()),
                str(df["pit"].max().date()),
            ],
            "metric_type_top": dict(
                Counter(
                    df.drop_duplicates("metric_id")["metric_type"]
                ).most_common(12)
            ),
            "metric_group_type": dict(
                Counter(df.drop_duplicates("metric_id")["metric_group_type"])
            ),
            "history_depth": depth,
        }
        per_metric.insert(0, "dataset", name)
        all_points.append(df.assign(dataset=name))
        per_metric.to_csv(OUT / f"per_metric_{name}.csv", index=False)
        print(json.dumps(summaries[name], indent=2))

    combined = pd.concat(all_points, ignore_index=True)
    labeled = combined[combined["is_anomaly"].notna()]
    r200 = set(combined.loc[combined["dataset"] == "random-a-200", "metric_id"])
    r500 = set(combined.loc[combined["dataset"] == "random-a-500", "metric_id"])
    summaries["_cross_dataset"] = {
        "unique_labeled_metric_pit_pairs": int(
            labeled.drop_duplicates(["metric_id", "pit"]).shape[0]
        ),
        "unique_labeled_metric_ids": int(labeled["metric_id"].nunique()),
        "random_200_500_metric_overlap": len(r200 & r500),
        "random_200_size": len(r200),
        "random_500_size": len(r500),
    }
    print(json.dumps(summaries["_cross_dataset"], indent=2))
    (OUT / "dataset_census.json").write_text(json.dumps(summaries, indent=2))
    combined.to_parquet(OUT / "all_labeled_points.parquet", index=False)


if __name__ == "__main__":
    main()
