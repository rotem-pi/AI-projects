"""Part 1 tail-only slice: score System A and GuardedBand (N=5) only on
points OUTSIDE System A's training window.

System A trains on the `all-minus-tail` split and the headline evaluates it
on `all`, i.e. largely in-sample. The tail (= keys in all/labels.csv that
are absent from all-minus-tail/labels.csv) is the honest out-of-sample
slice for A; GuardedBand is causal everywhere, so the slice only removes
its early points. Reported per arm on its own covered tail points, plus a
head-to-head on common tail points.

Inputs: part1_verdicts_{A,B5}_<dataset>.csv (from part1_scoring.py) and the
System A harness dataset splits. Output: part1_tail_results.json + stdout.

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/part1_tail_slice.py
"""

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "backend/tests/anomaly/outputs"
DATASETS = ["random-a-200", "random-a-500", "a_self_labeled", "definity_tagging"]
ARMS = ["A", "B5"]
KEY = ["metric_id", "pit"]


def confusion(gt: pd.Series, pred: pd.Series) -> dict:
    tp = int((gt & pred).sum())
    fp = int((~gt & pred).sum())
    fn = int((gt & ~pred).sum())
    tn = int((~gt & ~pred).sum())
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall_pct": round(100 * tp / (tp + fn), 1) if tp + fn else None,
        "fpr_pct": round(100 * fp / (fp + tn), 3) if fp + tn else None,
        "precision_pct": round(100 * tp / (tp + fp), 1) if tp + fp else None,
    }


def tail_keys(ds: str) -> pd.DataFrame:
    base = OUTPUTS / "prophet_labeler_grid_search" / ds / "datasets/1"
    full = pd.read_csv(base / "all/labels.csv")[KEY]
    train = pd.read_csv(base / "all-minus-tail/labels.csv")[KEY].assign(_train=True)
    merged = full.merge(train, on=KEY, how="left")
    tail = merged[merged["_train"].isna()][KEY]
    assert len(tail) == len(full) - len(train), f"tail mismatch in {ds}"
    return tail


def main() -> None:
    results: dict = {"per_dataset": {}, "pooled": {}}
    pooled: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    pooled_common: list[pd.DataFrame] = []

    for ds in DATASETS:
        tail = tail_keys(ds)
        preds = {}
        for arm in ARMS:
            p = pd.read_csv(HERE / f"part1_verdicts_{arm}_{ds}.csv")
            p = p.merge(tail, on=KEY, how="inner")
            p["gt_is_anomaly"] = p["gt_is_anomaly"].astype(bool)
            p["pred_is_anomaly"] = p["pred_is_anomaly"].astype(bool)
            preds[arm] = p
            pooled[arm].append(p.assign(dataset=ds))
        common = preds["A"].merge(preds["B5"], on=KEY, suffixes=("_a", "_b"))
        pooled_common.append(common.assign(dataset=ds))
        results["per_dataset"][ds] = {
            "tail_points": len(tail),
            **{
                arm: confusion(preds[arm]["gt_is_anomaly"], preds[arm]["pred_is_anomaly"])
                | {"points": len(preds[arm]),
                   "anomalies": int(preds[arm]["gt_is_anomaly"].sum())}
                for arm in ARMS
            },
        }

    for arm in ARMS:
        allp = pd.concat(pooled[arm], ignore_index=True)
        results["pooled"][arm] = confusion(
            allp["gt_is_anomaly"], allp["pred_is_anomaly"]
        ) | {"points": len(allp), "anomalies": int(allp["gt_is_anomaly"].sum())}

    com = pd.concat(pooled_common, ignore_index=True)
    gt = com["gt_is_anomaly_a"]
    results["pooled"]["common_tail"] = {
        "points": len(com),
        "anomalies": int(gt.sum()),
        "A": confusion(gt, com["pred_is_anomaly_a"]),
        "B5": confusion(gt, com["pred_is_anomaly_b"]),
    }

    (HERE / "part1_tail_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
