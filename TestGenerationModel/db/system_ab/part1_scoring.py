"""Part 1 scoring: System A vs System B (GuardedBand N=3 / N=5) on the
4 human-labeled harness datasets (B arm: N=5 only).

Inputs (produced by test_gen_e2e_pipeline.py runs):
  tests/anomaly/outputs/prophet_labeler_grid_search/<ds>/  (System A)
  tests/anomaly/outputs/guarded_band_system_n5/<ds>/       (GuardedBand N=5)

Computes, per dataset and pooled:
  1. headline per arm (its own covered points): recall/FPR/precision/coverage
  2. registry-symmetric slice (series whose metric_type is registry-allowed)
  3. head-to-head A vs each B arm: common points + exclusive catches +
     labeled-universe view with missing verdicts as "no flag"
  4. pooled tp/fp/fn/tn across the 4 datasets

Outputs under analysis_temp/guarded-band/:
  part1_results.json, part1_verdicts_<arm>_<ds>.csv

Run from backend/:
  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/part1_scoring.py
"""

import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest

from app.tests_gen.metrics.metrics_registry import MetricsRegistry

HERE = Path(__file__).resolve().parent
OUTPUTS = HERE.parents[1] / "backend/tests/anomaly/outputs"

ARMS = {
    "A": "prophet_labeler_grid_search",
    "B5": "guarded_band_system_n5",
}
ARM_LABELS = {
    "A": "System A (prophet grid search)",
    "B5": "GuardedBand (N=5)",
}
DATASETS = ["random-a-200", "random-a-500", "a_self_labeled", "definity_tagging"]
KEY = ["metric_id", "pit"]


def registry_allows(metric_type: str) -> bool:
    registry = MetricsRegistry.get_instance().registry
    lowered = (metric_type or "").lower()
    if lowered.startswith("custom."):
        return "custom" in registry
    return lowered in registry


def load_universe(ds: str) -> pd.DataFrame:
    base = OUTPUTS / ARMS["A"] / ds / "datasets/1/all"
    labels = pd.read_csv(base / "labels.csv")[KEY + ["is_anomaly"]]
    metrics = pd.read_csv(base / "metrics.csv")[
        KEY + ["metric_type", "metric_group_type"]
    ]
    uni = labels.merge(metrics, on=KEY, how="left")
    uni["is_anomaly"] = uni["is_anomaly"].astype(bool)
    uni["in_registry"] = uni["metric_type"].map(registry_allows)
    assert not uni.duplicated(subset=KEY).any(), f"dup universe keys in {ds}"
    return uni


def load_preds(arm: str, ds: str) -> pd.DataFrame:
    path = OUTPUTS / ARMS[arm] / ds / "results/1/eval/predictions.csv"
    p = pd.read_csv(path)
    p = p[KEY + ["metric_type", "metric_group_type", "gt_is_anomaly", "pred_is_anomaly"]]
    p["gt_is_anomaly"] = p["gt_is_anomaly"].astype(bool)
    p["pred_is_anomaly"] = p["pred_is_anomaly"].astype(bool)
    assert not p.duplicated(subset=KEY).any(), f"dup pred keys {arm}/{ds}"
    return p


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


def headline(preds: pd.DataFrame, universe: pd.DataFrame) -> dict:
    stats = confusion(preds["gt_is_anomaly"], preds["pred_is_anomaly"])
    stats.update(
        points=len(preds),
        anomalies=int(preds["gt_is_anomaly"].sum()),
        series_covered=int(preds["metric_id"].nunique()),
        series_total=int(universe["metric_id"].nunique()),
        series_coverage_pct=round(
            100 * preds["metric_id"].nunique() / universe["metric_id"].nunique(), 1
        ),
        point_coverage_pct=round(100 * len(preds) / len(universe), 1),
        universe_anomalies=int(universe["is_anomaly"].sum()),
    )
    return stats


def head_to_head(a: pd.DataFrame, b: pd.DataFrame, universe: pd.DataFrame) -> dict:
    merged = a.merge(b, on=KEY, suffixes=("_a", "_b"))
    gt = merged["gt_is_anomaly_a"]
    mism = int((merged["gt_is_anomaly_a"] != merged["gt_is_anomaly_b"]).sum())
    fa, fb = merged["pred_is_anomaly_a"], merged["pred_is_anomaly_b"]
    common = {
        "points": len(merged),
        "gt_mismatches": mism,
        "anomalies": int(gt.sum()),
        "agreement_pct": round(100 * (fa == fb).mean(), 1),
        "both_flag": int((fa & fb).sum()),
        "both_flag_true": int((fa & fb & gt).sum()),
        "only_a_flags": int((fa & ~fb).sum()),
        "only_a_flags_true": int((fa & ~fb & gt).sum()),
        "only_b_flags": int((~fa & fb).sum()),
        "only_b_flags_true": int((~fa & fb & gt).sum()),
        "neither_flags_true": int((~fa & ~fb & gt).sum()),
        "a_on_common": confusion(gt, fa),
        "b_on_common": confusion(gt, fb),
    }
    # labeled-universe view: missing verdict = no flag (production behavior)
    uni = universe.merge(
        a[KEY + ["pred_is_anomaly"]].rename(columns={"pred_is_anomaly": "flag_a"}),
        on=KEY, how="left",
    ).merge(
        b[KEY + ["pred_is_anomaly"]].rename(columns={"pred_is_anomaly": "flag_b"}),
        on=KEY, how="left",
    )
    anoms = uni[uni["is_anomaly"]]
    universe_view = {
        "points": len(uni),
        "anomalies": len(anoms),
        "a_covered_pct": round(100 * uni["flag_a"].notna().mean(), 1),
        "b_covered_pct": round(100 * uni["flag_b"].notna().mean(), 1),
        "anomalies_covered_a": int(anoms["flag_a"].notna().sum()),
        "anomalies_covered_b": int(anoms["flag_b"].notna().sum()),
        "a_catches": int(anoms["flag_a"].fillna(False).sum()),
        "b_catches": int(anoms["flag_b"].fillna(False).sum()),
        "a_exclusive_catches": int(
            (anoms["flag_a"].fillna(False) & ~anoms["flag_b"].fillna(False)).sum()
        ),
        "b_exclusive_catches": int(
            (anoms["flag_b"].fillna(False) & ~anoms["flag_a"].fillna(False)).sum()
        ),
        "a_recall_of_universe_pct": round(
            100 * anoms["flag_a"].fillna(False).mean(), 1
        ),
        "b_recall_of_universe_pct": round(
            100 * anoms["flag_b"].fillna(False).mean(), 1
        ),
    }
    return {"common_points": common, "universe_view": universe_view}


def mcnemar(b3: pd.DataFrame, b5: pd.DataFrame) -> dict:
    merged = b3.merge(b5, on=KEY, suffixes=("_3", "_5"))
    out = {
        "points_b3": len(b3),
        "points_b5": len(b5),
        "points_paired": len(merged),
        "identical_coverage": len(b3) == len(b5) == len(merged),
    }
    for is_anom, name in [(True, "anomalies"), (False, "normals")]:
        sub = merged[merged["gt_is_anomaly_3"] == is_anom]
        gained = int((~sub["pred_is_anomaly_3"] & sub["pred_is_anomaly_5"]).sum())
        lost = int((sub["pred_is_anomaly_3"] & ~sub["pred_is_anomaly_5"]).sum())
        p = binomtest(gained, gained + lost).pvalue if gained + lost else None
        out[name] = {
            "n5_gains": gained,
            "n5_losses": lost,
            "p_value": round(p, 4) if p is not None else None,
        }
    return out


def main() -> None:
    results: dict = {"per_dataset": {}, "pooled": {}, "registry": {}}
    pooled_frames: dict[str, list[pd.DataFrame]] = {arm: [] for arm in ARMS}
    pooled_universe: list[pd.DataFrame] = []

    for ds in DATASETS:
        universe = load_universe(ds)
        preds = {arm: load_preds(arm, ds) for arm in ARMS}
        for arm, p in preds.items():
            p.assign(dataset=ds).to_csv(
                HERE / f"part1_verdicts_{arm}_{ds}.csv", index=False
            )
            pooled_frames[arm].append(p.assign(dataset=ds))
        pooled_universe.append(universe.assign(dataset=ds))

        reg_universe = universe[universe["in_registry"]]
        reg_keys = set(universe.loc[universe["in_registry"], "metric_id"])
        ds_res = {
            "headline": {
                arm: headline(preds[arm], universe) for arm in ARMS
            },
            "registry_symmetric": {
                arm: headline(
                    preds[arm][preds[arm]["metric_id"].isin(reg_keys)], reg_universe
                )
                for arm in ARMS
            },
            "head_to_head_A_vs_B5": head_to_head(preds["A"], preds["B5"], universe),
        }
        results["per_dataset"][ds] = ds_res
        print(f"== {ds} scored ==", flush=True)

    uni_all = pd.concat(pooled_universe, ignore_index=True)
    reg_ids = set(
        zip(
            uni_all.loc[uni_all["in_registry"], "dataset"],
            uni_all.loc[uni_all["in_registry"], "metric_id"],
        )
    )
    for arm in ARMS:
        allp = pd.concat(pooled_frames[arm], ignore_index=True)
        results["pooled"][arm] = confusion(
            allp["gt_is_anomaly"], allp["pred_is_anomaly"]
        ) | {"points": len(allp), "anomalies": int(allp["gt_is_anomaly"].sum())}
        regp = allp[
            [(d, m) in reg_ids for d, m in zip(allp["dataset"], allp["metric_id"])]
        ]
        results["registry"][arm] = confusion(
            regp["gt_is_anomaly"], regp["pred_is_anomaly"]
        ) | {"points": len(regp), "anomalies": int(regp["gt_is_anomaly"].sum())}

    # pooled per metric-group breakdown, per arm on its own covered points
    group_rows = []
    for arm in ARMS:
        allp = pd.concat(pooled_frames[arm], ignore_index=True)
        for grp, sub in allp.groupby(allp["metric_group_type"].str.lower()):
            c = confusion(sub["gt_is_anomaly"], sub["pred_is_anomaly"])
            group_rows.append(
                {"arm": arm, "group": grp, "points": len(sub),
                 "anomalies": int(sub["gt_is_anomaly"].sum())} | c
            )
    results["pooled_by_group"] = group_rows

    (HERE / "part1_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps(results["pooled"], indent=2))
    print(json.dumps(results["registry"], indent=2))
    print("full results -> part1_results.json")


if __name__ == "__main__":
    main()
