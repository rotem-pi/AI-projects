"""
Regression / impact tests for commit 8f5e1c0 (test generation criteria and process).

Compares two **analytic train** implementations on the same CSV-backed history:

- **Legacy:** frozen ``LegacyAnalyticTestTimeSeriesModel`` in
  ``legacy_analytical_model_pre_8f5e1c0.py`` (same directory; parent of 8f5e1c0).
  Simulated **DB outcome** for ``branch="legacy"`` follows pre-8f5e1c0
  ``tests_generator`` rules (gen-fail delete auto, delete when new model scores
  worse than the snapshot test, replace only when much better).

- **Production (updated):** live ``AnalyticTestTimeSeriesModel`` from
  ``app.tests_gen.models.analytical_model``. Simulated outcome for
  ``branch="new"`` follows current ``tests_generator._generate`` intent: keep on
  gen failure unless the snapshot test is below the low-score threshold; delete
  auto when ``old_score`` is below that threshold and the new model is **not**
  much better; replace when the new model is much better.

Requires ``test_runs_base.csv`` in **this directory** (flat copy), or under
``alerts-tests-analysis/test_runs_base.csv`` when walking parents (monorepo layout).
If the file is missing, tests skip.

Run from ``alert-incidents-analysis/`` using ``./run_impact_tests.sh`` (see README).
Use **definity-app/backend/.venv** so sklearn and other backend deps resolve.

Eligible test_ids approximate the regeneration queue from `get_candidates.sql`
(`total_alerts > 0`): here we take distinct `test_id` with at least 8 runs and at
least one row with `is_passed_binary == 0` in the CSV export.

Environment:
- TEST_RUNS_ANALYSIS_LIMIT: optional positive int; cap simulated test_ids (unset =
  all eligible — can take several minutes). The cherry-pick deep dive needs at
  least CHERRY_PICK_N distinct simulated cases (see module constant).
- `0` or negative means no limit (all eligible).
- TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE: float threshold for the **new** branch
  only (mirrors ``SystemSettings.test_gen_auto_delete_below_score`` in
  ``tests_generator``). Default ``0.2``; set to ``0`` to disable low-score auto
  deletion in this simulation.
"""

from __future__ import annotations

import os
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pytest

from app.brain.anomaly.utils import calculate_scores, score_candidates
from app.brain.metric_tests.test_object_factory import create_test_instance
from app.models.metric_history_model import MetricHistory
from app.tests_gen.datasets.data_utils import TS_COL, VALUE_COL
from app.tests_gen.datasets.raw_data_loader import (
    convert_metrics_and_tests_to_df,
    normalize_metric_dict,
)
from app.tests_gen.models.analytic_utils import remove_outliers
from app.tests_gen.models.analytical_model import (
    AnalyticHyperParams,
    AnalyticTestTimeSeriesModel,
)
from legacy_analytical_model_pre_8f5e1c0 import LegacyAnalyticTestTimeSeriesModel

pytestmark = pytest.mark.slow

REPLACEMENT_SCORE_DELTA = 0.1
MAX_HISTORY_POINTS = 2000
MIN_POINTS = 8
CHERRY_PICK_N = 10
LONG_SERIES_CHERRY_THRESHOLD = 300
# Set to a test_id to print step-by-step legacy + new analytic pipeline traces (-s).
# Set to None to disable (default for CI / routine runs).
DEBUG_DIAGNOSE_TEST_ID: int | None = None

USECOLS = [
    "test_id",
    "run_value",
    "app_pit",
    "is_passed_binary",
    "test_type",
    "var1",
    "var2",
    "var3",
    "metric_type",
    "asset_name",
    "asset_type",
    "app_id",
]


def _resolve_test_runs_csv() -> Path | None:
    here = Path(__file__).resolve().parent
    local = here / "test_runs_base.csv"
    if local.is_file():
        return local
    for anchor in here.parents:
        candidate = anchor / "alerts-tests-analysis" / "test_runs_base.csv"
        if candidate.is_file():
            return candidate
    return None


def _analysis_limit() -> int | None:
    raw = os.environ.get("TEST_RUNS_ANALYSIS_LIMIT")
    if raw is None or raw == "":
        return None
    v = int(raw)
    if v <= 0:
        return None
    return v


def _impact_auto_delete_below_score() -> float:
    """Match ``SystemSettings.test_gen_auto_delete_below_score`` default (0.2); 0 disables."""
    raw = os.environ.get("TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE")
    if raw is None or str(raw).strip() == "":
        return 0.2
    return float(raw)


def _safe_float(x) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(x)


def _aggregate_eligible_test_ids(
    csv_path: Path,
) -> tuple[dict[int, int], dict[int, int]]:
    """Return (run_counts, fail_counts) per test_id using chunked reads."""
    run_counts: dict[int, int] = {}
    fail_counts: dict[int, int] = {}
    for chunk in pd.read_csv(
        csv_path, usecols=["test_id", "is_passed_binary"], chunksize=250_000
    ):
        g = chunk.groupby("test_id")
        for tid, sub in g:
            tid = int(tid)
            run_counts[tid] = run_counts.get(tid, 0) + len(sub)
            fail_counts[tid] = fail_counts.get(tid, 0) + int(
                (sub["is_passed_binary"] == 0).sum()
            )
    return run_counts, fail_counts


def _eligible_test_ids(
    run_counts: dict[int, int], fail_counts: dict[int, int]
) -> list[int]:
    return sorted(
        tid
        for tid, n in run_counts.items()
        if n >= MIN_POINTS and fail_counts.get(tid, 0) > 0
    )


def _load_rows_for_test_ids(csv_path: Path, want: set[int]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(csv_path, usecols=USECOLS, chunksize=250_000):
        sub = chunk[chunk["test_id"].isin(want)]
        if len(sub):
            pieces.append(sub)
    if not pieces:
        return pd.DataFrame(columns=USECOLS)
    return pd.concat(pieces, ignore_index=True)


def _build_metric_entry(sub: pd.DataFrame, test_id: int) -> dict[str, Any]:
    sub = sub.sort_values("app_pit").drop_duplicates(subset=["app_pit"], keep="last")
    if len(sub) > MAX_HISTORY_POINTS:
        sub = sub.iloc[-MAX_HISTORY_POINTS:]
    row0 = sub.iloc[0]
    pits = pd.to_datetime(sub["app_pit"], utc=True).tolist()
    vals = sub["run_value"].astype(float).tolist()
    app_id = row0["app_id"]
    try:
        app_id_int = int(app_id) if not pd.isna(app_id) else None
    except (TypeError, ValueError):
        app_id_int = None
    return {
        "metric_id": test_id,
        "metric_type": str(row0["metric_type"]),
        "asset_type": str(row0["asset_type"]),
        "asset_name": str(row0["asset_name"]),
        "schema_name": "test_runs_base",
        "app_id": app_id_int,
        "metric_data": [{"pit": pits[i], "value": vals[i]} for i in range(len(pits))],
        "test": {
            "test_id": test_id,
            "metric_id": test_id,
            "test_type": str(row0["test_type"]),
            "var1": _safe_float(row0["var1"]),
            "var2": _safe_float(row0["var2"]),
            "var3": _safe_float(row0["var3"]),
        },
    }


def _prepare_bundle(
    entry: dict[str, Any],
) -> tuple[dict[str, Any], MetricHistory, Any, dict[str, Any]]:
    normalize_metric_dict(entry)
    metrics, _tests = convert_metrics_and_tests_to_df([entry], with_metadata=True)
    mh = MetricHistory(
        metric_id=entry["metric_id"],
        metric_type=entry["metric_type"],
        metric_values=entry["metric_values"],
        app_pits=entry["app_pits"],
    )
    test_candidate = {
        **entry["test"],
        "is_auto": True,
        "asset_value": entry.get("asset_name"),
        "app_id": entry.get("app_id"),
    }
    return entry, mh, metrics, test_candidate


def _legacy_preprocess_stats(data: pd.DataFrame) -> dict[str, int]:
    hp = LegacyAnalyticTestTimeSeriesModel(0, 0).hyper_params
    clean, _, _ = remove_outliers(
        data,
        hp.anomaly_lower_bound_p,
        hp.anomaly_upper_bound_p,
        hp.base_iqr_factor,
    )
    return {"n_full_series": len(data), "n_after_outlier_removal_full": len(clean)}


def _new_preprocess_stats(data: pd.DataFrame) -> dict[str, int]:
    hp = AnalyticHyperParams()
    model = AnalyticTestTimeSeriesModel(0, 0, hyper_params=asdict(hp))
    fit_data = model._get_recent_fit_data(data)
    clean_recent, _, _ = remove_outliers(
        fit_data,
        hp.anomaly_lower_bound_p,
        hp.anomaly_upper_bound_p,
        hp.base_iqr_factor,
    )
    clean = model._protect_current_week_points(clean_recent, fit_data)
    return {
        "n_full_series": len(data),
        "n_recent_fit_window": len(fit_data),
        "n_after_outlier_removal_recent": len(clean_recent),
        "n_after_week_protection_merge": len(clean),
    }


def _train(
    model_cls, metric_id: int, mh: MetricHistory, metrics
) -> dict[str, Any] | None:
    try:
        model = model_cls(metric_id, metric_id, mh)
        return model.train(metrics, None)
    except Exception:
        return None


def _final_action(
    *,
    branch: Literal["legacy", "new"],
    fit: dict[str, Any] | None,
    mh: MetricHistory,
    test_candidate: dict[str, Any],
    auto_delete_below_score: float | None = None,
) -> tuple[str, float | None, float | None, bool]:
    """Return (action, old_score, new_score, new_is_much_better).

    ``branch="legacy"`` mirrors pre-8f5e1c0 ``tests_generator`` delete/replace rules.
    ``branch="new"`` mirrors current production ``tests_generator`` (including
    optional low-score auto delete when ``old_score`` is below the threshold).
    """
    old_test = create_test_instance(test_candidate)
    old_score = calculate_scores(mh, old_test)["score"]
    is_auto = bool(test_candidate.get("is_auto"))
    thr = (
        auto_delete_below_score
        if auto_delete_below_score is not None
        else _impact_auto_delete_below_score()
    )

    if fit is None:
        if branch == "legacy":
            return "gen_fail_delete_auto", old_score, None, False
        if is_auto and thr > 0 and old_score < thr:
            return "auto_delete_low_score", old_score, None, False
        return "gen_fail_keep_auto", old_score, None, False

    new_score = fit["scores"]["score"]
    new_is_much_better = new_score > old_score + REPLACEMENT_SCORE_DELTA
    if branch == "legacy":
        if new_is_much_better:
            return "replace", old_score, new_score, True
        if new_score < old_score:
            return "delete_no_replacement", old_score, new_score, False
        return "keep_existing", old_score, new_score, False

    if new_is_much_better:
        return "replace", old_score, new_score, True
    if is_auto and thr > 0 and old_score < thr:
        return "auto_delete_low_score", old_score, new_score, False
    return "keep_existing", old_score, new_score, False


def _candidate_rule(fit: dict[str, Any] | None) -> str | None:
    if fit is None:
        return None
    t = fit["test"]
    parts = [t.test_type, t.var1]
    if hasattr(t, "var2"):
        parts.append(t.var2)
    if hasattr(t, "var3"):
        parts.append(getattr(t, "var3", None))
    return str(parts)


def _existing_test_rule(test_candidate: dict[str, Any]) -> str:
    """Human-readable params for the test currently tied to the metric (CSV snapshot)."""
    parts = [
        test_candidate.get("test_type"),
        test_candidate.get("var1"),
        test_candidate.get("var2"),
        test_candidate.get("var3"),
    ]
    return str(parts)


def _legacy_quality_reject_buggy_parse(
    score: float, *, n_app_pits: int, hp_frac: float, max_cnt: int
) -> tuple[bool, str]:
    """Match legacy `filter_tests_by_policy` condition (pre-8f5e1c0 operator precedence)."""
    count_term = (1 - score) * n_app_pits
    part_and = "score" == "score" and score <= hp_frac
    part_or = count_term > max_cnt
    rejected = part_and or part_or
    detail = (
        f"legacy_parse reject={rejected} "
        f"((score_type=='score' AND score<={hp_frac})={part_and}) "
        f"OR ((1-score)*N={count_term:.4f}>{max_cnt})={part_or} N={n_app_pits}"
    )
    return rejected, detail


def _diagnose_legacy_analytic_pipeline(
    tid: int, mh: MetricHistory, metrics, legacy_fit: dict[str, Any] | None
) -> None:
    """Replay LegacyAnalyticTestTimeSeriesModel.train steps for side-by-side comparison."""
    banner = f" DIAGNOSTIC legacy analytic pipeline test_id={tid} "
    print(f"\n{'=' * 20}{banner}{'=' * 20}")
    print(
        f"metric_group_type={mh.metric_group_type!r} metric_type={mh.metric_type!r} "
        f"n_app_pits={len(mh.app_pits)}"
    )
    try:
        model = LegacyAnalyticTestTimeSeriesModel(tid, tid, mh)
        hp = model.hyper_params
        data = metrics.reset_index()
        clean, anomalies, _ = remove_outliers(
            data,
            hp.anomaly_lower_bound_p,
            hp.anomaly_upper_bound_p,
            hp.base_iqr_factor,
        )
        print(
            f"full_series_rows={len(data)} clean_after_iqr={len(clean)} "
            f"anomalies_removed={len(anomalies)}"
        )
        candidates = [
            t
            for t in (
                model.get_const(clean),
                model.get_trend(clean),
                model.get_range(clean),
                model.get_pct(data),
            )
            if t is not None
        ]
        print(f"built_candidates single_path={len(candidates)}")
        for j, t in enumerate(candidates):
            v1, v2, v3 = (
                getattr(t, "var1", None),
                getattr(t, "var2", None),
                getattr(t, "var3", None),
            )
            print(f"  [legacy#{j}] {t.test_type} var1={v1} var2={v2} var3={v3}")
        if not candidates:
            print(
                "No candidate test objects — get_range/get_trend/get_pct/get_const all None."
            )
            print("=" * (40 + len(banner)))
            return
        scored = score_candidates(
            candidates, mh, limit=model.hyper_params.scoring_threshold
        )
        n_pits = len(mh.app_pits)
        hp_frac = 1 - model.hyper_params.max_train_anomalies_fraction
        max_cnt = model.hyper_params.max_train_anomalies_count
        print(
            f"scoring_window_limit={model.hyper_params.scoring_threshold} "
            f"legacy_count_uses_N=len(app_pits)={n_pits}"
        )
        print("Per scored candidate (legacy: one path const/trend/range/pct):")
        for i, tc in enumerate(scored):
            s = tc["scores"]["score"]
            _, legacy_detail = _legacy_quality_reject_buggy_parse(
                s, n_app_pits=n_pits, hp_frac=hp_frac, max_cnt=max_cnt
            )
            one = dict(tc)
            kept = model.filter_tests_by_policy([one], mh, model.score_type)
            policy_ok = len(kept) > 0
            pri = one.get("priority", "<none>")
            print(
                f"  [{i}] {_candidate_rule(tc)} score={s:.4f} "
                f"{legacy_detail} "
                f"policy_kept={policy_ok} priority={pri} full_scores={tc['scores']}"
            )
        final = model.select_final_test(mh, scored)
        print(
            f"select_final_test -> "
            f"{'None' if final is None else _candidate_rule(final)}"
        )
        print(
            f"_train() legacy_fit -> "
            f"{'None' if legacy_fit is None else _candidate_rule(legacy_fit)}"
        )
    except Exception as e:
        print(f"DIAGNOSTIC raised: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"{'=' * (40 + len(banner))}\n")


def _diagnose_new_analytic_pipeline(
    tid: int, mh: MetricHistory, metrics, new_fit: dict[str, Any] | None
) -> None:
    """Replay AnalyticTestTimeSeriesModel.train steps; print why select_final_test may be None."""
    banner = f" DIAGNOSTIC new analytic pipeline test_id={tid} "
    print(f"\n{'=' * 20}{banner}{'=' * 20}")
    print(
        f"metric_group_type={mh.metric_group_type!r} metric_type={mh.metric_type!r} "
        f"n_app_pits={len(mh.app_pits)}"
    )
    try:
        hp = AnalyticHyperParams()
        model = AnalyticTestTimeSeriesModel(tid, tid, mh, hyper_params=asdict(hp))
        data = metrics.reset_index()
        fit_data = model._get_recent_fit_data(data)
        clean_recent, anomalies, _ = remove_outliers(
            fit_data,
            hp.anomaly_lower_bound_p,
            hp.anomaly_upper_bound_p,
            hp.base_iqr_factor,
        )
        clean = model._protect_current_week_points(clean_recent, fit_data)
        print(
            f"fit_data_rows={len(fit_data)} clean_after_iqr={len(clean_recent)} "
            f"clean_after_week_merge={len(clean)} anomalies_removed={len(anomalies)}"
        )
        print("--- IQR drops vs week protection re-inclusion ---")
        latest_pit = fit_data[TS_COL].max()
        if pd.isna(latest_pit):
            print("latest_pit is NaN; skip reinclusion detail")
        else:
            week_start = latest_pit.normalize() - pd.to_timedelta(
                latest_pit.weekday(), unit="D"
            )
            print(
                f"protection_window: pit >= {week_start} (week of latest pit {latest_pit})"
            )
        if len(anomalies):
            print(
                "Rows IQR-anomaly on fit_data (excluded from clean_recent):\n"
                f"{anomalies[[TS_COL, VALUE_COL]].to_string(index=False)}"
            )
        else:
            print("No IQR anomalies on fit_data.")
        pits_recent = set(clean_recent[TS_COL])
        extra_rows = clean.loc[~clean[TS_COL].isin(pits_recent), [TS_COL, VALUE_COL]]
        if len(extra_rows):
            print(
                "Rows re-inserted by week protection (in clean but NOT in clean_recent):\n"
                f"{extra_rows.to_string(index=False)}"
            )
        elif len(clean) != len(clean_recent):
            print(
                f"WARNING: len(clean)={len(clean)} != len(clean_recent)={len(clean_recent)} "
                "but no extra pits found (check duplicate pit timestamps)."
            )
        else:
            print("clean and clean_recent have the same pit set (no re-inclusion).")
        _, anomalies_legacy, _ = remove_outliers(
            data,
            model.hyper_params.anomaly_lower_bound_p,
            model.hyper_params.anomaly_upper_bound_p,
            model.hyper_params.base_iqr_factor,
        )
        if len(anomalies_legacy):
            print(
                "Legacy full-series IQR anomalies (same hyper_params, full data rows):\n"
                f"{anomalies_legacy[[TS_COL, VALUE_COL]].to_string(index=False)}"
            )
        else:
            print("Legacy full-series: no IQR anomalies.")
        cleaned_candidates = model._build_candidates(clean=clean, data_for_pct=fit_data)
        raw_candidates = model._build_candidates(clean=fit_data, data_for_pct=fit_data)
        print(
            f"built_candidates cleaned_path={len(cleaned_candidates)} "
            f"raw_path={len(raw_candidates)}"
        )
        for label, lst in (
            ("cleaned", cleaned_candidates),
            ("raw_recent", raw_candidates),
        ):
            for j, t in enumerate(lst):
                v1, v2, v3 = (
                    getattr(t, "var1", None),
                    getattr(t, "var2", None),
                    getattr(t, "var3", None),
                )
                print(f"  [{label}#{j}] {t.test_type} var1={v1} var2={v2} var3={v3}")
        merged = cleaned_candidates + raw_candidates
        if not merged:
            print(
                "No candidate test objects — get_range/get_trend/get_pct/get_const all None."
            )
            print("=" * (40 + len(banner)))
            return
        scored = score_candidates(
            merged, mh, limit=model.hyper_params.scoring_threshold
        )
        scoring_points = (
            min(len(mh.app_pits), model.hyper_params.scoring_threshold)
            if model.hyper_params.scoring_threshold
            else len(mh.app_pits)
        )
        hp_frac = 1 - model.hyper_params.max_train_anomalies_fraction
        max_cnt = model.hyper_params.max_train_anomalies_count
        print(
            f"scoring_window_limit={model.hyper_params.scoring_threshold} "
            f"scoring_points={scoring_points}"
        )
        print("Per scored candidate (same order as train: cleaned then raw):")
        for i, tc in enumerate(scored):
            s = tc["scores"]["score"]
            count_term = (1 - s) * scoring_points
            q_fail_frac = s <= hp_frac
            q_fail_cnt = count_term > max_cnt
            one = dict(tc)
            kept = model.filter_tests_by_policy([one], mh, model.score_type)
            policy_ok = len(kept) > 0
            pri = one.get("priority", "<none>")
            print(
                f"  [{i}] {_candidate_rule(tc)} score={s:.4f} "
                f"quality_fail_frac={q_fail_frac} (score<={hp_frac}) "
                f"quality_fail_cnt={q_fail_cnt} "
                f"((1-score)*scoring_points={count_term:.4f}>{max_cnt}) "
                f"policy_kept={policy_ok} priority={pri} full_scores={tc['scores']}"
            )
        final = model.select_final_test(mh, scored)
        print(
            f"select_final_test -> "
            f"{'None' if final is None else _candidate_rule(final)}"
        )
        print(
            f"_train() new_fit -> "
            f"{'None' if new_fit is None else _candidate_rule(new_fit)}"
        )
    except Exception as e:
        print(f"DIAGNOSTIC raised: {type(e).__name__}: {e}")
        traceback.print_exc()
    print(f"{'=' * (40 + len(banner))}\n")


def _quality_rule_explanation(
    branch: Literal["legacy", "new"], score: float | None, n_app_pits: int
) -> str:
    hp_frac = 1 - 0.15
    max_cnt = 10
    scoring_window = 50
    if score is None:
        return "no candidate score (generation failed)"
    if branch == "legacy":
        count_term = (1 - score) * n_app_pits
        return (
            f"legacy: reject if (score<={hp_frac}) OR ((1-score)*len(app_pits)>{max_cnt}); "
            f"(1-score)*N={count_term:.4f} with N={n_app_pits}"
        )
    count_term = (1 - score) * min(n_app_pits, scoring_window)
    return (
        f"new: reject if score<={hp_frac} OR ((1-score)*min(N,{scoring_window})>{max_cnt}); "
        f"(1-score)*min(N,{scoring_window})={count_term:.4f} with N={n_app_pits}"
    )


@dataclass
class CaseResult:
    test_id: int
    n_runs: int
    existing_test_summary: str
    legacy_fit: dict | None
    new_fit: dict | None
    legacy_action: str
    new_action: str
    old_score: float | None
    legacy_new_score: float | None
    new_new_score: float | None
    legacy_preprocess: dict
    new_preprocess: dict


def _run_cohort(
    csv_path: Path, limit: int | None
) -> tuple[list[int], list[CaseResult]]:
    run_counts, fail_counts = _aggregate_eligible_test_ids(csv_path)
    eligible = _eligible_test_ids(run_counts, fail_counts)
    if limit is not None:
        eligible = eligible[:limit]

    if not eligible:
        return [], []

    auto_delete_thr = _impact_auto_delete_below_score()
    rows = _load_rows_for_test_ids(csv_path, set(eligible))
    results: list[CaseResult] = []
    grouped = {tid: g for tid, g in rows.groupby("test_id")}
    for tid in eligible:
        sub = grouped.get(tid)
        if sub is None or len(sub) < MIN_POINTS:
            continue
        entry = _build_metric_entry(sub, int(tid))
        _entry, mh, metrics, test_candidate = _prepare_bundle(entry)
        data = metrics.reset_index()
        legacy_fit = _train(LegacyAnalyticTestTimeSeriesModel, tid, mh, metrics)
        new_fit = _train(AnalyticTestTimeSeriesModel, tid, mh, metrics)
        if DEBUG_DIAGNOSE_TEST_ID is not None and int(tid) == DEBUG_DIAGNOSE_TEST_ID:
            _diagnose_legacy_analytic_pipeline(int(tid), mh, metrics, legacy_fit)
            _diagnose_new_analytic_pipeline(int(tid), mh, metrics, new_fit)
        leg_act, old_s, lns, _ = _final_action(
            branch="legacy", fit=legacy_fit, mh=mh, test_candidate=test_candidate
        )
        new_act, _os2, nns, _ = _final_action(
            branch="new",
            fit=new_fit,
            mh=mh,
            test_candidate=test_candidate,
            auto_delete_below_score=auto_delete_thr,
        )
        assert old_s == _os2
        results.append(
            CaseResult(
                test_id=int(tid),
                n_runs=len(entry["metric_values"]),
                existing_test_summary=_existing_test_rule(test_candidate),
                legacy_fit=legacy_fit,
                new_fit=new_fit,
                legacy_action=leg_act,
                new_action=new_act,
                old_score=old_s,
                legacy_new_score=lns,
                new_new_score=nns,
                legacy_preprocess=_legacy_preprocess_stats(data),
                new_preprocess=_new_preprocess_stats(data),
            )
        )
    return eligible, results


def test_test_runs_base_overall_impact_vs_pre_8f5e1c0():
    csv_path = _resolve_test_runs_csv()
    if csv_path is None:
        pytest.skip(
            "alerts-tests-analysis/test_runs_base.csv not found next to repo roots"
        )

    run_counts, fail_counts = _aggregate_eligible_test_ids(csv_path)
    eligible_all = _eligible_test_ids(run_counts, fail_counts)
    limit = _analysis_limit()
    _, results = _run_cohort(csv_path, limit)

    assert results, "no cases simulated — check CSV path or filters"

    leg_c = Counter(r.legacy_action for r in results)
    neu_c = Counter(r.new_action for r in results)

    legacy_gen_success = sum(1 for r in results if r.legacy_fit is not None)
    new_gen_success = sum(1 for r in results if r.new_fit is not None)
    legacy_gen_fail = sum(1 for r in results if r.legacy_fit is None)
    new_gen_fail = sum(1 for r in results if r.new_fit is None)

    cohort_n = len(results)
    eligible_n = len(eligible_all)
    auto_delete_thr = _impact_auto_delete_below_score()
    print(
        "\n=== test_runs_base cohort ===\n"
        f"csv={csv_path}\n"
        f"eligible_test_ids (>= {MIN_POINTS} runs & any failure): {eligible_n}\n"
        f"simulated_test_ids (limit={limit!r}): {cohort_n}\n"
        f"impact_auto_delete_below_score (new branch only)={auto_delete_thr} "
        f"(env TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE; 0=off)\n"
        f"legacy generation success: {legacy_gen_success}\n"
        f"new generation success: {new_gen_success}\n"
        f"legacy generation failures: {legacy_gen_fail}\n"
        f"new generation failures: {new_gen_fail}\n"
        f"legacy action counts: {dict(leg_c)}\n"
        f"new action counts: {dict(neu_c)}\n"
        "Deletion-style outcomes use these labels (not the word 'delete' alone): "
        "legacy → gen_fail_delete_auto | delete_no_replacement; "
        "new → auto_delete_low_score (only if old_score < threshold and no replace). "
        "auto_delete_low_score is often 0 when TEST_RUNS_ANALYSIS_LIMIT keeps only "
        "the first eligible test_ids (sorted): they skew toward high old_score; "
        "run without limit or set TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE higher to "
        "see more new-branch deletions.\n"
    )

    assert new_gen_success >= legacy_gen_success
    assert new_gen_fail <= legacy_gen_fail

    # New production path never uses legacy's "worse new model → delete" bucket name.
    assert neu_c.get("delete_no_replacement", 0) == 0
    assert leg_c.get("delete_no_replacement", 0) >= 0


def test_test_runs_base_cherrypicked_deep_dive():
    csv_path = _resolve_test_runs_csv()
    if csv_path is None:
        pytest.skip(
            "alerts-tests-analysis/test_runs_base.csv not found next to repo roots"
        )

    limit = _analysis_limit()
    _, results = _run_cohort(csv_path, limit)
    assert len(results) >= CHERRY_PICK_N, (
        f"need at least {CHERRY_PICK_N} simulated cases for deep dive "
        "(unset TEST_RUNS_ANALYSIS_LIMIT or raise it)"
    )

    def interest_key(r: CaseResult) -> tuple[int, ...]:
        """Higher tuple = more interesting for documentation / deep dive."""
        cand_diff = 0
        if r.legacy_fit is not None and r.new_fit is not None:
            cand_diff = int(_candidate_rule(r.legacy_fit) != _candidate_rule(r.new_fit))
        week_protection = int(
            r.new_preprocess.get("n_after_week_protection_merge", 0)
            > r.new_preprocess.get("n_after_outlier_removal_recent", 0)
        )
        preprocess_divergence = int(
            r.legacy_preprocess.get("n_after_outlier_removal_full")
            != r.new_preprocess.get("n_after_week_protection_merge")
        )
        return (
            int(r.legacy_action == "keep_existing" and r.new_action == "replace"),
            int(r.legacy_action == "replace" and r.new_action == "keep_existing"),
            int(r.new_action == "auto_delete_low_score"),
            int(
                r.legacy_action == "delete_no_replacement"
                and r.new_action != "delete_no_replacement"
            ),
            int(
                r.legacy_action == "gen_fail_delete_auto"
                and r.new_action != "gen_fail_delete_auto"
            ),
            int(r.legacy_fit is None and r.new_fit is not None),
            int(r.new_fit is None and r.legacy_fit is not None),
            int(r.legacy_action == "replace" and r.new_action == "replace"),
            cand_diff,
            week_protection,
            preprocess_divergence,
            int(r.n_runs >= LONG_SERIES_CHERRY_THRESHOLD),
            int(r.legacy_action != r.new_action),
            int(r.legacy_fit is not None and r.new_fit is not None),
            r.n_runs,
        )

    ranked = sorted(results, key=interest_key, reverse=True)

    # Always include at least one case where legacy produced a candidate (legacy_fit).
    legacy_nonempty = [r for r in results if r.legacy_fit is not None]
    if not legacy_nonempty:
        pytest.skip(
            "no simulated cases had legacy generation success; widen cohort or CSV"
        )

    cherry: list[CaseResult] = []
    seen: set[int] = set()

    def _take_one(predicate) -> None:
        if len(cherry) >= CHERRY_PICK_N:
            return
        pool = [r for r in results if predicate(r) and r.test_id not in seen]
        if not pool:
            return
        pick = max(pool, key=interest_key)
        cherry.append(pick)
        seen.add(pick.test_id)

    # High-signal buckets (one exemplar each when the cohort has them).
    _take_one(
        lambda r: r.legacy_action == "keep_existing" and r.new_action == "replace"
    )
    _take_one(
        lambda r: r.legacy_action == "replace" and r.new_action == "keep_existing"
    )
    # Production low-score auto delete (early so CHERRY_PICK_N slots are not exhausted).
    _take_one(lambda r: r.new_action == "auto_delete_low_score")
    _take_one(
        lambda r: (
            r.legacy_action == "gen_fail_delete_auto"
            and r.new_action == "auto_delete_low_score"
        )
    )
    _take_one(
        lambda r: (
            r.legacy_action == "delete_no_replacement"
            and r.new_action != "delete_no_replacement"
        )
    )
    _take_one(
        lambda r: (
            r.legacy_action == "gen_fail_delete_auto"
            and r.new_action != "gen_fail_delete_auto"
        )
    )
    _take_one(lambda r: r.legacy_fit is None and r.new_fit is not None)
    _take_one(lambda r: r.new_fit is None and r.legacy_fit is not None)
    _take_one(lambda r: r.legacy_action == "replace" and r.new_action == "replace")
    _take_one(
        lambda r: (
            r.legacy_fit is not None
            and r.new_fit is not None
            and _candidate_rule(r.legacy_fit) != _candidate_rule(r.new_fit)
        )
    )
    _take_one(
        lambda r: (
            r.new_preprocess.get("n_after_week_protection_merge", 0)
            > r.new_preprocess.get("n_after_outlier_removal_recent", 0)
        )
    )
    _take_one(lambda r: r.n_runs >= LONG_SERIES_CHERRY_THRESHOLD)

    if not any(r.legacy_fit is not None for r in cherry):
        pick = max(legacy_nonempty, key=interest_key)
        cherry.append(pick)
        seen.add(pick.test_id)

    for r in ranked:
        if len(cherry) >= CHERRY_PICK_N:
            break
        if r.test_id not in seen:
            cherry.append(r)
            seen.add(r.test_id)
    for r in results:
        if len(cherry) >= CHERRY_PICK_N:
            break
        if r.test_id not in seen:
            cherry.append(r)
            seen.add(r.test_id)

    assert len(cherry) == CHERRY_PICK_N
    assert any(r.legacy_fit is not None for r in cherry)

    if any(
        r.legacy_action == "keep_existing" and r.new_action == "replace"
        for r in results
    ):
        assert any(
            r.legacy_action == "keep_existing" and r.new_action == "replace"
            for r in cherry
        ), "cohort has legacy=keep new=replace but cherry missed it"

    if any(
        r.legacy_action == "replace" and r.new_action == "keep_existing"
        for r in results
    ):
        assert any(
            r.legacy_action == "replace" and r.new_action == "keep_existing"
            for r in cherry
        ), "cohort has legacy=replace new=keep but cherry missed it"

    if any(
        r.legacy_action == "delete_no_replacement"
        and r.new_action != "delete_no_replacement"
        for r in results
    ):
        assert any(
            r.legacy_action == "delete_no_replacement"
            and r.new_action != "delete_no_replacement"
            for r in cherry
        ), "cohort has legacy delete_no_replacement mismatch but cherry missed it"

    if any(r.new_action == "auto_delete_low_score" for r in results):
        assert any(r.new_action == "auto_delete_low_score" for r in cherry), (
            "cohort has new auto_delete_low_score but cherry missed it"
        )

    lines = [
        f"=== deep dive ({CHERRY_PICK_N} test_ids) ===",
        f"impact_auto_delete_below_score={_impact_auto_delete_below_score()} "
        "(new-branch simulated delete; env TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE)",
    ]
    for r in cherry:
        n_pits = r.n_runs
        lines.append(f"\n--- test_id={r.test_id} n_runs={r.n_runs} ---")
        lines.append(
            "existing_test (params from test_runs export / DB snapshot)="
            f"{r.existing_test_summary}"
        )
        lines.append(f"preprocess legacy: {r.legacy_preprocess}")
        lines.append(f"preprocess new: {r.new_preprocess}")
        for label, fit, score in (
            ("legacy", r.legacy_fit, r.legacy_new_score),
            ("new", r.new_fit, r.new_new_score),
        ):
            lines.append(
                f"{label} candidate={_candidate_rule(fit)} score={score} "
                f"| {_quality_rule_explanation('legacy' if label == 'legacy' else 'new', score, n_pits)}"
            )
        lines.append(
            f"old_test_score_on_50={r.old_score} replacement_delta={REPLACEMENT_SCORE_DELTA}"
        )
        lines.append(f"legacy_action={r.legacy_action} new_action={r.new_action}")

    report = "\n".join(lines)
    print(report)

    for r in cherry:
        assert r.n_runs >= MIN_POINTS
        assert r.old_score is not None

    global_diff = any(
        r.legacy_action != r.new_action or (r.legacy_fit is None) != (r.new_fit is None)
        for r in results
    )
    if global_diff:
        assert any(
            r.legacy_action != r.new_action
            or (r.legacy_fit is None) != (r.new_fit is None)
            for r in cherry
        ), "expected top ranked cases to include a legacy vs new difference"
