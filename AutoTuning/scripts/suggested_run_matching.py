"""
Nearest-neighbor duration estimation for parametric config suggestions.

Given a dataset row with recommendations (config_key + suggested_value), build a
"suggested task run" config identity and find the closest historical task run with
the same app/task context. Returns that neighbor's duration as an estimate.

Duration is never used in the distance metric — only config and data-volume features
(vCore/memory time, IO record counts, and optional byte/spill metrics when present).
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from add_recommendations import CONFIG_KEY_TO_RUN_COLUMN

RUN_COLUMN_TO_CONFIG_KEY = {v: k for k, v in CONFIG_KEY_TO_RUN_COLUMN.items()}

DEFAULT_CONFIG_COLUMNS = list(CONFIG_KEY_TO_RUN_COLUMN.values())
DEFAULT_VOLUME_COLUMNS = [
    "vcore_seconds_used",
    "memory_gb_seconds_used",
]
RECORD_VOLUME_COLUMNS = [
    "input_records_read",
    "output_records_written",
    "shuffle_records_read",
    "shuffle_records_written",
]
# Legacy insights CSV names (pre xgboost parity)
LEGACY_VOLUME_COLUMNS = [
    "task__vcore_time__used",
    "executors__memory_time__used",
]
OPTIONAL_VOLUME_COLUMNS = [
    "executors_vcore_seconds_used",
    "total_io_bytes",
    "disk_bytes_spilled",
]
DEFAULT_CATEGORICAL_COLUMNS = [
    "worker_instance_type",
    "driver_instance_type",
    "aws_availability",
    "is_dynamic_allocation",
]
DURATION_COLUMN = "task_duration_seconds"
LEGACY_DURATION_COLUMN = "task__duration"
IDENTITY_COLUMNS = ["app_id", "task_name", "task_id"]


def _resolve_column(df: pd.DataFrame, primary: str, legacy: Optional[str] = None) -> str:
    if primary in df.columns:
        return primary
    if legacy and legacy in df.columns:
        return legacy
    return primary


def _volume_metric_columns(df: pd.DataFrame) -> list[str]:
    if "vcore_seconds_used" in df.columns:
        base = list(DEFAULT_VOLUME_COLUMNS)
    else:
        base = [c for c in LEGACY_VOLUME_COLUMNS if c in df.columns] or list(DEFAULT_VOLUME_COLUMNS)
    base.extend(c for c in RECORD_VOLUME_COLUMNS if c in df.columns and c not in base)
    return base


def _duration_column(df: pd.DataFrame) -> str:
    return _resolve_column(df, DURATION_COLUMN, LEGACY_DURATION_COLUMN)


@dataclass
class SimilarRunMatch:
    """Result of matching a suggested task run to the closest historical run."""

    suggested_task_run: dict[str, Any]
    similar_task_run: pd.Series
    distance: float
    config_distance: float
    volume_distance: float
    categorical_distance: float
    estimated_duration_seconds: float
    volume_ratios: dict[str, float] = field(default_factory=dict)
    max_volume_ratio: float = float("inf")
    volume_match_reliable: bool = False
    feature_weights: dict[str, float] = field(default_factory=dict)


def _row_metric_completeness(row: pd.Series, metric_columns: Optional[list[str]] = None) -> int:
    """Count non-null task-run metric columns (higher = more complete row)."""
    metric_columns = metric_columns or [
        "vcore_seconds_used",
        "memory_gb_seconds_used",
        "task_duration_seconds",
        *RECORD_VOLUME_COLUMNS,
        "task__vcore_time__used",
        "executors__memory_time__used",
        "task__duration",
    ]
    return sum(
        1 for col in metric_columns
        if col in row.index and pd.notna(row.get(col)) and str(row.get(col)).strip() != ""
    )


def dedupe_insight_task_rows(
    df: pd.DataFrame,
    *,
    subset: tuple[str, ...] = ("task_id", "insight_id"),
) -> pd.DataFrame:
    """
    Keep one row per (task_id, insight_id), preferring rows with task-run metrics.

    The insights CSV often contains duplicate pairs where one copy has null
    task_enrichment fields (from overlapping fetch checkpoints or sparse joins).
    """
    if df.empty:
        return df.copy()

    working = df.copy()
    working["_metric_completeness"] = working.apply(_row_metric_completeness, axis=1)
    working = working.sort_values("_metric_completeness", ascending=False)
    deduped = working.drop_duplicates(subset=list(subset), keep="first")
    return deduped.drop(columns=["_metric_completeness"]).reset_index(drop=True)


def dedupe_task_runs(
    df: pd.DataFrame,
    *,
    require_volume_metrics: bool = True,
) -> pd.DataFrame:
    """Keep one row per task_id with the most complete task-run metrics."""
    working = dedupe_insight_task_rows(df, subset=("task_id",))
    vcore_col = _resolve_column(working, "vcore_seconds_used", "task__vcore_time__used")
    if require_volume_metrics and vcore_col in working.columns:
        working = working[working[vcore_col].notna()]
    return working.reset_index(drop=True)


def prepare_matching_dataset(
    df: pd.DataFrame,
    *,
    for_recommendations: bool = False,
) -> pd.DataFrame:
    """
    Clean dataset for matching / recommendation analysis.

    - Deduplicates (task_id, insight_id) rows, keeping rows with metrics
    - Optionally filters to rows with recommendations and valid volume metrics
    """
    cleaned = dedupe_insight_task_rows(df)
    if for_recommendations:
        cleaned = cleaned[
            cleaned["suggested_value"].notna()
            & (cleaned["suggested_value"].astype(str).str.strip() != "")
        ]
    vcore_col = _resolve_column(cleaned, "vcore_seconds_used", "task__vcore_time__used")
    if vcore_col in cleaned.columns:
        cleaned = cleaned[cleaned[vcore_col].notna()]
    return cleaned.reset_index(drop=True)


def parse_list_or_scalar(raw: Any) -> Optional[list[Any]]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, list):
        return raw
    s = str(raw).strip()
    if not s:
        return None
    if s.startswith("["):
        try:
            parsed = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            try:
                parsed = json.loads(s)
            except json.JSONDecodeError:
                return None
        return parsed if isinstance(parsed, list) else [parsed]
    return [s]


def parse_recommendation_pairs(row: pd.Series) -> list[tuple[str, Any]]:
    """Return (config_key, suggested_value) pairs from a dataset row."""
    keys = parse_list_or_scalar(row.get("config_key"))
    values = parse_list_or_scalar(row.get("suggested_value"))
    if not keys or not values or len(keys) != len(values):
        return []
    return list(zip(keys, values))


def get_run_config(row: pd.Series, config_columns: Optional[list[str]] = None) -> dict[str, Any]:
    """Read the current task-run config as config_key -> value."""
    config_columns = config_columns or DEFAULT_CONFIG_COLUMNS
    config: dict[str, Any] = {}
    for source_col in config_columns:
        config_key = RUN_COLUMN_TO_CONFIG_KEY.get(source_col, source_col)
        val = row.get(source_col, row.get(config_key, np.nan))
        config[config_key] = val
        config[source_col] = val
    return config


def build_suggested_task_run(
    row: pd.Series,
    recommendation_pairs: Optional[list[tuple[str, Any]]] = None,
) -> dict[str, Any]:
    """
    Build a suggested task-run identity: same row config with recommendation overrides.

    Updates both config_key-named fields and their source CSV columns.
    """
    suggested = get_run_config(row)
    pairs = recommendation_pairs if recommendation_pairs is not None else parse_recommendation_pairs(row)

    for config_key, suggested_value in pairs:
        config_key = str(config_key).strip()
        suggested[config_key] = suggested_value
        source_col = CONFIG_KEY_TO_RUN_COLUMN.get(config_key)
        if source_col:
            suggested[source_col] = suggested_value

    suggested["_recommendation_pairs"] = pairs
    suggested["app_id"] = row.get("app_id")
    suggested["task_name"] = row.get("task_name")
    suggested["task_id"] = row.get("task_id")
    return suggested


def _numeric_series(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce")


def _log1p_median_impute(series: pd.Series, fallback: float = 0.0) -> pd.Series:
    numeric = _numeric_series(series)
    median = numeric.median()
    if np.isnan(median):
        median = fallback
    filled = numeric.fillna(median)
    return np.log1p(np.maximum(filled, 0))


class TaskRunMatcher:
    """
    Pre-compute feature statistics on a candidate pool, then match suggested runs.

    Candidates are filtered to the same (app_id, task_name) by default and exclude
    the source task_id.
    """

    def __init__(
        self,
        candidates: pd.DataFrame,
        *,
        config_columns: Optional[list[str]] = None,
        volume_columns: Optional[list[str]] = None,
        categorical_columns: Optional[list[str]] = None,
        config_weight: float = 1.0,
        volume_weight: float = 3.0,
        categorical_weight: float = 0.25,
        max_volume_ratio: float = 2.0,
        require_same_task_name: bool = True,
        require_same_app_id: bool = True,
        completed_only: bool = True,
    ):
        self.config_columns = config_columns or DEFAULT_CONFIG_COLUMNS
        self.volume_columns = volume_columns or self._resolve_volume_columns(candidates)
        self.categorical_columns = categorical_columns or [
            c for c in DEFAULT_CATEGORICAL_COLUMNS if c in candidates.columns
        ]
        self.config_weight = config_weight
        self.volume_weight = volume_weight
        self.categorical_weight = categorical_weight
        self.max_volume_ratio = max_volume_ratio
        self.require_same_task_name = require_same_task_name
        self.require_same_app_id = require_same_app_id
        self.completed_only = completed_only

        pool = dedupe_task_runs(candidates, require_volume_metrics=True)
        if self.completed_only and "status" in pool.columns:
            pool = pool[pool["status"] == "COMPLETED"]
        self.candidates = pool.reset_index(drop=True)

        self._config_medians: dict[str, float] = {}
        self._config_scales: dict[str, float] = {}
        self._fit_config_scalers()

    @staticmethod
    def _resolve_volume_columns(df: pd.DataFrame) -> list[str]:
        cols = _volume_metric_columns(df)
        cols.extend(c for c in OPTIONAL_VOLUME_COLUMNS if c in df.columns and c not in cols)
        cols.extend(c for c in LEGACY_VOLUME_COLUMNS if c in df.columns and c not in cols)
        return cols

    def _fit_config_scalers(self) -> None:
        for col in self.config_columns:
            numeric = _numeric_series(self.candidates[col])
            median = numeric.median()
            if np.isnan(median):
                median = 0.0
            mad = (numeric - median).abs().median()
            scale = mad if mad and not np.isnan(mad) else numeric.std()
            if not scale or np.isnan(scale):
                scale = 1.0
            self._config_medians[col] = median
            self._config_scales[col] = scale

    def _volume_distance_and_ratios(
        self,
        source_row: pd.Series,
        candidate_row: pd.Series,
    ) -> tuple[float, dict[str, float]]:
        """
        Symmetric log-ratio distance on used volume metrics, relative to the source run.

        Includes vCore/memory time, IO record counts (input/output/shuffle read/write),
        and optional byte/spill metrics when both runs have values.

        A 2x volume gap contributes log(2) ≈ 0.69 per metric; 5x contributes ≈ 1.61.
        """
        sq_terms: list[float] = []
        ratios: dict[str, float] = {}
        for col in self.volume_columns:
            source_val = pd.to_numeric(source_row.get(col), errors="coerce")
            candidate_val = pd.to_numeric(candidate_row.get(col), errors="coerce")
            if np.isnan(source_val) or np.isnan(candidate_val) or source_val <= 0 or candidate_val <= 0:
                continue
            ratio = float(candidate_val / source_val)
            ratios[col] = ratio
            sq_terms.append(float(np.log(ratio)) ** 2)
        if not sq_terms:
            return float("inf"), ratios
        return float(np.sqrt(sum(sq_terms))), ratios

    @staticmethod
    def _max_symmetric_ratio(ratios: dict[str, float]) -> float:
        if not ratios:
            return float("inf")
        return max(max(ratio, 1 / ratio) for ratio in ratios.values())

    def _vectorize_config(self, config: dict[str, Any]) -> np.ndarray:
        values = []
        for col in self.config_columns:
            config_key = RUN_COLUMN_TO_CONFIG_KEY.get(col, col)
            raw = config.get(col, config.get(config_key, np.nan))
            val = pd.to_numeric(raw, errors="coerce")
            if np.isnan(val):
                val = self._config_medians[col]
            values.append((val - self._config_medians[col]) / self._config_scales[col])
        return np.array(values, dtype=float)

    def _vectorize_volume(self, row: pd.Series) -> np.ndarray:
        values = []
        for col in self.volume_columns:
            val = pd.to_numeric(row.get(col), errors="coerce")
            if np.isnan(val) or val <= 0:
                val = 0.0
            values.append(float(np.log(val)))
        return np.array(values, dtype=float)

    def _categorical_distance(self, suggested: dict[str, Any], candidate: pd.Series) -> float:
        if not self.categorical_columns:
            return 0.0
        mismatches = 0
        compared = 0
        for col in self.categorical_columns:
            suggested_val = suggested.get(col, np.nan)
            if pd.isna(suggested_val):
                continue
            compared += 1
            left = str(suggested_val).strip()
            right = str(candidate.get(col, "")).strip()
            if left != right:
                mismatches += 1
        if compared == 0:
            return 0.0
        return mismatches / compared

    def _candidate_pool_for_row(self, row: pd.Series) -> pd.DataFrame:
        pool = self.candidates
        if self.require_same_app_id and "app_id" in row.index:
            pool = pool[pool["app_id"] == row["app_id"]]
        if self.require_same_task_name and "task_name" in row.index:
            pool = pool[pool["task_name"] == row["task_name"]]
        if "task_id" in row.index:
            pool = pool[pool["task_id"] != row["task_id"]]
        return pool

    def distance_to_row(
        self,
        suggested_task_run: dict[str, Any],
        source_row: pd.Series,
        candidate_row: pd.Series,
    ) -> tuple[float, float, float, float, dict[str, float]]:
        suggested_config = self._vectorize_config(suggested_task_run)
        candidate_config = self._vectorize_config(get_run_config(candidate_row, self.config_columns))
        config_distance = float(np.linalg.norm(suggested_config - candidate_config))

        volume_distance, volume_ratios = self._volume_distance_and_ratios(source_row, candidate_row)
        categorical_distance = self._categorical_distance(suggested_task_run, candidate_row)

        total = (
            self.config_weight * config_distance
            + self.volume_weight * volume_distance
            + self.categorical_weight * categorical_distance
        )
        return total, config_distance, volume_distance, categorical_distance, volume_ratios

    def find_similar_task_run(
        self,
        row: pd.Series,
        recommendation_pairs: Optional[list[tuple[str, Any]]] = None,
    ) -> Optional[SimilarRunMatch]:
        """
        Find the closest historical task run to the suggested config identity.

        Returns None if no candidates remain after filtering or if no candidate
        passes the max_volume_ratio threshold.
        """
        suggested_task_run = build_suggested_task_run(row, recommendation_pairs)
        pool = self._candidate_pool_for_row(row)
        if pool.empty:
            return None

        best_idx: Optional[int] = None
        best_total = best_config = best_volume = best_categorical = np.inf
        best_ratios: dict[str, float] = {}
        best_max_ratio = float("inf")

        for idx, candidate in pool.iterrows():
            total, config_d, volume_d, cat_d, ratios = self.distance_to_row(
                suggested_task_run, row, candidate
            )
            max_ratio = self._max_symmetric_ratio(ratios)
            if max_ratio > self.max_volume_ratio:
                continue
            if total < best_total:
                best_total, best_config, best_volume, best_categorical = (
                    total, config_d, volume_d, cat_d
                )
                best_ratios = ratios
                best_max_ratio = max_ratio
                best_idx = idx

        if best_idx is None:
            return None

        similar = pool.loc[best_idx]
        duration_col = _duration_column(self.candidates)
        duration = pd.to_numeric(similar.get(duration_col), errors="coerce")
        if np.isnan(duration):
            return None

        return SimilarRunMatch(
            suggested_task_run=suggested_task_run,
            similar_task_run=similar,
            distance=float(best_total),
            config_distance=float(best_config),
            volume_distance=float(best_volume),
            categorical_distance=float(best_categorical),
            estimated_duration_seconds=float(duration),
            volume_ratios=best_ratios,
            max_volume_ratio=float(best_max_ratio),
            volume_match_reliable=True,
            feature_weights={
                "config_weight": self.config_weight,
                "volume_weight": self.volume_weight,
                "categorical_weight": self.categorical_weight,
                "max_volume_ratio": self.max_volume_ratio,
            },
        )


def estimate_duration_for_row(
    row: pd.Series,
    candidates: pd.DataFrame,
    matcher: Optional[TaskRunMatcher] = None,
    recommendation_pairs: Optional[list[tuple[str, Any]]] = None,
) -> Optional[SimilarRunMatch]:
    """
    Convenience wrapper: build suggested run from a row and return the best match.

    Example
    -------
    >>> match = estimate_duration_for_row(row, data)
    >>> match.estimated_duration_seconds
    """
    engine = matcher or TaskRunMatcher(candidates)
    return engine.find_similar_task_run(row, recommendation_pairs)


def estimate_duration_for_suggestion(
    row: pd.Series,
    candidates: pd.DataFrame,
    config_key: str,
    suggested_value: Any,
    matcher: Optional[TaskRunMatcher] = None,
) -> Optional[SimilarRunMatch]:
    """Apply a single manual config suggestion and estimate duration."""
    return estimate_duration_for_row(
        row,
        candidates,
        matcher=matcher,
        recommendation_pairs=[(config_key, suggested_value)],
    )
