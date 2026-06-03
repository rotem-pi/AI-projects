"""
Gradual parameter-tuning experiment for multi-parameter recommendations.

Samples a task (app_id + task_name), takes the latest run, applies a partial step toward
all suggested values, estimates duration via similar-run matching, and computes run_cost_usd
using the same formula as the insights SQL fetch (not a separate cost model).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from add_recommendations import CONFIG_KEY_TO_RUN_COLUMN

# Memory keys stay fractional (GB); backend uses ceil on final suggestions.
MEMORY_CONFIG_KEYS = frozenset(
    {
        "spark.driver.memory",
        "spark.executor.memory",
        "spark.executor.memoryOverhead",
    }
)

# Count/partition keys are integers in task_enrichments and recommendations.
INTEGER_CONFIG_KEYS = frozenset(
    key for key in CONFIG_KEY_TO_RUN_COLUMN if key not in MEMORY_CONFIG_KEYS
)

# Minimums aligned with add_recommendations.py / Spark constraints.
CONFIG_KEY_INTEGER_MIN: dict[str, int] = {
    "spark.task.cpus": 1,
    "spark.executor.cores": 1,
    "spark.driver.cores": 1,
    "spark.executor.instances": 1,
    "spark.dynamicAllocation.minExecutors": 0,
    "spark.dynamicAllocation.maxExecutors": 1,
    "clusterMinWorkers": 0,
    "clusterWorkers": 0,
    "clusterMaxWorkers": 1,
    "spark.sql.shuffle.partitions": 1,
}

# Config keys used to infer effective executor count (first match wins).
EXECUTOR_COUNT_KEYS: list[tuple[str, str]] = [
    ("spark.executor.instances", "spark_executor_instances"),
    ("spark.dynamicAllocation.maxExecutors", "spark_dynamic_alloc_max_executors"),
    ("clusterWorkers", "cluster_workers"),
    ("clusterMaxWorkers", "cluster_max_workers"),
]
EXECUTOR_MEMORY_KEY = "spark.executor.memory"
EXECUTOR_MEMORY_COLUMN = "spark_executor_memory_gb"

from suggested_run_matching import (
    SimilarRunMatch,
    TaskRunMatcher,
    dedupe_task_runs,
    estimate_duration_for_row,
    parse_list_or_scalar,
    prepare_matching_dataset,
)


@dataclass
class ParamRecommendation:
    config_key: str
    current_value: Any
    suggested_value: Any
    current_num: Optional[float] = None
    suggested_num: Optional[float] = None

    def __post_init__(self) -> None:
        self.current_num = _to_float(self.current_value)
        self.suggested_num = _to_float(self.suggested_value)

    @property
    def is_numeric(self) -> bool:
        return self.current_num is not None and self.suggested_num is not None


@dataclass
class SimulationResult:
    label: str
    config_values: dict[str, Any]
    estimated_duration_seconds: Optional[float]
    run_cost_usd: Optional[float]
    match: Optional[SimilarRunMatch]
    match_found: bool
    executor_scale: float = 1.0
    memory_scale: float = 1.0
    notes: str = ""


@dataclass
class ExperimentResult:
    app_id: Any
    task_name: str
    task_id: Any
    baseline_duration_seconds: float
    duration_limit_seconds: float
    sensitivity_factor: float
    recommendations: list[ParamRecommendation]
    baseline: SimulationResult
    first_step: SimulationResult
    final: SimulationResult
    decision: str
    decision_detail: str


def _to_float(val: Any) -> Optional[float]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def duration_limit_seconds(
    current_duration_seconds: float,
    sensitivity_factor: float = 20.0,
) -> float:
    """
  Allowed max duration: current * (1 + sensitivity_factor/100).

  sensitivity_factor=20 → up to 20% longer than the current run.
  """
    return float(current_duration_seconds) * (1.0 + float(sensitivity_factor) / 100.0)


def compute_run_cost_usd(
    *,
    vcore_seconds_allocated: float,
    memory_gb_seconds_allocated: float,
    executor_total_memory_gb: float,
    spark_executor_cores: float,
    machine_memory_to_vcore_ratio: float,
    duration_seconds: float,
    vcore_price: float,
    memory_price: float,
) -> float:
    """
    Mirrors the SQL in data_exploration.ipynb (run_cost_usd).

    The hidden-vcore term scales with duration; vcore/memory *seconds* are taken from
    the run (or matched neighbor), not derived from Spark config keys directly.
    """
    ratio = machine_memory_to_vcore_ratio if machine_memory_to_vcore_ratio else np.nan
    hidden_vcores = 0.0
    if ratio and not np.isnan(ratio) and ratio > 0:
        hidden_vcores = max(
            (executor_total_memory_gb / ratio - spark_executor_cores) * duration_seconds,
            0.0,
        )
    vcore_billable = vcore_seconds_allocated + hidden_vcores
    return (
        vcore_billable * vcore_price / 3600.0
        + memory_gb_seconds_allocated * memory_price / 3600.0
    )


def compute_insight_usd_cost(
    impact_cost: float,
    impact_unit: str,
    vcore_price: float,
    memory_price: float,
    file_listing_price: float = 0.0,
) -> Optional[float]:
    """Mirrors insight_usd_cost CASE in the SQL fetch."""
    if impact_cost is None or (isinstance(impact_cost, float) and np.isnan(impact_cost)):
        return None
    unit = str(impact_unit or "").strip()
    if unit == "VCore":
        return float(impact_cost) * vcore_price / 3600.0
    if unit == "GB":
        return float(impact_cost) * memory_price / 3600.0
    if unit == "file-listing":
        return float(impact_cost) * file_listing_price / 1000.0
    if unit == "dollar":
        return float(impact_cost)
    return None


def _resource_row_for_cost(row: pd.Series) -> dict[str, float]:
    """Normalize legacy vs xgboost-parity column names for cost inputs."""
    duration = _to_float(row.get("task_duration_seconds", row.get("task__duration")))
    return {
        "vcore_seconds_allocated": _to_float(
            row.get("vcore_seconds_allocated", row.get("task__vcore_time__allocated"))
        )
        or 0.0,
        "memory_gb_seconds_allocated": _to_float(
            row.get("memory_gb_seconds_allocated", row.get("task__memory_time__allocated"))
        )
        or 0.0,
        "executor_total_memory_gb": _to_float(row.get("executor_total_memory_gb")) or 0.0,
        "spark_executor_cores": _to_float(row.get("spark_executor_cores")) or 0.0,
        "machine_memory_to_vcore_ratio": _to_float(row.get("machine_memory_to_vcore_ratio"))
        or 1.0,
        "duration_seconds": duration or 0.0,
        "vcore_price": _to_float(row.get("vcore_price")) or 0.0,
        "memory_price": _to_float(row.get("memory_price")) or 0.0,
    }


def run_cost_from_row(row: pd.Series, *, duration_seconds: Optional[float] = None) -> float:
    res = _resource_row_for_cost(row)
    dur = duration_seconds if duration_seconds is not None else res["duration_seconds"]
    return compute_run_cost_usd(
        vcore_seconds_allocated=res["vcore_seconds_allocated"],
        memory_gb_seconds_allocated=res["memory_gb_seconds_allocated"],
        executor_total_memory_gb=res["executor_total_memory_gb"],
        spark_executor_cores=res["spark_executor_cores"],
        machine_memory_to_vcore_ratio=res["machine_memory_to_vcore_ratio"],
        duration_seconds=dur,
        vcore_price=res["vcore_price"],
        memory_price=res["memory_price"],
    )


def _config_or_row_value(
    config_values: dict[str, Any],
    config_key: str,
    baseline_row: pd.Series,
    column: str,
) -> Optional[float]:
    if config_key in config_values:
        val = _to_float(config_values[config_key])
        if val is not None:
            return val
    return _to_float(baseline_row.get(column))


def effective_executor_count(
    config_values: dict[str, Any],
    baseline_row: pd.Series,
) -> Optional[float]:
    """Best-effort executor count from simulated config, falling back to the run row."""
    for config_key, column in EXECUTOR_COUNT_KEYS:
        val = _config_or_row_value(config_values, config_key, baseline_row, column)
        if val is not None and val > 0:
            return val
    return None


def effective_executor_memory_gb(
    config_values: dict[str, Any],
    baseline_row: pd.Series,
) -> Optional[float]:
    """Executor memory (GB) from spark.executor.memory or the run row."""
    val = _config_or_row_value(
        config_values,
        EXECUTOR_MEMORY_KEY,
        baseline_row,
        EXECUTOR_MEMORY_COLUMN,
    )
    if val is not None and val > 0:
        return val
    return _to_float(baseline_row.get("executor_total_memory_gb"))


def resource_scaling_factors(
    baseline_row: pd.Series,
    config_values: dict[str, Any],
) -> tuple[float, float, Optional[float], Optional[float]]:
    """
    Linear scaling for this experiment only.

    - vcore_seconds_allocated & spark_executor_cores ∝ executor count
    - memory_gb_seconds_allocated & executor_total_memory_gb ∝ spark.executor.memory
    """
    baseline_executors = effective_executor_count({}, baseline_row)
    simulated_executors = effective_executor_count(config_values, baseline_row)
    baseline_memory = effective_executor_memory_gb({}, baseline_row)
    simulated_memory = effective_executor_memory_gb(config_values, baseline_row)

    executor_scale = 1.0
    if (
        baseline_executors
        and simulated_executors
        and baseline_executors > 0
    ):
        executor_scale = simulated_executors / baseline_executors

    memory_scale = 1.0
    if baseline_memory and simulated_memory and baseline_memory > 0:
        memory_scale = simulated_memory / baseline_memory

    return executor_scale, memory_scale, simulated_executors, simulated_memory


def run_cost_from_scaled_config(
    baseline_row: pd.Series,
    config_values: dict[str, Any],
    *,
    duration_seconds: float,
) -> tuple[float, float, float]:
    """
    Apply experiment linear scaling to baseline run metrics, then the SQL cost formula.

    Returns (run_cost_usd, executor_scale, memory_scale).
    """
    base = _resource_row_for_cost(baseline_row)
    executor_scale, memory_scale, _, _ = resource_scaling_factors(
        baseline_row, config_values
    )
    cost = compute_run_cost_usd(
        vcore_seconds_allocated=base["vcore_seconds_allocated"] * executor_scale,
        memory_gb_seconds_allocated=base["memory_gb_seconds_allocated"] * memory_scale,
        executor_total_memory_gb=base["executor_total_memory_gb"] * memory_scale,
        spark_executor_cores=base["spark_executor_cores"] * executor_scale,
        machine_memory_to_vcore_ratio=base["machine_memory_to_vcore_ratio"],
        duration_seconds=duration_seconds,
        vcore_price=base["vcore_price"],
        memory_price=base["memory_price"],
    )
    return cost, executor_scale, memory_scale


def quantize_config_value(config_key: str, value: Any) -> Any:
    """
    Floor integer Spark/cluster params after interpolation.

    Task-run columns (spark_task_cpus, spark_executor_cores, etc.) are always
    whole numbers in the dataset; partial steps must not produce values like 1.8.
    """
    if config_key not in INTEGER_CONFIG_KEYS:
        return value
    num = _to_float(value)
    if num is None:
        return value
    out = math.floor(num + 1e-9)
    min_v = CONFIG_KEY_INTEGER_MIN.get(config_key)
    if min_v is not None:
        out = max(float(min_v), out)
    return int(out) if out == math.floor(out) else out


def _numeric_interval(
    start: Any,
    end: Any,
) -> Optional[tuple[float, float, float, float]]:
    """Return (lo, hi, start_num, end_num) when both endpoints are numeric."""
    start_num = _to_float(start)
    end_num = _to_float(end)
    if start_num is None or end_num is None:
        return None
    lo = min(start_num, end_num)
    hi = max(start_num, end_num)
    return lo, hi, start_num, end_num


def _clamp_numeric(value: float, lo: float, hi: float) -> float:
    return float(np.clip(value, lo, hi))


def _values_equal(config_key: str, left: Any, right: Any) -> bool:
    left_num = _to_float(left)
    right_num = _to_float(right)
    if left_num is not None and right_num is not None:
        return math.isclose(left_num, right_num, rel_tol=0.0, abs_tol=1e-9)
    return left == right


def interpolate_between(
    start: Any,
    end: Any,
    fraction: float,
    *,
    config_key: Optional[str] = None,
) -> Any:
    """
    Move ``fraction`` of the way from ``start`` → ``end``.

    Numeric values stay within [min(start, end), max(start, end)]; integer
    params are floored (e.g. 2 → 1 at 20% gives 1.8 → 1, the suggested value).
    """
    fraction = float(np.clip(fraction, 0.0, 1.0))
    interval = _numeric_interval(start, end)

    if interval is None:
        if fraction <= 0:
            return start
        if fraction >= 1:
            return end
        return end

    lo, hi, start_num, end_num = interval
    if math.isclose(start_num, end_num, rel_tol=0.0, abs_tol=1e-9):
        result_num = start_num
    else:
        result_num = _clamp_numeric(start_num + fraction * (end_num - start_num), lo, hi)

    if config_key is not None:
        result: Any = quantize_config_value(config_key, result_num)
        result_num = _to_float(result)
        if result_num is not None:
            result_num = _clamp_numeric(result_num, lo, hi)
            result = quantize_config_value(config_key, result_num)
        return result
    return result_num


def interpolate_value(
    current: Any,
    suggested: Any,
    fraction: float,
    *,
    config_key: Optional[str] = None,
) -> Any:
    """Move fraction of the way from current → suggested (fraction in [0, 1])."""
    return interpolate_between(
        current,
        suggested,
        fraction,
        config_key=config_key,
    )


def recommendation_reached(rec: ParamRecommendation, value: Any) -> bool:
    """True when ``value`` already equals this param's suggested recommendation."""
    return _values_equal(rec.config_key, value, rec.suggested_value)


def interpolate_config(
    recommendations: list[ParamRecommendation],
    fraction: float,
    *,
    from_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Interpolate all params between from_fraction and full suggested at `fraction`.

    At fraction=0.2: each param = current + 0.2 * (suggested - current), clamped to
    the [current, suggested] interval (integer params floored).
    """
    out: dict[str, Any] = {}
    for rec in recommendations:
        start = interpolate_between(
            rec.current_value,
            rec.suggested_value,
            from_fraction,
            config_key=rec.config_key,
        )
        out[rec.config_key] = interpolate_between(
            start,
            rec.suggested_value,
            fraction,
            config_key=rec.config_key,
        )
    return out


def collect_recommendations_for_task(
    df: pd.DataFrame,
    app_id: Any,
    task_name: str,
) -> list[ParamRecommendation]:
    """Merge recommendations from all insight rows for the latest task run."""
    task_rows = df[(df["app_id"] == app_id) & (df["task_name"] == task_name)].copy()
    if task_rows.empty:
        return []

    task_rows["start_time"] = pd.to_datetime(task_rows["start_time"], utc=True, errors="coerce")
    latest_task_id = task_rows.sort_values("start_time").iloc[-1]["task_id"]
    run_rows = task_rows[task_rows["task_id"] == latest_task_id]

    by_key: dict[str, ParamRecommendation] = {}
    for _, row in run_rows.iterrows():
        keys = parse_list_or_scalar(row.get("config_key")) or []
        currents = parse_list_or_scalar(row.get("current_value")) or []
        suggesteds = parse_list_or_scalar(row.get("suggested_value")) or []
        if len(keys) != len(currents) or len(keys) != len(suggesteds):
            continue
        for config_key, current_value, suggested_value in zip(keys, currents, suggesteds):
            config_key = str(config_key).strip()
            if not config_key:
                continue
            by_key[config_key] = ParamRecommendation(
                config_key=config_key,
                current_value=current_value,
                suggested_value=suggested_value,
            )
    return list(by_key.values())


def get_last_task_run_row(
    df: pd.DataFrame,
    app_id: Any,
    task_name: str,
) -> pd.Series:
    task_rows = df[(df["app_id"] == app_id) & (df["task_name"] == task_name)].copy()
    task_rows["start_time"] = pd.to_datetime(task_rows["start_time"], utc=True, errors="coerce")
    latest_task_id = task_rows.sort_values("start_time").iloc[-1]["task_id"]
    run_rows = task_rows[task_rows["task_id"] == latest_task_id]
    # Prefer row with metrics if duplicates exist
    if "vcore_seconds_used" in run_rows.columns:
        run_rows = run_rows.assign(
            _has_metrics=run_rows["vcore_seconds_used"].notna().astype(int)
        ).sort_values("_has_metrics", ascending=False)
    return run_rows.iloc[0]


def tasks_with_gradual_tuning_recommendations(df: pd.DataFrame) -> list[tuple[Any, str]]:
    """
    Tasks whose latest run has at least one Spark config recommendation.

    Rows with only instance-type / availability suggestions (no ``config_key``)
    are excluded — those cannot drive ``interpolate_config``.
    """
    required = {
        "app_id",
        "task_name",
        "task_id",
        "start_time",
        "config_key",
        "current_value",
        "suggested_value",
    }
    if not required.issubset(df.columns):
        return []

    work = df[
        df["suggested_value"].notna()
        & (df["suggested_value"].astype(str).str.strip() != "")
    ]
    if work.empty:
        return []

    tasks: list[tuple[Any, str]] = []
    for (app_id, task_name), _group in work.groupby(["app_id", "task_name"], sort=False):
        if collect_recommendations_for_task(df, app_id, task_name):
            tasks.append((app_id, task_name))
    return tasks


def sample_task_with_recommendations(
    df: pd.DataFrame,
    *,
    rng: Optional[random.Random] = None,
) -> tuple[Any, str]:
    rng = rng or random.Random()
    tasks = tasks_with_gradual_tuning_recommendations(df)
    if not tasks:
        raise ValueError(
            "No tasks with actionable config recommendations in dataset "
            "(need config_key + current_value + suggested_value on the latest run)."
        )
    return tasks[rng.randrange(len(tasks))]


def _pairs_from_config(config: dict[str, Any]) -> list[tuple[str, Any]]:
    return [(k, v) for k, v in config.items()]


def _no_recommendations_result(
    *,
    app_id: Any,
    task_name: str,
    baseline_row: pd.Series,
    sensitivity_factor: float,
    detail: str,
) -> ExperimentResult:
    """Return a skipped experiment when the latest run has no config_key recommendations."""
    baseline_duration = _to_float(
        baseline_row.get("task_duration_seconds", baseline_row.get("task__duration"))
    ) or 0.0
    skipped = SimulationResult(
        label="skipped (no actionable recommendations)",
        config_values={},
        estimated_duration_seconds=baseline_duration or None,
        run_cost_usd=None,
        match=None,
        match_found=False,
        notes=detail,
    )
    return ExperimentResult(
        app_id=app_id,
        task_name=task_name,
        task_id=baseline_row.get("task_id"),
        baseline_duration_seconds=baseline_duration,
        duration_limit_seconds=duration_limit_seconds(baseline_duration, sensitivity_factor)
        if baseline_duration > 0
        else 0.0,
        sensitivity_factor=sensitivity_factor,
        recommendations=[],
        baseline=skipped,
        first_step=skipped,
        final=skipped,
        decision="no_actionable_recommendations",
        decision_detail=detail,
    )


def is_failed_status(status: Any) -> bool:
    """True when a task run status indicates failure (e.g. FAILED)."""
    if status is None or (isinstance(status, float) and np.isnan(status)):
        return False
    return str(status).strip().upper() == "FAILED"


def is_failed_match(match: Optional[SimilarRunMatch | SimulationResult]) -> bool:
    """True when the closest similar historical run ended in failure."""
    if match is None:
        return False
    if isinstance(match, SimulationResult):
        match = match.match
    if match is None:
        return False
    return is_failed_status(match.similar_task_run.get("status"))


def build_experiment_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Candidate pool for gradual tuning: completed and failed runs.

    Failed runs are included so we can detect when a suggested config is most
    similar to a historical failure and revert to the prior safe suggestion.
    """
    if "status" not in df.columns:
        return dedupe_task_runs(df)
    statuses = df["status"].astype(str).str.strip().str.upper()
    mask = statuses.isin({"COMPLETED", "FAILED"})
    return dedupe_task_runs(df[mask])


def simulate_configuration(
    baseline_row: pd.Series,
    config_values: dict[str, Any],
    candidates: pd.DataFrame,
    matcher: TaskRunMatcher,
    *,
    label: str,
) -> SimulationResult:
    pairs = _pairs_from_config(config_values)
    exec_scale, mem_scale, sim_exec, sim_mem = resource_scaling_factors(
        baseline_row, config_values
    )

    match = estimate_duration_for_row(
        baseline_row, candidates, matcher=matcher, recommendation_pairs=pairs
    )

    baseline_duration = _to_float(
        baseline_row.get("task_duration_seconds", baseline_row.get("task__duration"))
    ) or 0.0

    if match is None:
        # Still compute cost at baseline duration with linear resource scaling.
        cost, exec_scale, mem_scale = run_cost_from_scaled_config(
            baseline_row,
            config_values,
            duration_seconds=baseline_duration,
        )
        return SimulationResult(
            label=label,
            config_values=config_values,
            estimated_duration_seconds=None,
            run_cost_usd=cost,
            match=None,
            match_found=False,
            executor_scale=exec_scale,
            memory_scale=mem_scale,
            notes=(
                "No duration match; cost used baseline duration with scaled resources "
                f"(executors×{exec_scale:.3f}, memory×{mem_scale:.3f})."
            ),
        )

    est_duration = match.estimated_duration_seconds
    cost, exec_scale, mem_scale = run_cost_from_scaled_config(
        baseline_row,
        config_values,
        duration_seconds=est_duration,
    )
    neighbor = match.similar_task_run
    neighbor_status = neighbor.get("status")
    status_note = (
        f" matched status={neighbor_status!r} (failed neighbor)."
        if is_failed_status(neighbor_status)
        else ""
    )
    return SimulationResult(
        label=label,
        config_values=config_values,
        estimated_duration_seconds=est_duration,
        run_cost_usd=cost,
        match=match,
        match_found=True,
        executor_scale=exec_scale,
        memory_scale=mem_scale,
        notes=(
            f"Duration from matched task_id={neighbor.get('task_id')};"
            f"{status_note} "
            f"cost from SQL formula on baseline resources scaled "
            f"(executors {sim_exec or '?'} → ×{exec_scale:.3f}, "
            f"memory {sim_mem or '?'} GB → ×{mem_scale:.3f})."
        ).strip(),
    )


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _print_recommendations(recs: list[ParamRecommendation]) -> None:
    print("\nParameter recommendations (latest run):")
    for rec in recs:
        if rec.is_numeric:
            delta = rec.suggested_num - rec.current_num
            print(
                f"  {rec.config_key}: {rec.current_value} → {rec.suggested_value} "
                f"(Δ {delta:+.4g})"
            )
        else:
            print(
                f"  {rec.config_key}: {rec.current_value!r} → {rec.suggested_value!r}"
            )


def _print_simulation(sim: SimulationResult, *, baseline_cost: float, limit: float) -> None:
    print(f"\n[{sim.label}]")
    for k, v in sim.config_values.items():
        print(f"  {k} = {v}")
    dur = sim.estimated_duration_seconds
    cost = sim.run_cost_usd or 0.0
    cost_delta = cost - baseline_cost
    if dur is not None:
        dur_pct = 100.0 * dur / limit if limit > 0 else float("nan")
        print(
            f"  estimated duration: {dur:,.1f}s (limit {limit:,.1f}s, {dur_pct:.1f}% of limit)"
        )
    else:
        print("  estimated duration: n/a (no similar run)")
    print(
        f"  resource scaling: executors ×{sim.executor_scale:.3f}, "
        f"memory ×{sim.memory_scale:.3f}"
    )
    print(f"  run_cost_usd: ${cost:,.4f} (Δ vs baseline ${cost_delta:+,.4f})")
    print(f"  {sim.notes}")
    if sim.match is not None and is_failed_match(sim):
        print(
            f"  WARNING: closest neighbor is a failed run "
            f"(task_id={sim.match.similar_task_run.get('task_id')})."
        )


def run_gradual_tuning_experiment(
    df: pd.DataFrame,
    *,
    app_id: Optional[Any] = None,
    task_name: Optional[str] = None,
    sensitivity_factor: float = 20.0,
    step_fraction: float = 0.20,
    fine_tune_fraction: float = 0.10,
    candidates: Optional[pd.DataFrame] = None,
    matcher: Optional[TaskRunMatcher] = None,
    rng: Optional[random.Random] = None,
    verbose: bool = True,
) -> ExperimentResult:
    """
    Run one gradual-tuning experiment for a single task.

    Flow
    ----
    1. Baseline = current config, actual duration, run_cost from current row.
    2. First step = move step_fraction (20%) toward suggested for all params.
    3. Decide final config (all params together):
       - cost increased → nudge fine_tune_fraction back toward original
       - cost decreased & duration OK → nudge fine_tune_fraction toward suggested
       - cost decreased & duration over limit → revert to baseline (previous step)
    4. If the closest similar run is FAILED (completed + failed in candidate pool),
       stop and revert to the last suggestion before that simulation step.
    """
    user_picked_task = app_id is not None and task_name is not None
    if not user_picked_task:
        app_id, task_name = sample_task_with_recommendations(df, rng=rng)

    baseline_row = get_last_task_run_row(df, app_id, task_name)
    recommendations = collect_recommendations_for_task(df, app_id, task_name)
    if not recommendations:
        detail = (
            f"Latest run for app_id={app_id} task_name={task_name!r} has no actionable "
            "config recommendations (insight rows may be instance-type / availability only, "
            "or config_key/current_value/suggested_value lists do not align). "
            "Gradual tuning applies to spark config keys only."
        )
        if verbose:
            _print_header("Gradual parameter-tuning experiment")
            print(f"Skipped: {detail}")
        return _no_recommendations_result(
            app_id=app_id,
            task_name=task_name,
            baseline_row=baseline_row,
            sensitivity_factor=sensitivity_factor,
            detail=detail,
        )

    baseline_duration = _to_float(
        baseline_row.get("task_duration_seconds", baseline_row.get("task__duration"))
    )
    if not baseline_duration or baseline_duration <= 0:
        raise ValueError("Latest run has no valid task_duration_seconds")

    limit = duration_limit_seconds(baseline_duration, sensitivity_factor)
    baseline_cost = run_cost_from_row(baseline_row, duration_seconds=baseline_duration)

    if candidates is None:
        candidates = build_experiment_candidates(df)
    if matcher is None:
        matcher = TaskRunMatcher(candidates, completed_only=False)

    baseline_config = {
        r.config_key: quantize_config_value(r.config_key, r.current_value)
        for r in recommendations
    }
    last_suggestion = dict(baseline_config)
    baseline_sim = SimulationResult(
        label="baseline (actual run)",
        config_values=baseline_config,
        estimated_duration_seconds=baseline_duration,
        run_cost_usd=baseline_cost,
        match=None,
        match_found=True,
        notes="Observed run; cost from SQL formula on actual metrics.",
    )

    first_config = interpolate_config(recommendations, step_fraction)
    first_sim = simulate_configuration(
        baseline_row, first_config, candidates, matcher, label=f"step {step_fraction:.0%} toward suggested"
    )

    decision = "unknown"
    decision_detail = ""
    final_config = dict(first_config)

    if is_failed_match(first_sim.match):
        neighbor = first_sim.match.similar_task_run if first_sim.match else None
        failed_id = neighbor.get("task_id") if neighbor is not None else "?"
        decision = "similar_to_failed_run"
        final_config = dict(last_suggestion)
        decision_detail = (
            f"Closest match for the {step_fraction:.0%} step is a failed run "
            f"(task_id={failed_id}); reverted to the prior suggestion (baseline config)."
        )
    elif not first_sim.match_found:
        decision = "no_match"
        if first_sim.run_cost_usd is not None and first_sim.run_cost_usd > baseline_cost:
            decision_detail = (
                "No duration match; scaled cost rose vs baseline — keeping baseline config."
            )
        else:
            decision_detail = (
                "No similar run for duration estimate; "
                "compare scaled cost only (see prints above)."
            )
        final_config = dict(baseline_config)
    else:
        last_suggestion = dict(first_config)
        if first_sim.run_cost_usd is not None and first_sim.run_cost_usd > baseline_cost:
            decision = "cost_increased"
            for rec in recommendations:
                at_step = first_config[rec.config_key]
                final_config[rec.config_key] = interpolate_between(
                    at_step,
                    rec.current_value,
                    fine_tune_fraction,
                    config_key=rec.config_key,
                )
            decision_detail = (
                f"Cost rose (${first_sim.run_cost_usd:,.4f} > ${baseline_cost:,.4f}); "
                f"nudged {fine_tune_fraction:.0%} back toward original from the {step_fraction:.0%} step."
            )
        elif (
            first_sim.estimated_duration_seconds is not None
            and first_sim.estimated_duration_seconds > limit
        ):
            decision = "duration_over_limit"
            final_config = dict(baseline_config)
            decision_detail = (
                f"Duration {first_sim.estimated_duration_seconds:,.1f}s exceeds limit {limit:,.1f}s; "
                "reverted to baseline (previous step before crossing limit)."
            )
        else:
            decision = "cost_decreased_duration_ok"
            already_at_target: list[str] = []
            for rec in recommendations:
                at_step = first_config[rec.config_key]
                if recommendation_reached(rec, at_step):
                    final_config[rec.config_key] = at_step
                    already_at_target.append(rec.config_key)
                else:
                    final_config[rec.config_key] = interpolate_between(
                        at_step,
                        rec.suggested_value,
                        fine_tune_fraction,
                        config_key=rec.config_key,
                    )
            decision_detail = (
                f"Cost decreased (${first_sim.run_cost_usd:,.4f} ≤ ${baseline_cost:,.4f}) "
                f"and duration within limit."
            )
            if already_at_target:
                decision_detail += (
                    f" No further step for {', '.join(already_at_target)} — already at suggested "
                    f"(cannot go past recommendation)."
                )
            elif fine_tune_fraction > 0:
                decision_detail += (
                    f" Nudged {fine_tune_fraction:.0%} further toward suggested for remaining params."
                )

    prior_suggestion = dict(last_suggestion)
    skip_final_sim = all(
        _values_equal(
            rec.config_key,
            final_config[rec.config_key],
            first_config[rec.config_key],
        )
        for rec in recommendations
    )
    if skip_final_sim:
        final_sim = SimulationResult(
            label="final recommendation (same as first step)",
            config_values=dict(final_config),
            estimated_duration_seconds=first_sim.estimated_duration_seconds,
            run_cost_usd=first_sim.run_cost_usd,
            match=first_sim.match,
            match_found=first_sim.match_found,
            executor_scale=first_sim.executor_scale,
            memory_scale=first_sim.memory_scale,
            notes=(
                (first_sim.notes or "")
                + " Final config unchanged after first step; no extra simulation."
            ).strip(),
        )
    else:
        final_sim = simulate_configuration(
            baseline_row,
            final_config,
            candidates,
            matcher,
            label="final recommendation",
        )

    if decision != "similar_to_failed_run" and is_failed_match(final_sim.match):
        neighbor = final_sim.match.similar_task_run if final_sim.match else None
        failed_id = neighbor.get("task_id") if neighbor is not None else "?"
        decision = "similar_to_failed_run"
        final_config = dict(prior_suggestion)
        decision_detail = (
            "Closest match for the final recommendation is a failed run "
            f"(task_id={failed_id}); reverted to the prior suggestion "
            "(config from before this simulation step)."
        )
        final_sim = simulate_configuration(
            baseline_row,
            final_config,
            candidates,
            matcher,
            label="final (reverted: failed neighbor)",
        )

    result = ExperimentResult(
        app_id=app_id,
        task_name=task_name,
        task_id=baseline_row.get("task_id"),
        baseline_duration_seconds=baseline_duration,
        duration_limit_seconds=limit,
        sensitivity_factor=sensitivity_factor,
        recommendations=recommendations,
        baseline=baseline_sim,
        first_step=first_sim,
        final=final_sim,
        decision=decision,
        decision_detail=decision_detail,
    )

    if verbose:
        _print_header("Gradual parameter-tuning experiment")
        print(f"Task: app_id={app_id}  task_name={task_name!r}  task_id={result.task_id}")
        print(
            f"Baseline duration: {baseline_duration:,.1f}s  |  "
            f"sensitivity_factor={sensitivity_factor}%  |  "
            f"duration limit: {limit:,.1f}s"
        )
        print(
            "\nCost model (experiment): SQL run_cost_usd on baseline allocated resources, "
            "with vcore_seconds_allocated & executor_cores scaled linearly by executor count, "
            "memory_gb_seconds_allocated & executor_memory_gb scaled linearly by spark.executor.memory. "
            "Duration from similar-run matching."
        )
        _print_recommendations(recommendations)
        _print_simulation(baseline_sim, baseline_cost=baseline_cost, limit=limit)
        _print_simulation(first_sim, baseline_cost=baseline_cost, limit=limit)
        print(f"\nDecision: {decision}")
        print(f"  {decision_detail}")
        _print_simulation(final_sim, baseline_cost=baseline_cost, limit=limit)

    return result
