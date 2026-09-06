"""Run the AutoRecommendation agent on a REST API dump instead of Athena.

Usage:
    ./run.sh run_from_dump.py data/dumps/dump_5453     # a directory made by tools/dump-rest-api.sh
    ./run.sh run_from_dump.py data/dumps/dump_5453 --json

Bridges the CSVs produced by dump-rest-api.sh into the same inputs main.py
injects before invoking the inference graph:

- enrichment row    <- task_metrics.csv + task_tsm.csv + tf_<id>.csv
  (REST metric_type values mapped to task_enrichments column names;
  time metrics stay in seconds — that is the enrichment schema's unit)
- sandbox tables    <- task_events/task_metrics/task_tfs/task_tsm/
  task_params.csv, each already close to one raw DB table (events, metrics,
  tfs, time_series_metrics, task_params respectively); task_tsm.csv's
  embedded metric_reports are reshaped into real time_series_metrics rows
  via app/metrics/timeseries_metric_types.csv's line_name -> metric_type
  reverse lookup (the REST endpoint's response carries line_name, not the
  raw metric_type); task.csv/tf_<id>.csv approximate a single-row
  task_enrichments AND `tasks` table (kb/analysis/store.py's
  Store._baseline_id() needs a `tasks` table unconditionally). Fed to
  agent/local_sandbox.py's build_run_sandbox() so
  run_deterministic_review/estimate_change_saving/get_advanced_config_catalog
  read real (if REST-shaped) data instead of {}. Single-PIT only — this
  run's rows, not multi-run history. Two known gaps, omitted rather than
  fabricated: tf_inputs/tfs_query_vars are NOT derivable from this dump
  shape (tf_<id>_lineage.csv is a single {"execution": {...}} blob, not a
  per-input row list); and the query_plan detector's physical-plan tier
  reads `events` rows with sub_category='physical_plan' — task_events.csv
  only carries sub_category='stage' events, and tf_<id>_physical_plan.csv's
  rows are pre-flattened by the REST endpoint (not the original nested
  {plan:{name,stages,metrics,children}} tree the event payload holds), so
  there's no safe re-nesting back into an `events` row — query_plan's T3
  tier degrades cleanly without it (kb/analysis/query_plan.py).
- sql insights      <- the `insights` JSON column embedded in task_tsm.csv
  (the same rows the production insights table holds for this task)

Known gaps of a REST dump vs Athena, surfaced at startup:
- no historical-runs endpoint in the REST API (dump-rest-api.sh only hits
  single-task_id-scoped endpoints) -> historical runs for the same
  (task_name, app_id) are instead fetched best-effort straight from Athena
  (main.py's existing _fetch_athena_historical_runs, same AWS SSO
  credentials as --athena mode); when boto3/AWS access isn't available this
  falls back to no history, same as before -> no trend analysis
- when task_params.csv is present it feeds the sandbox's task_params table
  (current_value resolution), the AQE effective state
  (task__aqe_enabled__param) and shuffle partitions; on older dumps where
  /api/tasks/<id>/params 500'd, shuffle partitions falls back to the
  insight payload (corroborated by stage numTasks) and AQE state stays
  unknown.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from datetime import datetime
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from bootstrap_worktree import resolved_backend_path

from main import (  # noqa: E402 — main wires sys.path to definity-app's backend
    HISTORICAL_RUNS_FETCH_LIMIT,
    SANDBOX_TEXT_MAX_CHARS,
    _MEMORY_PRICE,
    _VCORE_PRICE,
    _build_assembled_output,
    _cast_athena,
    _fetch_athena_historical_runs,
    _print_assembled_output,
    _print_plan,
    _persist_result,
    _results_dir,
)

from agent.cost_utils import compute_cost_profile  # noqa: E402
from app.brain.insights.recommendations.agent import (  # noqa: E402
    PAYLOAD_CONFIG_KEY,
    PAYLOAD_RECOMMENDATIONS,
)

MS_PER_SECOND = 1000

# REST metric_type (task-level) -> task_enrichments column. Values whose
# enrichment unit is milliseconds are flagged for seconds->ms conversion.
_METRIC_TO_ENRICHMENT: dict[str, tuple[str, bool]] = {
    "executor_memory": ("executor__memory__allocated", False),
    "executor_heap_memory": ("executor__memory_heap__allocated", False),
    "executor_max_heap_used_memory": ("executor__memory_heap__max_used", False),
    "executor_off_heap_memory": ("executor__memory_off_heap__allocated", False),
    "executor_max_off_heap_used_memory": ("executor__memory_off_heap__max_used", False),
    "driver_heap_memory": ("driver__heap_memory__allocated", False),
    "driver_max_heap_used_memory": ("driver__heap_memory__max_used", False),
    "executors_jvm_gc_time": ("executors__jvm_gc_time", False),
    "executors_used_vcore_time": ("executors__vcore_time__used", False),
    "executors_vcore_time": ("executors__vcore_time__allocated", False),
    "executors_used_cpu_time": ("executors__cpu_time__used", False),
    "used_vcore_time": ("task__vcore_time__used", False),
    "vcore_time": ("task__vcore_time__allocated", False),
    "task_idle_time": ("task__idle_time", True),
    "duration": ("task__duration", True),
    "disk_bytes_spilled": ("task__disk_bytes_spilled", False),
    "memory_time": ("task__memory_time__allocated", False),
    "executors_used_vcore_time_of_retried_tasks": (
        "executors__used_vcore_time_of_retried_tasks",
        False,
    ),
    "executor_cores": ("executor__cores", False),
    # Production maps skew_score straight onto task__skew_time
    # (task_run_enrichment.sql); converted like the other REST time metrics
    # so skew_ratio = skew_time/duration stays unit-consistent.
    "skew_score": ("task__skew_time", True),
}

_IO_METRIC_TYPES = (
    "input_bytes_read",
    "output_bytes_written",
    "shuffle_bytes_read",
    "shuffle_bytes_written",
)


def _read_rows(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [
            {k: _cast_athena(v, column=k) for k, v in raw.items()}
            for raw in csv.DictReader(f)
        ]


@lru_cache()
def _line_name_to_metric_type() -> dict[tuple[str, str], str]:
    """(grouping_metric, line_name) -> time_series_metric_type, from the
    branch's own app/metrics/timeseries_metric_types.csv — the authoritative
    source the real time_series_metrics_metadata_v view is built from
    (app/dal/sql/time_series_metrics.sql joins on it to attach line_name to
    each real time_series_metrics row). The REST time-series-metrics endpoint
    embeds line_name but not metric_type in its response (app/dal/sql/
    time_series_metrics.sql, TimeSeriesMetricReport schema), so recovering
    metric_type for task_tsm.csv's metric_reports needs this reverse lookup
    rather than a guess. Keyed on the (grouping_metric, line_name) pair, not
    line_name alone — the same line_name string is reused across groups with
    different metric_types. Comment/blank lines in the CSV are skipped;
    pattern rows (is_pattern=true, streaming-only in this file) have no
    line_name and are irrelevant here."""
    path = resolved_backend_path() / "app" / "metrics" / "timeseries_metric_types.csv"
    with path.open(newline="", encoding="utf-8") as f:
        lines = [line for line in f if line.strip() and not line.lstrip().startswith("#")]
    mapping: dict[tuple[str, str], str] = {}
    for row in csv.DictReader(lines):
        if row["is_pattern"].strip().lower() == "true" or not row["line_name"]:
            continue
        mapping[(row["grouping_metric"], row["line_name"])] = row["time_series_metric_type"]
    return mapping


def _time_series_metrics_rows(dump: Path, task_id: object) -> list[dict[str, object]]:
    """task_tsm.csv's embedded metric_reports -> real time_series_metrics rows
    (task_id, metric_type, kind, values, bucket_size_seconds, start_time_ms) —
    each report already carries per-bucket values/bucket_size_seconds/
    start_time_ms in the exact shape the real table's `values` column holds
    (a JSON array string), just missing metric_type, which
    _line_name_to_metric_type() recovers from the branch's own CSV. Reports
    whose (grouping_metric, line_name) isn't in that CSV (e.g. frontend-only
    virtual/synthetic lines with no backing metric_type) are skipped."""
    lookup = _line_name_to_metric_type()
    rows: list[dict[str, object]] = []
    for tsm_row in _read_rows(dump / "task_tsm.csv"):
        grouping_metric = str(tsm_row.get("grouping_metric") or "")
        try:
            reports = json.loads(str(tsm_row.get("metric_reports") or "[]"))
        except ValueError:
            continue
        for report in reports:
            metric_type = lookup.get((grouping_metric, report.get("line_name")))
            if metric_type is None:
                continue
            rows.append({
                "task_id": task_id,
                "metric_type": metric_type,
                "kind": report.get("kind"),
                "values": json.dumps(report.get("values")),
                "bucket_size_seconds": report.get("bucket_size_seconds"),
                "start_time_ms": report.get("start_time_ms"),
            })
    return rows


def _read_kv(path: Path) -> dict[str, object]:
    return {str(r["key"]): r["value"] for r in _read_rows(path)}


def _task_metrics(dump: Path) -> dict[str, float]:
    """Task-level metric_type -> metric_value from task_metrics.csv."""
    return {
        str(r["metric_type"]): float(r["metric_value"])
        for r in _read_rows(dump / "task_metrics.csv")
        if r.get("asset_type") == "task" and isinstance(r.get("metric_value"), (int, float))
    }


def _tf_detail(dump: Path) -> dict[str, object]:
    """The key/value tf_<id>.csv file (not the tf_<id>_events/stages siblings)."""
    for path in sorted(dump.glob("tf_*.csv")):
        if path.stem.removeprefix("tf_").isdigit():
            return _read_kv(path)
    return {}


def _task_detail(dump: Path) -> dict[str, object]:
    """task.csv merged over the first tf_<id>.csv (task.csv wins).

    Older dumps where /api/tasks/<id> 500'd only have the tf file; tasks
    with no transformations only have task.csv — either alone suffices.
    """
    merged = _tf_detail(dump)
    merged.update(
        {k: v for k, v in _read_kv(dump / "task.csv").items() if v is not None}
    )
    return merged


def _dump_insights(dump: Path) -> list[dict[str, object]]:
    """Production-shaped insight rows embedded in task_tsm.csv's `insights` column."""
    by_id: dict[object, dict[str, object]] = {}
    for row in _read_rows(dump / "task_tsm.csv"):
        for insight in json.loads(str(row.get("insights") or "[]")):
            insight.setdefault("lifecycle_status", "active")
            by_id[insight.get("insight_id", insight.get("type"))] = insight
    return list(by_id.values())


def _shuffle_partitions_param(insights: list[dict[str, object]]) -> str | None:
    """Fallback when task_params.csv is absent (older dumps where /params
    500'd): recover spark.sql.shuffle.partitions from the insight payload
    (its value is corroborated by stage numTasks)."""
    for insight in insights:
        payload = insight.get("insights_payload")
        if isinstance(payload, dict) and payload.get("shuffle_partitions"):
            return str(payload["shuffle_partitions"])
    return None


def read_spark_parameters(dump: Path) -> list[dict[str, object]]:
    """task_params.csv -> [{key, value}, ...], the run-context
    `spark_parameters` section shape resolve_current_config_value expects.

    Each CSV `value` cell is a JSON blob ({"value": "true", "black_listed":
    false, ...}); only the inner value is kept. Rows that fail to parse keep
    the raw cell — a raw string is still a usable current value.
    """
    params: list[dict[str, object]] = []
    for row in _read_rows(dump / "task_params.csv"):
        value = row.get("value")
        if isinstance(value, str):
            try:
                blob = json.loads(value)
                if isinstance(blob, dict) and "value" in blob:
                    value = blob["value"]
            except ValueError:
                pass
        params.append({"key": row.get("key"), "value": value})
    return params


def _param_value(params: list[dict[str, object]], key: str) -> object:
    return next((p["value"] for p in params if p.get("key") == key), None)


def _annual_runs_estimate(insights: list[dict[str, object]]) -> float | None:
    """Runs/year from the observed frequency any SQL insight payload carries
    (cnt_task_runs over time_span_days). Max across payloads: the widest
    observation window is the most representative."""
    best: float | None = None
    for insight in insights:
        payload = insight.get("insights_payload")
        if not isinstance(payload, dict):
            continue
        try:
            cnt = float(payload.get("cnt_task_runs"))  # type: ignore[arg-type]
            days = float(payload.get("time_span_days"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if cnt > 0 and days > 0:
            rate = cnt / days * 365.25
            best = rate if best is None else max(best, rate)
    return best


def _annual_runs_from_history(
    historical_runs: list[dict[str, object]], current_app_pit: object
) -> float | None:
    """Runs/year from real Athena run timestamps (app_pit) — preferred over
    the SQL-insight heuristic when historical runs were actually fetched,
    since it's the real run cadence rather than a payload happening to carry
    cnt_task_runs/time_span_days."""
    timestamps = [run.get("app_pit") for run in historical_runs]
    timestamps.append(current_app_pit)
    parsed: list[datetime] = []
    for ts in timestamps:
        if not ts:
            continue
        try:
            parsed.append(datetime.fromisoformat(str(ts)))
        except ValueError:
            continue
    if len(parsed) < 2:
        return None
    span_days = (max(parsed) - min(parsed)).total_seconds() / 86400
    if span_days <= 0:
        return None
    return len(parsed) / span_days * 365.25


def fill_missing_annual_costs(
    plan,
    row: dict[str, object],
    insights: list[dict[str, object]],
    historical_runs: list[dict[str, object]] | None = None,
):
    """Price recommendations a REST dump leaves unpriced.

    assemble_output derives usd_cost_annual from a run_sandbox this harness
    never builds — so every agent recommendation lands with usd_cost_annual=
    None even when the planner set estimated_saving_fraction. Recover the
    baseline from this run's cost (the same CALCULATE_USD_COST port main.py
    persists as cost_profile) x the run frequency — preferring the real
    cadence from Athena historical runs when available, else the frequency
    observed in the SQL insight payloads — then price each unpriced
    recommendation as baseline x estimated_saving_fraction. Also back-fills
    plan.plan_inputs.active_insights (the assembled_output/plan_inputs
    source of truth for usd_cost_annual, keyed by config_key) with the same
    prices — otherwise assembled_output keeps showing null even after
    plan.recommendations is fixed. Agent-discovered active_insights now group
    companion recommendations under payload[PAYLOAD_RECOMMENDATIONS] (one
    ActiveInsight per shared insight_ref — see definity-app's
    assemble_output.py::_agent_active_insights), so the group's price is the
    max of its member config_keys' priced usd_cost_annual, mirroring that
    same file's _max_usd_cost_annual (max, not sum, to avoid double-counting
    one insight's several companion changes).
    """
    if plan is None:
        return None
    annual_runs = _annual_runs_from_history(
        historical_runs or [], row.get("app_pit")
    ) or _annual_runs_estimate(insights)
    cost_per_run = compute_cost_profile(
        row, vcore_price=_VCORE_PRICE, memory_price=_MEMORY_PRICE
    ).cost_per_run_usd
    if not annual_runs or not cost_per_run:
        return plan
    annual_cost = annual_runs * cost_per_run

    def _priced_value(rec):
        if rec.usd_cost_annual is None and rec.estimated_saving_fraction is not None:
            return round(annual_cost * rec.estimated_saving_fraction, 2)
        return rec.usd_cost_annual

    def _price(recs):
        return [
            rec.model_copy(update={"usd_cost_annual": _priced_value(rec)})
            for rec in recs
        ]

    priced_recommendations = _price(plan.recommendations)
    priced_blocked = _price(plan.blocked_recommendations)

    priced_by_key = {
        rec.config_key: rec.usd_cost_annual
        for rec in (*priced_recommendations, *priced_blocked)
    }

    def _price_active_insight(insight):
        if insight.usd_cost_annual is not None:
            return insight
        payload = insight.payload or {}
        entries = payload.get(PAYLOAD_RECOMMENDATIONS)
        if entries is None:
            # Backward-compat: tolerate the old flat shape (payload IS the
            # single recommendation) for dumps saved before the insight_ref
            # grouping landed — same fallback AgentDiscoveredRule.extract()
            # applies on the definity-app side.
            entries = [payload]
        priced = [
            priced_by_key[config_key]
            for entry in entries
            if (config_key := entry.get(PAYLOAD_CONFIG_KEY)) in priced_by_key
        ]
        if not priced:
            return insight
        return insight.model_copy(update={"usd_cost_annual": max(priced)})

    plan_inputs = plan.plan_inputs
    if plan_inputs is not None:
        plan_inputs = plan_inputs.model_copy(
            update={
                "active_insights": [
                    _price_active_insight(insight)
                    for insight in plan_inputs.active_insights
                ]
            }
        )

    return plan.model_copy(
        update={
            "recommendations": priced_recommendations,
            "blocked_recommendations": priced_blocked,
            "plan_inputs": plan_inputs,
        }
    )


def build_enrichment_row(
    dump: Path,
    insights: list[dict[str, object]],
    spark_params: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    metrics = _task_metrics(dump)
    task = _task_detail(dump)
    spark_params = spark_params if spark_params is not None else read_spark_parameters(dump)

    row: dict[str, object] = {
        "task_id": task.get("task_id"),
        "task_name": task.get("task_name"),
        "app_id": task.get("app_id"),
        "app_pit": task.get("app_pit"),
        "env_id": task.get("env_id", task.get("env")),
        "status": task.get("status"),
    }
    for metric_type, (column, seconds_to_ms) in _METRIC_TO_ENRICHMENT.items():
        value = metrics.get(metric_type)
        # task_enrichments time columns are SECONDS (task_run_enrichment.sql:
        # EXTRACT(EPOCH ...)); the REST metrics are seconds too, so no conversion.
        # The seconds_to_ms flag is kept only to mark which columns are durations.
        row[column] = value

    io_values = [metrics[m] for m in _IO_METRIC_TYPES if m in metrics]
    row["task__total_io_bytes"] = sum(io_values) if io_values else None

    for tsm_row in _read_rows(dump / "task_tsm.csv"):
        if tsm_row.get("readable_name") == "Executors Memory Usage":
            row["executors__memory_time__used"] = tsm_row.get("used")

    # Submitted spark conf is the authoritative source; the insight payload
    # is only the fallback for older dumps without task_params.csv.
    shuffle_from_params = _param_value(spark_params, "spark.sql.shuffle.partitions")
    row["task__shuffle_partitions__param"] = (
        str(shuffle_from_params)
        if shuffle_from_params is not None
        else _shuffle_partitions_param(insights)
    )
    # AQE effective state — a fact from the submitted conf, so the agent
    # never has to guess whether a static partition count governs runtime.
    row["task__aqe_enabled__param"] = _param_value(
        spark_params, "spark.sql.adaptive.enabled"
    )
    return row


# Columns holding structured JSON that downstream code json.loads() — a
# truncated JSON blob fails to parse and the row is silently dropped. On task
# 18609 this cut every stage row over 2,000 chars (the three big
# shuffle-consuming stages) and left only the 1-partition final aggregate, so
# partition_distribution reported "AQE collapsed to 1 partition" and the agent
# recommended a coalesce fix for a stage that ran 20,000 tasks. Production's
# sandbox dump (agent_context.sql) is SELECT * with no truncation.
_STRUCTURED_COLUMNS = frozenset({"payload"})


def _truncate(value: object) -> object:
    text = value if isinstance(value, str) else None
    if text is not None and len(text) > SANDBOX_TEXT_MAX_CHARS:
        return text[:SANDBOX_TEXT_MAX_CHARS] + "…"
    return value


def _table_rows(path: Path, columns: dict[str, str]) -> list[dict[str, object]]:
    """Project dump rows onto raw-DB-table column names, truncating free text
    (never structured JSON columns — see _STRUCTURED_COLUMNS)."""
    return [
        {
            out_col: (
                row.get(src_col)
                if out_col in _STRUCTURED_COLUMNS
                else _truncate(row.get(src_col))
            )
            for src_col, out_col in columns.items()
        }
        for row in _read_rows(path)
    ]


def build_sandbox_tables(
    dump: Path,
    enrichment: dict[str, object],
    spark_params: list[dict[str, object]] | None = None,
) -> dict[str, list[dict[str, object]]]:
    """Raw-ish per-table rows for this single run, from the REST dump's CSVs —
    the local-sandbox equivalent of production's S3 CSV dump. Many of these
    CSVs are already close to one real DB table each; task.csv/tf_<id>.csv
    approximate a single-row task_enrichments table AND a single-row `tasks`
    table (there's no separate `tasks` endpoint in this dump shape, so the
    same merged task detail feeds both — kb/analysis/store.py's
    Store._baseline_id() calls self.table("tasks") unconditionally, unlike
    runs()'s graceful degradation when it's absent, so `tasks` must be
    present for run_deterministic_review to not crash on this dump source).
    tf_inputs/tfs_query_vars are NOT included: tf_<id>_lineage.csv is a
    single {"execution": {...}} blob, not a per-input row list, so there's no
    clean per-row analog in this dump — omitted rather than fabricated.
    Single-PIT only (this run's rows, no multi-run history).
    """
    task = _task_detail(dump)
    spark_params = spark_params if spark_params is not None else read_spark_parameters(dump)
    task_row = {
        **enrichment,
        "status": task.get("status"),
        "start_time": task.get("start_time"),
        "end_time": task.get("end_time"),
        "app_name": task.get("app_name"),
        "env_name": task.get("env_name", task.get("env")),
        "is_retry": task.get("is_retry", False),
        "parent_task_id": task.get("parent_task_id"),
        "params_baseline_id": task.get("params_baseline_id"),
    }
    task_id = enrichment.get("task_id")
    tables: dict[str, list[dict[str, object]]] = {
        "task_enrichments": [task_row],
        "tasks": [task_row],
        "events": _table_rows(
            dump / "task_events.csv",
            {c: c for c in (
                "task_id", "event_id", "category", "sub_category", "name",
                "description", "start_time_ms", "end_time_ms", "payload",
            )},
        ),
        # task_metrics.csv has no task_id column (single-task-scoped REST
        # endpoint) — store.py's PER_RUN_TABLES filtering and detector joins
        # need one, so it's stamped on every row below.
        "metrics": [
            {**row, "task_id": task_id}
            for row in _table_rows(
                dump / "task_metrics.csv",
                {"metric_type": "metric_type", "asset_type": "asset_type",
                 "asset_name": "asset_value", "metric_value": "metric_value"},
            )
        ],
        "tfs": _table_rows(
            dump / "task_tfs.csv",
            {"tf_id": "tf_id", "task_id": "task_id", "tf_type": "tf_type",
             "output_name": "output_name",
             "status": "status", "error": "error", "description": "description",
             "start_time": "start_time", "end_time": "end_time",
             "duration": "duration_seconds", "query_str": "query",
             "query_vars": "query_vars", "labels": "labels"},
        ),
        "time_series_metrics": _time_series_metrics_rows(dump, task_id),
    }
    if spark_params:
        # task_params.csv has no task_id column either (same single-task
        # REST scoping as task_metrics.csv) — _params_of()/_baseline_id()
        # look rows up by exact task_id match, so it's stamped here too.
        # The table name and {key, value, task_id} row shape are what
        # resolve_current_config_value reads for SUBMITTED_SPARK_CONF —
        # current_value on every recommendation resolves from here.
        tables["task_params"] = [{**p, "task_id": task_id} for p in spark_params]
    return {name: rows for name, rows in tables.items() if rows}


def _fetch_historical_runs_best_effort(
    row: dict[str, object],
) -> list[dict[str, object]]:
    """Real historical runs for (task_name, app_id) from Athena, when reachable.

    dump-rest-api.sh only hits single-task_id-scoped REST endpoints — the
    historical_enrichments query (agent_context.sql) that production uses for
    this is Postgres-internal only, never exposed over REST. main.py already
    queries Athena's mirror of task_enrichments for this exact job, so reuse
    that path via the AWS SSO credentials --athena mode already relies on;
    skip quietly (falling back to no trend analysis, as before) when
    boto3/AWS access isn't available.
    """
    task_name = row.get("task_name")
    app_id = row.get("app_id")
    task_id = row.get("task_id")
    if not task_name or app_id is None or task_id is None:
        return []
    try:
        import boto3  # noqa: F401
    except ImportError:
        print(
            "  Historical runs: boto3 not installed — skipping "
            "(pip install boto3 to enable trend analysis via Athena).",
            file=sys.stderr,
        )
        return []
    try:
        history = _fetch_athena_historical_runs(
            str(task_name), app_id, int(task_id), HISTORICAL_RUNS_FETCH_LIMIT
        )
    except Exception as exc:
        print(
            f"  Historical runs from Athena unavailable "
            f"({exc.__class__.__name__}: {exc}) — continuing without trend analysis.",
            file=sys.stderr,
        )
        return []
    print(
        f"  Historical runs from Athena: {len(history)} run(s) for "
        f"({task_name}, app_id={app_id})",
        file=sys.stderr,
    )
    return history


async def _run(dump: Path, as_json: bool, save_results: bool) -> None:
    from agent.inference_graph import run_analysis
    from agent.nodes.fetch_context import set_row_override
    from agent.nodes.sql_stream import set_insights_override

    insights = _dump_insights(dump)
    spark_params = read_spark_parameters(dump)
    row = build_enrichment_row(dump, insights, spark_params)
    sandbox_tables = build_sandbox_tables(dump, row, spark_params)

    print(f"  Enrichment columns recovered: "
          f"{sum(v is not None for v in row.values())}/{len(row)}", file=sys.stderr)
    print(f"  Sandbox tables: {', '.join(sandbox_tables)}", file=sys.stderr)
    print(f"  Spark params from dump: {len(spark_params)} "
          f"(aqe_enabled={row.get('task__aqe_enabled__param')!r})", file=sys.stderr)
    print(f"  Insights from dump: {[i.get('type') for i in insights]}", file=sys.stderr)

    history = _fetch_historical_runs_best_effort(row)

    set_row_override(row, history, sandbox_tables)
    set_insights_override(insights)
    try:
        plan, trace = await run_analysis(int(row["task_id"]), env_name="rest-dump")
    finally:
        set_row_override(None)
        set_insights_override(None)

    plan = fill_missing_annual_costs(plan, row, insights, history)

    _persist_result(plan, row=row, source="rest-dump", output_dir=_results_dir(),
                    summary_csv=None, save_results=save_results, trace=trace)
    _print_plan(plan, as_json, row)
    if not as_json:
        _print_assembled_output(_build_assembled_output(plan))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on a dump-rest-api.sh directory")
    parser.add_argument("dump_dir", type=Path, help="e.g. data/dumps/dump_5453")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    args = parser.parse_args()

    if not args.dump_dir.is_dir():
        parser.error(f"{args.dump_dir} is not a directory")
    asyncio.run(_run(args.dump_dir, args.json, save_results=not args.no_save))


if __name__ == "__main__":
    main()
