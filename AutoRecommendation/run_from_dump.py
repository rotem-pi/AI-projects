"""Run the AutoRecommendation agent on a REST API dump instead of Athena.

Usage:
    python run_from_dump.py data/dumps/dump_5453     # a directory made by tools/dump-rest-api.sh
    python run_from_dump.py data/dumps/dump_5453 --json

Bridges the CSVs produced by dump-rest-api.sh into the same three inputs
main.py injects before invoking the inference graph:

- enrichment row  <- task_metrics.csv + task_tsm.csv + tf_<id>.csv
  (REST metric_type values mapped to task_enrichments column names;
  time metrics converted seconds -> ms to match the enrichment schema)
- run_context     <- events / metrics / tfs / tsm / physical-plan CSVs,
  shaped like the agent_task_run_context.sql sections main.py builds
- sql insights    <- the `insights` JSON column embedded in task_tsm.csv
  (the same rows the production insights table holds for this task)

Known gaps of a REST dump vs Athena, surfaced at startup:
- no historical runs (single-run dump) -> no trend analysis
- /api/tasks/<id> and /api/tasks/<id>/params returned 500 for this dump ->
  no spark_parameters section; shuffle partitions is recovered from the
  insight payload (corroborated by stage numTasks) when available.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import (  # noqa: E402 — main wires sys.path to definity-app's backend
    RUN_CONTEXT_TEXT_MAX_CHARS,
    _build_assembled_output,
    _cast_athena,
    _print_assembled_output,
    _print_plan,
    _persist_result,
    _results_dir,
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
    """The /params endpoint 500s, so recover spark.sql.shuffle.partitions from
    the insight payload (its value is corroborated by stage numTasks)."""
    for insight in insights:
        payload = insight.get("insights_payload")
        if isinstance(payload, dict) and payload.get("shuffle_partitions"):
            return str(payload["shuffle_partitions"])
    return None


def build_enrichment_row(dump: Path, insights: list[dict[str, object]]) -> dict[str, object]:
    metrics = _task_metrics(dump)
    task = _task_detail(dump)

    row: dict[str, object] = {
        "task_id": task.get("task_id"),
        "task_name": task.get("task_name"),
        "app_id": task.get("app_id"),
        "app_pit": task.get("app_pit"),
        "env_id": task.get("env_id", task.get("env")),
    }
    for metric_type, (column, seconds_to_ms) in _METRIC_TO_ENRICHMENT.items():
        value = metrics.get(metric_type)
        row[column] = value * MS_PER_SECOND if (value is not None and seconds_to_ms) else value

    io_values = [metrics[m] for m in _IO_METRIC_TYPES if m in metrics]
    row["task__total_io_bytes"] = sum(io_values) if io_values else None

    for tsm_row in _read_rows(dump / "task_tsm.csv"):
        if tsm_row.get("readable_name") == "Executors Memory Usage":
            row["executors__memory_time__used"] = tsm_row.get("used")

    row["task__shuffle_partitions__param"] = _shuffle_partitions_param(insights)
    return row


def _truncate(value: object) -> object:
    text = value if isinstance(value, str) else None
    if text is not None and len(text) > RUN_CONTEXT_TEXT_MAX_CHARS:
        return text[:RUN_CONTEXT_TEXT_MAX_CHARS] + "…"
    return value


def _section_rows(path: Path, columns: dict[str, str]) -> list[dict[str, object]]:
    """Project dump rows onto run-context column names, truncating free text."""
    return [
        {out_col: _truncate(row.get(src_col)) for src_col, out_col in columns.items()}
        for row in _read_rows(path)
    ]


def build_run_context(dump: Path, enrichment: dict[str, object]) -> dict[str, object]:
    task = _task_detail(dump)
    sections: dict[str, object] = {
        "task_run": {
            "task_id": enrichment.get("task_id"),
            "task_name": enrichment.get("task_name"),
            "status": task.get("status"),
            "start_time": task.get("start_time"),
            "end_time": task.get("end_time"),
            "app_pit": enrichment.get("app_pit"),
            "app_name": task.get("app_name"),
            "env_name": task.get("env_name", task.get("env")),
        },
        "events": _section_rows(
            dump / "task_events.csv",
            {c: c for c in (
                "event_id", "category", "sub_category", "name", "description",
                "start_time_ms", "end_time_ms", "payload",
            )},
        ),
        "metrics": _section_rows(
            dump / "task_metrics.csv",
            {"metric_type": "metric_type", "asset_type": "asset_type",
             "asset_name": "asset_value", "metric_value": "metric_value"},
        ),
        "transformations": _section_rows(
            dump / "task_tfs.csv",
            {"tf_id": "tf_id", "tf_type": "tf_type", "output_name": "output_name",
             "status": "status", "error": "error", "description": "description",
             "start_time": "start_time", "end_time": "end_time",
             "duration": "duration_seconds", "query_str": "query",
             "query_vars": "query_vars", "labels": "labels", "inputs": "input_datasets"},
        ),
        "time_series_summary": _section_rows(
            dump / "task_tsm.csv",
            {c: c for c in (
                "readable_name", "allocated", "used", "utilization_pct",
                "units", "grouping_metric",
            )},
        ),
    }
    plan_rows = _section_rows(
        next(dump.glob("tf_*_physical_plan.csv"), dump / "missing"),
        {"tfId": "tf_id", "plan": "plan"},
    )
    if plan_rows:
        sections["physical_plans"] = plan_rows
    return {name: rows for name, rows in sections.items() if rows}


async def _run(dump: Path, as_json: bool, save_results: bool) -> None:
    from agent.inference_graph import run_analysis
    from agent.nodes.fetch_context import set_row_override
    from agent.nodes.sql_stream import set_insights_override

    insights = _dump_insights(dump)
    row = build_enrichment_row(dump, insights)
    run_context = build_run_context(dump, row)

    print(f"  Enrichment columns recovered: "
          f"{sum(v is not None for v in row.values())}/{len(row)}", file=sys.stderr)
    print(f"  Run-context sections: {', '.join(run_context)}", file=sys.stderr)
    print(f"  Insights from dump: {[i.get('type') for i in insights]}", file=sys.stderr)
    print("  No historical runs in a REST dump — trend analysis unavailable.",
          file=sys.stderr)

    set_row_override(row, [], run_context)
    set_insights_override(insights)
    try:
        plan, trace = await run_analysis(int(row["task_id"]), env_name="rest-dump")
    finally:
        set_row_override(None)
        set_insights_override(None)

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
