"""CLI entry point for AutoRecommendation.

Usage:
    python main.py --sample 10                      # fetch 10 random rows from Athena and run
    python main.py --sample 10 --json               # same, JSON output per task
    python main.py --task-id 2481696                # run one specific task_id from Athena

Mirrors the inference graph of definity-app's auto-recommendations-agent
branch (backend/app/brain/insights/agent) — same nodes, prompts, KB, tools
and safety rules — with Athena standing in for Postgres as the data source
(see agent/inference_graph.py for the two substituted data nodes).

Results are saved by default to data/results/ (override with RESULTS_DIR in .env).
Each run writes a JSON file with the input enrichment row, the full agent
plan (including plan_inputs and unactioned_insights), an assembled_output
section (the assemble_output node's ActiveInsight payload contract —
config_key / current_value / suggested_value — joined with each
recommendation's explanation), and a per-node graph trace. Batch runs also
get batch_summary.csv in a timestamped subfolder.

Reads from dev_app_analytics.task_enrichments using the AWS SSO profile
configured in .env (AWS_PROFILE, AWS_REGION, S3_OUTPUT).

--sample only picks (task_name, app_id) jobs with >= MIN_HISTORICAL_RUNS
(default 5) real enrichment rows in Athena, then feeds the agent's real
historical runs for that job (oldest -> newest) instead of synthetic data —
see _fetch_athena_sample_with_history / _fetch_athena_historical_runs.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Make definity-app's backend importable as "app.*" so the agent modules
# (models, state, tools, nodes/plan, etc.) resolve from the repo without duplication.
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "definity-app" / "backend"))

# csv is still used for Athena CSV result parsing and batch_summary writing

from dotenv import load_dotenv

load_dotenv()

from agent.cost_utils import DEFAULT_MEMORY_PRICE, DEFAULT_VCORE_PRICE, compute_cost_profile
import app.brain.insights.agent.constants as _agent_constants
from app.brain.insights.agent.constants import (
    HISTORICAL_RUNS_FETCH_LIMIT,
    RUN_CONTEXT_ROWS_PER_TABLE_LIMIT,
    RUN_CONTEXT_TEXT_MAX_CHARS,
)

# Optional .env overrides for the agent's per-node ReAct iteration budgets
# (e.g. SAFETY_REVIEW_RECURSION_LIMIT=24). A wandering agent that exhausts
# its budget raises GraphRecursionError and kills that task's run — the
# repo's discovery node handles it, the other LLM nodes currently don't.
# The repo defaults stay in force unless a variable is set.
#
# Each node module binds its limit by name at import time, and importing
# app.brain.insights.agent.constants above already ran the package __init__,
# which pulls in inference_graph and every node module — so patching the
# constants module alone is too late. Patch every loaded module that carries
# the name (constants + the node modules that from-imported it).
for _limit_name in (
    "DISCOVERY_RECURSION_LIMIT",
    "TRIAGE_RECURSION_LIMIT",
    "PLAN_RECURSION_LIMIT",
    "SAFETY_REVIEW_RECURSION_LIMIT",
    "EXPLAIN_RECURSION_LIMIT",
):
    _limit_value = os.getenv(_limit_name)
    if _limit_value:
        for _module in list(sys.modules.values()):
            if _module is not None and hasattr(_module, _limit_name):
                setattr(_module, _limit_name, int(_limit_value))
        print(
            f"  {_limit_name} = {_limit_value} (overridden via .env; repo default differs)",
            file=sys.stderr,
        )
from app.brain.insights.agent.models import SqlInsight
from app.brain.insights.tuning.rules.recommendations.agent import (
    PAYLOAD_ACTION,
    PAYLOAD_CONFIG_KEY,
    PAYLOAD_CURRENT_VALUE,
    PAYLOAD_SUGGESTED_VALUE,
)
from app.brain.insights.tuning.run_metrics import compute_run_metrics

_VCORE_PRICE = float(os.getenv("VCORE_PRICE", DEFAULT_VCORE_PRICE))
_MEMORY_PRICE = float(os.getenv("MEMORY_PRICE", DEFAULT_MEMORY_PRICE))

# Minimum number of real Athena enrichment rows a (task_name, app_id) job must
# have before --sample will pick it — guarantees enough real historical runs
# for trend analysis (analyze_metric_trend needs >= MIN_RUNS_REQUIRED_FOR_TREND=3;
# 5 leaves headroom after the current row is excluded from history).
_MIN_RUNS_FOR_SAMPLE = int(os.getenv("MIN_HISTORICAL_RUNS", "5"))

logging.basicConfig(
    level=logging.WARNING,   # keep LangGraph noise out of batch output
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Athena / AWS defaults (override in .env) ───────────────────────────────
_AWS_PROFILE = os.getenv("AWS_PROFILE", "dev-admin")
_AWS_REGION  = os.getenv("AWS_REGION",  "eu-north-1")
_ATHENA_DB   = os.getenv("ATHENA_DB",   "dev_app_analytics")
_ATHENA_TABLE = os.getenv("ATHENA_TABLE", "task_enrichments")
_S3_OUTPUT   = os.getenv("S3_OUTPUT",   "s3://definity-athena-results/notebooks/")

# env_name passed to run_analysis — the repo's graph uses it to scope the
# Postgres enrichment fetch; the Athena-backed fetch_context override ignores
# it, so it only shows up in the saved state.
_ENV_NAME = os.getenv("ENV_NAME", "athena")

_ATHENA_COLUMNS = """
    task_id, task_name, env_id, app_id, app_pit,
    executor__memory__allocated,
    executor__memory_heap__allocated,
    executor__memory_heap__max_used,
    executor__memory_off_heap__allocated,
    executor__memory_off_heap__max_used,
    driver__heap_memory__allocated,
    driver__heap_memory__max_used,
    executors__jvm_gc_time,
    executors__run_time__used,
    executors__vcore_time__used,
    executors__vcore_time__allocated,
    executors__cpu_time__used,
    task__vcore_time__used,
    task__vcore_time__allocated,
    task__idle_time,
    task__duration,
    task__skew_time,
    task__disk_bytes_spilled,
    task__total_io_bytes,
    task__memory_time__allocated,
    executors__memory_time__used,
    executors__used_vcore_time_of_retried_tasks,
    driver__cpu_utilization,
    task__dynamic_is_allocation_enabled__param,
    task__executors_dynamic_allocation_min_executors__param,
    task__executors_dynamic_allocation_max_executors__param,
    task__executor_instances__param,
    executor__cores,
    task__task_cpus__param,
    task__shuffle_partitions__param,
    dbx_autooptimizeshuffle,
    driver_type,
    worker_type,
    cloud_provider,
    cluster_min_workers,
    cluster_max_workers,
    cluster_workers,
    workers_availability,
    task_sub_type
""".strip()


# ── Athena helpers ─────────────────────────────────────────────────────────

_sso_login_checked = False


def _ensure_aws_sso_login(session) -> None:
    """Verify the AWS SSO session is valid, running `aws sso login` if it has expired."""
    global _sso_login_checked
    if _sso_login_checked:
        return

    from botocore.exceptions import ClientError, NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError

    try:
        session.client("sts").get_caller_identity()
    except (NoCredentialsError, TokenRetrievalError, UnauthorizedSSOTokenError, ClientError) as exc:
        print(f"  AWS SSO session invalid/expired ({exc.__class__.__name__}) — "
              f"running `aws sso login --profile {_AWS_PROFILE}`…")
        subprocess.run(["aws", "sso", "login", "--profile", _AWS_PROFILE], check=True)
        session.client("sts").get_caller_identity()

    _sso_login_checked = True


def _athena_session():
    try:
        import boto3
    except ImportError:
        print("boto3 is required for --athena mode. Run: pip install boto3")
        sys.exit(1)
    session = boto3.Session(profile_name=_AWS_PROFILE, region_name=_AWS_REGION)
    _ensure_aws_sso_login(session)
    return session


def _run_athena_query(sql: str) -> list[dict[str, Any]]:
    session = _athena_session()
    athena = session.client("athena")
    s3     = session.client("s3")

    resp = athena.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": _ATHENA_DB},
        ResultConfiguration={"OutputLocation": _S3_OUTPUT},
    )
    qid = resp["QueryExecutionId"]
    print(f"  Athena query {qid[:8]}… ", end="", flush=True)

    while True:
        status = athena.get_query_execution(QueryExecutionId=qid)["QueryExecution"]["Status"]
        state  = status["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        print(".", end="", flush=True)
        time.sleep(2)
    print(f" {state}")

    if state != "SUCCEEDED":
        raise RuntimeError(status.get("StateChangeReason", state))

    bucket = _S3_OUTPUT.split("/")[2]
    prefix = "/".join(_S3_OUTPUT.split("/")[3:]) + qid + ".csv"
    obj    = s3.get_object(Bucket=bucket, Key=prefix)
    df_bytes = obj["Body"].read()

    # Parse CSV → list of dicts, casting numeric strings
    reader = csv.DictReader(io.StringIO(df_bytes.decode("utf-8")))
    rows: list[dict[str, Any]] = []
    for raw in reader:
        rows.append({k: _cast_athena(v, column=k) for k, v in raw.items()})
    return rows


# Columns that are `character varying` in task_enrichments (per schema_export.sql)
# even though their values can look numeric — must stay strings, e.g.
# TaskProfile.shuffle_partitions_param expects "auto" or a numeric string.
_STRING_COLUMNS = {"task__shuffle_partitions__param"}


def _cast_athena(value: str, *, column: str | None = None) -> str | int | float | bool | None:
    if value == "" or value is None:
        return None
    if column in _STRING_COLUMNS:
        return value.strip() or None
    if value.strip().lower() in ("true", "false"):
        return value.strip().lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip() or None


def _sql_escape(value: str) -> str:
    return value.replace("'", "''")


def _fetch_athena_sample_with_history(n: int, min_runs: int) -> list[dict[str, Any]]:
    """Sample N distinct (task_name, app_id) jobs that have >= min_runs real
    enrichment rows in Athena, returning every eligible row for each sampled
    job (tagged with run_rank, 1 = most recent) so the caller can split each
    group into a "current" row plus real historical runs.

    Mirrors the (app_id, task_name) grouping and recency ordering that
    backend/app/dal/sql/insights/agent_historical_enrichments.sql uses
    against Postgres (start_time DESC); Athena's task_enrichments exposes the
    same point-in-time as app_pit (timestamp(3)), so it's used directly.
    """
    max_rows_per_group = 1 + HISTORICAL_RUNS_FETCH_LIMIT
    sql = f"""
    WITH filtered AS (
        SELECT {_ATHENA_COLUMNS}
        FROM {_ATHENA_DB}.{_ATHENA_TABLE}
        WHERE task__duration > 0
          AND executor__memory_heap__allocated > 0
          AND task__duration >= 15000
    ),
    ranked AS (
        SELECT *,
            COUNT(*) OVER (PARTITION BY task_name, app_id) AS run_count,
            ROW_NUMBER() OVER (PARTITION BY task_name, app_id ORDER BY app_pit DESC) AS run_rank
        FROM filtered
    ),
    eligible_groups AS (
        SELECT task_name, app_id
        FROM ranked
        WHERE run_rank = 1 AND run_count >= {min_runs}
        ORDER BY RANDOM()
        LIMIT {n}
    )
    SELECT ranked.*
    FROM ranked
    INNER JOIN eligible_groups eg
      ON ranked.task_name = eg.task_name AND ranked.app_id = eg.app_id
    WHERE ranked.run_rank <= {max_rows_per_group}
    ORDER BY ranked.task_name, ranked.app_id, ranked.run_rank
    """
    return _run_athena_query(sql)


def _strip_run_bookkeeping(row: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in row.items() if k not in ("run_rank", "run_count")}


def _group_current_and_history(
    rows: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], list[dict[str, Any]]]]:
    """Split rows from _fetch_athena_sample_with_history into (current_row,
    historical_runs) pairs, one per (task_name, app_id) group. historical_runs
    is oldest -> newest, matching fetch_context's documented convention."""
    groups: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        key = (row.get("task_name"), row.get("app_id"))
        groups.setdefault(key, []).append(row)

    result: list[tuple[dict[str, Any], list[dict[str, Any]]]] = []
    for group_rows in groups.values():
        group_rows.sort(key=lambda r: r["run_rank"])
        current, *history = group_rows
        history.reverse()  # run_rank ascends with recency (1=newest) -> oldest first
        result.append((_strip_run_bookkeeping(current), [_strip_run_bookkeeping(h) for h in history]))
    return result


def _fetch_athena_insights(
    app_id: Any, task_name: str, *, include_hidden: bool = False
) -> list[dict[str, Any]]:
    """The task's rows from the Athena `insights` table, latest snapshot only.

    Default filters mirror production insights.sql (lifecycle_status='active'
    AND visibility='visible'); --include-hidden-insights relaxes both so
    hidden/stale insights (e.g. task_profile) can be fed to the agent for
    experimentation.
    """
    visibility_filter = (
        "" if include_hidden
        else "AND lifecycle_status = 'active' AND visibility = 'visible'"
    )
    sql = f"""
    SELECT insight_id, task_name, type, impact_cost, impact_unit,
           insights_payload, lifecycle_status, visibility
    FROM {_ATHENA_DB}.insights
    WHERE app_id = {int(app_id)}
      AND task_name = '{_sql_escape(str(task_name))}'
      AND snapshot_date = (SELECT MAX(snapshot_date) FROM {_ATHENA_DB}.insights)
      {visibility_filter}
    """
    return _run_athena_query(sql)


def _fetch_athena_task(task_id: int) -> list[dict[str, Any]]:
    """The enrichment row for the EXACT task_id — the branch's
    insights/agent_latest_enrichment.sql selects `WHERE te.task_id = %s`,
    so the analyzed run is the one requested, not the task's newest run.
    Falls back to the logical task's latest run only when the exact run is
    missing from the Athena export (the caller prints a notice)."""
    sql = f"""
    SELECT {_ATHENA_COLUMNS}
    FROM {_ATHENA_DB}.{_ATHENA_TABLE}
    WHERE task_id = {int(task_id)}
    LIMIT 1
    """
    rows = _run_athena_query(sql)
    if rows:
        return rows

    print(
        f"  task_id {task_id} has no enrichment row in Athena — "
        "falling back to the logical task's latest run.",
        file=sys.stderr,
    )
    sql = f"""
    SELECT {_ATHENA_COLUMNS}
    FROM {_ATHENA_DB}.{_ATHENA_TABLE}
    WHERE task_name = (
        SELECT task_name FROM {_ATHENA_DB}.{_ATHENA_TABLE}
        WHERE task_id = {int(task_id)} LIMIT 1
      )
      AND app_id = (
        SELECT app_id FROM {_ATHENA_DB}.{_ATHENA_TABLE}
        WHERE task_id = {int(task_id)} LIMIT 1
      )
    ORDER BY app_pit DESC, task__duration DESC
    LIMIT 1
    """
    return _run_athena_query(sql)


def _fetch_athena_historical_runs(
    task_name: str, app_id: Any, exclude_task_id: int, limit: int
) -> list[dict[str, Any]]:
    """Real historical runs for one logical task (task_name, app_id), oldest
    -> newest — the --task-id counterpart of _fetch_athena_sample_with_history,
    mirroring agent_historical_enrichments.sql's semantics against Athena."""
    sql = f"""
    SELECT {_ATHENA_COLUMNS}
    FROM {_ATHENA_DB}.{_ATHENA_TABLE}
    WHERE task_name = '{_sql_escape(str(task_name))}'
      AND app_id = {int(app_id)}
      AND task_id != {int(exclude_task_id)}
      AND task__duration > 0
      AND executor__memory_heap__allocated > 0
      AND task__duration >= 15000
    ORDER BY app_pit DESC
    LIMIT {limit}
    """
    rows = _run_athena_query(sql)
    return list(reversed(rows))


# ── Run context (agent_task_run_context.sql equivalent) ────────────────────

def _run_context_section_sql(task_id: int) -> dict[str, tuple[str, str]]:
    """Per-section Athena SQL mirroring agent_task_run_context.sql (each JSON
    section of the production one-row query becomes one Athena query here —
    same columns, same ordering, same row/text caps).  Postgres-isms are
    translated to Trino: EXTRACT(EPOCH FROM a-b) → date_diff, STRING_AGG →
    array_join(array_agg(...)), SUBSTRING(x,1,n) → substr(cast(...)).
    Returns {section: (base_table, sql)}."""
    tid = int(task_id)
    rows_limit = RUN_CONTEXT_ROWS_PER_TABLE_LIMIT
    text_max = RUN_CONTEXT_TEXT_MAX_CHARS
    db = _ATHENA_DB
    return {
        "task_run": ("tasks", f"""
            SELECT t.task_id, t.task_name, t.task_type, t.task_sub_type,
                   t.status, t.error, t.start_time, t.end_time, t.app_pit,
                   t.is_retry, t.parent_task_id, t.user_task_id, t.pipeline_run_id,
                   a.app_name, e.env_name,
                   pra.status AS pipeline_status,
                   pra.sla_value AS pipeline_sla_value,
                   pra.duration_value AS pipeline_duration_value,
                   pra.task_runs AS pipeline_task_runs,
                   pra.task_retries AS pipeline_task_retries
            FROM {db}.tasks t
            INNER JOIN {db}.apps a ON t.app_id = a.app_id
            INNER JOIN {db}.envs e ON a.env_id = e.env_id
            LEFT JOIN {db}.pipeline_run_agg pra
              ON t.pipeline_run_id = pra.pipeline_run_id AND t.app_id = pra.app_id
            WHERE t.task_id = {tid}
            LIMIT 1
        """),
        "spark_parameters": ("task_params", f"""
            SELECT key, value FROM {db}.task_params
            WHERE task_id = {tid} ORDER BY key LIMIT {rows_limit}
        """),
        "events": ("events", f"""
            SELECT event_id, category, sub_category, name, description,
                   start_time_ms, end_time_ms,
                   substr(CAST(payload AS VARCHAR), 1, {text_max}) AS payload
            FROM {db}.events
            WHERE task_id = {tid} ORDER BY start_time_ms LIMIT {rows_limit}
        """),
        "metrics": ("metrics", f"""
            SELECT mc.metric_type, mc.asset_type, m.asset_value,
                   m.metric_value, m.tf_id, m.end_time
            FROM {db}.metrics m
            INNER JOIN {db}.metrics_conf mc ON m.metric_id = mc.metric_id
            WHERE m.task_id = {tid}
            ORDER BY mc.asset_type, mc.metric_type, m.asset_value
            LIMIT {rows_limit}
        """),
        "test_runs": ("test_runs", f"""
            SELECT tr.test_id, ts.test_type, mc.metric_type, mc.asset_type,
                   ts.asset_value, tr.run_value, tr.lower_bound, tr.upper_bound,
                   tr.is_passed, tr.task_broke, tr.tf_id
            FROM {db}.test_runs tr
            LEFT JOIN {db}.tests ts ON tr.test_id = ts.test_id
            LEFT JOIN {db}.metrics_conf mc ON tr.metric_id = mc.metric_id
            WHERE tr.task_id = {tid}
            ORDER BY tr.is_passed, ts.test_type
            LIMIT {rows_limit}
        """),
        "transformations": ("tfs", f"""
            SELECT tf.tf_id, tf.tf_type, tf.output_name, tf.status, tf.error,
                   tf.description, tf.start_time, tf.end_time,
                   date_diff('second', tf.start_time, tf.end_time) AS duration_seconds,
                   substr(q.query, 1, {text_max}) AS query,
                   substr(CAST(qv.query_vars AS VARCHAR), 1, {text_max}) AS query_vars,
                   tf.labels,
                   (
                     SELECT array_join(array_agg(ds_name), ', ')
                     FROM {db}.tf_inputs ti
                     WHERE ti.tf_id = tf.tf_id
                   ) AS input_datasets
            FROM {db}.tfs tf
            LEFT JOIN {db}.queries q ON tf.query_hash = q.query_hash
            LEFT JOIN {db}.tfs_query_vars qv ON tf.tf_id = qv.tf_id
            WHERE tf.task_id = {tid}
            ORDER BY tf.start_time
            LIMIT {rows_limit}
        """),
        "time_series_summary": ("time_series_metrics", f"""
            SELECT tsm.metric_type, tsm.kind, tsm.asset_name,
                   MIN(tsm.bucket_size_seconds) AS bucket_size_seconds,
                   COUNT(v.value) AS points_count,
                   MIN(v.value) AS min_value,
                   MAX(v.value) AS max_value,
                   AVG(v.value) AS avg_value
            FROM {db}.time_series_metrics tsm
            CROSS JOIN UNNEST(tsm."values") AS v (value)
            WHERE tsm.task_id = {tid}
            GROUP BY tsm.metric_type, tsm.kind, tsm.asset_name
            ORDER BY tsm.metric_type, tsm.asset_name
            LIMIT {rows_limit}
        """),
    }

_athena_tables_cache: set[str] | None = None


def _athena_table_names() -> set[str]:
    """All table names in the Athena DB (via Glue), cached for the process."""
    global _athena_tables_cache
    if _athena_tables_cache is not None:
        return _athena_tables_cache

    names: set[str] = set()
    try:
        glue = _athena_session().client("glue")
        for page in glue.get_paginator("get_tables").paginate(DatabaseName=_ATHENA_DB):
            names.update(t["Name"] for t in page.get("TableList", []))
    except Exception as exc:
        print(
            f"  Could not list Glue tables ({exc.__class__.__name__}: {exc}) — "
            "run-context sections disabled",
            file=sys.stderr,
        )
    _athena_tables_cache = names
    return names


def _truncate_text_values(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cap free-text values the way agent_task_run_context.sql SUBSTRINGs
    payload/query columns, so one pathological run can't flood the LLM."""
    return {
        k: (v[: max_chars] + "…") if isinstance(v, str) and len(v) > max_chars else v
        for k, v in row.items()
    }


def _fetch_athena_run_context(task_id: int) -> dict[str, Any] | None:
    """Athena equivalent of insights/agent_task_run_context.sql.

    The production query folds the per-run Postgres tables into one row of
    JSON sections; here each section runs as its own Athena query with the
    same columns, ordering and row/text caps. Sections whose base table is
    missing from the export are skipped; a section whose query fails (e.g.
    schema drift between Postgres and the export) is skipped with a notice.
    Returns None when nothing is available — the get_run_data tool then
    returns {}.
    """
    tables = _athena_table_names()
    sections: dict[str, Any] = {}
    for section, (table, sql) in _run_context_section_sql(task_id).items():
        if table not in tables:
            continue
        try:
            rows = _run_athena_query(sql)
        except Exception as exc:
            print(
                f"  run-context section {section!r} failed ({exc}) — skipped",
                file=sys.stderr,
            )
            continue
        rows = [_truncate_text_values(r, RUN_CONTEXT_TEXT_MAX_CHARS) for r in rows]
        sections[section] = (rows[0] if rows else None) if section == "task_run" else rows
    return sections or None


# ── Output helpers ─────────────────────────────────────────────────────────

_BATCH_SUMMARY_FIELDS = [
    "task_id",
    "task_name",
    "source",
    "idle_ratio",
    "vcore_utilization",
    "heap_headroom",
    "gc_pressure",
    "spill_mb",
    "duration_s",
    "gate_blocked",
    "recommendations",
    "blocked",
    "unactioned",
    "agent_discovered",
    "summary",
    "result_file",
]


def _results_dir() -> Path:
    return Path(os.getenv("RESULTS_DIR", "data/results"))


def _serialize_insight(insight: Any, index: int) -> dict[str, Any]:
    kind = "sql" if isinstance(insight, SqlInsight) else "discovery"
    identifier = insight.type if isinstance(insight, SqlInsight) else insight.title
    return {"index": index, "kind": kind, "identifier": identifier, **insight.model_dump()}


def _build_trace(state: dict[str, Any] | None) -> dict[str, Any] | None:
    """Surfaces the intermediate graph state that FinalPlan drops — merged
    insights and how triage classified/resolved them into tiers — so a run
    with unexpected recommendation counts can be diagnosed from the saved
    JSON alone, without re-running with --verbose."""
    if not state:
        return None

    merged = state.get("merged_insights", [])
    triage_result = state.get("triage_result")
    triage_dump = triage_result.model_dump() if triage_result else None

    tiered_insights = {
        f"tier{tier}": [
            _serialize_insight(merged[i], i)
            for i in triage_result.insights_for_tier(tier)
            if 0 <= i < len(merged)
        ]
        for tier in range(1, 5)
    } if triage_result else {}

    final_plan = state.get("final_plan")
    return {
        "gate_blocked": state.get("gate_blocked", False),
        "gate_reasons": state.get("gate_reasons", []),
        "sql_insights": [_serialize_insight(i, idx) for idx, i in enumerate(state.get("sql_insights", []))],
        "agent_insights": [_serialize_insight(i, idx) for idx, i in enumerate(state.get("agent_insights", []))],
        "merged_insights": [_serialize_insight(i, idx) for idx, i in enumerate(merged)],
        "triage_result": triage_dump,
        "tiered_insights": tiered_insights,
        "kb_version": final_plan.kb_version if final_plan else None,
        # Real Athena runs fed to analyze_metric_trend — surfaced so a saved
        # result can be audited for how much trend evidence actually backed it.
        "historical_runs_count": len(state.get("historical_runs", [])),
        # Which agent_task_run_context.sql sections the Athena export could
        # provide to the get_run_data tool (None -> tool returned {}).
        "run_context_sections": sorted((state.get("run_context") or {}).keys()),
        "sql_recommendations": [
            r.model_dump() for r in state.get("sql_recommendations", [])
        ],
    }


def _build_assembled_output(plan) -> list[dict[str, Any]]:
    """The assemble_output node's product, flattened for reading.

    One entry per ActiveInsight in plan_inputs.active_insights (what
    production feeds build_plan). agent_discovered entries expose the
    payload contract keys (PAYLOAD_CONFIG_KEY / PAYLOAD_CURRENT_VALUE /
    PAYLOAD_SUGGESTED_VALUE / PAYLOAD_ACTION) at the top level, joined with
    the explain node's explanation / expected_impact / risk_note (matched by
    config_key) or the safety block reasons. SQL-rule entries keep their
    original insight payload and list every explained recommendation that
    traces to their insight_type."""
    if plan is None or plan.plan_inputs is None:
        return []

    explained_by_key = {r.config_key: r for r in plan.recommendations}
    blocked_by_key = {r.config_key: r for r in plan.blocked_recommendations}

    assembled: list[dict[str, Any]] = []
    for insight in plan.plan_inputs.active_insights:
        payload = insight.payload or {}
        entry: dict[str, Any] = {
            "insight_type": str(insight.insight_type),
            "usd_cost_annual": insight.usd_cost_annual,
        }
        config_key = payload.get(PAYLOAD_CONFIG_KEY)
        if config_key:  # agent_discovered payload contract
            entry.update(
                config_key=config_key,
                current_value=payload.get(PAYLOAD_CURRENT_VALUE),
                suggested_value=payload.get(PAYLOAD_SUGGESTED_VALUE),
                action=payload.get(PAYLOAD_ACTION),
            )
            rec = explained_by_key.get(config_key)
            if rec is not None:
                entry.update(
                    explanation=rec.explanation,
                    expected_impact=rec.expected_impact,
                    risk_note=rec.risk_note,
                )
            blocked = blocked_by_key.get(config_key)
            if blocked is not None:
                entry["blocked_by"] = blocked.blocked_by
        else:  # SQL insight: original payload + the recommendations tracing to it
            entry["payload"] = payload
            entry["recommendations"] = [
                {
                    "config_key": r.config_key,
                    "current_value": r.current_value,
                    "suggested_value": r.suggested_value,
                    "action": str(r.action),
                    "explanation": r.explanation,
                    "expected_impact": r.expected_impact,
                    "risk_note": r.risk_note,
                }
                for r in plan.recommendations
                if str(r.insight_type or "") == str(insight.insight_type)
            ]
        assembled.append(entry)
    return assembled


def _print_assembled_output(assembled: list[dict[str, Any]]) -> None:
    if not assembled:
        return
    print(f"  Assembled output — plan_inputs.active_insights ({len(assembled)}):")
    for entry in assembled:
        if "config_key" in entry:
            state = "BLOCKED" if entry.get("blocked_by") else "safe"
            print(f"    - [{entry['insight_type']}] {entry['config_key']}: "
                  f"{entry['current_value']} → {entry['suggested_value']} ({state})")
            if entry.get("explanation"):
                print(f"        explanation: {entry['explanation']}")
            if entry.get("expected_impact"):
                print(f"        expected_impact: {entry['expected_impact']}")
            if entry.get("risk_note"):
                print(f"        risk: {entry['risk_note']}")
            if entry.get("blocked_by"):
                print(f"        blocked_by: {'; '.join(entry['blocked_by'])}")
        else:
            cost = (f" (${entry['usd_cost_annual']:,.0f}/yr)"
                    if entry.get("usd_cost_annual") is not None else "")
            print(f"    - [{entry['insight_type']}]{cost}")
            for rec in entry.get("recommendations", []):
                print(f"        → {rec['config_key']}: "
                      f"{rec['current_value']} → {rec['suggested_value']}")
                if rec.get("explanation"):
                    print(f"          explanation: {rec['explanation']}")
    print()


def _save_run_result(
    plan,
    *,
    row: dict[str, Any] | None,
    source: str,
    output_dir: Path,
    trace: dict[str, Any] | None = None,
) -> Path | None:
    if plan is None:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_at = datetime.now(timezone.utc)
    stamp = saved_at.strftime("%Y%m%dT%H%M%SZ")
    path = output_dir / f"task_{plan.task_id}_{stamp}.json"

    run_metrics = compute_run_metrics(row) if row else None
    cost_profile = compute_cost_profile(row, vcore_price=_VCORE_PRICE, memory_price=_MEMORY_PRICE) if row else None

    payload = {
        "saved_at": saved_at.isoformat(),
        "source": source,
        "task_id": plan.task_id,
        "task_name": (row or {}).get("task_name"),
        "input": row,
        "plan": plan.model_dump(),
        "run_metrics": run_metrics.model_dump() if run_metrics else None,
        "cost_profile": cost_profile.model_dump() if cost_profile else None,
        # assemble_output's product flattened for reading (payload contract
        # keys + explanations); plan.plan_inputs stays the canonical form.
        "assembled_output": _build_assembled_output(plan),
        "trace": _build_trace(trace),
    }
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def _append_batch_summary_row(
    summary_csv: Path,
    *,
    row: dict[str, Any] | None,
    plan,
    source: str,
    result_file: Path,
    trace: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_csv.exists()

    m = compute_run_metrics(row) if row else None
    summary_row = {
        "task_id": plan.task_id if plan else (row or {}).get("task_id"),
        "task_name": (row or {}).get("task_name") or "",
        "source": source,
        "idle_ratio": m.idle_ratio if m else None,
        "vcore_utilization": m.vcore_utilization if m else None,
        "heap_headroom": m.memory_headroom if m else None,
        "gc_pressure": m.gc_pressure if m else None,
        "spill_mb": ((row or {}).get("task__disk_bytes_spilled") or 0) / 1024 / 1024 if row else None,
        "duration_s": ((row or {}).get("task__duration") or 0) / 1000 if row else None,
        "gate_blocked": bool((trace or {}).get("gate_blocked")),
        "recommendations": len(plan.recommendations) if plan else 0,
        "blocked": len(plan.blocked_recommendations) if plan else 0,
        "unactioned": len(plan.unactioned_insights) if plan else 0,
        "agent_discovered": len((trace or {}).get("agent_insights", []) or []),
        "summary": (plan.summary or "") if plan else (f"ERROR: {error}" if error else ""),
        "result_file": result_file.name,
    }

    with summary_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_BATCH_SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(summary_row)


def _persist_result(
    plan,
    *,
    row: dict[str, Any] | None,
    source: str,
    output_dir: Path,
    summary_csv: Path | None,
    save_results: bool,
    trace: dict[str, Any] | None = None,
) -> Path | None:
    if not save_results:
        return None

    path = _save_run_result(
        plan, row=row, source=source, output_dir=output_dir, trace=trace,
    )
    if path is None:
        return None

    if summary_csv is not None:
        _append_batch_summary_row(
            summary_csv,
            row=row,
            plan=plan,
            source=source,
            result_file=path,
            trace=trace,
        )

    print(f"  Saved → {path.resolve()}", file=sys.stderr, flush=True)
    return path


def _fmt(v: float | None, pct: bool = False, decimals: int = 3) -> str:
    if v is None:
        return "—"
    if pct:
        return f"{v:.1%}"
    return f"{v:.{decimals}f}"


def _print_plan(plan, as_json: bool, row: dict | None = None) -> None:
    if plan is None:
        print("  !! No plan produced — check logs.")
        return

    if as_json:
        print(json.dumps(plan.model_dump(), indent=2, default=str))
        return

    m = compute_run_metrics(row) if row else None
    cost_profile = compute_cost_profile(row, vcore_price=_VCORE_PRICE, memory_price=_MEMORY_PRICE) if row else None
    print(f"\n{'─'*72}")
    print(f"  Task {plan.task_id}  |  {getattr(plan, 'task_name', None) or (row or {}).get('task_name', '?')}")
    print(f"{'─'*72}")

    if m:
        spill_mb = ((row or {}).get("task__disk_bytes_spilled") or 0) / 1024 / 1024
        duration_ms = (row or {}).get("task__duration")
        print(f"  Run metrics:")
        print(f"    idle_ratio       {_fmt(m.idle_ratio, pct=True):<10}  "
              f"vcore_util   {_fmt(m.vcore_utilization, pct=True)}")
        print(f"    heap_headroom    {_fmt(m.memory_headroom, pct=True):<10}  "
              f"gc_pressure  {_fmt(m.gc_pressure, pct=True)}")
        print(f"    skew_ratio       {_fmt(m.skew_ratio, pct=True):<10}  "
              f"spill_MB     {_fmt(spill_mb, decimals=1)}")
        print(f"    duration_ms      {_fmt(duration_ms, decimals=0):<10}  "
              f"retried_waste {_fmt(m.retried_task_waste, pct=True)}")

    if cost_profile and cost_profile.cost_per_run_usd is not None:
        print(f"  Cost per run: {cost_profile.cost_per_run_usd:.4f} USD  "
              f"| workers: {cost_profile.cluster_workers} × {cost_profile.workers_availability or '?'}")

    if plan.recommendations:
        print(f"\n  Recommendations ({len(plan.recommendations)}):")
        for i, rec in enumerate(plan.recommendations, 1):
            impact = f"  [{rec.expected_impact}]" if rec.expected_impact else ""
            print(f"    {i}. [{rec.action}] {rec.config_key}: "
                  f"{rec.current_value} → {rec.suggested_value}{impact}")
            print(f"       {rec.explanation or rec.rationale}")
            if rec.risk_note:
                print(f"       risk: {rec.risk_note}")
    else:
        print("\n  No recommendations.")

    if plan.blocked_recommendations:
        print(f"\n  Blocked ({len(plan.blocked_recommendations)}):")
        for rec in plan.blocked_recommendations:
            print(f"    - {rec.config_key}: {'; '.join(rec.blocked_by or [])}")

    if plan.unactioned_insights:
        print(f"\n  Unactioned insights ({len(plan.unactioned_insights)}):")
        for insight in plan.unactioned_insights:
            print(f"    - [{insight.source}] {insight.title}: {insight.reason}")

    if plan.summary:
        print(f"\n  Summary: {plan.summary}")
    print()


def _print_batch_table(results: list[tuple[dict, object]]) -> None:
    """Print a compact comparison table across all tasks."""
    COL = 10
    headers = ["task_id", "task_name", "idle%", "vcore%", "heap%", "gc%",
               "spill_MB", "duration_s", "#recs", "blocked", "summary"]
    widths   = [10,        30,           7,       7,        7,       6,
                9,          10,           6,       8,        50]

    sep = "  ".join(f"{'─'*w}" for w in widths)
    hdr = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(f"\n{'═'*len(sep)}")
    print("  BATCH RESULTS")
    print(f"{'═'*len(sep)}")
    print(hdr)
    print(sep)

    for row, plan in results:
        if plan is None:
            vals = [
                str(row.get("task_id", "?"))[:10],
                str(row.get("task_name", "?"))[:30],
                *["—"] * 8,
                "ERROR — see stderr / batch_summary.csv",
            ]
            print("  ".join(f"{v:<{w}}" for v, w in zip(vals, widths)))
            continue

        m    = compute_run_metrics(row) if row else None
        vals = [
            str(plan.task_id)[:10],
            str(getattr(plan, "task_name", None) or row.get("task_name", "?"))[:30],
            _fmt(m.idle_ratio if m else None, pct=True)[:7],
            _fmt(m.vcore_utilization if m else None, pct=True)[:7],
            _fmt(m.memory_headroom if m else None, pct=True)[:7],
            _fmt(m.gc_pressure if m else None, pct=True)[:6],
            _fmt((row.get("task__disk_bytes_spilled") or 0)/1024/1024, decimals=1)[:9],
            _fmt((row.get("task__duration") or 0)/1000, decimals=1)[:10],
            str(len(plan.recommendations)),
            str(len(plan.blocked_recommendations)),
            (plan.summary or "")[:50],
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(vals, widths)))

    print()


# ── Runner ─────────────────────────────────────────────────────────────────

async def _run_on_row(
    row: dict,
    as_json: bool,
    *,
    show_progress: bool = True,
    save_results: bool = True,
    output_dir: Path | None = None,
    summary_csv: Path | None = None,
    source: str = "athena",
    historical_runs: list[dict] | None = None,
    include_hidden_insights: bool = False,
    fetch_run_context: bool = True,
) -> tuple[dict, object]:
    from agent.inference_graph import run_analysis
    from agent.nodes.fetch_context import set_row_override
    from agent.nodes.sql_stream import set_insights_override

    task_id = int(row.get("task_id", 0))
    insight_rows = _fetch_athena_insights(
        row.get("app_id"), str(row.get("task_name")),
        include_hidden=include_hidden_insights,
    )
    print(
        f"  Insights table: {len(insight_rows)} row(s) for "
        f"({row.get('task_name')}, app_id={row.get('app_id')})"
        + (" [including hidden/stale]" if include_hidden_insights else ""),
        file=sys.stderr, flush=True,
    )

    run_context = _fetch_athena_run_context(task_id) if fetch_run_context else None

    set_row_override(row, historical_runs, run_context)
    set_insights_override(insight_rows)
    try:
        plan, trace = await run_analysis(
            task_id, env_name=_ENV_NAME, show_progress=show_progress
        )
    finally:
        set_row_override(None)
        set_insights_override(None)

    _persist_result(
        plan,
        row=row,
        source=source,
        output_dir=output_dir or _results_dir(),
        summary_csv=summary_csv,
        save_results=save_results,
        trace=trace,
    )
    _print_plan(plan, as_json, row)
    if not as_json:
        _print_assembled_output(_build_assembled_output(plan))
    return row, plan


async def _run_athena_batch(
    sample: int | None,
    task_id: int | None,
    as_json: bool,
    sequential: bool,
    *,
    show_progress: bool = True,
    save_results: bool = True,
    include_hidden_insights: bool = False,
    fetch_run_context: bool = True,
) -> None:
    print(f"\nFetching from Athena ({_ATHENA_DB}.{_ATHENA_TABLE})…")
    if task_id is not None:
        target_rows = _fetch_athena_task(task_id)
        if not target_rows:
            print(f"No row found in Athena for task_id={task_id}")
            sys.exit(1)
        target_row = target_rows[0]
        analyzed_task_id = int(target_row["task_id"])
        if analyzed_task_id != task_id:
            print(
                f"  task_id {task_id} resolved to the latest run task_id "
                f"{analyzed_task_id} of ({target_row.get('task_name')}, "
                f"app_id={target_row.get('app_id')})",
                file=sys.stderr,
            )
        history = _fetch_athena_historical_runs(
            target_row.get("task_name"),
            target_row.get("app_id"),
            analyzed_task_id,
            HISTORICAL_RUNS_FETCH_LIMIT,
        )
        jobs: list[tuple[dict, list[dict]]] = [(target_row, history)]
    else:
        sampled_rows = _fetch_athena_sample_with_history(sample or 10, _MIN_RUNS_FOR_SAMPLE)
        jobs = _group_current_and_history(sampled_rows)
        requested = sample or 10
        if len(jobs) < requested:
            print(
                f"  Note: only {len(jobs)} task(s) with >= {_MIN_RUNS_FOR_SAMPLE} historical "
                f"runs were found in Athena (requested {requested}).",
                file=sys.stderr,
            )

    print(f"Running agent on {len(jobs)} task(s)…\n")

    batch_dir: Path | None = None
    summary_csv: Path | None = None
    if save_results:
        batch_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        batch_dir = _results_dir() / f"batch_{batch_stamp}"
        summary_csv = batch_dir / "batch_summary.csv"
        print(f"  Results dir → {batch_dir.resolve()}", file=sys.stderr, flush=True)

    if not sequential and len(jobs) > 1:
        # Module-global overrides are not safe for concurrent use.
        print("Note: running sequentially to avoid context conflicts between tasks.")

    results: list[tuple[dict, object]] = []
    for row, history in jobs:
        try:
            result = await _run_on_row(
                row,
                as_json,
                show_progress=show_progress,
                save_results=save_results,
                output_dir=batch_dir or _results_dir(),
                summary_csv=summary_csv,
                source="athena",
                historical_runs=history,
                include_hidden_insights=include_hidden_insights,
                fetch_run_context=fetch_run_context,
            )
        except Exception as exc:
            # One task's failure (e.g. a GraphRecursionError from an LLM node
            # that exhausted its iteration budget — production propagates it
            # the same way) shouldn't kill the rest of the batch.
            logger.exception("Agent run failed for task_id=%s", row.get("task_id"))
            print(
                f"  !! Task {row.get('task_id')} ({row.get('task_name')}) failed: "
                f"{exc.__class__.__name__}: {exc}",
                file=sys.stderr, flush=True,
            )
            if save_results and summary_csv is not None:
                _append_batch_summary_row(
                    summary_csv,
                    row=row,
                    plan=None,
                    source="athena",
                    result_file=Path(""),
                    trace=None,
                    error=f"{exc.__class__.__name__}: {exc}",
                )
            results.append((row, None))
            continue
        results.append(result)

    if save_results and summary_csv is not None and summary_csv.exists():
        print(f"  Summary CSV → {summary_csv.resolve()}", file=sys.stderr, flush=True)

    if not as_json and len(results) > 1:
        _print_batch_table(results)


# ── Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AutoRecommendation — Spark tuning agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--task-id", type=int, metavar="ID",
                      help="Run on a specific task_id from Athena")
    mode.add_argument("--sample",  type=int, metavar="N",
                      help="Fetch N random rows from Athena and run the agent on each")

    parser.add_argument("--json",    action="store_true", help="Output raw JSON per task")
    parser.add_argument("--include-hidden-insights", action="store_true",
                        help="Feed hidden/stale insights (e.g. task_profile) to the agent too — "
                             "by default only active+visible ones are used, matching production")
    parser.add_argument("--skip-run-context", action="store_true",
                        help="Do not query Athena for the raw per-run context tables "
                             "(params/events/metrics/tests/transformations) — the "
                             "get_run_data tool then returns {}")
    parser.add_argument("--quiet",   action="store_true", help="Suppress per-node progress output")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    parser.add_argument("--verbose", action="store_true", help="Show full agent log output")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    show_progress = not args.quiet
    save_results = not args.no_save

    asyncio.run(_run_athena_batch(
        sample=args.sample,
        task_id=args.task_id,
        as_json=args.json,
        sequential=True,
        show_progress=show_progress,
        save_results=save_results,
        include_hidden_insights=args.include_hidden_insights,
        fetch_run_context=not args.skip_run_context,
    ))


if __name__ == "__main__":
    main()
