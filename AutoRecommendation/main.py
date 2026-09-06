"""CLI entry point for AutoRecommendation.

Usage:
    ./run.sh main.py --sample 10                      # fetch 10 random rows from Athena and run
    ./run.sh main.py --sample 10 --json               # same, JSON output per task
    ./run.sh main.py --task-id 2481696                # run one specific task_id from Athena

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
# (models, state, tools, nodes/plan, etc.) resolve from the repo without
# duplication. The path is resolved by bootstrap_worktree.py (a dedicated,
# harness-managed worktree pinned to auto-recommendations-agent — see that
# module's docstring for why), not assumed from directory layout — run
# `python bootstrap_worktree.py` once (or use ./run.sh, which does this
# automatically) before running this file directly.
from bootstrap_worktree import resolved_backend_path

sys.path.insert(0, str(resolved_backend_path()))

# csv is still used for Athena CSV result parsing and batch_summary writing
# Athena result CSVs carry whole JSON columns (events.payload cast to VARCHAR
# can exceed 128 KB); csv's default field limit made those queries raise
# "field larger than field limit" and the sandbox silently dropped the table.
csv.field_size_limit(sys.maxsize)

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from agent.cost_utils import DEFAULT_MEMORY_PRICE, DEFAULT_VCORE_PRICE, compute_cost_profile
import app.brain.insights.agent.constants as _agent_constants
from app.brain.insights.agent.constants import HISTORICAL_RUNS_FETCH_LIMIT

# Row/text caps for the raw per-table Athena queries that feed the local run
# sandbox (agent/local_sandbox.py) — same values the branch used for its old
# pre-joined run-context sections, before the S3-sandbox change (dd494cb3d).
SANDBOX_ROWS_PER_TABLE_LIMIT = 300
SANDBOX_TEXT_MAX_CHARS = 2000

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
    # Soft model-call limit for plan's ForceFinalizeMiddleware — must stay
    # below PLAN_RECURSION_LIMIT/2 (hard steps ≈ 2 × model calls) or it
    # never fires and the node degrades via GraphRecursionError instead.
    "PLAN_FORCE_FINALIZE_MODEL_CALLS",
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
from app.brain.insights.recommendations.agent import (
    PAYLOAD_ACTION,
    PAYLOAD_CONFIG_KEY,
    PAYLOAD_CURRENT_VALUE,
    PAYLOAD_RECOMMENDATIONS,
    PAYLOAD_SUGGESTED_VALUE,
)
from app.brain.insights.tuning.entities.run_metrics import compute_run_metrics

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


# The same columns qualified for queries that join task_enrichments (aliased
# `e`) to the Athena `tasks` mirror (aliased `t`) for the run status — the
# enrichment export has no status column, and the entry gate only lets
# COMPLETED runs through (agent/nodes/entry_gate.py). tasks is unique per
# task_id in the export, so a plain LEFT JOIN adds no rows.
_ATHENA_COLUMNS_E = ", ".join(f"e.{c.strip()}" for c in _ATHENA_COLUMNS.split(","))
_COMPLETED_STATUS = "COMPLETED"


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
        # AWS_SSO_AUTO_LOGIN=0: never shell out to `aws sso login` from here —
        # a long-running service owns the (device-code) login flow itself and
        # needs the failure surfaced, not an interactive prompt on its stdout.
        if os.getenv("AWS_SSO_AUTO_LOGIN", "1") == "0":
            raise
        print(f"  AWS SSO session invalid/expired ({exc.__class__.__name__}) — "
              f"running `aws sso login --profile {_AWS_PROFILE}`…")
        cmd = ["aws", "sso", "login", "--profile", _AWS_PROFILE]
        # Headless hosts (containers) can't open a browser: the device-code
        # flow prints a URL + code for the user to approve elsewhere.
        if os.getenv("AWS_SSO_USE_DEVICE_CODE") == "1":
            cmd.append("--use-device-code")
        subprocess.run(cmd, check=True)
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
    backend/app/dal/sql/insights/agent_context.sql (@query:
    historical_enrichments) uses against Postgres (start_time DESC);
    Athena's task_enrichments exposes the same point-in-time as app_pit
    (timestamp(3)), so it's used directly.
    """
    max_rows_per_group = 1 + HISTORICAL_RUNS_FETCH_LIMIT
    sql = f"""
    WITH filtered AS (
        SELECT {_ATHENA_COLUMNS_E}, t.status
        FROM {_ATHENA_DB}.{_ATHENA_TABLE} e
        LEFT JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
        WHERE e.task__duration > 0
          AND e.executor__memory_heap__allocated > 0
          AND e.task__duration >= 15
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

    The default filter mirrors the production agent's active_insights query
    (agent_context.sql): lifecycle_status='active' ONLY — production does
    NOT filter on visibility, so hidden-but-active insights (e.g.
    task_profile) reach the agent there too.  Agent runs always use this
    prod filter; include_hidden exists only for diagnostics
    (evaluation/compare_sql_stream.py) that need the unfiltered rows.
    """
    visibility_filter = (
        "" if include_hidden
        else "AND lifecycle_status = 'active'"
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
    insights/agent_context.sql (@query: latest_enrichment) resolves the
    logical task from task_id and takes its latest COMPLETED run, not the
    exact run requested; this fetches the exact run_id instead, so the
    analyzed run is the one requested, not the task's newest run.
    Falls back to the logical task's latest run only when the exact run is
    missing from the Athena export (the caller prints a notice)."""
    sql = f"""
    SELECT {_ATHENA_COLUMNS_E}, t.status
    FROM {_ATHENA_DB}.{_ATHENA_TABLE} e
    LEFT JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
    WHERE e.task_id = {int(task_id)}
    LIMIT 1
    """
    rows = _run_athena_query(sql)
    if rows:
        return rows

    print(
        f"  task_id {task_id} has no enrichment row in Athena — "
        "falling back to the logical task's latest COMPLETED run.",
        file=sys.stderr,
    )
    sql = f"""
    SELECT {_ATHENA_COLUMNS_E}, t.status
    FROM {_ATHENA_DB}.{_ATHENA_TABLE} e
    LEFT JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
    WHERE e.task_name = (
        SELECT task_name FROM {_ATHENA_DB}.{_ATHENA_TABLE}
        WHERE task_id = {int(task_id)} LIMIT 1
      )
      AND e.app_id = (
        SELECT app_id FROM {_ATHENA_DB}.{_ATHENA_TABLE}
        WHERE task_id = {int(task_id)} LIMIT 1
      )
      AND t.status = '{_COMPLETED_STATUS}'
    ORDER BY e.app_pit DESC, e.task__duration DESC
    LIMIT 1
    """
    return _run_athena_query(sql)


def _fetch_athena_historical_runs(
    task_name: str, app_id: Any, exclude_task_id: int, limit: int
) -> list[dict[str, Any]]:
    """Real historical runs for one logical task (task_name, app_id), oldest
    -> newest — the --task-id counterpart of _fetch_athena_sample_with_history,
    mirroring agent_context.sql's (@query: historical_enrichments) semantics
    against Athena."""
    sql = f"""
    SELECT {_ATHENA_COLUMNS_E}, t.status
    FROM {_ATHENA_DB}.{_ATHENA_TABLE} e
    LEFT JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
    WHERE e.task_name = '{_sql_escape(str(task_name))}'
      AND e.app_id = {int(app_id)}
      AND e.task_id != {int(exclude_task_id)}
      AND e.task__duration > 0
      AND e.executor__memory_heap__allocated > 0
      AND e.task__duration >= 15
    ORDER BY e.app_pit DESC
    LIMIT {limit}
    """
    rows = _run_athena_query(sql)
    return list(reversed(rows))


def _fetch_athena_task_candidates(task_name: str) -> list[dict[str, Any]]:
    """Every (app_id, env_id) a task name runs under, with its latest
    analyzable run — the disambiguation step a name-based entry point
    needs, since task_name alone is not a key in task_enrichments (the same
    name recurs across apps/environments; generic Databricks task names like
    "compute" run under hundreds of apps). Joined to the Athena `apps` /
    `envs` mirrors for human-readable app_name / env_name (deduped by
    app_id/env_id in case the export holds more than one row per key).
    Rows use the same analyzability filter _fetch_athena_historical_runs
    applies plus tasks.status = COMPLETED (the entry gate's requirement), so
    latest_task_id is a run the agent can actually work on. Newest app
    first."""
    db = _ATHENA_DB
    sql = f"""
    WITH runs AS (
        SELECT e.app_id, e.env_id,
               COUNT(*)                     AS run_count,
               MAX(e.app_pit)               AS latest_app_pit,
               MAX_BY(e.task_id, e.app_pit) AS latest_task_id
        FROM {db}.{_ATHENA_TABLE} e
        INNER JOIN {db}.tasks t ON t.task_id = e.task_id
        WHERE e.task_name = '{_sql_escape(str(task_name))}'
          AND t.status = '{_COMPLETED_STATUS}'
          AND e.task__duration > 0
          AND e.executor__memory_heap__allocated > 0
          AND e.task__duration >= 15
        GROUP BY e.app_id, e.env_id
    ),
    apps AS (
        SELECT app_id, app_name, deleted_at
        FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY app_id ORDER BY created_at DESC) AS rn
              FROM {db}.apps)
        WHERE rn = 1
    ),
    envs AS (
        SELECT env_id, env_name, tenant_id
        FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY env_id ORDER BY env_id) AS rn
              FROM {db}.envs)
        WHERE rn = 1
    )
    SELECT runs.app_id, apps.app_name, apps.deleted_at AS app_deleted_at,
           runs.env_id, envs.env_name, envs.tenant_id,
           runs.run_count, runs.latest_app_pit, runs.latest_task_id
    FROM runs
    LEFT JOIN apps ON apps.app_id = runs.app_id
    LEFT JOIN envs ON envs.env_id = runs.env_id
    ORDER BY runs.latest_app_pit DESC
    """
    return _run_athena_query(sql)


def _search_athena_task_names(fragment: str, limit: int = 20) -> list[dict[str, Any]]:
    """Task names matching `fragment` (case-insensitive substring), each with
    a real analyzable COMPLETED run — a UI's type-ahead when the exact name
    isn't known. Two things a plain `LIKE '%frag%' ORDER BY task_name` gets
    wrong, fixed here:

    - Relevance: alphabetical order puts a dotted/prefixed name like
      "com.att.eg....compute.Application" ahead of the literal "compute"
      (ASCII '.' sorts before letters), so a search for "compute" buried the
      one name a person typed under a page of noise. Ranked instead: exact
      match, then prefix match, then plain substring match; ties broken by
      which name most recently had an analyzable run, so live names surface
      over one-off historical ones.
    - Dead ends: a task_name with zero COMPLETED, long-enough runs would
      suggest a name that then fails with "no analyzable runs" the moment
      it's picked — filtered out via the same join/thresholds
      _fetch_athena_task_candidates uses.

    Full scan of task_enrichments joined to tasks; fine for an internal
    tool, not for a hot path."""
    frag = _sql_escape(str(fragment).lower())
    sql = f"""
    WITH matches AS (
        SELECT e.task_name,
               COUNT(*)     AS run_count,
               MAX(e.app_pit) AS latest_app_pit
        FROM {_ATHENA_DB}.{_ATHENA_TABLE} e
        JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
        WHERE LOWER(e.task_name) LIKE '%{frag}%'
          AND t.status = '{_COMPLETED_STATUS}'
          AND e.task__duration >= 15
          AND e.executor__memory_heap__allocated > 0
        GROUP BY e.task_name
    )
    SELECT task_name, run_count, latest_app_pit
    FROM matches
    ORDER BY
        CASE
            WHEN LOWER(task_name) = '{frag}' THEN 0
            WHEN LOWER(task_name) LIKE '{frag}%' THEN 1
            ELSE 2
        END,
        latest_app_pit DESC
    LIMIT {int(limit)}
    """
    return _run_athena_query(sql)


def _search_athena_apps_by_name(fragment: str, limit: int = 10) -> list[dict[str, Any]]:
    """Apps whose app_name matches `fragment`, each paired with the real
    Spark task_name(s) it runs — the fallback for when someone searches (or
    types into "Find runs") the Databricks cluster/job label instead of the
    task name task_enrichments actually keys on (e.g. "platinum_feedback_
    cluster" is an app_name; the task that runs on it is "compute"). Only
    analyzable COMPLETED runs count, same thresholds as
    _fetch_athena_task_candidates, so every suggestion here leads somewhere.
    Newest run first."""
    frag = _sql_escape(str(fragment).lower())
    sql = f"""
    WITH app_match AS (
        SELECT app_id, app_name
        FROM (
            SELECT *, ROW_NUMBER() OVER (PARTITION BY app_id ORDER BY created_at DESC) AS rn
            FROM {_ATHENA_DB}.apps
        )
        WHERE rn = 1 AND LOWER(app_name) LIKE '%{frag}%'
    )
    SELECT am.app_id, am.app_name, e.task_name,
           COUNT(*) AS run_count, MAX(e.app_pit) AS latest_app_pit
    FROM app_match am
    JOIN {_ATHENA_DB}.{_ATHENA_TABLE} e ON e.app_id = am.app_id
    JOIN {_ATHENA_DB}.tasks t ON t.task_id = e.task_id
    WHERE t.status = '{_COMPLETED_STATUS}'
      AND e.task__duration >= 15
      AND e.executor__memory_heap__allocated > 0
    GROUP BY am.app_id, am.app_name, e.task_name
    ORDER BY latest_app_pit DESC
    LIMIT {int(limit)}
    """
    return _run_athena_query(sql)


def _fetch_athena_export_freshness() -> dict[str, Any]:
    """How far behind production the Athena export is: the newest run point-
    in-time in task_enrichments and the newest insights snapshot. Surfaced
    to users so "latest run" is understood as "latest exported run"."""
    sql = f"""
    SELECT
      (SELECT MAX(app_pit)       FROM {_ATHENA_DB}.{_ATHENA_TABLE}) AS latest_enrichment_pit,
      (SELECT MAX(snapshot_date) FROM {_ATHENA_DB}.insights)        AS latest_insights_snapshot
    """
    rows = _run_athena_query(sql)
    return rows[0] if rows else {}


# ── Local run sandbox (raw per-table Athena queries) ───────────────────────
#
# Mirrors backend/app/brain/insights/agent/constants.py's SANDBOX_DIRECT_TABLES
# / SANDBOX_TF_KEYED_TABLES: each table dumped verbatim (SELECT * WHERE
# task_id = ...) rather than the old pre-joined agent_task_run_context.sql
# sections. tasks/task_enrichments aren't queried again here — this harness
# already has the enrichment row (and no separate `tasks` row) — the
# enrichment row itself is wrapped as a one-row task_enrichments table.


def _sandbox_table_sql(task_id: int) -> dict[str, tuple[str, str]]:
    """Per-table Athena SQL mirroring production's sandbox dump exactly
    (agent_context.sql `sandbox_table_dump` / `sandbox_tf_table_dump`:
    SELECT * FROM <table> WHERE task_id = ..., tf-keyed tables reached
    through tfs.tf_id). The KB analysis layer (kb/analysis/*) reads the raw
    column names — metrics.metric_id, time_series_metrics.values as a JSON
    array string, events.payload — so any projection or aggregation here
    breaks a detector somewhere (estimate_change_saving crashed on a
    min/max/avg rollup of time_series_metrics that dropped `values`).
    Athena-specific casts only: array columns become JSON text, as Postgres'
    CSV export renders them. Returns {table: (base_table, sql)} (the base
    table is used to skip a table the Athena export doesn't have)."""
    tid = int(task_id)
    rows_limit = SANDBOX_ROWS_PER_TABLE_LIMIT
    db = _ATHENA_DB
    direct = {
        "tasks": "*",
        "task_params": "*",
        "events": "*",
        "metrics": "*",
        "test_runs": "*",
        "tfs": "*",
        "time_series_metrics": (
            "app_id, task_id, metric_type, created_time, start_time_ms, bucket_size_seconds, "
            "json_format(CAST(\"values\" AS JSON)) AS \"values\", kind, "
            "json_format(CAST(server_metrics AS JSON)) AS server_metrics, asset_name"
        ),
    }
    order = {"events": "start_time_ms", "tfs": "start_time", "task_params": "key",
             "metrics": "metric_id", "test_runs": "test_id", "time_series_metrics": "metric_type",
             "tasks": "task_id"}
    out: dict[str, tuple[str, str]] = {
        table: (table, f"""
            SELECT {cols} FROM {db}.{table}
            WHERE task_id = {tid} ORDER BY {order[table]} LIMIT {rows_limit}
        """)
        for table, cols in direct.items()
    }
    for table in ("tf_inputs", "tfs_query_vars"):
        out[table] = (table, f"""
            SELECT * FROM {db}.{table}
            WHERE tf_id IN (SELECT tf_id FROM {db}.tfs WHERE task_id = {tid})
            LIMIT {rows_limit}
        """)
    return out

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
            "sandbox tables disabled",
            file=sys.stderr,
        )
    _athena_tables_cache = names
    return names


# Structured JSON columns that downstream code json.loads() — truncating them
# yields unparseable rows that are silently dropped (stage events over 2,000
# chars vanished, so partition_distribution saw only a 1-partition final stage).
# Production's sandbox dump does not truncate; neither may the harness.
_STRUCTURED_COLUMNS = frozenset({"payload", "values", "server_metrics", "query_vars", "columns"})


def _truncate_text_values(row: dict[str, Any], max_chars: int) -> dict[str, Any]:
    """Cap free-text values (query, description, ...) so one pathological run
    can't flood the downstream review/LLM tools. Structured JSON columns are
    passed through untouched — see _STRUCTURED_COLUMNS."""
    return {
        k: (
            (v[:max_chars] + "…")
            if isinstance(v, str) and len(v) > max_chars and k not in _STRUCTURED_COLUMNS
            else v
        )
        for k, v in row.items()
    }


def _fetch_athena_sandbox_tables(
    task_id: int, enrichment: dict[str, Any], insight_rows: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Raw per-table rows for this single run, Athena-sourced — the local
    run-sandbox equivalent of production's S3 CSV dump. Tables whose base
    table is missing from the Athena export are skipped; a table whose query
    fails (e.g. schema drift) is skipped with a notice. Single-PIT only (this
    run's rows, no multi-run history).
    """
    available = _athena_table_names()
    tables: dict[str, list[dict[str, Any]]] = {"task_enrichments": [enrichment]}
    for table, (base_table, sql) in _sandbox_table_sql(task_id).items():
        if base_table not in available:
            continue
        try:
            rows = _run_athena_query(sql)
        except Exception as exc:
            print(
                f"  sandbox table {table!r} failed ({exc}) — skipped",
                file=sys.stderr,
            )
            continue
        rows = [_truncate_text_values(r, SANDBOX_TEXT_MAX_CHARS) for r in rows]
        if rows:
            tables[table] = rows
    if insight_rows:
        tables["insights"] = insight_rows
    return tables


def _spark_param_value(sandbox_tables: dict[str, Any] | None, key: str) -> Any:
    """{key, value} row lookup in the sandbox's task_params table — same
    shape run_from_dump.py's _param_value reads."""
    params = (sandbox_tables or {}).get("task_params")
    if not isinstance(params, list):
        return None
    return next((p.get("value") for p in params if p.get("key") == key), None)


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
    configured = Path(os.getenv("RESULTS_DIR", "data/results"))
    # Relative paths anchor to the repo root, not the caller's CWD, so the
    # evaluation/ scripts write to the same place as root-level runs.
    return configured if configured.is_absolute() else Path(__file__).parent / configured


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

    return {
        "gate_blocked": state.get("gate_blocked", False),
        "gate_reasons": state.get("gate_reasons", []),
        "sql_insights": [_serialize_insight(i, idx) for idx, i in enumerate(state.get("sql_insights", []))],
        "agent_insights": [_serialize_insight(i, idx) for idx, i in enumerate(state.get("agent_insights", []))],
        "merged_insights": [_serialize_insight(i, idx) for idx, i in enumerate(merged)],
        "triage_result": triage_dump,
        "tiered_insights": tiered_insights,
        # Real Athena runs fed to analyze_metric_trend — surfaced so a saved
        # result can be audited for how much trend evidence actually backed it.
        "historical_runs_count": len(state.get("historical_runs", [])),
        # Which raw DB tables the local run sandbox could provide to
        # run_deterministic_review/estimate_change_saving/
        # get_advanced_config_catalog (empty -> those tools returned {}).
        "sandbox_tables": sorted(getattr(state.get("run_sandbox"), "tables", [])),
        "sql_recommendations": [
            r.model_dump() for r in state.get("sql_recommendations", [])
        ],
    }


def _build_assembled_output(plan) -> list[dict[str, Any]]:
    """The assemble_output node's product, flattened for reading.

    One entry per ActiveInsight in plan_inputs.active_insights (what
    production feeds build_plan). agent_discovered entries hold a
    PAYLOAD_RECOMMENDATIONS list — one dict per companion config change
    that shares this insight's insight_ref (see assemble_output.py's
    _agent_active_insights grouping) — each joined with the explain node's
    explanation / expected_impact / risk_note (matched by config_key) or
    the safety block reasons. SQL-rule entries keep their original insight
    payload and list every explained recommendation that traces to their
    insight_type. Both branches converge on the same per-entry shape
    (config_key/current_value/suggested_value/action/explanation/
    expected_impact/risk_note/status/blocked_by) under "recommendations"."""
    if plan is None or plan.plan_inputs is None:
        return []

    explained_by_key = {r.config_key: r for r in plan.recommendations}
    blocked_by_key = {r.config_key: r for r in plan.blocked_recommendations}

    def _explained(config_key: str | None) -> dict[str, Any]:
        blocked = blocked_by_key.get(config_key)
        rec: dict[str, Any] = {
            # Explicit disposition so a blocked entry (which carries no
            # explanation — explain only runs on safe recommendations)
            # can never read as a live recommendation with missing text.
            "status": "blocked" if blocked is not None else "recommended",
        }
        explained = explained_by_key.get(config_key)
        if explained is not None:
            rec.update(
                explanation=explained.explanation,
                expected_impact=explained.expected_impact,
                risk_note=explained.risk_note,
            )
        if blocked is not None:
            rec["blocked_by"] = blocked.blocked_by
        return rec

    assembled: list[dict[str, Any]] = []
    for insight in plan.plan_inputs.active_insights:
        payload = insight.payload or {}
        entry: dict[str, Any] = {
            "insight_type": str(insight.insight_type),
            "usd_cost_annual": insight.usd_cost_annual,
        }
        entries = payload.get(PAYLOAD_RECOMMENDATIONS)
        if entries is not None:  # agent_discovered payload contract
            entry["recommendations"] = [
                {
                    "config_key": rec_payload.get(PAYLOAD_CONFIG_KEY),
                    "current_value": rec_payload.get(PAYLOAD_CURRENT_VALUE),
                    "suggested_value": rec_payload.get(PAYLOAD_SUGGESTED_VALUE),
                    "action": rec_payload.get(PAYLOAD_ACTION),
                    **_explained(rec_payload.get(PAYLOAD_CONFIG_KEY)),
                }
                for rec_payload in entries
            ]
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
        cost = (f" (${entry['usd_cost_annual']:,.0f}/yr)"
                if entry.get("usd_cost_annual") is not None else "")
        print(f"    - [{entry['insight_type']}]{cost}")
        for rec in entry.get("recommendations", []):
            state = "BLOCKED" if rec.get("blocked_by") else "safe"
            print(f"        → {rec['config_key']}: "
                  f"{rec['current_value']} → {rec['suggested_value']} ({state})")
            if rec.get("explanation"):
                print(f"          explanation: {rec['explanation']}")
            if rec.get("expected_impact"):
                print(f"          expected_impact: {rec['expected_impact']}")
            if rec.get("risk_note"):
                print(f"          risk: {rec['risk_note']}")
            if rec.get("blocked_by"):
                print(f"          blocked_by: {'; '.join(rec['blocked_by'])}")
    print()


def _build_run_config(
    insight_rows: list[dict[str, Any]],
    *,
    dump_dir: str | None = None,
) -> dict[str, Any]:
    """Provenance block persisted with every result so two runs of the same
    task can be diffed on their actual inputs: which insight rows were
    injected (and under which filter), which model/temperature ran, and —
    for dump runs — which dump directory fed the graph."""
    from agent.inference_graph import _DEFAULT_LLM_MODEL

    return {
        "dump_dir": dump_dir,
        "insight_filter": "lifecycle_status='active' (mimics prod active_insights)",
        "injected_insights": [
            {
                "type": r.get("type"),
                "lifecycle_status": r.get("lifecycle_status"),
                "visibility": r.get("visibility"),
            }
            for r in insight_rows
        ],
        "llm_model": os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL),
        "llm_temperature": float(os.getenv("LLM_TEMPERATURE", "0")),
    }


def _save_run_result(
    plan,
    *,
    row: dict[str, Any] | None,
    source: str,
    output_dir: Path,
    trace: dict[str, Any] | None = None,
    run_config: dict[str, Any] | None = None,
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
        "run_config": run_config,
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
        "duration_s": (row or {}).get("task__duration") if row else None,
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
    run_config: dict[str, Any] | None = None,
) -> Path | None:
    if not save_results:
        return None

    path = _save_run_result(
        plan, row=row, source=source, output_dir=output_dir, trace=trace,
        run_config=run_config,
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
        duration_s = (row or {}).get("task__duration")
        print(f"  Run metrics:")
        print(f"    idle_ratio       {_fmt(m.idle_ratio, pct=True):<10}  "
              f"vcore_util   {_fmt(m.vcore_utilization, pct=True)}")
        print(f"    heap_headroom    {_fmt(m.memory_headroom, pct=True):<10}  "
              f"gc_pressure  {_fmt(m.gc_pressure, pct=True)}")
        print(f"    skew_ratio       {_fmt(m.skew_ratio, pct=True):<10}  "
              f"spill_MB     {_fmt(spill_mb, decimals=1)}")
        print(f"    duration_s       {_fmt(duration_s, decimals=0):<10}  "
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
            _fmt(row.get("task__duration") or 0, decimals=1)[:10],
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
    fetch_run_context: bool = True,
) -> tuple[dict, object]:
    from agent.inference_graph import run_analysis
    from agent.nodes.fetch_context import set_row_override
    from agent.nodes.sql_stream import set_insights_override

    task_id = int(row.get("task_id", 0))
    insight_rows = _fetch_athena_insights(
        row.get("app_id"), str(row.get("task_name")),
    )
    print(
        f"  Insights table: {len(insight_rows)} row(s) for "
        f"({row.get('task_name')}, app_id={row.get('app_id')})",
        file=sys.stderr, flush=True,
    )

    sandbox_tables = (
        _fetch_athena_sandbox_tables(task_id, row, insight_rows)
        if fetch_run_context
        else None
    )
    # TaskProfile.aqe_enabled_param reads this off the enrichment row (a
    # submitted-conf fact, not derivable from task_enrichments columns) —
    # mirrors run_from_dump.py's read_spark_parameters()/_param_value() path.
    if row.get("task__aqe_enabled__param") is None:
        row["task__aqe_enabled__param"] = _spark_param_value(
            sandbox_tables, "spark.sql.adaptive.enabled"
        )

    set_row_override(row, historical_runs, sandbox_tables)
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
        run_config=_build_run_config(insight_rows),
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
    parser.add_argument("--skip-run-context", action="store_true",
                        help="Do not query Athena for the raw per-run sandbox tables "
                             "(params/events/metrics/tests/tfs/time_series_metrics) — "
                             "run_deterministic_review/estimate_change_saving/"
                             "get_advanced_config_catalog then return {}")
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
        fetch_run_context=not args.skip_run_context,
    ))


if __name__ == "__main__":
    main()
