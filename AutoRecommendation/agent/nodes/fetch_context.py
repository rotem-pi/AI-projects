"""Fetch-context node — reads a task_enrichments row injected via set_row_override().

The row, its real historical runs, and (when available) a local run-sandbox
built from raw-ish per-run tables are fetched by main.py/run_from_dump.py/
run_from_pg_dump.py before the graph is invoked and passed in via
set_row_override(). This mirrors backend/app/brain/insights/agent/nodes/fetch_context.py
— same run_metrics/task_profile computation, same "historical runs for trend
analysis" contract — the only difference is the data source (Athena export
vs. Postgres) and how the run sandbox is materialized: the repo node dumps
raw tables to S3 (this harness must not touch S3 for now); here, the entry
point builds a {table: [row-dict,...]} mapping and agent/local_sandbox.py's
build_run_sandbox() constructs a real RunSandbox pointing at it, materialized
to a local temp folder on tool demand instead of downloaded from S3
(agent/local_sandbox.py's patch of sandbox_dal.materialize_to_folder). This
harness is single-PIT only: the tables hold this one run's rows, not
multi-run history — no config-change-regime detection across runs. No
run-id resolution is needed here (main.py already injects the run to
analyze).

The historical_enrichments query (repo: dal/sql/insights/agent_context.sql)
adds a computed run_cost_usd column (CALCULATE_USD_COST over vcore/memory
time) that the terminal node uses for the pre-tuning cost baseline; Athena
rows don't carry it, so the same formula is applied here to every
historical run before it enters the state.
"""

from __future__ import annotations

import logging
import os

from app.brain.insights.agent.models import DbRow, RunSandbox
from app.brain.insights.agent.state import AgentState
from app.brain.insights.tuning.entities.run_metrics import compute_run_metrics
from app.brain.insights.tuning.entities.task_profile import compute_task_profile

from agent.cost_utils import DEFAULT_MEMORY_PRICE, DEFAULT_VCORE_PRICE
from agent.local_sandbox import build_run_sandbox

logger = logging.getLogger(__name__)

# Module-level overrides: set before invoking the graph (set by main.py's batch mode).
_ROW_OVERRIDE: DbRow | None = None
_HISTORICAL_RUNS_OVERRIDE: list[DbRow] = []
_RUN_SANDBOX_OVERRIDE: RunSandbox | None = None


def set_row_override(
    row: DbRow | None,
    historical_runs: list[DbRow] | None = None,
    sandbox_tables: dict[str, list[dict]] | None = None,
) -> None:
    """sandbox_tables maps raw-DB-table name (e.g. "tasks", "task_params",
    "events", "insights") to this run's row-dicts for that table — whatever
    the calling entry point can populate; omitted tables degrade gracefully
    (kb/analysis/store.py's build_from_csv and Store tolerate missing
    tables)."""
    global _ROW_OVERRIDE, _HISTORICAL_RUNS_OVERRIDE, _RUN_SANDBOX_OVERRIDE
    _ROW_OVERRIDE = row
    _HISTORICAL_RUNS_OVERRIDE = historical_runs or []
    _RUN_SANDBOX_OVERRIDE = (
        build_run_sandbox(int(row["task_id"]), sandbox_tables)
        if row is not None and sandbox_tables
        else None
    )


def _with_run_cost_usd(run: DbRow) -> DbRow:
    """Mirror agent_context.sql's (@query: historical_enrichments)
    run_cost_usd column: CALCULATE_USD_COST('VCore', vcore_time_allocated)
    + CALCULATE_USD_COST('GB', memory_time_allocated)."""
    if run.get("run_cost_usd") is not None:
        return run
    vcore_price = float(os.getenv("VCORE_PRICE", DEFAULT_VCORE_PRICE))
    memory_price = float(os.getenv("MEMORY_PRICE", DEFAULT_MEMORY_PRICE))
    vcore_seconds = run.get("task__vcore_time__allocated")
    gb_seconds = run.get("task__memory_time__allocated")
    cost = 0.0
    if isinstance(vcore_seconds, (int, float)):
        cost += vcore_seconds * vcore_price / 3600
    if isinstance(gb_seconds, (int, float)):
        cost += gb_seconds * memory_price / 3600
    return {**run, "run_cost_usd": cost}


def _with_start_time(run: DbRow) -> DbRow:
    """compute_annual_task_cost reads run["start_time"] to derive runs_per_year;
    Athena's task_enrichments exposes the same point-in-time as app_pit."""
    if run.get("start_time") is not None:
        return run
    app_pit = run.get("app_pit")
    if app_pit is None:
        return run
    return {**run, "start_time": app_pit}


def fetch_context(state: AgentState) -> dict:
    task_id = state["task_id"]
    enrichment = _ROW_OVERRIDE

    if enrichment is None:
        logger.error(
            "No row override set for task_id=%s — call set_row_override() before invoking the graph",
            task_id,
        )
        return {
            "latest_enrichment": None,
            "run_metrics": None,
            "task_profile": None,
            "historical_runs": [],
            "run_sandbox": None,
        }

    run_metrics = compute_run_metrics(enrichment)
    task_profile = compute_task_profile(enrichment)

    return {
        "latest_enrichment": enrichment,
        "run_metrics": run_metrics,
        "task_profile": task_profile,
        "historical_runs": [
            _with_start_time(_with_run_cost_usd(r)) for r in _HISTORICAL_RUNS_OVERRIDE
        ],
        "run_sandbox": _RUN_SANDBOX_OVERRIDE,
    }
