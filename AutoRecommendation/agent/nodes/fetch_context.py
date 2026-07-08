"""Fetch-context node — reads a task_enrichments row injected via set_row_override().

The row, its real historical runs, and (when available) the raw per-run
context are fetched from Athena by main.py before the graph is invoked and
passed in via set_row_override(). This mirrors
backend/app/brain/insights/agent/nodes/context.py — same
run_metrics/task_profile computation, same "historical runs for trend
analysis" contract, same run_context sections — the only difference is the
data source (Athena export vs. Postgres via fetch_sql_file).

agent_historical_enrichments.sql adds a computed run_cost_usd column
(CALCULATE_USD_COST over vcore/memory time) that the terminal node uses for
the pre-tuning cost baseline; Athena rows don't carry it, so the same
formula is applied here to every historical run before it enters the state.
"""

from __future__ import annotations

import logging
import os

from app.brain.insights.agent.models import DbRow, RunContext
from app.brain.insights.agent.state import AgentState
from app.brain.insights.tuning.run_metrics import compute_run_metrics
from app.brain.insights.tuning.task_profile import compute_task_profile

from agent.cost_utils import DEFAULT_MEMORY_PRICE, DEFAULT_VCORE_PRICE

logger = logging.getLogger(__name__)

# Module-level overrides: set before invoking the graph (set by main.py's batch mode).
_ROW_OVERRIDE: DbRow | None = None
_HISTORICAL_RUNS_OVERRIDE: list[DbRow] = []
_RUN_CONTEXT_OVERRIDE: RunContext | None = None


def set_row_override(
    row: DbRow | None,
    historical_runs: list[DbRow] | None = None,
    run_context: RunContext | None = None,
) -> None:
    global _ROW_OVERRIDE, _HISTORICAL_RUNS_OVERRIDE, _RUN_CONTEXT_OVERRIDE
    _ROW_OVERRIDE = row
    _HISTORICAL_RUNS_OVERRIDE = historical_runs or []
    _RUN_CONTEXT_OVERRIDE = run_context


def _with_run_cost_usd(run: DbRow) -> DbRow:
    """Mirror agent_historical_enrichments.sql's run_cost_usd column:
    CALCULATE_USD_COST('VCore', vcore_time_allocated)
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
            "run_context": None,
        }

    run_metrics = compute_run_metrics(enrichment)
    task_profile = compute_task_profile(enrichment)

    return {
        "latest_enrichment": enrichment,
        "run_metrics": run_metrics,
        "task_profile": task_profile,
        "historical_runs": [_with_run_cost_usd(r) for r in _HISTORICAL_RUNS_OVERRIDE],
        "run_context": _RUN_CONTEXT_OVERRIDE,
    }
