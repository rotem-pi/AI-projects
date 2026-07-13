"""SQL stream node — real insights from Athena.

Mirrors backend/app/brain/insights/agent/nodes/sql_stream.py: production
fetches the task's rows via insights.sql (Postgres) and builds SqlInsight
objects straight from the row dicts, then extracts engine recommendations
with the same extract_recommendations dispatcher.

Here main.py fetches the task's rows from the Athena `insights` table
(latest snapshot, active+visible by default — the same filters insights.sql
applies) and injects them via set_insights_override().  Each Athena row is
mapped to the insights.sql output shape first — usd_cost computed the way
CALCULATE_USD_COST does, insights_payload parsed to a dict, rows ordered by
usd_cost DESC — so SqlInsight(**row) sees exactly what production sees.
"""

from __future__ import annotations

import json
import logging
import os

from app.brain.insights.agent.enums import InsightSource, RecommendationAction
from app.brain.insights.agent.models import DbRow, Recommendation, SqlInsight
from app.brain.insights.agent.state import AgentState
from app.brain.insights.recommendations import extract_recommendations
from app.brain.insights.recommendations.protocol import (
    Recommendation as TuningRecommendation,
)
from app.brain.insights.tuning.task_profile import compute_task_profile

from agent.cost_utils import DEFAULT_MEMORY_PRICE, DEFAULT_VCORE_PRICE

logger = logging.getLogger(__name__)

# Module-level override: set before invoking the graph (set by main.py's
# batch mode, same pattern as fetch_context.set_row_override).  None means
# "no Athena insights were fetched"; a list (possibly empty) means "these
# are the task's real insights".
_INSIGHTS_OVERRIDE: list[dict] | None = None


def set_insights_override(rows: list[dict] | None) -> None:
    global _INSIGHTS_OVERRIDE
    _INSIGHTS_OVERRIDE = rows


def _impact_usd_cost(impact_unit: str | None, impact_cost: float | None) -> float | None:
    """Python port of CALCULATE_USD_COST for the units the agent meets.

    impact_cost is annualized by the insight rules (VCore-seconds or
    GB-seconds per year), so the result is annual USD.
    """
    if impact_unit is None or impact_cost is None:
        return None
    vcore_price = float(os.getenv("VCORE_PRICE", DEFAULT_VCORE_PRICE))
    memory_price = float(os.getenv("MEMORY_PRICE", DEFAULT_MEMORY_PRICE))
    if impact_unit == "VCore":
        return impact_cost * vcore_price / 3600
    if impact_unit == "GB":
        return impact_cost * memory_price / 3600
    if impact_unit == "dollar":
        return float(impact_cost)
    logger.warning("Unpriced impact_unit %r — usd_cost left unset", impact_unit)
    return None


def _parse_payload(raw: object) -> dict | None:
    payload: dict | None = None
    if isinstance(raw, dict):
        payload = raw
    elif isinstance(raw, str) and raw.strip():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("insights_payload is not valid JSON: %.120r", raw)
    if payload is None:
        return None
    # The Athena snapshot enriches payloads with nested display metadata
    # (e.g. a task_metrics list) that Postgres payloads don't carry;
    # SqlInsight.insights_payload is a flat scalar dict and the rule engine
    # only reads scalar keys, so nested values are dropped.
    dropped = [k for k, v in payload.items() if isinstance(v, (dict, list))]
    if dropped:
        logger.debug("Dropped nested insights_payload key(s): %s", dropped)
    return {k: v for k, v in payload.items() if not isinstance(v, (dict, list))}


def _describe(insight_type: str, usd_cost: float | None) -> str:
    """insights.sql joins insights_metadata_v for a human description; the
    Athena export has no such column, so a compact equivalent is built."""
    if usd_cost is not None:
        return f"{insight_type} — estimated annual saving opportunity ${usd_cost:,.0f}"
    return insight_type


def _to_insights_sql_row(row: dict) -> dict:
    """Map an Athena `insights` row to the column set insights.sql returns,
    so SqlInsight(**row) carries the same fields (id stays None and
    usd_cost/insight_id land in model_extra) as in production."""
    usd_cost = _impact_usd_cost(row.get("impact_unit"), row.get("impact_cost"))
    return {
        "insight_id": row.get("insight_id"),
        "type": row["type"],
        "description": _describe(row["type"], usd_cost),
        "insights_payload": _parse_payload(row.get("insights_payload")),
        "usd_cost": usd_cost,
        "impact_cost": row.get("impact_cost"),
        "impact_unit": row.get("impact_unit"),
        "task_name": row.get("task_name"),
        "lifecycle_status": row.get("lifecycle_status"),
        "visibility": row.get("visibility"),
    }


def _build_recommendation(
    rec: TuningRecommendation, insight: SqlInsight
) -> Recommendation:
    return Recommendation(
        config_key=rec.config_key,
        action=RecommendationAction(rec.action.value),
        suggested_value=rec.suggested_value,
        current_value=rec.current_value,
        source=InsightSource.SQL_RULE,
        source_detail=insight.type,
        confidence=1.0,
        priority=0,
        rationale=rec.reason or "",
        insight_type=insight.type,
        insight_id=insight.id,
    )


def run_sql_stream(state: AgentState) -> dict:
    if _INSIGHTS_OVERRIDE is None:
        logger.warning(
            "No insights override set for task_id=%s — the SQL stream will "
            "report zero insights; call set_insights_override() before "
            "invoking the graph.",
            state.get("task_id"),
        )
        return {"sql_insights": [], "sql_recommendations": []}

    enrichment: DbRow | None = state.get("latest_enrichment")
    task_profile = compute_task_profile(enrichment) if enrichment else None

    mapped = [_to_insights_sql_row(row) for row in _INSIGHTS_OVERRIDE]
    # insights.sql orders by usd_cost DESC with NULLs last.
    mapped.sort(key=lambda r: (r["usd_cost"] is None, -(r["usd_cost"] or 0.0)))
    insights = [SqlInsight(**row) for row in mapped]

    recommendations: list[Recommendation] = []
    for insight in insights:
        raw_recs = extract_recommendations(
            insight_type=insight.type,
            payload=insight.insights_payload or {},
            task_profile=task_profile,
        )
        recommendations.extend(_build_recommendation(raw, insight) for raw in raw_recs)

    return {
        "sql_insights": insights,
        "sql_recommendations": recommendations,
    }


# Backwards-compatible alias (pg_nodes_runner / older callers).
sql_stream = run_sql_stream
