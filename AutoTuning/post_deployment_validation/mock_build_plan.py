#!/usr/bin/env python3
"""Run build_plan (backend/app/brain/insights/tuning/api/build_plan.py) for a
single task_id against the real Postgres insights/task_enrichments tables.

Mirrors app/brain/insights/tuning/preview_service.py::build_preview_plan, but
keyed by task_id instead of (app_id, task_name) so it can be pointed at any
task directly: task_id -> tasks row -> (app_id, task_name) -> the same DAL
queries production's preview endpoint uses (tuning/latest_enrichments.sql,
tuning/active_insights_for_task.sql).

Usage:
    export DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/app
    python scripts/mock_build_plan.py <task_id>
    python scripts/mock_build_plan.py <task_id> --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

import psycopg2
import psycopg2.extras

sys.path.insert(0, str(Path(__file__).parent))
from paths import definity_backend_root  # noqa: E402

BACKEND_ROOT = definity_backend_root()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.brain.insights.tuning.api import PlanInputs, build_plan  # noqa: E402
from app.brain.insights.tuning.entities import ActiveInsight, TuningConstraints  # noqa: E402
from app.brain.insights.tuning.run_metrics import TaskRunMetrics  # noqa: E402
from app.brain.insights.tuning.task_profile import TaskProfile  # noqa: E402

# 1 latest + TARGET prior, matching preview_service.py's baseline depth.
TARGET_BASELINE_RUNS = 5
MIN_BASELINE_RUNS = 2

# task__duration is stored in seconds (EXTRACT(EPOCH FROM ...) in
# task_run_enrichment.sql); pre_tuning_durations_ms must be in ms, matching
# preview_service.py's SECONDS_TO_MILISECONDS conversion.
SECONDS_TO_MILISECONDS = 1_000.0

_TASK_SQL = """
SELECT t.app_id, t.task_name, a.app_name, e.tenant_id, e.env_name
FROM tasks t
JOIN apps a ON a.app_id = t.app_id
JOIN envs e ON e.env_id = a.env_id
WHERE t.task_id = %(task_id)s
"""

# Same shape as backend/app/dal/sql/tuning/latest_enrichments.sql.
_LATEST_ENRICHMENTS_SQL = """
SELECT
  te.*,
  t.start_time,
  COALESCE(CALCULATE_USD_COST('VCore'::TEXT, te.task__vcore_time__allocated), 0)
  + COALESCE(CALCULATE_USD_COST('GB'::TEXT, te.task__memory_time__allocated), 0)
    AS run_cost_usd
FROM task_enrichments AS te
JOIN tasks AS t USING (task_id)
WHERE t.app_id = %(app_id)s
  AND t.task_name = %(task_name)s
  AND te.enrichments_updated_at IS NOT NULL
ORDER BY t.start_time DESC
LIMIT %(limit)s
"""

# Same shape as backend/app/dal/sql/tuning/active_insights_for_task.sql.
_ACTIVE_INSIGHTS_SQL = """
SELECT
  type AS insight_type,
  insights_payload || COALESCE(manual_overrides, '{}'::JSONB) AS insights_payload,
  CALCULATE_USD_COST(impact_unit, impact_cost) AS usd_cost_annual
FROM insights
WHERE app_id = %(app_id)s
  AND task_name = %(task_name)s
  AND lifecycle_status = 'active'
"""


def _connect():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        raise SystemExit("DATABASE_URL is not set (see .env.example)")
    return psycopg2.connect(db_url)


def _fetch_task(conn, task_id: int) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_TASK_SQL, {"task_id": task_id})
        row = cur.fetchone()
    if row is None:
        raise SystemExit(f"No task found for task_id={task_id}")
    return dict(row)


def _set_tenant_context(conn, tenant_id: int) -> None:
    """CALCULATE_USD_COST (used by latest_enrichments/active_insights) reads
    app.tenant_id from the session GUC, same as production request scoping."""
    with conn.cursor() as cur:
        cur.execute("SELECT set_config('app.tenant_id', %s, false)", (str(tenant_id),))


def _fetch_latest_enrichments(conn, app_id: int, task_name: str, limit: int) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            _LATEST_ENRICHMENTS_SQL,
            {"app_id": app_id, "task_name": task_name, "limit": limit},
        )
        return [dict(row) for row in cur.fetchall()]


def _fetch_active_insights(conn, app_id: int, task_name: str) -> list[ActiveInsight]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(_ACTIVE_INSIGHTS_SQL, {"app_id": app_id, "task_name": task_name})
        rows = [dict(row) for row in cur.fetchall()]
    return [
        ActiveInsight(
            insight_type=row["insight_type"],
            payload=row["insights_payload"],
            usd_cost_annual=(
                float(row["usd_cost_annual"]) if row["usd_cost_annual"] is not None else None
            ),
        )
        for row in rows
    ]


def _extract_durations(rows: list[dict[str, Any]]) -> list[float]:
    return [
        float(row["task__duration"]) * SECONDS_TO_MILISECONDS
        for row in reversed(rows)
        if row.get("task__duration") is not None
    ]


def _mean_cost(rows: list[dict[str, Any]]) -> float | None:
    costs = [float(row["run_cost_usd"]) for row in rows if row.get("run_cost_usd") is not None]
    return mean(costs) if costs else None


def mock_build_plan_with_meta(task_id: int, conn=None):
    """Return (plan, task_meta) for task_id, where task_meta has tenant_id,
    env_name, app_name, task_name. Reuses `conn` if given, else opens/closes its own."""
    owns_conn = conn is None
    conn = conn or _connect()
    try:
        task = _fetch_task(conn, task_id)
        app_id, task_name = task["app_id"], task["task_name"]
        _set_tenant_context(conn, task["tenant_id"])

        rows = _fetch_latest_enrichments(conn, app_id, task_name, TARGET_BASELINE_RUNS + 1)
        if not rows:
            raise SystemExit(
                f"No enriched runs found for task_id={task_id} "
                f"(app_id={app_id}, task_name={task_name!r})"
            )

        latest, baseline_rows = rows[0], rows[1:]
        if len(baseline_rows) < MIN_BASELINE_RUNS:
            print(
                f"Warning: only {len(baseline_rows)} prior enriched run(s); "
                f"regression baseline needs >= {MIN_BASELINE_RUNS}.",
                file=sys.stderr,
            )

        active_insights = _fetch_active_insights(conn, app_id, task_name)
        print(
            f"  Task {task_id} ({task_name}, app_id={app_id})\n"
            f"  Active insights: {[i.insight_type for i in active_insights]}\n"
            f"  Baseline runs: {len(baseline_rows)}",
            file=sys.stderr,
        )

        plan = build_plan(
            PlanInputs(
                active_insights=active_insights,
                run_metrics=TaskRunMetrics.from_enrichment_row(latest),
                task_profile=TaskProfile.from_enrichment_row(latest),
                pre_tuning_durations_ms=_extract_durations(baseline_rows),
                pre_tuning_cost_usd=_mean_cost(baseline_rows),
                constraints=TuningConstraints(),
            )
        )
        return plan, task
    finally:
        if owns_conn:
            conn.close()


def mock_build_plan(task_id: int):
    plan, _task = mock_build_plan_with_meta(task_id)
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task_id", type=int)
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    plan = mock_build_plan(args.task_id)
    if args.json:
        print(json.dumps(plan.model_dump(mode="json"), indent=2))
    else:
        for knob in plan.knobs:
            spec = knob.spec
            print(f"  [{spec.priority_score:.3f}] {knob.phase} {spec.config_key}: "
                  f"{spec.initial_value} -> {spec.target_value} (next={knob.next_value}) "
                  f"({spec.insight_type})")
        if plan.blocked_knobs:
            print("\nBlocked:")
            for knob in plan.blocked_knobs:
                print(f"  {knob.spec.config_key} ({knob.spec.insight_type}): {knob.spec.block_reason}")


if __name__ == "__main__":
    main()
