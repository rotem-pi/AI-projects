"""Generate SQL-rule insights for one task by running the production insight
SQLs read-only against a Postgres replica.

Usage:
    ./run.sh tools/generate_insights.py \
        --task-id 2681161 --tenant-id 63 \
        --dsn postgresql://... \
        --dump-dir data/dumps/pg_2681161 [--dump-dir data/dumps/pg_2681459]

Why this exists: production writes insights via RunInsightsAction, which runs
each backend/app/dal/sql/insights/*.sql into insights_staging and merges. A
tenant whose tenant_settings thresholds suppress generation (or a task newer
than the last sweep) has no rows, so the agent's SQL stream sees nothing.
This script replays the exact same rule SQLs against a read-only replica:

- the single `INSERT INTO insights_staging (...)` of each rule is converted
  to a plain SELECT (column names taken from the INSERT list, by position);
- `tenant_settings` is shadowed by an injected CTE (CTE names take
  precedence over tables) so min_runs/min_days/min_cost can be relaxed
  without UPDATEs — every other tenant_settings column keeps its real value;
- `insights` is shadowed the same way with the freshly generated
  task_profile rows, satisfying the rules that JOIN
  `insights WHERE type = 'task_profile'` (long_spill_time,
  over_provisioned_machine_type, orphaned_machine_vcores,
  spark_task_retries) without the merge step;
- GET_TENANT_ID() is satisfied with `SET app.tenant_id`.

Rows for the target task are written to <dump-dir>/insights.json in the
shape tools/dump_postgres.py produces (insight_id is synthetic-negative to
mark generated rows; lifecycle_status='active' as _merge_staging would set).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from bootstrap_worktree import resolved_backend_path  # noqa: E402

# Dedicated worktree pinned to auto-recommendations-agent (see bootstrap_worktree.py)
SQL_DIR = resolved_backend_path() / "app" / "dal" / "sql" / "insights"

# Mirrors run_insights_action.py: task_profile first (downstream rules JOIN
# against it), then insights_to_run in the same order.
DEPENDENCY_INSIGHT = "task_profile"
INSIGHTS_TO_RUN = [
    "long_gc_time",
    "long_idle_time",
    "long_skew_time",
    "long_spill_time",
    "over_provisioned_driver_heap",
    "over_provisioned_driver_cores",
    "over_provisioned_executor_heap",
    "over_provisioned_executors",
    "small_files",
    "task_retries",
    "under_utilized_executors_cpu",
    "spark_task_retries",
    "over_provisioned_executor_off_heap",
    "s3_file_over_listing",
    "orphaned_machine_vcores",
    "orphaned_node_resources_databricks",
    "orphaned_node_resources_yarn_cloud",
    "over_provisioned_machine_type",
    "captured_cost_savings",
    "over_provisioned_driver_machine",
]

# SystemSettings defaults (models/system_settings_models.py) — the values
# RunInsightsAction passes as sql_params.
SQL_PARAMS = {
    "days_back": "30 days",
    "days_back_num": 30,
    "exp_decay": 3,
    "max_runs_per_task": 100,
}

_INSERT_RE = re.compile(
    r"INSERT\s+INTO\s+insights_staging\s*\(([^)]*)\)", re.IGNORECASE
)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _convert_to_select(sql: str) -> tuple[str, list[str]]:
    """Strip the rule's single INSERT INTO insights_staging (...) so the file
    runs as a plain SELECT; return (sql, insert column list) — output columns
    map onto the INSERT list by position."""
    match = _INSERT_RE.search(sql)
    if not match:
        raise ValueError("no INSERT INTO insights_staging found")
    columns = [c.strip() for c in match.group(1).split(",")]
    return sql[: match.start()] + sql[match.end():], columns


def _inject_shadow_ctes(sql: str, shadow_ctes: str) -> str:
    """Prepend shadow CTEs to the rule's WITH clause (or add one). The first
    top-of-line WITH outside comments is the query's own; rules are single
    statements so this is unambiguous."""
    match = re.search(r"^\s*WITH\b", sql, re.MULTILINE)
    if match:
        return sql[: match.end()] + " " + shadow_ctes + "," + sql[match.end():]
    match = re.search(r"^\s*\(?\s*SELECT\b", sql, re.MULTILINE)
    if not match:
        raise ValueError("neither WITH nor SELECT found")
    return sql[: match.start()] + "WITH " + shadow_ctes + "\n" + sql[match.start():]


def _tenant_settings_shadow(cur) -> str:
    """A CTE named tenant_settings selecting the real table with the three
    generation thresholds overridden (CTE bodies still see the real table)."""
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_name = 'tenant_settings' ORDER BY ordinal_position
    """)
    overridden = {
        "min_runs_for_insight": "%(relax_min_runs)s::INT",
        "min_days_for_insight": "%(relax_min_days)s::INT",
        "min_cost_for_insight": "%(relax_min_cost)s::NUMERIC",
    }
    cols = ", ".join(
        f"{overridden[name]} AS {name}" if name in overridden else name
        for (name,) in cur.fetchall()
    )
    return f"tenant_settings AS (SELECT {cols} FROM tenant_settings)"


_INSIGHTS_SHADOW = """insights AS (
  SELECT (e ->> 'app_id')::INT   AS app_id,
         e ->> 'task_name'       AS task_name,
         'task_profile'          AS type,
         e -> 'insights_payload' AS insights_payload
  FROM JSONB_ARRAY_ELEMENTS(%(task_profile_rows)s::JSONB) AS e
)"""


def _run_rule(cur, name: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    sql, columns = _convert_to_select((SQL_DIR / f"{name}.sql").read_text())
    shadows = _tenant_settings_shadow(cur) + ", " + _INSIGHTS_SHADOW
    cur.execute(_inject_shadow_ctes(sql, shadows), params)
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _to_dump_row(staged: dict[str, Any], synthetic_id: int) -> dict[str, Any]:
    """Shape a staged row like tools/dump_postgres.py's insights.json rows
    (what _merge_staging would have produced, minus DB-assigned fields)."""
    return {
        "insight_id": synthetic_id,
        "task_name": staged["task_name"],
        "type": staged["type"],
        "impact_cost": staged.get("impact_cost"),
        "impact_unit": staged.get("impact_unit"),
        "insights_payload": staged.get("insights_payload"),
        "insights_payload_override": None,
        "lifecycle_status": "active",
        "visibility": staged.get("visibility") or "visible",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task-id", type=int, required=True,
                        help="Any run of the logical task (resolves app_id + task_name)")
    parser.add_argument("--tenant-id", type=int, required=True)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--dump-dir", action="append", type=Path, default=[],
                        help="Dump dir(s) whose insights.json should receive the rows")
    parser.add_argument("--min-runs", type=int, default=3,
                        help="Relaxed min_runs_for_insight (production default 3)")
    parser.add_argument("--min-days", type=int, default=0,
                        help="Relaxed min_days_for_insight (0 = allow brand-new tasks)")
    parser.add_argument("--min-cost", type=float, default=1000,
                        help="Relaxed min_cost_for_insight in $ (production default 1000)")
    args = parser.parse_args()

    import psycopg

    with psycopg.connect(args.dsn, connect_timeout=15) as conn, conn.cursor() as cur:
        cur.execute(f"SET app.tenant_id = '{int(args.tenant_id)}'")
        cur.execute("SET statement_timeout = '300s'")

        cur.execute("SELECT app_id, task_name FROM tasks WHERE task_id = %s", (args.task_id,))
        target = cur.fetchone()
        if target is None:
            raise SystemExit(f"task_id {args.task_id} not found")
        app_id, task_name = target
        print(f"Target: app_id={app_id}, task_name={task_name!r}, tenant={args.tenant_id}")
        print(f"Thresholds: min_runs={args.min_runs}, min_days={args.min_days}, "
              f"min_cost=${args.min_cost:,.0f}\n")

        params: dict[str, Any] = {
            **SQL_PARAMS,
            "relax_min_runs": args.min_runs,
            "relax_min_days": args.min_days,
            "relax_min_cost": args.min_cost,
            "task_profile_rows": "[]",
        }

        profile_rows = _run_rule(cur, DEPENDENCY_INSIGHT, params)
        app_profiles = [r for r in profile_rows if r["app_id"] == app_id]
        params["task_profile_rows"] = json.dumps(
            [{"app_id": r["app_id"], "task_name": r["task_name"],
              "insights_payload": r["insights_payload"]} for r in app_profiles],
            default=_jsonable,
        )
        print(f"  task_profile: {len(app_profiles)} row(s) for app {app_id} "
              f"(of {len(profile_rows)} tenant-wide)")

        task_rows = [r for r in profile_rows
                     if r["app_id"] == app_id and r["task_name"] == task_name]
        for rule in INSIGHTS_TO_RUN:
            try:
                rows = _run_rule(cur, rule, params)
            except Exception as exc:
                conn.rollback()
                cur.execute(f"SET app.tenant_id = '{int(args.tenant_id)}'")
                cur.execute("SET statement_timeout = '300s'")
                print(f"  {rule}: FAILED — {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            hits = [r for r in rows if r["app_id"] == app_id and r["task_name"] == task_name]
            task_rows.extend(hits)
            print(f"  {rule}: {len(hits)} row(s) for the task ({len(rows)} total)")

    generated = [_to_dump_row(r, -(i + 1)) for i, r in enumerate(task_rows)]
    print(f"\nGenerated {len(generated)} insight row(s) for ({app_id}, {task_name!r}):")
    for row in generated:
        cost = f" impact={row['impact_cost']:.0f} {row['impact_unit']}" if row["impact_cost"] else ""
        print(f"  - {row['type']} [{row['visibility']}]{cost}")

    for dump_dir in args.dump_dir:
        path = dump_dir / "insights.json"
        path.write_text(json.dumps(generated, indent=2, default=_jsonable), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
