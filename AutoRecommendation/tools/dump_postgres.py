"""Dump everything the agent needs for one task from a Postgres replica.

Usage:
    uv run --project ../definity-app/backend python tools/dump_postgres.py 2681247 \
        --dsn postgresql://user:pass@host:5432/app

    ->  data/dumps/pg_2681247/
          enrichment.json        the task_enrichments row (+ task/app identity)
          historical_runs.json   up to --history-limit prior runs, oldest first
          insights.json          all insights rows for the logical task
          sandbox_tables.json    raw per-table rows for this single run (one
                                  key per SANDBOX_DIRECT_TABLES/
                                  SANDBOX_TF_KEYED_TABLES/"insights" table),
                                  the local-sandbox equivalent of production's
                                  S3 run-sandbox CSVs (single-PIT only — this
                                  run's rows, not multi-run history)

Unlike tools/dump-rest-api.sh (REST-shaped CSVs that run_from_dump.py has to
map back onto enrichment columns), this dumps the enrichment row directly, so
run_from_pg_dump.py replays it with no metric-name translation and with real
historical runs (trend analysis works).

The DSN comes from --dsn or the PG_DSN environment variable (.env is read).
The enrichment/history/insights queries mirror
backend/app/dal/sql/insights/agent_context.sql, minus tenant-session
functions (GET_TENANT_ID, CALCULATE_USD_COST) which need an app session —
task_id is globally unique and run_cost_usd is recomputed from env prices by
the harness. The sandbox-table queries are plain `SELECT * FROM <table>
WHERE task_id = ...` (or the tf-keyed join for tf_inputs/tfs_query_vars),
mirroring backend/app/brain/insights/agent/nodes/fetch_context.py's
_dump_direct_table/_dump_tf_keyed_table/_dump_insights against the real
schema instead of the old pre-joined run_context sections.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")

# Row cap per sandbox table (mirrors the old run-context section cap).
ROWS_PER_SECTION_LIMIT = 300
DEFAULT_HISTORY_LIMIT = 20  # HISTORICAL_RUNS_FETCH_LIMIT

_ENRICHMENT_SELECT = """
    SELECT te.*,
           t.task_name, t.app_id, t.start_time, t.parent_task_id, t.status,
           a.app_name
    FROM task_enrichments te
    INNER JOIN tasks t USING (task_id)
    INNER JOIN apps a ON t.app_id = a.app_id
"""


def _jsonable(value: Any) -> Any:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def _fetch(cur, sql: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    cur.execute(sql, params or {})
    columns = [d.name for d in cur.description]
    return [dict(zip(columns, row)) for row in cur.fetchall()]


def _fetch_enrichment(cur, task_id: int) -> dict[str, Any]:
    rows = _fetch(cur, _ENRICHMENT_SELECT + "WHERE te.task_id = %(task_id)s", {"task_id": task_id})
    if rows:
        return rows[0]

    # Same fallback as main.py's Athena path: the logical task's latest run.
    print(f"  task_id {task_id} has no enrichment row — falling back to the "
          "logical task's latest enriched run.", file=sys.stderr)
    rows = _fetch(cur, _ENRICHMENT_SELECT + """
        WHERE (t.app_id, t.task_name) = (
            SELECT app_id, task_name FROM tasks WHERE task_id = %(task_id)s
        )
        ORDER BY t.start_time DESC
        LIMIT 1
    """, {"task_id": task_id})
    if not rows:
        raise SystemExit(f"No enrichment row found for task_id={task_id} (nor for its logical task)")
    return rows[0]


def _fetch_historical_runs(cur, task_id: int, limit: int) -> list[dict[str, Any]]:
    """Prior runs of the same logical task, oldest -> newest (fetch_context's
    documented convention). Mirrors agent_context.sql's historical_enrichments
    but keeps te.* — a superset of the production column list — and leaves
    run_cost_usd to the harness (env-price formula in fetch_context)."""
    rows = _fetch(cur, _ENRICHMENT_SELECT + """
        WHERE (t.app_id, t.task_name) = (
            SELECT app_id, task_name FROM tasks WHERE task_id = %(task_id)s
        )
          AND t.task_id != %(analyzed_task_id)s
          AND t.status IN ('COMPLETED', 'FAILED')
        ORDER BY t.start_time DESC
        LIMIT %(limit)s
    """, {"task_id": task_id, "analyzed_task_id": task_id, "limit": limit})
    return list(reversed(rows))


def _fetch_insights(cur, task_id: int) -> list[dict[str, Any]]:
    """Every insights row for the logical task — lifecycle/visibility filters
    are applied at replay time so one dump serves both default and
    --include-hidden-insights runs."""
    return _fetch(cur, """
        SELECT insight_id, task_name, type, impact_cost, impact_unit,
               insights_payload, insights_payload_override, lifecycle_status,
               visibility
        FROM insights
        WHERE (app_id, task_name) = (
            SELECT app_id, task_name FROM tasks WHERE task_id = %(task_id)s
        )
    """, {"task_id": task_id})


def _unwrap_param_value(value: Any) -> Any:
    """task_params.value may be a JSON blob ({"value": ...}); keep the inner
    value — the shape resolve_current_config_value expects."""
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    if isinstance(value, str):
        try:
            blob = json.loads(value)
            if isinstance(blob, dict) and "value" in blob:
                return blob["value"]
        except ValueError:
            pass
    return value


# Raw DB tables dumped verbatim for the local run-sandbox, mirroring
# backend/app/brain/insights/agent/constants.py's SANDBOX_DIRECT_TABLES /
# SANDBOX_TF_KEYED_TABLES (task_enrichments is the enrichment row itself,
# already fetched separately — wrapped as a one-row table below).
_SANDBOX_DIRECT_TABLES = (
    "tasks",
    "task_params",
    "events",
    "metrics",
    "test_runs",
    "tfs",
    "time_series_metrics",
)
_SANDBOX_TF_KEYED_TABLES = ("tf_inputs", "tfs_query_vars")


def _fetch_direct_table(cur, task_id: int, table: str) -> list[dict[str, Any]]:
    return _fetch(cur, f"SELECT * FROM {table} WHERE task_id = %(tid)s LIMIT %(rows)s",
                   {"tid": task_id, "rows": ROWS_PER_SECTION_LIMIT})


def _fetch_tf_keyed_table(cur, task_id: int, table: str) -> list[dict[str, Any]]:
    """No task_id column on these tables — reached by hopping through the
    task's tfs rows (tf_id), mirroring fetch_context.py's
    sandbox_tf_table_dump query."""
    return _fetch(cur, f"""
        SELECT x.* FROM {table} x
        INNER JOIN tfs tf ON x.tf_id = tf.tf_id
        WHERE tf.task_id = %(tid)s
        LIMIT %(rows)s
    """, {"tid": task_id, "rows": ROWS_PER_SECTION_LIMIT})


def _build_sandbox_tables(
    cur, task_id: int, enrichment: dict[str, Any], insights: list[dict[str, Any]]
) -> dict[str, list[dict[str, Any]]]:
    """Raw per-table rows for this single run — the local-sandbox equivalent
    of production's S3 CSV dump (fetch_context.py's _build_run_sandbox),
    single-PIT only (this run's rows, no multi-run history)."""
    tables: dict[str, list[dict[str, Any]]] = {"task_enrichments": [enrichment]}
    for table in _SANDBOX_DIRECT_TABLES:
        try:
            rows = _fetch_direct_table(cur, task_id, table)
        except Exception as exc:
            print(f"  sandbox table {table!r} failed ({exc}) — skipped", file=sys.stderr)
            cur.connection.rollback()
            continue
        if table == "task_params":
            rows = [{**r, "value": _unwrap_param_value(r["value"])} for r in rows]
        if rows:
            tables[table] = rows
    for table in _SANDBOX_TF_KEYED_TABLES:
        try:
            rows = _fetch_tf_keyed_table(cur, task_id, table)
        except Exception as exc:
            print(f"  sandbox table {table!r} failed ({exc}) — skipped", file=sys.stderr)
            cur.connection.rollback()
            continue
        if rows:
            tables[table] = rows
    if insights:
        tables["insights"] = insights
    return tables


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one task from Postgres for run_from_pg_dump.py")
    parser.add_argument("task_id", type=int)
    parser.add_argument("--dsn", default=os.getenv("PG_DSN"),
                        help="postgresql://user:pass@host:port/db (or PG_DSN in .env)")
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Write here instead of data/dumps/pg_<task_id> "
                             "(e.g. a temp dir for a one-off run)")
    args = parser.parse_args()
    if not args.dsn:
        parser.error("No DSN: pass --dsn or set PG_DSN in .env")

    import psycopg

    out = args.out_dir or (ROOT / "data" / "dumps" / f"pg_{args.task_id}")
    out.mkdir(parents=True, exist_ok=True)

    with psycopg.connect(args.dsn, connect_timeout=15) as conn, conn.cursor() as cur:
        enrichment = _fetch_enrichment(cur, args.task_id)
        analyzed_task_id = int(enrichment["task_id"])
        history = _fetch_historical_runs(cur, analyzed_task_id, args.history_limit)
        insights = _fetch_insights(cur, analyzed_task_id)
        sandbox_tables = _build_sandbox_tables(cur, analyzed_task_id, enrichment, insights)

    files = {
        "enrichment.json": enrichment,
        "historical_runs.json": history,
        "insights.json": insights,
        "sandbox_tables.json": sandbox_tables,
    }
    for name, payload in files.items():
        (out / name).write_text(json.dumps(payload, indent=2, default=_jsonable),
                                encoding="utf-8")

    print(f"Dumped task {analyzed_task_id} ({enrichment.get('task_name')}, "
          f"app_id={enrichment.get('app_id')}) -> {out}")
    print(f"  historical runs: {len(history)}")
    print(f"  insights rows:   {len(insights)}")
    print(f"  sandbox tables:  {', '.join(sandbox_tables) or '(none)'}")


if __name__ == "__main__":
    main()
