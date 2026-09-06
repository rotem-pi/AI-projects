"""Dump everything the agent needs for one task from Athena.

Usage:
    uv run python tools/dump_athena.py 2681247

    ->  data/dumps/athena_2681247/
          enrichment.json        the task_enrichments row for this exact task_id
                                  (falls back to the logical task's latest run
                                  if the exact task_id is missing, same as
                                  main.py's _fetch_athena_task)
          historical_runs.json   up to --history-limit prior runs, oldest first
          insights.json          the task's rows from the Athena `insights`
                                  table (latest snapshot_date only)
          sandbox_tables.json    raw per-table rows for this single run (one
                                  key per table in main.py's
                                  _sandbox_table_sql, plus "insights"), the
                                  Athena-sourced equivalent of production's S3
                                  run-sandbox CSVs (single-PIT only — this
                                  run's rows, not multi-run history)

This is the Athena counterpart of tools/dump_postgres.py — same output shape
(so run_from_pg_dump.py / run_from_dump.py can replay either dump
interchangeably), but sourced from the Athena tables main.py already queries
(_run_athena_query, _fetch_athena_task, _fetch_athena_historical_runs,
_fetch_athena_insights, _fetch_athena_sandbox_tables). AWS auth/query/casting
logic is not reimplemented here — it's imported from main.py so both entry
points stay in sync with a single Athena query implementation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# main.py does all the env/path setup this dump needs (loads .env incl.
# ENV_FILE_NAME before backend modules are ever imported, resolves the
# worktree backend path, inserts it on sys.path) — import it first and reuse
# its already-configured Athena helpers rather than redoing that setup here
# in a different order (app.config.Settings requires ENV_FILE_NAME to be in
# the environment *before* app.config is imported).
import main as _main  # noqa: E402

DEFAULT_HISTORY_LIMIT = _main.HISTORICAL_RUNS_FETCH_LIMIT


def dump_task(
    task_id: int,
    out: Path,
    *,
    history_limit: int = DEFAULT_HISTORY_LIMIT,
    include_hidden_insights: bool = False,
) -> dict[str, Any]:
    """Fetch one task's agent inputs from Athena and write them to `out` in
    the tools/dump_postgres.py shape. Returns the in-memory payloads keyed by
    file name plus "analyzed_task_id" (the exact run, or the logical task's
    latest run when the requested task_id isn't in the export). Library
    entry point shared by the CLI below and service/run_job.py."""
    out.mkdir(parents=True, exist_ok=True)

    rows = _main._fetch_athena_task(task_id)
    if not rows:
        raise LookupError(
            f"No enrichment row found for task_id={task_id} (nor for its logical task)"
        )
    enrichment = rows[0]
    analyzed_task_id = int(enrichment["task_id"])

    history = _main._fetch_athena_historical_runs(
        enrichment["task_name"], enrichment["app_id"], analyzed_task_id, history_limit
    )
    insights = _main._fetch_athena_insights(
        enrichment["app_id"], enrichment["task_name"], include_hidden=include_hidden_insights
    )
    sandbox_tables = _main._fetch_athena_sandbox_tables(analyzed_task_id, enrichment, insights)

    files: dict[str, Any] = {
        "enrichment.json": enrichment,
        "historical_runs.json": history,
        "insights.json": insights,
        "sandbox_tables.json": sandbox_tables,
    }
    for name, payload in files.items():
        (out / name).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {**files, "analyzed_task_id": analyzed_task_id, "out_dir": out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Dump one task from Athena for run_from_pg_dump.py")
    parser.add_argument("task_id", type=int)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--include-hidden-insights", action="store_true",
                        help="Fetch all insights rows, not just lifecycle_status='active'")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="Write here instead of data/dumps/athena_<task_id> "
                             "(e.g. a temp dir for a one-off run)")
    args = parser.parse_args()

    out = args.out_dir or (ROOT / "data" / "dumps" / f"athena_{args.task_id}")
    try:
        dumped = dump_task(
            args.task_id, out,
            history_limit=args.history_limit,
            include_hidden_insights=args.include_hidden_insights,
        )
    except LookupError as exc:
        raise SystemExit(str(exc))

    enrichment = dumped["enrichment.json"]
    print(f"Dumped task {dumped['analyzed_task_id']} ({enrichment.get('task_name')}, "
          f"app_id={enrichment.get('app_id')}) -> {out}")
    print(f"  historical runs: {len(dumped['historical_runs.json'])}")
    print(f"  insights rows:   {len(dumped['insights.json'])}")
    print(f"  sandbox tables:  {', '.join(dumped['sandbox_tables.json']) or '(none)'}")


if __name__ == "__main__":
    main()
