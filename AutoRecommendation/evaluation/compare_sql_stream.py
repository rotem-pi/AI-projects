"""Compare the production fetch_context → entry_gate → sql_stream pipeline
(run unmodified against Postgres) with the insights recorded in Athena.

    python evaluation/compare_sql_stream.py --task-id 1473514   # from the repo root

Steps:
  1. Run pg_nodes_runner.py inside the definity-app backend venv against
     PG_DATABASE_URL (from .env) — real nodes, real Postgres.
  2. Fetch ALL rows for the same (app_id, task_name) from the Athena
     `insights` table (latest snapshot), annotating each with whether it
     passes the production filters (lifecycle_status='active' AND
     visibility='visible').
  3. Diff by insight type and save a report JSON next to the batch results.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(_REPO_ROOT / ".env")

import main as harness  # Athena helpers + results dir

_BACKEND = _REPO_ROOT.parent.parent / "definity-app" / "backend"
_BACKEND_PYTHON = _BACKEND / ".venv" / "bin" / "python"
_RUNNER = Path(__file__).parent / "pg_nodes_runner.py"


def _run_pg_side(task_id: int, out_path: Path) -> dict:
    database_url = os.getenv("PG_DATABASE_URL")
    if not database_url:
        print("ERROR: PG_DATABASE_URL is not set in .env")
        sys.exit(2)

    # Must be absolute: the subprocess runs with cwd=_BACKEND, so a relative
    # path would resolve there instead of wherever this script's caller runs from.
    out_path = out_path.resolve()

    env = {**os.environ, "DATABASE_URL": database_url}
    result = subprocess.run(
        [str(_BACKEND_PYTHON), str(_RUNNER), "--task-id", str(task_id), "--out", str(out_path)],
        cwd=str(_BACKEND),  # so app.config resolves the backend .env / VERSION
        env=env,
    )
    if result.returncode != 0:
        print("ERROR: pg_nodes_runner failed — see output above.")
        sys.exit(result.returncode)
    return json.loads(out_path.read_text(encoding="utf-8"))


def _passes_production_filters(row: dict) -> bool:
    return row.get("lifecycle_status") == "active" and row.get("visibility") == "visible"


def _compare(pg: dict, athena_rows: list[dict]) -> dict:
    pg_insights = pg["sql_stream"]["sql_insights"]
    pg_by_type = {i["type"]: i for i in pg_insights}

    athena_by_type: dict[str, dict] = {}
    for row in athena_rows:
        annotated = {**row, "passes_production_filters": _passes_production_filters(row)}
        # Latest snapshot has one row per (type, task); keep the passing one if both.
        existing = athena_by_type.get(row["type"])
        if existing is None or annotated["passes_production_filters"]:
            athena_by_type[row["type"]] = annotated

    matched, pg_only, athena_only = [], [], []
    for insight_type, pg_insight in pg_by_type.items():
        athena_row = athena_by_type.get(insight_type)
        if athena_row:
            matched.append({
                "type": insight_type,
                "pg_insight_id": pg_insight.get("id"),
                "athena_insight_id": athena_row.get("insight_id"),
                "pg_usd_cost": pg_insight.get("usd_cost"),
                "athena_lifecycle_status": athena_row.get("lifecycle_status"),
                "athena_visibility": athena_row.get("visibility"),
            })
        else:
            pg_only.append({"type": insight_type, "pg_insight_id": pg_insight.get("id")})

    for insight_type, athena_row in athena_by_type.items():
        if insight_type not in pg_by_type:
            athena_only.append({
                "type": insight_type,
                "athena_insight_id": athena_row.get("insight_id"),
                "lifecycle_status": athena_row.get("lifecycle_status"),
                "visibility": athena_row.get("visibility"),
                "passes_production_filters": athena_row["passes_production_filters"],
                "expected_gap_reason": (
                    None if athena_row["passes_production_filters"]
                    else "excluded by production filters (hidden and/or stale) — sql_stream is correct to skip it"
                ),
            })

    unexplained = [a for a in athena_only if a["passes_production_filters"]]
    return {
        "matched": matched,
        "pg_only": pg_only,
        "athena_only": athena_only,
        "verdict": (
            "MATCH — every visible+active Athena insight was found by sql_stream"
            if not unexplained and not pg_only
            else "DIFFERENCES — see pg_only / unexplained athena_only entries"
        ),
        "unexplained_athena_only_count": len(unexplained),
    }


def _print_report(pg: dict, comparison: dict, athena_rows: list[dict]) -> None:
    identity = pg["identity"]
    gate = pg["entry_gate"]
    fetch = pg["fetch_context"]
    run_sandbox = fetch.get("run_sandbox") or {}

    print(f"\n{'═' * 70}")
    print(f"  Task {pg['task_id']} — {identity['task_name']} "
          f"(app {identity['app_id']}, env {identity['env_name']}, tenant {identity['tenant_id']})")
    print(f"  Postgres: {pg['database_url_host']}")
    print(f"{'═' * 70}")

    if pg.get("diagnostic_notes"):
        print("\n  ⚠ DIAGNOSTIC FALLBACK(S) WERE USED — this run did not execute the")
        print("    literal, unmodified production query end-to-end:")
        for note in pg["diagnostic_notes"]:
            print(f"      - {note}")

    print("\n  fetch_context:")
    print(f"    enrichment row: {'yes' if fetch.get('latest_enrichment') else 'MISSING'}")
    print(f"    historical runs: {len(fetch.get('historical_runs', []))}")
    tables = run_sandbox.get("tables") or []
    print(f"    run_sandbox tables: {', '.join(tables) if tables else 'MISSING'}")
    if pg.get("sandbox_local_dir"):
        print(f"    sandbox CSVs (local, S3 skipped): {pg['sandbox_local_dir']}")

    print(f"\n  entry_gate: {'BLOCKED — ' + '; '.join(gate['gate_reasons']) if gate['gate_blocked'] else 'open'}")

    pg_insights = pg["sql_stream"]["sql_insights"]
    print(f"\n  sql_stream (production node, Postgres): {len(pg_insights)} insight(s)")
    for insight in pg_insights:
        print(f"    - [{insight.get('id')}] {insight['type']}  usd_cost={insight.get('usd_cost')}")
    print(f"  sql_stream recommendations: {len(pg['sql_stream']['sql_recommendations'])}")

    print(f"\n  Athena insights table (latest snapshot): {len(athena_rows)} row(s)")
    print(f"\n  Comparison: {comparison['verdict']}")
    for m in comparison["matched"]:
        print(f"    ✓ matched   {m['type']} (pg id {m['pg_insight_id']}, athena id {m['athena_insight_id']})")
    for p in comparison["pg_only"]:
        print(f"    ! pg-only   {p['type']} (id {p['pg_insight_id']}) — in Postgres but not in the Athena snapshot")
    for a in comparison["athena_only"]:
        marker = "! UNEXPLAINED" if a["passes_production_filters"] else "· expected  "
        print(f"    {marker} athena-only {a['type']} "
              f"(lifecycle={a['lifecycle_status']}, visibility={a['visibility']})")
        if a["expected_gap_reason"]:
            print(f"                  {a['expected_gap_reason']}")
    print()


def main_cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = harness._results_dir() / "compare_sql_stream"
    out_dir.mkdir(parents=True, exist_ok=True)
    pg_path = out_dir / f"task_{args.task_id}_{stamp}_postgres.json"

    print(f"\n[1/3] Running production nodes against Postgres…")
    pg = _run_pg_side(args.task_id, pg_path)

    identity = pg["identity"]
    print(f"\n[2/3] Fetching Athena insights for ({identity['task_name']}, app {identity['app_id']})…")
    athena_rows = harness._fetch_athena_insights(
        identity["app_id"], identity["task_name"], include_hidden=True
    )

    print(f"\n[3/3] Comparing…")
    comparison = _compare(pg, athena_rows)

    report_path = out_dir / f"task_{args.task_id}_{stamp}_report.json"
    report_path.write_text(json.dumps({
        "task_id": args.task_id,
        "identity": dict(identity),
        "comparison": comparison,
        "athena_rows": athena_rows,
        "postgres_output_file": pg_path.name,
    }, indent=2, default=str), encoding="utf-8")

    _print_report(pg, comparison, athena_rows)
    print(f"  Full Postgres-side content → {pg_path}")
    print(f"  Comparison report          → {report_path}\n")


if __name__ == "__main__":
    main_cli()
