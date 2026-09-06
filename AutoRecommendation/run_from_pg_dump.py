"""Run the AutoRecommendation agent on a tools/dump_postgres.py dump.

Usage:
    ./run.sh run_from_pg_dump.py data/dumps/pg_2681247
    ./run.sh run_from_pg_dump.py data/dumps/pg_2681247 --json

The dump already holds the task_enrichments row, real historical runs,
insights rows and raw per-table sandbox rows in their native (Postgres)
shapes, so — unlike run_from_dump.py's REST-CSV bridging — this only loads
the JSON files and injects them via the same overrides main.py uses. Trend
analysis works (historical runs are real); run_cost_usd is recomputed from
env prices by fetch_context, matching the Athena path. sandbox_tables.json
feeds set_row_override()'s sandbox_tables param, which agent/local_sandbox.py
turns into a local RunSandbox for run_deterministic_review/
estimate_change_saving/get_advanced_config_catalog to read (single-PIT only
— this run's rows, no multi-run history).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from main import (  # noqa: E402 — main wires sys.path to definity-app's backend
    _MEMORY_PRICE,
    _VCORE_PRICE,
    _build_assembled_output,
    _build_run_config,
    _print_assembled_output,
    _print_plan,
    _persist_result,
    _results_dir,
)

from agent.cost_utils import compute_cost_profile  # noqa: E402


def _load(dump: Path, name: str, default):
    path = dump / name
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def _filter_insights(rows: list[dict]) -> list[dict]:
    """Mirrors the production agent's active_insights query
    (agent_context.sql): lifecycle_status='active' only.  Production does
    NOT filter on visibility — hidden-but-active insights reach the agent
    there too."""
    return [r for r in rows if r.get("lifecycle_status") == "active"]


def _annual_runs_from_history(
    historical_runs: list[dict], current_app_pit: object
) -> float | None:
    """Runs/year from real historical-run timestamps (app_pit/start_time).
    dump_postgres.py already gives real history here — no Athena/insight
    fallback needed, unlike run_from_dump.py's REST-CSV path."""
    timestamps = [
        run.get("app_pit") or run.get("start_time") for run in historical_runs
    ]
    timestamps.append(current_app_pit)
    parsed: list[datetime] = []
    for ts in timestamps:
        if not ts:
            continue
        try:
            parsed.append(datetime.fromisoformat(str(ts)))
        except ValueError:
            continue
    if len(parsed) < 2:
        return None
    span_days = (max(parsed) - min(parsed)).total_seconds() / 86400
    if span_days <= 0:
        return None
    return len(parsed) / span_days * 365.25


def fill_missing_annual_costs(plan, row: dict, historical_runs: list[dict]):
    """Price recommendations this harness leaves unpriced.

    assemble_output derives usd_cost_annual from a run_sandbox (raw stage/
    timeline tables materialized to a local folder for kb-shay's audited
    cost_basis() stack) that this harness never builds — real historical
    runs alone aren't enough, so every agent recommendation lands with
    usd_cost_annual=None even when the planner set estimated_saving_fraction.
    Recover an approximate baseline instead: this run's cost (the same
    CALCULATE_USD_COST port main.py persists as cost_profile) x the run
    frequency measured from real historical-run timestamps, then price each
    unpriced recommendation as baseline x estimated_saving_fraction. This is
    a cruder estimate than the audited folder-based one production uses —
    it skips the allocated-vs-used basis reconciliation cost_basis() does —
    but it's better than leaving every figure null in a standalone harness.
    """
    if plan is None:
        return None
    annual_runs = _annual_runs_from_history(historical_runs, row.get("app_pit"))
    cost_per_run = compute_cost_profile(
        row, vcore_price=_VCORE_PRICE, memory_price=_MEMORY_PRICE
    ).cost_per_run_usd
    if not annual_runs or not cost_per_run:
        return plan
    annual_cost = annual_runs * cost_per_run

    def _price(recs):
        return [
            rec.model_copy(
                update={
                    "usd_cost_annual": round(
                        annual_cost * rec.estimated_saving_fraction, 2
                    )
                }
            )
            if rec.usd_cost_annual is None and rec.estimated_saving_fraction is not None
            else rec
            for rec in recs
        ]

    return plan.model_copy(
        update={
            "recommendations": _price(plan.recommendations),
            "blocked_recommendations": _price(plan.blocked_recommendations),
        }
    )


def _merge_manual_overrides(row: dict) -> dict:
    """agent_context.sql's active_insights merges the user override into the
    payload (insights_payload || insights_payload_override); do the same here.
    The old manual_overrides key is read as a fallback so dumps taken before
    the 2026-07-31 column rename still replay."""
    overrides = row.get("insights_payload_override") or row.get("manual_overrides")
    if isinstance(overrides, dict) and overrides:
        payload = row.get("insights_payload")
        row = {**row, "insights_payload": {**(payload or {}), **overrides}}
    return row


def load_dump(dump: Path) -> tuple[dict, list[dict], list[dict], dict]:
    """(enrichment row, historical runs, active insights, sandbox tables)
    from a dump_postgres.py / dump_athena.py directory, with the same
    production-mimicking adjustments _run has always applied."""
    row = _load(dump, "enrichment.json", None)
    if row is None:
        raise FileNotFoundError(
            f"{dump}/enrichment.json not found — run tools/dump_postgres.py or tools/dump_athena.py first"
        )
    history = _load(dump, "historical_runs.json", [])
    insights = [
        _merge_manual_overrides(r)
        for r in _filter_insights(_load(dump, "insights.json", []))
    ]
    sandbox_tables = _load(dump, "sandbox_tables.json", {}) or {}

    # task__aqe_enabled__param is not a task_enrichments column — production
    # and run_from_dump.py alike derive it from the submitted Spark conf.
    # TaskProfile._safe_bool handles the 'true'/'false' string form.
    if row.get("task__aqe_enabled__param") is None:
        row["task__aqe_enabled__param"] = next(
            (p.get("value") for p in sandbox_tables.get("task_params", [])
             if p.get("key") == "spark.sql.adaptive.enabled"),
            None,
        )
    return row, history, insights, sandbox_tables


async def run_dump(
    dump: Path,
    *,
    output_dir: Path,
    save_results: bool = True,
    source: str = "pg-dump",
    env_name: str = "pg-dump",
) -> tuple[object, dict, list[dict], dict, Path | None]:
    """Run the agent on a dump directory and persist the result.

    Returns (plan, enrichment row, active insights, graph trace, saved result
    path) — plan is None when the entry gate blocked the run (the trace then
    carries gate_reasons), and the path is None when nothing was saved.
    Library entry point shared by the CLI below and service/run_job.py.
    """
    from agent.inference_graph import run_analysis
    from agent.nodes.fetch_context import set_row_override
    from agent.nodes.sql_stream import set_insights_override

    row, history, insights, sandbox_tables = load_dump(dump)

    print(f"  Task {row.get('task_id')} ({row.get('task_name')}, app_id={row.get('app_id')})",
          file=sys.stderr)
    print(f"  AQE enabled (from submitted conf): {row['task__aqe_enabled__param']!r}",
          file=sys.stderr)
    print(f"  Historical runs: {len(history)}", file=sys.stderr)
    print(f"  Insights: {[i.get('type') for i in insights]}", file=sys.stderr)
    print(f"  Sandbox tables: {', '.join(sandbox_tables) or '(none)'}", file=sys.stderr)

    set_row_override(row, history, sandbox_tables)
    set_insights_override(insights)
    try:
        plan, trace = await run_analysis(int(row["task_id"]), env_name=env_name)
    finally:
        set_row_override(None)
        set_insights_override(None)

    plan = fill_missing_annual_costs(plan, row, history)

    path = _persist_result(plan, row=row, source=source, output_dir=output_dir,
                           summary_csv=None, save_results=save_results, trace=trace,
                           run_config=_build_run_config(insights, dump_dir=str(dump)))
    return plan, row, insights, trace, path


async def _run(dump: Path, as_json: bool, save_results: bool) -> None:
    try:
        plan, row, _insights, _trace, _path = await run_dump(
            dump, output_dir=_results_dir(), save_results=save_results
        )
    except FileNotFoundError as exc:
        raise SystemExit(str(exc))
    _print_plan(plan, as_json, row)
    if not as_json:
        _print_assembled_output(_build_assembled_output(plan))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the agent on a dump_postgres.py directory")
    parser.add_argument("dump_dir", type=Path, help="e.g. data/dumps/pg_2681247")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--no-save", action="store_true", help="Do not write results to disk")
    args = parser.parse_args()

    if not args.dump_dir.is_dir():
        parser.error(f"{args.dump_dir} is not a directory")
    asyncio.run(_run(args.dump_dir, args.json, save_results=not args.no_save))


if __name__ == "__main__":
    main()
