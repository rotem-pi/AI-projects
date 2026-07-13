"""Consistency gate — CI-style pass/fail wrapper around the consistency test.

The notebook (consistency_test.ipynb) explores consistency interactively;
this script turns the same measurement into a shippability gate:

    avg key consistency >= THRESHOLD on the benchmark set,
    AND plan_degraded rate <= MAX_DEGRADED_PCT,
    or the prompt/KB change doesn't ship.

plan_degraded runs (the planner exhausted its iteration budget and produced
nothing) are a quality regression signal in their own right: a prompt/KB
change that makes the planner wander must fail the gate even when the runs
that DID finish were consistent.

Unlike the notebook, the task set is FIXED: a benchmark file pins the
(task_name, app_id, task_id) jobs and their historical runs, so every gate
run measures the same inputs and scores are comparable across prompt/KB
versions.  Results are stamped with the kb_version that produced them.

Usage:
    # From the repo root:
    python evaluation/consistency_gate.py --create-benchmark 5   # sample & pin the benchmark set
    python evaluation/consistency_gate.py                        # run the gate (5 cycles, 80%)
    python evaluation/consistency_gate.py --cycles 5 --threshold 80
    python evaluation/consistency_gate.py --benchmark data/benchmark_tasks.json

Exit codes:
    0 — gate passed (avg key consistency >= threshold)
    1 — gate FAILED (below threshold): don't ship the change
    2 — measurement invalid (benchmark missing, or a task produced no
        successful cycles) — fix the run, don't read the score

Metric (per the notebook, with one fix): for each task, every config_key
recommended in >=1 successful cycle scores cycles_seen / cycles_ok; the
task's score is the mean over its keys (100 when the agent consistently
recommends nothing).  The gate metric is the mean of task scores.  Unlike
the notebook, cycles_ok counts every successful agent run — a cycle that
produced zero recommendations still counts, so "sometimes finds nothing"
correctly reads as inconsistency instead of inflating the score.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import main as arm

DEFAULT_BENCHMARK_PATH = _REPO_ROOT / "data" / "benchmark_tasks.json"
DEFAULT_CYCLES = 5
DEFAULT_THRESHOLD_PCT = 80.0
DEFAULT_MAX_DEGRADED_PCT = 10.0

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_INVALID = 2

_NUMERIC_VALUE_RE = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*[a-zA-Z%]*\s*$")


# ── Benchmark file ──────────────────────────────────────────────────────────


def create_benchmark(path: Path, n_tasks: int) -> None:
    rows = arm._fetch_athena_sample_with_history(n_tasks, arm._MIN_RUNS_FOR_SAMPLE)
    jobs = arm._group_current_and_history(rows)
    payload = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "min_historical_runs": arm._MIN_RUNS_FOR_SAMPLE,
        "jobs": [{"row": row, "history": history} for row, history in jobs],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Benchmark of {len(jobs)} task(s) saved -> {path}")
    for job in payload["jobs"]:
        row = job["row"]
        print(f"  task_id={row['task_id']}  {row['task_name']} (app_id={row['app_id']})")


def load_benchmark(path: Path) -> list[tuple[dict, list[dict]]]:
    payload = json.loads(path.read_text())
    return [(job["row"], job["history"]) for job in payload["jobs"]]


# ── Run cycles ──────────────────────────────────────────────────────────────


async def run_cycles(
    jobs: list[tuple[dict, list[dict]]], cycles: int, output_dir: Path
) -> tuple[list[dict], str | None]:
    """Sequential (set_row_override / set_insights_override are module-global).

    Returns flattened recommendation records plus the kb_version that
    produced them (from the first successful plan).
    """
    records: list[dict] = []
    kb_version: str | None = None
    for row, history in jobs:
        task_id = row["task_id"]
        for cycle in range(1, cycles + 1):
            print(f"--- task_id={task_id} ({row['task_name']}) cycle {cycle}/{cycles} ---")
            try:
                _, plan = await arm._run_on_row(
                    row,
                    as_json=False,
                    show_progress=False,
                    save_results=True,
                    output_dir=output_dir,
                    summary_csv=output_dir / "batch_summary.csv",
                    source="athena",
                    historical_runs=history,
                )
            except Exception as exc:
                print(f"  !! cycle {cycle} failed: {exc.__class__.__name__}: {exc}")
                records.append({"task_id": task_id, "cycle": cycle, "ok": False})
                continue
            if plan is None:
                records.append({"task_id": task_id, "cycle": cycle, "ok": False})
                continue
            kb_version = kb_version or plan.kb_version
            recs = list(_iter_recommendations(arm._build_assembled_output(plan)))
            records.append(
                {
                    "task_id": task_id,
                    "cycle": cycle,
                    "ok": True,
                    "plan_degraded": bool(getattr(plan, "plan_degraded", False)),
                    "recommendations": recs,
                }
            )
    return records, kb_version


def _iter_recommendations(assembled: list[dict]):
    """One record per recommended config_key (same flattening as the notebook):
    agent_discovered entries carry the key at top level, SQL-rule entries
    nest theirs under `recommendations`."""
    for entry in assembled:
        if "config_key" in entry:
            yield {
                "config_key": entry["config_key"],
                "suggested_value": entry.get("suggested_value"),
                "blocked": bool(entry.get("blocked_by")),
            }
        else:
            for rec in entry.get("recommendations", []):
                yield {
                    "config_key": rec["config_key"],
                    "suggested_value": rec.get("suggested_value"),
                    "blocked": False,
                }


# ── Scoring ─────────────────────────────────────────────────────────────────


def score_task(task_records: list[dict]) -> dict:
    """Key-consistency score for one task across its cycles."""
    ok_cycles = [r for r in task_records if r["ok"]]
    cycles_ok = len(ok_cycles)
    key_cycles: dict[str, set[int]] = {}
    key_values: dict[str, list[str]] = {}
    for record in ok_cycles:
        for rec in record["recommendations"]:
            key = rec["config_key"]
            key_cycles.setdefault(key, set()).add(record["cycle"])
            if rec["suggested_value"] is not None:
                key_values.setdefault(key, []).append(str(rec["suggested_value"]))

    keys = [
        {
            "config_key": key,
            "cycles_seen": len(seen),
            "cycles_ok": cycles_ok,
            "consistency_pct": round(100.0 * len(seen) / cycles_ok, 1),
            **_value_spread(key_values.get(key, [])),
        }
        for key, seen in sorted(key_cycles.items())
    ]
    task_pct = (
        round(statistics.mean(k["consistency_pct"] for k in keys), 1)
        if keys
        else 100.0  # consistently recommends nothing
    )
    return {"cycles_ok": cycles_ok, "keys": keys, "task_consistency_pct": task_pct}


def _value_spread(values: list[str]) -> dict:
    """Informational: how much suggested_value varies for one key."""
    distinct = sorted(set(values))
    numeric = [float(m.group(1)) for v in values if (m := _NUMERIC_VALUE_RE.match(v))]
    spread_pct = None
    if len(numeric) == len(values) and numeric and min(numeric) > 0:
        spread_pct = round(100.0 * (max(numeric) - min(numeric)) / min(numeric), 1)
    return {"distinct_values": distinct, "value_spread_pct": spread_pct}


def score_all(records: list[dict], jobs: list[tuple[dict, list[dict]]]) -> dict:
    tasks = {}
    for row, _ in jobs:
        task_id = row["task_id"]
        task_records = [r for r in records if r["task_id"] == task_id]
        tasks[str(task_id)] = {
            "task_name": row["task_name"],
            **score_task(task_records),
        }
    scores = [t["task_consistency_pct"] for t in tasks.values()]
    ok_runs = [r for r in records if r["ok"]]
    degraded = sum(1 for r in ok_runs if r.get("plan_degraded"))
    return {
        "tasks": tasks,
        "avg_key_consistency_pct": round(statistics.mean(scores), 1) if scores else None,
        "plan_degraded_pct": (
            round(100.0 * degraded / len(ok_runs), 1) if ok_runs else None
        ),
    }


def rebuild_records(run_dir: Path) -> list[dict]:
    """Reconstruct gate records from the per-run task_*.json files a run
    directory holds (also works for legacy consistency_* notebook runs —
    same _save_run_result schema).  Cycle numbers come from saved_at order
    within each task.  Failed cycles save no file, so they cannot be
    reconstructed — rebuilt scores treat every saved run as ok."""
    per_task: dict[int, list[dict]] = {}
    for path in sorted(run_dir.glob("task_*.json")):
        payload = json.loads(path.read_text())
        per_task.setdefault(payload["task_id"], []).append(payload)

    records: list[dict] = []
    for task_id, payloads in per_task.items():
        payloads.sort(key=lambda p: p["saved_at"])
        for cycle, payload in enumerate(payloads, start=1):
            records.append(
                {
                    "task_id": task_id,
                    "task_name": payload.get("task_name"),
                    "cycle": cycle,
                    "ok": True,
                    "plan_degraded": bool(payload["plan"].get("plan_degraded")),
                    "kb_version": payload["plan"].get("kb_version"),
                    "recommendations": list(
                        _iter_recommendations(payload["assembled_output"])
                    ),
                }
            )
    return records


def score_run_dir(run_dir: Path) -> dict:
    """Score a saved run directory from its task_*.json files — for
    exploring runs that predate gate_result.json (legacy notebook runs)."""
    records = rebuild_records(run_dir)
    jobs = [
        ({"task_id": task_id, "task_name": name, "app_id": None}, [])
        for task_id, name in sorted(
            {(r["task_id"], r["task_name"]) for r in records}
        )
    ]
    result = score_all(records, jobs)
    result["kb_version"] = next(
        (r["kb_version"] for r in records if r.get("kb_version")), None
    )
    result["cycles"] = max((r["cycle"] for r in records), default=0)
    return result


# ── Reporting / gate ────────────────────────────────────────────────────────


def print_report(result: dict, threshold: float, max_degraded_pct: float) -> None:
    print("\n================ consistency gate ================")
    for task_id, task in result["tasks"].items():
        print(
            f"task {task_id} ({task['task_name']}): "
            f"{task['task_consistency_pct']}% over {task['cycles_ok']} cycle(s)"
        )
        for key in task["keys"]:
            spread = (
                f", value spread {key['value_spread_pct']}%"
                if key["value_spread_pct"] is not None
                else f", values {key['distinct_values']}"
            )
            print(
                f"    {key['config_key']}: {key['cycles_seen']}/{key['cycles_ok']} "
                f"cycles ({key['consistency_pct']}%){spread}"
            )
    print(
        f"\navg key consistency: {result['avg_key_consistency_pct']}% "
        f"(threshold {threshold}%)"
    )
    print(
        f"plan_degraded rate: {result['plan_degraded_pct']}% "
        f"(max {max_degraded_pct}%)"
    )


def gate_verdict(result: dict, threshold: float, max_degraded_pct: float) -> int:
    broken = [
        task_id for task_id, task in result["tasks"].items() if task["cycles_ok"] == 0
    ]
    if broken:
        print(f"INVALID: task(s) {broken} produced no successful cycles — "
              "score is not trustworthy.")
        return EXIT_INVALID
    if result["avg_key_consistency_pct"] is None:
        print("INVALID: no tasks scored.")
        return EXIT_INVALID
    failures = []
    if result["avg_key_consistency_pct"] < threshold:
        failures.append(
            f"avg key consistency {result['avg_key_consistency_pct']}% "
            f"< {threshold}%"
        )
    degraded_pct = result.get("plan_degraded_pct")
    if degraded_pct is not None and degraded_pct > max_degraded_pct:
        failures.append(
            f"plan_degraded rate {degraded_pct}% > {max_degraded_pct}%"
        )
    if failures:
        print(f"GATE FAILED ({'; '.join(failures)}) — "
              "the prompt/KB change doesn't ship.")
        return EXIT_FAIL
    print("GATE PASSED.")
    return EXIT_PASS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK_PATH)
    parser.add_argument(
        "--create-benchmark", type=int, metavar="N_TASKS",
        help="Sample N tasks from Athena, pin them as the benchmark set, and exit.",
    )
    parser.add_argument("--cycles", type=int, default=DEFAULT_CYCLES)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD_PCT)
    parser.add_argument(
        "--max-degraded-pct", type=float, default=DEFAULT_MAX_DEGRADED_PCT,
        help="Max share of successful runs allowed to be plan_degraded "
        "(planner exhausted its iteration budget).",
    )
    args = parser.parse_args()

    if args.create_benchmark is not None:
        create_benchmark(args.benchmark, args.create_benchmark)
        return EXIT_PASS

    if not args.benchmark.exists():
        print(f"INVALID: benchmark file {args.benchmark} not found — create one with "
              "--create-benchmark N.")
        return EXIT_INVALID

    jobs = load_benchmark(args.benchmark)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = arm._results_dir() / f"consistency_gate_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Gate over {len(jobs)} task(s) x {args.cycles} cycle(s) -> {output_dir}")

    records, kb_version = asyncio.run(run_cycles(jobs, args.cycles, output_dir))
    result = score_all(records, jobs)
    result.update(
        kb_version=kb_version,
        cycles=args.cycles,
        threshold_pct=args.threshold,
        max_degraded_pct=args.max_degraded_pct,
        benchmark=str(args.benchmark),
        run_utc=stamp,
    )
    print_report(result, args.threshold, args.max_degraded_pct)
    print(f"kb_version: {kb_version}")

    verdict = gate_verdict(result, args.threshold, args.max_degraded_pct)
    result["verdict"] = {EXIT_PASS: "pass", EXIT_FAIL: "fail"}.get(verdict, "invalid")
    (output_dir / "gate_result.json").write_text(json.dumps(result, indent=2))
    print(f"Saved -> {output_dir / 'gate_result.json'}")
    return verdict


if __name__ == "__main__":
    sys.exit(main())
