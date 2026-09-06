"""Task name -> Athena dump -> agent run -> result.json + tldr.md, in one workdir.

    ./run.sh service/run_job.py --task-name compute --workdir /tmp/job1
    ./run.sh service/run_job.py --task-name compute --app-id 21483 --workdir /tmp/job1
    ./run.sh service/run_job.py --task-id 2681247 --workdir /tmp/job1

The library function run_job() is what the web service calls (one subprocess
per job — the harness injects data through module globals, so two jobs must
never share a process). It composes existing pieces rather than adding a
third data path:

  main._fetch_athena_task_candidates   task name -> (app_id, env_id, latest run)
  tools/dump_athena.dump_task          Athena -> <workdir>/dump/*.json
  run_from_pg_dump.run_dump            dump -> agent -> <workdir>/result/task_*.json

and then writes two stable-named deliverables next to them:

  <workdir>/result.json   the full saved result (+ a "service" block: how the
                          name was resolved, Athena export freshness, timings)
  <workdir>/tldr.md       service/tldr.build_tldr() rendering of result.json

Everything lives under --workdir; deleting that directory removes the dump,
the run sandbox folders (LOCAL_SANDBOX_DIR is pointed inside it) and the
outputs — the service's session cleanup is a single rmtree.

Exit codes: 0 ok, 2 task name unknown, 3 ambiguous name (candidates printed
as JSON on stdout for the caller to offer as choices), 1 anything else.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import main as _main  # noqa: E402 — wires sys.path/.env before backend imports
from dump_athena import dump_task  # noqa: E402
from run_from_pg_dump import run_dump  # noqa: E402
from service.tldr import build_tldr  # noqa: E402

SOURCE = "athena-service"
EXIT_NOT_FOUND = 2
EXIT_AMBIGUOUS = 3


class TaskNotFound(LookupError):
    pass


class AmbiguousTask(LookupError):
    def __init__(self, task_name: str, candidates: list[dict[str, Any]]):
        super().__init__(f"{task_name!r} runs under {len(candidates)} apps — pass --app-id")
        self.task_name = task_name
        self.candidates = candidates


@dataclass
class JobResult:
    workdir: Path
    result_path: Path
    tldr_path: Path
    dump_dir: Path
    task_id: int
    gate_blocked: bool


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def resolve_task(task_name: str, app_id: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """The candidate to analyze plus every candidate, for provenance."""
    candidates = _main._fetch_athena_task_candidates(task_name)
    if not candidates:
        raise TaskNotFound(f"No analyzable runs of task {task_name!r} in Athena")
    if app_id is not None:
        match = [c for c in candidates if int(c["app_id"]) == int(app_id)]
        if not match:
            raise TaskNotFound(
                f"Task {task_name!r} has no analyzable runs under app_id={app_id} "
                f"(known app_ids: {[c['app_id'] for c in candidates]})"
            )
        return match[0], candidates
    if len(candidates) > 1:
        raise AmbiguousTask(task_name, candidates)
    return candidates[0], candidates


def _gate_blocked_payload(row: dict[str, Any], insights: list[dict], trace: dict, dump_dir: Path) -> dict[str, Any]:
    """_save_run_result skips plan=None runs; the service still needs a result
    to show (the gate reasons ARE the answer), so mirror its shape here."""
    from agent.cost_utils import compute_cost_profile
    from app.brain.insights.tuning.entities.run_metrics import compute_run_metrics

    run_metrics = compute_run_metrics(row)
    cost_profile = compute_cost_profile(
        row, vcore_price=_main._VCORE_PRICE, memory_price=_main._MEMORY_PRICE
    )
    return {
        "saved_at": _now(),
        "source": SOURCE,
        "task_id": row.get("task_id"),
        "task_name": row.get("task_name"),
        "run_config": _main._build_run_config(insights, dump_dir=str(dump_dir)),
        "input": row,
        "plan": None,
        "run_metrics": run_metrics.model_dump() if run_metrics else None,
        "cost_profile": cost_profile.model_dump() if cost_profile else None,
        "assembled_output": [],
        "trace": _main._build_trace(trace),
    }


async def run_job(
    *,
    workdir: Path,
    task_name: str | None = None,
    app_id: int | None = None,
    task_id: int | None = None,
    history_limit: int | None = None,
) -> JobResult:
    if (task_name is None) == (task_id is None):
        raise ValueError("pass exactly one of task_name / task_id")

    workdir = workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    dump_dir = workdir / "dump"
    os.environ["LOCAL_SANDBOX_DIR"] = str(workdir / "sandbox")
    started = _now()

    service: dict[str, Any] = {
        "requested_task_name": task_name,
        "requested_app_id": app_id,
        "requested_task_id": task_id,
        "started_at": started,
    }

    _log("  Checking Athena export freshness…")
    try:
        service["athena_freshness"] = _main._fetch_athena_export_freshness()
    except Exception as exc:  # freshness is a note, never a blocker
        _log(f"  (freshness unavailable: {exc.__class__.__name__}: {exc})")
        service["athena_freshness"] = {}

    if task_name is not None:
        _log(f"  Resolving task name {task_name!r}…")
        chosen, candidates = resolve_task(task_name, app_id)
        service["candidates"] = candidates
        service["chosen"] = chosen
        task_id = int(chosen["latest_task_id"])
        _log(f"  -> app_id={chosen['app_id']} env={chosen.get('env_id')} "
             f"latest run task_id={task_id} at {chosen.get('latest_app_pit')} "
             f"({chosen.get('run_count')} analyzable runs)")

    _log(f"  Dumping task {task_id} from Athena -> {dump_dir}")
    dumped = dump_task(
        int(task_id), dump_dir,
        history_limit=history_limit or _main.HISTORICAL_RUNS_FETCH_LIMIT,
    )
    analyzed_task_id = int(dumped["analyzed_task_id"])
    service["analyzed_task_id"] = analyzed_task_id
    service["dump_finished_at"] = _now()

    _log("  Running the agent…")
    plan, row, insights, trace, saved_path = await run_dump(
        dump_dir, output_dir=workdir / "result", save_results=True,
        source=SOURCE, env_name="athena-service",
    )
    if saved_path is not None:
        payload = json.loads(saved_path.read_text(encoding="utf-8"))
    else:
        payload = _gate_blocked_payload(row, insights, trace, dump_dir)

    service["finished_at"] = _now()
    payload["service"] = service

    result_path = workdir / "result.json"
    result_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    tldr_path = workdir / "tldr.md"
    tldr_path.write_text(build_tldr(payload), encoding="utf-8")
    _log(f"  Wrote {result_path} and {tldr_path}")

    return JobResult(
        workdir=workdir, result_path=result_path, tldr_path=tldr_path, dump_dir=dump_dir,
        task_id=analyzed_task_id, gate_blocked=plan is None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    who = parser.add_mutually_exclusive_group(required=True)
    who.add_argument("--task-name")
    who.add_argument("--task-id", type=int)
    parser.add_argument("--app-id", type=int, help="disambiguate --task-name across apps")
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--history-limit", type=int, default=None)
    parser.add_argument("--print-tldr", action="store_true")
    args = parser.parse_args()

    try:
        result = asyncio.run(run_job(
            workdir=args.workdir, task_name=args.task_name, app_id=args.app_id,
            task_id=args.task_id, history_limit=args.history_limit,
        ))
    except AmbiguousTask as exc:
        _log(f"  {exc}")
        json.dump({"error": "ambiguous", "task_name": exc.task_name,
                   "candidates": exc.candidates}, sys.stdout, indent=2, default=str)
        print()
        sys.exit(EXIT_AMBIGUOUS)
    except TaskNotFound as exc:
        _log(f"  {exc}")
        json.dump({"error": "not_found", "message": str(exc)}, sys.stdout, indent=2)
        print()
        sys.exit(EXIT_NOT_FOUND)

    if args.print_tldr:
        sys.stdout.write(result.tldr_path.read_text(encoding="utf-8"))
    else:
        json.dump({"result": str(result.result_path), "tldr": str(result.tldr_path),
                   "task_id": result.task_id, "gate_blocked": result.gate_blocked},
                  sys.stdout, indent=2)
        print()


if __name__ == "__main__":
    main()
