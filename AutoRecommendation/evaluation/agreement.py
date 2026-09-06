"""Agreement report across saved result files of one task.

consistency_gate.py measures consistency by re-running the agent N times;
this tool is the offline complement: it reads the task_<id>_*.json files
that data/results already holds (e.g. the four task_1685 runs of one
afternoon) and reports how much the runs agree — without spending a token.

Reported per task:
- per-run recommendation sets (config_key: current -> suggested, disposition)
- key agreement: for every config_key seen anywhere, the share of runs that
  recommended it (the consistency_gate metric, applied to saved files)
- value spread: keys recommended with materially different suggested values
  across runs (same lever, unanchored sizing)
- contradictions: keys recommended in one run and safety-blocked in another

Usage:
    python evaluation/agreement.py 1685
    python evaluation/agreement.py 1685 --results-dir data/results --min-agreement 80
Exit codes: 0 = report printed (and agreement >= --min-agreement when given),
1 = agreement below --min-agreement, 2 = fewer than two result files.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel

_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = _REPO_ROOT / "data" / "results"

EXIT_OK = 0
EXIT_BELOW_THRESHOLD = 1
EXIT_TOO_FEW_RUNS = 2

_STATUS_RECOMMENDED = "recommended"
_STATUS_BLOCKED = "blocked"


class RunRecommendation(BaseModel):
    config_key: str
    current_value: object = None
    suggested_value: object = None
    status: str = _STATUS_RECOMMENDED


class RunSummary(BaseModel):
    file_name: str
    saved_at: str | None = None
    recommendations: list[RunRecommendation]


def load_runs(task_id: int, results_dir: Path) -> list[RunSummary]:
    runs: list[RunSummary] = []
    for path in sorted(results_dir.glob(f"task_{task_id}_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        plan = payload.get("plan") or {}
        recommended = {
            rec.get("config_key"): rec for rec in plan.get("recommendations") or []
        }
        blocked = {
            rec.get("config_key"): rec
            for rec in plan.get("blocked_recommendations") or []
        }
        recommendations = [
            RunRecommendation(
                config_key=key,
                current_value=rec.get("current_value"),
                suggested_value=rec.get("suggested_value"),
                status=_STATUS_BLOCKED if key in blocked else _STATUS_RECOMMENDED,
            )
            for key, rec in {**recommended, **blocked}.items()
            if key
        ]
        runs.append(
            RunSummary(
                file_name=path.name,
                saved_at=payload.get("saved_at"),
                recommendations=recommendations,
            )
        )
    return runs


def _recommended_keys(run: RunSummary) -> set[str]:
    return {
        rec.config_key
        for rec in run.recommendations
        if rec.status == _STATUS_RECOMMENDED
    }


def key_agreement_pct(runs: list[RunSummary]) -> tuple[float, dict[str, int]]:
    """The consistency_gate metric over saved files: every config_key
    recommended in >=1 run scores runs_seen / runs_total; the task score is
    the mean over keys (100.0 when no run recommended anything)."""
    seen_counts: dict[str, int] = {}
    for run in runs:
        for key in _recommended_keys(run):
            seen_counts[key] = seen_counts.get(key, 0) + 1
    if not seen_counts:
        return 100.0, {}
    shares = [count / len(runs) for count in seen_counts.values()]
    return 100.0 * sum(shares) / len(shares), seen_counts


def value_spread(runs: list[RunSummary]) -> dict[str, list[object]]:
    """config_key -> distinct suggested values, for keys with more than one."""
    values_by_key: dict[str, list[object]] = {}
    for run in runs:
        for rec in run.recommendations:
            if rec.status != _STATUS_RECOMMENDED:
                continue
            bucket = values_by_key.setdefault(rec.config_key, [])
            if rec.suggested_value not in bucket:
                bucket.append(rec.suggested_value)
    return {key: vals for key, vals in values_by_key.items() if len(vals) > 1}


def contradictions(runs: list[RunSummary]) -> list[str]:
    """Keys recommended in one run but safety-blocked in another."""
    recommended: set[str] = set()
    blocked: set[str] = set()
    for run in runs:
        for rec in run.recommendations:
            target = recommended if rec.status == _STATUS_RECOMMENDED else blocked
            target.add(rec.config_key)
    return sorted(recommended & blocked)


def print_report(task_id: int, runs: list[RunSummary]) -> float:
    print(f"Agreement report — task {task_id}, {len(runs)} saved run(s)\n")
    for run in runs:
        print(f"  {run.file_name}  saved_at={run.saved_at}")
        if not run.recommendations:
            print("    (no recommendations)")
        for rec in run.recommendations:
            print(
                f"    - {rec.config_key}: {rec.current_value} -> "
                f"{rec.suggested_value} [{rec.status}]"
            )
    agreement, seen_counts = key_agreement_pct(runs)
    print(f"\n  Key agreement: {agreement:.1f}%")
    for key, count in sorted(seen_counts.items(), key=lambda item: -item[1]):
        print(f"    {key}: recommended in {count}/{len(runs)} run(s)")
    spread = value_spread(runs)
    if spread:
        print("\n  Value spread (same lever, different sizing):")
        for key, values in spread.items():
            print(f"    {key}: {values}")
    conflicting = contradictions(runs)
    if conflicting:
        print("\n  Contradictions (recommended in one run, blocked in another):")
        for key in conflicting:
            print(f"    {key}")
    return agreement


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report recommendation agreement across saved result files."
    )
    parser.add_argument("task_id", type=int, help="Task id, e.g. 1685")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--min-agreement",
        type=float,
        default=None,
        help="Exit 1 when key agreement (%%) is below this value.",
    )
    args = parser.parse_args()

    runs = load_runs(args.task_id, args.results_dir)
    if len(runs) < 2:
        print(
            f"Need >=2 result files for task {args.task_id} in {args.results_dir} "
            f"(found {len(runs)}).",
            file=sys.stderr,
        )
        return EXIT_TOO_FEW_RUNS
    agreement = print_report(args.task_id, runs)
    if args.min_agreement is not None and agreement < args.min_agreement:
        return EXIT_BELOW_THRESHOLD
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
