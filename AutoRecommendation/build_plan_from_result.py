"""Build a tuning plan offline from a saved agent-run result JSON.

Usage:
    ./run.sh build_plan_from_result.py data/results/task_2990006_20260820T110814Z.json

Loads the result's plan.plan_inputs — the PlanInputs the agent's
assemble_output node prepared for the tuning loop — and runs the real
build_plan() on it. Fully offline: no database, no Bedrock; every input is
embedded in the result file. Writes <result-stem>.plan.json next to the
input (override with --out) and prints a knob summary.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_json", type=Path, help="a data/results/task_*.json file")
    parser.add_argument("--out", help="Output path (default: <result-stem>.plan.json)")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_worktree import resolved_backend_path

    sys.path.insert(0, str(resolved_backend_path()))

    from app.brain.insights.tuning import (
        ActiveInsight,
        PlanInputs,
        TuningConstraints,
        build_plan,
    )
    from app.brain.insights.tuning.entities import TaskProfile, TaskRunMetrics

    result = json.loads(args.result_json.read_text())
    pi = result["plan"]["plan_inputs"]

    plan = build_plan(
        PlanInputs(
            active_insights=[ActiveInsight(**ai) for ai in pi["active_insights"]],
            run_metrics=TaskRunMetrics(**pi["run_metrics"]) if pi.get("run_metrics") else None,
            task_profile=TaskProfile(**pi["task_profile"]) if pi.get("task_profile") else None,
            pre_tuning_durations_ms=pi.get("pre_tuning_durations_ms") or [],
            pre_tuning_cost_usd=pi.get("pre_tuning_cost_usd"),
            constraints=TuningConstraints(**pi["constraints"]),
        )
    )

    out_path = (
        Path(args.out) if args.out else args.result_json.with_suffix(".plan.json")
    )
    out_path.write_text(json.dumps(plan.model_dump(mode="json"), indent=2, default=str))
    print(f"Wrote {out_path}\n")

    for label, knobs in (("ACTIVE", plan.knobs), ("BLOCKED", plan.blocked_knobs)):
        print(f"{label}:")
        for k in knobs:
            spec = k.spec
            line = f"  {spec.config_key}: {spec.initial_value} -> {spec.target_value}"
            if spec.staircase and len(spec.staircase) > 1:
                line += f" via {spec.staircase}"
            if spec.block_reason:
                line += f"  [{spec.block_reason[:80]}]"
            print(line)


if __name__ == "__main__":
    main()
