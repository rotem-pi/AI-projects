"""Deterministic TL;DR of a saved agent result — no LLM call.

    python service/tldr.py data/results/task_18609_20260906T081331Z.json

Input is the JSON main.py's _save_run_result writes (input, plan, run_metrics,
cost_profile, assembled_output, trace), optionally with a "service" block
from service/run_job.py (task-name resolution + Athena export freshness).
Output is Markdown with four fixed sections:

  What to change   plan.recommendations   -> change / why / how / risk / $
  Blocked          plan.blocked_recommendations -> intended change / why blocked
  Not actioned     plan.unactioned_insights     -> insight / why not actioned
  Data & provenance  analyzed run, cost/run, Athena export lag note, model

Everything shown is already in the result; this only orders and words it,
so two runs with the same plan always produce the same TL;DR.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

# How to apply each RecommendationAction, in operator terms. Unknown actions
# fall back to a generic sentence so a new enum value can't crash the page.
_HOW_BY_ACTION: dict[str, str] = {
    # app.brain.insights.recommendations.models.RecommendationAction values
    "set_spark_config": (
        "Set `{key}` to `{new}` in the job's Spark configuration "
        "(spark-submit `--conf`, the cluster's Spark conf, or the job's SparkSession builder)."
    ),
    "set_cluster_config": (
        "Set `{key}` to `{new}` in the cluster / job-cluster definition the task runs on."
    ),
    "change_instance_type": (
        "Change the instance type for `{key}` to `{new}` in the cluster definition, "
        "then re-check memory and core sizing on the next run."
    ),
}


def _fmt_value(value: Any) -> str:
    if value is None or value == "None":
        return "unset"
    return str(value)


def _fmt_usd(value: Any) -> str | None:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return None
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.0f}/yr"


def _fmt_pct(value: Any) -> str | None:
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return None


def _how(rec: dict[str, Any]) -> str:
    action = str(rec.get("action") or "")
    template = _HOW_BY_ACTION.get(action, "Apply action `{action}` to `{key}` (new value `{new}`).")
    return template.format(
        key=rec.get("config_key"),
        new=_fmt_value(rec.get("suggested_value")),
        action=action,
    )


def _change_line(rec: dict[str, Any]) -> str:
    return (
        f"`{rec.get('config_key')}`: {_fmt_value(rec.get('current_value'))} → "
        f"{_fmt_value(rec.get('suggested_value'))}"
    )


def _recommended_section(recs: list[dict[str, Any]]) -> list[str]:
    out = [f"## What to change ({len(recs)})", ""]
    if not recs:
        out += ["Nothing to change: no recommendation passed the safety review.", ""]
        return out
    for i, rec in enumerate(recs, 1):
        cost = _fmt_usd(rec.get("usd_cost_annual"))
        saving = _fmt_pct(rec.get("estimated_saving_fraction"))
        headline = f"### {i}. {_change_line(rec)}"
        if cost:
            headline += f" · est. {cost}"
        out.append(headline)
        out.append(f"- **Why:** {rec.get('explanation') or rec.get('rationale') or '(no explanation recorded)'}")
        out.append(f"- **How:** {_how(rec)}")
        if rec.get("expected_impact"):
            out.append(f"- **Expected impact:** {rec['expected_impact']}")
        if rec.get("risk_note"):
            out.append(f"- **Risk:** {rec['risk_note']}")
        meta = [f"source: {rec.get('source')}"]
        if rec.get("insight_type"):
            meta.append(f"insight: {rec['insight_type']}")
        if rec.get("confidence") is not None:
            meta.append(f"confidence: {rec['confidence']}")
        if saving:
            meta.append(f"estimated saving: {saving} of run cost")
        out.append(f"- _{' · '.join(meta)}_")
        out.append("")
    return out


def _blocked_section(recs: list[dict[str, Any]]) -> list[str]:
    out = [f"## Blocked ({len(recs)})", ""]
    if not recs:
        out += ["No recommendation was blocked.", ""]
        return out
    out += [
        "These changes were proposed but the safety review held them back this visit. "
        "They are listed so the reasoning is visible, not as actions to take.",
        "",
    ]
    for rec in recs:
        out.append(f"### {_change_line(rec)}")
        if rec.get("rationale"):
            out.append(f"- **Intended because:** {rec['rationale']}")
        reasons = rec.get("blocked_by") or []
        if reasons:
            out.append("- **Blocked because:**")
            out += [f"  - {reason}" for reason in reasons]
        else:
            out.append("- **Blocked because:** (no reason recorded)")
        if rec.get("risk_note"):
            out.append(f"- **Risk noted by planner:** {rec['risk_note']}")
        out.append("")
    return out


def _unactioned_section(items: list[dict[str, Any]]) -> list[str]:
    out = [f"## Not actioned ({len(items)})", ""]
    if not items:
        out += ["Every active insight was either turned into a recommendation or blocked above.", ""]
        return out
    out += ["Insights the agent saw but did not turn into a change, and why:", ""]
    for item in items:
        label = item.get("title") or item.get("insight_type") or "insight"
        source = item.get("source")
        tag = f" _({source})_" if source else ""
        out.append(f"- **{label}**{tag}: {item.get('reason') or '(no reason recorded)'}")
    out.append("")
    return out


def _summary_section(payload: dict[str, Any]) -> list[str]:
    plan = payload.get("plan") or {}
    trace = payload.get("trace") or {}
    out = ["## Summary", ""]
    if plan.get("summary"):
        out.append(plan["summary"])
    elif trace.get("gate_blocked"):
        out.append("The entry gate stopped the analysis before any recommendation was produced.")
    else:
        out.append("No plan was produced.")
    flags = []
    if trace.get("gate_blocked"):
        flags.append("**Entry gate blocked:** " + "; ".join(trace.get("gate_reasons") or ["(no reason recorded)"]))
    if plan.get("plan_degraded"):
        flags.append("**Plan degraded:** part of the pipeline fell back to a reduced mode; treat figures as indicative.")
    if plan.get("health_breach_unactioned"):
        flags.append("**Health breach not actioned:** a health-tier issue was detected but no safe fix was found for it.")
    if flags:
        out.append("")
        out += [f"- {f}" for f in flags]
    total = sum(
        float(r["usd_cost_annual"])
        for r in plan.get("recommendations") or []
        if isinstance(r.get("usd_cost_annual"), (int, float))
    )
    if total:
        out += ["", f"Estimated annual saving if all recommended changes land: **{_fmt_usd(total)}** "
                    "(sum of per-change estimates; changes may overlap)."]
    out.append("")
    return out


def _data_section(payload: dict[str, Any]) -> list[str]:
    row = payload.get("input") or {}
    cost = payload.get("cost_profile") or {}
    metrics = payload.get("run_metrics") or {}
    service = payload.get("service") or {}
    run_config = payload.get("run_config") or {}
    trace = payload.get("trace") or {}

    out = ["## Data & provenance", ""]
    chosen = service.get("chosen") or {}
    app = f"app_id {row.get('app_id')}"
    if chosen.get("app_name"):
        app = f"app `{chosen['app_name']}` ({app})"
    env = f"env {chosen.get('env_name') or row.get('env_id')}"
    out.append(f"- **Analyzed run:** task_id {payload.get('task_id')} of `{payload.get('task_name')}`"
               f" — {app}, {env}, run at {row.get('app_pit')}")
    if cost.get("cost_per_run_usd") is not None:
        workers = cost.get("cluster_workers")
        avail = cost.get("workers_availability")
        cluster = f", {workers} × {avail or '?'} workers" if workers else ""
        out.append(f"- **Cost per run:** ${float(cost['cost_per_run_usd']):.2f}{cluster}")
    if metrics:
        bits = []
        for key, label in (("idle_ratio", "idle"), ("vcore_utilization", "vCPU util"),
                           ("memory_headroom", "heap headroom"), ("gc_pressure", "GC pressure"),
                           ("skew_ratio", "skew")):
            pct = _fmt_pct(metrics.get(key))
            if pct:
                bits.append(f"{label} {pct}")
        if bits:
            out.append(f"- **Run metrics:** {', '.join(bits)}")
    if trace.get("historical_runs_count") is not None:
        out.append(f"- **Trend evidence:** {trace['historical_runs_count']} earlier run(s) of this task")

    freshness = service.get("athena_freshness") or {}
    note = ("- **Data source: Athena export, not live production.** "
            "The export lags the app; runs newer than the export are not visible here")
    if freshness.get("latest_enrichment_pit"):
        note += f" (newest exported run: {freshness['latest_enrichment_pit']}"
        if freshness.get("latest_insights_snapshot"):
            note += f"; insights snapshot: {freshness['latest_insights_snapshot']}"
        note += ")"
    out.append(note + ".")

    if service.get("requested_task_name") and service.get("candidates"):
        n = len(service["candidates"])
        if n > 1:
            out.append(f"- **Name resolution:** `{service['requested_task_name']}` runs under {n} apps; "
                       f"app_id {row.get('app_id')} was analyzed.")
    if run_config.get("llm_model"):
        out.append(f"- **Model:** {run_config['llm_model']}")
    if payload.get("saved_at"):
        out.append(f"- **Generated:** {payload['saved_at']} (source: {payload.get('source')})")
    out.append("")
    return out


def _no_plan_section(payload: dict[str, Any]) -> list[str]:
    """When the entry gate stops the run there is no plan at all — say so
    once instead of rendering three empty sections that read as 'all clear'."""
    trace = payload.get("trace") or {}
    reasons = trace.get("gate_reasons") or ["(no reason recorded)"]
    out = ["## Not evaluated", "",
           "The agent did not analyze this run, so there is nothing to change, block, or skip yet. "
           "The entry gate stopped it because:", ""]
    out += [f"- {r}" for r in reasons]
    out += ["", "Typical fixes: pick a run that completed, or wait for the Athena export to "
                "catch up if the latest run finished after the export.", ""]
    return out


def build_tldr(payload: dict[str, Any]) -> str:
    plan = payload.get("plan")
    lines = [f"# Recommendations for `{payload.get('task_name') or payload.get('task_id')}`", ""]
    lines += _summary_section(payload)
    if plan is None:
        lines += _no_plan_section(payload)
    else:
        lines += _recommended_section(plan.get("recommendations") or [])
        lines += _blocked_section(plan.get("blocked_recommendations") or [])
        lines += _unactioned_section(plan.get("unactioned_insights") or [])
    lines += _data_section(payload)
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python service/tldr.py <result.json>")
    payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    sys.stdout.write(build_tldr(payload))


if __name__ == "__main__":
    main()
