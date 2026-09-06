"""Inference graph — the repo graph rebuilt with progress-printing wrappers.

Mirrors backend/app/brain/insights/agent/inference_graph.py on the
auto-recommendations-agent branch (same node names, same edges, same
routing — fetch_context → entry_gate → sql_stream → discovery_stream →
merge_insights → triage → plan → safety_check → explain → assemble_output;
merge/route helpers and _empty_state copied — not imported, to avoid the
repo module's app.config/app.dal dependencies), with two local
substitutions:

- fetch_context reads the Athena row/history/run-sandbox injected via
  set_row_override() instead of Postgres (nodes/fetch_context.py); the
  run-sandbox tables are materialized locally instead of from S3 (see
  agent/local_sandbox.py, imported here for its import-time
  sandbox_dal.materialize_to_folder patch);
- sql_stream reads the task's real Athena `insights` rows injected via
  set_insights_override() (nodes/sql_stream.py).

Every other node — entry_gate, discovery_stream, triage, plan,
safety_check, explain, assemble_output — is imported directly from the
repo, so their behavior (prompts, tools, KB, blocking rules, structured
output) is exactly what the branch ships.

LLM is configured from environment variables (see .env.example); the
default model string matches the repo's settings.llm_model.
"""

from __future__ import annotations

import contextvars
import functools
import logging
import os
import sys
import time

from langchain.chat_models import init_chat_model
from langgraph.graph import END, StateGraph

from app.brain.insights.agent.knowledge_base import KnowledgeBase, load_knowledge_base
from app.brain.insights.agent.models import (
    AgentInsight,
    FinalPlan,
    SqlInsight,
)
from app.brain.insights.agent.nodes.assemble_output import assemble_output
from app.brain.insights.agent.nodes.discovery import run_discovery_stream
from app.brain.insights.agent.nodes.entry_gate import entry_gate
from app.brain.insights.agent.nodes.explain import explain
from app.brain.insights.agent.nodes.plan import plan
from app.brain.insights.agent.nodes.safety import safety_check
from app.brain.insights.agent.nodes.triage import triage
from app.brain.insights.agent.state import AgentState
import agent.local_sandbox  # noqa: F401 — patches sandbox_dal.materialize_to_folder at import time
from agent.nodes.fetch_context import fetch_context
from agent.nodes.sql_stream import run_sql_stream

logger = logging.getLogger(__name__)

# Bedrock inference-profile ID for the same model the repo's settings.llm_model
# defaults to (app/config.py: "anthropic:claude-sonnet-4-6"). The bare model ID
# ("anthropic.claude-sonnet-4-6") exists in eu-north-1 but only supports the
# INFERENCE_PROFILE invocation type, not direct on-demand invocation — the
# "eu." cross-region inference profile is the one that's actually invokable
# (confirmed via `aws bedrock list-inference-profiles`). Override via
# LLM_MODEL if you're on a different region/profile.
_DEFAULT_LLM_MODEL = "eu.anthropic.claude-sonnet-4-6"

_show_progress: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "show_progress", default=True
)

_NODE_LABELS: dict[str, str] = {
    "fetch_context": "Fetch context",
    "entry_gate": "Entry gate",
    "sql_stream": "SQL rules",
    "discovery_stream": "Agent discovery",
    "merge_insights": "Merge insights",
    "triage": "Triage",
    "plan": "Plan recommendations",
    "safety_check": "Safety check",
    "explain": "Explain plan",
    "assemble_output": "Assemble output",
}


def _node_summary(name: str, result: dict) -> str:
    if name == "fetch_context":
        if result.get("run_metrics") is None:
            return "no enrichment found"
        parts = []
        duration_s = (result.get("latest_enrichment") or {}).get("task__duration")
        if duration_s is not None:  # task_enrichments.task__duration is in seconds
            parts.append(f"duration {duration_s / 60:.0f}m")
        parts.append(f"{len(result.get('historical_runs', []))} historical run(s)")
        run_sandbox = result.get("run_sandbox")
        table_count = len(run_sandbox.tables) if run_sandbox is not None else 0
        parts.append(f"{table_count} sandbox table(s)")
        return ", ".join(parts)
    if name == "entry_gate":
        if result.get("gate_blocked"):
            return f"blocked — {'; '.join(result.get('gate_reasons', []))}"
        return "open"
    if name == "sql_stream":
        return (
            f"{len(result.get('sql_insights', []))} insight(s), "
            f"{len(result.get('sql_recommendations', []))} engine rec(s)"
        )
    if name == "discovery_stream":
        return f"{len(result.get('agent_insights', []))} insight(s)"
    if name == "merge_insights":
        return f"{len(result.get('merged_insights', []))} merged"
    if name == "triage":
        triage_result = result.get("triage_result")
        if triage_result is not None:
            counts = "/".join(
                str(len(triage_result.insights_for_tier(tier))) for tier in range(1, 5)
            )
            summary = f"tiers 1-4: {counts}"
            if triage_result.active_conflicts:
                summary += f", {len(triage_result.active_conflicts)} conflict(s)"
            return summary
        return ""
    if name == "plan":
        return f"{len(result.get('planned_recommendations', []))} recommendation(s)"
    if name == "safety_check":
        safe = len(result.get("safe_recommendations", []))
        blocked = len(result.get("blocked_recommendations", []))
        return f"{safe} safe, {blocked} blocked"
    if name == "explain":
        final_plan = result.get("final_plan")
        if final_plan is not None:
            return f"{len(final_plan.recommendations)} explained"
        return ""
    if name == "assemble_output":
        final_plan = result.get("final_plan")
        if final_plan is not None:
            return f"done, {len(final_plan.unactioned_insights)} unactioned insight(s)"
        return ""
    return ""


def _wrap_node(name: str, fn):
    label = _NODE_LABELS.get(name, name)

    @functools.wraps(fn)
    def wrapper(state: AgentState) -> dict:
        if _show_progress.get():
            print(f"  → {label}…", file=sys.stderr, flush=True)
        started = time.perf_counter()
        result = fn(state)
        if _show_progress.get():
            elapsed = time.perf_counter() - started
            summary = _node_summary(name, result)
            detail = f" — {summary}" if summary else ""
            print(f"  ✓ {label} ({elapsed:.1f}s{detail})", file=sys.stderr, flush=True)
        return result

    return wrapper


# _merge_insights / routes / _empty_state are copied from
# the repo's inference_graph rather than imported: importing that module would
# pull in app.config and app.dal (settings, psycopg) — the DB-tied
# dependencies this harness exists to avoid.  Keep them in sync with the repo
# when it changes.


def _merge_insights(state: AgentState) -> dict:
    seen_keys: set[str] = set()
    merged: list[SqlInsight | AgentInsight] = []

    for insight in state.get("sql_insights", []):
        key = insight.type
        if key not in seen_keys:
            seen_keys.add(key)
            merged.append(insight)

    for insight in state.get("agent_insights", []):
        if insight.title not in seen_keys:
            seen_keys.add(insight.title)
            merged.append(insight)

    return {"merged_insights": merged}


def _route_after_entry_gate(state: AgentState) -> str:
    return END if state.get("gate_blocked") else "sql_stream"


def _route_after_merge_insights(state: AgentState) -> str:
    """Skip the reasoning nodes (triage → … → explain) when there is nothing
    to reason about — a clean run costs zero LLM calls past discovery."""
    return "triage" if state.get("merged_insights") else "assemble_output"


def _empty_state(task_id: int, env_name: str) -> AgentState:
    return AgentState(
        task_id=task_id,
        env_name=env_name,
        latest_enrichment=None,
        run_metrics=None,
        task_profile=None,
        historical_runs=[],
        run_sandbox=None,
        gate_blocked=False,
        gate_reasons=[],
        sql_insights=[],
        sql_recommendations=[],
        agent_insights=[],
        merged_insights=[],
        triage_result=None,
        planned_recommendations=[],
        skipped_insight_reasons={},
        plan_degraded=False,
        safe_recommendations=[],
        blocked_recommendations=[],
        final_plan=None,
        messages=[],
    )


def _build_llm():
    model = os.getenv("LLM_MODEL", _DEFAULT_LLM_MODEL)
    # Claude via Amazon Bedrock — reuses the same AWS_PROFILE/AWS_REGION
    # credential chain already used for Athena (main.py), so no separate
    # Anthropic API key is needed. The default botocore read_timeout (60s) is
    # too tight for this agent's longer tool-heavy generations and was
    # observed to time out under load (mirrors the repo's own _build_llm on
    # auto-recommendations-agent).
    #
    # Passing config= through init_chat_model(model_provider="bedrock_converse", ...)
    # is NOT enough: ChatBedrockConverse only applies self.config when it
    # builds its OWN client (validate_environment(), gated on
    # `self.client is None`), and _get_effective_config() re-derives the
    # botocore Config from self.timeout/self.max_retries whenever either is
    # non-None — a later .bind()/retry wrapper setting either of those
    # silently drops our read_timeout back to botocore's 60s default.
    # Building the boto3 client ourselves and handing it to
    # ChatBedrockConverse(client=...) sidesteps validate_environment's
    # client-building branch entirely, so no later rebind can ever recompute
    # (and reset) the timeout.
    from botocore.config import Config
    from langchain_aws.utils import create_aws_client

    region = os.getenv("AWS_REGION", "eu-north-1")
    profile = os.getenv("AWS_PROFILE")
    bedrock_client = create_aws_client(
        service_name="bedrock-runtime",
        region_name=region,
        credentials_profile_name=profile,
        config=Config(read_timeout=600, connect_timeout=30),
    )
    bedrock_llm = init_chat_model(
        model,
        model_provider="bedrock_converse",
        client=bedrock_client,
        # Deterministic sampling by default: run-to-run recommendation
        # agreement is a shipping gate (evaluation/consistency_gate.py), and
        # temperature is the cheapest variance to remove. Override via
        # LLM_TEMPERATURE.
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
    )
    # cached_system_message() (prompts.py) embeds Anthropic-native per-block
    # cache_control, which ChatBedrockConverse silently drops — it only
    # inserts cachePoint blocks via this bind-time kwarg. Without it, every
    # node reprocesses the full KB-laden system prompt from scratch on every
    # call instead of hitting the cache, which is why multi-insight tasks'
    # plan step was consistently timing out.
    return bedrock_llm.bind(cache_control={"type": "default"})


def build_inference_graph(llm=None, kb: KnowledgeBase | None = None):
    if kb is None:
        kb = load_knowledge_base()
    if llm is None:
        llm = _build_llm()

    graph = StateGraph(AgentState)

    graph.add_node("fetch_context", _wrap_node("fetch_context", fetch_context))
    graph.add_node("entry_gate", _wrap_node("entry_gate", entry_gate))
    graph.add_node("sql_stream", _wrap_node("sql_stream", run_sql_stream))
    graph.add_node(
        "discovery_stream",
        _wrap_node(
            "discovery_stream",
            functools.partial(run_discovery_stream, llm=llm, kb=kb),
        ),
    )
    graph.add_node("merge_insights", _wrap_node("merge_insights", _merge_insights))
    graph.add_node(
        "triage", _wrap_node("triage", functools.partial(triage, llm=llm, kb=kb))
    )
    graph.add_node("plan", _wrap_node("plan", functools.partial(plan, llm=llm, kb=kb)))
    graph.add_node(
        "safety_check",
        _wrap_node("safety_check", functools.partial(safety_check, llm=llm, kb=kb)),
    )
    graph.add_node(
        "explain", _wrap_node("explain", functools.partial(explain, llm=llm, kb=kb))
    )
    graph.add_node("assemble_output", _wrap_node("assemble_output", assemble_output))

    graph.set_entry_point("fetch_context")
    graph.add_edge("fetch_context", "entry_gate")
    graph.add_conditional_edges(
        "entry_gate",
        _route_after_entry_gate,
        {"sql_stream": "sql_stream", END: END},
    )
    graph.add_edge("sql_stream", "discovery_stream")
    graph.add_edge("discovery_stream", "merge_insights")
    graph.add_conditional_edges(
        "merge_insights",
        _route_after_merge_insights,
        {"triage": "triage", "assemble_output": "assemble_output"},
    )
    graph.add_edge("triage", "plan")
    graph.add_edge("plan", "safety_check")
    graph.add_edge("safety_check", "explain")
    graph.add_edge("explain", "assemble_output")
    graph.add_edge("assemble_output", END)

    return graph.compile()


_compiled_graph = None


def get_inference_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_inference_graph()
    return _compiled_graph


async def run_analysis(
    task_id: int,
    env_name: str = "athena",
    *,
    show_progress: bool = True,
) -> tuple[FinalPlan | None, AgentState]:
    """Returns (final_plan, full_graph_state) — the full state lets callers
    persist triage/merge diagnostics that FinalPlan alone drops.

    The repo's run_analysis(task_id, env_name) returns only the plan; the
    extra state and progress printing are harness additions."""
    token = _show_progress.set(show_progress)
    try:
        if show_progress:
            print(f"\n  Analysing task {task_id}…", file=sys.stderr, flush=True)
        result = await get_inference_graph().ainvoke(_empty_state(task_id, env_name))
        return result.get("final_plan"), result
    finally:
        _show_progress.reset(token)
