"""Runs the literal production fetch_context -> entry_gate -> sql_stream nodes
against a real Postgres database for one task_id, and dumps their outputs as
JSON. Invoked by compare_sql_stream.py as a subprocess, in the definity-app
backend venv, with cwd=backend/ (so app.config resolves backend/.env) and
DATABASE_URL pointing at whichever Postgres to inspect.

    DATABASE_URL=... backend/.venv/bin/python pg_nodes_runner.py \
        --task-id 1473514 --out /path/out.json

The nodes enforce tenant isolation via GET_TENANT_ID(), a Postgres session
variable normally set by session/session_init.sql as part of a real HTTP
request (see apply_db_session_ctx in app/dal/dal.py). Outside a request
there is no access token to derive that from, so this script resolves the
task's tenant_id via a plain identity query (tenant scoping on tasks/apps/
envs is enforced by explicit WHERE clauses in app SQL, not native Postgres
RLS, so this read needs no session context) and then calls session_init.sql
itself with that tenant's real license flags -- the same session state a
real request for that tenant would establish, without impersonating a user
or minting any token.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# python puts the script's own directory (AutoRecommendation/) on sys.path,
# not cwd -- but this must run with cwd=backend/ so app.config resolves
# backend/.env, so "app" (the backend package) needs cwd added explicitly.
sys.path.insert(0, os.getcwd())


def _serialize(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _serialize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize(v) for v in value]
    return value


def _resolve_identity(fetch_from_db, task_id: int) -> dict:
    rows = fetch_from_db(
        """
        SELECT t.tenant_id, t.app_id, t.task_name, a.app_name, e.env_name
        FROM tasks t
        JOIN apps a ON a.app_id = t.app_id
        JOIN envs e ON e.env_id = a.env_id
        WHERE t.task_id = %(task_id)s
        """,
        {"task_id": task_id},
    )
    if not rows:
        raise SystemExit(f"task_id={task_id} not found")
    return dict(rows[0])


def _init_session(fetch_sql_file, get_tenant_license, tenant_id: int) -> None:
    """Establish the same Postgres session GUCs (app.tenant_id and friends)
    that apply_db_session_ctx sets for a real request, using that tenant's
    actual license flags -- no auth token involved."""
    license_features = get_tenant_license(tenant_id)
    fetch_sql_file(
        "session/session_init",
        {
            "is_session": False,
            "tenant_id": str(tenant_id),
            "count_allowed_pipelines": str(license_features.count_allowed_pipelines),
            "is_insights_allowed": str(license_features.is_insights_allowed),
            "is_recommendations_allowed": str(
                license_features.is_recommendations_allowed
            ),
        },
    )


def _run_sql_stream_with_fallback(sql_stream, dal_module, state: dict) -> tuple[dict, list[str]]:
    """The literal production insights.sql references insights.auto_tune_supported,
    a column that may not exist yet on every environment (e.g. a read replica
    pending migration). Retry once with that column stripped so the run can
    still complete — but say so, since that's no longer the literal query."""
    try:
        return sql_stream(state), []
    except Exception as e:  # noqa: BLE001 - deliberately broad, we branch on message content
        if "auto_tune_supported" not in str(e):
            raise
        note = (
            "DIAGNOSTIC FALLBACK USED: insights.auto_tune_supported is missing on this "
            "database — retried sql_stream with that column stripped from insights.sql. "
            "This is NOT proof the literal production query succeeds; re-test once the "
            "schema is migrated."
        )
        original_read = dal_module.read_sql_file

        def patched_read(sql_name, stack=None):
            content = original_read(sql_name, stack)
            if sql_name == "insights":
                content = content.replace("insights.auto_tune_supported,\n", "")
            return content

        dal_module.read_sql_file = patched_read
        try:
            return sql_stream(state), [note]
        finally:
            dal_module.read_sql_file = original_read


def _patch_sandbox_to_local(sandbox_dal_module, out_dir: Path) -> Path:
    """fetch_context dumps every raw per-run table to S3 via
    sandbox_dal.write_table. Local inspection runs must not touch S3 (for
    now), so write the same CSVs under <out_dir>/sandbox/ instead — same
    keys, same csv serialization, no AWS credentials or network needed."""
    sandbox_dir = out_dir / "sandbox"

    def local_write_table(task_id: int, table: str, rows: list) -> str:
        key = sandbox_dal_module._table_key(task_id, table)
        path = sandbox_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(sandbox_dal_module._rows_to_csv(rows))
        return key

    sandbox_dal_module.write_table = local_write_table
    return sandbox_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    import app.dal.dal as dal_module
    import app.dal.sandbox_dal as sandbox_dal
    from app.brain.insights.agent.nodes.fetch_context import fetch_context
    from app.brain.insights.agent.nodes.entry_gate import entry_gate
    from app.brain.insights.agent.nodes.sql_stream import run_sql_stream as sql_stream
    from app.config import settings
    from app.dal.dal import fetch_from_db, fetch_sql_file
    from app.dal.tenant_dal import get_tenant_license
    from app.utils.fastapi_ext.middleware.db_context import CursorContextWrapper

    sandbox_dir = _patch_sandbox_to_local(sandbox_dal, args.out.parent)

    with CursorContextWrapper():
        identity = _resolve_identity(fetch_from_db, args.task_id)
        _init_session(fetch_sql_file, get_tenant_license, identity["tenant_id"])

        state: dict = {"task_id": args.task_id, "env_name": identity["env_name"]}
        state.update(fetch_context(state))
        state.update(entry_gate(state))

        sql_result, diagnostic_notes = _run_sql_stream_with_fallback(
            sql_stream, dal_module, state
        )
        state.update(sql_result)

    database_url = settings.database_url.get_secret_value()
    database_url_host = database_url.split("@")[-1].split("/")[0]

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "database_url_host": database_url_host,
        "task_id": args.task_id,
        "analyzed_task_id": state["task_id"],
        "identity": {
            "env_name": identity["env_name"],
            "tenant_id": identity["tenant_id"],
            "app_id": identity["app_id"],
            "app_name": identity["app_name"],
            "task_name": identity["task_name"],
        },
        "diagnostic_notes": diagnostic_notes,
        "fetch_context": _serialize(
            {
                "task_id": state["task_id"],
                "latest_enrichment": state.get("latest_enrichment"),
                "run_metrics": state.get("run_metrics"),
                "task_profile": state.get("task_profile"),
                "historical_runs": state.get("historical_runs"),
                "run_sandbox": state.get("run_sandbox"),
            }
        ),
        "sandbox_local_dir": str(sandbox_dir),
        "entry_gate": {
            "gate_blocked": state.get("gate_blocked"),
            "gate_reasons": state.get("gate_reasons"),
        },
        "sql_stream": _serialize(
            {
                "sql_insights": state.get("sql_insights"),
                "sql_recommendations": state.get("sql_recommendations"),
            }
        ),
    }

    args.out.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
