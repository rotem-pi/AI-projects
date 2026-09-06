"""Build the production tuning-preview plan for one (app_id, task_name) and
save the JSON — the exact payload GET /api/envs/{env}/tuning/preview returns.

Usage:
    ./run.sh run_tuning_preview.py --app-id 21483 --task-name compute

The app's display name (apps.app_name, e.g. "Job Cluster - <job> - <cluster>")
is resolved from the DB and embedded in the output alongside app_id, task_name
and tenant_id; the default output filename is "<app_name>.json".

Runs the real build_preview_plan() from the synced auto-recommendations-agent
worktree (see bootstrap_worktree.py) against the Postgres in PG_DATABASE_URL
(.env, read replica) — same convention as evaluation/compare_sql_stream.py.
Strictly read-only: build_preview_plan is called with its default
save_to_db=False and only ever SELECTs.

app.dal.dal.apply_db_session_ctx is patched to a no-op because calling it for
real requires a request-scoped JWT that doesn't exist outside an HTTP request.
The one session GUC the tuning SQL actually depends on — app.tenant_id, read
by calculate_usd_cost()'s tenant price functions in latest_enrichments.sql —
is set directly here instead, after resolving the tenant from the app_id.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-id", type=int, required=True)
    parser.add_argument(
        "--task-name",
        required=True,
        help="Task name within the app (tasks.task_name, often just 'compute')",
    )
    parser.add_argument(
        "--out",
        help="Output path (default: ~/Downloads/<app_name>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from bootstrap_worktree import resolved_backend_path

    sys.path.insert(0, str(resolved_backend_path()))

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent / ".env")

    database_url = os.getenv("PG_DATABASE_URL")
    if not database_url:
        print("ERROR: PG_DATABASE_URL is not set in .env", file=sys.stderr)
        raise SystemExit(1)
    os.environ["DATABASE_URL"] = database_url

    import app.dal.dal as dal_module

    dal_module.apply_db_session_ctx = lambda: None  # see module docstring

    from app.dal.dal import execute, fetch_one_from_db
    from app.brain.insights.tuning.preview_service import build_preview_plan
    from app.utils.fastapi_ext.middleware.db_context import CursorContextWrapper

    # The DAL reads its cursor from a contextvar that DbContextMiddleware sets
    # per HTTP request; outside a request the wrapper below provides it.
    with CursorContextWrapper():
        app_row = fetch_one_from_db(
            "SELECT a.app_name, e.tenant_id "
            "FROM apps a JOIN envs e USING (env_id) "
            "WHERE a.app_id = %(app_id)s",
            {"app_id": args.app_id},
        )
        if not app_row:
            print(f"ERROR: app_id {args.app_id} not found", file=sys.stderr)
            raise SystemExit(1)
        # calculate_usd_cost()'s tenant price functions read this GUC.
        execute(
            "SELECT set_config('app.tenant_id', %(tenant_id)s, false)",
            {"tenant_id": str(app_row["tenant_id"])},
        )
        plan = build_preview_plan(app_id=args.app_id, task_name=args.task_name)

    # PreviewResult itself carries app_id / app_name / task_name / tenant_id
    # since definity-app commit after 623fb487b; no wrapping needed here.
    out_path = (
        Path(args.out)
        if args.out
        else Path.home() / "Downloads" / f"{app_row['app_name']}.json"
    )
    out_path.write_text(
        json.dumps(plan.model_dump(mode="json"), indent=2, default=str)
    )
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
