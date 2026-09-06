#!/usr/bin/env bash
# Entry point for every agent invocation: syncs the dedicated
# auto-recommendations-agent worktree (see bootstrap_worktree.py) first, so
# the backend `uv run --project` resolves dependencies from and the backend
# agent/'s symlinks point at are always the exact same commit — regardless of
# what's checked out in your own DEFINITY_APP_REPO clone or your CWD.
#
# Usage:
#   ./run.sh main.py --sample 10
#   ./run.sh run_from_pg_dump.py data/dumps/pg_2681247
#   ./run.sh run_from_dump.py data/dumps/dump_5453
#   ./run.sh --no-sync run_from_pg_dump.py data/dumps/pg_2681247   # skip the fetch (offline)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

NO_SYNC=""
if [[ "${1:-}" == "--no-sync" ]]; then
  NO_SYNC="--no-sync"
  shift
fi

python3 bootstrap_worktree.py $NO_SYNC
BACKEND=$(cat .worktrees/.backend_path)

exec uv run --project "$BACKEND" python "$@"
