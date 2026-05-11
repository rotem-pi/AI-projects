#!/usr/bin/env bash
# Run test_test_generation_8f5e1c0_impact.py using definity-app/backend/.venv.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${DEFINITY_BACKEND_ROOT:-}" ]]; then
  BACKEND="$(cd "${DEFINITY_BACKEND_ROOT}" && pwd)"
else
  BACKEND=""
  d="$ROOT"
  while [[ "$d" != "/" ]]; do
    if [[ -d "$d/definity-app/backend/app" ]]; then
      BACKEND="$d/definity-app/backend"
      break
    fi
    d="$(dirname "$d")"
  done
fi

if [[ -z "$BACKEND" || ! -d "$BACKEND/app" ]]; then
  echo "Could not find definity-app/backend. Set DEFINITY_BACKEND_ROOT." >&2
  exit 1
fi

if [[ ! -x "$BACKEND/.venv/bin/pytest" ]]; then
  echo "Missing $BACKEND/.venv/bin/pytest" >&2
  exit 1
fi

export PYTHONPATH="${ROOT}:${BACKEND}"
export DATABASE_URL="${DATABASE_URL:-postgresql://pytest_placeholder:pytest@127.0.0.1:5432/pytest_no_db}"
export IS_SAAS="${IS_SAAS:-true}"

exec "$BACKEND/.venv/bin/pytest" --confcutdir="$ROOT" \
  "$ROOT/test_test_generation_8f5e1c0_impact.py" "$@"
