#!/bin/bash
# Dump everything the definity REST API knows about one task into CSVs that
# run_from_dump.py can replay. Kept for the familiar invocation; the work is
# done by tools/dump_rest_api.py (also used by the web service), so the list
# of endpoints lives in one place.
#
#   ./tools/dump-rest-api.sh 5453     ->  data/dumps/dump_5453/
#
# Requires DEFINITY_API_TOKEN (and optionally DEFINITY_API_BASE) in .env or
# the environment. Never hardcode the token here — this file is committed.
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$DIR/dump_rest_api.py" "$@"
