#!/bin/bash
# Dump everything the definity REST API knows about one task into CSVs that
# run_from_dump.py can replay: task detail/params/metrics/tfs/lineage/events/
# time-series, plus per-TF detail/events/physical-plan/lineage/stages.
#
#   ./tools/dump-rest-api.sh 5453     ->  data/dumps/dump_5453/
#
# Requires DEFINITY_API_TOKEN (and optionally DEFINITY_API_BASE) in .env or
# the environment. Never hardcode the token here — this file is committed.
set -u

DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$DIR")"

# Pull DEFINITY_* vars from .env without executing it wholesale.
if [ -f "$ROOT/.env" ]; then
  while IFS='=' read -r k v; do
    case "$k" in DEFINITY_API_TOKEN|DEFINITY_API_BASE) export "$k"="$v" ;; esac
  done < "$ROOT/.env"
fi

BASE="${DEFINITY_API_BASE:-https://definity-ai.infra.aks.prod.akamaicsi.net}"
API="${DEFINITY_API_TOKEN:?Set DEFINITY_API_TOKEN in .env (see .env.example)}"

TID="${1:?Usage: $0 <task_id>}"
OUT="$ROOT/data/dumps/dump_$TID"
mkdir -p "$OUT"

# Fetch a JSON endpoint and convert the response to CSV.
#   - list of objects  -> one row per object, columns = union of keys
#   - single object     -> two columns: key,value (nested values JSON-encoded)
# Nested/complex cell values are serialized back to compact JSON.
# The prod API 500s intermittently, so each endpoint is tried up to 3 times.
hit() { # path outfile(.csv)
  local url="$BASE$1"
  local try code http body
  for try in 1 2 3; do
    code=$(curl -s -H "Authorization: Bearer $API" -w $'\n%{http_code}' "$url")
    http="${code##*$'\n'}"
    body="${code%$'\n'*}"
    [ "$http" = "200" ] && break
    sleep 2
  done
  if [ "$http" != "200" ]; then
    printf "[%s] HTTP-ERR   %s  -> %s\n" "$http" "$1" "$(printf '%s' "$body" | head -c 120)"
    return
  fi
  printf '%s' "$body" | python3 "$DIR/json2csv.py" "$OUT/$2"
  local rc=$?
  local size; size=$(wc -c < "$OUT/$2" 2>/dev/null | tr -d ' ')
  if [ "$rc" -ne 0 ]; then
    printf "[%s] CONV-FAIL  %s\n" "$http" "$1"
  else
    printf "[%s] %8sB  %s\n" "$http" "$size" "$1"
  fi
}

echo "### TASK $TID ###"
hit "/api/tasks/$TID"                     task.csv
hit "/api/tasks/$TID/params"              task_params.csv
hit "/api/tasks/$TID/metrics"             task_metrics.csv
hit "/api/tasks/$TID/tfs"                 task_tfs.csv
hit "/api/tasks/$TID/lineage"             task_lineage.csv
hit "/api/tasks/$TID/events"              task_events.csv
hit "/api/tasks/$TID/time-series-metrics" task_tsm.csv

echo "### TFs ###"
# tf_ids come from the already-dumped (retry-hardened) task_tfs.csv rather
# than a second live call that could hit the flaky 500.
TFIDS=$(python3 -c "
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
except OSError:
    rows = []
ids = {int(r['tf_id']) for r in rows if (r.get('tf_id') or '').isdigit()}
print(' '.join(str(x) for x in sorted(ids)))
" "$OUT/task_tfs.csv")
echo "tf_ids: $TFIDS"
for tf in $TFIDS; do
  hit "/api/tfs/$tf"               tf_${tf}.csv
  hit "/api/tfs/$tf/events"        tf_${tf}_events.csv
  hit "/api/tfs/$tf/physical-plan" tf_${tf}_physical_plan.csv
  hit "/api/tfs/$tf/lineage"       tf_${tf}_lineage.csv
  hit "/api/tfs/$tf/stages"        tf_${tf}_stages.csv
done
echo "### DONE -> $OUT ###"
