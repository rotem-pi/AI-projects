"""Dump everything the definity REST API knows about one task into CSVs that
run_from_dump.py can replay: task detail/params/metrics/tfs/lineage/events/
time-series, plus per-TF detail/events/physical-plan/lineage/stages.

    python tools/dump_rest_api.py 5453                 ->  data/dumps/dump_5453/
    python tools/dump_rest_api.py 5453 --out /tmp/d    ->  /tmp/d/

The token comes from DEFINITY_API_TOKEN (environment, else .env); the base
URL from --base, else DEFINITY_API_BASE, else the default below. Never pass
the token on the command line: it would show up in `ps` and shell history.

dump_task() is the library form service/run_job.py uses for REST-sourced
jobs; it takes the base URL and token explicitly so the service can run
against whichever definity deployment the user points it at. Only the
standard library is used so this also works outside the backend venv
(tools/dump-rest-api.sh delegates here).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from json2csv import write_csv  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE = "https://definity-ai.infra.aks.prod.akamaicsi.net"
RETRIES = 3          # the prod API 500s intermittently
RETRY_SLEEP_S = 2.0
TIMEOUT_S = 60.0
ERROR_PREVIEW_CHARS = 120

TASK_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("", "task.csv"),
    ("/params", "task_params.csv"),
    ("/metrics", "task_metrics.csv"),
    ("/tfs", "task_tfs.csv"),
    ("/lineage", "task_lineage.csv"),
    ("/events", "task_events.csv"),
    ("/time-series-metrics", "task_tsm.csv"),
)
TF_ENDPOINTS: tuple[tuple[str, str], ...] = (
    ("", "tf_{tf}.csv"),
    ("/events", "tf_{tf}_events.csv"),
    ("/physical-plan", "tf_{tf}_physical_plan.csv"),
    ("/lineage", "tf_{tf}_lineage.csv"),
    ("/stages", "tf_{tf}_stages.csv"),
)

Log = Callable[[str], None]


class RestDumpError(RuntimeError):
    """The task itself could not be fetched (bad token, unknown id, unreachable host)."""

    def __init__(self, status: int | None, message: str):
        super().__init__(message)
        self.status = status


def env_from_dotenv() -> dict[str, str]:
    """DEFINITY_* values from .env without executing it (the file may hold
    other tools' settings)."""
    out: dict[str, str] = {}
    path = ROOT / ".env"
    if not path.is_file():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        if key.strip() in ("DEFINITY_API_TOKEN", "DEFINITY_API_BASE"):
            out[key.strip()] = value.strip().strip("'\"")
    return out


def normalize_base(base: str) -> str:
    base = base.strip().rstrip("/")
    if not base.startswith(("http://", "https://")):
        base = "https://" + base
    return base


def fetch_json(base: str, path: str, token: str) -> tuple[int, Any, str]:
    """(http status, parsed JSON or None, error preview). Retries non-200s."""
    url = f"{base}{path}"
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {token}",
                                               "Accept": "application/json"})
    status, preview = 0, ""
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                body = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            status, body = exc.code, exc.read()
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            status, body = 0, str(exc).encode()
        if status == 200:
            try:
                return status, json.loads(body), ""
            except ValueError:
                return status, None, "response is not JSON"
        preview = body.decode(errors="replace")[:ERROR_PREVIEW_CHARS]
        # 4xx are deterministic (bad token / unknown id): retrying cannot help.
        if 400 <= status < 500:
            break
        if attempt < RETRIES:
            time.sleep(RETRY_SLEEP_S)
    return status, None, preview


def probe_task(base: str, task_id: int, token: str) -> dict[str, Any]:
    """GET /api/tasks/{id}: the task's own record, or RestDumpError with the
    HTTP status so callers can tell a bad token (401) from a wrong id (404)
    from an unreachable host (0)."""
    base = normalize_base(base)
    status, data, preview = fetch_json(base, f"/api/tasks/{task_id}", token)
    if status == 200 and isinstance(data, dict):
        return data
    if status == 200:
        raise RestDumpError(status, f"GET /api/tasks/{task_id} returned unexpected JSON")
    reason = {0: "host unreachable", 401: "token rejected", 403: "token not allowed",
              404: "no such task"}.get(status, f"HTTP {status}")
    raise RestDumpError(status, f"GET {base}/api/tasks/{task_id}: {reason}"
                                f"{' — ' + preview if preview else ''}")


def _hit(base: str, path: str, token: str, out: Path, log: Log) -> bool:
    status, data, preview = fetch_json(base, path, token)
    if data is None:
        why = "CONV-FAIL" if status == 200 else "HTTP-ERR  "
        log(f"[{status:>3}] {why} {path}  -> {preview}")
        return False
    write_csv(data, out)
    log(f"[{status:>3}] {out.stat().st_size:>8}B  {path}")
    return True


def _tf_ids(task_tfs_csv: Path) -> list[int]:
    """tf_ids from the already-dumped task_tfs.csv rather than a second live
    call that could hit the flaky 500."""
    import csv
    try:
        with task_tfs_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except OSError:
        return []
    return sorted({int(r["tf_id"]) for r in rows if (r.get("tf_id") or "").isdigit()})


def dump_task(task_id: int, out_dir: Path, *, base: str, token: str,
              log: Log = lambda m: print(m, file=sys.stderr, flush=True)) -> dict[str, Any]:
    """Dump one task (and its TFs) to out_dir. Raises RestDumpError when the
    task record itself cannot be fetched; individual sub-endpoints failing is
    logged and reported in the returned dict, not fatal (run_from_dump.py
    degrades per missing file)."""
    base = normalize_base(base)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = probe_task(base, task_id, token)
    write_csv(task, out_dir / "task.csv")
    log(f"### TASK {task_id} ({task.get('task_name')}) from {base} ###")
    log(f"[200] {(out_dir / 'task.csv').stat().st_size:>8}B  /api/tasks/{task_id}")

    failed: list[str] = []
    for suffix, name in TASK_ENDPOINTS[1:]:
        path = f"/api/tasks/{task_id}{suffix}"
        if not _hit(base, path, token, out_dir / name, log):
            failed.append(path)

    tf_ids = _tf_ids(out_dir / "task_tfs.csv")
    log(f"### TFs: {tf_ids} ###")
    for tf in tf_ids:
        for suffix, name in TF_ENDPOINTS:
            path = f"/api/tfs/{tf}{suffix}"
            if not _hit(base, path, token, out_dir / name.format(tf=tf), log):
                failed.append(path)
    log(f"### DONE -> {out_dir} ({len(failed)} endpoint(s) failed) ###")
    return {"analyzed_task_id": task_id, "task_name": task.get("task_name"),
            "app_id": task.get("app_id"), "out_dir": str(out_dir), "api_base": base,
            "tf_ids": tf_ids, "failed_endpoints": failed}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("task_id", type=int)
    parser.add_argument("--out", type=Path, help="default data/dumps/dump_<task_id>")
    parser.add_argument("--base", help="API base URL (default DEFINITY_API_BASE)")
    args = parser.parse_args()

    dotenv = env_from_dotenv()
    token = os.environ.get("DEFINITY_API_TOKEN") or dotenv.get("DEFINITY_API_TOKEN")
    if not token:
        parser.error("set DEFINITY_API_TOKEN in the environment or .env (see .env.example)")
    base = args.base or os.environ.get("DEFINITY_API_BASE") or dotenv.get("DEFINITY_API_BASE") \
        or DEFAULT_BASE
    out = args.out or ROOT / "data" / "dumps" / f"dump_{args.task_id}"
    try:
        dump_task(args.task_id, out, base=base, token=token)
    except RestDumpError as exc:
        sys.exit(f"error: {exc}")


if __name__ == "__main__":
    main()
