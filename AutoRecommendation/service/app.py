"""Recommendation Agent — the self-serve web service over the AutoRecommendation harness.

    ./run.sh --no-sync -m uvicorn service.app:app --host 0.0.0.0 --port 8765

One FastAPI process serves the page (service/static/index.html), answers the
Athena lookups (task-name search, per-app candidates, export freshness) and
runs each analysis as its **own subprocess** of service/run_job.py — the
harness injects data through module globals, so two analyses must never share
a process. Job state lives on disk under REC_AGENT_JOBS_DIR/<job_id>/:

    job.json     status, request, timings, exit code   (this module)
    log.txt      run_job.py's stderr, streamed          (progress source)
    result.json  full saved result                     (run_job.py)
    tldr.md      deterministic TL;DR                   (run_job.py)
    dump/ sandbox/ result/                             (run_job.py)

so a restart of the web process loses nothing that hasn't been swept.
Retention is REC_AGENT_TTL_HOURS after a job finishes (default 4), enforced
by a background sweeper; DELETE /api/jobs/{id} removes one immediately.

AWS: the process uses the AWS_PROFILE SSO profile with AWS_SSO_AUTO_LOGIN=0
(main.py never shells out to `aws sso login` on its own here). When the
session is missing/expired, POST /api/aws/login starts
`aws sso login --profile <p> --use-device-code --no-browser`, parses the
verification URL + user code from its stdout for the page to show, and
reports when the CLI finishes. The resulting token is cached by the CLI in
~/.aws/sso/cache and shared by every user of this site — acceptable behind
the team's own SSO on the ingress, which is the deployment this targets.

Environment (all optional):
    REC_AGENT_JOBS_DIR        default <repo>/data/jobs
    REC_AGENT_TTL_HOURS       default 4
    REC_AGENT_MAX_CONCURRENT  default 2 concurrent analyses
    REC_AGENT_JOB_TIMEOUT_MIN default 30 — a job running longer is killed
    AWS_PROFILE               default dev-admin (same as main.py)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AWS_SSO_AUTO_LOGIN", "0")

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

import main as _main  # noqa: E402 — .env, sys.path to the backend, Athena helpers

STATIC = Path(__file__).parent / "static"
JOBS_DIR = Path(os.getenv("REC_AGENT_JOBS_DIR") or ROOT / "data" / "jobs").resolve()
TTL = timedelta(hours=float(os.getenv("REC_AGENT_TTL_HOURS", "4")))
MAX_CONCURRENT = int(os.getenv("REC_AGENT_MAX_CONCURRENT", "2"))
JOB_TIMEOUT = timedelta(minutes=float(os.getenv("REC_AGENT_JOB_TIMEOUT_MIN", "30")))
AWS_PROFILE = os.getenv("AWS_PROFILE", "dev-admin")
SWEEP_EVERY_S = 300
FRESHNESS_CACHE_S = 600
AWS_CHECK_CACHE_S = 60

# Lines of run_job.py's stderr worth showing as progress (everything else —
# the backend's settings dump, Athena polling dots, recursion-limit notes —
# stays in log.txt for debugging).
_PROGRESS_RE = re.compile(
    r"^\s*(→|✓|Checking|Resolving|->|Dumping|Running the agent|Wrote|sandbox table|"
    r"Historical runs|Insights:|Sandbox tables|Task \d|AWS SSO|Traceback|\w+Error\b|"
    r"No analyzable|Task '.*' has no)"
)
_JOB_ID_RE = re.compile(r"^[a-z0-9]{10}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime | None) -> str | None:
    return dt.isoformat(timespec="seconds") if dt else None


def _parse(ts: str | None) -> datetime | None:
    return datetime.fromisoformat(ts) if ts else None


# ── AWS session ──────────────────────────────────────────────────────────────

class _AwsState:
    """Cached identity check + the single in-flight device-code login."""

    def __init__(self) -> None:
        self.checked_at = 0.0
        self.status: dict[str, Any] = {"state": "unknown"}
        self.login: dict[str, Any] | None = None  # {url, code, started_at, done, ok, error}
        self._proc: asyncio.subprocess.Process | None = None

    def _sso_expiry(self) -> str | None:
        """expiresAt of the newest SSO access token for our start URL."""
        cache = Path.home() / ".aws" / "sso" / "cache"
        best: str | None = None
        for f in cache.glob("*.json"):
            try:
                d = json.loads(f.read_text())
            except (OSError, ValueError):
                continue
            if "accessToken" in d and d.get("expiresAt"):
                best = max(best or "", d["expiresAt"])
        return best

    def _check_sync(self) -> dict[str, Any]:
        try:
            import boto3
            session = boto3.Session(profile_name=AWS_PROFILE, region_name=_main._AWS_REGION)
            ident = session.client("sts").get_caller_identity()
            arn = ident.get("Arn", "")
            return {"state": "ok", "profile": AWS_PROFILE, "identity": arn.rsplit("/", 1)[-1],
                    "expires_at": self._sso_expiry()}
        except Exception as exc:
            return {"state": "expired", "profile": AWS_PROFILE,
                    "error": f"{exc.__class__.__name__}: {exc}"[:300]}

    async def check(self, force: bool = False) -> dict[str, Any]:
        if force or time.time() - self.checked_at > AWS_CHECK_CACHE_S:
            self.status = await asyncio.to_thread(self._check_sync)
            self.checked_at = time.time()
        return self.status

    async def start_login(self) -> dict[str, Any]:
        if self.login and not self.login["done"]:
            return self.login
        self.login = {"url": None, "code": None, "started_at": _iso(_now()),
                      "done": False, "ok": False, "error": None}
        self._proc = await asyncio.create_subprocess_exec(
            "aws", "sso", "login", "--profile", AWS_PROFILE, "--use-device-code", "--no-browser",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
        )
        asyncio.create_task(self._pump())
        # Give the CLI a moment to print the URL + code so the first response
        # can already carry them.
        for _ in range(40):
            if self.login["code"] or self.login["done"]:
                break
            await asyncio.sleep(0.25)
        return self.login

    async def _pump(self) -> None:
        assert self._proc and self._proc.stdout and self.login is not None
        out: list[str] = []
        while True:
            raw = await self._proc.stdout.readline()
            if not raw:
                break
            line = raw.decode(errors="replace").strip()
            out.append(line)
            url = re.search(r"https://\S+", line)
            if url:
                # The CLI prints the bare device URL first, then an autofill
                # variant carrying user_code= — prefer that one (one click).
                code_in_url = re.search(r"user_code=([A-Z0-9-]+)", url.group(0))
                if self.login["url"] is None or code_in_url:
                    self.login["url"] = url.group(0)
                if code_in_url:
                    self.login["code"] = code_in_url.group(1)
            code = re.fullmatch(r"[A-Z0-9]{4}-[A-Z0-9]{4}", line)
            if code:
                self.login["code"] = code.group(0)
        rc = await self._proc.wait()
        self.login["done"] = True
        self.login["ok"] = rc == 0
        if rc != 0:
            self.login["error"] = "\n".join(out[-5:])[:500] or f"aws sso login exited {rc}"
        self.checked_at = 0.0  # force a fresh identity check


AWS = _AwsState()

# ── Athena lookups (blocking boto3 → thread) ────────────────────────────────

_freshness: dict[str, Any] = {"at": 0.0, "value": {}}


async def _athena(fn, *args):
    status = await AWS.check()
    if status["state"] != "ok":
        raise HTTPException(401, "AWS session expired — sign in first")
    try:
        return await asyncio.to_thread(fn, *args)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(502, f"Athena query failed: {exc.__class__.__name__}: {exc}"[:500])


async def freshness() -> dict[str, Any]:
    if time.time() - _freshness["at"] > FRESHNESS_CACHE_S:
        try:
            _freshness["value"] = await _athena(_main._fetch_athena_export_freshness)
            _freshness["at"] = time.time()
        except HTTPException:
            pass
    return _freshness["value"]


# ── Jobs ─────────────────────────────────────────────────────────────────────

class JobRequest(BaseModel):
    task_name: str = Field(min_length=1, max_length=500)
    app_id: int | None = None
    app_name: str | None = None
    env_name: str | None = None


_sem = asyncio.Semaphore(MAX_CONCURRENT)
_running: dict[str, asyncio.subprocess.Process] = {}


def _job_dir(job_id: str) -> Path:
    if not _JOB_ID_RE.match(job_id):
        raise HTTPException(404, "no such job")
    d = JOBS_DIR / job_id
    if not d.is_dir():
        raise HTTPException(404, "no such job")
    return d


def _read_meta(d: Path) -> dict[str, Any]:
    try:
        return json.loads((d / "job.json").read_text())
    except (OSError, ValueError):
        return {"id": d.name, "status": "corrupt"}


def _write_meta(d: Path, meta: dict[str, Any]) -> None:
    tmp = d / "job.json.tmp"
    tmp.write_text(json.dumps(meta, indent=1, default=str))
    tmp.replace(d / "job.json")


def _progress(d: Path) -> list[str]:
    try:
        lines = (d / "log.txt").read_text(errors="replace").splitlines()
    except OSError:
        return []
    return [ln.strip() for ln in lines if _PROGRESS_RE.match(ln)][-60:]


def _log_tail(d: Path, n: int = 40) -> str:
    try:
        return "\n".join((d / "log.txt").read_text(errors="replace").splitlines()[-n:])
    except OSError:
        return ""


def _expires_at(meta: dict[str, Any]) -> datetime | None:
    end = _parse(meta.get("finished_at")) or (_parse(meta.get("created_at")) or _now()) + JOB_TIMEOUT
    return end + TTL


def _summary(d: Path) -> dict[str, Any] | None:
    try:
        r = json.loads((d / "result.json").read_text())
    except (OSError, ValueError):
        return None
    plan = r.get("plan") or {}
    return {
        "task_id": r.get("task_id"),
        "gate_blocked": (r.get("trace") or {}).get("gate_blocked", False),
        "recommendations": len(plan.get("recommendations") or []),
        "blocked": len(plan.get("blocked_recommendations") or []),
        "unactioned": len(plan.get("unactioned_insights") or []),
        "summary": plan.get("summary"),
    }


def _public(d: Path, meta: dict[str, Any], *, detail: bool) -> dict[str, Any]:
    out = {
        **{k: meta.get(k) for k in ("id", "status", "task_name", "app_id", "app_name", "env_name",
                                    "created_at", "started_at", "finished_at", "exit_code", "error")},
        "expires_at": _iso(_expires_at(meta)),
    }
    if meta.get("status") == "done":
        out["result"] = _summary(d)
    if detail:
        out["progress"] = _progress(d)
        if meta.get("status") == "failed":
            out["log_tail"] = _log_tail(d)
    return out


async def _run(job_id: str) -> None:
    d = JOBS_DIR / job_id
    meta = _read_meta(d)
    async with _sem:
        meta.update(status="running", started_at=_iso(_now()))
        _write_meta(d, meta)
        cmd = [sys.executable, str(ROOT / "service" / "run_job.py"),
               "--task-name", meta["task_name"], "--workdir", str(d)]
        if meta.get("app_id") is not None:
            cmd += ["--app-id", str(meta["app_id"])]
        env = {**os.environ, "AWS_SSO_AUTO_LOGIN": "0", "PYTHONUNBUFFERED": "1"}
        with (d / "log.txt").open("ab") as log:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(ROOT), env=env, stdout=asyncio.subprocess.PIPE, stderr=log,
            )
            _running[job_id] = proc
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), JOB_TIMEOUT.total_seconds())
                rc = proc.returncode
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                stdout, rc = b"", -9
            finally:
                _running.pop(job_id, None)
        meta.update(finished_at=_iso(_now()), exit_code=rc)
        if rc == 0 and (d / "result.json").exists():
            meta["status"] = "done"
        else:
            meta["status"] = "failed"
            meta["error"] = _failure_message(rc, stdout)
        _write_meta(d, meta)


def _failure_message(rc: int, stdout: bytes) -> str:
    text = stdout.decode(errors="replace")
    # run_job.py prints a JSON object on stdout for the two expected failures.
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            try:
                payload = json.loads(line)
                if payload.get("error") == "ambiguous":
                    return "ambiguous task name — choose an app first"
                if payload.get("error") == "not_found":
                    return payload.get("message", "task not found")
            except ValueError:
                pass
    if rc == -9:
        return f"killed after {JOB_TIMEOUT.total_seconds()/60:.0f} minutes"
    return f"run_job.py exited {rc} — see log"


async def _sweep_forever() -> None:
    while True:
        try:
            now = _now()
            for d in JOBS_DIR.iterdir():
                if not d.is_dir():
                    continue
                meta = _read_meta(d)
                exp = _expires_at(meta)
                if meta.get("status") == "running" and d.name in _running:
                    continue
                if exp and exp <= now:
                    shutil.rmtree(d, ignore_errors=True)
        except Exception as exc:  # the sweeper must never die
            print(f"sweeper: {exc!r}", file=sys.stderr)
        await asyncio.sleep(SWEEP_EVERY_S)


# ── App ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="Recommendation Agent", docs_url="/api/docs", redoc_url=None)


@app.on_event("startup")
async def _startup() -> None:
    JOBS_DIR.mkdir(parents=True, exist_ok=True)
    # Jobs left "running" by a previous process are dead: mark them failed.
    for d in JOBS_DIR.iterdir():
        if d.is_dir():
            meta = _read_meta(d)
            if meta.get("status") in ("queued", "running"):
                meta.update(status="failed", finished_at=_iso(_now()),
                            error="the service restarted while this job was running")
                _write_meta(d, meta)
    asyncio.create_task(_sweep_forever())


@app.get("/", response_class=HTMLResponse)
@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def index(job_id: str | None = None) -> HTMLResponse:
    return HTMLResponse((STATIC / "index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness/readiness for the ALB and kubelet: no AWS or Athena call, so a
    stale SSO session never makes the pod look unhealthy."""
    return {"ok": True, "running": len(_running), "jobs_dir": str(JOBS_DIR)}


def _build_info() -> dict[str, str | None]:
    """harness/backend commit SHAs baked in by deploy/build.sh's BUILD_INFO
    (absent outside a built image, e.g. running locally via preview_start) —
    surfaced so the "How this works" panel can link to the exact definity-app
    commit this deployment's knowledge base and agent code came from, not
    just a branch name that keeps moving."""
    path = ROOT / "BUILD_INFO"
    if not path.exists():
        return {"harness_commit": None, "backend_commit": None}
    info: dict[str, str] = {}
    for line in path.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            info[k.strip()] = v.strip()
    return {"harness_commit": info.get("harness"), "backend_commit": info.get("backend")}


@app.get("/api/status")
async def status() -> dict[str, Any]:
    aws = await AWS.check()
    return {
        "aws": aws,
        "login": AWS.login,
        "athena_freshness": await freshness() if aws["state"] == "ok" else {},
        "ttl_hours": TTL.total_seconds() / 3600,
        "max_concurrent": MAX_CONCURRENT,
        "running": len(_running),
        "athena_db": _main._ATHENA_DB,
        "llm_model": os.environ.get("LLM_MODEL", "eu.anthropic.claude-sonnet-4-6"),
        "build": _build_info(),
    }


@app.post("/api/aws/login")
async def aws_login() -> dict[str, Any]:
    if (await AWS.check(force=True))["state"] == "ok":
        return {"already": True, **AWS.status}
    return await AWS.start_login()


@app.get("/api/tasks/search")
async def search(q: str) -> dict[str, Any]:
    q = q.strip()
    if len(q) < 2:
        return {"names": []}
    return {"names": await _athena(_main._search_athena_task_names, q, 12)}


@app.get("/api/tasks/candidates")
async def candidates(task_name: str) -> dict[str, Any]:
    task_name = task_name.strip()
    rows = await _athena(_main._fetch_athena_task_candidates, task_name)
    out: dict[str, Any] = {"task_name": task_name, "candidates": rows}
    if not rows:
        # Common miss: the query was actually an app/cluster name (e.g. the
        # Databricks job's display name), not a Spark task_name — the two
        # are different columns and neither contains the other. Point at the
        # real task_name(s) that run under any matching app so "no runs
        # found" isn't a dead end.
        out["app_matches"] = await _athena(_main._search_athena_apps_by_name, task_name, 10)
    return out


@app.get("/api/jobs")
async def list_jobs() -> dict[str, Any]:
    jobs = []
    for d in JOBS_DIR.iterdir():
        if d.is_dir():
            meta = _read_meta(d)
            jobs.append(_public(d, meta, detail=False))
    jobs.sort(key=lambda j: j.get("created_at") or "", reverse=True)
    return {"jobs": jobs[:50]}


@app.post("/api/jobs", status_code=202)
async def create_job(req: JobRequest) -> dict[str, Any]:
    if (await AWS.check())["state"] != "ok":
        raise HTTPException(401, "AWS session expired — sign in first")
    job_id = secrets.token_hex(5)
    d = JOBS_DIR / job_id
    d.mkdir(parents=True)
    meta = {"id": job_id, "status": "queued", "task_name": req.task_name.strip(),
            "app_id": req.app_id, "app_name": req.app_name, "env_name": req.env_name,
            "created_at": _iso(_now()), "started_at": None, "finished_at": None,
            "exit_code": None, "error": None}
    _write_meta(d, meta)
    asyncio.create_task(_run(job_id))
    return _public(d, meta, detail=True)


@app.get("/api/jobs/{job_id}")
async def get_job(job_id: str) -> dict[str, Any]:
    d = _job_dir(job_id)
    return _public(d, _read_meta(d), detail=True)


@app.get("/api/jobs/{job_id}/result.json")
async def get_result(job_id: str) -> FileResponse:
    d = _job_dir(job_id)
    f = d / "result.json"
    if not f.exists():
        raise HTTPException(404, "no result yet")
    meta = _read_meta(d)
    name = f"recommendation-agent_{meta.get('task_name','task')}_{meta.get('app_id') or ''}_{job_id}.json"
    return FileResponse(f, media_type="application/json", filename=re.sub(r"[^\w.-]+", "_", name))


@app.get("/api/jobs/{job_id}/tldr.md")
async def get_tldr(job_id: str) -> PlainTextResponse:
    f = _job_dir(job_id) / "tldr.md"
    if not f.exists():
        raise HTTPException(404, "no TL;DR yet")
    return PlainTextResponse(f.read_text(encoding="utf-8"), media_type="text/markdown")


@app.delete("/api/jobs/{job_id}")
async def delete_job(job_id: str) -> JSONResponse:
    d = _job_dir(job_id)
    proc = _running.get(job_id)
    if proc is not None:
        proc.kill()
    shutil.rmtree(d, ignore_errors=True)
    return JSONResponse({"deleted": job_id})
