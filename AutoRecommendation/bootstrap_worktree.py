"""Ensures a dedicated git worktree of definity-app's auto-recommendations-agent
branch exists and is up to date, then (re)points agent/'s symlinks at it.

Why a dedicated worktree instead of using DEFINITY_APP_REPO directly: that repo
is someone's own live definity-app clone, on whatever branch they're using for
other things, often with uncommitted changes. Checking out or resetting it here
would clobber their work. A worktree at a fixed, harness-owned path is a second,
disposable checkout of the same repository that never touches the original —
see `git worktree add --detach`, which checks out a commit (not a branch name),
so it never conflicts with whatever branch is checked out elsewhere, including
DEFINITY_APP_REPO itself.

Run standalone (`python bootstrap_worktree.py`), or via ./run.sh, which calls
this before every `uv run` so the backend used for dependency resolution and
the backend the agent/ symlinks point at are always the same commit.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
BRANCH = os.environ.get("AUTO_RECS_BRANCH", "auto-recommendations-agent")
WORKTREE_DIR = REPO_ROOT / ".worktrees" / "definity-app-auto-recs"
BACKEND_PATH_FILE = REPO_ROOT / ".worktrees" / ".backend_path"

AGENT_DIR = REPO_ROOT / "agent"
# Kept in sync by hand with agent/'s actual symlinks (see README's
# "Important: symlinked agent code" section) — not auto-discovered, so a
# stale entry here fails loudly (missing source file) rather than silently.
SYMLINKED_FILES = [
    "constants.py", "enums.py", "models.py", "prompts.py", "state.py",
    "tools.py", "knowledge_base.py", "kb",
]
SYMLINKED_NODE_FILES = [
    "__init__.py", "assemble_output.py", "discovery.py", "entry_gate.py",
    "explain.py", "plan.py", "safety.py", "triage.py",
]


def _load_dotenv_var(name: str) -> str | None:
    """Minimal .env reader — avoids depending on python-dotenv so this script
    also works from a bare `python3` before any venv/deps are set up."""
    env_path = REPO_ROOT / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def _definity_app_repo() -> Path:
    raw = (
        os.environ.get("DEFINITY_APP_REPO")
        or _load_dotenv_var("DEFINITY_APP_REPO")
        or "~/GitCode/definity-app"
    )
    return Path(raw).expanduser().resolve()


def _run(cmd: list[str]) -> None:
    print(f"  $ {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, check=True)


def ensure_worktree(no_sync: bool = False) -> Path:
    repo = _definity_app_repo()
    if not (repo / ".git").exists():
        raise SystemExit(
            f"DEFINITY_APP_REPO={repo} doesn't look like a definity-app git checkout "
            "(no .git found). Set DEFINITY_APP_REPO in .env to your local clone — "
            "any branch, it's only used as the source to fetch from, never modified."
        )

    WORKTREE_DIR.parent.mkdir(parents=True, exist_ok=True)

    if not WORKTREE_DIR.exists():
        print(f"Creating dedicated worktree for {BRANCH} at {WORKTREE_DIR} ...", file=sys.stderr)
        _run(["git", "-C", str(repo), "fetch", "origin", BRANCH])
        _run(["git", "-C", str(repo), "worktree", "add", "--detach", str(WORKTREE_DIR), "FETCH_HEAD"])
    elif not no_sync:
        print(f"Syncing worktree to latest origin/{BRANCH} ...", file=sys.stderr)
        _run(["git", "-C", str(WORKTREE_DIR), "fetch", "origin", BRANCH])
        _run(["git", "-C", str(WORKTREE_DIR), "checkout", "--detach", "FETCH_HEAD"])
    else:
        print(f"--no-sync: reusing worktree at {WORKTREE_DIR} as-is", file=sys.stderr)

    backend = WORKTREE_DIR / "backend"
    if not backend.exists():
        raise SystemExit(f"{backend} not found inside the worktree — unexpected repo layout")
    return backend


def _relink(link: Path, target: Path) -> None:
    if not target.exists():
        raise SystemExit(f"Expected symlink source {target} not found — repo layout changed?")
    if link.is_symlink() and Path(os.readlink(link)) == target:
        return
    if link.is_symlink() or link.exists():
        link.unlink()
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def relink_agent_symlinks(backend: Path) -> None:
    agent_src = backend / "app" / "brain" / "insights" / "agent"
    for name in SYMLINKED_FILES:
        _relink(AGENT_DIR / name, agent_src / name)
    for name in SYMLINKED_NODE_FILES:
        _relink(AGENT_DIR / "nodes" / name, agent_src / "nodes" / name)


def resolved_backend_path() -> Path:
    """For other scripts (main.py, evaluation/compare_sql_stream.py, ...) that
    need the resolved backend dir but shouldn't each re-implement the git/env
    logic above. Requires main() to have run at least once (via ./run.sh or a
    manual `python bootstrap_worktree.py`)."""
    if not BACKEND_PATH_FILE.exists():
        raise SystemExit(
            "No resolved backend path yet — run `python bootstrap_worktree.py` "
            "(or use ./run.sh, which does this automatically) first."
        )
    path = Path(BACKEND_PATH_FILE.read_text().strip())
    if not path.exists():
        raise SystemExit(f"Resolved backend path {path} no longer exists — re-run bootstrap_worktree.py")
    return path


def main() -> None:
    no_sync = "--no-sync" in sys.argv
    backend = ensure_worktree(no_sync=no_sync)
    relink_agent_symlinks(backend)
    BACKEND_PATH_FILE.write_text(f"{backend}\n")
    print(f"OK — agent code resolved from {backend} (branch {BRANCH})", file=sys.stderr)


if __name__ == "__main__":
    main()
