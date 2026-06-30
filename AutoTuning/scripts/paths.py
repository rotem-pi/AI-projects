"""Resolve cost-optimizer root and common data paths from any working directory."""

from __future__ import annotations

import os
from pathlib import Path


def cost_optimizer_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "case_studies").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find cost-optimizer root (expected data/case_studies/)"
    )


def definity_backend_root() -> Path:
    """Locate definity-app/backend for framework imports.

    Resolution order:
    1. ``DEFINITY_BACKEND_ROOT`` env var (relative paths are from cost-optimizer root)
    2. Walk parents for a sibling ``definity-app/backend`` with an ``app/`` package
    """
    root = cost_optimizer_root()
    env = os.environ.get("DEFINITY_BACKEND_ROOT")
    if env:
        path = Path(env)
        if not path.is_absolute():
            path = (root / path).resolve()
        else:
            path = path.resolve()
        if not (path / "app").is_dir():
            raise FileNotFoundError(
                f"DEFINITY_BACKEND_ROOT does not contain app/: {path}"
            )
        return path

    for base in [root, *root.parents]:
        for candidate in (
            base.parent / "definity-app" / "backend",
            base / "definity-app" / "backend",
        ):
            if (candidate / "app").is_dir():
                return candidate.resolve()

    raise FileNotFoundError(
        "Could not find definity-app/backend. Set DEFINITY_BACKEND_ROOT "
        "(e.g. export DEFINITY_BACKEND_ROOT=/path/to/definity-app/backend)"
    )


def data_dir() -> Path:
    return cost_optimizer_root() / "data"


def case_studies_dir() -> Path:
    return data_dir() / "case_studies"


def scripts_dir() -> Path:
    return cost_optimizer_root() / "scripts"
