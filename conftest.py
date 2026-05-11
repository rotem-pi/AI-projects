"""
Flat ``incidents-test-analysis/`` layout: this directory is on ``sys.path`` first
(for ``legacy_analytical_model_pre_8f5e1c0``), then ``definity-app/backend`` (for ``app``).
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pytest

try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    pytest.skip(
        "Impact tests require scikit-learn. Use definity-app/backend/.venv (see README).",
        allow_module_level=True,
    )


def _definity_backend_root() -> Path:
    raw = os.environ.get("DEFINITY_BACKEND_ROOT")
    if raw:
        p = Path(raw).resolve()
        if not (p / "app").is_dir():
            raise RuntimeError(f"DEFINITY_BACKEND_ROOT has no app/: {p}")
        return p
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "definity-app" / "backend"
        if (cand / "app").is_dir():
            return cand.resolve()
    raise RuntimeError(
        "Could not find definity-app/backend. Set DEFINITY_BACKEND_ROOT or place "
        "definity-app next to a parent of incidents-test-analysis (e.g. GitCode/)."
    )


_here = Path(__file__).resolve().parent
_definity = _definity_backend_root()
for p in (_definity, _here):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

os.environ.setdefault(
    "DATABASE_URL", "postgresql://pytest_placeholder:pytest@127.0.0.1:5432/pytest_no_db"
)
os.environ.setdefault("IS_SAAS", "true")

logging.getLogger("psycopg.pool").setLevel(logging.ERROR)
