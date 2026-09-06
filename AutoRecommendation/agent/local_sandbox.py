"""Local-filesystem stand-in for app.dal.sandbox_dal's S3-backed run sandbox.

Production's fetch_context dumps one CSV per raw DB table to S3 and puts a
RunSandbox(task_id, bucket, prefix, tables) pointer in graph state; tools.py
then calls sandbox_dal.materialize_to_folder(task_id, tables) on demand,
which downloads those CSVs into a temp folder and builds task_store.h5
there (kb/analysis/store.py) for the deterministic review layer to read.

This harness must not touch S3 (see agent/nodes/fetch_context.py's
docstring), so it needs a same-shaped local equivalent: each entry point
builds a {table_name: [row-dict, ...]} mapping from whatever raw-ish data its
source can offer (single-PIT only — no multi-run history, no cross-run
config-change-regime detection; that is an accepted gap of this harness, not
production), hands it to set_sandbox_tables_override(), and fetch_context
wraps it in a real RunSandbox. patch_sandbox_dal() then makes tools.py's
sandbox_dal.materialize_to_folder(...) calls resolve to a folder built from
these tables instead of hitting real S3.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from app.brain.insights.agent.kb.analysis import store as kb_store
from app.brain.insights.agent.models import RunSandbox
from app.dal import sandbox_dal

# task_id -> {table_name: [row-dict, ...]}, set by set_sandbox_tables_override()
# before the graph runs, mirroring nodes/fetch_context.py's _ROW_OVERRIDE pattern.
_SANDBOX_TABLES: dict[int, dict[str, list[dict]]] = {}


def set_sandbox_tables_override(task_id: int, tables: dict[str, list[dict]]) -> None:
    _SANDBOX_TABLES[task_id] = tables


def clear_sandbox_tables_override(task_id: int) -> None:
    _SANDBOX_TABLES.pop(task_id, None)


def build_run_sandbox(task_id: int, tables: dict[str, list[dict]]) -> RunSandbox:
    """RunSandbox pointer for a locally-held tables dict. bucket/prefix are
    placeholders — materialize_to_folder is patched to never read them."""
    set_sandbox_tables_override(task_id, tables)
    return RunSandbox(
        task_id=task_id,
        bucket="local",
        prefix=f"task-{task_id}",
        tables=list(tables.keys()),
    )


def _materialize_local(task_id: int, tables: tuple[str, ...]) -> str:
    """The harness's local equivalent of sandbox_dal.materialize_to_folder:
    write each overridden table as <table>.csv into a fresh temp folder and
    build task_store.h5 there via the same kb/analysis/store.py used in
    production, so every downstream tool reads the identical on-disk shape."""
    tables_by_name = _SANDBOX_TABLES.get(task_id, {})
    # LOCAL_SANDBOX_DIR: parent for the per-materialization folders, so a
    # caller that owns a job workdir can put them under it and remove them
    # with the job instead of leaking mkdtemp() dirs into the system temp.
    base = os.environ.get("LOCAL_SANDBOX_DIR") or None
    if base:
        Path(base).mkdir(parents=True, exist_ok=True)
    folder = Path(tempfile.mkdtemp(prefix=f"local_sandbox_{task_id}_", dir=base))
    for table in tables:
        rows = tables_by_name.get(table, [])
        _write_csv(folder / f"{table}.csv", rows)
    kb_store.build_from_csv(str(folder))
    return str(folder)


def _write_csv(path: Path, rows: list[dict]) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


_original_materialize_to_folder = sandbox_dal.materialize_to_folder


def _materialize_to_folder_override(task_id: int, tables: tuple[str, ...]) -> str:
    if task_id in _SANDBOX_TABLES:
        return _materialize_local(task_id, tables)
    return _original_materialize_to_folder(task_id, tables)


def patch_sandbox_dal() -> None:
    """Redirect tools.py's sandbox_dal.materialize_to_folder(...) calls to the
    local builder for any task_id this harness has an override for. tools.py
    does `from app.dal import sandbox_dal` then `sandbox_dal.materialize_to_folder(...)`
    (module-attribute access, not a direct name import), so reassigning the
    attribute here affects that call site. Idempotent — safe to call from
    every entry point's import path."""
    sandbox_dal.materialize_to_folder = _materialize_to_folder_override


patch_sandbox_dal()
