"""Resolve cost-optimizer root and common data paths from any working directory."""

from __future__ import annotations

from pathlib import Path


def cost_optimizer_root() -> Path:
    for candidate in [Path.cwd(), *Path.cwd().parents]:
        if (candidate / "data" / "case_studies").is_dir():
            return candidate
    raise FileNotFoundError(
        "Could not find cost-optimizer root (expected data/case_studies/)"
    )


def data_dir() -> Path:
    return cost_optimizer_root() / "data"


def case_studies_dir() -> Path:
    return data_dir() / "case_studies"


def scripts_dir() -> Path:
    return cost_optimizer_root() / "scripts"
