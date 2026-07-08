"""Cost-per-run calculation.

Ported from the SQL functions that back definity-app's cost dashboard —
CALCULATE_USD_COST / CALCULATE_UNUSED_VCORE_TIME
(backend/alembic_sql/sql_utils/functions_create.sql) combined the way
backend/app/dal/sql/enhanced_tasks.sql does for a single task run:

    cost = CALCULATE_USD_COST('VCore', task__vcore_time__allocated)
         + CALCULATE_USD_COST('GB',    task__memory_time__allocated)
         + CALCULATE_USD_COST('VCore', unused_vcore_time)

This logic only exists as a Postgres function today (no importable Python
equivalent), so it's reproduced here rather than mocked from scratch.
"""

from __future__ import annotations

from pydantic import BaseModel

from app.brain.insights.agent.models import DbRow

# Match tenant_settings defaults (backend/app/models/system_settings_models.py TenantSettings).
DEFAULT_VCORE_PRICE = 0.05
DEFAULT_MEMORY_PRICE = 0.005
DEFAULT_MACHINE_MEMORY_TO_VCORE_RATIO = 0.0


class CostProfile(BaseModel):
    cost_per_run_usd: float | None = None
    cluster_workers: int | None = None
    workers_availability: str | None = None


def _calculate_usd_cost(
    units: str, value: float | None, *, vcore_price: float, memory_price: float
) -> float:
    if value is None:
        return 0.0
    if units == "VCore":
        return value * vcore_price / 3600
    if units == "GB":
        return value * memory_price / 3600
    return 0.0


def _calculate_unused_vcore_time(
    executor_memory_allocated: float | None,
    executor_cores: float | None,
    duration: float | None,
    machine_memory_to_vcore_ratio: float,
) -> float:
    if machine_memory_to_vcore_ratio <= 0:
        return 0.0
    if executor_memory_allocated is None or executor_cores is None or duration is None:
        return 0.0
    return max((executor_memory_allocated / machine_memory_to_vcore_ratio - executor_cores) * duration, 0.0)


def compute_cost_profile(
    row: DbRow,
    *,
    vcore_price: float = DEFAULT_VCORE_PRICE,
    memory_price: float = DEFAULT_MEMORY_PRICE,
    machine_memory_to_vcore_ratio: float = DEFAULT_MACHINE_MEMORY_TO_VCORE_RATIO,
) -> CostProfile:
    unused_vcore_time = _calculate_unused_vcore_time(
        row.get("executor__memory__allocated"),
        row.get("executor__cores"),
        row.get("task__duration"),
        machine_memory_to_vcore_ratio,
    )
    cost_per_run_usd = (
        _calculate_usd_cost(
            "VCore", row.get("task__vcore_time__allocated"), vcore_price=vcore_price, memory_price=memory_price
        )
        + _calculate_usd_cost(
            "GB", row.get("task__memory_time__allocated"), vcore_price=vcore_price, memory_price=memory_price
        )
        + _calculate_usd_cost("VCore", unused_vcore_time, vcore_price=vcore_price, memory_price=memory_price)
    )

    return CostProfile(
        cost_per_run_usd=round(cost_per_run_usd, 4),
        cluster_workers=row.get("cluster_workers"),
        workers_availability=row.get("workers_availability"),
    )
