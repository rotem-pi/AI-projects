"""
Add recommendation columns to insights CSV by mirroring backend extract_recommendations().

Reads `data/insights_with_recommendations_data.csv` and writes
`data/insights_with_recommendations_output.csv`
with one row per recommendation (matching backend recommendations[] format).

Each insight produces one output row. Single recommendations use scalar values; when an
insight has multiple recommendations (e.g. min + max executors), values are stored as
JSON arrays in the same order as backend recommendations[].

Usage:
    python add_recommendations.py
    python add_recommendations.py input.csv output.csv
"""

import ast
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional


# Spark/cluster config keys emitted by recommendations.py, mapped to task-run CSV columns.
CONFIG_KEY_TO_RUN_COLUMN: dict[str, str] = {
    "spark.sql.shuffle.partitions": "spark_shuffle_partitions",
    "spark.executor.cores": "spark_executor_cores",
    "clusterMaxWorkers": "cluster_max_workers",
    "clusterWorkers": "cluster_workers",
    "clusterMinWorkers": "cluster_min_workers",
    "spark.driver.cores": "spark_driver_cores",
    "spark.driver.memory": "spark_driver_memory_gb",
    "spark.executor.memory": "spark_executor_memory_gb",
    "spark.executor.memoryOverhead": "spark_executor_memory_overhead_gb",
    "spark.dynamicAllocation.maxExecutors": "spark_dynamic_alloc_max_executors",
    "spark.dynamicAllocation.minExecutors": "spark_dynamic_alloc_min_executors",
    "spark.executor.instances": "spark_executor_instances",
    "spark.task.cpus": "spark_task_cpus",
}


# ---------------------------------------------------------------------------
# Payload parsing
# ---------------------------------------------------------------------------

def _parse_payload(raw: str) -> Optional[dict]:
    """Parse insights_payload (JSON or Python dict literal) into a dict."""
    if not raw or not raw.strip():
        return None
    raw = raw.strip()
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# Recommendation extraction (mirrors backend recommendations.py)
# ---------------------------------------------------------------------------

def _extract_over_provisioned_executors(payload: dict) -> list[dict]:
    recommendations = []

    if payload.get("min_executors_suggestion"):
        recommendations.append({
            "action": "set_spark_config",
            "config_key": "spark.dynamicAllocation.minExecutors",
            "suggested_value": payload["min_executors_suggestion"],
            "current_value": payload.get("dynamic_min_executors_conf"),
        })

    if payload.get("max_executors_suggestion"):
        config_key = (
            "spark.dynamicAllocation.maxExecutors"
            if payload.get("is_dynamic")
            else "spark.executor.instances"
        )
        current_value = (
            payload.get("dynamic_max_executors_conf")
            if payload.get("is_dynamic")
            else payload.get("static_executors_conf")
        )
        recommendations.append({
            "action": "set_spark_config",
            "config_key": config_key,
            "suggested_value": payload["max_executors_suggestion"],
            "current_value": current_value,
        })

    return recommendations


def _extract_over_provisioned_cluster_machines(payload: dict) -> list[dict]:
    recommendations = []

    if payload.get("min_executors_suggestion") and payload.get("is_cluster_auto_scale"):
        recommendations.append({
            "action": "set_cluster_config",
            "config_key": "clusterMinWorkers",
            "suggested_value": payload["min_executors_suggestion"],
            "current_value": payload.get("cluster_min_workers"),
        })

    if payload.get("max_executors_suggestion"):
        config_key = (
            "clusterMaxWorkers" if payload.get("is_cluster_auto_scale") else "clusterWorkers"
        )
        current_value = (
            payload.get("cluster_max_workers")
            if payload.get("is_cluster_auto_scale")
            else payload.get("cluster_workers")
        )
        recommendations.append({
            "action": "set_cluster_config",
            "config_key": config_key,
            "suggested_value": payload["max_executors_suggestion"],
            "current_value": current_value,
        })

    return recommendations


def _extract_over_provisioned_executor_heap_memory(payload: dict) -> list[dict]:
    if not payload.get("suggested_value"):
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.executor.memory",
        "suggested_value": math.ceil(payload["suggested_value"]),
        "current_value": round(payload["current_allocated"]) if payload.get("current_allocated") else None,
    }]


def _extract_over_provisioned_executor_off_heap_memory(payload: dict) -> list[dict]:
    if not payload.get("suggested_value"):
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.executor.memoryOverhead",
        "suggested_value": math.ceil(payload["suggested_value"]),
        "current_value": round(payload["current_allocated"]) if payload.get("current_allocated") else None,
    }]


def _extract_over_provisioned_driver_heap_memory(payload: dict) -> list[dict]:
    if not payload.get("suggested_value"):
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.driver.memory",
        "suggested_value": math.ceil(payload["suggested_value"]),
        "current_value": round(payload["current_allocated"]) if payload.get("current_allocated") else None,
    }]


def _round_up_to_machine_size(cores: int) -> int:
    for size in (2, 4, 8, 16, 32, 64):
        if cores <= size:
            return size
    return cores


def _extract_over_provisioned_driver_cores(payload: dict) -> list[dict]:
    suggested = payload.get("suggested_driver_cores")
    current = payload.get("current_driver_cores")
    if not suggested or not current or suggested >= current:
        return []

    if payload.get("has_dedicated_cluster"):
        driver_type = payload.get("driver_type")
        suggested_driver_type = payload.get("suggested_driver_type")

        if suggested_driver_type and driver_type:
            return [{
                "action": "change_instance_type",
                "suggested_value": suggested_driver_type,
                "current_value": f"{driver_type} ({current}-vCPU)",
            }]

        suggested_machine_size = _round_up_to_machine_size(suggested)
        if suggested_machine_size >= current:
            return []

        if driver_type:
            return [{
                "action": "change_instance_type",
                "suggested_value": f"{suggested_machine_size}-vCPU driver machine",
                "current_value": f"{driver_type} ({current}-vCPU)",
            }]

        return [{
            "action": "change_instance_type",
            "suggested_value": f"{suggested_machine_size}-vCPU driver machine",
            "current_value": f"{current}-vCPU driver machine",
        }]

    return [{
        "action": "set_spark_config",
        "config_key": "spark.driver.cores",
        "suggested_value": suggested,
        "current_value": current,
    }]


def _extract_over_provisioned_machine_type(payload: dict) -> list[dict]:
    task_profile = payload.get("task_profile") or {}
    worker_types = task_profile.get("worker_types") or []
    cloud_provider = task_profile.get("cloud_provider")

    if cloud_provider != "AWS" or len(worker_types) != 1:
        return []

    worker_type = worker_types[0]
    if worker_type.startswith("c"):
        return []

    if worker_type.startswith("r"):
        suggested_type = worker_type.replace("r", "m", 1)
    elif worker_type.startswith("m"):
        suggested_type = worker_type.replace("m", "c", 1)
    else:
        return []

    return [{
        "action": "change_instance_type",
        "suggested_value": suggested_type,
        "current_value": worker_type,
        "cloud_provider": "AWS",
    }]


def _extract_executors_long_spill_time(payload: dict) -> list[dict]:
    recommendations = []
    task_profile = payload.get("task_profile") or {}

    shuffle_partitions = task_profile.get("shuffle_partitions")
    if shuffle_partitions and shuffle_partitions != "auto":
        try:
            partitions_num = int(shuffle_partitions)
            if partitions_num < 20000:
                recommendations.append({
                    "action": "set_spark_config",
                    "config_key": "spark.sql.shuffle.partitions",
                    "suggested_value": partitions_num * 2,
                    "current_value": partitions_num,
                })
        except (ValueError, TypeError):
            pass

    worker_types = task_profile.get("worker_types") or []
    cloud_provider = task_profile.get("cloud_provider")

    if cloud_provider == "AWS" and len(worker_types) == 1:
        worker_type = worker_types[0]
        if not worker_type.startswith("r"):
            suggested_type = worker_type.replace("m", "r", 1).replace("c", "r", 1)
            recommendations.append({
                "action": "change_instance_type",
                "suggested_value": suggested_type,
                "current_value": worker_type,
                "cloud_provider": "AWS",
                "reason": "memory_optimized_for_spill",
            })

    return recommendations


def _extract_under_provisioned_executor_heap_memory(payload: dict) -> list[dict]:
    if not payload.get("suggested_value"):
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.executor.memory",
        "suggested_value": math.ceil(payload["suggested_value"]),
        "current_value": round(payload["current_allocated"]) if payload.get("current_allocated") else None,
    }]


def _extract_under_utilized_executors_cpu(payload: dict) -> list[dict]:
    spark_task_cpus = payload.get("spark_task_cpus")
    if not spark_task_cpus or spark_task_cpus <= 1:
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.task.cpus",
        "suggested_value": max(1, round(spark_task_cpus) - 1),
        "current_value": round(spark_task_cpus),
    }]


def _extract_spark_task_retries_cost(payload: dict) -> list[dict]:
    task_profile = payload.get("task_profile") or {}

    if task_profile.get("cluster_availability") != "SPOT_WITH_FALLBACK":
        return []

    return [{
        "action": "set_cluster_config",
        "config_key": "aws_attributes.availability",
        "suggested_value": "ON_DEMAND",
        "current_value": "SPOT_WITH_FALLBACK",
    }]


def _extract_orphaned_machine_vcores(payload: dict) -> list[dict]:
    recommendations = []

    shuffle_partitions = payload.get("shuffle_partitions")
    if shuffle_partitions:
        try:
            partitions_num = int(shuffle_partitions)
            recommendations.append({
                "action": "set_spark_config",
                "config_key": "spark.sql.shuffle.partitions",
                "suggested_value": partitions_num * 2,
                "current_value": partitions_num,
                "reason": "reduce_memory_per_task",
            })
        except (ValueError, TypeError):
            pass

    executor_cores = payload.get("executor__cores")
    if executor_cores is not None:
        recommendations.append({
            "action": "set_spark_config",
            "config_key": "spark.executor.cores",
            "suggested_value": executor_cores + 1,
            "current_value": executor_cores,
            "reason": "utilize_orphaned_vcores",
        })

    return recommendations


def _extract_executors_long_gc_time(payload: dict) -> list[dict]:
    if not payload.get("suggested_value"):
        return []

    return [{
        "action": "set_spark_config",
        "config_key": "spark.executor.memory",
        "suggested_value": math.ceil(payload["suggested_value"]),
        "current_value": round(payload["current_allocated"]) if payload.get("current_allocated") else None,
    }]


def _extract_task_retries_cost(payload: dict) -> list[dict]:
    return []


def _extract_long_idle_time(payload: dict) -> list[dict]:
    return []


def _extract_long_skew_time(payload: dict) -> list[dict]:
    task_profile = payload.get("task_profile") or {}
    shuffle_partitions = task_profile.get("shuffle_partitions")

    if shuffle_partitions and shuffle_partitions != "auto":
        try:
            partitions_num = int(shuffle_partitions)
            if partitions_num < 20000:
                return [{
                    "action": "set_spark_config",
                    "config_key": "spark.sql.shuffle.partitions",
                    "suggested_value": partitions_num * 2,
                    "current_value": partitions_num,
                    "reason": "reduce_skew",
                }]
        except (ValueError, TypeError):
            pass

    return []


def _extract_small_files(payload: dict) -> list[dict]:
    return []


def _extract_stale_task_runs(payload: dict) -> list[dict]:
    return []


def _extract_excessive_listing_operation(payload: dict) -> list[dict]:
    return []


def _extract_orphaned_node_resources(payload: dict) -> list[dict]:
    return []


def _extract_captured_cost_savings(payload: dict) -> list[dict]:
    return []


RECOMMENDATION_EXTRACTORS = {
    "over_provisioned_executors": _extract_over_provisioned_executors,
    "over_provisioned_cluster_machines": _extract_over_provisioned_cluster_machines,
    "over_provisioned_executor_heap_memory": _extract_over_provisioned_executor_heap_memory,
    "over_provisioned_executor_off_heap_memory": _extract_over_provisioned_executor_off_heap_memory,
    "over_provisioned_driver_heap_memory": _extract_over_provisioned_driver_heap_memory,
    "over_provisioned_driver_cores": _extract_over_provisioned_driver_cores,
    "over_provisioned_machine_type": _extract_over_provisioned_machine_type,
    "executors_long_spill_time": _extract_executors_long_spill_time,
    "under_provisioned_executor_heap_memory": _extract_under_provisioned_executor_heap_memory,
    "under_utilized_executors_cpu": _extract_under_utilized_executors_cpu,
    "spark_task_retries_cost": _extract_spark_task_retries_cost,
    "orphaned_machine_vcores": _extract_orphaned_machine_vcores,
    "executors_long_gc_time": _extract_executors_long_gc_time,
    "task_retries_cost": _extract_task_retries_cost,
    "long_idle_time": _extract_long_idle_time,
    "long_skew_time": _extract_long_skew_time,
    "small_files": _extract_small_files,
    "stale_task_runs": _extract_stale_task_runs,
    "excessive_listing_operation": _extract_excessive_listing_operation,
    "orphaned_node_resources": _extract_orphaned_node_resources,
    "captured_cost_savings": _extract_captured_cost_savings,
}


def extract_recommendations(insight_type: str, payload: dict) -> list[dict]:
    """Extract recommendations from insight payload (mirrors backend recommendations.py)."""
    extractor = RECOMMENDATION_EXTRACTORS.get(insight_type)
    if not extractor:
        return []

    recommendations = extractor(payload)
    if not recommendations:
        return []

    if isinstance(recommendations, dict):
        return [recommendations]
    return recommendations


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def _serialize(v: Any) -> str:
    if v is None:
        return ""
    return str(v)


def _serialize_recommendation_field(values: list[Any]) -> str:
    """Scalar for one recommendation; JSON array when multiple."""
    if not values:
        return ""
    if len(values) == 1:
        return _serialize(values[0])
    return json.dumps(values)


def _apply_run_config_values(row: dict) -> dict:
    """Copy task-run config values into columns named by recommendations config_key."""
    out = dict(row)
    for config_key, source_col in CONFIG_KEY_TO_RUN_COLUMN.items():
        val = row.get(source_col, "")
        out[config_key] = "" if val is None or str(val).strip() == "" else val
    return out


def _apply_recommendations(row: dict, recommendations: list[dict]) -> dict:
    out = _apply_run_config_values(row)
    if not recommendations:
        out["action"] = ""
        out["config_key"] = ""
        out["suggested_value"] = ""
        out["current_value"] = ""
        return out

    out["action"] = _serialize_recommendation_field([r.get("action") for r in recommendations])
    out["config_key"] = _serialize_recommendation_field([r.get("config_key") for r in recommendations])
    out["suggested_value"] = _serialize_recommendation_field([r.get("suggested_value") for r in recommendations])
    out["current_value"] = _serialize_recommendation_field([r.get("current_value") for r in recommendations])
    return out


def _validate_run_config_sources(fieldnames: list[str]) -> list[str]:
    """Return source columns required for config keys but missing from the input CSV."""
    return [
        source_col
        for source_col in CONFIG_KEY_TO_RUN_COLUMN.values()
        if source_col not in fieldnames
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def process(input_path: str, output_path: str) -> None:
    input_path = Path(input_path)
    output_path = Path(output_path)

    stats = {
        "input_rows": 0,
        "output_rows": 0,
        "payload_parsed": 0,
        "payload_missing": 0,
        "with_recommendations": 0,
        "no_recommendations": 0,
        "unknown_type": 0,
        "multi_recommendation_insights": 0,
    }

    coverage_counts = {config_key: 0 for config_key in CONFIG_KEY_TO_RUN_COLUMN}

    with input_path.open(newline="") as fin, output_path.open("w", newline="") as fout:
        reader = csv.DictReader(fin)
        missing_sources = _validate_run_config_sources(list(reader.fieldnames or []))
        if missing_sources:
            raise ValueError(
                "Input CSV is missing task-run columns required for recommendations config keys: "
                + ", ".join(missing_sources)
            )

        extra_columns = (
            list(CONFIG_KEY_TO_RUN_COLUMN.keys())
            + ["action", "config_key", "suggested_value", "current_value"]
        )
        fieldnames = list(dict.fromkeys(list(reader.fieldnames) + extra_columns))
        writer = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for row in reader:
            stats["input_rows"] += 1
            insight_type = row.get("insight_type", "")

            raw_payload = row.get("insights_payload", "")
            p = _parse_payload(raw_payload)
            if p:
                stats["payload_parsed"] += 1
            elif raw_payload.strip():
                stats["payload_missing"] += 1

            if insight_type and insight_type not in RECOMMENDATION_EXTRACTORS:
                stats["unknown_type"] += 1
                recommendations: list[dict] = []
            elif insight_type and p is not None:
                recommendations = extract_recommendations(insight_type, p)
            else:
                recommendations = []

            if recommendations:
                stats["with_recommendations"] += 1
                if len(recommendations) > 1:
                    stats["multi_recommendation_insights"] += 1
            else:
                stats["no_recommendations"] += 1

            out_row = _apply_recommendations(row, recommendations)
            writer.writerow(out_row)
            for config_key in CONFIG_KEY_TO_RUN_COLUMN:
                if str(out_row.get(config_key, "")).strip() != "":
                    coverage_counts[config_key] += 1
            stats["output_rows"] += 1

    if stats["output_rows"]:
        print("\nTask-run config_key coverage (non-empty values per row):")
        for config_key, source_col in CONFIG_KEY_TO_RUN_COLUMN.items():
            filled = coverage_counts[config_key]
            total = stats["output_rows"]
            print(
                f"  {config_key:<45} {filled:>8,}/{total:,} "
                f"({100 * filled / total:5.1f}%)  <- {source_col}"
            )

    print(f"Output written to: {output_path}")
    print(f"Input rows:                    {stats['input_rows']:>8,}")
    print(f"Output rows:                   {stats['output_rows']:>8,}")
    print(f"  payload parsed:              {stats['payload_parsed']:>8,}")
    print(f"  payload missing/bad:         {stats['payload_missing']:>8,}")
    print(f"  insights with recommendations: {stats['with_recommendations']:>6,}")
    print(f"  multi-rec insights:          {stats['multi_recommendation_insights']:>8,}")
    print(f"  no recommendation:           {stats['no_recommendations']:>8,}")
    print(f"  unknown insight type:        {stats['unknown_type']:>8,}")


if __name__ == "__main__":
    from paths import data_dir

    data = data_dir()
    default_in = data / "insights_with_recommendations_data.csv"
    default_out = data / "insights_with_recommendations_output.csv"

    if len(sys.argv) == 3:
        input_csv, output_csv = sys.argv[1], sys.argv[2]
    elif len(sys.argv) == 2:
        input_csv, output_csv = sys.argv[1], str(default_out)
    else:
        input_csv, output_csv = str(default_in), str(default_out)

    process(input_csv, output_csv)
