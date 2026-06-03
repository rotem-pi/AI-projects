# Auto-Tuning Policy — Brief HLD

**Scope:** `app/brain/insights/tuning/` · **Status:** Implemented (165 unit tests) · **Updated:** 2026-06-01

Condensed companion to [tuning_hld.md](./tuning_hld.md) in this folder. Same process, less prose.

---

## Goal

From active insights for a task, produce a **ranked, safe, gradual, self-driving** config optimization:

| Step | Question | Where |
|------|----------|--------|
| What to change | Expert target from L1 | `recommendations.py` |
| How far this iteration | Step cap (gradual ramp) | `step.py`, `knob_registry.py` |
| Safe to apply now? | 8 blocking rules | `blocking.py` |
| After each run | Advance / pause / rollback / abandon / confirm | `loop.py`, `run_assessor.py` |

Designed for autonomous operation: over-provisioned → confirmed optimized, with anomalies surfaced.

---

## Pipeline (L1 → L2 → loop)

```
L1 Insight SQL (schedule)
  → insights row: type, payload, impact_cost, usd_cost
  → optional estimated_duration_impact_pct (executor sims)

L2a build_tuning_plan(insights, constraints, run_metrics, task_profile)
  → extract rec → next_value → block? → cost/duration est → priority
  → TuningPlan: actions (ranked), blocked_actions

Apply next_value → run job

L2b process_run_outcome(state, outcome) → LoopDecision
  → search → confirming → completed | abandoned | paused

Post-run: assess_duration_regression, get_confirmation_status (3 clean runs)
```

---

## Inputs

| Input | Source | Required |
|-------|--------|----------|
| `active_insights` | insights table | Yes |
| `constraints` | user/org (`TuningConstraints`) | No (defaults) |
| `run_metrics` | latest `task_enrichments` | No (graceful degrade) |
| `task_profile` | enrichments / cluster config (incl. `executor_cores`) | No |

**Constraints (defaults):** duration tolerance 10% (`strict` 5% / `standard` 10% / `flexible` 20%); step cap 20% or abs 5; `apply_mode` suggest/approve/auto.

**Run metrics** (from enrichments; missing → `None`, weaker blocking):

| Signal | Formula (summary) | Used for |
|--------|-------------------|----------|
| `memory_headroom` | 1 − heap_used/allocated | Rule 4a |
| `gc_pressure` | gc_time / run_time | Rule 3 |
| `off_heap_headroom` | 1 − off_heap_used/allocated | Rule 4b |
| `vcore_utilization` | vcore_used / allocated | Rule 5 |
| `idle_ratio`, `skew_ratio` | idle or skew / duration | Rule 6, labels |
| `has_spill` | disk_spilled > **100 MB** | Rule 2 |
| `spill_ratio` | spilled / total_io (if spill > 100MB) | informational |
| `retried_task_waste`, `cpu_efficiency` | ratios | labels / heuristics |

---

## Step 1 — Extract recommendations

`extract_recommendations(insight_type, payload[, task_profile])` → list of:

```python
{ "action": "set_spark_config"|"set_cluster_config"|"change_instance_type",
  "config_key": "...", "current_value", "suggested_value",
  "estimated_duration_impact_pct": ... }  # optional, from L1 SQL
```

| Insight | Recommendation |
|---------|----------------|
| `over_provisioned_executors` | ↓ max/min executors |
| `over_provisioned_cluster_machines` | ↓ cluster max/min workers |
| `over_provisioned_executor_heap_memory` | ↓ `spark.executor.memory` |
| `over_provisioned_executor_off_heap_memory` | ↓ `spark.executor.memoryOverhead` |
| `over_provisioned_driver_heap_memory` | ↓ `spark.driver.memory` |
| `over_provisioned_driver_cores` | ↓ `spark.driver.cores` **or** `change_instance_type` |
| `over_provisioned_machine_type` | `change_instance_type` (AWS r→m, m→c) |
| `executors_long_spill_time` | ↑ shuffle partitions; AWS r-instance; cap 20k |
| `under_utilized_executors_cpu` | ↓ `spark.task.cpus` |
| `spark_task_retries_cost` | SPOT → ON_DEMAND |
| `orphaned_node_resources` | ↓ cluster workers (vCore orphan only) |
| `under_provisioned_executor_heap_memory` | ↑ executor memory |
| `long_idle_time` | ↓ executors/workers (needs `task_profile`) |
| `long_skew_time` | 2× shuffle partitions (needs `task_profile`; AQE → []) |

**No automatable lever (`[]`):** `task_retries_cost`, `executors_long_gc_time` (no suggested_value), `small_files`, `stale_task_runs`, `excessive_listing_operation`.

**Special guards:**
- **Fix B2 (driver cores):** `driver__cores` is often machine vCPU, not `spark.driver.cores`. If `current > 64` → `current_value = None` → single step to target. Rule 7 backstop.
- **INT_MAX guard (`long_idle_time`):** if maxExecutors ≥ 2e9 → `[]`.
- **`long_skew_time`:** reads `task_profile.shuffle_partitions_param`; AQE/`auto` → `[]`.

---

## Step 2 — Step cap (`next_value`)

Per-knob `KnobStepCap`: `max_step = max(abs_cap, current × pct_cap)`; `next = current ± min(|Δ|, max_step)`. Integer knobs clamp to target.

`current is None` → `next = target` (one step).

| Config key | pct | abs | Notes |
|------------|-----|-----|-------|
| maxExecutors, executor.instances, cluster workers | 20% | 5–10 | cost_reduction_only |
| executor/driver memory, memoryOverhead | 15% | 4 GB | |
| driver.cores, executor.cores, task.cpus | 25% | 1–2 | driver: max_sane_current=64 |
| shuffle.partitions | 50% | 500 | faster ramp |
| change_instance_type, availability | — | — | expert_single_step (one hop) |

---

## Step 3 — Blocking (8 rules, first match wins)

| # | Condition | Block when |
|---|-----------|------------|
| **7** | Data quality | `current > max_sane_current` (e.g. driver.cores > 64) |
| **1** | AQE / Databricks | tune `spark.sql.shuffle.partitions` while AQE/auto-optimize/active |
| **2** | Spill + shrink capacity | `has_spill` and ↓ executor count or memory |
| **3** | GC + shrink memory | `gc_pressure > 20%` and ↓ memory |
| **4a** | Tight heap | `memory_headroom < 10%` and ↓ heap memory |
| **4b** | Tight off-heap | `off_heap_headroom < 10%` and ↓ memoryOverhead |
| **5** | CPU saturated | `vcore_util ≥ 85%` and ↓ executor count |
| **6** | Skew + partition load | `skew > 30%`, ↓ executors, partitions/executor > 100 (static partitions only) |
| **8** | vCore budget increase | `config_key` in MAX_EXECUTOR_KEYS, `executor_cores` known, `target × cores > current × cores` |

Thresholds: GC 0.20, headroom 0.10, vcore 0.85, skew 0.30, partitions/executor 100.

During **confirmation** at target, Rules 2–6 (all need `direction==decrease`) do not apply; safety = failure rollback, cost abandon, duration step_back.

---

## Step 4 — Cost & duration (informational)

- **Full savings:** `cost_savings_usd_annual` from insight `usd_cost`.
- **Step savings:** `full × (current − next) / (current − target)` (linear assumption).
- **Duration Δ:** payload `estimated_duration_impact_pct` if present; else `risk_rules.py` heuristics (executor/memory/shuffle downscale modifiers). **Not used for blocking** — only priority adjustment.

---

## Step 5 — Priority

```
base = usd_cost_annual/10_000  OR  savings_pct × (2 if cost-insight else 1)
priority = base / (1 + duration_delta_pct/100)   if duration_delta > 0
         = base                                   otherwise
blocked → 0
```

---

## Output: `TuningPlan`

- **`actions`:** ranked `TuningAction` (config_key, current/target/next, step_policy, insight_type, savings, duration_delta_est, risk.observed_signals, priority_score).
- **`blocked_actions`:** same shape + `block_reason`.

---

## Post-run assessment (`run_assessor.py`)

- **Regression:** z = (duration − mean(baseline)) / std; needs ≥2 baseline samples. Thresholds: strict 1.28, standard 1.645, flexible 2.054.
- **Confirmation:** 3 consecutive runs at target with z below threshold (`CONFIRMATION_RUNS_REQUIRED = 3`).

---

## Autonomous loop (`loop.py`)

**Per-knob phases:** `search` → (at target) `confirming` → `completed` | `abandoned` | `paused`.

**Decision priority:**

| # | Trigger | Action |
|---|---------|--------|
| 1 | Job failed/OOM/timeout | **ROLLBACK** to previous staircase value |
| 2 | `run_cost > pre_tuning_cost × 1.15` for 2 consecutive runs | **ABANDON** knob |
| 2a | Cost drops ≥30 % while config unchanged | Re-anchor baseline → **continue** |
| 3–4 | Blocking signal (paused / new) | **PAUSE** |
| 5 | Duration regression in search | **PAUSE** |
| 6 | Regression in confirming | **STEP_BACK** one level |
| 7a | 3 clean runs + refreshed target differs >10% (≤3 refreshes) | **REFRESH** → rebuild staircase, re-enter search |
| 7b | 3 clean confirmation runs (no meaningful target change) | **COMPLETE** |
| 8 | Confirming, waiting | **STAY** |
| 9 | Else, not at target | **ADVANCE** |

**Rollback:** revert to previous staircase entry (handles fractional/halved steps between rungs).

**Paused:** hold config; re-check blocking each run; resume search or confirming when clear.

**Multi-knob:** `TaskTuningSession` — one state machine per knob; knobs applied in parallel (no cross-knob interaction model).

**Cycle budget:** steps + 3 confirmation runs (e.g. 1 step → 4 runs; 6 steps → 9 runs). Dataset median ~4–5 runs/knob.

**API sketch:** `create_knob_state(...)` → `get_initial_next_value(state)` → each run `process_run_outcome(state, RunOutcome(...), target_refresher=fn)` → `decision.next_action`, `decision.next_value`.

`target_refresher(config_key, current_value) → new_target` is optional; called once per confirmation. If the returned target differs >10% from the previous target (and refresh budget < 3), a `"refresh"` action is returned and the staircase is rebuilt from the current applied value.

---

## End-to-end example

```
L1: over_provisioned_executors, current=100, target=10, $8.4k/yr
→ staircase [80,64,51,41,33,26,21,17,13,10]
→ run @80: success → ADVANCE @64 … → run @10: STAY (1/3, 2/3) → COMPLETE @10
```

Failure mid-ramp → ROLLBACK; cost above pre-tuning baseline → ABANDON; regression at target → STEP_BACK.

---

## Assumptions & limits

| Topic | Note |
|-------|------|
| Linear cost vs resource | Step savings approximate; ranking OK |
| Duration sim | SQL for executor insights only; else heuristics |
| Z-scores | Assumes Gaussian baselines |
| Independent knobs | No interaction modeling in parallel session |
| Cost gate | Requires 2 consecutive cost-exceeding runs (+15 % tolerance) before ABANDON; single transient spike is tolerated |
| External cost shift | If cost drops ≥30 % while config is unchanged, baseline is re-anchored (workload changed externally, not due to Spark config) |
| Regression detection | Needs ≥2 baseline durations |
| High-variance tasks | Confirmation step-back may oscillate |
| GC insight | No L1 `suggested_value` yet |
| ABANDON | Does not auto-revert last applied value |
| Target refresh | At most 3 refreshes per knob (`_MAX_TARGET_REFRESHES`); threshold 10% relative change |
| vCore budget guard | Rule 8 fires only when `executor_cores` is known on `TaskProfile`; prevents executor-count increase that raises total vCore footprint |
| Instance pricing | Not computed in this layer |

---

## Out of scope (this layer)

Pre-run failure probability; cold-task duration ML; multi-knob interaction effects; live cloud pricing for SKU changes; automatic revert on ABANDON.
