# Auto-Tuning Policy — Brief HLD

**Scope:** `app/brain/insights/tuning/` · **Status:** Implemented (188 unit tests) · **Updated:** 2026-06-06

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

Targets are fixed at session-start (`POST /start`). After a session completes the user starts a new one to pick up refreshed insights.

---

## Pipeline (L1 → L2 → loop)

```
L1 Insight SQL (every N days)
  → insights row: type, payload, impact_cost, usd_cost
  → optional estimated_duration_impact_pct (executor sims)

L2a build_plan(insights, run_metrics, task_profile,
               pre_tuning_durations_ms, pre_tuning_cost_usd,
               tolerance, step_pct_cap, step_abs_cap, apply_mode)
  → extract rec → build staircase → block? → cost/duration est → priority
  → Plan: knobs (ranked), blocked_knobs, constraints, session_started_at

Apply active_knob.next_value → run job

L2b process_run_outcome(plan, outcome) → (Plan, str action)
  → search → confirming → completed | abandoned | paused
  → on terminal: activate next queued knob automatically

Post-run: assess_duration_regression, get_confirmation_status (3 clean runs)
```

---

## Inputs to `build_plan`

| Input | Source | Required |
|-------|--------|----------|
| `active_insights` | insights table | Yes |
| `tolerance` | user (`"strict"` 5% / `"standard"` 10% / `"flexible"` 20%) | No (default `"standard"`) |
| `step_pct_cap` | user | No (default 0.20) |
| `step_abs_cap` | user | No (default 5) |
| `apply_mode` | user (`"suggest"` / `"approve"` / `"auto"`) | No (default `"suggest"`) |
| `run_metrics` | latest `task_enrichments` | No (graceful degrade) |
| `task_profile` | enrichments / cluster config (incl. `executor_cores`) | No |
| `pre_tuning_durations_ms` | last N run durations (ms) | No (disables regression detection) |
| `pre_tuning_cost_usd` | avg cost of pre-tuning runs | No (disables cost abandon gate) |

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

---

## Step 2 — Staircase / step cap

Per-knob `KnobStepCap`: `max_step = max(abs_cap, current × pct_cap)`, further bounded by `gap_pct_cap × remaining_gap`. Staircase pre-built at plan time; stored on `Knob` for UI preview and rollback reference.

| Config key | pct | abs | gap_pct_cap | Notes |
|------------|-----|-----|-------------|-------|
| maxExecutors, executor.instances, cluster workers | 20% | 5–10 | 40% | cost_reduction_only |
| executor/driver memory, memoryOverhead | 15% | 4 GB | 33% | |
| driver.cores, executor.cores, task.cpus | 25% | 1–2 | 50% | driver: max_sane_current=64 |
| shuffle.partitions | 50% | 500 | 60% | faster ramp |
| change_instance_type, availability | — | — | — | expert_single_step (one hop) |

---

## Step 3 — Blocking (8 rules, first match wins)

| # | Condition | Block when |
|---|-----------|------------|
| **7** | Data quality | `current > max_sane_current` (e.g. driver.cores > 64) |
| **1** | AQE / Databricks | tune `spark.sql.shuffle.partitions` while AQE/auto-optimize active |
| **8** | vCore budget increase | `target × executor_cores > current × executor_cores` |
| **2** | Spill + shrink capacity | `has_spill` and ↓ executor count or memory |
| **3** | GC + shrink memory | `gc_pressure > 20%` and ↓ memory |
| **4a** | Tight heap | `memory_headroom < 10%` and ↓ heap memory |
| **4b** | Tight off-heap | `off_heap_headroom < 10%` and ↓ memoryOverhead |
| **5** | CPU saturated | `vcore_util ≥ 85%` and ↓ executor count |
| **6** | Skew + partition load | `skew > 30%`, ↓ executors, partitions/floor_executor > 100 |

During **confirmation** at target: Rules 2–6 do not apply (direction≠decrease); safety = failure rollback, cost abandon, duration step_back.

---

## Step 4 — Cost & duration (informational)

- **Full savings:** `cost_savings_usd_annual` from insight `usd_cost`.
- **Step savings:** `full × (initial − next) / (initial − target)` (linear assumption).
- **Duration Δ:** payload `estimated_duration_impact_pct` if present; else `risk_rules.py` heuristics. **Not used for blocking** — only priority adjustment.

---

## Step 5 — Priority

```
base = usd_cost_annual/10_000  OR  savings_pct × (2 if cost-insight else 1)
priority = base / (1 + duration_delta_pct/100)   if duration_delta > 0
         = base                                   otherwise
blocked → 0
```

---

## Output: `Plan`

```python
Plan:
    knobs: list[Knob]          # ranked, first non-queued/non-terminal = active
    blocked_knobs: list[Knob]  # shown in UI, never executed
    constraints: TuningConstraints
    session_started_at: str    # ISO-8601 UTC

    active_knob: Knob | None
    pending_knobs: list[Knob]  # "queued" phase — not yet started
    completed_knobs: list[Knob]
    is_complete: bool

Knob:                          # merges plan-time metadata + session state
    config_key, insight_type, action
    initial_value, target_value
    staircase: list             # pre-built; UI + rollback reference
    current_value, next_value   # updated each run
    phase: queued|search|confirming|paused|completed|abandoned
    run_history, step_back_count, ...
    cost_savings_usd_annual, duration_delta_pct_est
    confidence, observed_signals
```

---

## Post-run assessment (`run_assessor.py`)

- **Regression:** z = (duration − mean(baseline)) / std; needs ≥2 baseline samples. Thresholds: strict 1.28, standard 1.645, flexible 2.054.
- **Confirmation:** 3 consecutive runs at target with z below threshold (`CONFIRMATION_RUNS_REQUIRED = 3`).

---

## Autonomous loop (`loop.py`)

**Per-knob phases:** `queued` → `search` → (at target) `confirming` → `completed` | `abandoned` | `paused`.

**`process_run_outcome(plan, outcome) → (Plan, str)`** — mutates the active knob in-place; on terminal, activates next queued knob with fresh baselines.

**Decision priority:**

| # | Trigger | Action |
|---|---------|--------|
| 1 | Job failed/OOM/timeout | **ROLLBACK** to previous staircase value |
| 2 | `run_cost > pre_tuning_cost × 1.15` for 2 consecutive runs | **ABANDON** knob |
| 3–4 | Blocking signal (persists / new) | **PAUSE** |
| 5 | Duration regression in search | **PAUSE** |
| 6 | Regression in confirming | **STEP_BACK** one level |
| 7 | 3 clean confirmation runs | **COMPLETE**; activate next queued knob |
| 8 | Confirming, waiting | **STAY** |
| 9 | Else, not at target | **ADVANCE** |

**Rollback:** reverts to previous staircase entry; halves next step cap to avoid immediate retry of failed value.

**Paused:** holds config; re-checks blocking each run; resumes search or confirming when clear. Open-ended — no auto-abandon from paused state.

**Multi-knob:** `Plan.knobs` holds all knobs in priority order. One knob is active at a time; others are `"queued"`. When active knob completes/abandons, the next queued knob activates automatically with baselines from the finishing knob's last 5 runs.

**Cycle budget:** steps + 3 confirmation runs (e.g. 1 step → 4 runs; 6 steps → 9 runs). Dataset median ~4–5 runs/knob.

**API sketch:**
```python
# POST /start
plan = build_plan(insights, pre_tuning_durations_ms=[...], pre_tuning_cost_usd=50.0,
                  tolerance="standard", apply_mode="auto")

# GET /config (before every run)
config_override = plan.active_knob.next_value  # e.g. {"maxExecutors": 80}

# After each run (enrichment pipeline, inline)
plan, action = process_run_outcome(plan, RunOutcome(applied_value=80, ...))
# plan persisted to task_enrichments.tuning_state
```

---

## End-to-end example

```
L1: over_provisioned_executors, current=100, target=10, $8.4k/yr
→ staircase [80, 64, 51, 41, 33, 26, 21, 17, 13, 10]
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
| Sequential knobs | One knob at a time; no interaction modelling |
| Cost gate | Requires 2 consecutive cost-exceeding runs (+15% tolerance) before ABANDON |
| Regression detection | Needs ≥2 baseline durations |
| High-variance tasks | Confirmation step-back may oscillate |
| GC insight | No L1 `suggested_value` yet |
| ABANDON | Does not auto-revert last applied value |
| Paused state | Open-ended; a new POST /start is the mechanism to reset |
| vCore budget guard | Rule 8 fires only when `executor_cores` is known on `TaskProfile` |
| Instance pricing | Not computed in this layer |
| Target freshness | Targets fixed at POST /start; new session needed for refreshed insights |

---

## Out of scope (this layer)

Pre-run failure probability; cold-task duration ML; multi-knob interaction effects; live cloud pricing for SKU changes; automatic revert on ABANDON; auto-abandon from paused state.
