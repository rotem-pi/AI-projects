# Auto-Tuning Policy — High-Level Design

**Scope:** Spark / Databricks task auto-tuning policy layer (`app/brain/insights/tuning/`)
**Status:** Implemented, unit-tested (188 tests)
**Branch:** `cost-optimization-auto-tuning`
**Last updated:** 2026-06-06
**Audience:** DS model review

---

## 1. Goal

Given a set of active insights for a task (e.g. "executors are over-provisioned"), produce a **ranked, safe, gradual, self-driving config optimization** that:

1. Decides **what to change** — expert target from the insight layer
2. Decides **how far to go this iteration** — step cap (gradual ramp, not a cliff)
3. Decides **whether it is safe to apply now** — 8 deterministic blocking rules
4. Decides **after each run** — advance / pause / rollback / abandon / confirm

The system is designed for fully autonomous operation: it drives itself from the initial over-provisioned state to a confirmed optimized state without human intervention, while surfacing any anomaly that requires attention.

Targets are **fixed at session-start time** (POST /start). When a session completes, the user starts a new session to pick up any refreshed insight targets.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  L1 — Insight Detection  (SQL, runs on schedule)            │
│                                                             │
│  Per insight type (executors, memory, GC, spill, …):       │
│  • Detects condition from task_enrichments                  │
│  • Computes expert target config value                      │
│  • Computes impact_cost (VCore-seconds or GB-seconds/year)  │
│  • Converts to usd_cost via CALCULATE_USD_COST()            │
│  • For executor insights: simulates duration impact         │
│    (estimated_duration_impact_pct stored in payload)        │
│                                                             │
│  Output: insights table row                                 │
│    type, insights_payload, impact_cost, usd_cost            │
└──────────────────────────────┬──────────────────────────────┘
                               │  ActiveInsight(type, payload, usd_cost_annual)
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  L2a — Tuning Policy  (Python, called on POST /start)       │
│                                                             │
│  build_plan(insights, run_metrics, task_profile,            │
│             tolerance, step_pct_cap, …)  →  Plan            │
│                                                             │
│  Per recommendation:                                        │
│  1. Extract recommendation  (what to change, to what)       │
│  2. Compute staircase       (step cap, full ramp)           │
│  3. Check blocking          (8 deterministic rules)         │
│  4. Estimate cost / duration                                │
│  5. Score priority                                          │
│                                                             │
│  Output: Plan                                               │
│    knobs (ranked), blocked_knobs, constraints,              │
│    session_started_at                                       │
└──────────────────────────────┬──────────────────────────────┘
                               │  apply active_knob.next_value to cluster
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  L2b — Autonomous Loop  (loop.py, stateful)                 │
│                                                             │
│  process_run_outcome(plan, outcome)  →  (Plan, str)         │
│                                                             │
│  Per completed run:                                         │
│  1. Job failed?     → ROLLBACK to previous staircase value  │
│  2. Cost increased? → ABANDON, move to other knobs          │
│  3. Blocking signal? → PAUSE; re-evaluate next run          │
│  4. Regression mid-ramp? → PAUSE                            │
│  5. Confirmation regression? → STEP_BACK one level          │
│  6. 3 clean runs at target? → COMPLETE; activate next knob  │
│  7. Not at target yet? → ADVANCE to next step               │
│                                                             │
│  Knob phases: queued → search → confirming → completed      │
│               (or abandoned)                                │
└──────────────────────────────┬──────────────────────────────┘
                               │  confirmed config saved
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  Post-Run Assessment  (run_assessor.py, called by loop)     │
│                                                             │
│  assess_duration_regression(run_duration, baseline)         │
│  get_confirmation_status(run_configs, run_z_scores, target) │
│                                                             │
│  Output: DurationAssessment, ConfirmationStatus             │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. Inputs

| Input | Type | Source | Required |
|---|---|---|---|
| `active_insights` | `list[ActiveInsight]` | insights table | Yes |
| `tolerance` | `"strict"` / `"standard"` / `"flexible"` | user choice (POST /start) | No (default `"standard"`) |
| `step_pct_cap` | `float` | user choice | No (default 0.20) |
| `step_abs_cap` | `int` | user choice | No (default 5) |
| `apply_mode` | `"suggest"` / `"approve"` / `"auto"` | user choice | No (default `"suggest"`) |
| `run_metrics` | `TaskRunMetrics` | task_enrichments latest row | No (degrades gracefully) |
| `task_profile` | `TaskProfile` | task_enrichments / cluster config | No (degrades gracefully) |
| `pre_tuning_durations_ms` | `list[float]` | last N run durations | No (disables regression detection) |
| `pre_tuning_cost_usd` | `float` | avg of pre-tuning runs | No (disables cost abandon gate) |

**`TuningConstraints` — built internally by `build_plan`:**

```python
class TuningConstraints:
    def __init__(
        self,
        tolerance: "strict" | "standard" | "flexible" = "standard",
        step_pct_cap: float = 0.20,
        step_abs_cap: int = 5,
        apply_mode: "suggest" | "approve" | "auto" = "suggest",
    )
    # tolerance maps to max_duration_regression_pct:
    #   "strict" → 5%, "standard" → 10%, "flexible" → 20%
    # max_failure_prob = 0.05  (internal constant, not user-settable)
    # apply_mode: stored on Plan; enforcement is the API layer's responsibility
```

**`TaskRunMetrics` signals** (derived from `task_enrichments`):

| Signal | Formula | Threshold labels |
|---|---|---|
| `memory_headroom` | `1 − heap_max_used / heap_allocated` | ≥40% = ample, <15% = tight |
| `gc_pressure` | `gc_time / run_time` | >20% = severe, >10% = moderate |
| `off_heap_headroom` | `1 − off_heap_used / off_heap_allocated` | — |
| `vcore_utilization` | `vcore_time_used / vcore_time_allocated` | ≤40% = low, ≥80% = high |
| `idle_ratio` | `idle_time / task_duration` | >20% = high |
| `skew_ratio` | `skew_time / task_duration` | >25% = skew present |
| `has_spill` | `disk_bytes_spilled > 100 MB` | boolean |
| `spill_ratio` | `disk_bytes_spilled / total_io_bytes` | — |
| `retried_task_waste` | `retried_vcore_time / total_vcore_time` | >10% = high |
| `cpu_efficiency` | `cpu_time_ns / 1e6 / run_time_ms` | ≥80% = CPU-bound |

All ratios are clamped to `[0, 1]`. Missing columns → `None` (no blocking, no signals).

---

## 4. Step 1 — Extract Recommendations

**File:** `recommendations.py`

`extract_recommendations(insight_type, payload, task_profile=None)` dispatches to a per-insight extractor. Each extractor returns a list of structured dicts:

```python
{
    "action": "set_spark_config" | "set_cluster_config" | "change_instance_type",
    "config_key": "spark.dynamicAllocation.maxExecutors",   # None for instance changes
    "current_value": 20,
    "suggested_value": 14,
    "estimated_duration_impact_pct": 17,   # from SQL simulation, if available
}
```

**Active extractors and their recommendations:**

| Insight | Recommendation | Notes |
|---|---|---|
| `over_provisioned_executors` | reduce `maxExecutors` / `minExecutors` | |
| `over_provisioned_cluster_machines` | reduce `clusterMaxWorkers` / `clusterMinWorkers` | |
| `over_provisioned_executor_heap_memory` | reduce `spark.executor.memory` | |
| `over_provisioned_executor_off_heap_memory` | reduce `spark.executor.memoryOverhead` | |
| `over_provisioned_driver_heap_memory` | reduce `spark.driver.memory` | |
| `over_provisioned_driver_cores` | reduce `spark.driver.cores` OR `change_instance_type` | See Fix B2 below |
| `over_provisioned_machine_type` | `change_instance_type` (AWS: r→m, m→c) | |
| `executors_long_spill_time` | increase `spark.sql.shuffle.partitions`; AWS: switch to r-instance | Result capped at 20,000 |
| `under_utilized_executors_cpu` | reduce `spark.task.cpus` | |
| `spark_task_retries_cost` | `SPOT_WITH_FALLBACK` → `ON_DEMAND` | |
| `orphaned_node_resources` | reduce `clusterWorkers` / `clusterMaxWorkers` | vCore orphaning only |
| `under_provisioned_executor_heap_memory` | increase `spark.executor.memory` | |
| `long_idle_time` | reduce executor/worker count | Requires `task_profile`; INT_MAX guard; floor 50% of current |
| `long_skew_time` | double `spark.sql.shuffle.partitions` | Requires `task_profile`; AQE → [] |

**Extractors that always return `[]`**: `task_retries_cost`, `executors_long_gc_time`, `small_files`, `stale_task_runs`, `excessive_listing_operation`

---

## 5. Step 2 — Compute staircase (Step Cap)

**Files:** `knob_registry.py`, `step.py`

Each config key has a `KnobStepCap` entry. The full staircase is pre-computed when the `Knob` is created:

```
max_step = max(abs_cap, current × pct_cap)
           further bounded by gap_pct_cap × remaining_gap
next     = current + sign(target − current) × min(|target − current|, max_step)
```

The staircase is built once at plan time and stored on the `Knob`. It serves as a UI preview and a rollback reference.

**Registered caps:**

| Config key | pct_cap | abs_cap | gap_pct_cap | Special |
|---|---|---|---|---|
| `spark.dynamicAllocation.maxExecutors` | 20% | 10 | 40% | cost_reduction_only |
| `spark.executor.instances` | 20% | 10 | 40% | cost_reduction_only |
| `clusterMaxWorkers` / `clusterWorkers` | 20% | 10 | 40% | cost_reduction_only |
| `spark.executor.memory` | 15% | 4 GB | 33% | — |
| `spark.executor.memoryOverhead` | 15% | 4 GB | 33% | — |
| `spark.driver.memory` | 15% | 4 GB | 33% | — |
| `spark.driver.cores` | 25% | 2 | 50% | cost_reduction_only; `max_sane_current=64` |
| `spark.executor.cores` | 25% | 2 | 50% | — |
| `spark.sql.shuffle.partitions` | 50% | 500 | 60% | Faster ramp |
| `change_instance_type` | — | — | — | `expert_single_step`: one hop |
| `aws_attributes.availability` | — | — | — | `expert_single_step`: discrete |

---

## 6. Step 3 — Blocking Rules

**File:** `blocking.py`

Eight deterministic rules. Checked in order; first that fires returns. No probabilities, no heuristics.

| # | Condition | Scope |
|---|---|---|
| **7** | `current > max_sane_current` (data quality gate, e.g. driver.cores > 64) | any knob with registry limit |
| **1** | AQE or Databricks auto-optimize active; `spark.sql.shuffle.partitions` would be silently ignored | shuffle partitions only |
| **8** | Proposed change raises `target × executor_cores > current × executor_cores` (vCore budget increase) | max-executor knobs |
| **2** | `has_spill=True` and decreasing executor count or memory | executor / memory reduction |
| **3** | `gc_pressure > 20%` and decreasing memory | memory reduction |
| **4a** | `memory_headroom < 10%` and decreasing heap memory | heap memory reduction |
| **4b** | `off_heap_headroom < 10%` and decreasing `memoryOverhead` | memoryOverhead reduction |
| **5** | `vcore_utilization ≥ 85%` and decreasing executor count | executor reduction |
| **6** | `skew_ratio > 30%` AND `shuffle_partitions / floor_executors > 100` AND decreasing executors | executor reduction |

**Named thresholds** (`blocking.py`):

| Constant | Value |
|---|---|
| `GC_BLOCK_THRESHOLD` | 0.20 |
| `HEADROOM_BLOCK_THRESHOLD` | 0.10 |
| `VCORE_SATURATION_THRESHOLD` | 0.85 |
| `SKEW_BLOCK_THRESHOLD` | 0.30 |
| `PARTITIONS_PER_EXECUTOR_THRESHOLD` | 100 |

During **confirmation** (at target value), Rules 2–6 all require `direction=="decrease"` and do not apply. Safety relies on failure rollback, cost abandon, and duration step_back.

---

## 7. Step 4 — Cost & Duration Estimation

### 7a. Cost Savings

**Full potential** (`Knob.cost_savings_usd_annual`): annual savings at `target_value`, from `insights.usd_cost`.

**Step savings** (`Knob.step_cost_savings_usd()`):

```
step_savings = cost_savings_usd_annual × (initial_value − next_value) / (initial_value − target_value)
```

*Assumption: linear in resource count.*

### 7b. Duration Delta Estimate

**Priority 1** — SQL simulation (`estimated_duration_impact_pct` in payload, executor insights only).

**Priority 2** — `risk_rules.py` heuristic (all other types):

| Change type | Base duration delta | Modifiers |
|---|---|---|
| Executor large downscale (≥50%) | +18% | vcore ≤40% → ×0.70; vcore ≥80% → ×1.40; idle ≥60% → ×0.45 |
| Executor medium downscale (25–50%) | +10% | idle 30–60% → ×0.75; skew > 20% → ×(1+skew) |
| Memory large downscale (≥30%) | +12% | headroom ≥40% → ×0.60; headroom ≤15% → ×1.50 |
| Shuffle partition increase | +10–12% | has_spill → benefit discount applied |

Duration estimate is **informational only** — used only for priority score adjustment, never for blocking.

---

## 8. Step 5 — Priority Scoring

```
base =
    usd_cost_annual / 10,000          if usd_cost_annual is available
    savings_pct × cost_multiplier     otherwise

priority_score = base / (1 + duration_delta_pct / 100)   if delta > 0
               = base                                      otherwise
```

- `cost_multiplier = 2.0` for cost-labelled insights, `1.0` otherwise
- Blocked knobs → `priority_score = 0.0`

---

## 9. Output: Plan

```python
Plan:
    knobs: list[Knob]          # ranked by priority_score DESC; first non-queued is active
    blocked_knobs: list[Knob]  # blocked at plan-build time; shown in UI, never executed
    constraints: TuningConstraints
    session_started_at: str    # ISO-8601 UTC

    # Properties
    active_knob: Knob | None   # first knob in search/confirming/paused phase
    pending_knobs: list[Knob]  # knobs in "queued" phase (not yet started)
    completed_knobs: list[Knob]
    is_complete: bool

Knob:
    # Identity
    config_key, insight_type, action

    # The journey
    initial_value, target_value
    staircase: list[Any]       # pre-built ramp; UI preview + rollback reference
    current_value              # what was applied in the most recent run
    next_value                 # what to apply on the next run

    # Phase
    phase: "queued"|"search"|"confirming"|"paused"|"completed"|"abandoned"
    phase_reason: str

    # Session state
    run_history: list[RunOutcome]
    step_back_count: int
    consecutive_cost_exceedances: int
    halve_next_step: bool
    pause_cause: "blocking"|"regression"|None
    pre_tuning_durations_ms: list[float]
    pre_tuning_cost_usd: float | None

    # Plan metadata
    cost_savings_usd_annual: float | None
    step_cost_savings_usd() -> float | None
    priority_score: float
    blocked: bool
    block_reason: str | None
    duration_delta_pct_est: float | None
    confidence: "high"|"medium"|"low"
    observed_signals: tuple[str, ...]
```

---

## 10. Post-Run Assessment

**File:** `run_assessor.py` — called by the autonomous loop after every run.

### Duration Regression Detection

```
z = (run_duration_ms − mean(baseline)) / std(baseline)
```

Requires ≥ 2 baseline samples. Returns `DurationAssessment(is_regression, z_score, threshold_used)`.

| Preset | z-threshold | α | Meaning |
|---|---|---|---|
| `"strict"` (5%) | 1.28 | 10% | Any noticeable slowdown |
| `"standard"` (10%) | 1.645 | 5% | Default |
| `"flexible"` (20%) | 2.054 | 2% | Obvious outliers only |

### Confirmation Window

`get_confirmation_status(run_config_values, run_z_scores, target_value)` scans the tail of run history for consecutive runs at `target_value` with `z < threshold`.

`CONFIRMATION_RUNS_REQUIRED = 3`.

---

## 11. Autonomous Loop

**File:** `loop.py`

The loop is a **per-knob state machine** driven by `process_run_outcome(plan, outcome) → (Plan, str)`. The `Plan` is the unit of state — it wraps all knobs and is persisted to `task_enrichments.tuning_state` after every run.

### Knob Phases

| Phase | Meaning |
|---|---|
| `queued` | Created in the plan; not yet active; no baselines set |
| `search` | Actively stepping toward target |
| `confirming` | At target; collecting N consecutive clean runs |
| `paused` | Blocking signal or regression; holding at current value |
| `completed` | Confirmed at target — terminal |
| `abandoned` | Cost increased or max step-backs reached — terminal |

When the active knob completes or is abandoned, the loop automatically activates the next `queued` knob, using the finishing knob's last 5 successful runs as the new regression/cost baseline.

### State Machine

```
         ┌─────────────────────────────────────────────────────────┐
         │                   Per-knob phases                       │
         │                                                         │
         │   queued ──activate──▶ search                          │
         │                          │                             │
         │   ┌──────────┐   reach   ┌────────────┐   3 clean     │
         │   │  search  │──target──▶│ confirming │──runs──▶ COMPLETE│
         │   └────┬─────┘           └─────┬──────┘               │
         │        │                       │                        │
         │   signal/regress          regression                    │
         │        │                       │                        │
         │        ▼                       ▼                        │
         │   ┌─────────┐            step_back                     │
         │   │ paused  │◀───────────(returns to search)           │
         │   └────┬────┘                                          │
         │        │ signal resolves                                │
         │        └──────▶ resume (search or confirming)          │
         │                                                         │
         │   cost increase (2 consecutive runs) ──▶ ABANDONED     │
         │   job failure            ──▶ ROLLBACK + search         │
         └─────────────────────────────────────────────────────────┘
```

### Decision Rules (priority order)

| Priority | Trigger | Action | Effect |
|---|---|---|---|
| 1 | `job_status ∈ {oom, timeout, failed}` | **ROLLBACK** | Revert to previous staircase value; halve next step cap |
| 2 | `run_cost > pre_tuning_cost × 1.15` for 2 consecutive runs | **ABANDON** | Stop this knob; activate next queued knob |
| 3 | Blocking signal still present (was paused) | **PAUSE** (stay) | Keep current value; re-check next run |
| 4 | New blocking signal (in search) | **PAUSE** | Stop advancing; re-check next run |
| 5 | Duration regression in search (z ≥ threshold) | **PAUSE** | Stop advancing; re-check next run |
| 6 | Duration regression in confirmation | **STEP_BACK** | Revert one staircase level; reopen search |
| 7 | 3 clean confirmation runs | **COMPLETE** | Activate next queued knob; or session complete |
| 8 | In confirming, waiting | **STAY** | No config change; count runs |
| 9 | No issues, not at target | **ADVANCE** | Apply next staircase step |

### API

```python
# Build plan at POST /start
plan = build_plan(
    active_insights=[
        ActiveInsight("over_provisioned_executors",
                      {"max_executors_suggestion": 10, "dynamic_max_executors_conf": 100},
                      usd_cost_annual=8400.0)
    ],
    run_metrics=compute_run_metrics(enrichment_row),
    task_profile=compute_task_profile(enrichment_row),
    pre_tuning_durations_ms=[1000.0, 1010.0, 990.0],
    pre_tuning_cost_usd=50.0,
    tolerance="standard",  # or "strict" / "flexible"
    apply_mode="auto",
)

# Before every run
config_override = plan.active_knob.next_value   # e.g. 80

# After each run (called by enrichment pipeline inline)
plan, action = process_run_outcome(plan, RunOutcome(
    applied_value   = 80,
    job_status      = "success",
    run_cost_usd    = 45.20,
    run_duration_ms = 1050.0,
    run_metrics     = compute_run_metrics(enrichment_row),
    task_profile    = compute_task_profile(enrichment_row),
))
# action ∈ {"advance","stay","pause","rollback","step_back","abandon","complete"}
# plan.active_knob.next_value = config to apply on next run
# plan is serialized to task_enrichments.tuning_state
```

### Session Lifecycle

```
POST /start → build_plan() → Plan persisted to tuning_state
  │
  ▼ (repeat until plan.is_complete)
GET /config → plan.active_knob.next_value → applied to job
  │
  ▼ (job completes, enrichment pipeline runs)
process_run_outcome(plan, outcome) → (plan, action)
  → plan re-persisted to tuning_state
  │
  ├── action == "complete" + pending_knobs → activate next knob, continue
  └── plan.is_complete → session done; user may POST /start again
```

### Total Cycle Budget

| Staircase steps | + Confirmation runs | = Total runs |
|---|---|---|
| 1 (target in one step) | 3 | **4** |
| 2 | 3 | **5** |
| 6 (max seen in dataset) | 3 | **9** |

Dataset median ~4–5 runs/knob.

---

## 12. Full End-to-End Flow

```
[Task runs normally]
        │
        ▼
[L1 SQL detects over-provisioning]
  → insight row: type=over_provisioned_executors,
    payload={max_executors_suggestion:10, current:100},
    usd_cost=$8,400/yr
        │
        ▼
[POST /start]
plan = build_plan([ActiveInsight(...)],
                  pre_tuning_durations_ms=[...],
                  pre_tuning_cost_usd=50.0)
  → staircase: [80, 64, 51, 41, 33, 26, 21, 17, 13, 10]
  → plan.active_knob.next_value = 80
        │
        ▼
[Apply 80 → run job]
  → RunOutcome(applied=80, cost=$40, duration=1050ms, metrics=...)
  → process_run_outcome → ADVANCE → next_value=64
        │
[Apply 64 → run job]
  → RunOutcome(applied=64, cost=$32, duration=1080ms, has_spill=False)
  → process_run_outcome → ADVANCE → next_value=51
        │
  ...   (continue stepping)
        │
[Apply 10 → run job]   ← first confirmation run
  → process_run_outcome → STAY (1/3 confirms)
        │
[Apply 10 → run job]   ← second confirmation run
  → process_run_outcome → STAY (2/3 confirms)
        │
[Apply 10 → run job]   ← third confirmation run
  → process_run_outcome → COMPLETE ✓ → plan.is_complete = True
        │
        ▼
[Confirmed: maxExecutors=10 saves $8,400/yr]
[User may POST /start again to pick up refreshed insights]
```

---

## 13. Assumptions & Limitations

| Assumption / Limitation | Impact |
|---|---|
| Cost savings are linear in resource count | Slightly inaccurate `step_cost_savings` for large changes; priority ranking unaffected |
| L1 SQL duration simulation covers executor insights only | Other knobs fall back to `risk_rules` heuristics |
| `risk_rules` heuristics calibrated for typical Spark jobs | May over/under-estimate for unusual workloads |
| Baseline z-score is Gaussian | Heavy-tailed duration distributions may give spurious regressions |
| Config changes applied one knob at a time | Interaction effects between knobs are not modelled |
| `run_cost_usd` availability | Cost-increase ABANDON rule is disabled when cost is unavailable |
| Pre-tuning baseline quality | Duration regression detection is disabled when `pre_tuning_durations_ms` has < 2 entries |
| Paused state is open-ended | A persistent blocking condition (spill that never resolves) keeps the session paused indefinitely; a new POST /start restarts it |
| Single confirmation regression → step back | One noisy run causes a step-back; a very high-variance task could oscillate between the last two staircase values |
| Targets fixed at session start | If insights are refreshed mid-session, the new targets take effect only in the next session |

---

## 14. What Is Not In This Layer

| Capability | Status |
|---|---|
| Pre-run failure probability | Removed — no labeled outcome data to calibrate |
| Cold-task duration prediction | Not attempted — config alone explains ~6% variance |
| Multi-step interaction effects | Not modelled — knobs are tuned sequentially, one at a time |
| Cloud pricing for instance type changes | Not computed — requires live pricing API |
| `executors_long_gc_time` recommendations | L1 gap — payload has no `suggested_value`; requires fix in SQL layer |
| Automatic rollback after ABANDON | Loop abandons knob and moves on; it does not revert the last applied step (caller responsibility) |
| ML-based duration prediction | Future — requires paired (before/after) outcome data per config change |
| Auto-abandon after N paused runs | Not implemented — paused state is open-ended by design |
