# Auto-Tuning Policy — High-Level Design

**Scope:** Spark / Databricks task auto-tuning policy layer (`app/brain/insights/tuning/`)
**Status:** Implemented, unit-tested (153 tests)
**Branch:** `cost-optimization-auto-tuning`
**Last updated:** 2026-05-30
**Audience:** DS model review

---

## 1. Goal

Given a set of active insights for a task (e.g. "executors are over-provisioned"), produce a **ranked, safe, gradual, self-driving config optimization** that:

1. Decides **what to change** — expert target from the insight layer
2. Decides **how far to go this iteration** — step cap (gradual ramp, not a cliff)
3. Decides **whether it is safe to apply now** — 7 deterministic blocking rules
4. Decides **after each run** — advance / pause / rollback / abandon / confirm

The system is designed for fully autonomous operation: it drives itself from the initial over-provisioned state to a confirmed optimized state without human intervention, while surfacing any anomaly that requires attention.

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
│  L2a — Tuning Policy  (Python, called on-demand)            │
│                                                             │
│  build_tuning_plan(insights, constraints, run_metrics,      │
│                    task_profile)  →  TuningPlan             │
│                                                             │
│  Per recommendation:                                        │
│  1. Extract recommendation  (what to change, to what)       │
│  2. Compute next_value      (step cap)                      │
│  3. Check blocking          (7 deterministic rules)         │
│  4. Estimate cost / duration                                │
│  5. Score priority                                          │
│                                                             │
│  Output: TuningPlan                                         │
│    actions (ranked), blocked_actions, constraints           │
└──────────────────────────────┬──────────────────────────────┘
                               │  apply next_value to cluster
                               ▼
┌─────────────────────────────────────────────────────────────┐
│  L2b — Autonomous Loop  (loop.py, stateful)                 │
│                                                             │
│  process_run_outcome(state, outcome)  →  LoopDecision       │
│                                                             │
│  Per completed run:                                         │
│  1. Job failed?     → ROLLBACK to previous staircase value  │
│  2. Cost increased? → ABANDON, move to other knobs          │
│  3. Blocking signal? → PAUSE; re-evaluate next run          │
│  4. Regression mid-ramp? → PAUSE                            │
│  5. Confirmation regression? → STEP_BACK one level          │
│  6. 3 clean runs at target? → COMPLETE                      │
│  7. Not at target yet? → ADVANCE to next step               │
│                                                             │
│  Phases: search → confirming → completed / abandoned        │
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
| `constraints` | `TuningConstraints` | user/org config | No (defaults apply) |
| `run_metrics` | `TaskRunMetrics` | task_enrichments latest row | No (degrades gracefully) |
| `task_profile` | `TaskProfile` | task_enrichments / cluster config | No (degrades gracefully) |

**`TuningConstraints` fields (user-facing):**

| Field | Default | Meaning |
|---|---|---|
| `max_duration_regression_pct` | 10% | Post-run regression tolerance (→ z-threshold) |
| `default_step_pct_cap` | 20% | Max % change per iteration |
| `default_step_abs_cap` | 5 | Max absolute change per iteration |
| `apply_mode` | `"suggest"` | suggest / approve / auto |

User-facing preset aliases: `"strict"` → 5%, `"standard"` → 10%, `"flexible"` → 20%.

**`TaskRunMetrics` signals** (derived from `task_enrichments`):

| Signal | Formula | Threshold labels |
|---|---|---|
| `memory_headroom` | `1 − heap_max_used / heap_allocated` | ≥40% = ample, <15% = tight |
| `gc_pressure` | `gc_time / run_time` | >20% = severe, >10% = moderate |
| `off_heap_headroom` | `1 − off_heap_used / off_heap_allocated` | — |
| `vcore_utilization` | `vcore_time_used / vcore_time_allocated` | ≤40% = low, ≥80% = high |
| `idle_ratio` | `idle_time / task_duration` | >20% = high |
| `skew_ratio` | `skew_time / task_duration` | >25% = skew present |
| `has_spill` | `disk_bytes_spilled > 0` | boolean |
| `spill_ratio` | `disk_bytes_spilled / total_io_bytes` | — |
| `retried_task_waste` | `retried_vcore_time / total_vcore_time` | >10% = high |
| `cpu_efficiency` | `cpu_time_ns / 1e6 / run_time_ms` | ≥80% = CPU-bound |

All ratios are clamped to `[0, 1]`. Missing columns → `None` (no blocking, no signals).

---

## 4. Step 1 — Extract Recommendations

**File:** `recommendations.py`

`extract_recommendations(insight_type, payload, task_profile=None)` dispatches to a per-insight extractor. Extractors registered in `_TASK_PROFILE_EXTRACTORS` receive `task_profile` as a keyword argument — all others take only the payload.

Each extractor returns a list of structured dicts:

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
| `executors_long_spill_time` | increase `spark.sql.shuffle.partitions`; AWS: switch to r-instance | Result capped at 20,000; `min_sug > max_sug` guarded |
| `under_utilized_executors_cpu` | reduce `spark.task.cpus` | |
| `spark_task_retries_cost` | `SPOT_WITH_FALLBACK` → `ON_DEMAND` | |
| `orphaned_node_resources` | reduce `clusterWorkers` / `clusterMaxWorkers` | vCore orphaning only; memory-only orphaning → []. SQL also guards with `AND sum_orphan_vcore_s > 0` |
| `under_provisioned_executor_heap_memory` | increase `spark.executor.memory` | |
| `long_idle_time` | reduce executor/worker count | Requires `task_profile`; INT_MAX guard (see below); `idle_pct` clamped to 95%; suggested value floored at 50% of current |
| `long_skew_time` | double `spark.sql.shuffle.partitions` | Requires `task_profile`; AQE → []. Result capped at 20,000 |

**Extractors that always return `[]`** (no automatable lever):
`task_retries_cost`, `executors_long_gc_time` (L1 gap — no `suggested_value` in payload),
`small_files`, `stale_task_runs`, `excessive_listing_operation`

### Fix B2 — Driver Cores Sanitization (`over_provisioned_driver_cores`)

`driver__cores` in `task_enrichments` stores the driver **machine's allocated vCPU count**, not the `spark.driver.cores` Spark parameter. For a 256-vCPU cloud node, this produces `current=256` — which would create a meaningless 15-step staircase to `target=2`.

**Fix:** the extractor remaps `current_value = None` when `current > 64` (the maximum plausible `spark.driver.cores` Spark config value). `compute_next_value(None, target, ...)` returns `target` directly → single-step recommendation. Rule 7 in `blocking.py` acts as a backstop for any case the extractor misses.

### INT_MAX Guard (`long_idle_time`)

`spark.dynamicAllocation.maxExecutors` defaults to `Integer.MAX_VALUE` (2,147,483,647) when not configured — it means "no upper limit." Applying the idle-fraction formula to this value produces targets in the hundreds of millions of executors.

**Guard:** when `max_executors_param >= 2_000_000_000`, the extractor returns `[]`. Without a real ceiling there is no actionable recommendation.

### `long_skew_time` — Requires `task_profile`

The `long_skew_time` payload does not carry shuffle partition info. The extractor reads `task_profile.shuffle_partitions_param` instead. Registered in `_TASK_PROFILE_EXTRACTORS` so the dispatch layer injects `task_profile`. When AQE is active (`param == "auto"`), returns `[]` — Rule 1 would block it anyway.

---

## 5. Step 2 — Compute next_value (Step Cap)

**Files:** `knob_registry.py`, `step.py`

Each config key has a `KnobStepCap` entry:

```
max_step = max(abs_cap, current × pct_cap)
step     = min(|target − current|, max_step)
next     = current + sign(target − current) × step
```

When `current is None` (e.g. after Fix B2 remaps an implausible value): `next = target` directly (single step).

**Registered caps:**

| Config key | pct_cap | abs_cap | Special |
|---|---|---|---|
| `spark.dynamicAllocation.maxExecutors` | 20% | 10 | cost_reduction_only |
| `spark.executor.instances` | 20% | 10 | cost_reduction_only |
| `clusterMaxWorkers` / `clusterWorkers` | 20% | 10 | cost_reduction_only |
| `spark.executor.memory` | 15% | 4 GB | — |
| `spark.executor.memoryOverhead` | 15% | 4 GB | — |
| `spark.driver.memory` | 15% | 4 GB | — |
| `spark.driver.cores` | 25% | 2 | cost_reduction_only; `max_sane_current=64` |
| `spark.executor.cores` | 25% | 2 | — |
| `spark.sql.shuffle.partitions` | 50% | 500 | Faster ramp (expert often doubles) |
| `change_instance_type` | — | — | `expert_single_step`: one hop |
| `aws_attributes.availability` | — | — | `expert_single_step`: discrete |

---

## 6. Step 3 — Blocking Rules

**File:** `blocking.py`

Seven deterministic rules. Checked in order; first that fires returns. No probabilities, no heuristics.

### Rule 7: Implausible Current Value (Enrichment Data Quality Gate)

**Checked first — requires no run_metrics.**

Each `KnobStepCap` in the registry can declare `max_sane_current`. When `current > max_sane_current`, the recorded value almost certainly reflects an enrichment bug (e.g. machine vCPU count stored in the wrong column). Tuning through bad data produces extreme changes — blocked and surfaced to the operator.

Currently applies to: `spark.driver.cores` (`max_sane_current=64`).

```
IF config_key has max_sane_current
AND current > max_sane_current
THEN  BLOCK (data quality issue — verify task_enrichments)
```

Note: Fix B2 sanitizes `over_provisioned_driver_cores` payloads before this rule is reached, so Rule 7 acts as a backstop for any other path that passes a raw implausible value.

### Rule 1: AQE / Databricks Override (`spark.sql.shuffle.partitions`)

AQE coalesces shuffle partitions at runtime; Databricks auto-optimize shuffle does the same. The static config value is silently overwritten — tuning it is physically pointless.

```
IF config_key == "spark.sql.shuffle.partitions"
AND (AQE active OR Databricks auto-optimize OR param is "auto" / None)
THEN  BLOCK
```

### Rule 2: Active Spill + Reducing Executor Count or Memory

Spill → data overflows executor memory to disk. Reducing capacity (executor count or memory) worsens spill throughput or causes OOM on the hot executor.

```
IF has_spill == True
AND direction == "decrease"
AND config_key ∈ {executor count keys OR memory keys}
THEN  BLOCK
```

### Rule 3: Severe GC Pressure + Reducing Memory

GC consuming > 20% of executor runtime = heap under severe pressure ("GC death spiral" threshold). Reducing heap tightens it further.

```
IF gc_pressure > 0.20
AND direction == "decrease"
AND config_key ∈ {all memory keys}
THEN  BLOCK
```

### Rule 4a: Tight Heap Headroom + Reducing Heap Memory

Executor using > 90% of heap with no visible GC pressure. The next large operation (broadcast, sort, aggregation) will OOM.

```
IF memory_headroom < 0.10
AND direction == "decrease"
AND config_key ∈ {spark.executor.memory, spark.driver.memory}
THEN  BLOCK
```

### Rule 4b: Tight Off-Heap Headroom + Reducing memoryOverhead

Off-heap exhaustion → OS kills the executor process (not JVM). Harder to debug than Java OOM.

```
IF off_heap_headroom < 0.10
AND direction == "decrease"
AND config_key == "spark.executor.memoryOverhead"
THEN  BLOCK
```

### Rule 5: CPU-Saturated Executors + Reducing Executor Count

≥85% vCore utilization = no idle capacity. Reducing executor count queues more tasks per executor; may push marginal executors over memory budget.

```
IF vcore_utilization >= 0.85
AND direction == "decrease"
AND config_key ∈ {executor count keys}
THEN  BLOCK
```

### Rule 6: High Partition-to-Executor Ratio + Task Skew + Reducing Executors

Skew + many partitions per executor → hot executor concentrates disproportionate data → OOM. Requires static partition count; skipped when AQE controls partitions.

```
IF skew_ratio > 0.30
AND direction == "decrease"
AND config_key ∈ {executor count keys}
AND shuffle_partitions is a parseable integer
AND shuffle_partitions / target_executor_count > 100
THEN  BLOCK
```

**Blocking rule thresholds** (named constants in `blocking.py`):

| Constant | Value |
|---|---|
| `GC_BLOCK_THRESHOLD` | 0.20 |
| `HEADROOM_BLOCK_THRESHOLD` | 0.10 |
| `VCORE_SATURATION_THRESHOLD` | 0.85 |
| `SKEW_BLOCK_THRESHOLD` | 0.30 |
| `PARTITIONS_PER_EXECUTOR_THRESHOLD` | 100 |

---

## 7. Step 4 — Cost & Duration Estimation

### 7a. Cost Savings

**Full potential** (`TuningAction.cost_savings_usd_annual`): annual savings at `target_value`, from `insights.usd_cost`.

**Step savings** (`step_cost_savings_usd()`):

```
step_savings = cost_savings_usd_annual × (current − next_value) / (current − target_value)
```

*Assumption: linear in resource count.*

### 7b. Duration Delta Estimate

**Priority 1** — SQL simulation (`estimated_duration_impact_pct` in payload, executor insights only).

**Priority 2** — `risk_rules.py` heuristic (all other types):

| Change type | Base duration delta | Modifiers |
|---|---|---|
| Executor large downscale (≥50%) | +18% | vcore ≤40% → ×0.70; vcore ≥80% → ×1.40 |
| Executor medium downscale (25–50%) | +10% | skew_ratio > 0.20 → ×(1+skew) |
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
- Duration risk adjustment: lower-impact actions (e.g. +3% duration) rank above higher-risk ones at similar dollar value
- Blocked actions → `priority_score = 0.0`

---

## 9. Output: TuningPlan

```python
TuningPlan:
    actions: list[TuningAction]          # ranked by priority_score DESC
    blocked_actions: list[TuningAction]
    active_insight_types: list[str]
    constraints: TuningConstraints

TuningAction:
    config_key, current_value, target_value, next_value
    step_policy                          # "pct_cap" | "expert_single_step"
    insight_type
    blocked, block_reason
    cost_savings_usd_annual              # full potential at target
    step_cost_savings_usd()              # this step's share
    duration_delta_pct_est               # informational, positive = slower
    risk: RiskEstimate
        confidence                       # "high" if run_metrics present
        observed_signals                 # tuple of descriptive labels
    priority_score
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
| `strict` (5%) | 1.28 | 10% | Any noticeable slowdown |
| `standard` (10%) | 1.645 | 5% | Default |
| `flexible` (20%) | 2.054 | 2% | Obvious outliers only |

### Confirmation Window

`get_confirmation_status(run_config_values, run_z_scores, target_value)` scans the tail of run history for consecutive runs at `target_value` with `z < threshold`.

`CONFIRMATION_RUNS_REQUIRED = 3`. Why 3: at σ=1.5 a single-run false-negative rate is ~7%; over 3 runs it drops to ~0.03%.

---

## 11. Autonomous Loop

**File:** `loop.py`

The loop is a **per-knob state machine** that drives from the initial value to a confirmed optimized value across multiple job runs, reacting to each run's outcome.

### State Machine

```
         ┌─────────────────────────────────────────────────────────┐
         │                   Per-knob phases                       │
         │                                                         │
         │   ┌─────────┐   reach    ┌────────────┐   3 clean      │
         │   │ search  │──target──▶│ confirming │──runs──▶ COMPLETE│
         │   └────┬────┘           └─────┬──────┘                 │
         │        │                      │                         │
         │   signal/regress         regression                     │
         │        │                      │                         │
         │        ▼                      ▼                         │
         │   ┌─────────┐            step_back                      │
         │   │ paused  │◀──────────(returns to search)             │
         │   └────┬────┘                                           │
         │        │ signal                                         │
         │        │ resolves                                        │
         │        └──────▶ resume (search or confirming)           │
         │                                                         │
         │   cost increase (any phase) ──▶ ABANDONED               │
         │   job failure   (any phase) ──▶ ROLLBACK + search       │
         └─────────────────────────────────────────────────────────┘
```

### Decision Rules (priority order)

| Priority | Trigger | Action | Effect |
|---|---|---|---|
| 1 | `job_status ∈ {oom, timeout, failed}` | **ROLLBACK** | Revert to previous staircase value; stay in search |
| 2 | `run_cost_usd > pre_tuning_cost_usd` | **ABANDON** | Stop tuning this knob; other knobs continue. Compares against pre-tuning baseline (not previous run), so legitimate savings are not penalised by natural run-cost variance |
| 3 | Blocking signal still present (was paused) | **PAUSE** (stay) | Keep current value; re-check next run |
| 4 | New blocking signal appeared (in search) | **PAUSE** | Stop advancing; re-check next run |
| 5 | Duration regression in search (z ≥ threshold) | **PAUSE** | Stop advancing; re-check next run |
| 6 | Duration regression in confirmation | **STEP_BACK** | Revert one staircase level; reopen search |
| 7 | 3 clean confirmation runs | **COMPLETE** | Knob is done |
| 8 | In confirming, waiting | **STAY** | No config change; count runs |
| 9 | No issues, not at target | **ADVANCE** | Apply next staircase step |

### ROLLBACK — Mid-Stair Handling

When a job fails, the loop reverts to the _previous staircase value_. The staircase is the sequence of planned steps (e.g. `[80, 64, 51, …, 10]`). If a halved-step or floating-point rounding left `current_applied_value` between two staircase entries, the previous-value lookup finds the nearest entry **between current and initial** (i.e. the entry just above `current` when going down). This prevents jumping all the way back to the initial value when only a fractional step had been applied.

### Confirmation Phase Safety

During confirmation the job runs at `target_value` — no further decrease is attempted. The directional blocking rules (Rules 2–6 all require `direction=="decrease"`) do not apply. Safety during confirmation relies on:
- Job failure → ROLLBACK
- Cost increase → ABANDON
- Duration regression (z-score) → STEP_BACK

### Paused State Behaviour

When paused:
- The orchestrator keeps the current config value unchanged
- After each subsequent run, blocking is re-checked
- When the signal clears: if `current == target` → resume confirming; else → resume search
- `consecutive_paused_runs` is incremented each run (surfaced for operator visibility)

### API

```python
# Initialization
state = create_knob_state(
    config_key              = "spark.dynamicAllocation.maxExecutors",
    current_value           = 100,
    target_value            = 10,
    pre_tuning_durations_ms = [1000.0, 1010.0, 990.0],  # baseline for z-scores
    pre_tuning_cost_usd     = 50.0,                      # baseline cost for Rule 2 gate
)

# Before the first run
first_val = get_initial_next_value(state)   # e.g. 80

# After each run
state, decision = process_run_outcome(state, RunOutcome(
    applied_value   = 80,
    job_status      = "success",
    run_cost_usd    = 45.20,
    run_duration_ms = 1050.0,
    run_metrics     = compute_run_metrics(enrichments_row),
    task_profile    = compute_task_profile(enrichments_row),
))
# decision.next_action ∈ {"advance","stay","pause","rollback","step_back","abandon","complete"}
# decision.next_value  = config value to apply on next run
```

### Multi-Knob Sessions

```python
session = TaskTuningSession(task_id=42, knob_states={
    "spark.dynamicAllocation.maxExecutors": state_a,
    "spark.executor.memory": state_b,
})

# Loop
while not session.is_complete:
    config = {k: get_initial_next_value(s) for k, s in session.knob_states.items()}
    # ... submit job with config ...
    for key, state in session.knob_states.items():
        _, decision = process_run_outcome(state, run_outcome_for(key))
        config[key] = decision.next_value
```

### Total Cycle Budget

Typical runs to fully optimize one knob:

| Staircase steps | + Confirmation runs | = Total runs |
|---|---|---|
| 1 (target in one step) | 3 | **4** |
| 2 | 3 | **5** |
| 6 (max seen in dataset) | 3 | **9** |

In the current dataset: mean staircase = 1.5 steps → median total = **4–5 job runs per knob**.

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
[create_knob_state("spark.dynamicAllocation.maxExecutors", 100, 10,
                   pre_tuning_durations_ms=[...])]
  → staircase: [80, 64, 51, 41, 33, 26, 21, 17, 13, 10]
        │
        ▼
[Apply 80 → run job]
  → RunOutcome(applied=80, cost=$40, duration=1050ms, metrics=...)
  → process_run_outcome → ADVANCE → next=64
        │
[Apply 64 → run job]
  → RunOutcome(applied=64, cost=$32, duration=1080ms, has_spill=False)
  → process_run_outcome → ADVANCE → next=51
        │
  ...   (continue stepping)
        │
[Apply 10 → run job]   ← first confirmation run
  → RunOutcome(applied=10, cost=$5, duration=1100ms)
  → process_run_outcome → STAY (1/3 confirms)
        │
[Apply 10 → run job]   ← second confirmation run
  → process_run_outcome → STAY (2/3 confirms)
        │
[Apply 10 → run job]   ← third confirmation run
  → process_run_outcome → COMPLETE ✓
        │
        ▼
[Confirmed: maxExecutors=10 saves $8,400/yr]
```

---

## 13. Assumptions & Limitations

| Assumption / Limitation | Impact |
|---|---|
| Cost savings are linear in resource count | Slightly inaccurate `step_cost_savings` for large changes; priority ranking unaffected |
| L1 SQL duration simulation covers executor insights only | Other knobs fall back to `risk_rules` heuristics |
| `risk_rules` heuristics calibrated for typical Spark jobs | May over/under-estimate for unusual workloads |
| Baseline z-score is Gaussian | Heavy-tailed duration distributions may give spurious regressions |
| Config changes applied independently per knob | Interaction effects between simultaneous changes are not modelled; `TaskTuningSession` applies knobs in parallel without cross-knob re-evaluation |
| `run_cost_usd` availability | Cost-increase ABANDON rule is disabled when cost is unavailable |
| `pre_tuning_cost_usd` availability | ABANDON rule falls back to comparing against previous run cost when `pre_tuning_cost_usd` is `None`; set it from the mean of baseline runs to avoid false abandonments |
| Pre-tuning baseline quality | Duration regression detection is disabled when `pre_tuning_durations_ms` has < 2 entries |
| Single confirmation regression → step back | One noisy run causes a regression and step-back; a very high-variance task could oscillate between the last two staircase values |

---

## 14. What Is Not In This Layer

| Capability | Status |
|---|---|
| Pre-run failure probability | Removed — no labeled outcome data to calibrate |
| Cold-task duration prediction | Not attempted — config alone explains ~6% variance |
| Multi-step interaction effects | Not modelled — knobs are assumed independent |
| Cloud pricing for instance type changes | Not computed — requires live pricing API |
| `executors_long_gc_time` recommendations | L1 gap — payload has no `suggested_value`; requires fix in SQL layer |
| Automatic rollback after ABANDON | Loop abandons knob and moves on; it does not revert the last applied step (caller responsibility) |
| ML-based duration prediction | Future — requires paired (before/after) outcome data per config change |
