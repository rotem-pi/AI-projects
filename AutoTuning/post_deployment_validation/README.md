# Post-deployment validation: auto-tuning plan computation (2026-07-26)

**Post-deployment validation: `mock_build_plan.py` vs. live `/tuning/preview` endpoint.**
To validate the auto-tuning plan computation after deployment, we compared the local
`mock_build_plan.py` script against the live `GET /tuning/preview` endpoint across 9
representative production apps (tenant_id=29, env_name=prod), spanning a mix of Spark job
clusters and downstream compute tasks. Both were run against the same production Postgres
instance the backend connects to
(`terraform-20231213162521335900000002.canevgnvqwuj.eu-north-1.rds.amazonaws.com`), within
an ~11-minute window of each other. For each app, we compared every knob's `config_key`,
`target_value`, `cost_savings_usd_annual`, `priority_score`, `step_cost_savings_usd`,
`observed_signals`, and `blocked` status. All 9 apps produced **identical results** between
the mock script and the live endpoint, confirming the deployed `build_plan` algorithm and
`preview_service.py` are computing correctly in production, and that `mock_build_plan.py`
remains a faithful local stand-in for the live endpoint for future offline testing.

## Results

| seed task_id | app_id | app_name | task_name | endpoint's resolved task_id | match |
|---|---|---|---|---|---|
| 2719372 | 21776 | Job Cluster - prodgrowth_holdouts - job_cluster | compute | 2734361 | ✅ |
| 2719591 | 13704 | [prod]growth_holdouts | growth_holdouts_user_metrics_day_full_refresh | 2734613 | ✅ |
| 2723860 | 21544 | Job Cluster - prodplatinum_feedback_daily - platinum_feedback_cluster | compute | 2739134 | ✅ |
| 2723887 | 13964 | [prod]platinum_feedback_daily | platinum_feedback_user_day | 2739163 | ✅ |
| 2724220 | 28507 | Job Cluster - events_client_ui_interaction - events_job_cluster | compute | 2738454 | ✅ |
| 2724767 | 21609 | Job Cluster - prod prod-growth-user-day-statistics-workflow - growth_cluster | compute | 2740195 | ✅ |
| 2724800 | 7089 | [prod] prod-growth-user-day-statistics-workflow | text-check-statistics-feature-generation | 2740223 | ✅ |
| 2725667 | 21689 | Job Cluster - PROD NetSpring Demux - netspring_test_write_small | compute | 2741376 | ✅ |
| 2725893 | 7125 | [PROD] NetSpring Demux | metadata | 2733430 | ✅ |

"Match" = identical `config_key`, `target_value`, `cost_savings_usd_annual`,
`priority_score`, `step_cost_savings_usd`, `observed_signals`, and `blocked` status for
every knob and blocked_knob in the plan.

The "endpoint's resolved task_id" differs from each "seed task_id" in every row — this is
expected: neither the endpoint nor `mock_build_plan.py` accepts a specific historical
task_id as input. Both resolve `(app_id, task_name)` from the seed task_id and then build
the plan from whichever run is latest in the database at call time. The seed task_id is
only used to identify which app/task pair to test.

## Contents

- `mock_build_plan/task_<seed_task_id>.json` — local script output
- `endpoint_preview/task_<seed_task_id>.json` — live endpoint output (`plan` + `inputs`)
