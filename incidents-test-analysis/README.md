# Alert incidents analysis — 8f5e1c0 test generation regression

In **[rotem-pi/AI-projects](https://github.com/rotem-pi/AI-projects)** this content lives under **`incidents-test-analysis/`** (not at the repo root).

## Links

- **Pull request:** [definity-app#4898](https://github.com/definity-ai/definity-app/pull/4898)
- **Design / write-up (Google Doc):** [Test generation & 8f5e1c0](https://docs.google.com/document/d/1t6GpLp5P31b0wVCkwgIrseXEHViFjrmHem81q-A51-I/edit?tab=t.0#heading=h.901ine1si5z0)

## Files

| File | Role |
|------|------|
| `README.md` | This document |
| `run_impact_tests.sh` | Runs pytest using **definity-app/backend/.venv** |
| `conftest.py` | Adds this directory + `definity-app/backend` to `PYTHONPATH`; placeholder DB env |
| `test_test_generation_8f5e1c0_impact.py` | Impact test (imports legacy from this directory) |
| `test_runs_base.csv` | Metric export (~480MB). **Gitignored** — copy here after clone, or rely on `../alerts-tests-analysis/test_runs_base.csv` when walking parents |
| `legacy_analytical_model_pre_8f5e1c0.py` | Frozen analytic model before 8f5e1c0 |
| `analytical_model_post_8f5e1c0_reference.py` | Reference copy of current `analytical_model.py` for diffs only |

## Process (short)

1. Build `MetricHistory` + snapshot test from CSV per `test_id` (eligible: ≥8 runs, any failure).
2. Train **legacy** vs **current** analytic models; `_final_action` simulates legacy vs production generator outcomes.

## Run

Needs **definity-app/backend** with `.venv` (sklearn, pandas, hydra, `app`).

```bash
git clone https://github.com/rotem-pi/AI-projects.git
cd AI-projects/incidents-test-analysis
# copy test_runs_base.csv into this directory (or use monorepo parent path)
./run_impact_tests.sh
```

Optional:

```bash
# Cohort cap — use ≥10 or omit (cherry-pick test needs 10+ ids)
TEST_RUNS_ANALYSIS_LIMIT=100 ./run_impact_tests.sh -q

TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE=0.2 ./run_impact_tests.sh -s

DEFINITY_BACKEND_ROOT=/path/to/definity-app/backend ./run_impact_tests.sh
```

Canonical Definity sources: refresh these copies from `definity-app` when you want the bundle to match the main repo.
