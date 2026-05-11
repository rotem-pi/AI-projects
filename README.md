# Alert incidents analysis — 8f5e1c0 test generation regression

Flat folder: **README, runner, conftest, impact test copy, legacy model snapshot, reference analytic model.** The CSV export is **not committed** to GitHub (file size); copy `test_runs_base.csv` locally (see below).

Canonical code lives in **definity-app**; refresh copies here when you want this folder to track the repo.

**Clone of [rotem-pi/AI-projects](https://github.com/rotem-pi/AI-projects):** after `git clone`, place `test_runs_base.csv` in this directory (or rely on `../alerts-tests-analysis/test_runs_base.csv` in a monorepo checkout).

## Links

- **Pull request:** [definity-app#4898](https://github.com/definity-ai/definity-app/pull/4898)
- **Design / write-up (Google Doc):** [Test generation & 8f5e1c0](https://docs.google.com/document/d/1t6GpLp5P31b0wVCkwgIrseXEHViFjrmHem81q-A51-I/edit?tab=t.0#heading=h.901ine1si5z0)

## Files

| File | Role |
|------|------|
| `README.md` | This document |
| `run_impact_tests.sh` | Runs pytest on the impact test using **definity-app/backend/.venv** |
| `conftest.py` | Puts this directory + `definity-app/backend` on `PYTHONPATH`; placeholder DB env |
| `test_test_generation_8f5e1c0_impact.py` | Copy of Definity impact test (imports legacy from this dir) |
| `test_runs_base.csv` | Metric run export (~480MB). **Gitignored** for GitHub; add locally after clone. |
| `legacy_analytical_model_pre_8f5e1c0.py` | Frozen `AnalyticTestTimeSeriesModel` before 8f5e1c0 |
| `analytical_model_post_8f5e1c0_reference.py` | Copy of current `app/tests_gen/models/analytical_model.py` for diffing; not imported by the test |

## Process (short)

1. Build `MetricHistory` + snapshot test from CSV per `test_id` (eligible: ≥8 runs, any failure).
2. Train **legacy** vs **current** analytic models; `_final_action` simulates legacy vs production generator outcomes (including optional low-score delete on the new side).

## Run

Needs **definity-app/backend** with `.venv` (sklearn, pandas, hydra, `app`).

```bash
cd /path/to/incidents-test-analysis   # or …/alert-quality-evaluation/incidents-test-analysis
./run_impact_tests.sh
```

Optional:

```bash
# Cohort cap — use ≥10 or omit (cherry-pick test needs 10+ ids)
TEST_RUNS_ANALYSIS_LIMIT=100 ./run_impact_tests.sh -q

TEST_RUNS_IMPACT_AUTO_DELETE_BELOW_SCORE=0.2 ./run_impact_tests.sh -s

DEFINITY_BACKEND_ROOT=/path/to/definity-app/backend ./run_impact_tests.sh
```

CSV is resolved as **`./test_runs_base.csv`** first, else **`../alerts-tests-analysis/test_runs_base.csv`** when walking parents.

## Push this folder to GitHub ([rotem-pi/AI-projects](https://github.com/rotem-pi/AI-projects))

`test_runs_base.csv` is **gitignored** (GitHub’s ~100MB blob limit). Everything else is small enough for a normal push.

```bash
cd /path/to/alert-quality-evaluation/incidents-test-analysis

git init
git add README.md run_impact_tests.sh conftest.py test_test_generation_8f5e1c0_impact.py \
  legacy_analytical_model_pre_8f5e1c0.py analytical_model_post_8f5e1c0_reference.py .gitignore
git commit -m "8f5e1c0 test generation impact bundle (CSV not in repo)"
git branch -M main
git remote add origin https://github.com/rotem-pi/AI-projects.git   # skip if origin exists
git push -u origin main
```

Use **SSH** if you prefer: `git remote add origin git@github.com:rotem-pi/AI-projects.git`

If the remote repo already has commits (e.g. a README added on GitHub), use `git pull origin main --allow-unrelated-histories` before the first push, or create an empty repo on GitHub and push.

Authenticate with **GitHub CLI** (`gh auth login`) or SSH keys / credential helper; the empty repo [rotem-pi/AI-projects](https://github.com/rotem-pi/AI-projects) is ready for a first push.
