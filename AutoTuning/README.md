# Spark Auto-Tuning — Research & Validation

Offline research, validation notebooks, and design docs for the Spark/Databricks **auto-tuning policy** implemented in [definity-app](https://github.com/definity-ai/definity-app) at `app/brain/insights/tuning/`.

This folder does **not** contain the production tuning engine — it mirrors and tests it against real task-run data.

## Layout

| Path | Role |
|------|------|
| `docs/tuning_hld.md` | Full high-level design (production policy) |
| `docs/tuning_hld_brief.md` | Condensed HLD |
| `scripts/add_recommendations.py` | Offline mirror of `extract_recommendations()` |
| `scripts/suggested_run_matching.py` | Nearest-neighbor duration estimation for config suggestions |
| `scripts/gradual_tuning_experiment.py` | Multi-knob partial-step simulation experiment |
| `scripts/paths.py` | Resolve repo root and `data/` paths |
| `notebooks/data_exploration.ipynb` | **Fetch** insights + task-run data → `data/insights_with_recommendations_data.csv` |
| `notebooks/tuning_framework_test.ipynb` | End-to-end validation of tuning framework on real data |
| `notebooks/historical_pov_exploration.ipynb` | Case studies (Impression-Pre-Process, ENGAGEMENT) |
| `experiments/xgboost/xgboost_exploration.ipynb` | XGBoost duration modeling experiment |
| `data/case_studies/` | Small committed CSV snapshots for case-study notebooks |
| `data/` (gitignored) | Large exports reproduced from notebook SQL |

## Setup

```bash
cd AutoTuning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set DATABASE_URL locally; never commit .env
```

For notebooks that import the production tuning framework (`tuning_framework_test`, `historical_pov`):

```bash
export DEFINITY_BACKEND_ROOT=/path/to/definity-app/backend
# Use the backend venv if you need exact dependency parity with production tests.
```

## Data workflow

Large CSVs are **not** in git. Reproduce them from notebook SQL:

1. **`notebooks/data_exploration.ipynb`** — fetches task runs joined with insights → `data/insights_with_recommendations_data.csv` (requires `DATABASE_URL`).
2. **`scripts/add_recommendations.py`** — enriches fetch output → `data/insights_with_recommendations_output.csv`.
3. **`notebooks/tuning_framework_test.ipynb`** — consumes the output CSV; needs `DEFINITY_BACKEND_ROOT`.
4. **`experiments/xgboost/xgboost_exploration.ipynb`** — fetches task-run cohort → `data/xgboost_tasks_data.csv` (requires `DATABASE_URL`).
5. **`notebooks/historical_pov_exploration.ipynb`** — uses committed `data/case_studies/*.csv`; includes refetch SQL for both case studies (commented; requires `DATABASE_URL`).

```bash
cd scripts
python add_recommendations.py
# Or: python add_recommendations.py ../data/insights_with_recommendations_data.csv ../data/insights_with_recommendations_output.csv
```

## Environment variables

| Variable | Required for | Default |
|----------|--------------|---------|
| `DATABASE_URL` | Data-fetch notebooks | — (set in `.env` locally) |
| `DEFINITY_BACKEND_ROOT` | Framework validation notebooks | Auto-detect `../../definity-app/backend` from AutoTuning root |

## Production implementation

Canonical code and unit tests live in **definity-app**:

- `app/brain/insights/tuning/` — `build_plan`, `update_plan`, blocking rules, autonomous loop
- Design documented in `docs/tuning_hld.md` (kept in sync with production reviews)
