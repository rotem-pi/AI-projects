# Test Generation Model - Seasonality & Guarded-Band Redesign

Research investigating seasonality in [definity-app](https://github.com/definity-ai/definity-app)'s automatic test generation, and validating a proposed replacement design (a calibrated "guarded band" test) against the current Prophet + Optuna generation pipeline.

This folder does **not** contain the production generator - it analyzes real production data (via the read replica) to compare the current approach against the proposed one, and produces the write-up and comparison report below.

Full write-up: **`findings.docx`**. Illustrative side-by-side comparison report: **`db/guarded_band_report.html`**.

## Layout

| Path | Role |
|---|---|
| `findings.docx` | Full write-up: motivation, seasonality findings, proposed architecture, results, deployment considerations |
| `make_findings_doc.js` | Generates `findings.docx` (Node + `docx` package) |
| `seasonality_analysis.py` | Day-of-week / day-of-month / month-of-year seasonality scan |
| `hour_effect_deep_dive.py` | Robustness check (block permutation) for the hour-of-day effect |
| `db/model_bakeoff.py` | Architecture comparison: trailing-mean/median baselines vs. Prophet vs. MSTL |
| `db/round2_representative.py`, `db/round3_recent_tiers.py` | Representative sampling, stratified by metric cadence tier |
| `db/stage2_*.py` | Existing-vs-suggested comparison stages: coverage, generation success, budget sweep, negative-bounds research |
| `db/stage3_guarded_band.py` | Guarded-band tuning: floor / rare-spike-exclusion / asymmetric-tolerance grid search |
| `db/stage4_final_eval.py` | Final band definition (`band_with_flag`) + full evaluation + HTML report generation |
| `db/stage5_conversion_matrices.py` | Sampled metric-level and run-level conversion matrices |
| `db/resilient_matrices.py`, `db/aggregate_resilient.py` | Checkpointed, retry-safe large-sample conversion matrix (survives replica connection drops) |
| `db/visual_side_by_side.py` | Shared DB helpers: connection string, SQL, current-production-test predictor |
| `db/guarded_band_report.html` | Consolidated comparison report (9 illustrative cases, with definity task links) |
| `db/bounds_model_lab.ipynb` | Interactive notebook to reproduce or challenge the band design |

## Setup

Scripts import from the definity-app backend (`app.brain.metric_tests`, `app.models`, `app.tests_gen`, etc.), so they need to run with that project's dependencies available:

```bash
cd db   # or repo root, for the top-level scripts
uv run --project /path/to/definity-app/backend --with psycopg2-binary python <script>.py
```

## Data

All CSVs are gitignored. They are raw pulls from the definity production read replica, and are reproduced by rerunning the scripts, not committed. The DB connection string is currently hardcoded near the top of each script (`prod-read-replica...`) - update it if you need to point at a different environment.

## Production implementation

Canonical code lives in **definity-app**:

- `backend/app/brain/anomaly/tests_generator.py` - current generation pipeline (Prophet + Optuna)
- `backend/app/tests_gen/` - the models, config, and offline research harness behind that pipeline
- `backend/app/brain/metric_tests/` - the four test types (`Const`, `Range`, `PctDiff`, `Trend`), evaluated at ingest time
- `backend/app/brain/tests.py` - ingest-time bounds resolution (generic across test types; unaffected by which test type produced the bounds)
