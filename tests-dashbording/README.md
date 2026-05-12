# tests-dashbording

Dash dashboard over exported Definity test runs (`test_runs_base.csv`).

## Data

The CSV is **not** in git (large export; ignored at repo root). Either:

- Run `python3 refresh_data.py` after `export DATABASE_URL='postgresql+psycopg2://…'`, or  
- Copy `test_runs_base.csv` into this folder next to `dashboard.py`.

## Run

```bash
pip install dash dash-bootstrap-components plotly pandas scipy numpy sqlalchemy psycopg2-binary
python3 dashboard.py
```

Open the URL printed in the terminal (Dash defaults to port 8050 unless configured otherwise).
