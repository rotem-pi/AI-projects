"""
Fetches the latest test run data from PostgreSQL and overwrites test_runs_base.csv.

Run manually:
    python3 refresh_data.py

Schedule via cron (every Sunday at 3 AM):
    crontab -e
    0 3 * * 0 /usr/bin/python3 /path/to/tests-dashbording/refresh_data.py >> /path/to/refresh.log 2>&1
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text as sa_text


def _notify(title: str, message: str):
    """Send a macOS desktop notification (silent no-op on non-Mac)."""
    try:
        script = (
            f'display notification "{message}" '
            f'with title "{title}" '
            f'sound name "default"'
        )
        subprocess.run(["osascript", "-e", script], check=False, timeout=5)
    except Exception:
        pass

# ── Config ────────────────────────────────────────────────────────────────────
PG_URL = os.environ.get("DATABASE_URL")
if not PG_URL:
    sys.exit(
        "DATABASE_URL is not set. Example:\n"
        "  export DATABASE_URL='postgresql+psycopg2://USER:PASS@HOST:5432/DBNAME'\n"
    )

# How far back to pull data. Change to e.g. '2 months' or '7 days' as needed.
LOOKBACK = os.environ.get("LOOKBACK_INTERVAL", "2 months")

CSV_PATH = Path(__file__).parent / "test_runs_base.csv"
CSV_TMP  = CSV_PATH.with_suffix(".csv.tmp")

QUERY = f"""
SELECT
    r.test_run_id,
    r.test_id,
    r.test_id        AS test_def_id,
    r.created_time,
    r.run_value,
    r.is_passed,
    r.lower_bound,
    r.upper_bound,
    r.app_pit,
    r.task_id,
    r.app_id,
    CASE WHEN r.is_passed THEN 1 ELSE 0 END AS is_passed_binary,
    t.test_type,
    t.var1,
    t.var2,
    t.var3,
    mc.asset_name,
    mc.metric_type,
    mc.asset_type,
    mc.task_name,
    env.tenant_id,
    env.env_id
FROM test_runs r
JOIN tests        t   ON r.test_id   = t.test_id
JOIN metrics_conf mc  ON t.metric_id  = mc.metric_id
JOIN apps             ON t.app_id    = apps.app_id
JOIN envs         env ON apps.env_id  = env.env_id
WHERE r.created_time >= NOW() - INTERVAL '{LOOKBACK}'
"""


def main():
    print(f"[refresh_data] Connecting to DB…")
    engine = create_engine(PG_URL, pool_pre_ping=True)

    t0 = time.time()
    print(f"[refresh_data] Running query (lookback: {LOOKBACK})…")
    try:
        with engine.connect() as conn:
            df = pd.read_sql(sa_text(QUERY), conn)
    except Exception as exc:
        msg = f"Query failed: {exc}"
        print(f"[refresh_data] ERROR: {msg}", file=sys.stderr)
        _notify("Dashboard refresh FAILED", msg)
        sys.exit(1)

    elapsed = time.time() - t0
    print(f"[refresh_data] Fetched {len(df):,} rows in {elapsed:.1f}s")

    # Write to a temp file first, then atomically replace the real CSV
    # so the dashboard never reads a half-written file.
    df.to_csv(CSV_TMP, index=False)
    CSV_TMP.replace(CSV_PATH)

    print(f"[refresh_data] Saved to {CSV_PATH}")
    _restart_dashboard()
    _notify(
        "Dashboard data refreshed",
        f"{len(df):,} rows loaded in {elapsed:.0f}s. Dashboard restarted.",
    )


DASHBOARD_SCRIPT = Path(__file__).parent / "dashboard.py"
DASHBOARD_LOG    = Path(__file__).parent / "dashboard.log"

def _restart_dashboard():
    print("[refresh_data] Restarting dashboard…")
    # Kill any running instance
    subprocess.run(["pkill", "-f", f"python3 {DASHBOARD_SCRIPT}"], check=False)
    time.sleep(2)
    # Start a new detached instance, appending to the dashboard log
    with open(DASHBOARD_LOG, "a") as log:
        subprocess.Popen(
            [sys.executable, str(DASHBOARD_SCRIPT)],
            stdout=log,
            stderr=log,
            start_new_session=True,
        )
    print(f"[refresh_data] Dashboard restarted — logs at {DASHBOARD_LOG}")


if __name__ == "__main__":
    main()
