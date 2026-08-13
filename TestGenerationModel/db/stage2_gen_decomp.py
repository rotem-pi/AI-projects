"""Stage 2: run the REAL prod generator (prophet path) offline on 300
eligible-but-untested series to decompose why they have no test.

Eligible-untested = registry metric type, >= 14 points, > 7 days span, active,
enabled, non-temporary, auto-tests tenant, and no enabled auto test today
(1,955 series in the full population; deterministic md5 sample of 300).
"""

import logging
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))

from app.brain.anomaly.tests_generator import TestsGenerator
from app.models.metric_history_model import MetricHistory
from app.tests_gen.metrics.metrics_registry import MetricsRegistry

HERE = Path(__file__).resolve().parent
DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")
SAMPLE_N = 300
HISTORY_LIMIT = 2000
START = time.time()

SAMPLE_SQL = """
WITH auto_apps AS (
  SELECT app_id FROM tenant_settings
  LEFT JOIN envs USING (tenant_id) INNER JOIN apps USING (env_id)
  WHERE auto_tests_enabled
)
SELECT ma.metric_id, ma.asset_value, mc.metric_type
FROM metrics_agg ma
JOIN metrics_conf mc ON mc.metric_id = ma.metric_id
JOIN auto_apps a ON a.app_id = mc.app_id
LEFT JOIN tests t ON t.metric_id = ma.metric_id AND t.asset_value = ma.asset_value
     AND t.is_enabled AND t.is_auto
WHERE ma.max_app_pit >= now() - interval '30 days'
  AND ma.total_metrics >= 14
  AND ma.max_app_pit - ma.min_app_pit > interval '7 days'
  AND mc.is_metric_enabled AND NOT mc.is_temporary
  AND t.test_id IS NULL
  AND (LOWER(mc.metric_type) = ANY(%(reg)s) OR mc.metric_type LIKE 'custom.%%')
ORDER BY md5(ma.metric_id::text || ma.asset_value)
LIMIT %(n)s
"""

VALUES_SQL = """
SELECT app_pit, metric_value FROM (
  SELECT m.app_pit, m.metric_value,
         ROW_NUMBER() OVER (PARTITION BY m.app_pit ORDER BY m.end_time DESC) rn
  FROM metrics m
  WHERE m.metric_id = %(mid)s AND m.asset_value = %(av)s
    AND m.metric_value IS NOT NULL
) x WHERE rn = 1 ORDER BY app_pit DESC LIMIT %(lim)s
"""


class LogCapture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def classify_failure(messages: list[str]) -> str:
    text = " | ".join(messages).lower()
    if "error while generating test" in text:
        return "exception"
    if "couldn't find" in text and "configuration" in text:
        return "unknown_metric_type"
    if "too noisy" in text or "suspected" in text:
        return "rejected_too_noisy"
    if "cannot process" in text:
        return "labeler_refused_type"
    return "no_test_found_other"


def main() -> None:
    registry_types = list(MetricsRegistry.get_instance().registry.keys())
    conn = psycopg2.connect(DB)
    conn.set_session(readonly=True, autocommit=True)
    cur = conn.cursor()
    cur.execute("SET statement_timeout = 300000")
    cur.execute(SAMPLE_SQL, {"reg": registry_types, "n": SAMPLE_N})
    sample = cur.fetchall()
    print(f"sampled {len(sample)} eligible-untested series", flush=True)

    capture = LogCapture()
    logging.getLogger().addHandler(capture)
    logging.getLogger().setLevel(logging.INFO)

    rows = []
    for i, (mid, av, mtype) in enumerate(sample):
        cur.execute(VALUES_SQL, {"mid": mid, "av": av, "lim": HISTORY_LIMIT})
        vals = cur.fetchall()[::-1]
        pits = [v[0] for v in vals]
        values = [float(v[1]) for v in vals]
        metric_dict = {"metric_id": mid, "metric_type": mtype,
                       "metric_values": list(values), "app_pits": list(pits)}
        history = MetricHistory(metric_id=mid, metric_type=mtype,
                                metric_values=values, app_pits=pits)
        capture.messages.clear()
        t0 = time.time()
        result = TestsGenerator._generate_test(
            metric_dict, False, history, "prophet")
        rows.append({
            "metric_id": mid, "metric_type": mtype, "n_points": len(values),
            "seconds": round(time.time() - t0, 2),
            "success": result is not None,
            "test_type": result["test"].test_type if result else None,
            "score": (result.get("score") or
                      result.get("scores", {}).get("score")) if result else None,
            "failure_reason": None if result else classify_failure(
                capture.messages),
            "last_log": capture.messages[-1][:200] if capture.messages else "",
        })
        if i % 25 == 0:
            print(f"{i}/{len(sample)} ({time.time() - START:.0f}s)", flush=True)
    conn.close()

    df = pd.DataFrame(rows)
    df.to_csv(HERE / "stage2_gen_decomp.csv", index=False)
    print(f"\nsuccess rate: {df['success'].mean():.1%} ({df['success'].sum()}"
          f"/{len(df)})")
    print("\nfailure reasons:")
    print(df[~df["success"]]["failure_reason"].value_counts().to_string())
    print("\nsuccess by test_type:")
    print(df[df["success"]]["test_type"].value_counts().to_string())
    print(f"\nmedian generation seconds: {df['seconds'].median():.1f}")


if __name__ == "__main__":
    main()
