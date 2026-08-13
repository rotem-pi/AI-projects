"""Shared DB helpers for the guarded-band evaluation scripts: connection
string, the raw-values/tests SQL, the definity-task link lookup, and the
current-production-test predictor.

This module is a pure library, not a report generator. The single
consolidated comparison report (existing test vs. guarded band, all cases)
is built by stage4_final_eval.py -> guarded_band_report.html. Keeping the
report-building logic in exactly one place is what this consolidation was
for; do not add a second report generator here.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

DB = ("postgresql://postgres:postgres@prod-read-replica.coe3zosbcs5l"
     ".eu-north-1.rds.amazonaws.com:5432/app")
EVAL_DAYS = 30

VALUES_SQL = """
SELECT metric_id, asset_value, app_pit, metric_value
FROM (
  SELECT m.metric_id, m.asset_value, m.app_pit, m.metric_value,
         ROW_NUMBER() OVER (PARTITION BY m.metric_id, m.asset_value, m.app_pit
                            ORDER BY m.end_time DESC) AS rn
  FROM metrics m
  JOIN (SELECT UNNEST(%(mids)s::int[]) AS metric_id,
               UNNEST(%(avs)s::text[]) AS asset_value) u
    USING (metric_id, asset_value)
  WHERE m.app_pit >= now() - interval '180 days' AND m.metric_value IS NOT NULL
) x WHERE rn = 1
"""

TESTS_SQL = """
SELECT t.metric_id, t.asset_value, t.test_type, t.var1, t.var2, t.var3,
       mc.metric_type
FROM tests t JOIN metrics_conf mc ON mc.metric_id = t.metric_id
JOIN (SELECT UNNEST(%(mids)s::int[]) m, UNNEST(%(avs)s::text[]) a) u
  ON u.m = t.metric_id AND u.a = t.asset_value
WHERE t.is_auto AND t.is_enabled
"""

LINK_SQL = """
SELECT x.task_id, e.tenant_id
FROM metrics_conf mc
JOIN apps a ON a.app_id = mc.app_id
JOIN envs e ON e.env_id = a.env_id
LEFT JOIN LATERAL (
  SELECT m.task_id FROM metrics m
  WHERE m.metric_id = mc.metric_id AND m.asset_value = %(av)s
    AND m.task_id IS NOT NULL
  ORDER BY m.app_pit DESC LIMIT 1
) x ON TRUE
WHERE mc.metric_id = %(mid)s
"""


def incumbent_bounds(t_row: pd.Series, pits: list, values: np.ndarray):
    """Replay the metric's CURRENT production test (whatever test_type/
    var1-3 it has today) over its own history, returning (lower, upper)
    per point - the "existing method" side of every comparison."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "backend"))
    from app.brain.metric_tests.test_object_factory import create_test_instance
    from app.models.metric_history_model import MetricHistory

    mh = MetricHistory(metric_id=int(t_row["metric_id"]),
                       metric_type=t_row["metric_type"],
                       metric_values=[float(v) for v in values], app_pits=pits)
    test = create_test_instance({"test_type": t_row["test_type"],
                                 "var1": t_row["var1"], "var2": t_row["var2"],
                                 "var3": t_row["var3"]})
    preds = test.predict_all_pits(mh)
    lower = np.array([p.lower_bound if p and p.lower_bound is not None
                      else np.nan for p in preds], dtype=float)
    upper = np.array([p.upper_bound if p and p.upper_bound is not None
                      else np.nan for p in preds], dtype=float)
    return lower, upper
