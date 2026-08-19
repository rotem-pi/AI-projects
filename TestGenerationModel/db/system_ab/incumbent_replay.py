"""Replay of the EXISTING (pre-guarded-band) test generation pipeline on the
same population as the GuardedBand shadow replay, scored on the same
guarded_band_qualified_samples.csv points.

Protocol, mirroring production auto-test generation:
- one generation per series (production generates when a metric has no test),
  at the first Monday boundary where the series has >= test_gen_min_count
  (14) points and > test_gen_min_days (7) span, exactly like
  test-candidates.sql's HAVING clause
- history for generation: the last <= 2000 points up to that boundary
  (GET_METRIC_REPORTS limit)
- model: the production default, test_model_type='prophet'
  (prophet_labeler_grid_search), called via TestsGenerator._generate_test -
  the same entry point production uses
- generation may legitimately return None (no candidate passes the score
  policy): in production such series simply have no auto test, so they are
  evaluated here with zero flags
- forward evaluation: every point strictly after the generation boundary is
  checked against the generated test's bounds, with the full prior history
  as prediction context (same as the live evaluation path)

Outputs: incumbent_replay_points.csv (all evaluated points),
incumbent_replay_qualified_points.csv (restricted to the qualified samples,
with label + test_type + var1/2/3), incumbent_replay_tests.csv (one row per
series: generated test or none), incumbent_replay_summary.json.

Run from backend/:  PYTHONPATH=. uv run python ../analysis_temp/guarded-band/incumbent_replay.py
"""

import json
import logging
import time
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from app.brain.anomaly.tests_generator import TestsGenerator  # noqa: E402
from app.models.metric_history_model import MetricHistory  # noqa: E402

HERE = Path(__file__).resolve().parent
MIN_COUNT = 14  # system_settings.test_gen_min_count default
MIN_DAYS = 7  # system_settings.test_gen_min_days default
HISTORY_CAP = 2000  # GET_METRIC_REPORTS limit in test-candidates.sql
MODEL_TYPE = "prophet"  # system_settings.test_model_type default


def week_start(ts: pd.Timestamp) -> pd.Timestamp:
    return (ts - pd.Timedelta(days=ts.weekday())).normalize()


def first_generation_boundary(g: pd.DataFrame) -> pd.Timestamp | None:
    """First Monday 00:00 at which the production HAVING clause is met."""
    pits = g["pit"]
    first = pits.iloc[0]
    as_of = week_start(first) + pd.Timedelta(days=7)
    last = pits.iloc[-1]
    while as_of <= last:
        hist = pits[pits <= as_of]
        if len(hist) >= MIN_COUNT and (hist.iloc[-1] - hist.iloc[0]) > pd.Timedelta(
            days=MIN_DAYS
        ):
            return as_of
        as_of += pd.Timedelta(days=7)
    return None


def replay_series(g: pd.DataFrame) -> tuple[list[dict], dict]:
    g = g.sort_values("pit").reset_index(drop=True)
    boundary = first_generation_boundary(g)
    meta = {
        "dataset": g["dataset"].iloc[0],
        "metric_id": int(g["metric_id"].iloc[0]),
        "metric_type": g["metric_type"].iloc[0],
        "generated_at": boundary,
        "test_type": None,
        "var1": None,
        "var2": None,
        "var3": None,
        "gen_seconds": 0.0,
    }
    if boundary is None:
        return [], meta

    hist = g[g["pit"] <= boundary].tail(HISTORY_CAP)
    cand = {
        "metric_id": meta["metric_id"],
        "metric_type": meta["metric_type"],
        "metric_values": hist["value"].astype(float).tolist(),
        "app_pits": [p.to_pydatetime() for p in hist["pit"]],
    }
    mh = MetricHistory(**cand)
    t0 = time.monotonic()
    generated = TestsGenerator._generate_test(dict(cand), False, mh, MODEL_TYPE)
    meta["gen_seconds"] = round(time.monotonic() - t0, 3)

    test = generated.get("test") if generated else None
    if test is not None:
        meta["test_type"] = test.test_type
        meta["var1"] = getattr(test, "var1", None)
        meta["var2"] = getattr(test, "var2", None)
        meta["var3"] = getattr(test, "var3", None)

    eval_df = g[g["pit"] > boundary]
    if not len(eval_df):
        return [], meta

    if test is not None:
        full_mh = MetricHistory(
            metric_id=meta["metric_id"],
            metric_type=meta["metric_type"],
            metric_values=g["value"].astype(float).tolist(),
            app_pits=[p.to_pydatetime() for p in g["pit"]],
        )
        predictions = test.predict_all_pits(full_mh)
    else:
        predictions = [None] * len(g)

    point_rows = []
    for j in eval_df.index:
        pred = predictions[j]
        flagged = (
            not pred.is_passed(float(g["value"].iloc[j])) if pred is not None else False
        )
        point_rows.append(
            {
                "dataset": meta["dataset"],
                "metric_id": meta["metric_id"],
                "pit": g["pit"].iloc[j],
                "value": g["value"].iloc[j],
                "is_anomaly": bool(g["is_anomaly"].iloc[j]),
                "test_type": meta["test_type"] or "none",
                "var1": meta["var1"],
                "var2": meta["var2"],
                "var3": meta["var3"],
                "predicted": pred.predicted_value if pred else None,
                "lower": pred.lower_bound if pred else None,
                "upper": pred.upper_bound if pred else None,
                "flagged": flagged,
            }
        )
    return point_rows, meta


def main() -> None:
    t_start = time.monotonic()
    df = pd.read_parquet(HERE / "all_labeled_points.parquet")
    df = df[df["is_anomaly"].notna()].copy()
    df["is_anomaly"] = df["is_anomaly"].astype(bool)
    df = df[~df.duplicated(subset=["metric_id", "pit"], keep="first")]

    points_all: list[dict] = []
    tests_all: list[dict] = []
    groups = list(df.groupby(["dataset", "metric_id"]))
    for i, (_, g) in enumerate(groups):
        p, meta = replay_series(g)
        points_all.extend(p)
        tests_all.append(meta)
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(groups)} series, {time.monotonic() - t_start:.0f}s")

    points = pd.DataFrame(points_all)
    tests = pd.DataFrame(tests_all)
    points.to_csv(HERE / "incumbent_replay_points.csv", index=False)
    tests.to_csv(HERE / "incumbent_replay_tests.csv", index=False)
    total_seconds = time.monotonic() - t_start

    qualified = pd.read_csv(
        HERE / "guarded_band_qualified_samples.csv", parse_dates=["pit"]
    )
    qkeys = set(zip(qualified["metric_id"], qualified["pit"]))
    points["qualified"] = [
        (m, p) in qkeys for m, p in zip(points["metric_id"], points["pit"])
    ]
    qpts = points[points["qualified"]].drop(columns=["qualified"])
    qpts.to_csv(HERE / "incumbent_replay_qualified_points.csv", index=False)

    def score(sub: pd.DataFrame) -> dict:
        anom, norm = sub[sub["is_anomaly"]], sub[~sub["is_anomaly"]]
        return {
            "points": len(sub),
            "anomalies": len(anom),
            "detected": int(anom["flagged"].sum()),
            "detection_pct": round(100 * anom["flagged"].mean(), 1) if len(anom) else None,
            "normals": len(norm),
            "false_alarms": int(norm["flagged"].sum()),
            "false_alarm_pct": round(100 * norm["flagged"].mean(), 2) if len(norm) else None,
        }

    with_test = qpts[qpts["test_type"] != "none"]
    summary = {
        "timing": {
            "replay_seconds_total": round(total_seconds, 1),
            "generation_seconds_total": round(float(tests["gen_seconds"].sum()), 1),
            "seconds_per_generation": round(
                float(tests.loc[tests["generated_at"].notna(), "gen_seconds"].mean()), 2
            ),
            "ms_per_evaluated_point": round(1000 * total_seconds / len(points), 3),
        },
        "series_total": len(tests),
        "series_generation_attempted": int(tests["generated_at"].notna().sum()),
        "tests_generated_by_type": tests["test_type"].fillna("none").value_counts().to_dict(),
        "qualified_scored": score(qpts),
        "qualified_scored_only_series_with_test": score(with_test),
        "by_dataset_qualified": {ds: score(sub) for ds, sub in qpts.groupby("dataset")},
    }
    (HERE / "incumbent_replay_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
