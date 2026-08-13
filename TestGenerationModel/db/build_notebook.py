"""Generate bounds_model_lab.ipynb - a self-contained lab for challenging the
bounds-producer benchmark: data, splits, models, calibration, and metrics."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {},
            "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": source.splitlines(keepends=True)}


CELLS = [
    md("""# Bounds-producer lab: data, training method, and challenges

Self-contained notebook to reproduce and challenge the alert-bounds benchmark.
No DB access needed: it runs on the saved extracts in this directory.

## Data provenance
- Source: prod read replica, `metrics` table, aggregated server-side.
- Sample: **active** series (reported in last 30 days) carrying an **enabled
  auto test**, stratified by cadence tier: HF (3+/day) 120, daily 530,
  weekly 480, sparse 70. Sampled deterministically by
  `md5(metric_id||asset_value)` order. No history-length filter at sampling
  time, so cold-start ineligibility shows up as a result.
- Grain: HF = day x hour averages (last 90d); others = daily medians (last 420d).
- Files: `{PREFIX}_frame_{tier}.csv` (values), `{PREFIX}_sample_{tier}.csv` (metadata).

## Two sampling rounds
- **round2**: tiers assigned by LIFETIME average cadence (total reports / span).
- **round3**: tiers assigned by RECENT cadence (reports in last 45 days / 45).
  Motivation: 42% of series change tier between the two definitions; 77% of
  lifetime-"weekly" series actually report daily now. `PREFIX` below selects
  which round the interactive cells run on; a comparison section at the end
  puts both side by side.

## Org prior
The org expects **< 3% true anomalies** in real data. There are no labels here,
so "FPR" below is really *breach rate on unlabeled data*. Interpretation rule:
breach rates <= 3% are consistent with a calibrated band; only series above
`CLEAR_MISCALIBRATION_RATE` (default 10%) are treated as clear miscalibration."""),

    code("""from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(".")          # directory holding {PREFIX}_frame_*.csv
PREFIX = "round3"             # "round2" = lifetime tiers, "round3" = recent tiers
TIERS = ["hf", "daily", "weekly", "sparse"]

# --- knobs to challenge ---------------------------------------------------
TRAILING_WINDOW = 3           # how many recent values the point forecast averages
MISS_BUDGETS = [0.01, 0.02, 0.05]   # allowed false-alert rate (conformal alpha)
WINDOWS = {"hf": (21, 14), "daily": (56, 28),   # (calibration_days, test_days)
           "weekly": (112, 56), "sparse": (168, 84)}
EXPECTED_ANOMALY_RATE = 0.03  # org's upper bound on true anomaly rate
CLEAR_MISCALIBRATION_RATE = 0.10  # breach rate above this = band is clearly wrong
INJECTED_CHANGES = [0.3, 0.5, 1.0, -0.5, -1.0]  # -1.0 = value drops to zero
MIN_CALIBRATION_POINTS, MIN_TEST_POINTS = 10, 5
SEED = 1337"""),

    code("""values_by_tier, series_info_by_tier = {}, {}
for tier in TIERS:
    values_df = pd.read_csv(DATA_DIR / f"{PREFIX}_frame_{tier}.csv",
                            parse_dates=["day"])
    values_df["asset_value"] = values_df["asset_value"].fillna("")
    values_by_tier[tier] = values_df
    series_info = pd.read_csv(DATA_DIR / f"{PREFIX}_sample_{tier}.csv")
    series_info["asset_value"] = series_info["asset_value"].fillna("")
    series_info_by_tier[tier] = series_info
    print(f"{tier}: {series_info.shape[0]} series sampled, "
          f"{values_df.shape[0]} value rows")"""),

    md("""## Split protocol (leakage rules)

Per series, strictly chronological: everything before the calibration window is
"burn-in" for the causal forecaster; the **calibration** window supplies the
typical-error quantile; the final **test** window supplies every reported number.

- The point forecast is causal: the prediction at time t uses only values
  observed before t.
- The band half-width comes from calibration-window errors only, never test.
- Injections perturb a test point only; bands never see the perturbed value
  (the trailing average uses past values, so a single injected point cannot
  influence its own band).
- No per-series tuning: every knob above is global."""),

    code("""def predict_trailing_average(values: np.ndarray,
                             window: int = TRAILING_WINDOW) -> np.ndarray:
    \"\"\"Causal point forecast: prediction at t = mean of the last `window`
    values strictly before t. First point has no prediction (NaN).\"\"\"
    predicted = np.full(len(values), np.nan)
    for t in range(1, len(values)):
        predicted[t] = np.mean(values[max(0, t - window):t])
    return predicted


def calibration_error_quantile(abs_errors: np.ndarray,
                               miss_budget: float) -> float:
    \"\"\"Conformal quantile: the error size that only `miss_budget` of
    calibration errors exceeded (with the standard +1 finite-sample bump).\"\"\"
    rank = int(np.ceil((len(abs_errors) + 1) * (1 - miss_budget))) - 1
    rank = min(len(abs_errors) - 1, rank)
    return float(np.sort(abs_errors)[rank])


def split_series_for_evaluation(series_df: pd.DataFrame, tier: str) -> dict | None:
    \"\"\"Sort one series, compute causal predictions, and mark which points
    belong to the calibration window and which to the test window.\"\"\"
    order_cols = ["day", "hr"] if tier == "hf" else ["day"]
    series_df = series_df.sort_values(order_cols)
    if tier == "hf":
        timestamps = pd.DatetimeIndex(
            series_df["day"] + pd.to_timedelta(series_df["hr"], unit="h"))
    else:
        timestamps = pd.DatetimeIndex(series_df["day"])
    values = series_df["val"].astype(float).values

    calibration_days, test_days = WINDOWS[tier]
    end = timestamps.max()
    test_mask = np.asarray(timestamps >= end - pd.Timedelta(days=test_days))
    calibration_mask = np.asarray(
        (timestamps >= end - pd.Timedelta(days=calibration_days + test_days))
        & ~test_mask)

    predicted = predict_trailing_average(values)
    calibration_points = calibration_mask & ~np.isnan(predicted)
    if test_mask.sum() < MIN_TEST_POINTS or calibration_points.sum() < MIN_CALIBRATION_POINTS:
        return None
    return {"timestamps": timestamps, "values": values, "predicted": predicted,
            "calibration_points": calibration_points, "test_mask": test_mask}


def compute_bounds(series: dict, miss_budget: float, relative: bool):
    \"\"\"Band around the prediction sized by calibration-window errors.
    relative=True sizes the error as a fraction of the predicted value.\"\"\"
    abs_error = np.abs(series["values"] - series["predicted"])
    prediction_size = np.maximum(np.abs(series["predicted"]), 1e-9)
    calibration_errors = (abs_error / prediction_size if relative else abs_error)
    typical_error = calibration_error_quantile(
        calibration_errors[series["calibration_points"]], miss_budget)
    half_width = typical_error * prediction_size if relative else typical_error
    lower_bound = series["predicted"] - half_width
    upper_bound = series["predicted"] + half_width
    return lower_bound, upper_bound"""),

    code("""# Reproduce the benchmark scoring (per-series abs/rel configs)
result_rows = []
prepared_series = {}
for tier in TIERS:
    for key, one_series_df in values_by_tier[tier].groupby(
            ["metric_id", "asset_value"]):
        series = split_series_for_evaluation(one_series_df, tier)
        if series is None:
            result_rows.append({"tier": tier, "metric_id": key[0],
                                "config": "INELIGIBLE"})
            continue
        prepared_series[key] = (tier, series)
        for miss_budget in MISS_BUDGETS:
            for relative in [False, True]:
                lower_bound, upper_bound = compute_bounds(
                    series, miss_budget, relative)
                on_test = series["test_mask"] & ~np.isnan(series["predicted"])
                actual = series["values"][on_test]
                low, high = lower_bound[on_test], upper_bound[on_test]
                row = {"tier": tier, "metric_id": key[0],
                       "config": (f"{'rel' if relative else 'abs'}"
                                  f"_a{int(miss_budget * 100)}"),
                       "breach_rate": np.mean((actual < low) | (actual > high))}
                for change in INJECTED_CHANGES:
                    injected = actual * (1 + change)
                    row[f"det_{change}"] = np.mean(
                        (injected < low) | (injected > high))
                result_rows.append(row)
results = pd.DataFrame(result_rows)
scored = results[results["config"] != "INELIGIBLE"]
for tier in TIERS:
    tier_scores = scored[scored["tier"] == tier]
    n_ineligible = ((results["tier"] == tier)
                    & (results["config"] == "INELIGIBLE")).sum()
    print(f"\\n===== {tier} (ineligible {n_ineligible}) =====")
    print(tier_scores.groupby("config").agg(
        n=("metric_id", "nunique"),
        breach_mean=("breach_rate", "mean"),
        clearly_miscalibrated=("breach_rate",
                               lambda x: (x > CLEAR_MISCALIBRATION_RATE).mean()),
        ambiguous=("breach_rate",
                   lambda x: ((x > EXPECTED_ANOMALY_RATE)
                              & (x <= CLEAR_MISCALIBRATION_RATE)).mean()),
        det50=("det_0.5", "mean"), det_drop=("det_-1.0", "mean"),
    ).round(3).to_string())"""),

    md("""## Breach-pattern diagnostic: miscalibration vs regime change

A band failure scatters breaches; a genuine regime change produces a
*consecutive run* of breaches right after the shift. Series whose breaches are
mostly one long run are cases where the model was arguably right and the
unlabeled "FPR" punished it."""),

    code("""def longest_consecutive_breach_run(breach_flags: np.ndarray) -> int:
    longest = current = 0
    for breached in breach_flags:
        current = current + 1 if breached else 0
        longest = max(longest, current)
    return longest

high_breach_rows = []
for key, (tier, series) in prepared_series.items():
    lower_bound, upper_bound = compute_bounds(series, 0.01, relative=True)
    on_test = series["test_mask"] & ~np.isnan(series["predicted"])
    breach_flags = ((series["values"] < lower_bound)
                    | (series["values"] > upper_bound))[on_test]
    if breach_flags.mean() > EXPECTED_ANOMALY_RATE:
        high_breach_rows.append({
            "tier": tier, "metric_id": key[0],
            "breach_rate": breach_flags.mean(),
            "consecutive_share": (longest_consecutive_breach_run(breach_flags)
                                  / max(breach_flags.sum(), 1))})
high_breach_series = pd.DataFrame(high_breach_rows)
print("high-breach series: consecutive_share ~1.0 -> regime change, "
      "low -> miscalibration")
print(high_breach_series.groupby("tier").agg(
    n=("metric_id", "size"),
    regime_like=("consecutive_share", lambda x: (x >= 0.8).mean()),
    scattered=("consecutive_share", lambda x: (x < 0.5).mean()),
).round(2).to_string())"""),

    md("""## Series explorer

Plot any series with its band; breaches marked red. Use it to eyeball whether
a "storm" is a real shift or a band failure."""),

    code("""def plot_series_with_bounds(metric_id: int, miss_budget: float = 0.01,
                            relative: bool = True):
    key = next(k for k in prepared_series if k[0] == metric_id)
    tier, series = prepared_series[key]
    lower_bound, upper_bound = compute_bounds(series, miss_budget, relative)
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(series["timestamps"], series["values"], ".", ms=3, label="value")
    ax.plot(series["timestamps"], lower_bound, lw=0.8, color="gray")
    ax.plot(series["timestamps"], upper_bound, lw=0.8, color="gray", label="band")
    breached = ((series["values"] < lower_bound)
                | (series["values"] > upper_bound))
    in_test = breached & series["test_mask"]
    ax.plot(series["timestamps"][in_test], series["values"][in_test], "rx",
            label="test breach")
    test_start = series["timestamps"][np.argmax(series["test_mask"])]
    ax.axvline(test_start, color="orange", ls="--", label="test start")
    ax.set_title(f"metric {metric_id} ({tier}), "
                 f"miss_budget={miss_budget}, relative={relative}")
    ax.legend()
    plt.show()

if len(high_breach_series):
    worst = high_breach_series.sort_values("breach_rate").iloc[-1]
    plot_series_with_bounds(int(worst["metric_id"]))"""),

    md("""## Challenge checklist

Things this notebook makes one-line-editable, and what to look for:

1. **`TRAILING_WINDOW`**: try 1, 5, 10. Larger = smoother baseline, slower to
   adapt after regime changes. Round 1 found 3 near-optimal; verify.
2. **`MISS_BUDGETS` / `WINDOWS`**: does a longer calibration window fix the
   daily tier, or is the tier just young?
3. **`CLEAR_MISCALIBRATION_RATE` / `EXPECTED_ANOMALY_RATE`**: the storm story
   is threshold-sensitive; move them and watch the deployability conclusion.
4. **Trimmed calibration**: drop the top 3% of calibration errors (assume they
   are anomalies) before the quantile - tighter bands, higher detection, but
   check the breach cost.
5. **Injection realism**: negative changes are included; add drifts
   (multiply a whole tail window) to challenge the single-point assumption.
6. **Survivorship**: this sample requires an existing auto test. Re-run
   sampling without that join (see round2_representative.py) to include
   metrics the generator failed on.
7. **Tenant concentration**: if a few apps dominate, slice the summary per
   app (next cell)."""),

    code("""# quick tenant-concentration check
for tier in TIERS:
    app_share = series_info_by_tier[tier]["app_id"].value_counts(normalize=True)
    print(f"{tier}: top app {app_share.iloc[0]:.0%}, "
          f"top3 {app_share.iloc[:3].sum():.0%} of sampled series")"""),

    md("""## Round 2 vs round 3: lifetime tiers vs recent-cadence tiers

Same windows, same configs, same scoring; only the tier assignment (and hence
the sampled population per tier) differs. Shown for the per-series relative
band at miss budget 1% (the recommended config)."""),

    code("""comparison_tables = []
for round_name in ["round2", "round3"]:
    results_path = DATA_DIR / f"{round_name}_results.csv"
    if not results_path.exists():
        print(f"{results_path} missing - run the {round_name} script first")
        continue
    round_results = pd.read_csv(results_path)
    ineligible_per_tier = (round_results[round_results["config"] == "INELIGIBLE"]
                           .groupby("tier").size())
    series_per_tier = round_results.groupby("tier")["metric_id"].nunique()
    chosen_config = round_results[round_results["config"] == "rel_a1"]
    by_tier = chosen_config.groupby("tier").agg(
        n=("metric_id", "nunique"),
        breach_mean=("fpr", "mean"),
        storm5=("fpr", lambda x: (x > 0.05).mean()),
        storm10=("fpr", lambda x: (x > 0.10).mean()),
        det50=("det_0.5", "mean"))
    by_tier["ineligible_pct"] = (ineligible_per_tier
                                 / series_per_tier).reindex(by_tier.index).fillna(0)
    by_tier.insert(0, "round", round_name)
    comparison_tables.append(by_tier)
if comparison_tables:
    print(pd.concat(comparison_tables).round(3).sort_index(kind="stable")
          .to_string())"""),
]

nb = {"cells": CELLS, "metadata": {"kernelspec": {
    "display_name": "Python 3", "language": "python", "name": "python3"}},
    "nbformat": 4, "nbformat_minor": 5}

out = HERE / "bounds_model_lab.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out}")
