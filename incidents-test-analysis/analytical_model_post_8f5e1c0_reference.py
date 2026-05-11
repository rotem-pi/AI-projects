import dataclasses
import logging

import numpy as np
import pandas as pd

from app.brain.metric_tests.base_test import BaseTest
from app.brain.metric_tests.const_test import ConstTest
from app.brain.metric_tests.pct_diff_test import PctDiffTest
from app.brain.metric_tests.range_test import RangeTest
from app.brain.metric_tests.trend_test import TrendTest
from app.metrics.metrics import EPS
from app.tests_gen.datasets.data_utils import (
    TS_COL,
    VALUE_COL,
    get_metric_history,
    timestamp_col_to_int,
)
from app.tests_gen.models.analytic_utils import (
    is_trend_series,
    remove_outliers,
    fit_trend,
    percent_variability,
)
from app.tests_gen.models.metric_test_model import (
    TestTimeSeriesModel,
)

from dataclasses import asdict

from app.brain.anomaly.utils import score_candidates
from app.models.metric_history_model import MetricHistory
from app.utils.significant_digits import SignificantDigits

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class AnalyticHyperParams:
    anomaly_upper_bound_p: float = 95.0
    anomaly_lower_bound_p: float = 5.0
    base_iqr_factor: float = 0.5
    pct_iqr_factor: float = 1.5
    max_window_length: int = 25
    max_window_fraction: float = 0.2
    max_pct_value: float = 25
    min_pct_value: float = 5
    max_train_anomalies_fraction = 0.15
    max_train_anomalies_count: int = 10
    low_variability_threshold: float = 0.1
    scoring_threshold: int = 50
    recent_fit_window: int = 300


default_hyper_params = asdict(AnalyticHyperParams())


class AnalyticTestTimeSeriesModel(TestTimeSeriesModel):
    def __init__(
        self,
        metric_id,
        model_id,
        metric_history=None,
        model: BaseTest = None,
        regime=None,
        optimize_metrics=["score"],
        fallback_to_baseline=False,
        hyper_params=default_hyper_params,
    ):
        super().__init__(
            metric_id=metric_id,
            model_id=model_id,
            model=model,
            regime=regime,
            optimize_metrics=optimize_metrics,
        )
        self.score_type = optimize_metrics[0]
        self.fallback_to_baseline = fallback_to_baseline
        self.hyper_params = AnalyticHyperParams(**hyper_params)
        self.metric_history = metric_history
        logger.debug(
            f"Metric {self.metric_id} analytical test generation hyper parameters: {self.hyper_params}"
        )

    def train(self, metrics, labels):
        if not self.metric_history:
            self.metric_history = get_metric_history(metrics=metrics, labels=labels)

        data = metrics.reset_index()
        fit_data = self._get_recent_fit_data(data)
        clean_recent, anomalies, _ = remove_outliers(
            fit_data,
            self.hyper_params.anomaly_lower_bound_p,
            self.hyper_params.anomaly_upper_bound_p,
            self.hyper_params.base_iqr_factor,
        )
        clean = self._protect_current_week_points(clean_recent, fit_data)
        anomaly_ratio = len(anomalies) / max(1, len(fit_data))
        logger.debug(
            f"Metric {self.metric_id} training: filtered {anomaly_ratio} of fit points as anomalous, fit_points={len(fit_data)}, clean_points={len(clean)}"
        )
        cleaned_candidates = self._build_candidates(clean=clean, data_for_pct=fit_data)
        raw_candidates = self._build_candidates(clean=fit_data, data_for_pct=fit_data)
        candidates = cleaned_candidates + raw_candidates

        scored_candidates = score_candidates(
            candidates, self.metric_history, limit=self.hyper_params.scoring_threshold
        )
        logger.debug(f"Metric {self.metric_id} candidates:")
        for candidate in scored_candidates:
            logger.debug(f"\t{candidate.get('test')}: {candidate.get('scores')}")
        fit_results = self.select_final_test(self.metric_history, scored_candidates)
        if fit_results:
            logger.debug(
                f"Metric {self.metric_id} selected model:{fit_results.get('test')}, scores: {fit_results.get('scores')}"
            )
        if not fit_results and self.fallback_to_baseline:
            fit_results = super().train(metrics=metrics, labels=labels)
        return fit_results

    def _get_recent_fit_data(self, data):
        if (
            self.hyper_params.recent_fit_window
            and self.hyper_params.recent_fit_window > 0
        ):
            return data.iloc[-self.hyper_params.recent_fit_window :].copy()
        return data.copy()

    def _protect_current_week_points(self, clean_data, fit_data):
        if fit_data.empty or TS_COL not in fit_data.columns:
            return clean_data

        latest_pit = fit_data[TS_COL].max()
        if pd.isna(latest_pit):
            return clean_data

        week_start = latest_pit.normalize() - pd.to_timedelta(
            latest_pit.weekday(), unit="D"
        )
        protected_tail = fit_data[fit_data[TS_COL] >= week_start]
        if protected_tail.empty:
            return clean_data

        combined = pd.concat([clean_data, protected_tail], ignore_index=True)
        combined = combined.drop_duplicates(subset=[TS_COL], keep="last")
        return combined.sort_values(by=TS_COL).reset_index(drop=True)

    def _build_candidates(self, clean, data_for_pct):
        trend_test = self.get_trend(clean)
        pct_test = self.get_pct(data_for_pct)
        range_test = self.get_range(clean)
        const_test = self.get_const(clean)
        return [
            test
            for test in [const_test, trend_test, range_test, pct_test]
            if test is not None
        ]

    def select_final_test(self, metric_history, scored_candidates):
        if not scored_candidates:
            return None
        final_tests = self.filter_tests_by_policy(
            scored_candidates, metric_history, self.score_type
        )
        if len(final_tests) == 0:
            return None
        if len(final_tests) == 1:  # do we need this if?
            return final_tests[0]
        else:
            index_max_score = max(
                range(len(final_tests)), key=lambda i: final_tests[i][self.score_type]
            )
            return final_tests[index_max_score]

    def filter_tests_by_policy(
        self, candidates, metric_history: MetricHistory, score_type
    ):
        final_tests = []
        known_group_types = set(["volume", "time", "distribution", "schema"])
        scoring_points = (
            min(len(metric_history.app_pits), self.hyper_params.scoring_threshold)
            if self.hyper_params.scoring_threshold
            else len(metric_history.app_pits)
        )
        for tc in candidates:
            metric_group_type = metric_history.metric_group_type
            metric_type = metric_history.metric_type
            test_type = tc["test"].test_type
            tc["score"] = tc.get("scores").get(score_type)
            logger.debug(
                f"metric_id {self.metric_id} test_type {test_type} training {score_type} score: {tc['score']}"
            )
            if score_type == "score" and (
                # "too many" anomalies in %
                tc["score"] <= (1 - self.hyper_params.max_train_anomalies_fraction)
                # "too many" anomalies in count
                or (1 - tc["score"]) * scoring_points
                > self.hyper_params.max_train_anomalies_count
            ):
                logger.debug(
                    f"metric_id {self.metric_id} test_type {test_type} training '{score_type}' did not pass the quality threshold. Model is considered invalid."
                )
                continue

            elif metric_group_type == "volume" and test_type == "PctDiff":
                tc["priority"] = 1
            elif metric_group_type == "volume" and test_type == "Trend":
                tc["priority"] = 2
            elif metric_group_type == "time":
                # don't create test on low duration times (maybe should be fine tuned with asset_type tf/task)
                if metric_type == "duration" and metric_history.max <= 300:
                    pass
                elif metric_type != "freshness" and test_type == "Range":
                    tc["priority"] = 1
                elif metric_type == "freshness" and test_type == "Trend":
                    tc["priority"] = 1
                elif test_type == "PctDiff":
                    tc["priority"] = 1
            elif metric_group_type == "schema" and test_type == "PctDiff":
                tc["priority"] = 1
            elif (
                metric_group_type == "distribution"
                and test_type == "Const"
                and tc["test"].var1 == 0
            ):
                tc["priority"] = 1
            elif metric_group_type == "distribution" and test_type == "PctDiff":
                tc["priority"] = 2
            elif metric_group_type == "distribution" and test_type == "Trend":
                tc["priority"] = 3
            elif metric_group_type not in known_group_types:
                tc["priority"] = 1

            if "priority" in tc:
                final_tests.append(tc)
        if len(final_tests):
            final_tests = sorted(final_tests, key=lambda x: x["priority"])
        return final_tests

    def get_const(self, clean):
        min_val = clean[VALUE_COL].min()
        # restrict const test to const zero metrics only
        if min_val == 0 and min_val == clean[VALUE_COL].max():
            return ConstTest(var1=clean[VALUE_COL].min())

    def get_range(self, clean_data):
        lower = np.percentile(
            clean_data[VALUE_COL], self.hyper_params.anomaly_lower_bound_p
        )
        upper = np.percentile(
            clean_data[VALUE_COL], self.hyper_params.anomaly_upper_bound_p
        )
        threshold = self.hyper_params.base_iqr_factor * (upper - lower)
        if not threshold:
            return None
        avg = clean_data[VALUE_COL].mean()
        return RangeTest._get_clean_test(self.metric_history, avg, threshold)

    def get_pct(self, data):
        clean, _, iqr = remove_outliers(
            data,
            self.hyper_params.anomaly_lower_bound_p,
            self.hyper_params.anomaly_upper_bound_p,
            1,
        )
        values = clean[VALUE_COL]
        filtered_avg = values.mean()
        cap_factor = (
            100 * (self.hyper_params.pct_iqr_factor * iqr) / (filtered_avg + EPS)
        )
        percentage = max(
            self.hyper_params.min_pct_value,
            min(cap_factor, self.hyper_params.max_pct_value),
        )

        var_range_low, var_range_high = percent_variability(
            data[VALUE_COL],
            self.hyper_params.anomaly_lower_bound_p,
            self.hyper_params.anomaly_upper_bound_p,
        )
        if (
            var_range_low < self.hyper_params.low_variability_threshold
            and var_range_high < self.hyper_params.low_variability_threshold
        ):
            window = 1
        else:
            anomaly_count = []
            for window in range(
                1,
                min(
                    self.hyper_params.max_window_length,
                    int(self.hyper_params.max_window_fraction * len(values)) + 1,
                ),
            ):
                kernel = np.ones(window) / window
                rolling_average = np.convolve(values, kernel, mode="valid")
                diffs = (
                    abs(values[window:] - rolling_average[:-1]) / rolling_average[:-1]
                ) * 100
                anomalies = diffs[diffs > percentage]
                anom_count = len(anomalies)
                anomaly_count.append(anom_count)
            if len(anomaly_count) > 0:
                window = anomaly_count.index(min(anomaly_count)) + 1
            else:
                window = 1
        return PctDiffTest(var1=percentage, var2=window, normalize=True)

    def get_trend(self, data):
        x = timestamp_col_to_int(data)
        y = data[VALUE_COL]
        is_trend = is_trend_series(x, y)
        if is_trend:
            x = x.reshape(-1, 1)
            model = fit_trend(x, y)
            (
                slope,
                intercept,
            ) = (
                model.coef_[0],
                model.intercept_,
            )
            predictions = model.predict(x)

            tolerance_calc = (
                max(abs(y - predictions)) * self.hyper_params.pct_iqr_factor
            )
            tolerance_min = abs(0.005 * intercept)
            tolerance_max = abs(0.05 * intercept)
            tolerance = SignificantDigits.round(
                max(tolerance_min, min(tolerance_calc, tolerance_max)), 11
            )
            slope = SignificantDigits.round(slope, 11)
            intercept = SignificantDigits.round(intercept, 11)
            return TrendTest(var1=slope, var2=intercept, var3=tolerance)
        else:
            return None
