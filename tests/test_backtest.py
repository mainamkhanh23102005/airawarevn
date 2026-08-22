import io
import unittest
from unittest.mock import patch
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.modeling.backtest import (
    FINAL_TEST_MONTH,
    VALIDATION_MONTHS,
    calculate_metrics,
    create_walk_forward_folds,
    evaluate_walk_forward,
    calculate_error_metrics,
)
from scripts.modeling.experimental_features import (
    ABLATION_FEATURE_COLUMNS,
    CYCLICAL_CALENDAR_COLUMNS,
    FROZEN_A0_FEATURE_COLUMNS,
    DYNAMICS_COLUMNS,
    LONG_LAG_COLUMNS,
    SHORT_LAG_COLUMNS,
    build_experimental_features,
)
from scripts.modeling.ablation import main as ablation_main, run_feature_ablation
from scripts.modeling.features import TARGET_COLUMN, V1_FEATURE_COLUMNS, build_v1_features
from scripts.modeling.train_cli import build_modeling_dataframe


ARTIFACT = Path(".artifacts/data_spike/coverage/openaq_normalized_sensor_13502151_20250731T170000+0000.json")
CANONICAL_ARTIFACT_AVAILABLE = ARTIFACT.exists()
CANONICAL_ARTIFACT_SKIP_REASON = "canonical frozen PM2.5 artifact is not available"


@unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
class CanonicalWalkForwardRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_modeling_dataframe(ARTIFACT)
        cls.folds, cls.final_test = create_walk_forward_folds(cls.data)

    def test_canonical_fold_boundaries_and_counts_are_deterministic(self):
        expected = [
            ("2026-02", "2026-01-31T17:00:00+00:00", 3655, 672),
            ("2026-03", "2026-02-28T17:00:00+00:00", 4327, 701),
            ("2026-04", "2026-03-31T17:00:00+00:00", 5028, 689),
            ("2026-05", "2026-04-30T17:00:00+00:00", 5717, 667),
            ("2026-06", "2026-05-31T17:00:00+00:00", 6384, 327),
        ]
        actual = [(fold.name, fold.validation_start.isoformat(), len(fold.training), len(fold.validation)) for fold in self.folds]
        self.assertEqual(actual, expected)
        self.assertEqual([fold.name for fold in self.folds], list(VALIDATION_MONTHS))

    def test_folds_expand_without_overlap_or_future_labels(self):
        previous_train_count = 0
        previous_validation_end = None
        for fold in self.folds:
            self.assertGreater(len(fold.training), previous_train_count)
            self.assertTrue(((fold.training["event_time"] + pd.Timedelta(hours=6)) < fold.validation_start).all())
            self.assertTrue((fold.training["event_time"] < fold.validation_start).all())
            self.assertTrue((fold.validation["event_time"] >= fold.validation_start).all())
            self.assertTrue((fold.validation["event_time"] < fold.validation_end).all())
            if previous_validation_end is not None:
                self.assertEqual(previous_validation_end, fold.validation_start)
            previous_train_count = len(fold.training)
            previous_validation_end = fold.validation_end

    def test_equal_target_boundary_is_rejected_and_six_rows_are_purged(self):
        for fold in self.folds:
            equal_target_time = fold.validation_start - pd.Timedelta(hours=6)
            self.assertNotIn(equal_target_time, set(fold.training["event_time"]))
            self.assertEqual(fold.purged_boundary_count, 6)

    def test_july_is_reserved_and_absent_from_folds(self):
        self.assertEqual(FINAL_TEST_MONTH, "2026-07")
        self.assertEqual(self.final_test.start.isoformat(), "2026-06-30T17:00:00+00:00")
        self.assertEqual(self.final_test.end.isoformat(), "2026-07-31T17:00:00+00:00")
        self.assertEqual(len(self.final_test.data), 613)
        july_times = set(self.final_test.data["event_time"])
        for fold in self.folds:
            self.assertTrue(july_times.isdisjoint(fold.training["event_time"]))
            self.assertTrue(july_times.isdisjoint(fold.validation["event_time"]))

    def test_models_share_validation_timestamps_and_july_is_not_scored(self):
        report = evaluate_walk_forward(self.data)
        self.assertEqual(report.total_oof_count, 3056)
        self.assertEqual(set(report.models), {"persistence", "linear_regression", "hist_gradient_boosting"})
        for model in report.models.values():
            self.assertEqual(model.total_oof_count, 3056)
            self.assertEqual([result.validation_count for result in model.folds], [672, 701, 689, 667, 327])
            self.assertTrue(all(result.validation_end <= self.final_test.start for result in model.folds))


class BacktestUnitTests(unittest.TestCase):
    def test_metric_calculation(self):
        metrics = calculate_metrics(pd.Series([1.0, 2.0]), pd.Series([2.0, 4.0]))
        self.assertAlmostEqual(metrics.mae, 1.5)
        self.assertAlmostEqual(metrics.rmse, (2.5 ** 0.5))

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_configurable_features_factories_and_oof_records(self):
        data = build_modeling_dataframe(ARTIFACT)
        report = evaluate_walk_forward(
            data,
            feature_columns=["pm25_lag_1h"],
            model_factories={"linear": LinearRegression},
        )
        self.assertEqual(set(report.models), {"linear"})
        self.assertEqual(list(report.predictions.columns), ["timestamp", "fold", "model", "actual_pm25", "predicted_pm25"])
        self.assertEqual(len(report.predictions), 3056)
        self.assertTrue((report.predictions["timestamp"] < report.final_test.start).all())

    def test_error_analysis_metrics(self):
        metrics = calculate_error_metrics(
            pd.Series([10.0, 40.0, 80.0, 100.0]),
            pd.Series([12.0, 30.0, 90.0, 60.0]),
        )
        self.assertAlmostEqual(metrics.bias, -9.5)
        self.assertAlmostEqual(metrics.absolute_error_p50, 10.0)
        self.assertAlmostEqual(metrics.absolute_error_p90, 31.0)
        self.assertEqual(metrics.maximum_absolute_error, 40.0)
        self.assertEqual(metrics.high_pm25[35].count, 3)
        self.assertAlmostEqual(metrics.high_pm25[35].underprediction_rate, 2 / 3)
        self.assertEqual(metrics.high_pm25[75].count, 2)
        self.assertAlmostEqual(metrics.high_pm25[75].signed_bias, -15.0)

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_macro_high_pm25_metrics_are_count_weighted(self):
        data = build_modeling_dataframe(ARTIFACT)
        report = evaluate_walk_forward(data, model_factories={"persistence": None})
        macro = report.models["persistence"].macro
        pooled = report.models["persistence"].pooled
        for threshold in (35, 75):
            self.assertEqual(macro.high_pm25[threshold].count, pooled.high_pm25[threshold].count)
            self.assertAlmostEqual(macro.high_pm25[threshold].mae, pooled.high_pm25[threshold].mae)
            self.assertAlmostEqual(macro.high_pm25[threshold].signed_bias, pooled.high_pm25[threshold].signed_bias)
            self.assertAlmostEqual(macro.high_pm25[threshold].underprediction_rate, pooled.high_pm25[threshold].underprediction_rate)

    def test_experimental_features_are_shifted_hourly_and_preserve_missingness(self):
        values = [float(value) for value in range(200)]
        values[50] = float("nan")
        raw = pd.DataFrame({"event_time": pd.date_range("2026-01-01", periods=200, freq="h", tz="UTC"), "pm25": values})
        result = build_experimental_features(raw)
        self.assertEqual(result.loc[10, "pm25_lag_2h"], 8.0)
        self.assertEqual(result.loc[10, "pm25_lag_4h"], 6.0)
        self.assertEqual(result.loc[10, "pm25_diff_1h"], 1.0)
        self.assertEqual(result.loc[10, "pm25_diff_3h"], 3.0)
        self.assertAlmostEqual(result.loc[10, "pm25_rolling_std_6h"], pd.Series([4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).std())
        self.assertAlmostEqual(result.loc[30, "pm25_mean_gap_6h_24h"], 26.5 - 17.5)
        self.assertTrue(pd.isna(result.loc[52, "pm25_lag_2h"]))
        self.assertTrue(pd.isna(result.loc[51, "pm25_diff_1h"]))

    def test_cyclical_features_use_hanoi_time(self):
        raw = pd.DataFrame({"event_time": pd.to_datetime(["2026-01-01T17:00:00Z"]), "pm25": [1.0]})
        result = build_experimental_features(raw)
        self.assertAlmostEqual(result.loc[0, "hour_sin"], 0.0, places=12)
        self.assertAlmostEqual(result.loc[0, "hour_cos"], 1.0, places=12)
        self.assertEqual(set(ABLATION_FEATURE_COLUMNS), {"A0", "A1", "A2", "A3", "A4", "A5"})
        self.assertEqual(FROZEN_A0_FEATURE_COLUMNS, [
            "pm25_lag_1h",
            "pm25_lag_3h",
            "pm25_lag_6h",
            "pm25_lag_12h",
            "pm25_lag_24h",
            "pm25_rolling_mean_6h",
            "pm25_rolling_mean_12h",
            "pm25_rolling_mean_24h",
            "hour",
            "day_of_week",
            "month",
            "is_weekend",
        ])
        self.assertEqual(ABLATION_FEATURE_COLUMNS["A0"], FROZEN_A0_FEATURE_COLUMNS)
        self.assertEqual(ABLATION_FEATURE_COLUMNS["A2"], [*FROZEN_A0_FEATURE_COLUMNS, *SHORT_LAG_COLUMNS])
        self.assertEqual(ABLATION_FEATURE_COLUMNS["A5"], [*FROZEN_A0_FEATURE_COLUMNS, *DYNAMICS_COLUMNS, *SHORT_LAG_COLUMNS])
        self.assertNotEqual(ABLATION_FEATURE_COLUMNS["A0"], ABLATION_FEATURE_COLUMNS["A2"])
        self.assertNotEqual(ABLATION_FEATURE_COLUMNS["A1"], ABLATION_FEATURE_COLUMNS["A5"])
        self.assertEqual(V1_FEATURE_COLUMNS, ABLATION_FEATURE_COLUMNS["A2"])
        self.assertTrue(set(DYNAMICS_COLUMNS).isdisjoint(SHORT_LAG_COLUMNS))
        self.assertTrue(set(LONG_LAG_COLUMNS).isdisjoint(CYCLICAL_CALENDAR_COLUMNS))

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_ablation_variants_share_model_timestamps_and_exclude_july(self):
        results = run_feature_ablation(ARTIFACT)
        self.assertEqual(set(results), {"A0", "A1", "A2", "A3", "A4", "A5"})
        self.assertEqual(results["A5"].retained_validation_count, 3056)
        expected = {
            "A0": (10.1062, 14.3186),
            "A2": (9.8558, 14.0273),
            "A5": (9.8923, 14.0781),
        }
        for name, (mae, rmse) in expected.items():
            pooled = results[name].report.models["linear_regression"].pooled
            self.assertAlmostEqual(pooled.mae, mae, places=4)
            self.assertAlmostEqual(pooled.rmse, rmse, places=4)
        for result in results.values():
            report = result.report
            timestamps = [set(group["timestamp"]) for _, group in report.predictions.groupby("model")]
            self.assertTrue(all(value == timestamps[0] for value in timestamps[1:]))
            self.assertTrue((report.predictions["timestamp"] < report.final_test.start).all())
            self.assertEqual(result.retained_validation_count, report.total_oof_count)

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_ablation_cli_prints_february_through_june_only(self):
        stdout = io.StringIO()
        with patch("sys.stdout", stdout):
            ablation_main(["--input", str(ARTIFACT)])
        output = stdout.getvalue()
        self.assertIn("A0", output)
        self.assertIn("2026-02", output)
        self.assertIn("2026-06", output)
        self.assertNotIn("2026-07", output)

    def test_missing_pm25_is_not_filled(self):
        raw = pd.DataFrame({
            "event_time": pd.date_range("2026-01-01", periods=32, freq="h", tz="UTC"),
            "pm25": [1.0] * 10 + [float("nan")] + [1.0] * 21,
        })
        featured = build_v1_features(raw, include_target=True)
        self.assertTrue(pd.isna(featured.loc[10, "pm25"]))
        self.assertTrue(featured[V1_FEATURE_COLUMNS + [TARGET_COLUMN]].isna().any().any())


if __name__ == "__main__":
    unittest.main()
