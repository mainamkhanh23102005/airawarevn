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


class BaselineComparisonTests(unittest.TestCase):
    def records(self, ml=11.0, baseline=12.0):
        return pd.DataFrame([
            {"fold": "2026-02", "timestamp": pd.Timestamp("2026-02-01", tz="UTC"),
             "model": model, "actual_pm25": 10.0, "predicted_pm25": value}
            for model, value in [("linear_regression", ml), ("persistence", baseline)]
        ])

    def compare(self, records):
        from scripts.modeling.backtest import compare_baseline
        return compare_baseline(records, "linear_regression")

    def test_metrics_better_equal_worse_empty_and_zero(self):
        for ml, baseline, percentage in [(11, 12, 50), (12, 12, 0), (14, 12, -100), (11, 10, None), (10, 10, None)]:
            with self.subTest(ml=ml, baseline=baseline):
                result = self.compare(self.records(ml, baseline))
                self.assertEqual(result.ml_mae, abs(ml - 10))
                self.assertEqual(result.baseline_mae, abs(baseline - 10))
                self.assertEqual(result.improvement_percent, percentage)
                self.assertEqual(result.evaluated_forecast_count, 1)
        result = self.compare(self.records().iloc[:0])
        self.assertEqual(result.evaluated_forecast_count, 0)
        self.assertIsNone(result.ml_mae)
        self.assertIsNone(result.baseline_mae)
        self.assertIsNone(result.improvement_percent)

    def test_pairing_by_unique_fold_timestamp_not_order(self):
        first = self.records()
        second = self.records(14, 12).assign(fold="2026-03")
        records = pd.concat([first, second], ignore_index=True).iloc[::-1]
        result = self.compare(records)
        self.assertEqual(result.evaluated_forecast_count, 2)
        self.assertEqual(result.ml_mae, 2.5)
        self.assertEqual(result.improvement_percent, -25)
        with patch("scripts.modeling.backtest.calculate_error_metrics", wraps=calculate_error_metrics) as metrics:
            self.compare(records)
        self.assertEqual(metrics.call_count, 2)

    def test_rejects_missing_extra_duplicate_keys_and_different_truth(self):
        records = self.records()
        invalid = [records.iloc[:1], records.iloc[1:], pd.concat([records, records.iloc[:1]]), pd.concat([records, records.iloc[1:]])]
        for column, value in [("fold", "2026-03"), ("timestamp", pd.Timestamp("2026-02-02", tz="UTC")), ("actual_pm25", 10.000000001), ("fold", None), ("timestamp", pd.NaT)]:
            changed = records.copy()
            changed.loc[1, column] = value
            invalid.append(changed)
        for changed in invalid:
            with self.subTest(records=changed), self.assertRaises(ValueError):
                self.compare(changed)

    def test_rejects_nonfinite_predictions_and_targets_for_either_model(self):
        for row in (0, 1):
            for column in ("actual_pm25", "predicted_pm25"):
                for value in (float("nan"), float("inf"), -float("inf")):
                    records = self.records()
                    records.loc[row, column] = value
                    with self.subTest(row=row, column=column, value=value), self.assertRaises(ValueError):
                        self.compare(records)

    def test_synthetic_cohorts_persistence_and_strict_chronology(self):
        times = pd.date_range("2025-08-01", "2026-07-02", freq="h", tz="UTC")
        raw = pd.DataFrame({"event_time": times, "pm25": [float(i % 97) for i in range(len(times))]})
        missing_time = pd.Timestamp("2026-03-10", tz="UTC")
        raw.loc[raw.event_time == missing_time, "pm25"] = float("nan")
        with patch("scripts.modeling.train_cli.load_frozen_pm25_dataframe", return_value=raw):
            data = build_modeling_dataframe(Path("synthetic.json"))
        self.assertNotIn(missing_time + pd.Timedelta(hours=2), set(data.event_time))
        report = evaluate_walk_forward(data, model_factories={"linear_regression": LinearRegression, "persistence": None})
        result = self.compare(report.predictions)
        self.assertEqual(result.evaluated_forecast_count, report.total_oof_count)
        indexed = raw.set_index("event_time").pm25
        baseline = report.predictions.query("model == 'persistence'")
        self.assertEqual(baseline.predicted_pm25.tolist(), indexed.reindex(baseline.timestamp - pd.Timedelta(hours=1)).tolist())
        self.assertEqual(baseline.actual_pm25.tolist(), indexed.reindex(baseline.timestamp + pd.Timedelta(hours=6)).tolist())
        self.assertTrue((baseline.predicted_pm25.to_numpy() != indexed.reindex(baseline.timestamp).to_numpy()).any())
        folds, reserved = create_walk_forward_folds(data)
        self.assertEqual(tuple(fold.name for fold in folds), VALIDATION_MONTHS)
        previous = 0
        for fold in folds:
            self.assertGreater(len(fold.training), previous)
            self.assertTrue(fold.training.event_time.is_monotonic_increasing)
            self.assertTrue(((fold.training.event_time + pd.Timedelta(hours=6)) < fold.validation_start).all())
            self.assertEqual(fold.purged_boundary_count, 6)
            self.assertTrue((fold.validation.event_time >= fold.validation_start).all())
            self.assertTrue((fold.validation.event_time < fold.validation_end).all())
            previous = len(fold.training)
        self.assertTrue((baseline.timestamp < reserved.start).all())
        origin = pd.Timestamp("2026-04-10", tz="UTC")
        changed = raw.copy()
        changed.loc[changed.event_time >= origin, "pm25"] = 9999.0
        before = build_v1_features(raw, include_target=True).set_index("event_time")
        after = build_v1_features(changed, include_target=True).set_index("event_time")
        pd.testing.assert_series_equal(before.loc[origin, V1_FEATURE_COLUMNS], after.loc[origin, V1_FEATURE_COLUMNS])
        self.assertNotEqual(before.loc[origin, TARGET_COLUMN], after.loc[origin, TARGET_COLUMN])

    def test_cli_fixed_a2_configuration_and_visible_verdicts(self):
        from types import SimpleNamespace
        for ml, baseline, percentage, verdict in [(14, 12, "-100.0000%", "worse"), (11, 12, "50.0000%", "better"), (12, 12, "0.0000%", "equal"), (11, 10, "unavailable", "worse"), (10, 10, "unavailable", "equal")]:
            stdout = io.StringIO()
            data = pd.DataFrame()
            with patch("scripts.modeling.ablation.build_modeling_dataframe", return_value=data) as build, patch("scripts.modeling.ablation.evaluate_walk_forward", return_value=SimpleNamespace(predictions=self.records(ml, baseline))) as evaluate, patch("scripts.modeling.ablation.run_feature_ablation") as ablate, patch("sys.stdout", stdout):
                ablation_main(["--baseline-comparison", "--input", "synthetic.json"])
            build.assert_called_once_with(Path("synthetic.json"))
            self.assertIs(evaluate.call_args.args[0], data)
            self.assertEqual(evaluate.call_args.kwargs["feature_columns"], ABLATION_FEATURE_COLUMNS["A2"])
            self.assertEqual(evaluate.call_args.kwargs["model_factories"], {"linear_regression": LinearRegression, "persistence": None})
            ablate.assert_not_called()
            output = stdout.getvalue()
            for text in ("ml_mae=", "baseline_mae=", f"ml_rmse={abs(ml - 10):.1f}", f"ml_bias={ml - 10:.1f}", f"baseline_rmse={abs(baseline - 10):.1f}", f"baseline_bias={baseline - 10:.1f}", "Bias = predicted - actual", "positive = overprediction", "negative = underprediction", "July origins reserved", "six late-June +6h targets in July retained", "improvement_percent=" + percentage, "evaluated_forecast_count=1", "verdict=" + verdict, "pm25_lag_1h", "latest completed hour", "t+6h", "2025-08", "2026-02", "2026-06", "2026-07", "Asia/Ho_Chi_Minh", "strict", "pooled", "A2"):
                self.assertIn(text, output)


class M8ExperimentTests(unittest.TestCase):
    def raw(self):
        times = pd.date_range("2025-08-01", "2026-07-02", freq="h", tz="UTC")
        return pd.DataFrame({"event_time": times, "pm25": [float(i % 37) for i in range(len(times))]})

    def test_registry_is_frozen_separate_and_features_are_completed(self):
        from scripts.modeling.experimental_features import M8_FEATURE_COLUMNS
        self.assertEqual(list(M8_FEATURE_COLUMNS), ["A2", "M8_STD6", "M8_STD6_12", "M8_STD6_12_24", "M8_HOUR"])
        for name, extra in [("A2", []), ("M8_STD6", ["pm25_rolling_std_6h"]), ("M8_STD6_12", ["pm25_rolling_std_6h", "pm25_rolling_std_12h"]), ("M8_STD6_12_24", ["pm25_rolling_std_6h", "pm25_rolling_std_12h", "pm25_rolling_std_24h"]), ("M8_HOUR", ["hour_sin", "hour_cos"])]:
            self.assertEqual(list(M8_FEATURE_COLUMNS[name]), V1_FEATURE_COLUMNS + extra)
        self.assertEqual(set(ABLATION_FEATURE_COLUMNS), {"A0", "A1", "A2", "A3", "A4", "A5"})
        raw = self.raw().iloc[:60].copy()
        raw["pm25"] = [float((i * i + 7 * i) % 43) for i in range(len(raw))]
        before = build_experimental_features(raw, include_target=True)
        origin = 30
        for width in (6, 12, 24):
            self.assertAlmostEqual(before.loc[origin, f"pm25_rolling_std_{width}h"], raw.pm25.iloc[origin-width:origin].std(ddof=1))
        self.assertEqual(before.loc[origin, "pm25_diff_1h"], raw.pm25.iloc[29] - raw.pm25.iloc[28])
        self.assertEqual(before.loc[origin, "pm25_diff_3h"], raw.pm25.iloc[29] - raw.pm25.iloc[26])
        self.assertEqual(before.loc[origin, "pm25_trend_6h"], raw.pm25.iloc[29] - raw.pm25.iloc[24])
        self.assertEqual(before.loc[origin, TARGET_COLUMN], raw.pm25.iloc[36])
        raw.loc[origin:, "pm25"] = 9999.0
        after = build_experimental_features(raw.iloc[::-1], include_target=True)
        columns = list(dict.fromkeys(column for columns in M8_FEATURE_COLUMNS.values() for column in columns))
        pd.testing.assert_series_equal(before.loc[origin, columns], after.loc[origin, columns])
        self.assertNotEqual(before.loc[origin, TARGET_COLUMN], after.loc[origin, TARGET_COLUMN])

    def test_common_cohort_refits_training_and_validation_and_hashes(self):
        from scripts.modeling.ablation import run_m8_experiment
        from scripts.modeling.experimental_features import build_experimental_features as build
        raw = self.raw()
        def missing(frame, include_target=False):
            result = build(frame, include_target=include_target)
            result.loc[result.event_time.isin(pd.to_datetime(["2025-10-10", "2026-03-10"], utc=True)), "pm25_rolling_std_6h"] = float("nan")
            return result
        with patch("scripts.modeling.ablation.load_frozen_pm25_dataframe", return_value=raw), patch("scripts.modeling.ablation.build_experimental_features", side_effect=missing):
            result = run_m8_experiment(Path(__file__))
            repeated = run_m8_experiment(Path(__file__))
        self.assertEqual(result, repeated)
        native, common = result["cohorts"]["native"], result["cohorts"]["common"]
        self.assertEqual(native["A2"]["evaluated_count"] - common["A2"]["evaluated_count"], 1)
        self.assertEqual(native["A2"]["folds"][0]["training_count"] - common["A2"]["folds"][0]["training_count"], 1)
        for variant in common.values():
            self.assertEqual(variant["cohort_id"], common["A2"]["cohort_id"])
            self.assertEqual([(f["training_id"], f["validation_id"]) for f in variant["folds"]], [(f["training_id"], f["validation_id"]) for f in common["A2"]["folds"]])
            self.assertEqual([f["name"] for f in variant["folds"]], list(VALIDATION_MONTHS))
            self.assertEqual(variant["late_june_target_count"], 6)
        self.assertNotEqual(common["A2"]["schema_id"], common["M8_HOUR"]["schema_id"])
        self.assertEqual(len(common["A2"]["schema_id"]), 64)
        self.assertEqual(common["A2"]["paired_a2_mae_change"], 0)
        self.assertIn("pm25_rolling_std_6h", result["missing_counts"])
        self.assertTrue(common["A2"]["cohort_differs_from_a2"])
        self.assertEqual(common["A2"]["paired_a2_mae"], common["A2"]["paired_candidate_mae"])

    def test_provenance_and_actual_estimator_cohorts(self):
        import hashlib
        from scripts.modeling.ablation import run_m8_experiment
        from scripts.modeling.experimental_features import M8_FEATURE_DEFINITION_VERSION
        raw = self.raw()
        featured = build_experimental_features(raw, include_target=True)
        featured.loc[featured.event_time.isin(pd.to_datetime(["2025-10-10", "2026-03-10"], utc=True)), "pm25_rolling_std_6h"] = float("nan")
        calls = []
        class RecordingLinearRegression(LinearRegression):
            def fit(self, X, y):
                calls.append(("fit", X.copy(), y.copy()))
                return super().fit(X, y)

            def predict(self, X):
                calls.append(("predict", X.copy(), None))
                return super().predict(X)
        with patch("scripts.modeling.ablation.load_frozen_pm25_dataframe", return_value=raw), patch("scripts.modeling.ablation.build_experimental_features", return_value=featured), patch("scripts.modeling.ablation.LinearRegression", RecordingLinearRegression):
            result = run_m8_experiment(Path(__file__))
        self.assertEqual(result["input_sha256"], hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        self.assertEqual(result["experimental_feature_definition_version"], M8_FEATURE_DEFINITION_VERSION)
        from scripts.modeling.experimental_features import M8_FEATURE_COLUMNS
        union = list(dict.fromkeys(c for columns in M8_FEATURE_COLUMNS.values() for c in columns))
        common = featured.dropna(subset=union + [TARGET_COLUMN])
        self.assertEqual(len(calls), 100)
        for index, columns in enumerate(M8_FEATURE_COLUMNS.values()):
            for month_index, month in enumerate(VALIDATION_MONTHS):
                start = pd.Timestamp(month + "-01", tz="Asia/Ho_Chi_Minh").tz_convert("UTC")
                end = start.tz_convert("Asia/Ho_Chi_Minh") + pd.offsets.MonthBegin(1)
                training = common.loc[(common.event_time >= pd.Timestamp("2025-07-31T17:00:00Z")) & (common.event_time + pd.Timedelta(hours=6) < start)]
                validation = common.loc[(common.event_time >= start) & (common.event_time < end)]
                fit, predict = calls[50 + index * 10 + month_index * 2:52 + index * 10 + month_index * 2]
                pd.testing.assert_frame_equal(fit[1].reset_index(drop=True), training[list(columns)].reset_index(drop=True))
                pd.testing.assert_series_equal(fit[2].reset_index(drop=True), training[TARGET_COLUMN].reset_index(drop=True))
                pd.testing.assert_frame_equal(predict[1].reset_index(drop=True), validation[list(columns)].reset_index(drop=True))
                self.assertTrue((validation.event_time < pd.Timestamp("2026-06-30T17:00:00Z")).all())
        baseline = result["cohorts"]["common"]["A2"]
        self.assertAlmostEqual(baseline["canonical_a2_mae_change_percent"], baseline["canonical_a2_mae_change"] / 9.855844823038945 * 100)
        self.assertAlmostEqual(baseline["canonical_persistence_mae_change_percent"], baseline["canonical_persistence_mae_change"] / 11.307251308900524 * 100)

    def test_duplicate_origins_rejected(self):
        from scripts.modeling.ablation import run_m8_experiment
        raw = self.raw()
        with patch("scripts.modeling.ablation.load_frozen_pm25_dataframe", return_value=pd.concat([raw, raw.iloc[:1]])):
            with self.assertRaisesRegex(ValueError, "unique"):
                run_m8_experiment(Path("synthetic.json"))

    def test_m8_missingness_calendar_and_repeatability(self):
        raw = self.raw().iloc[:100].copy()
        raw.loc[40, "pm25"] = float("nan")
        result = build_experimental_features(raw, include_target=True)
        pd.testing.assert_frame_equal(result, build_experimental_features(raw.iloc[::-1], include_target=True))
        for width in (6, 12, 24):
            self.assertTrue(result.loc[41:40+width, f"pm25_rolling_std_{width}h"].isna().all())
            self.assertTrue(pd.notna(result.loc[41+width, f"pm25_rolling_std_{width}h"]))
        self.assertTrue(pd.isna(result.loc[34, TARGET_COLUMN]))
        self.assertAlmostEqual(result.loc[17, "hour_sin"], 0, places=12)
        self.assertAlmostEqual(result.loc[17, "hour_cos"], 1, places=12)

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_m8_canonical_cohorts_and_cli(self):
        from scripts.modeling.ablation import run_m8_experiment
        result = run_m8_experiment(ARTIFACT)
        for mode, variants in result["cohorts"].items():
            baseline = variants["A2"]
            self.assertAlmostEqual(baseline["models"]["linear_regression"]["mae"], 9.855844823038945, places=12)
            self.assertAlmostEqual(baseline["models"]["persistence"]["mae"], 11.307251308900524, places=12)
            for variant in variants.values():
                self.assertEqual(variant["evaluated_count"], 3056)
                self.assertEqual(variant["late_june_target_count"], 6)
                self.assertEqual(variant["common_removed_count"], 0)
                self.assertTrue(variant["paired_a2_training_matched"])
                self.assertEqual([f["training_count"] for f in variant["folds"]], [3655, 4327, 5028, 5717, 6384])
                self.assertEqual([f["validation_count"] for f in variant["folds"]], [672, 701, 689, 667, 327])
        stdout = io.StringIO()
        with patch("scripts.modeling.ablation.run_m8_experiment", return_value=result), patch("sys.stdout", stdout):
            self.assertEqual(ablation_main(["--m8"]), result)
        import json
        self.assertEqual(json.loads(stdout.getvalue()), result)


class M7MetricsTests(unittest.TestCase):
    def synthetic_report(self):
        rows = [{"event_time": pd.Timestamp("2025-08-01", tz="UTC"), "pm25_lag_1h": 10.0, TARGET_COLUMN: 10.0}]
        for count, month in enumerate(VALIDATION_MONTHS, 1):
            for hour in range(count):
                rows.append({"event_time": pd.Timestamp(month + "-02", tz="UTC") + pd.Timedelta(hours=hour), "pm25_lag_1h": 10.0 + count, TARGET_COLUMN: 10.0})
        return evaluate_walk_forward(pd.DataFrame(rows), feature_columns=["pm25_lag_1h"], model_factories={"persistence": None})

    def test_exact_mae_rmse_bias_and_rmse_bound(self):
        metrics = calculate_error_metrics([10, 20, 30], [12, 16, 30])
        self.assertEqual(metrics.mae, 2.0)
        self.assertAlmostEqual(metrics.rmse, (20 / 3) ** 0.5)
        self.assertAlmostEqual(metrics.bias, -2 / 3)
        self.assertGreaterEqual(metrics.rmse, metrics.mae)

    def test_positive_negative_and_zero_bias(self):
        for predicted, bias in [([12, 24], 3), ([8, 16], -3), ([8, 22], 0)]:
            with self.subTest(predicted=predicted):
                self.assertEqual(calculate_error_metrics([10, 20], predicted).bias, bias)

    def test_evaluated_forecast_count(self):
        self.assertEqual(calculate_error_metrics([10, 20, 30], [12, 16, 30]).evaluated_forecast_count, 3)

    def test_row_order_independence(self):
        self.assertEqual(calculate_error_metrics([10, 20, 30], [12, 16, 30]), calculate_error_metrics([30, 10, 20], [30, 12, 16]))

    def test_nonfinite_actuals_and_predictions(self):
        for side in (0, 1):
            for value in (float("nan"), float("inf"), -float("inf")):
                values = [[10.0, 20.0], [12.0, 16.0]]
                values[side][0] = value
                with self.subTest(side=side, value=value), self.assertRaisesRegex(ValueError, "finite"):
                    calculate_error_metrics(*values)

    def test_empty_metrics_explicit_error(self):
        with self.assertRaisesRegex(ValueError, "actual and predicted must be nonempty"):
            calculate_error_metrics([], [])

    def test_mismatched_and_non_1d_shapes(self):
        for actual, predicted in [([1, 2], [1]), ([1], [1, 2]), ([[1, 2]], [[1, 2]]), ([1, 2], [[1], [2]]), (1, 2)]:
            with self.subTest(actual=actual, predicted=predicted), self.assertRaisesRegex(ValueError, "matching one-dimensional shapes"):
                calculate_error_metrics(actual, predicted)

    def test_unequal_folds_pool_rows_not_fold_metrics(self):
        report = self.synthetic_report()
        model = report.models["persistence"]
        self.assertEqual([fold.validation_count for fold in model.folds], [1, 2, 3, 4, 5])
        self.assertEqual(model.pooled.mae, 55 / 15)
        self.assertEqual(model.pooled.bias, 55 / 15)
        self.assertAlmostEqual(model.pooled.rmse, (225 / 15) ** 0.5)
        self.assertEqual(model.macro.mae, 3.0)
        self.assertEqual(model.macro.rmse, 3.0)
        self.assertEqual(model.macro.bias, 3.0)
        self.assertNotEqual(model.pooled.rmse, model.macro.rmse)
        self.assertEqual(model.pooled.evaluated_forecast_count, 15)
        self.assertEqual(model.macro.evaluated_forecast_count, 15)
        self.assertEqual([fold.metrics.evaluated_forecast_count for fold in model.folds], [1, 2, 3, 4, 5])

    def test_baseline_new_metrics_and_empty_cohort(self):
        from scripts.modeling.backtest import BaselineComparison, compare_baseline
        records = BaselineComparisonTests().records(8, 13)
        result = compare_baseline(records, "linear_regression")
        self.assertEqual((result.ml_rmse, result.ml_bias, result.baseline_rmse, result.baseline_bias), (2, -2, 3, 3))

    def test_empty_baseline_new_metrics_unavailable(self):
        from scripts.modeling.backtest import BaselineComparison, compare_baseline
        records = BaselineComparisonTests().records()
        empty = compare_baseline(records.iloc[:0], "linear_regression")
        self.assertEqual(empty.evaluated_forecast_count, 0)
        for field in ("ml_mae", "baseline_mae", "improvement_percent", "ml_rmse", "ml_bias", "baseline_rmse", "baseline_bias"):
            self.assertIsNone(getattr(empty, field))
        self.assertEqual(empty, BaselineComparison(None, None, None, 0))

    def test_normal_cli_fold_and_pooled_bias_count(self):
        from scripts.modeling.ablation import AblationResult
        report = self.synthetic_report()
        stdout = io.StringIO()
        with patch("scripts.modeling.ablation.run_feature_ablation", return_value={"A2": AblationResult("A2", ("pm25_lag_1h",), 15, report)}), patch("sys.stdout", stdout):
            ablation_main([])
        lines = stdout.getvalue().splitlines()
        for count, month in enumerate(VALIDATION_MONTHS, 1):
            line = next(line for line in lines if month in line)
            self.assertIn(f"Bias={count:.4f}", line)
            self.assertIn(f"evaluated_forecast_count={count}", line)
        pooled = next(line for line in lines if "pooled:" in line)
        self.assertIn("Bias=3.6667", pooled)
        self.assertIn("evaluated_forecast_count=15", pooled)

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, CANONICAL_ARTIFACT_SKIP_REASON)
    def test_canonical_m6_metrics_and_late_june_targets(self):
        from scripts.modeling.backtest import compare_baseline
        report = evaluate_walk_forward(build_modeling_dataframe(ARTIFACT), model_factories={"linear_regression": LinearRegression, "persistence": None})
        comparison = compare_baseline(report.predictions, "linear_regression")
        self.assertAlmostEqual(comparison.ml_mae, 9.855844823038945, places=12)
        self.assertAlmostEqual(comparison.baseline_mae, 11.307251308900524, places=12)
        self.assertEqual(comparison.evaluated_forecast_count, 3056)
        for model in report.models.values():
            self.assertEqual(model.pooled.evaluated_forecast_count, 3056)
        june = report.predictions.query("model == 'persistence' and fold == '2026-06'")
        self.assertEqual(int(((june.timestamp + pd.Timedelta(hours=6)) >= report.final_test.start).sum()), 6)
        self.assertTrue((report.predictions.timestamp < report.final_test.start).all())


if __name__ == "__main__":
    unittest.main()
