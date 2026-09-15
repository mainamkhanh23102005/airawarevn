import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from scripts.modeling.backtest import create_walk_forward_folds, evaluate_walk_forward
from scripts.modeling.features import TARGET_COLUMN, V1_FEATURE_COLUMNS
from scripts.modeling.multi_horizon import (
    HORIZONS,
    build_horizon_dataframe,
    run_multi_horizon_experiment,
    target_column_for_horizon,
)


ARTIFACT = Path(".artifacts/data_spike/coverage/openaq_normalized_sensor_13502151_20250731T170000+0000.json")
CANONICAL_ARTIFACT_AVAILABLE = ARTIFACT.exists()


class MultiHorizonUnitTests(unittest.TestCase):
    def raw(self):
        times = pd.date_range("2025-08-01", "2026-07-02", freq="h", tz="UTC")
        return pd.DataFrame({"event_time": times, "pm25": [float(index % 97) for index in range(len(times))]})

    def test_targets_use_exact_event_time_offsets_for_one_and_six_hours(self):
        raw = self.raw()
        origin = pd.Timestamp("2026-03-10T12:00:00Z")
        for horizon in (1, 6):
            data = build_horizon_dataframe(raw, horizon).set_index("event_time")
            self.assertEqual(target_column_for_horizon(horizon), f"target_pm25_t_plus_{horizon}")
            self.assertEqual(data.loc[origin, target_column_for_horizon(horizon)], raw.set_index("event_time").loc[origin + pd.Timedelta(hours=horizon), "pm25"])
        self.assertEqual(target_column_for_horizon(6), TARGET_COLUMN)

    def test_missing_intermediate_row_does_not_shift_target(self):
        raw = self.raw()
        origin = pd.Timestamp("2026-03-10T12:00:00Z")
        raw = raw.loc[raw.event_time != origin + pd.Timedelta(hours=3)].reset_index(drop=True)
        data = build_horizon_dataframe(raw, 6).set_index("event_time")
        self.assertEqual(data.loc[origin, target_column_for_horizon(6)], raw.set_index("event_time").loc[origin + pd.Timedelta(hours=6), "pm25"])

    def test_features_at_origin_do_not_use_future_pm25(self):
        raw = self.raw()
        origin = pd.Timestamp("2026-03-10T12:00:00Z")
        changed = raw.copy()
        changed.loc[changed.event_time >= origin, "pm25"] = 9999.0
        before = build_horizon_dataframe(raw, 1).set_index("event_time")
        after = build_horizon_dataframe(changed, 1).set_index("event_time")
        pd.testing.assert_series_equal(before.loc[origin, V1_FEATURE_COLUMNS], after.loc[origin, V1_FEATURE_COLUMNS])
        self.assertNotEqual(before.loc[origin, target_column_for_horizon(1)], after.loc[origin, target_column_for_horizon(1)])

    def test_each_horizon_purges_its_own_unavailable_targets(self):
        raw = self.raw()
        for horizon in HORIZONS:
            data = build_horizon_dataframe(raw, horizon)
            folds, _ = create_walk_forward_folds(data, target_column=target_column_for_horizon(horizon), horizon_hours=horizon)
            self.assertEqual([fold.purged_boundary_count for fold in folds], [horizon] * len(folds))
            for fold in folds:
                self.assertTrue(((fold.training.event_time + pd.Timedelta(hours=horizon)) < fold.validation_start).all())
                self.assertNotIn(fold.validation_start - pd.Timedelta(hours=horizon), set(fold.training.event_time))

    def test_runner_is_ordered_deterministic_and_pairs_each_horizon(self):
        raw = self.raw()
        with TemporaryDirectory() as directory:
            artifact = Path(directory) / "synthetic.json"
            artifact.write_bytes(b"synthetic")
            with patch("scripts.modeling.multi_horizon.load_frozen_pm25_dataframe", return_value=raw):
                first = run_multi_horizon_experiment(artifact)
                second = run_multi_horizon_experiment(artifact)
        self.assertEqual(first, second)
        self.assertEqual(first["horizons"], list(HORIZONS))
        for horizon in HORIZONS:
            result = first["results"][str(horizon)]
            self.assertEqual(result["linear_regression"]["count"], result["persistence"]["count"])
            self.assertEqual(result["comparison"]["count"], result["linear_regression"]["count"])

    def test_persistence_uses_latest_completed_origin_observation(self):
        raw = self.raw()
        data = build_horizon_dataframe(raw, 1)
        report = evaluate_walk_forward(data, feature_columns=V1_FEATURE_COLUMNS, model_factories={"persistence": None}, target_column=target_column_for_horizon(1), horizon_hours=1)
        predictions = report.predictions
        lookup = data.set_index("event_time")["pm25_lag_1h"]
        self.assertEqual(predictions.predicted_pm25.tolist(), lookup.reindex(predictions.timestamp).tolist())

    @unittest.skipUnless(CANONICAL_ARTIFACT_AVAILABLE, "canonical frozen PM2.5 artifact is not available")
    def test_canonical_six_hour_regression_is_unchanged(self):
        result = run_multi_horizon_experiment(ARTIFACT)["results"]["6"]
        self.assertAlmostEqual(result["linear_regression"]["mae"], 9.855844823038945, places=12)
        self.assertAlmostEqual(result["linear_regression"]["rmse"], 14.027286230773658, places=12)
        self.assertAlmostEqual(result["linear_regression"]["bias"], 0.9370505512624444, places=12)
        self.assertEqual(result["linear_regression"]["count"], 3056)
        self.assertAlmostEqual(result["persistence"]["mae"], 11.307251308900524, places=12)
        self.assertAlmostEqual(result["persistence"]["rmse"], 16.16191190424337, places=12)
        self.assertAlmostEqual(result["persistence"]["bias"], 0.11611256544502616, places=12)
        self.assertEqual(result["persistence"]["count"], 3056)
