import tempfile
import unittest
from pathlib import Path

import pandas as pd

from scripts.modeling.features import (
    TARGET_COLUMN,
    V1_FEATURE_COLUMNS,
    WEATHER_COLUMNS,
    build_v1_features,
)
from scripts.modeling.predict import load_artifact, predict_pm25_t_plus_6
from scripts.modeling.train import save_artifact, train_v1_model


class FeatureTests(unittest.TestCase):
    def test_v1_feature_contract_excludes_raw_pm25_and_weather(self):
        self.assertEqual(
            V1_FEATURE_COLUMNS,
            [
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
                "pm25_lag_2h",
                "pm25_lag_4h",
                "pm25_lag_8h",
                "pm25_lag_18h",
            ],
        )
        self.assertNotIn("pm25", V1_FEATURE_COLUMNS)
        self.assertTrue(WEATHER_COLUMNS.isdisjoint(V1_FEATURE_COLUMNS))

    def test_builds_lags_and_completed_interval_rolling_features(self):
        frame = pd.DataFrame(
            {
                "event_time": pd.date_range(
                    "2025-01-01T00:00:00Z",
                    periods=32,
                    freq="1h",
                ),
                "pm25": range(32),
            }
        )

        result = build_v1_features(frame, include_target=True)
        row = result.iloc[24]

        self.assertEqual(row["pm25_lag_1h"], 23)
        self.assertEqual(row["pm25_lag_2h"], 22)
        self.assertEqual(row["pm25_lag_3h"], 21)
        self.assertEqual(row["pm25_lag_4h"], 20)
        self.assertEqual(row["pm25_lag_8h"], 16)
        self.assertEqual(row["pm25_lag_18h"], 6)
        self.assertEqual(row["pm25_lag_24h"], 0)
        self.assertEqual(row["pm25_rolling_mean_6h"], 20.5)
        self.assertEqual(row["pm25_rolling_mean_24h"], 11.5)
        self.assertEqual(row[TARGET_COLUMN], 30)

    def test_missing_pm25_propagates_to_new_lags_without_filling(self):
        frame = pd.DataFrame({
            "event_time": pd.date_range("2025-01-01T00:00:00Z", periods=30, freq="1h"),
            "pm25": [float(value) for value in range(30)],
        })
        frame.loc[10, "pm25"] = None
        result = build_v1_features(frame, include_target=True)
        for lag in (2, 4, 8, 18):
            self.assertTrue(pd.isna(result.loc[10 + lag, f"pm25_lag_{lag}h"]))
        self.assertTrue(pd.isna(result.loc[10 - 6, TARGET_COLUMN]))

    def test_builds_vietnam_local_calendar_features_and_sorts_rows(self):
        frame = pd.DataFrame(
            {
                "event_time": pd.to_datetime(
                    ["2025-01-05T18:00:00Z", "2025-01-05T17:00:00Z"]
                ),
                "pm25": [20.0, 10.0],
            }
        )

        result = build_v1_features(frame)

        self.assertTrue(result["event_time"].is_monotonic_increasing)
        self.assertEqual(result.iloc[0]["hour"], 0)
        self.assertEqual(result.iloc[0]["day_of_week"], 0)
        self.assertEqual(result.iloc[0]["month"], 1)
        self.assertEqual(result.iloc[0]["is_weekend"], 0)


class ModelingTests(unittest.TestCase):
    def setUp(self):
        rows = 40
        source = pd.DataFrame(
            {
                "event_time": pd.date_range(
                    "2025-01-01T00:00:00Z",
                    periods=rows,
                    freq="1h",
                ),
                "pm25": [float(value) for value in range(rows)],
            }
        )
        self.prepared = build_v1_features(source, include_target=True).dropna(
            subset=[*V1_FEATURE_COLUMNS, TARGET_COLUMN]
        )

    def test_training_rejects_missing_required_values(self):
        invalid = self.prepared.copy()
        invalid.loc[invalid.index[0], "pm25_lag_1h"] = None

        with self.assertRaisesRegex(ValueError, "missing required values"):
            train_v1_model(invalid)

    def test_prediction_uses_trained_feature_order(self):
        model, metadata = train_v1_model(self.prepared)
        reordered = self.prepared.iloc[[-1]][list(reversed(V1_FEATURE_COLUMNS))]

        prediction = predict_pm25_t_plus_6(model, metadata, reordered)
        expected = model.predict(
            self.prepared.iloc[[-1]][V1_FEATURE_COLUMNS]
        )

        self.assertEqual(prediction.tolist(), expected.tolist())

    def test_serialization_round_trip_preserves_prediction(self):
        model, metadata = train_v1_model(self.prepared)
        inputs = self.prepared.iloc[[-1]][V1_FEATURE_COLUMNS]
        before = predict_pm25_t_plus_6(model, metadata, inputs)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "v1.joblib"
            save_artifact(path, model, metadata)
            loaded_model, loaded_metadata = load_artifact(path)
            after = predict_pm25_t_plus_6(
                loaded_model,
                loaded_metadata,
                inputs,
            )

        self.assertEqual(before.tolist(), after.tolist())
        self.assertEqual(loaded_metadata["feature_columns"], V1_FEATURE_COLUMNS)
        self.assertEqual(loaded_metadata["feature_configuration"], "A2")
        self.assertEqual(loaded_metadata["forecast_horizon_hours"], 6)


if __name__ == "__main__":
    unittest.main()
