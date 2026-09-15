import tempfile
import unittest
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.modeling.features import V1_FEATURE_COLUMNS
from scripts.modeling.forecast_bundle import (
    HORIZONS,
    build_bundle,
    load_bundle,
    predict_trajectory,
    save_bundle,
)


class ForecastBundleTests(unittest.TestCase):
    def _models(self):
        inputs = pd.DataFrame(
            [[float(column_index + row_index) for column_index in range(len(V1_FEATURE_COLUMNS))] for row_index in range(3)],
            columns=V1_FEATURE_COLUMNS,
        )
        return {horizon: LinearRegression().fit(inputs, [float(horizon + row_index) for row_index in range(3)]) for horizon in HORIZONS}, inputs.iloc[[0]]

    def test_round_trip_predicts_all_ordered_horizons(self):
        models, inputs = self._models()
        bundle = build_bundle(models, model_version="m9b-test", training_input_sha256="a" * 64, training_cutoff="2025-01-02T00:00:00Z", horizon_metadata={horizon: {"training_row_count": 3, "training_origin_start": "2025-01-01T00:00:00+00:00", "training_origin_end": "2025-01-01T02:00:00+00:00", "target_interval_start": "2025-01-01T01:00:00+00:00", "target_interval_end": "2025-01-01T09:00:00+00:00", "cohort_sha256": "b" * 64} for horizon in HORIZONS})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bundle.joblib"
            save_bundle(path, bundle)
            loaded = load_bundle(path)
        predictions = predict_trajectory(loaded, inputs)
        self.assertEqual([item["forecast_horizon_hours"] for item in predictions], list(HORIZONS))
        self.assertEqual(len(predictions), 6)

    def test_rejects_metadata_with_target_end_after_cutoff(self):
        models, _ = self._models()
        metadata = {horizon: {"training_row_count": 3, "training_origin_start": "2025-01-01T00:00:00+00:00", "training_origin_end": "2025-01-01T02:00:00+00:00", "target_interval_start": "2025-01-01T01:00:00+00:00", "target_interval_end": "2025-01-03T00:00:00+00:00", "cohort_sha256": "b" * 64} for horizon in HORIZONS}
        with self.assertRaises(ValueError):
            build_bundle(models, model_version="m9b-test", training_input_sha256="a" * 64, training_cutoff="2025-01-02T00:00:00Z", horizon_metadata=metadata)

    def test_rejects_missing_horizon_and_nonfinite_estimator(self):
        models, _ = self._models()
        with self.assertRaises(ValueError):
            build_bundle({horizon: model for horizon, model in models.items() if horizon != 6}, model_version="m9b-test", training_input_sha256="a" * 64, training_cutoff="2025-01-02T00:00:00Z", horizon_metadata={})
        models[1].coef_[0] = float("nan")
        with self.assertRaises(ValueError):
            build_bundle(models, model_version="m9b-test", training_input_sha256="a" * 64, training_cutoff="2025-01-02T00:00:00Z", horizon_metadata={})


if __name__ == "__main__":
    unittest.main()
