import hashlib
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from scripts.modeling.features import V1_FEATURE_COLUMNS
from scripts.modeling.train_multi_horizon import (
    HORIZONS,
    build_horizon_dataframe,
    main,
)

UTC = timezone.utc


def make_artifact(directory, rows=40):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records = [
        {
            "sensor_id": 13502151,
            "event_time": (start + timedelta(hours=index)).isoformat(),
            "period_end_utc": (start + timedelta(hours=index + 1)).isoformat(),
            "value": float(index + 1),
            "unit": "µg/m³",
        }
        for index in range(rows)
    ]
    artifact = {
        "sensor_metadata": {
            "sensor_id": 13502151,
            "coordinates": {"latitude": 21.0031, "longitude": 105.7947},
        },
        "normalized_records": records,
        "frozen_candidate": {
            "start_utc": start.isoformat(),
            "end_utc": (start + timedelta(hours=rows)).isoformat(),
        },
    }
    path = Path(directory) / "input.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


class MultiHorizonTrainingTests(unittest.TestCase):
    def test_exact_timestamp_targets_and_independent_cohorts(self):
        raw = pd.DataFrame(
            {
                "event_time": pd.date_range("2025-01-01T00:00:00Z", periods=40, freq="1h"),
                "pm25": [float(index) for index in range(40)],
            }
        )
        raw.loc[35, "pm25"] = None

        one_hour = build_horizon_dataframe(raw, 1)
        six_hour = build_horizon_dataframe(raw, 6)

        self.assertEqual(HORIZONS, (1, 2, 3, 4, 5, 6))
        self.assertTrue((one_hour["target_pm25_t_plus_1"] - one_hour["event_time"].map(lambda value: value.hour) >= 0).all())
        self.assertNotEqual(len(one_hour), len(six_hour))
        self.assertEqual(one_hour[V1_FEATURE_COLUMNS].columns.tolist(), V1_FEATURE_COLUMNS)

    def test_cli_accepts_target_end_equal_to_cutoff_and_records_input_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_artifact(directory)
            output_path = Path(directory) / "bundle.joblib"
            cutoff = "2025-01-02T16:00:00Z"

            returned = main([
                "--input", str(input_path), "--output", str(output_path),
                "--training-cutoff", cutoff, "--model-version", "m9b-test",
            ])

            self.assertEqual(returned, output_path)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                hashlib.sha256(input_path.read_bytes()).hexdigest(),
                __import__("scripts.modeling.forecast_bundle", fromlist=["load_bundle"]).load_bundle(output_path).metadata["training_input_sha256"],
            )

    def test_cli_rejects_existing_output_without_force(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_artifact(directory)
            output_path = Path(directory) / "bundle.joblib"
            arguments = ["--input", str(input_path), "--output", str(output_path), "--training-cutoff", "2025-01-02T16:00:00Z", "--model-version", "m9b-test"]
            main(arguments)
            with self.assertRaises(SystemExit):
                main(arguments)


if __name__ == "__main__":
    unittest.main()
