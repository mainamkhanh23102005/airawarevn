import hashlib
import io
import json
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.modeling.features import (
    FORECAST_HORIZON_HOURS,
    TARGET_COLUMN,
    V1_FEATURE_COLUMNS,
)
from scripts.modeling.predict import load_artifact, predict_pm25_t_plus_6
from scripts.modeling.train import MODEL_TYPE
from scripts.modeling.train_cli import (
    build_modeling_dataframe,
    load_frozen_pm25_dataframe,
    main,
)


UTC = timezone.utc


def make_frozen_artifact(
    directory,
    rows=48,
    missing_indices=None,
    extra_fields=None,
    include_window=True,
):
    start = datetime(2025, 1, 1, tzinfo=UTC)
    records = []
    for index in range(rows):
        record = {
            "sensor_id": 13502151,
            "event_time": (start + timedelta(hours=index)).isoformat(),
            "value": float(index + 1),
            "unit": "µg/m³",
        }
        if extra_fields:
            record.update(extra_fields)
        records.append(record)
    if missing_indices:
        for index in missing_indices:
            records[index]["value"] = None

    artifact = {
        "sensor_metadata": {
            "sensor_id": 13502151,
            "coordinates": {"latitude": 21.0031, "longitude": 105.7947},
        },
        "normalized_records": records,
    }
    if include_window:
        artifact["frozen_candidate"] = {
            "start_utc": start.isoformat(),
            "end_utc": (start + timedelta(hours=rows)).isoformat(),
        }

    path = Path(directory) / "pm25.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


class TrainCliHelpTests(unittest.TestCase):
    def test_help_exits_zero_and_describes_workflow(self):
        with self.assertRaises(SystemExit) as context:
            main(["--help"])

        self.assertEqual(context.exception.code, 0)


class TrainCliSmokeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.input_path = make_frozen_artifact(self.directory.name)
        self.output_path = Path(self.directory.name) / "model.joblib"

    def tearDown(self):
        self.directory.cleanup()

    def test_trains_and_writes_loadable_artifact(self):
        returned = main(
            [
                "--input",
                str(self.input_path),
                "--output",
                str(self.output_path),
            ]
        )

        self.assertEqual(returned, self.output_path)
        self.assertTrue(self.output_path.exists())

        model, metadata = load_artifact(self.output_path)
        self.assertEqual(metadata["model_type"], MODEL_TYPE)
        self.assertEqual(metadata["feature_columns"], V1_FEATURE_COLUMNS)
        self.assertEqual(metadata["target_column"], TARGET_COLUMN)
        self.assertEqual(metadata["forecast_horizon_hours"], FORECAST_HORIZON_HOURS)
        self.assertFalse(metadata["raw_pm25_is_feature"])
        self.assertFalse(metadata["weather_is_feature"])
        self.assertEqual(
            metadata["training_input_sha256"],
            hashlib.sha256(self.input_path.read_bytes()).hexdigest(),
        )

        modeling_df = build_modeling_dataframe(self.input_path)
        prediction = predict_pm25_t_plus_6(
            model, metadata, modeling_df.iloc[[-1]][V1_FEATURE_COLUMNS]
        )
        self.assertEqual(len(prediction), 1)

    def test_verbose_prints_summary(self):
        stdout = io.StringIO()
        with unittest.mock.patch("sys.stdout", stdout):
            main(
                [
                    "--input",
                    str(self.input_path),
                    "--output",
                    str(self.output_path),
                    "--verbose",
                ]
            )

        output = stdout.getvalue()
        self.assertIn(str(self.output_path), output)
        self.assertIn("training rows:", output)
        self.assertIn("features:", output)
        self.assertIn(TARGET_COLUMN, output)

    def test_refuses_to_overwrite_without_force(self):
        main(
            [
                "--input",
                str(self.input_path),
                "--output",
                str(self.output_path),
            ]
        )

        with self.assertRaises(SystemExit) as context:
            main(
                [
                    "--input",
                    str(self.input_path),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(context.exception.code, 2)

    def test_force_overwrites_existing_artifact(self):
        main(
            [
                "--input",
                str(self.input_path),
                "--output",
                str(self.output_path),
            ]
        )
        _, first_metadata = load_artifact(self.output_path)

        main(
            [
                "--input",
                str(self.input_path),
                "--output",
                str(self.output_path),
                "--force",
            ]
        )

        self.assertTrue(self.output_path.exists())
        _, second_metadata = load_artifact(self.output_path)
        self.assertNotEqual(
            first_metadata["trained_at_utc"],
            second_metadata["trained_at_utc"],
        )

    def test_missing_input_exits_with_error(self):
        missing = Path(self.directory.name) / "missing.json"
        with self.assertRaises(SystemExit) as context:
            main(
                [
                    "--input",
                    str(missing),
                    "--output",
                    str(self.output_path),
                ]
            )

        self.assertEqual(context.exception.code, 2)


class TrainCliDataSemanticsTests(unittest.TestCase):
    def test_missing_pm25_hours_are_not_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            # Drop a single hour in the middle of the feature-history range so
            # that the row at that hour has incomplete lags and is excluded.
            input_path = make_frozen_artifact(
                directory, rows=48, missing_indices=[26]
            )
            output_path = Path(directory) / "model.joblib"
            main(
                [
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ]
            )

            modeling_df = build_modeling_dataframe(input_path)
            self.assertFalse(modeling_df[V1_FEATURE_COLUMNS].isna().any().any())
            self.assertFalse(modeling_df[TARGET_COLUMN].isna().any())

    def test_artifact_without_frozen_window_exits_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_frozen_artifact(
                directory, rows=48, include_window=False
            )
            output_path = Path(directory) / "model.joblib"
            with self.assertRaises(SystemExit) as context:
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(context.exception.code, 2)

    def test_duplicate_units_use_shared_resolution_semantics(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_frozen_artifact(directory)
            artifact = json.loads(input_path.read_text(encoding="utf-8"))
            duplicate = artifact["normalized_records"][30].copy()
            duplicate["value"] /= 1000
            duplicate["unit"] = "mg/m³"
            artifact["normalized_records"].append(duplicate)
            input_path.write_text(json.dumps(artifact), encoding="utf-8")

            modeling_df = build_modeling_dataframe(input_path)

            self.assertFalse(modeling_df.empty)

    def test_ambiguous_duplicate_is_not_used(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_frozen_artifact(directory)
            artifact = json.loads(input_path.read_text(encoding="utf-8"))
            duplicate = artifact["normalized_records"][30].copy()
            duplicate["value"] += 1
            artifact["normalized_records"].append(duplicate)
            input_path.write_text(json.dumps(artifact), encoding="utf-8")

            raw = load_frozen_pm25_dataframe(input_path)

            ambiguous_row = raw.loc[
                raw["event_time"] == datetime(2025, 1, 2, 6, tzinfo=UTC)
            ]
            self.assertTrue(ambiguous_row["pm25"].isna().all())

    def test_sensor_metadata_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_frozen_artifact(directory)
            artifact = json.loads(input_path.read_text(encoding="utf-8"))
            del artifact["sensor_metadata"]
            input_path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaises(ValueError):
                build_modeling_dataframe(input_path)

    def test_record_outside_frozen_window_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_frozen_artifact(directory)
            artifact = json.loads(input_path.read_text(encoding="utf-8"))
            artifact["normalized_records"][0]["event_time"] = (
                datetime(2024, 12, 31, 23, tzinfo=UTC).isoformat()
            )
            input_path.write_text(json.dumps(artifact), encoding="utf-8")

            with self.assertRaises(ValueError):
                build_modeling_dataframe(input_path)

    def test_empty_artifact_exits_with_error(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "empty.json"
            input_path.write_text(
                json.dumps({"normalized_records": []}), encoding="utf-8"
            )
            output_path = Path(directory) / "model.joblib"

            with self.assertRaises(SystemExit) as context:
                main(
                    [
                        "--input",
                        str(input_path),
                        "--output",
                        str(output_path),
                    ]
                )

            self.assertEqual(context.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
