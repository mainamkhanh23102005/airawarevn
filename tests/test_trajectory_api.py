import json
import pickle
import tempfile
import unittest
from unittest.mock import patch
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from scripts.modeling.train_multi_horizon import main as train_main
from scripts.modeling.train_cli import main as train_legacy_main
from tests.test_train_multi_horizon import make_artifact

UTC = timezone.utc


class TrajectoryApiTests(unittest.TestCase):
    def test_missing_optional_bundle_preserves_legacy_health_and_disables_trajectory(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.joblib"
            input_path = make_artifact(directory, rows=48)
            legacy_path = Path(directory) / "legacy.joblib"
            train_legacy_main(["--input", str(input_path), "--output", str(legacy_path)])
            with TestClient(create_app(model_path=legacy_path, multi_horizon_model_path=missing)) as client:
                response = client.get("/forecast/trajectory")
            self.assertEqual(response.status_code, 503)

    def test_corrupt_optional_bundle_does_not_block_legacy_startup(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_artifact(directory, rows=48)
            legacy_path = Path(directory) / "legacy.joblib"
            train_legacy_main(["--input", str(input_path), "--output", str(legacy_path)])
            with patch("app.main.load_bundle", side_effect=pickle.UnpicklingError("bad bundle")):
                with TestClient(create_app(model_path=legacy_path, multi_horizon_model_path=Path(directory) / "bundle.joblib")) as client:
                    self.assertEqual(client.get("/health").status_code, 200)
                    self.assertEqual(client.get("/forecast/trajectory").status_code, 503)

    def test_trajectory_returns_six_ordered_intervals(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = make_artifact(directory, rows=56)
            bundle_path = Path(directory) / "bundle.joblib"
            train_main(["--input", str(input_path), "--output", str(bundle_path), "--training-cutoff", "2025-01-03T08:00:00Z", "--model-version", "m9b-test"])
            legacy_path = Path(directory) / "legacy.joblib"
            train_legacy_main(["--input", str(input_path), "--output", str(legacy_path)])
            current = json.loads(input_path.read_text(encoding="utf-8"))
            current.update({"artifact_version": 1, "sensor_id": 13502151, "retrieved_at": "2025-01-03T08:00:00Z"})
            current_path = Path(directory) / "current.json"
            current_path.write_text(json.dumps(current), encoding="utf-8")
            with TestClient(create_app(model_path=legacy_path, multi_horizon_model_path=bundle_path, current_pm25_artifact_path=current_path, now=lambda: datetime(2025, 1, 3, 8, tzinfo=UTC))) as client:
                response = client.get("/forecast/trajectory")
            self.assertEqual(response.status_code, 200)
            body = response.json()
            self.assertEqual([item["forecast_horizon_hours"] for item in body["forecasts"]], [1, 2, 3, 4, 5, 6])
            self.assertEqual(body["forecasts"][0]["target_interval_start"], "2025-01-03T09:00:00Z")


if __name__ == "__main__":
    unittest.main()
