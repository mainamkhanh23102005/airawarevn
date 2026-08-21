import asyncio
import json
import math
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import joblib
import pandas as pd
from fastapi.testclient import TestClient

from app import main
from app.main import create_app
from scripts.modeling.features import V1_FEATURE_COLUMNS, build_v1_features
from scripts.modeling.predict import predict_pm25_t_plus_6
from scripts.modeling.train import save_artifact, train_v1_model


class ApiTests(unittest.TestCase):
    def setUp(self):
        source = pd.DataFrame(
            {
                "event_time": pd.date_range(
                    "2025-01-01T00:00:00Z",
                    periods=60,
                    freq="1h",
                ),
                "pm25": [float(value) for value in range(60)],
            }
        )
        prepared = build_v1_features(source, include_target=True).dropna()
        self.model, self.metadata = train_v1_model(prepared)
        self.directory = tempfile.TemporaryDirectory()
        self.artifact_path = Path(self.directory.name) / "v1.joblib"
        save_artifact(
            self.artifact_path,
            self.model,
            self.metadata,
        )
        self.prediction_time = datetime(2025, 2, 2, 0, tzinfo=timezone.utc)
        self.pm25_artifact_path = Path(self.directory.name) / "pm25.json"
        self.current_pm25_artifact_path = Path(self.directory.name) / "live" / "current_pm25.json"
        self.history = [
            {
                "event_time": (
                    self.prediction_time - timedelta(hours=24 - index)
                ).isoformat(),
                "pm25": float(index + 1),
            }
            for index in range(24)
        ]
        self.write_pm25_artifact(self.history)

    def tearDown(self):
        self.directory.cleanup()

    def write_pm25_artifact(self, history):
        records = [
            {
                "event_time": entry["event_time"],
                "period_end_utc": (
                    datetime.fromisoformat(entry["event_time"])
                    + timedelta(hours=1)
                ).isoformat(),
                "record_id": index,
                "sensor_id": 13502151,
                "unit": "µg/m³",
                "value": entry["pm25"],
            }
            for index, entry in enumerate(history)
        ]
        artifact = {
            "frozen_candidate": {
                "start_utc": records[0]["event_time"],
                "end_utc": self.prediction_time.isoformat(),
            },
            "normalized_records": records,
            "sensor_metadata": {
                "sensor_id": 13502151,
                "coordinates": {"latitude": 21.0031, "longitude": 105.7947},
            },
        }
        self.pm25_artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    def write_current_pm25_artifact(self, history=None, retrieved_at=None):
        history = self.history if history is None else history
        records = [
            {
                "event_time": entry["event_time"],
                "period_end_utc": (
                    datetime.fromisoformat(entry["event_time"]) + timedelta(hours=1)
                ).isoformat(),
                "record_id": index,
                "sensor_id": 13502151,
                "unit": "µg/m³",
                "value": entry["pm25"],
            }
            for index, entry in enumerate(history)
        ]
        artifact = {
            "artifact_version": 1,
            "sensor_id": 13502151,
            "retrieved_at": (retrieved_at or self.prediction_time).isoformat(),
            "source": {"provider": "OpenAQ", "endpoint": "/sensors/13502151/hours"},
            "provenance": [],
            "normalized_records": records,
        }
        self.current_pm25_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.current_pm25_artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    def client(self, artifact_path=None, now=None):
        path = self.artifact_path if artifact_path is None else artifact_path
        return TestClient(create_app(path, self.pm25_artifact_path, self.current_pm25_artifact_path, now=now))

    def payload(self, history=None):
        return {
            "prediction_time": self.prediction_time.isoformat(),
            "history": self.history if history is None else history,
        }

    def test_health_reports_loaded_v1_model(self):
        with self.client() as client:
            response = client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "status": "ok",
                "model": "LinearRegression",
                "model_version": "v1",
                "forecast_horizon_hours": 6,
            },
        )

    def test_valid_history_returns_numeric_prediction_and_target_interval(self):
        with self.client() as client:
            response = client.post("/predict", json=self.payload())

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(math.isfinite(body["predicted_pm25"]))
        self.assertEqual(
            datetime.fromisoformat(body["target_interval_start"]),
            self.prediction_time + timedelta(hours=6),
        )
        self.assertEqual(
            datetime.fromisoformat(body["target_interval_end"]),
            self.prediction_time + timedelta(hours=7),
        )
        self.assertEqual(body["forecast_horizon_hours"], 6)
        self.assertEqual(body["unit"], "µg/m³")
        self.assertEqual(body["model_version"], "v1")

    def test_api_prediction_matches_direct_modeling_prediction(self):
        history_frame = pd.DataFrame(self.history)
        prediction_row = pd.DataFrame(
            {"event_time": [self.prediction_time], "pm25": [float("nan")]}
        )
        features = build_v1_features(
            pd.concat([history_frame, prediction_row], ignore_index=True)
        ).iloc[[-1]]
        expected = predict_pm25_t_plus_6(
            self.model,
            self.metadata,
            features,
        )[0]

        with self.client() as client:
            response = client.post("/predict", json=self.payload())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["predicted_pm25"], expected)

    def test_insufficient_history_is_rejected(self):
        with self.client() as client:
            response = client.post(
                "/predict",
                json=self.payload(self.history[1:]),
            )

        self.assertEqual(response.status_code, 422)

    def test_missing_hourly_interval_is_rejected(self):
        history = self.history.copy()
        del history[10]
        history.append(
            {
                "event_time": (
                    self.prediction_time - timedelta(hours=25)
                ).isoformat(),
                "pm25": 1.0,
            }
        )

        with self.client() as client:
            response = client.post("/predict", json=self.payload(history))

        self.assertEqual(response.status_code, 422)

    def test_duplicate_timestamp_is_rejected(self):
        history = self.history.copy()
        history[-1] = history[-2].copy()

        with self.client() as client:
            response = client.post("/predict", json=self.payload(history))

        self.assertEqual(response.status_code, 422)

    def test_future_or_unavailable_observation_is_rejected(self):
        history = self.history.copy()
        history[-1] = {
            "event_time": self.prediction_time.isoformat(),
            "pm25": 999999.0,
        }

        with self.client() as client:
            response = client.post("/predict", json=self.payload(history))

        self.assertEqual(response.status_code, 422)

    def test_prediction_time_must_align_to_hour_boundary(self):
        payload = self.payload()
        payload["prediction_time"] = (
            self.prediction_time + timedelta(minutes=30)
        ).isoformat()

        with self.client() as client:
            response = client.post("/predict", json=payload)

        self.assertEqual(response.status_code, 422)

    def test_non_finite_pm25_is_rejected(self):
        for value in ["NaN", "Infinity", "-Infinity"]:
            with self.subTest(value=value):
                history = [entry.copy() for entry in self.history]
                history[0]["pm25"] = value
                with self.client() as client:
                    response = client.post(
                        "/predict",
                        json=self.payload(history),
                    )
                self.assertEqual(response.status_code, 422)

    def test_contract_exposes_neither_weather_nor_engineered_features(self):
        schema = create_app(self.artifact_path).openapi()
        request_schema = schema["components"]["schemas"]["PredictionRequest"]
        observation_schema = schema["components"]["schemas"]["Pm25Observation"]

        self.assertEqual(
            set(request_schema["properties"]),
            {"prediction_time", "history"},
        )
        self.assertEqual(
            set(observation_schema["properties"]),
            {"event_time", "pm25"},
        )
        self.assertTrue(
            set(V1_FEATURE_COLUMNS).isdisjoint(request_schema["properties"])
        )
        self.assertNotIn("weather", request_schema["properties"])

    def test_raw_row_t_pm25_cannot_be_supplied(self):
        history = self.history + [
            {
                "event_time": self.prediction_time.isoformat(),
                "pm25": 999999.0,
            }
        ]

        with self.client() as client:
            response = client.post("/predict", json=self.payload(history))

        self.assertEqual(response.status_code, 422)

    def test_latest_forecast_uses_latest_contiguous_completed_history(self):
        with self.client() as client:
            response = client.get("/forecast/latest")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(
            datetime.fromisoformat(body["prediction_time"]),
            self.prediction_time,
        )
        self.assertEqual(
            datetime.fromisoformat(body["history_start"]),
            self.prediction_time - timedelta(hours=24),
        )
        self.assertEqual(
            datetime.fromisoformat(body["history_end"]),
            self.prediction_time - timedelta(hours=1),
        )
        self.assertEqual(body["latest_completed_pm25"], 24.0)
        self.assertEqual(body["data_mode"], "historical_artifact")
        self.assertEqual(body["forecast_horizon_hours"], 6)
        self.assertNotIn("weather", body)
        self.assertTrue(set(V1_FEATURE_COLUMNS).isdisjoint(body))

    def test_latest_forecast_matches_existing_prediction_pipeline(self):
        with self.client() as client:
            latest = client.get("/forecast/latest")
            direct = client.post("/predict", json=self.payload())

        self.assertEqual(latest.status_code, 200)
        self.assertEqual(direct.status_code, 200)
        self.assertEqual(
            latest.json()["predicted_pm25"],
            direct.json()["predicted_pm25"],
        )

    def test_latest_forecast_never_uses_row_at_prediction_time(self):
        artifact = json.loads(self.pm25_artifact_path.read_text())
        row_t = artifact["normalized_records"][-1].copy()
        row_t["event_time"] = self.prediction_time.isoformat()
        row_t["period_end_utc"] = (
            self.prediction_time + timedelta(hours=1)
        ).isoformat()
        row_t["value"] = 999999.0
        artifact["normalized_records"].append(row_t)
        self.pm25_artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

        with self.client() as client:
            response = client.get("/forecast/latest")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            datetime.fromisoformat(response.json()["prediction_time"]),
            self.prediction_time,
        )
        self.assertEqual(response.json()["latest_completed_pm25"], 24.0)

    def test_gap_causes_clear_latest_forecast_failure(self):
        self.write_pm25_artifact(self.history[:12] + self.history[13:])

        with self.client() as client:
            response = client.get("/forecast/latest")

        self.assertEqual(response.status_code, 422)
        self.assertIn("no contiguous 24-hour", response.json()["detail"])

    def test_current_forecast_uses_exactly_24_completed_hours_and_existing_pipeline(self):
        self.write_current_pm25_artifact()
        with self.client(now=lambda: self.prediction_time) as client:
            current = client.get("/forecast/current")
            direct = client.post("/predict", json=self.payload())

        self.assertEqual(current.status_code, 200)
        body = current.json()
        self.assertEqual(body["predicted_pm25"], direct.json()["predicted_pm25"])
        self.assertEqual(datetime.fromisoformat(body["prediction_time"]), self.prediction_time)
        self.assertEqual(datetime.fromisoformat(body["history_start"]), self.prediction_time - timedelta(hours=24))
        self.assertEqual(datetime.fromisoformat(body["history_end"]), self.prediction_time - timedelta(hours=1))
        self.assertEqual(body["latest_completed_pm25"], 24.0)
        self.assertEqual(body["sensor_id"], 13502151)
        self.assertEqual(body["data_mode"], "fresh_openaq")
        self.assertEqual(body["freshness_status"], "fresh")
        self.assertEqual(body["age_minutes"], 0.0)
        self.assertNotIn("weather", body)
        self.assertTrue(set(V1_FEATURE_COLUMNS).isdisjoint(body))

    def test_current_forecast_rejects_gap_in_required_window(self):
        self.write_current_pm25_artifact(self.history[:10] + self.history[11:])
        with self.client(now=lambda: self.prediction_time) as client:
            response = client.get("/forecast/current")
        self.assertEqual(response.status_code, 422)
        self.assertIn("contiguous", response.json()["detail"])

    def test_current_forecast_rejects_incomplete_current_interval(self):
        incomplete = self.history[:-1] + [{"event_time": self.prediction_time.isoformat(), "pm25": 999999.0}]
        self.write_current_pm25_artifact(incomplete)
        with self.client(now=lambda: self.prediction_time + timedelta(minutes=30)) as client:
            response = client.get("/forecast/current")
        self.assertEqual(response.status_code, 422)
        self.assertIn("contiguous", response.json()["detail"])

    def test_current_forecast_rejects_conflicting_duplicate_timestamp(self):
        duplicate = self.history + [{**self.history[-1], "pm25": 999.0}]
        self.write_current_pm25_artifact(duplicate)
        with self.client(now=lambda: self.prediction_time) as client:
            response = client.get("/forecast/current")
        self.assertEqual(response.status_code, 422)
        self.assertIn("contiguous", response.json()["detail"])

    def test_current_forecast_identifies_stale_artifact(self):
        self.write_current_pm25_artifact()
        stale_now = self.prediction_time + timedelta(hours=4, minutes=1)
        with self.client(now=lambda: stale_now) as client:
            response = client.get("/forecast/current")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["freshness_status"], "stale")
        self.assertEqual(response.json()["data_mode"], "stale_openaq")
        self.assertEqual(response.json()["age_minutes"], 241.0)

    def test_current_forecast_missing_artifact_is_explicit(self):
        with self.client(now=lambda: self.prediction_time) as client:
            response = client.get("/forecast/current")
        self.assertEqual(response.status_code, 503)
        self.assertIn("fresh PM2.5 artifact", response.json()["detail"])

    def test_historical_forecast_still_works_with_current_route(self):
        with self.client(now=lambda: self.prediction_time) as client:
            response = client.get("/forecast/latest")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data_mode"], "historical_artifact")

    def test_status_reports_healthy_current_forecast_state(self):
        self.write_current_pm25_artifact()
        with self.client(now=lambda: self.prediction_time) as client:
            response = client.get("/status")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["overall_status"], "healthy")
        self.assertEqual(body["api_status"], "ok")
        self.assertTrue(body["model_loaded"])
        self.assertEqual(body["model_type"], "LinearRegression")
        self.assertEqual(body["model_version"], "v1")
        self.assertEqual(body["forecast_horizon_hours"], 6)
        self.assertTrue(body["fresh_artifact_available"])
        self.assertEqual(body["freshness_status"], "fresh")
        self.assertEqual(datetime.fromisoformat(body["source_retrieved_at"]), self.prediction_time)
        self.assertEqual(body["age_minutes"], 0.0)
        self.assertEqual(body["latest_completed_pm25"], 24.0)
        self.assertEqual(datetime.fromisoformat(body["latest_completed_interval_end"]), self.prediction_time)
        self.assertEqual(body["sensor_id"], 13502151)
        self.assertTrue(body["current_forecast_available"])
        self.assertEqual(body["data_mode"], "fresh_openaq")

    def test_status_reports_stale_artifact_as_degraded(self):
        self.write_current_pm25_artifact()
        with self.client(now=lambda: self.prediction_time + timedelta(hours=4, minutes=1)) as client:
            body = client.get("/status").json()
        self.assertEqual(body["overall_status"], "degraded")
        self.assertEqual(body["freshness_status"], "stale")
        self.assertTrue(body["current_forecast_available"])
        self.assertEqual(body["data_mode"], "stale_openaq")

    def test_status_reports_missing_and_corrupt_artifact_without_path_leakage(self):
        cases = ("missing", "corrupt")
        for case in cases:
            with self.subTest(case=case):
                if case == "corrupt":
                    self.current_pm25_artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    self.current_pm25_artifact_path.write_text("not json", encoding="utf-8")
                with self.client(now=lambda: self.prediction_time) as client:
                    response = client.get("/status")
                body = response.json()
                self.assertEqual(response.status_code, 200)
                self.assertEqual(body["overall_status"], "degraded")
                self.assertFalse(body["fresh_artifact_available"])
                self.assertFalse(body["current_forecast_available"])
                self.assertEqual(body["freshness_status"], "unavailable")
                serialized = json.dumps(body)
                self.assertNotIn(str(self.current_pm25_artifact_path), serialized)
                self.assertNotIn("OPENAQ_API_KEY", serialized)
                self.current_pm25_artifact_path.unlink(missing_ok=True)

    def test_status_reports_insufficient_history_as_degraded(self):
        self.write_current_pm25_artifact(self.history[1:])
        with self.client(now=lambda: self.prediction_time) as client:
            body = client.get("/status").json()
        self.assertEqual(body["overall_status"], "degraded")
        self.assertTrue(body["fresh_artifact_available"])
        self.assertFalse(body["current_forecast_available"])
        self.assertEqual(body["freshness_status"], "unavailable")

    def test_status_reports_forecast_failure_without_internal_error(self):
        self.write_current_pm25_artifact()
        with patch("app.main._predict", side_effect=ValueError("sensitive internal failure")):
            with self.client(now=lambda: self.prediction_time) as client:
                body = client.get("/status").json()
        self.assertEqual(body["overall_status"], "degraded")
        self.assertFalse(body["current_forecast_available"])
        self.assertNotIn("sensitive internal failure", json.dumps(body))

    def test_webpage_distinguishes_current_and_historical_forecasts(self):
        with self.client() as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/html"))
        self.assertIn("AirAware VN", response.text)
        self.assertIn("/status", response.text)
        self.assertIn("AirAware status", response.text)
        self.assertIn("OpenAQ data", response.text)
        self.assertIn("Last retrieval", response.text)
        self.assertIn("/forecast/current", response.text)
        self.assertIn("/forecast/latest", response.text)
        self.assertIn("Fresh OpenAQ data", response.text)
        self.assertIn("Historical artifact", response.text)

    def test_missing_or_corrupt_artifact_fails_during_startup(self):
        missing_path = Path(self.directory.name) / "missing.joblib"
        with self.assertRaises(Exception):
            with self.client(missing_path):
                pass

        corrupt_path = Path(self.directory.name) / "corrupt.joblib"
        corrupt_path.write_text("not a joblib artifact", encoding="utf-8")
        with self.assertRaises(Exception):
            with self.client(corrupt_path):
                pass

        incompatible_path = Path(self.directory.name) / "incompatible.joblib"
        incompatible_metadata = self.metadata.copy()
        incompatible_metadata["forecast_horizon_hours"] = 12
        joblib.dump(
            {"model": self.model, "metadata": incompatible_metadata},
            incompatible_path,
        )
        with self.assertRaisesRegex(ValueError, "does not match V1 contract"):
            with self.client(incompatible_path):
                pass


class RefreshLoopTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_loop_survives_unexpected_refresh_failures(self):
        with patch(
            "app.main.asyncio.to_thread",
            side_effect=[RuntimeError("OpenAQ failure"), KeyError("period"), asyncio.CancelledError()],
        ) as to_thread, patch("app.main.asyncio.sleep", return_value=None) as sleep:
            with self.assertRaises(asyncio.CancelledError):
                await main._refresh_loop("secret", Path("current_pm25.json"))

        self.assertEqual(to_thread.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        sleep.assert_called_with(3600)


if __name__ == "__main__":
    unittest.main()
