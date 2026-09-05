import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from scripts import run_monitoring_cycle


ICT = ZoneInfo("Asia/Ho_Chi_Minh")


class MonitoringCycleTests(unittest.TestCase):
    def test_cycle_runs_issuer_reconciliation_and_snapshot_worker_once(self):
        issue = object()
        reconciliation = object()
        evaluation = object()
        with patch.object(run_monitoring_cycle, "issue_forecast", return_value=issue) as issue_forecast, patch.object(
                run_monitoring_cycle, "run_reconciliation", return_value=reconciliation) as reconcile, patch.object(
                run_monitoring_cycle.SQLiteForecastStore, "initialize") as initialize, patch.object(
                run_monitoring_cycle, "materialize_available_evaluations", return_value=evaluation) as materialize:
            result = run_monitoring_cycle.run_monitoring_cycle("ledger.sqlite3", "raw", "key", "model.joblib", "current.json")
        self.assertEqual(result, (issue, reconciliation, evaluation))
        issue_forecast.assert_called_once()
        reconcile.assert_called_once()
        initialize.assert_called_once()
        materialize.assert_called_once()

    def test_monitoring_cycle_owns_ledger_initialization(self):
        with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                run_monitoring_cycle, "run_reconciliation"), patch.object(
                run_monitoring_cycle.SQLiteForecastStore, "initialize") as initialize, patch.object(
                run_monitoring_cycle, "materialize_available_evaluations"):
            run_monitoring_cycle.run_monitoring_cycle("ledger.sqlite3", "raw", "key", "model.joblib", "current.json")
        initialize.assert_called_once()

    def test_missing_model_materializes_without_consumer_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.joblib"
            evaluation = object()
            with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                    run_monitoring_cycle, "run_reconciliation"), patch.object(
                    run_monitoring_cycle.SQLiteForecastStore, "initialize"), patch.object(
                    run_monitoring_cycle, "materialize_available_evaluations", return_value=evaluation) as materialize:
                result = run_monitoring_cycle.run_monitoring_cycle("ledger.sqlite3", "raw", "key", missing, "current.json")
            self.assertIs(result[2], evaluation)
            materialize.assert_called_once()
            call = materialize.call_args
            self.assertEqual(len(call.args), 1)
            self.assertEqual(call.kwargs, {})

    def test_existing_model_passes_exact_configured_cohort_and_reference_now(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "v1.joblib"
            model_path.write_bytes(b"model-bytes")
            reference = datetime(2026, 1, 2, 3, 15, 0, tzinfo=ICT).astimezone(timezone.utc)
            evaluation = object()
            with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                    run_monitoring_cycle, "run_reconciliation"), patch.object(
                    run_monitoring_cycle.SQLiteForecastStore, "initialize"), patch.object(
                    run_monitoring_cycle, "sha256_file", return_value="c" * 64) as sha, patch.object(
                    run_monitoring_cycle, "feature_schema_sha256", return_value="d" * 64) as schema, patch.object(
                    run_monitoring_cycle, "materialize_available_evaluations", return_value=evaluation) as materialize:
                result = run_monitoring_cycle.run_monitoring_cycle(
                    "ledger.sqlite3", "raw", "key", model_path, "current.json",
                    now=lambda: reference)
            self.assertIs(result[2], evaluation)
            sha.assert_called_once_with(model_path)
            schema.assert_called_once()
            materialize.assert_called_once()
            _, kwargs = materialize.call_args
            expected_cohort = {
                "model_version": run_monitoring_cycle.MODEL_VERSION,
                "model_artifact_sha256": "c" * 64,
                "feature_schema_sha256": "d" * 64,
                "sensor_id": run_monitoring_cycle.TARGET_SENSOR_ID,
                "evaluation_policy_version": 1,
                "forecast_horizon_hours": run_monitoring_cycle.FORECAST_HORIZON_HOURS,
            }
            self.assertEqual(kwargs["consumer_cohort"], expected_cohort)
            self.assertEqual(kwargs["now"], reference)
            self.assertEqual(kwargs["now"].tzinfo, timezone.utc)

    def test_before_gate_now_suppresses_consumer_publication_via_materializer(self):
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "ledger.sqlite3"
            model_path = Path(directory) / "v1.joblib"
            model_path.write_bytes(b"model-bytes")
            before_gate = datetime(2026, 1, 2, 3, 14, 59, tzinfo=ICT)
            issue = object()
            reconciliation = object()
            with patch.object(run_monitoring_cycle, "issue_forecast", return_value=issue), patch.object(
                    run_monitoring_cycle, "run_reconciliation", return_value=reconciliation):
                result = run_monitoring_cycle.run_monitoring_cycle(
                    database, "raw", "key", model_path, "current.json", now=lambda: before_gate)
            self.assertIs(result[0], issue)
            self.assertIs(result[1], reconciliation)
            self.assertIsNone(result[2].consumer_publication)


if __name__ == "__main__":
    unittest.main()
