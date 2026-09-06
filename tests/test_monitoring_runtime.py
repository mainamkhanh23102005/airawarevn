import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from app.forecast_ledger import LedgerDatabaseError
from scripts import run_monitoring_cycle


ICT = ZoneInfo("Asia/Ho_Chi_Minh")


class MonitoringCycleTests(unittest.TestCase):
    def test_cycle_runs_issuer_reconciliation_and_snapshot_worker_once(self):
        issue = object()
        reconciliation = object()
        evaluation = object()
        store = Mock()
        with patch.object(run_monitoring_cycle, "issue_forecast", return_value=issue) as issue_forecast, patch.object(
                run_monitoring_cycle, "run_reconciliation", return_value=reconciliation) as reconcile, patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store) as create_store, patch.object(
                run_monitoring_cycle, "materialize_available_evaluations", return_value=evaluation) as materialize:
            result = run_monitoring_cycle.run_monitoring_cycle("ledger.sqlite3", "raw", "key", "model.joblib", "current.json")
        self.assertEqual(result, (issue, reconciliation, evaluation))
        issue_forecast.assert_called_once()
        reconcile.assert_called_once()
        create_store.assert_called_once_with("ledger.sqlite3")
        store.initialize.assert_called_once()
        store.acquire_monitoring_lease.assert_not_called()
        store.release_monitoring_lease.assert_not_called()
        materialize.assert_called_once_with(store)

    def test_leased_cycle_acquires_before_work_and_releases_afterward(self):
        issue = object()
        reconciliation = object()
        evaluation = object()
        store = Mock()
        store.acquire_monitoring_lease.return_value = True
        store.release_monitoring_lease.return_value = True
        reference = datetime(2026, 1, 2, 3, 15, tzinfo=timezone.utc)

        def issue_after_acquire(*args, **kwargs):
            self.assertTrue(store.acquire_monitoring_lease.called)
            return issue

        with patch.object(run_monitoring_cycle, "issue_forecast", side_effect=issue_after_acquire) as issue_forecast, patch.object(
                run_monitoring_cycle, "run_reconciliation", return_value=reconciliation), patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store) as create_store, patch.object(
                run_monitoring_cycle, "materialize_available_evaluations", return_value=evaluation):
            result = run_monitoring_cycle.run_monitoring_cycle(
                "ledger.sqlite3", "raw", "key", "model.joblib", "current.json",
                now=lambda: reference, lease_owner_id="owner-a", lease_ttl_seconds=120)

        self.assertEqual(result, (issue, reconciliation, evaluation))
        create_store.assert_called_once_with("ledger.sqlite3")
        store.initialize.assert_called_once()
        store.acquire_monitoring_lease.assert_called_once_with(
            run_monitoring_cycle.DEFAULT_MONITORING_LEASE_NAME, "owner-a", reference, 120)
        issue_forecast.assert_called_once()
        store.release_monitoring_lease.assert_called_once_with(
            run_monitoring_cycle.DEFAULT_MONITORING_LEASE_NAME, "owner-a")

    def test_successful_leased_cycle_fails_if_lease_was_lost_before_release(self):
        store = Mock()
        store.acquire_monitoring_lease.return_value = True
        store.release_monitoring_lease.return_value = False
        reference = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with patch.object(run_monitoring_cycle, "issue_forecast", return_value=object()), patch.object(
                run_monitoring_cycle, "run_reconciliation", return_value=object()), patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store), patch.object(
                run_monitoring_cycle, "materialize_available_evaluations", return_value=object()), self.assertRaisesRegex(
                run_monitoring_cycle.MonitoringLeaseUnavailable, "lost production-monitoring lease"):
            run_monitoring_cycle.run_monitoring_cycle(
                "ledger.sqlite3", "raw", "key", lease_owner_id="owner-a", now=lambda: reference)
        store.release_monitoring_lease.assert_called_once_with(
            run_monitoring_cycle.DEFAULT_MONITORING_LEASE_NAME, "owner-a")

    def test_leased_cycle_skips_all_monitoring_work_when_lease_is_held(self):
        store = Mock()
        store.acquire_monitoring_lease.return_value = False
        reference = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with patch.object(run_monitoring_cycle, "issue_forecast") as issue_forecast, patch.object(
                run_monitoring_cycle, "run_reconciliation") as reconciliation, patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store), patch.object(
                run_monitoring_cycle, "materialize_available_evaluations") as materialize, self.assertRaisesRegex(
                run_monitoring_cycle.MonitoringLeaseUnavailable, "already held"):
            run_monitoring_cycle.run_monitoring_cycle(
                "ledger.sqlite3", "raw", "key", lease_owner_id="owner-b", now=lambda: reference)
        issue_forecast.assert_not_called()
        reconciliation.assert_not_called()
        materialize.assert_not_called()
        store.release_monitoring_lease.assert_not_called()

    def test_leased_cycle_releases_after_monitoring_exception(self):
        store = Mock()
        store.acquire_monitoring_lease.return_value = True
        store.release_monitoring_lease.return_value = False
        reference = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with patch.object(run_monitoring_cycle, "issue_forecast", side_effect=RuntimeError("monitoring failed")), patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store), self.assertRaisesRegex(
                RuntimeError, "monitoring failed"):
            run_monitoring_cycle.run_monitoring_cycle(
                "ledger.sqlite3", "raw", "key", lease_owner_id="owner-a", now=lambda: reference)
        store.release_monitoring_lease.assert_called_once_with(
            run_monitoring_cycle.DEFAULT_MONITORING_LEASE_NAME, "owner-a")

    def test_failed_release_does_not_mask_monitoring_exception(self):
        store = Mock()
        store.acquire_monitoring_lease.return_value = True
        store.release_monitoring_lease.side_effect = LedgerDatabaseError("release failed")
        reference = datetime(2026, 1, 2, tzinfo=timezone.utc)
        with patch.object(run_monitoring_cycle, "issue_forecast", side_effect=ValueError("original failure")), patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store), self.assertRaisesRegex(
                ValueError, "original failure"):
            run_monitoring_cycle.run_monitoring_cycle(
                "ledger.sqlite3", "raw", "key", lease_owner_id="owner-a", now=lambda: reference)
        store.release_monitoring_lease.assert_called_once()

    def test_monitoring_cycle_owns_ledger_initialization(self):
        store = Mock()
        with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                run_monitoring_cycle, "run_reconciliation"), patch.object(
                run_monitoring_cycle, "create_forecast_store", return_value=store), patch.object(
                run_monitoring_cycle, "materialize_available_evaluations"):
            run_monitoring_cycle.run_monitoring_cycle("ledger.sqlite3", "raw", "key", "model.joblib", "current.json")
        store.initialize.assert_called_once()

    def test_missing_model_materializes_without_consumer_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.joblib"
            evaluation = object()
            store = Mock()
            with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                    run_monitoring_cycle, "run_reconciliation"), patch.object(
                    run_monitoring_cycle, "create_forecast_store", return_value=store), patch.object(
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
            store = Mock()
            with patch.object(run_monitoring_cycle, "issue_forecast"), patch.object(
                    run_monitoring_cycle, "run_reconciliation"), patch.object(
                    run_monitoring_cycle, "create_forecast_store", return_value=store), patch.object(
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
