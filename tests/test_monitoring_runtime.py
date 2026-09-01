import unittest
from unittest.mock import patch

from scripts import run_monitoring_cycle


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


if __name__ == "__main__":
    unittest.main()
