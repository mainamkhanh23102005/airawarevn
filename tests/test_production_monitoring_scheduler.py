import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github/workflows/production-monitoring.yml"
MODEL_URL = (
    "https://github.com/mainamkhanh23102005/airawarevn/releases/download/"
    "model-v1.1.0/airaware_v1.joblib"
)
MODEL_SHA256 = "af27f76aca9dd637814f2a6c83d50ceb50fd1b7309762cfdf6898b2e94cb8605"


class ProductionMonitoringSchedulerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workflow = WORKFLOW.read_text(encoding="utf-8")

    def test_runtime_pins_and_python_match_across_ci_monitoring_and_render(self):
        expected = ["fastapi==0.141.1", "httpx==0.28.1", "joblib==1.6.0", "libsql==0.1.11",
            "numpy==2.5.3", "pandas==3.0.5", "scikit-learn==1.9.1", "tzdata==2026.3", "uvicorn==0.52.4"]
        self.assertEqual((ROOT / "requirements-stage0.txt").read_text().splitlines(), expected)
        ci = (ROOT / ".github/workflows/ci.yml").read_text()
        render = (ROOT / "render.yaml").read_text()
        for workflow in (ci, self.workflow):
            self.assertIn('python-version: "3.14.4"', workflow)
            self.assertIn("python -m pip install -r requirements-stage0.txt", workflow)
        self.assertIn("buildCommand: pip install -r requirements-stage0.txt", render)
        self.assertIn("value: 3.14.4", render)

    def test_triggers_only_hourly_schedule_and_manual_dispatch(self):
        self.assertIn('cron: "15 * * * *"', self.workflow)
        self.assertEqual(self.workflow.count("cron:"), 1)
        self.assertRegex(self.workflow, r"(?m)^\s+workflow_dispatch:\s*$")
        self.assertNotRegex(self.workflow, r"(?m)^\s+push:\s*$")
        self.assertNotRegex(self.workflow, r"(?m)^\s+pull_request:\s*$")

    def test_runner_permissions_concurrency_timeout_and_main_guard_are_pinned(self):
        self.assertIn("runs-on: ubuntu-latest", self.workflow)
        self.assertIn('python-version: "3.14.4"', self.workflow)
        self.assertRegex(self.workflow, r"(?ms)^permissions:\s*\n\s+contents: read\s*$")
        self.assertNotRegex(self.workflow, r"(?m)^\s+[A-Za-z-]+:\s+write\s*$")
        self.assertIn("group: airaware-production-monitoring", self.workflow)
        self.assertIn("cancel-in-progress: false", self.workflow)
        self.assertIn("timeout-minutes: 20", self.workflow)
        self.assertIn("if: github.ref == 'refs/heads/main'", self.workflow)

    def test_canonical_model_is_pinned_and_provisioned_before_monitoring(self):
        self.assertIn(f"AIRAWARE_MODEL_URL: {MODEL_URL}", self.workflow)
        self.assertIn(f"AIRAWARE_MODEL_SHA256: {MODEL_SHA256}", self.workflow)
        self.assertIn("from scripts.start_render import provision_model", self.workflow)
        self.assertIn("provision_model(", self.workflow)
        self.assertLess(
            self.workflow.index("provision_model("),
            self.workflow.index("python -m scripts.run_monitoring_cycle"),
        )
        self.assertNotIn("actions/cache", self.workflow)
        self.assertNotRegex(self.workflow, r"(?m)^\s+cache:\s*")

    def test_refresh_is_explicit_and_runs_before_monitoring_cycle(self):
        refresh = "python -m scripts.refresh_pm25"
        monitoring = "python -m scripts.run_monitoring_cycle"
        self.assertIn(refresh, self.workflow)
        self.assertIn("--sensor-id 13502151", self.workflow)
        self.assertIn("--history-hours 72", self.workflow)
        self.assertIn("--artifact-directory .artifacts/live", self.workflow)
        self.assertLess(self.workflow.index(refresh), self.workflow.index(monitoring))

    def test_production_environment_and_secret_references_are_wired(self):
        expected = {
            "AIRAWARE_LEDGER_BACKEND": "libsql",
            "AIRAWARE_FORECAST_LEDGER_PATH": ".artifacts/forecast-ledger/forecasts.sqlite3",
            "AIRAWARE_TURSO_DATABASE_URL": "libsql://airaware-prod-mainamkhanh23102005.aws-ap-northeast-1.turso.io",
            "AIRAWARE_MODEL_PATH": ".artifacts/models/airaware_v1.joblib",
            "AIRAWARE_CURRENT_PM25_ARTIFACT_PATH": ".artifacts/live/current_pm25.json",
            "AIRAWARE_RECONCILIATION_RAW_DIRECTORY": ".artifacts/monitoring/reconciliation_raw",
            "AIRAWARE_MONITORING_LEASE_OWNER_ID": "github-${{ github.run_id }}",
        }
        for name, value in expected.items():
            self.assertIn(f"{name}: {value}", self.workflow)

        self.assertIn("OPENAQ_API_KEY: ${{ secrets.OPENAQ_API_KEY }}", self.workflow)
        self.assertIn(
            "AIRAWARE_TURSO_AUTH_TOKEN: ${{ secrets.AIRAWARE_TURSO_AUTH_TOKEN }}",
            self.workflow,
        )
        self.assertNotIn("github.run_attempt", self.workflow)

    def test_workflow_does_not_publish_artifacts_retry_or_embed_credentials(self):
        self.assertNotIn("actions/upload-artifact", self.workflow)
        self.assertNotIn("upload-artifact", self.workflow)
        self.assertNotIn("continue-on-error", self.workflow)
        self.assertEqual(self.workflow.count("python -m scripts.run_monitoring_cycle"), 1)
        credential_lines = [
            line.strip()
            for line in self.workflow.splitlines()
            if line.strip().startswith(("OPENAQ_API_KEY:", "AIRAWARE_TURSO_AUTH_TOKEN:"))
        ]
        self.assertEqual(len(credential_lines), 3)
        self.assertTrue(all("secrets." in line for line in credential_lines))

if __name__ == "__main__":
    unittest.main()
