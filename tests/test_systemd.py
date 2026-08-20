import tempfile
import unittest
from pathlib import Path

from scripts import install_systemd_user


ROOT = Path(__file__).resolve().parents[1]
SERVICE_TEMPLATE = ROOT / "deploy/systemd/airaware-refresh.service"
TIMER_TEMPLATE = ROOT / "deploy/systemd/airaware-refresh.timer"
API_SERVICE_TEMPLATE = ROOT / "deploy/systemd/airaware-api.service"


class SystemdDeploymentTests(unittest.TestCase):
    def test_service_uses_module_execution_and_user_environment_file(self):
        service = SERVICE_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("WorkingDirectory=@REPO_ROOT@", service)
        self.assertIn("ExecStart=@REPO_ROOT@/.venv/bin/python -m scripts.refresh_pm25", service)
        self.assertIn("EnvironmentFile=%h/.config/airaware/airaware.env", service)
        self.assertNotIn("OPENAQ_API_KEY=", service)
        self.assertNotIn("python scripts/refresh_pm25.py", service)

    def test_timer_runs_hourly_after_top_of_hour_and_is_persistent(self):
        timer = TIMER_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("OnCalendar=*-*-* *:12:00", timer)
        self.assertIn("Persistent=true", timer)
        self.assertIn("Unit=airaware-refresh.service", timer)

    def test_api_service_uses_production_module_invocation_and_restart_policy(self):
        service = API_SERVICE_TEMPLATE.read_text(encoding="utf-8")

        self.assertIn("WorkingDirectory=@REPO_ROOT@", service)
        self.assertIn("ExecStart=@REPO_ROOT@/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000", service)
        self.assertNotIn("--reload", service)
        self.assertIn("Restart=on-failure", service)
        self.assertIn("RestartSec=5s", service)
        self.assertIn("WantedBy=default.target", service)
        self.assertNotIn("OPENAQ_API_KEY=", service)

    def test_installer_renders_repository_path_without_secret_material(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory)
            installed = install_systemd_user.install_units(ROOT, destination)

            service = (destination / "airaware-refresh.service").read_text(encoding="utf-8")
            timer = (destination / "airaware-refresh.timer").read_text(encoding="utf-8")
            api_service = (destination / "airaware-api.service").read_text(encoding="utf-8")

        self.assertEqual(set(installed), {destination / "airaware-refresh.service", destination / "airaware-refresh.timer", destination / "airaware-api.service"})
        self.assertIn(f"WorkingDirectory={ROOT}", service)
        self.assertIn(f"WorkingDirectory={ROOT}", api_service)
        self.assertIn(f"ReadWritePaths={ROOT / '.artifacts/live'}", service)
        self.assertNotIn("@REPO_ROOT@", service + timer + api_service)
        self.assertNotIn("OPENAQ_API_KEY=", service + timer + api_service)

    def test_installer_rejects_repository_paths_with_systemd_newlines(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "newline"):
                install_systemd_user.install_units(Path("/tmp/unsafe\npath"), Path(directory))


if __name__ == "__main__":
    unittest.main()
