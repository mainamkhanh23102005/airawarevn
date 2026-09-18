import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts import start_render
from scripts.start_render import MAX_MODEL_BYTES, provision_model


class Response:
    def __init__(self, content):
        self.content = content
        self.offset = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def read(self, size):
        chunk = self.content[self.offset:self.offset + size]
        self.offset += len(chunk)
        return chunk


class StartRenderTests(unittest.TestCase):
    def test_existing_model_needs_no_remote_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            destination.write_bytes(b"trusted")
            self.assertEqual(provision_model(destination, None, None), destination)
            self.assertEqual(destination.read_bytes(), b"trusted")

    def test_download_requires_matching_sha256(self):
        content = b"trusted model"
        digest = hashlib.sha256(content).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            with patch("urllib.request.urlopen", return_value=Response(content)) as urlopen:
                provision_model(destination, "https://models.example/v1", digest)
            self.assertEqual(destination.read_bytes(), content)
            urlopen.assert_called_once_with("https://models.example/v1", timeout=30)

    def test_failed_verification_leaves_no_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            with patch("urllib.request.urlopen", return_value=Response(b"wrong")):
                with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                    provision_model(destination, "https://models.example/v1", "0" * 64)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_suffix(".joblib.tmp").exists())

    def test_download_size_is_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "model.joblib"
            content = b"x" * (MAX_MODEL_BYTES + 1)
            with patch("urllib.request.urlopen", return_value=Response(content)):
                with self.assertRaisesRegex(RuntimeError, "exceeds 50 MiB"):
                    provision_model(destination, "https://models.example/v1", "0" * 64)
            self.assertFalse(destination.exists())

    def test_main_provisions_both_models_and_exports_both_paths(self):
        v1_content = b"trusted v1"
        multi_horizon_content = b"trusted multi horizon"
        with tempfile.TemporaryDirectory() as directory:
            v1_destination = Path(directory) / "v1.joblib"
            multi_horizon_destination = Path(directory) / "mh.joblib"
            environment = {
                "PORT": "8000",
                "AIRAWARE_MODEL_URL": "https://models.example/v1",
                "AIRAWARE_MODEL_SHA256": hashlib.sha256(v1_content).hexdigest(),
                "AIRAWARE_MULTI_HORIZON_MODEL_URL": "https://models.example/mh",
                "AIRAWARE_MULTI_HORIZON_MODEL_SHA256": hashlib.sha256(multi_horizon_content).hexdigest(),
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "scripts.start_render.DEFAULT_MODEL_PATH", v1_destination
            ), patch(
                "scripts.start_render.DEFAULT_MULTI_HORIZON_MODEL_PATH",
                multi_horizon_destination,
            ), patch(
                "urllib.request.urlopen",
                side_effect=[Response(v1_content), Response(multi_horizon_content)],
            ) as urlopen, patch("os.execv") as execv:
                start_render.main()
                self.assertEqual(os.environ["AIRAWARE_MODEL_PATH"], str(v1_destination))
                self.assertEqual(
                    os.environ["AIRAWARE_MULTI_HORIZON_MODEL_PATH"],
                    str(multi_horizon_destination),
                )
            self.assertEqual(v1_destination.read_bytes(), v1_content)
            self.assertEqual(multi_horizon_destination.read_bytes(), multi_horizon_content)
            self.assertEqual(urlopen.call_count, 2)
            execv.assert_called_once()

    def test_missing_multi_horizon_sha_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            v1_destination = Path(directory) / "v1.joblib"
            v1_destination.write_bytes(b"trusted v1")
            environment = {
                "PORT": "8000",
                "AIRAWARE_MODEL_PATH": str(v1_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_PATH": str(Path(directory) / "mh.joblib"),
                "AIRAWARE_MULTI_HORIZON_MODEL_URL": "https://models.example/mh",
            }
            with patch.dict(os.environ, environment, clear=True), self.assertRaisesRegex(
                RuntimeError, "AIRAWARE_MULTI_HORIZON_MODEL_SHA256"
            ):
                start_render.main()

    def test_bad_multi_horizon_sha_fails_before_download(self):
        with tempfile.TemporaryDirectory() as directory:
            v1_destination = Path(directory) / "v1.joblib"
            v1_destination.write_bytes(b"trusted v1")
            multi_horizon_destination = Path(directory) / "mh.joblib"
            environment = {
                "PORT": "8000",
                "AIRAWARE_MODEL_PATH": str(v1_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_PATH": str(multi_horizon_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_URL": "https://models.example/mh",
                "AIRAWARE_MULTI_HORIZON_MODEL_SHA256": "not-a-sha256",
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "urllib.request.urlopen"
            ) as urlopen:
                with self.assertRaisesRegex(
                    RuntimeError,
                    "AIRAWARE_MULTI_HORIZON_MODEL_SHA256 must be a 64-character hexadecimal digest",
                ):
                    start_render.main()
            urlopen.assert_not_called()
            self.assertFalse(multi_horizon_destination.exists())

    def test_oversized_multi_horizon_artifact_fails_without_installing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            v1_destination = Path(directory) / "v1.joblib"
            v1_destination.write_bytes(b"trusted v1")
            multi_horizon_destination = Path(directory) / "mh.joblib"
            environment = {
                "PORT": "8000",
                "AIRAWARE_MODEL_PATH": str(v1_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_PATH": str(multi_horizon_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_URL": "https://models.example/mh",
                "AIRAWARE_MULTI_HORIZON_MODEL_SHA256": "0" * 64,
            }
            with patch.dict(os.environ, environment, clear=True), patch.object(
                start_render, "MAX_MODEL_BYTES", 4
            ), patch("urllib.request.urlopen", return_value=Response(b"12345")), self.assertRaisesRegex(
                RuntimeError, "exceeds 50 MiB"
            ):
                start_render.main()
            self.assertFalse(multi_horizon_destination.exists())
            self.assertFalse(multi_horizon_destination.with_suffix(".joblib.tmp").exists())

    def test_corrupt_multi_horizon_download_fails_without_installing_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            v1_destination = Path(directory) / "v1.joblib"
            v1_destination.write_bytes(b"trusted v1")
            multi_horizon_destination = Path(directory) / "mh.joblib"
            expected_content = b"trusted multi horizon"
            environment = {
                "PORT": "8000",
                "AIRAWARE_MODEL_PATH": str(v1_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_PATH": str(multi_horizon_destination),
                "AIRAWARE_MULTI_HORIZON_MODEL_URL": "https://models.example/mh",
                "AIRAWARE_MULTI_HORIZON_MODEL_SHA256": hashlib.sha256(expected_content).hexdigest(),
            }
            with patch.dict(os.environ, environment, clear=True), patch(
                "urllib.request.urlopen", return_value=Response(b"corrupt")
            ), self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                start_render.main()
            self.assertFalse(multi_horizon_destination.exists())
            self.assertFalse(multi_horizon_destination.with_suffix(".joblib.tmp").exists())

    def test_render_blueprint_wires_production_runtime(self):
        blueprint = Path("render.yaml").read_text(encoding="utf-8")
        self.assertIn('key: AIRAWARE_REFRESH_ENABLED\n        value: "1"', blueprint)
        self.assertIn(
            "key: AIRAWARE_CURRENT_PM25_ARTIFACT_PATH\n"
            "        value: .artifacts/live/current_pm25.json",
            blueprint,
        )
        self.assertIn('key: AIRAWARE_LEDGER_BACKEND\n        value: libsql', blueprint)
        self.assertIn(
            "key: AIRAWARE_FORECAST_LEDGER_PATH\n"
            "        value: .artifacts/forecast-ledger/forecasts.sqlite3",
            blueprint,
        )
        self.assertIn(
            "key: AIRAWARE_TURSO_DATABASE_URL\n"
            "        value: libsql://airaware-prod-mainamkhanh23102005.aws-ap-northeast-1.turso.io",
            blueprint,
        )
        self.assertIn("key: AIRAWARE_TURSO_AUTH_TOKEN\n        sync: false", blueprint)
        self.assertIn("key: OPENAQ_API_KEY\n        sync: false", blueprint)
        self.assertIn(
            "value: https://github.com/mainamkhanh23102005/airawarevn/releases/download/"
            "model-v1.1.0/airaware_v1.joblib",
            blueprint,
        )
        self.assertIn(
            "value: af27f76aca9dd637814f2a6c83d50ceb50fd1b7309762cfdf6898b2e94cb8605",
            blueprint,
        )
        self.assertIn(
            "value: https://github.com/mainamkhanh23102005/airawarevn/releases/download/"
            "model-mh-v1.0.0/airaware-mh-v1.joblib",
            blueprint,
        )
        self.assertIn(
            "value: 8fb851e64ba51c010d6869ecc0180585f832dae9f65c884a7cc8c018e8b6e505",
            blueprint,
        )


if __name__ == "__main__":
    unittest.main()
