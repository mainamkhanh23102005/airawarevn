import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()
