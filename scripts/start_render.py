from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import urllib.request

DEFAULT_MODEL_PATH = Path(".artifacts/models/airaware_v1.joblib")
DEFAULT_MULTI_HORIZON_MODEL_PATH = Path(".artifacts/models/airaware-mh-v1.joblib")
MAX_MODEL_BYTES = 50 * 1024 * 1024


def provision_model(
    destination,
    url,
    expected_sha256,
    url_variable="AIRAWARE_MODEL_URL",
    sha256_variable="AIRAWARE_MODEL_SHA256",
):
    if destination.is_file():
        return destination
    if not url or not expected_sha256:
        raise RuntimeError(
            f"model artifact is absent; set {url_variable} and {sha256_variable}"
        )
    if len(expected_sha256) != 64 or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256):
        raise RuntimeError(f"{sha256_variable} must be a 64-character hexadecimal digest")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    digest = hashlib.sha256()
    size = 0
    try:
        with urllib.request.urlopen(url, timeout=30) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                size += len(chunk)
                if size > MAX_MODEL_BYTES:
                    raise RuntimeError("model artifact exceeds 50 MiB limit")
                digest.update(chunk)
                output.write(chunk)
        if digest.hexdigest() != expected_sha256.lower():
            raise RuntimeError("model artifact SHA-256 mismatch")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def main():
    port = os.environ.get("PORT")
    if not port or not port.isdigit() or not 1 <= int(port) <= 65535:
        raise RuntimeError("PORT must be an integer from 1 through 65535")
    destination = Path(os.environ.get("AIRAWARE_MODEL_PATH", DEFAULT_MODEL_PATH))
    multi_horizon_destination = Path(
        os.environ.get(
            "AIRAWARE_MULTI_HORIZON_MODEL_PATH",
            DEFAULT_MULTI_HORIZON_MODEL_PATH,
        )
    )
    provision_model(
        destination,
        os.environ.get("AIRAWARE_MODEL_URL"),
        os.environ.get("AIRAWARE_MODEL_SHA256"),
    )
    provision_model(
        multi_horizon_destination,
        os.environ.get("AIRAWARE_MULTI_HORIZON_MODEL_URL"),
        os.environ.get("AIRAWARE_MULTI_HORIZON_MODEL_SHA256"),
        "AIRAWARE_MULTI_HORIZON_MODEL_URL",
        "AIRAWARE_MULTI_HORIZON_MODEL_SHA256",
    )
    os.environ["AIRAWARE_MODEL_PATH"] = str(destination)
    os.environ["AIRAWARE_MULTI_HORIZON_MODEL_PATH"] = str(
        multi_horizon_destination
    )
    os.execv(
        sys.executable,
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            port,
        ],
    )


if __name__ == "__main__":
    main()
