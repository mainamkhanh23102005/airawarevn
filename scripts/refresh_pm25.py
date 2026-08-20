from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path

import httpx

from scripts import openaq


UTC = timezone.utc
DEFAULT_SENSOR_ID = 13502151
DEFAULT_HISTORY_HOURS = 72
DEFAULT_ARTIFACT_DIRECTORY = Path(".artifacts/live")


def refresh_current_pm25(client, api_key, sensor_id, artifact_directory, history_hours=DEFAULT_HISTORY_HOURS, now=lambda: datetime.now(UTC)):
    retrieved_at = now().astimezone(UTC)
    end = retrieved_at
    start = retrieved_at.replace(minute=0, second=0, microsecond=0) - timedelta(hours=history_hours)
    raw_directory = artifact_directory / "raw" / "openaq"
    records, provenance = openaq.fetch_hours(client, api_key, sensor_id, start, end, raw_directory, now=lambda: retrieved_at)
    artifact = {
        "artifact_version": 1,
        "sensor_id": sensor_id,
        "retrieved_at": retrieved_at.isoformat(),
        "source": {
            "provider": "OpenAQ",
            "endpoint": f"/sensors/{sensor_id}/hours",
            "datetime_from": start.isoformat(),
            "datetime_to": end.isoformat(),
        },
        "provenance": provenance,
        "normalized_records": [
            {
                **record,
                "event_time": record["event_time"].isoformat(),
                "period_end_utc": record["period_end_utc"].isoformat(),
                "ingested_at": retrieved_at.isoformat(),
            }
            for record in records
        ],
    }
    artifact_directory.mkdir(parents=True, exist_ok=True)
    path = artifact_directory / "current_pm25.json"
    temporary_path = artifact_directory / "current_pm25.json.tmp"
    temporary_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--sensor-id", type=int, default=DEFAULT_SENSOR_ID)
    parser.add_argument("--history-hours", type=int, default=DEFAULT_HISTORY_HOURS)
    parser.add_argument("--artifact-directory", type=Path, default=DEFAULT_ARTIFACT_DIRECTORY)
    args = parser.parse_args(argv)
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        parser.error("OPENAQ_API_KEY is required")
    with httpx.Client(timeout=30) as client:
        path = refresh_current_pm25(client, api_key, args.sensor_id, args.artifact_directory, args.history_hours)
    print(path)


if __name__ == "__main__":
    main()
