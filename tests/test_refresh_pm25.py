import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

from scripts import refresh_pm25


UTC = timezone.utc


def hour_row(start, value=12.5, row_id=1):
    end = start + timedelta(hours=1)
    return {
        "id": row_id,
        "value": value,
        "parameter": {"id": 2, "name": "pm25", "units": "µg/m³"},
        "period": {
            "datetimeFrom": {"utc": start.isoformat().replace("+00:00", "Z")},
            "datetimeTo": {"utc": end.isoformat().replace("+00:00", "Z")},
        },
    }


class RefreshPm25Tests(unittest.TestCase):
    def test_refresh_writes_separate_fresh_artifact_with_provenance(self):
        now = datetime(2026, 8, 20, 12, 34, tzinfo=UTC)
        rows = [hour_row(now.replace(minute=0) - timedelta(hours=hour), row_id=hour) for hour in range(1, 49)]
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(200, json={"results": rows, "meta": {"found": len(rows)}})

        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=httpx.MockTransport(handler)) as client:
            root = Path(directory)
            artifact_path = refresh_pm25.refresh_current_pm25(client, "secret", 13502151, root, now=lambda: now)
            artifact = json.loads(artifact_path.read_text())

        self.assertEqual(artifact_path, root / "current_pm25.json")
        self.assertEqual(artifact["sensor_id"], 13502151)
        self.assertEqual(artifact["retrieved_at"], now.isoformat())
        self.assertEqual(len(artifact["normalized_records"]), 48)
        self.assertEqual(artifact["source"]["endpoint"], "/sensors/13502151/hours")
        self.assertTrue(artifact["provenance"])
        self.assertNotIn("secret", json.dumps(artifact))
        self.assertEqual(requests[0].url.params["datetime_from"], (now.replace(minute=0) - timedelta(hours=72)).isoformat())
        self.assertEqual(requests[0].url.params["datetime_to"], now.isoformat())

    def test_refresh_network_is_mocked_and_current_interval_is_preserved_for_api_filtering(self):
        now = datetime(2026, 8, 20, 12, 30, tzinfo=UTC)
        rows = [hour_row(now.replace(minute=0), row_id=1)]
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json={"results": rows}))
        with tempfile.TemporaryDirectory() as directory, httpx.Client(transport=transport) as client:
            path = refresh_pm25.refresh_current_pm25(client, "key", 13502151, Path(directory), now=lambda: now)
            record = json.loads(path.read_text())["normalized_records"][0]
        self.assertGreater(datetime.fromisoformat(record["period_end_utc"]), now)

    def test_failed_refresh_preserves_existing_artifact(self):
        now = datetime(2026, 8, 20, 12, 30, tzinfo=UTC)
        with tempfile.TemporaryDirectory() as directory:
            artifact_directory = Path(directory)
            artifact_path = artifact_directory / "current_pm25.json"
            artifact_path.write_bytes(b"known-good")
            client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(401)))
            with client, self.assertRaisesRegex(Exception, "HTTP 401"):
                refresh_pm25.refresh_current_pm25(client, "key", 13502151, artifact_directory, now=lambda: now)
            self.assertEqual(artifact_path.read_bytes(), b"known-good")


if __name__ == "__main__":
    unittest.main()
