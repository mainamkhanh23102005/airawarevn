import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.forecast_ledger import create_forecast_store
from app.ground_truth_reconciler import GroundTruthReconciler, TruthPolicy
from scripts.openaq import OpenAQObservationSource


def run_reconciliation(database, raw_directory, api_key, limit=None, now=lambda: datetime.now(timezone.utc)):
    store = create_forecast_store(database)
    store.initialize()
    with httpx.Client(timeout=30) as client:
        source = OpenAQObservationSource(client, api_key, raw_directory, now=now, evidence_store=store)
        return GroundTruthReconciler(store, now=now, policy=TruthPolicy(1, 120), source=source, limit=limit).run()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--raw-directory", required=True, type=Path)
    parser.add_argument("--policy-version", type=int, default=1, choices=(1,))
    parser.add_argument("--limit", type=int)
    arguments = parser.parse_args(argv)
    api_key = os.environ.get("OPENAQ_API_KEY")
    if not api_key:
        parser.error("OPENAQ_API_KEY is required")
    result = run_reconciliation(arguments.database, arguments.raw_directory, api_key, arguments.limit)
    payload = {"counts": result.counts, "results": [
        {"forecast_id": item.forecast_id, "status": item.status, "reason_code": item.reason_code}
        for item in result.results
    ]}
    print(json.dumps(payload, sort_keys=True, separators=(",", ":")))
    return 1 if any(item.status == "failed" for item in result.results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
