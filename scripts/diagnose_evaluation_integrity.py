import json
import os

from app.forecast_ledger import create_forecast_store


def main():
    database = os.environ.get("AIRAWARE_FORECAST_LEDGER_PATH")
    if not database:
        raise SystemExit("AIRAWARE_FORECAST_LEDGER_PATH is required")
    store = create_forecast_store(database, read_only=True)
    store.validate_existing()
    evaluation_reports = store.invalid_evaluation_rows()
    snapshot_reports = store.snapshot_metric_recomputation_diagnostics()
    print(json.dumps({"evaluation_count": len(evaluation_reports), "evaluation_rows": evaluation_reports,
        "snapshot_count": len(snapshot_reports), "snapshots": snapshot_reports}, sort_keys=True, separators=(",", ":")))
    return 1 if evaluation_reports or snapshot_reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
