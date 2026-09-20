import json
import os

from app.forecast_ledger import create_forecast_store


def main():
    database = os.environ.get("AIRAWARE_FORECAST_LEDGER_PATH")
    if not database:
        raise SystemExit("AIRAWARE_FORECAST_LEDGER_PATH is required")
    store = create_forecast_store(database, read_only=True)
    store.validate_existing()
    reports = store.invalid_evaluation_rows()
    print(json.dumps({"count": len(reports), "rows": reports}, sort_keys=True, separators=(",", ":")))
    return 1 if reports else 0


if __name__ == "__main__":
    raise SystemExit(main())
