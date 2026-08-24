import argparse

from app.forecast_ledger import issue_forecast


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--current-pm25", required=True)
    arguments = parser.parse_args()
    result = issue_forecast(arguments.database, arguments.model, arguments.current_pm25)
    print(result.outcome)
    return 0 if result.outcome in {"issued", "already_exists", "not_eligible"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
