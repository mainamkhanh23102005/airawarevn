# AirAware VN M1 Forecast Ledger

## Purpose

M1 adds a traffic-independent foundation for recording scheduled production forecasts. It provides deterministic forecast identity, immutable SQLite development storage, a directly callable Forecast Issuer, and an opt-in `/forecast/current` read path. It does not reconcile observations, calculate errors, activate Render ledgering, or select a production scheduler/database.

## Logical forecast identity

Natural identity contains exactly:

```text
sensor_id
prediction_time
target_interval_start
target_interval_end
forecast_horizon_hours
model_version
model_artifact_sha256
feature_schema_sha256
```

SQLite enforces a unique constraint across these fields. `forecast_id` is also the table primary key.

## Canonical serialization and UUID

Identity serialization version is `1`. Namespace is:

```text
e60f879b-90cf-51a4-8936-aad3c962371c
```

Canonical identity JSON adds `identity_serialization_version`, NFC-normalizes strings, omits nulls, sorts object keys, uses UTF-8 and `ensure_ascii=False`, and uses compact separators without whitespace or a trailing newline. Identity timestamps are aware UTC, hour-aligned, have no fractional seconds, and serialize as `YYYY-MM-DDTHH:MM:SSZ`. SHA-256 values must be lowercase 64-character hexadecimal.

`forecast_id` is lowercase hyphenated UUIDv5 over canonical identity JSON text.

## Feature-schema hash

Feature-schema serialization version is `1`. Canonical fields are:

```text
calendar_timezone
feature_columns
feature_configuration
feature_schema_serialization_version
forecast_horizon_hours
raw_pm25_is_feature
target_column
weather_is_feature
```

Production A2 uses `Asia/Ho_Chi_Minh`, exact ordered `V1_FEATURE_COLUMNS`, configuration `A2`, horizon `6`, target `target_pm25_t_plus_6`, and false raw-PM2.5/weather flags. Computed hash:

```text
a6b1307497011bd8f54dae9a6b198938de2175d5c2843445a3c474a2c0447049
```

Model artifact SHA-256 is computed from exact loaded `.joblib` bytes, not metadata or a hard-coded production digest.

## SQLite schema

Schema version uses SQLite `PRAGMA user_version = 1`.

Table `forecasts` contains:

```text
forecast_id TEXT PRIMARY KEY
sensor_id INTEGER NOT NULL
prediction_time TEXT NOT NULL
target_interval_start TEXT NOT NULL
target_interval_end TEXT NOT NULL
forecast_horizon_hours INTEGER NOT NULL
model_version TEXT NOT NULL
model_artifact_sha256 TEXT NOT NULL
feature_schema_sha256 TEXT NOT NULL
predicted_pm25 REAL NOT NULL
persistence_prediction REAL NOT NULL
unit TEXT NOT NULL
artifact_version INTEGER NOT NULL
feature_configuration TEXT NOT NULL
source_retrieved_at TEXT NOT NULL
input_data_cutoff TEXT NOT NULL
history_start TEXT NOT NULL
history_end TEXT NOT NULL
data_mode_at_issue TEXT NOT NULL
freshness_status_at_issue TEXT NOT NULL
source_age_minutes_at_issue REAL NOT NULL
issuance_mode TEXT NOT NULL
issued_at TEXT NOT NULL
```

A database `UNIQUE` constraint covers all eight natural identity columns. Times are canonical UTC strings. Parent directories are created explicitly. Each operation opens and closes its own connection. Writes use `BEGIN IMMEDIATE`; failed transactions roll back without partial rows.

## Idempotency and immutability

First insert returns `inserted`. Exact repeated content returns stored content with `already_exists`. Same natural identity with changed immutable content raises `ForecastIntegrityError`. No update or delete API exists. Concurrent duplicate insertion leaves one row.

## Forecast Issuer

Direct invocation:

```bash
.venv/bin/python -m scripts.issue_forecast \
  --database .artifacts/forecast-ledger/forecasts.sqlite3 \
  --model .artifacts/models/airaware_v1.joblib \
  --current-pm25 .artifacts/live/current_pm25.json
```

Issuer:

1. initializes SQLite;
2. loads and validates V1 model artifact;
3. hashes exact model bytes;
4. loads current OpenAQ artifact;
5. selects latest eligible completed prediction hour;
6. runs existing A2 V1 prediction pipeline;
7. uses latest safely completed PM2.5 as persistence prediction;
8. constructs scheduled immutable record;
9. inserts idempotently.

Machine outcomes:

```text
issued
already_exists
not_eligible
failed
```

CLI exits zero for all non-failure outcomes and one for `failed`. No HTTP request is involved.

## SQLite development use

Database files are ignored through `*.sqlite3` and `.artifacts/forecast-ledger/`. Use that ignored artifact path for local development.

Safe inspection:

```bash
sqlite3 -readonly .artifacts/forecast-ledger/forecasts.sqlite3 \
  'SELECT forecast_id, prediction_time, predicted_pm25, persistence_prediction, issued_at FROM forecasts ORDER BY issued_at DESC LIMIT 5;'
```

Do not edit rows manually. Recreate a disposable development ledger instead.

## Ledger-disabled compatibility mode

Default remains ledger-disabled. Without `AIRAWARE_FORECAST_LEDGER_PATH`, `/forecast/current` preserves V1 behavior: it reads current OpenAQ data and computes a request-time forecast. `/status`, `/forecast/latest`, and `/predict` remain unchanged and never write forecasts.

## Ledger-enabled read mode

Set a local ledger path before API startup:

```bash
export AIRAWARE_FORECAST_LEDGER_PATH=.artifacts/forecast-ledger/forecasts.sqlite3
.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

In this mode, `/forecast/current` reads latest successfully issued row. It never issues or inserts a forecast. Response schema remains V1-compatible. `age_minutes`, `freshness_status`, and `data_mode` are computed for request time; immutable issuance metadata remains unchanged. Empty ledger or internal database errors return sanitized source-unavailable responses.

## Production activation gates

```text
Render production ledger:
NOT ACTIVATED

Canonical production scheduler:
NOT SELECTED

Durable production storage:
NOT SELECTED
```

`render.yaml` is unchanged. SQLite is development persistence only.

## Known limitations

- single configured sensor;
- one direct `+6h` forecast;
- SQLite is not selected production storage;
- no production scheduler is selected;
- no opportunistic request-time recovery;
- no delayed truth reconciliation or scoring;
- no production artifact bytes in repository, so exact production artifact SHA test skips when artifact is absent;
- local `.venv` uses Python 3.14.4 while CI/Render use Python 3.11.

## Verification commands

```bash
.venv/bin/python -m unittest tests.test_forecast_ledger tests.test_api
.venv/bin/python -m unittest discover -s tests
.venv/bin/python -m compileall -q app scripts tests
git diff --check
git status
git diff --stat
```
