# AirAware VN

AirAware VN is an end-to-end PM2.5 forecasting system for Hanoi. It ingests real hourly observations from OpenAQ, validates temporal and data-quality semantics, builds leakage-safe historical features, and serves a six-hour-ahead forecast through FastAPI and a minimal web interface.

The project demonstrates more than model fitting: it connects reproducible offline evaluation to fresh-data inference, explicit freshness and gap handling, operational status reporting, and local service automation.

## What users see

The web application displays:

- latest completed hourly PM2.5 measurement;
- predicted PM2.5 six hours ahead;
- prediction and target intervals;
- fresh, stale, unavailable, or explicitly historical data state;
- model and operational status.

The current forecast never silently falls back to historical data. Historical output remains available as a clearly labeled offline fallback.

> **Public demo:** a temporary Cloudflare Quick Tunnel may be available during active development. Its random URL is not permanent hosting; stable deployment is planned.

## Architecture

AirAware has separate offline training and live inference paths. The hourly refresh updates data only—it does not retrain or modify the frozen model.

```mermaid
flowchart LR
  subgraph offline["TRAINING / OFFLINE PATH"]
    OH["Stored OpenAQ hourly PM2.5"] --> DQ["Coverage and data-quality validation"]
    DQ --> FB1["Leakage-safe feature builder"]
    FB1 --> EV["Chronological model evaluation"]
    EV --> M["Frozen V1 Linear Regression artifact"]
  end

  subgraph live["LIVE INFERENCE PATH"]
    T["systemd refresh timer<br/>hourly at HH:12"] --> R["scripts.refresh_pm25"]
    OA["OpenAQ API"] --> R
    R --> A[".artifacts/live/current_pm25.json"]
    A --> C["24 completed contiguous hours"]
    C --> FB2["Same V1 feature builder"]
    M --> P["V1 prediction"]
    FB2 --> P
    P --> API["FastAPI<br/>systemd API service"]
    API --> UI["FastAPI-served HTML/JS UI"]
    API --> H["GET /health"]
    API --> S["GET /status"]
    API --> F["GET /forecast/current"]
  end
```

## Data and prediction target

V1 uses hourly PM2.5 means from the configured OpenAQ sensor in Hanoi. Each normalized record preserves sensor identity, event time, provider interval end, value, unit, and source record ID. The live artifact adds an ingestion timestamp to each row, while retrieval time, request metadata, raw payload paths, and payload hashes are stored at artifact/request-provenance level.

The model predicts PM2.5 **six hours ahead**. If `event_time = t` identifies the completed interval `[t, t+1h)`, then at prediction time `t` the latest safe source interval has `event_time = t-1h`. Consequently, `pm25_lag_1h` is the latest safely completed measurement—not the raw PM2.5 value at the prediction row.

Live inference requires exactly 24 contiguous completed hourly intervals. It rejects missing or conflicting hours and never interpolates PM2.5.

## Leakage-safe V1 features

The frozen V1 feature contract contains:

| PM2.5 history | Calendar context (Asia/Ho_Chi_Minh) |
|---|---|
| `pm25_lag_1h` | `hour` |
| `pm25_lag_3h` | `day_of_week` |
| `pm25_lag_6h` | `month` |
| `pm25_lag_12h` | `is_weekend` |
| `pm25_lag_24h` | |
| `pm25_rolling_mean_6h` | |
| `pm25_rolling_mean_12h` | |
| `pm25_rolling_mean_24h` | |

Leakage controls include:

- lag features use positive shifts;
- rolling means first shift PM2.5 by one hour, then aggregate;
- raw row-`t` PM2.5 is excluded from the feature contract;
- weather is excluded from V1;
- train, validation, and test partitions are chronological and never shuffled;
- runtime inference constructs a synthetic prediction row with no PM2.5 value and reuses the training feature builder.

## Modeling and evaluation

After target and feature availability filtering, the modeling dataset contains 7,330 rows:

| Split | Rows | Method |
|---|---:|---|
| Train | 5,131 | First 70% chronologically |
| Validation | 1,099 | Following 15% |
| Test | 1,100 | Final 15%; no shuffle |

Model selection and feature analysis used validation performance. The test partition was treated as the final held-out report rather than choosing whichever model happened to have the lowest test metric.

| Model | Validation MAE | Validation RMSE | Test MAE | Test RMSE |
|---|---:|---:|---:|---:|
| Persistence baseline | 11.1580 | 14.9924 | 11.3754 | 16.0023 |
| **Linear Regression (V1)** | **9.0894** | **12.0908** | 9.6931 | **13.2059** |
| Ridge (`alpha=0.01`) | 9.0895 | **12.0908** | 9.6931 | 13.2058 |
| Random Forest | 10.0154 | 13.7373 | 9.7502 | 13.5275 |
| HistGradientBoosting | 10.0834 | 14.1157 | **9.6853** | 13.5020 |

Linear Regression produced the best validation MAE, essentially tied Ridge without an additional scaling/tuning path, and outperformed the tree models on validation MAE and RMSE. Its test MAE improves on persistence by approximately **14.8%**. HistGradientBoosting's test MAE is 0.0078 lower, but that held-out result was not used to reverse the validation-based selection decision.

### Error analysis

V1 improves average performance but regresses toward the mean:

- low-pollution periods can be overpredicted;
- high-pollution periods can be underpredicted;
- for actual PM2.5 `>=55 µg/m³` (`n=27`), MAE was approximately `34.43 µg/m³` and signed bias was approximately `-34.36 µg/m³`.

This makes extreme pollution spikes the clearest modeling weakness and a priority for future feature and model work.

## Why weather is excluded from V1

The historical Open-Meteo artifact preserved valid-time weather, but not the forecast issue time, model run, or vintage metadata needed to prove that each value was available at the corresponding prediction time. Using those rows as if they were production forecast snapshots would risk point-in-time leakage and offline/online skew.

V1 therefore excludes weather by design. A weather-enabled V2 should ingest production-equivalent forecast snapshots with retrieval and issue timestamps before those features enter training or inference.

## Fresh inference and API

Every hour, the user-level systemd timer runs:

```text
airaware-refresh.timer
  → airaware-refresh.service
  → python -m scripts.refresh_pm25
  → .artifacts/live/current_pm25.json
```

The API loads the saved V1 model once at startup. `/forecast/current` reads the latest local artifact, resolves duplicate-hour semantics, rejects incomplete intervals and gaps, builds the same V1 features, and returns the six-hour forecast. Data older than the centralized four-hour threshold is explicitly marked stale.

| Endpoint | Purpose |
|---|---|
| `GET /` | Web interface |
| `GET /health` | Cheap API/model liveness check |
| `GET /status` | Model, artifact, freshness, and current-forecast state |
| `GET /forecast/current` | Fresh or explicitly stale OpenAQ-based forecast |
| `GET /forecast/latest` | Historical/offline artifact fallback |
| `POST /predict` | Low-level prediction contract for exactly 24 hourly observations |

## Run locally

### 1. Install

```bash
git clone https://github.com/mainamkhanh23102005/airawarevn.git airaware-vn
cd airaware-vn
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-stage0.txt
```

Model and historical artifacts are intentionally stored under ignored `.artifacts/` paths. A working deployment needs the saved V1 model artifact expected by `app.main`, or an `AIRAWARE_MODEL_PATH` override. The repository currently exposes reusable training and serialization functions, and the offline experiment reproduces evaluation, but no supported clean CLI rebuilds the frozen production model artifact from a fresh clone. Producing and promoting that artifact through the existing offline workflow is a known reproducibility gap reserved for a later engineering task.

### 2. Configure OpenAQ securely

Create a user-owned environment file; never place the real key in the repository:

```bash
mkdir -p ~/.config/airaware
chmod 700 ~/.config/airaware
${EDITOR:-vi} ~/.config/airaware/airaware.env
chmod 600 ~/.config/airaware/airaware.env
```

File contents:

```text
OPENAQ_API_KEY=replace-with-your-key
```

For a manual shell refresh, export the same variable without passing it as a command argument, then use module execution:

```bash
set -a
source ~/.config/airaware/airaware.env
set +a
python -m scripts.refresh_pm25
```

Do not invoke `python scripts/refresh_pm25.py`; repository package imports require module execution from the repository root.

### 3. Start the API manually

```bash
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

## User-level systemd operation

Repository-managed templates live in `deploy/systemd/`. Install rendered user units with:

```bash
python -m scripts.install_systemd_user
systemctl --user daemon-reload
systemctl --user enable --now airaware-refresh.timer
systemctl --user enable --now airaware-api.service
```

Components:

- `airaware-refresh.timer`: persistent hourly schedule at approximately `HH:12`;
- `airaware-refresh.service`: oneshot data refresh with restricted write access to `.artifacts/live`;
- `airaware-api.service`: Uvicorn without `--reload`, bound to `127.0.0.1:8000`, restarting on failure after five seconds.

Useful checks:

```bash
systemctl --user status airaware-refresh.timer
systemctl --user status airaware-api.service
journalctl --user -u airaware-refresh.service
journalctl --user -u airaware-api.service
```

The timer refreshes data only. Model retraining is a separate offline workflow.

## Testing

```bash
python -m unittest discover -s tests
python -m compileall -q app scripts tests
```

Current verified state: **123 passing tests**, with external OpenAQ calls mocked in unit tests.

## Repository structure

```text
app/                  FastAPI application and server-served HTML UI
scripts/              OpenAQ ingestion, refresh, data-quality, and setup tools
scripts/modeling/     Frozen V1 features, training, serialization, and prediction
experiments/          Reproducible dataset inspection, baselines, comparisons, and error analysis
tests/                Unit and API integration tests without Internet/systemd dependency
deploy/systemd/       User-level refresh timer/service and API service templates
```

## V1 limitations

- One configured Hanoi PM2.5 sensor; no multi-station or spatial context.
- One six-hour forecast horizon.
- No production-equivalent weather forecast features.
- Extreme PM2.5 spikes remain difficult and are systematically underpredicted.
- Live data uses a local JSON artifact rather than durable shared storage.
- The model does not retrain automatically.
- Cloudflare Quick Tunnel is temporary demo exposure, not permanent hosting or an SLA-backed deployment.

## V2 roadmap

- point-in-time-safe weather forecast snapshots with issue and retrieval timestamps;
- multi-station and spatial features;
- spike-aware features, objectives, and stronger nonlinear models;
- explicit retraining and model-promotion strategy;
- stable cloud deployment and CI/CD;
- richer monitoring, metrics, and UI diagnostics.
