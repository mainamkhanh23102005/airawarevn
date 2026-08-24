# AirAware VN — V1 to V2 M1 Engineering History

## 1. Project purpose

AirAware VN is an end-to-end PM2.5 forecasting and air-quality intelligence project focused on a practical, technically defensible AI Engineering portfolio system. Its progression is:

```text
data
→ leakage-safe ML
→ deployed V1 forecast
→ production evidence loop
→ Air Quality Intelligence V2
```

V1 established a real data, modeling, deployment, and operational path. V2 begins by making production model behavior measurable before expanding model or product complexity.

## 2. V1 data engineering

V1 uses hourly PM2.5 means from OpenAQ sensor `13502151` in Hanoi. Normalized observations preserve sensor identity, UTC event time, provider interval end, value, unit, and source record ID. Request provenance records retrieval time, request metadata, raw payload location, and payload SHA-256.

Modeling uses a complete hourly UTC grid. Hanoi calendar features use `Asia/Ho_Chi_Minh`. Naive timestamps are rejected in normalized training input, and expected interval bounds are timezone-aware and hour-aligned.

Missing PM2.5 remains missing. V1 does not interpolate or impute PM2.5. Modeling rows without complete required features or targets are excluded. Duplicate records with equivalent valid values resolve to one value; conflicting, invalid, or ambiguous records remain unusable rather than being silently selected. Live inference similarly rejects missing, conflicting, incomplete, or non-contiguous hourly history.

These rules support point-in-time reasoning: inputs must represent information safely available when a prediction is issued.

## 3. V1 modeling

V1 directly predicts PM2.5 at `+6h` using target `target_pm25_t_plus_6`. Persistence, represented by the latest safely completed PM2.5 value, is the required baseline. Evaluation uses chronological expanding-window validation rather than random splitting. July 2026 remained isolated until model and feature selection were frozen.

Production V1 uses `sklearn.linear_model.LinearRegression` with feature configuration A2. Exact ordered feature contract:

```text
pm25_lag_1h
pm25_lag_3h
pm25_lag_6h
pm25_lag_12h
pm25_lag_24h
pm25_rolling_mean_6h
pm25_rolling_mean_12h
pm25_rolling_mean_24h
hour
day_of_week
month
is_weekend
pm25_lag_2h
pm25_lag_4h
pm25_lag_8h
pm25_lag_18h
```

Raw PM2.5 from the current prediction row is excluded. Weather is excluded. Runtime inference creates a synthetic prediction row without PM2.5 and applies the same feature builder used for training.

Repository-reported evaluation results, not newly reproduced during this documentation step, are:

- February–June 2026: 3,056 out-of-fold validation predictions;
- validation persistence MAE/RMSE: `11.3073` / `16.1619`;
- validation A2 LinearRegression MAE/RMSE: `9.8558` / `14.0273`;
- July 2026 final test: 613 shared timestamps;
- pre-July fit: 6,711 eligible rows;
- July A2 MAE/RMSE: `8.265` / `10.468`;
- July persistence MAE/RMSE: `9.840` / `12.530`;
- unchanged production specification retrained on 7,330 eligible rows after final evaluation.

LinearRegression was retained because it produced measurable gains over persistence under leakage-safe evaluation while remaining explainable, reproducible, and operationally small. Additional complexity was not added for appearance. Known underprediction during elevated pollution remains an explicit research problem rather than justification for unsupported complexity.

## 4. Leakage-safety decisions

V1 freezes these safeguards:

- lag features use positive shifts;
- rolling statistics first shift PM2.5 by one hour;
- current-row raw PM2.5 is not a feature;
- future target values never enter features;
- partitions are chronological and never shuffled;
- expanding folds purge rows whose `+6h` target is not strictly before validation start;
- validation supports model and feature selection;
- final test remains isolated until selection is frozen;
- weather stays excluded because available historical weather lacks issue-time/model-run vintage evidence needed to prove point-in-time availability.

## 5. Production V1

Production V1 is served through FastAPI. Public paths include `/forecast/current`, `/status`, `/forecast/latest`, `/health`, and `/predict`. `/forecast/current` reads a local refreshed OpenAQ artifact and returns a fresh or explicitly stale forecast. It does not silently substitute historical data. `/forecast/latest` is a separately labeled historical/offline path when its ignored frozen artifact is provisioned. `/status` reports healthy/degraded operation and fresh/stale/unavailable source state.

Render runs one Uvicorn web service. Startup uses an existing model path or downloads an immutable model artifact with a timeout, size bound, exact SHA-256 verification, temporary-file handling, and atomic replacement. Render's managed refresh is opt-in and preserves the last good local data artifact after refresh failure. Its free-service spin-down and ephemeral filesystem prevent it from being a reliable durable evaluation ledger or canonical scheduler.

Frozen production model record:

```text
Release:
model-v1.0.0

Release target:
d58495804554a34d4420f00d23d511565bc5feee

Active A2-compatible main:
c4cd9e341701ccdd0072529fb8af744f0f1d4488

Artifact:
airaware_v1.joblib

Artifact SHA-256:
1f6ba6a9cb1ca232ba65c80356f5f4cf1f84619ea3a42043bebe46816d26fbdd

Model:
sklearn.linear_model.LinearRegression

Feature configuration:
A2

Forecast horizon:
+6h
```

## 6. Why V1 was not enough

V1 can produce a real forecast, but before M1 it does not create a durable, traffic-independent record of every production forecast. It therefore cannot continuously prove:

```text
prediction
→ eventual observation
→ error
→ persistence error
→ production model skill
```

Production evidence is the first V2 priority. Visual redesign or a more complicated model cannot substitute for a trustworthy record of what was predicted, when, with which artifact and inputs.

## 7. AirAware VN 2.0 architecture council

V2 planning used independent perspectives covering ML and forecasting, MLOps, product, frontend and data visualization, adversarial AI Engineer review, and synthesis. This process was not proof of technical correctness. It challenged scope, assumptions, architecture, and portfolio value before human review.

Resulting principles:

- user value before dashboard metrics;
- measurable ML evidence;
- no fake complexity;
- no unsupported citywide claims;
- no medical advice;
- no fashionable infrastructure without demonstrated need.

## 8. Frozen V2 direction

### Build

- production forecast ledger;
- delayed ground-truth reconciliation;
- production comparison with persistence;
- direct `+1h`/`+3h`/`+6h` evaluation;
- calibrated uncertainty;
- timeline API;
- consumer UI;
- gated Outdoor Window;
- station-specific scope.

### Research

- issue-time weather vintages;
- weather ablation;
- `+12h`/`+24h` horizons;
- elevated-pollution improvements;
- multi-station and spatial work.

### Explicitly rejected for current V2

- LLM chatbot;
- RAG;
- vector DB;
- Kafka;
- Kubernetes;
- Spark;
- Airflow;
- microservices;
- feature store;
- framework rewrite solely for aesthetics;
- deep learning by default;
- automatic model promotion;
- unsupported Hanoi heatmap;
- medical recommendations.

## 9. Why M1 comes first

Core design:

```text
new eligible completed hour
        ↓
Forecast Issuer
        ↓
compute point-in-time forecast
        ↓
persist immutable forecast
        ↓
later ground truth arrives
        ↓
reconcile and score
```

Website traffic must not determine the production evaluation dataset. Request-driven issuance would bias evidence toward periods with users and omit quiet periods. M1 therefore establishes an independently callable issuer and immutable ledger before reconciliation or scoring.

## 10. M0 decision

```text
M0 — AirAware VN 2.0 Frozen Specification
Status: COMPLETE / HUMAN APPROVED
```

Production-activation gates:

```text
Canonical traffic-independent free production scheduler:
NOT YET SELECTED

Durable free production ledger backend:
NOT YET SELECTED
```

These gates do not block local M1 implementation. Production activation remains disabled until both decisions receive human approval.

## 11. Milestone roadmap

```text
M0  Frozen V2 specification                COMPLETE
M1  Forecast storage + scheduled issuer    CURRENT
M1R Weather vintage collection             PARALLEL RESEARCH AFTER M1 FOUNDATION
M2  Delayed truth reconciliation
M3  Production model-vs-persistence
M4  Direct +1h/+3h/+6h experiment
M5  Calibrated prediction intervals
M6  V2 timeline/product API
M7  Consumer frontend
M8  Gated Outdoor Window
M10 Weather ablation
M11 Multi-station research
```

M4 is offline and does not logically require M3 after evaluation contracts are frozen.

## 12. Engineering principles going forward

- no random time-series split;
- no PM2.5 interpolation;
- no test-set model selection;
- no forecast metric without baseline comparison;
- no weather leakage;
- no invented cross-validation metrics;
- production artifacts are immutable and versioned;
- a human approves model promotion;
- a human performs Git commits.

# M1 Implementation Completed

Implementation date: 2026-08-24.

M1 introduced `app/forecast_ledger.py`, `scripts/issue_forecast.py`, `tests/test_forecast_ledger.py`, and `docs/m1-forecast-ledger.md`, with focused compatibility changes in `app/main.py`, `tests/test_api.py`, and `.gitignore`. Result: deterministic UUIDv5 identity, canonical A2 schema hashing, exact artifact-byte hashing, immutable SQLite development storage, independently callable scheduled issuance, and opt-in ledger-backed `/forecast/current` reads. Render ledger activation, production scheduler selection, and durable production storage remain deliberately disabled or undecided.
