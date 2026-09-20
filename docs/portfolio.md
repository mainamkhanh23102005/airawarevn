# AirAware VN — Technical Portfolio

## 30-second explanation

AirAware VN is an end-to-end PM2.5 forecasting system for Hanoi. It turns OpenAQ hourly observations into forecasts, serves a six-point outlook through FastAPI, and checks issued forecasts against later observations. The canonical +6h LinearRegression model reduced walk-forward MAE by **12.84% versus persistence** on **3,056 paired predictions**. Render serves the application; Turso/libSQL preserves forecast monitoring records; GitHub Actions schedules the monitoring worker.

## Two-minute technical explanation

The project connects data engineering, time-series evaluation, and deployment rather than treating a trained model as the finished product. OpenAQ records preserve hourly interval semantics. Missing and conflicting measurements are checked explicitly; live inference requires 24 contiguous completed hours and never invents missing observations.

The selected A2 model uses lagged PM2.5, shifted rolling means, and Hanoi calendar features. Expanding monthly walk-forward validation tests future months using only prior training targets. Persistence predicts the latest completed observation and provides a strong, understandable baseline. On February–June 2026 pooled validation, A2 MAE is **9.855844823038945**, versus **11.307251308900524** for persistence: **12.836068167328174%** lower MAE. This is an offline validation result, not a claim about current production accuracy.

A separate bundle produces direct +1h through +6h predictions. Each horizon has its own estimator; missing final +1h–+5h score reports are disclosed rather than filled with estimates. Richer M8 feature sets did not beat A2, so the simpler specification remained selected.

Production monitoring is independent of user traffic. A worker issues immutable forecasts, reconciles mature targets with OpenAQ, materializes evaluations, and publishes consumer metrics. Deterministic identities and a database lease make retries safe without presenting a best-effort scheduler as guaranteed delivery.

## Architecture walkthrough

1. **Observation ingestion:** OpenAQ hourly PM2.5 for sensor `13502151` is normalized with interval and provenance metadata. Render and the monitoring worker refresh separate local artifacts.
2. **Feature construction:** validated past measurements feed the shared A2 feature contract. Training and inference use compatible feature semantics.
3. **Serving:** Render hosts FastAPI and its HTML/JS interface. With the store enabled, `/forecast/current` reads the latest issued V1 forecast from Turso; `/forecast/trajectory` computes six direct forecasts using Render-local observations.
4. **Durability:** Turso/libSQL holds forecast identities, reconciliation/evaluation records, snapshots, and consumer publication. It does not replace local live PM2.5 JSON storage.
5. **Scheduling:** GitHub Actions runs the main-branch worker at minute 15 hourly, with manual dispatch for recovery. It provisions V1, refreshes observations, then executes the monitoring cycle.
6. **Feedback:** verified future observations become forecast errors and immutable snapshots, then consumer-facing performance. Offline benchmark tables and production performance remain separate evidence.

See the [architecture diagram and operational details](../README.md#architecture), [API implementation](../app/main.py), and [monitoring cycle](../scripts/run_monitoring_cycle.py).

## Why chronological walk-forward evaluation matters

A random split would mix past and future observations and would not represent a forecast made before its target exists. Expanding folds preserve time order and allow training history to grow naturally. A training target must be strictly earlier than the validation start; merely checking the feature timestamp is insufficient.

For prediction origin `t`, `pm25_lag_1h` represents the completed interval `[t-1h, t)`. The +6h target covers `[t+6h, t+7h)`. Rolling windows shift before aggregation. Multi-horizon targets use exact timestamps, so missing rows cannot shift a label to the wrong hour.

These controls address feature and target leakage. They do not reconstruct historical OpenAQ delivery times. Six late-June validation targets cross into July; July is not a pristine holdout for new experiments after its earlier results were reported.

## Why persistence matters

Hourly PM2.5 is autocorrelated: retaining the last completed value can already be competitive. A sophisticated-looking model is not useful merely because its error is nonzero but small. Comparing against persistence on identical origins measures whether the model adds value beyond recent conditions.

The +6h canonical comparison uses **3,056** rows for both models. A2 improves MAE by **12.836068167328174%** and RMSE by **13.207754664899884%**, but has a larger positive signed bias: **+0.9370505512624444**, versus persistence **+0.11611256544502616** µg/m³. Better absolute error does not mean every diagnostic improved.

## Why A2 remained selected after M8

M8 investigated rolling standard deviations and cyclical hour encoding. Native and common-cohort comparisons avoid rewarding a feature set simply for dropping harder rows; paired A2 fits also match training membership.

The retained project conclusion is that no tested richer feature set beat A2. Keeping A2 avoids extra feature complexity without demonstrated benefit. Detailed candidate scores are not committed, so this portfolio does not invent rankings or margins. [Experiment definitions](../scripts/modeling/experimental_features.py) and [ablation checks](../tests/test_backtest.py) preserve the evaluation mechanics, not a complete candidate-score report.

## Production monitoring

Forecast issuance precedes truth availability. A deterministic identity records sensor, origin, target interval, horizon, model version, artifact hash, and feature-schema hash. Once the interval ends and the observation delay elapses, reconciliation checks actual OpenAQ evidence. Only verified outcomes contribute to evaluations; missing truth is not scored as an invented observation.

Immutable snapshots preserve evaluation membership. Consumer publication uses a trailing 30-day window ending at local midnight, becomes eligible at/after 03:15 ICT, and requires at least 48 verified forecasts for metrics. `/consumer/forecast-performance` can explicitly report unavailable performance.

A transactional 3,600-second lease protects scheduled execution. GitHub reruns reuse the run-based owner; deterministic records prevent duplicate logical issuance and scoring. The lease has no in-cycle heartbeat. GitHub scheduling is best-effort, so delayed/dropped events and manual recovery remain operational concerns.

## Engineering challenges solved

- **Temporal correctness:** aligned hourly interval semantics, shifted features, strict training-target boundaries, and timestamp-based targets.
- **Offline/online consistency:** preserved A2 feature ordering and separate model-artifact, schema, and model-version identities.
- **Data-quality failure modes:** explicit stale, missing, conflicting, and historical states instead of silently substituting or interpolating data.
- **Retryable monitoring:** durable forecast identity, observation reconciliation, evaluations, snapshots, and publication without dependence on page visits.
- **Deployment boundaries:** separated ephemeral web observations from durable monitoring history; provisioned frozen models without startup training.
- **Evidence-based selection:** retained a simple model after richer features failed to justify promotion.

## Key tradeoffs

| Decision | Benefit | Cost / boundary |
|---|---|---|
| A2 LinearRegression | Inspectable, compact baseline with measured +6h improvement | High-pollution underprediction remains |
| Single primary sensor | Clear data contract and reproducible scope | No spatial generalization claim |
| Require contiguous live history | Avoids fabricated features | Gaps can make forecasts unavailable |
| Separate direct horizon estimators | No recursive prediction feedback | Separate models and horizon-specific evidence required |
| Turso/libSQL ledger | Durable shared monitoring across worker and web service | Remote service credentials and availability dependency |
| Local current PM2.5 JSON | Simple refresh/serving boundary | Ephemeral state, no shared artifact storage |
| GitHub Actions scheduling | Reuses repository automation with manual recovery | No guaranteed hourly delivery or backfill |
| Frozen model provisioning | Explicit artifact identity; no startup training | Trusted artifact handling and version compatibility required |

## Known limitations and evidence boundaries

- Canonical history: **8,339 valid hours / 8,760 expected**, approximately **95.19%** coverage; longest observed gap **86 hours**. Sparse coverage-test fixtures do not describe this dataset.
- Frozen input window: `2025-07-31T17:00:00Z` through `2026-07-31T17:00:00Z`, end-exclusive. Retained SHA-256: `585d2068dc66d0f66ffbb9cb51d402b2e7cc5fe2ce59804e163f86d435235bc1`. The ignored artifact is absent from a fresh clone; these provenance records are not a new local hash verification.
- Exact +1h–+5h final benchmark scores and detailed M8 candidate reports are missing from the repository. Only +6h exact validation metrics are published in the [README](../README.md#model-results).
- Historical July metrics are rounded and belong to the pre-July fit, not the later production artifact. High-pollution underprediction remains unresolved.
- No calibrated uncertainty, demonstrated multi-station performance, or production-equivalent weather vintage history.
- Best-effort scheduling, limited alerting, no lease heartbeat, and ephemeral live-data storage remain operational limits.
- Dedicated remote OpenAQ/Turso smoke environment and complete transitive dependency locking remain future work.
- Recorded pre-M10 WSL baseline: **508 tests, OK (skipped=37)**. This is not a new execution claim or proof that conditional tests ran.
- No screenshots are included: this checkout lacks the ignored model/data artifacts needed for a genuine populated local application capture. No fabricated UI evidence is substituted.

## V2 roadmap

Prioritize stronger evidence and operations before adding model complexity:

1. Investigate newer valid observations and historical missing-hour gaps.
2. Strengthen scheduler delivery, alerting/watchdog coverage, and shared PM2.5 artifact durability.
3. Establish a dedicated integration environment for OpenAQ/Turso smoke checks.
4. Study uncertainty/calibration and high-pollution errors.
5. Evaluate additional sensors and weather only with point-in-time provenance.
6. Define retraining/promotion gates using paired chronological comparisons against persistence.

These are prospective directions, not implemented functionality or delivery promises.

## CV bullets

- Built an end-to-end Hanoi PM2.5 forecasting system using OpenAQ and leakage-safe expanding walk-forward validation, reducing +6h MAE by **12.84% versus persistence across 3,056 paired predictions**.
- Developed a **six-horizon (+1h to +6h)** direct forecasting API with shared A2 feature contracts, explicit freshness states, and validation of **24 contiguous completed observation hours**.
- Designed durable forecast monitoring with **Turso/libSQL**, immutable identities, observation reconciliation, evaluation snapshots, and retry-safe scheduled issuance independent of web traffic.
- Integrated **Render** serving and **GitHub Actions** monitoring with trusted model provisioning, SHA-256-checked downloads, database lease protection, and manual recovery for best-effort schedules.

## Interview questions and answers

### Why LinearRegression?

It is an inspectable model that already improves +6h error over persistence on paired chronological validation. Complexity needs evidence; richer M8 features did not justify replacing A2. This does not imply linear regression always wins or resolves extreme pollution.

### Why not a random split?

Production predicts future observations from past information. Random splits mix temporal regimes and may let future labels influence training. Expanding walk-forward folds and target-boundary purges better represent that task.

### What counts as leakage here?

Using an unfinished current interval, an unshifted rolling mean, a training label not available before validation, or historical weather whose forecast vintage is unknown. Event-time safeguards are explicit; historical provider delivery-time availability remains unverifiable.

### Why use persistence?

It tests whether the model improves on the latest completed observation, a strong baseline for hourly autocorrelated data. Both methods must use identical eligible forecast origins.

### Why keep A2 after richer features?

The recorded M8 outcome did not show an improvement. Common cohorts and matched training comparisons protect against apparent gains caused by dropping observations. Keeping A2 avoids unsupported complexity; detailed candidate score reports still need preservation.

### How does forecast reconciliation work?

The issued forecast remains immutable. Once its target interval is mature, the worker fetches and validates corresponding observations, records reconciliation evidence, and materializes evaluations and snapshots. Missing or unverified truth does not become a fabricated score.

### How do duplicate scheduler runs stay safe?

A database lease guards execution; deterministic forecast and evaluation identities make repeated logical work idempotent. Workflow concurrency adds a scheduler-level guard. These safeguards do not guarantee delivery or eliminate the no-heartbeat lease limitation.

### Why Turso/libSQL?

Web and monitoring workers need shared durable records across separate, ephemeral runtimes. Turso supplies that ledger while local SQLite supports development. It does not currently store the web runtime's live PM2.5 artifact.

### What happens when OpenAQ data is missing?

Refresh failures preserve the last good artifact. Stale data is labeled; a missing or conflicting hour can prevent construction of the required 24-hour feature history. Empty ledger mode returns unavailable rather than silently recomputing a historical forecast. Reconciliation waits for valid truth instead of interpolating it.

### Why can the current forecast and trajectory disagree in availability?

The configured current endpoint reads the latest ledger issuance, while trajectory inference uses Render-local observations and a separate bundle. They can have different origins and freshness. `/status` describes local inference, not ledger readiness.

### What would improve in V2?

First improve observation coverage, scheduler guarantees, alerting, shared live-data storage, and integration evidence. Then test uncertainty, additional data, or models using point-in-time-safe, paired chronological comparisons. Retain persistence as the minimum benchmark.
