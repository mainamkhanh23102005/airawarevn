from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.modeling.backtest import (
    FINAL_TEST_MONTH,
    INITIAL_TRAINING_MONTH,
    VALIDATION_MONTHS,
    compare_baseline,
    evaluate_walk_forward,
)
from scripts.modeling.features import TARGET_COLUMN, V1_FEATURE_COLUMNS, build_v1_features
from scripts.modeling.train_cli import DEFAULT_PM25_ARTIFACT, load_frozen_pm25_dataframe


HORIZONS = (1, 2, 3, 4, 5, 6)


def target_column_for_horizon(horizon):
    if horizon not in HORIZONS:
        raise ValueError(f"horizon must be one of {HORIZONS}")
    return TARGET_COLUMN if horizon == 6 else f"target_pm25_t_plus_{horizon}"


def build_horizon_dataframe(raw, horizon):
    target_column = target_column_for_horizon(horizon)
    features = build_v1_features(raw)
    targets = raw[["event_time", "pm25"]].copy()
    targets["event_time"] = pd.to_datetime(targets["event_time"], utc=True) - pd.Timedelta(hours=horizon)
    targets = targets.rename(columns={"pm25": target_column})
    result = features.merge(targets, on="event_time", how="left", validate="one_to_one")
    return result.dropna(subset=[*V1_FEATURE_COLUMNS, target_column]).reset_index(drop=True)


def _metrics(metrics):
    return {"mae": metrics.mae, "rmse": metrics.rmse, "bias": metrics.bias, "count": metrics.evaluated_forecast_count}


def run_multi_horizon_experiment(artifact_path):
    raw = load_frozen_pm25_dataframe(artifact_path)
    results = {}
    for horizon in HORIZONS:
        target_column = target_column_for_horizon(horizon)
        data = build_horizon_dataframe(raw, horizon)
        report = evaluate_walk_forward(
            data,
            feature_columns=V1_FEATURE_COLUMNS,
            model_factories={"linear_regression": LinearRegression, "persistence": None},
            target_column=target_column,
            horizon_hours=horizon,
        )
        comparison = compare_baseline(report.predictions, "linear_regression")
        results[str(horizon)] = {
            "target_column": target_column,
            "linear_regression": _metrics(report.models["linear_regression"].pooled),
            "persistence": _metrics(report.models["persistence"].pooled),
            "comparison": {"mae_improvement_percent": comparison.improvement_percent, "count": comparison.evaluated_forecast_count},
        }
    return {
        "input_sha256": hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        "horizons": list(HORIZONS),
        "folds": {"initial_training_month": INITIAL_TRAINING_MONTH, "validation_months": list(VALIDATION_MONTHS), "reserved_origin_month": FINAL_TEST_MONTH},
        "persistence_definition": "pm25_lag_1h: latest completed PM2.5 observation available at forecast origin",
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run frozen A2 direct multi-horizon PM2.5 walk-forward evaluation.")
    parser.add_argument("--input", type=Path, default=DEFAULT_PM25_ARTIFACT)
    args = parser.parse_args(argv)
    result = run_multi_horizon_experiment(args.input)
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
