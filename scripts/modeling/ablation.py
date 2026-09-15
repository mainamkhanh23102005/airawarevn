from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.modeling.backtest import WalkForwardReport, compare_baseline, create_walk_forward_folds, evaluate_walk_forward
from scripts.modeling.experimental_features import ABLATION_FEATURE_COLUMNS, M8_FEATURE_COLUMNS, M8_FEATURE_DEFINITION_VERSION, build_ablation_dataframe, build_experimental_features
from scripts.modeling.features import TARGET_COLUMN, V1_FEATURE_COLUMNS
from scripts.modeling.train_cli import DEFAULT_PM25_ARTIFACT, build_modeling_dataframe, load_frozen_pm25_dataframe


@dataclass(frozen=True)
class AblationResult:
    name: str
    feature_columns: tuple[str, ...]
    retained_validation_count: int
    report: WalkForwardReport


def run_feature_ablation(artifact_path):
    raw = load_frozen_pm25_dataframe(artifact_path)
    results = {}
    for name, feature_columns in ABLATION_FEATURE_COLUMNS.items():
        dataframe = build_ablation_dataframe(raw, feature_columns)
        report = evaluate_walk_forward(dataframe, feature_columns=feature_columns)
        results[name] = AblationResult(name, tuple(feature_columns), report.total_oof_count, report)
    return results


def _m8_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _m8_rows(data):
    return [[time.isoformat(), float(target)] for time, target in zip(data.event_time, data[TARGET_COLUMN])]


def _m8_metrics(metrics):
    return {"mae": metrics.mae, "rmse": metrics.rmse, "bias": metrics.bias, "count": metrics.evaluated_forecast_count}


def run_m8_experiment(artifact_path):
    raw = load_frozen_pm25_dataframe(artifact_path)
    if raw.event_time.isna().any() or raw.event_time.duplicated().any():
        raise ValueError("M8 requires unique nonmissing origin timestamps")
    featured = build_experimental_features(raw, include_target=True)
    columns = list(dict.fromkeys(column for values in M8_FEATURE_COLUMNS.values() for column in values))
    required = [*columns, TARGET_COLUMN]
    if np.isinf(featured[required].to_numpy(dtype=float)).any():
        raise ValueError("M8 features and targets must be finite or missing")
    common = featured.dropna(subset=required).reset_index(drop=True)
    native_a2 = featured.dropna(subset=[*M8_FEATURE_COLUMNS["A2"], TARGET_COLUMN])
    result = {
        "input_sha256": hashlib.sha256(Path(artifact_path).read_bytes()).hexdigest(),
        "experimental_feature_definition_version": M8_FEATURE_DEFINITION_VERSION,
        "raw_count": len(raw),
        "observed_count": int(raw.pm25.notna().sum()),
        "missing_counts": {column: int(featured[column].isna().sum()) for column in required},
        "ineligibility_reasons": {column: [time.isoformat() for time in featured.loc[featured[column].isna(), "event_time"]] for column in required},
        "limitations": ["Archive ingestion availability unverifiable", "July previously reported, not pristine; no July-origin tuning/scoring"],
        "cohorts": {},
    }
    for mode in ("native", "common"):
        summaries = {}
        reference_folds = None
        reference_predictions = None
        for name, features in M8_FEATURE_COLUMNS.items():
            native = featured.dropna(subset=[*features, TARGET_COLUMN]).reset_index(drop=True)
            data = native if mode == "native" else common
            folds, _ = create_walk_forward_folds(data, features)
            if mode == "common" and reference_folds is not None:
                for fold, reference in zip(folds, reference_folds):
                    for part in ("training", "validation"):
                        if _m8_rows(getattr(fold, part)) != _m8_rows(getattr(reference, part)):
                            raise ValueError("common training/validation keys and truth must match")
            report = evaluate_walk_forward(data, feature_columns=features, model_factories={"linear_regression": LinearRegression, "persistence": None})
            comparison = compare_baseline(report.predictions, "linear_regression")
            predictions = report.predictions.query("model == 'linear_regression'")
            if reference_predictions is None:
                reference_predictions = predictions
                reference_folds = folds
            paired = predictions.merge(reference_predictions, on=["fold", "timestamp"], suffixes=("", "_a2"), validate="one_to_one")
            if not np.array_equal(paired.actual_pm25, paired.actual_pm25_a2):
                raise ValueError("paired A2 actual targets must match")
            paired_change = float((paired.predicted_pm25 - paired.actual_pm25).abs().mean() - (paired.predicted_pm25_a2 - paired.actual_pm25_a2).abs().mean())
            fold_summaries = []
            for index, fold in enumerate(folds):
                fold_summaries.append({
                    "name": fold.name,
                    "training_count": len(fold.training),
                    "validation_count": len(fold.validation),
                    "purged_count": fold.purged_boundary_count,
                    "training_id": _m8_hash(_m8_rows(fold.training)),
                    "validation_id": _m8_hash(_m8_rows(fold.validation)),
                    "models": {model: _m8_metrics(value.folds[index].metrics) for model, value in report.models.items()},
                })
            summaries[name] = {
                "features": list(features),
                "schema_id": _m8_hash({"version": 1, "experimental_feature_definition_version": M8_FEATURE_DEFINITION_VERSION, "configuration": name, "features": list(features), "target": TARGET_COLUMN, "horizon": 6, "timezone": "Asia/Ho_Chi_Minh"}),
                "cohort_id": _m8_hash(_m8_rows(data)),
                "cohort_differs_from_a2": _m8_rows(data) != _m8_rows(native_a2),
                "cohort_difference_reason": "additional selected-feature missingness relative to native A2" if _m8_rows(data) != _m8_rows(native_a2) else "identical origins and targets to native A2",
                "eligible_count": len(data),
                "native_eligible_count": len(native),
                "common_removed_count": len(native) - len(common),
                "common_removed_origins": [time.isoformat() for time in native.loc[~native.event_time.isin(common.event_time), "event_time"]],
                "evaluated_count": report.total_oof_count,
                "raw_validation_count": int(((featured.event_time >= folds[0].validation_start) & (featured.event_time < folds[-1].validation_end)).sum()),
                "excluded_validation_count": int(((featured.event_time >= folds[0].validation_start) & (featured.event_time < folds[-1].validation_end)).sum()) - report.total_oof_count,
                "late_june_target_count": int(((predictions.fold == "2026-06") & ((predictions.timestamp + pd.Timedelta(hours=6)) >= report.final_test.start)).sum()),
                "models": {model: _m8_metrics(value.pooled) for model, value in report.models.items()},
                "canonical_a2_mae_change": comparison.ml_mae - 9.855844823038945,
                "canonical_persistence_mae_change": comparison.ml_mae - 11.307251308900524,
                "canonical_a2_mae_change_percent": (comparison.ml_mae - 9.855844823038945) / 9.855844823038945 * 100,
                "canonical_persistence_mae_change_percent": (comparison.ml_mae - 11.307251308900524) / 11.307251308900524 * 100,
                "paired_persistence_mae_change": comparison.ml_mae - comparison.baseline_mae,
                "paired_a2_mae_change": paired_change,
                "paired_a2_count": len(paired),
                "paired_a2_mae": float((paired.predicted_pm25_a2 - paired.actual_pm25_a2).abs().mean()),
                "paired_candidate_mae": float((paired.predicted_pm25 - paired.actual_pm25).abs().mean()),
                "paired_a2_training_matched": all(_m8_rows(fold.training) == _m8_rows(reference.training) for fold, reference in zip(folds, reference_folds)),
                "folds": fold_summaries,
            }
        result["cohorts"][mode] = summaries
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m scripts.modeling.ablation",
        description="Rerun frozen A0-A5 February-June walk-forward ablations.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PM25_ARTIFACT)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--baseline-comparison", action="store_true", help="Compare fixed A2 LinearRegression with persistence on identical OOF rows.")
    mode.add_argument("--m8", action="store_true", help="Run fixed M8 native/common experimental cohorts; print JSON.")
    args = parser.parse_args(argv)
    if args.m8:
        result = run_m8_experiment(args.input)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return result
    if args.baseline_comparison:
        dataframe = build_modeling_dataframe(args.input)
        report = evaluate_walk_forward(
            dataframe,
            feature_columns=V1_FEATURE_COLUMNS,
            model_factories={"linear_regression": LinearRegression, "persistence": None},
        )
        comparison = compare_baseline(report.predictions, "linear_regression")
        print("Fixed A2 LinearRegression vs persistence; no tuning.")
        print("Baseline: pm25_lag_1h = latest completed hour available at origin t; forecast target t+6h.")
        print("Methodology: pooled OOF MAEs on identical complete feature/target rows; no imputation. RMSE and Bias use the same rows.")
        print("Bias = predicted - actual; positive = overprediction; negative = underprediction.")
        print("Expanding monthly training from 2025-08; validation 2026-02 through 2026-06 (Asia/Ho_Chi_Minh); 2026-07 July origins reserved, not scored; canonical cohort: six late-June +6h targets in July retained.")
        print("Chronology: strict training target time < validation start; six-hour boundary purge.")
        print(f"ml_mae={comparison.ml_mae} baseline_mae={comparison.baseline_mae} evaluated_forecast_count={comparison.evaluated_forecast_count}")
        print(f"ml_rmse={comparison.ml_rmse} ml_bias={comparison.ml_bias} baseline_rmse={comparison.baseline_rmse} baseline_bias={comparison.baseline_bias}")
        percentage = "unavailable (empty cohort or zero baseline MAE)" if comparison.improvement_percent is None else f"{comparison.improvement_percent:.4f}%"
        verdict = "unavailable"
        if comparison.ml_mae is not None and comparison.baseline_mae is not None:
            verdict = "better" if comparison.ml_mae < comparison.baseline_mae else "worse" if comparison.ml_mae > comparison.baseline_mae else "equal"
        print(f"improvement_percent={percentage} verdict={verdict}")
        return comparison
    results = run_feature_ablation(args.input)
    for name, result in results.items():
        print(f"{name}: {result.retained_validation_count} OOF rows")
        for model_name, model in result.report.models.items():
            for fold in model.folds:
                print(
                    f"  {model_name} {fold.name}: "
                    f"n={fold.validation_count} "
                    f"MAE={fold.metrics.mae:.4f} "
                    f"RMSE={fold.metrics.rmse:.4f} "
                    f"Bias={fold.metrics.bias:.4f} "
                    f"evaluated_forecast_count={fold.metrics.evaluated_forecast_count}"
                )
            print(
                f"  {model_name} pooled: n={model.total_oof_count} "
                f"MAE={model.pooled.mae:.4f} "
                f"RMSE={model.pooled.rmse:.4f} "
                f"Bias={model.pooled.bias:.4f} "
                f"evaluated_forecast_count={model.pooled.evaluated_forecast_count}"
            )
    return results


if __name__ == "__main__":
    main()
