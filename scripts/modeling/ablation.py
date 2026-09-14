from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LinearRegression

from scripts.modeling.backtest import WalkForwardReport, compare_baseline, evaluate_walk_forward
from scripts.modeling.experimental_features import ABLATION_FEATURE_COLUMNS, build_ablation_dataframe
from scripts.modeling.features import V1_FEATURE_COLUMNS
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


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m scripts.modeling.ablation",
        description="Rerun frozen A0-A5 February-June walk-forward ablations.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_PM25_ARTIFACT)
    parser.add_argument("--baseline-comparison", action="store_true", help="Compare fixed A2 LinearRegression with persistence on identical OOF rows.")
    args = parser.parse_args(argv)
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
        print("Methodology: pooled OOF MAEs on identical complete feature/target rows; no imputation.")
        print("Expanding monthly training from 2025-08; validation 2026-02 through 2026-06 (Asia/Ho_Chi_Minh); 2026-07 reserved, not scored.")
        print("Chronology: strict training target time < validation start; six-hour boundary purge.")
        print(f"ml_mae={comparison.ml_mae} baseline_mae={comparison.baseline_mae} evaluated_forecast_count={comparison.evaluated_forecast_count}")
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
                    f"RMSE={fold.metrics.rmse:.4f}"
                )
            print(
                f"  {model_name} pooled: n={model.total_oof_count} "
                f"MAE={model.pooled.mae:.4f} "
                f"RMSE={model.pooled.rmse:.4f}"
            )
    return results


if __name__ == "__main__":
    main()
