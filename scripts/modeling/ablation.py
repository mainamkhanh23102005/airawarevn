from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from scripts.modeling.backtest import WalkForwardReport, evaluate_walk_forward
from scripts.modeling.experimental_features import ABLATION_FEATURE_COLUMNS, build_ablation_dataframe
from scripts.modeling.train_cli import DEFAULT_PM25_ARTIFACT, load_frozen_pm25_dataframe


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
    args = parser.parse_args(argv)
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
