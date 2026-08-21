from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd

from scripts.openmeteo import TARGET_LATITUDE, TARGET_LONGITUDE, TARGET_SENSOR_ID
from scripts.run_data_spike import resolve_hour_duplicates

from scripts.modeling.features import (
    FORECAST_HORIZON_HOURS,
    TARGET_COLUMN,
    V1_FEATURE_COLUMNS,
    build_v1_features,
)
from scripts.modeling.train import save_artifact, train_v1_model

DEFAULT_PM25_ARTIFACT = Path(
    ".artifacts/data_spike/coverage/"
    "openaq_normalized_sensor_13502151_20250731T170000+0000.json"
)
DEFAULT_OUTPUT_ARTIFACT = Path(".artifacts/models/airaware_v1.joblib")


def load_frozen_pm25_dataframe(path: Path) -> pd.DataFrame:
    """Load a frozen OpenAQ normalized PM2.5 artifact into a chronological hourly grid.

    The returned dataframe contains one row per hour in the frozen window and uses
    ``NaN`` for missing measurements. No interpolation or imputation is performed.
    """
    with path.open("r", encoding="utf-8") as file:
        artifact = json.load(file)

    if not isinstance(artifact, dict):
        raise ValueError("input artifact must be a JSON object")

    metadata = artifact.get("sensor_metadata")
    coordinates = metadata.get("coordinates") if isinstance(metadata, dict) else None
    latitude = coordinates.get("latitude") if isinstance(coordinates, dict) else None
    longitude = coordinates.get("longitude") if isinstance(coordinates, dict) else None
    if (
        not isinstance(metadata, dict)
        or metadata.get("sensor_id") != TARGET_SENSOR_ID
        or not isinstance(latitude, (int, float))
        or isinstance(latitude, bool)
        or not isinstance(longitude, (int, float))
        or isinstance(longitude, bool)
        or not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not math.isclose(latitude, TARGET_LATITUDE, rel_tol=0, abs_tol=1e-6)
        or not math.isclose(longitude, TARGET_LONGITUDE, rel_tol=0, abs_tol=1e-6)
    ):
        raise ValueError("input artifact sensor does not match fixed target")

    records = artifact.get("normalized_records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("input artifact contains no normalized_records")

    grouped = {}
    try:
        for record in records:
            if not isinstance(record, dict) or record.get("sensor_id") != TARGET_SENSOR_ID:
                raise ValueError("normalized record sensor does not match fixed target")
            raw_event_time = record["event_time"]
            if not isinstance(raw_event_time, str):
                raise ValueError("normalized record event_time must be an aware UTC timestamp")
            event_time = pd.Timestamp(raw_event_time)
            if event_time.tzinfo is None or event_time.utcoffset() != pd.Timedelta(0):
                raise ValueError("normalized record event_time must be an aware UTC timestamp")
            event_time = event_time.tz_convert("UTC")
            if event_time.minute or event_time.second or event_time.microsecond:
                raise ValueError("normalized record event_time must be hour-aligned")
            grouped.setdefault(event_time, []).append(record)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("invalid normalized record schema") from error
    pm25 = pd.DataFrame(
        (
            {
                "event_time": event_time,
                "pm25": resolve_hour_duplicates(rows).selected_value_ug_m3,
            }
            for event_time, rows in sorted(grouped.items())
        )
    )

    frozen_window = artifact.get("frozen_candidate")
    if not isinstance(frozen_window, dict) or {
        "start_utc",
        "end_utc",
    } - frozen_window.keys():
        raise ValueError("invalid frozen_candidate")
    try:
        start = pd.Timestamp(frozen_window["start_utc"])
        end = pd.Timestamp(frozen_window["end_utc"])
    except (TypeError, ValueError) as error:
        raise ValueError("invalid frozen_candidate") from error
    if (
        start.tzinfo is None
        or end.tzinfo is None
        or start.utcoffset() != pd.Timedelta(0)
        or end.utcoffset() != pd.Timedelta(0)
        or start.minute
        or start.second
        or start.microsecond
        or end.minute
        or end.second
        or end.microsecond
    ):
        raise ValueError("frozen_candidate bounds must be hour-aligned aware UTC timestamps")
    start = start.tz_convert("UTC")
    end = end.tz_convert("UTC")
    if start >= end:
        raise ValueError("frozen_candidate start must precede end")
    if (pm25["event_time"] < start).any() or (pm25["event_time"] >= end).any():
        raise ValueError("normalized record falls outside frozen_candidate")

    hourly_grid = pd.DataFrame(
        {
            "event_time": pd.date_range(
                start=start,
                end=end,
                freq="1h",
                inclusive="left",
            )
        }
    )
    return hourly_grid.merge(pm25, on="event_time", how="left")


def build_modeling_dataframe(path: Path) -> pd.DataFrame:
    """Build a fully trainable V1 modeling dataframe from a frozen PM2.5 artifact.

    This reuses the same feature builder and target construction used in production
    inference. Rows with incomplete features or targets are dropped (no imputation).
    """
    raw = load_frozen_pm25_dataframe(path)
    features = build_v1_features(raw, include_target=True)
    required = [*V1_FEATURE_COLUMNS, TARGET_COLUMN]
    return features.dropna(subset=required).reset_index(drop=True)


def main(argv=None) -> Path:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.modeling.train_cli",
        description=(
            "Train the frozen AirAware V1 PM2.5 forecasting model from a "
            "frozen OpenAQ normalized artifact. No weather, no interpolation, "
            "no retuning, and no live OpenAQ substitution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_PM25_ARTIFACT,
        help="Path to the frozen OpenAQ normalized PM2.5 artifact",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ARTIFACT,
        help="Path to write the frozen V1 joblib model artifact",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output artifact if it already exists",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print a summary of the training result",
    )
    args = parser.parse_args(argv)

    if not args.input.exists():
        parser.error(f"input artifact not found: {args.input}")

    if args.output.exists() and not args.force:
        parser.error(
            f"output artifact already exists: {args.output} "
            "(use --force to overwrite)"
        )

    try:
        modeling_df = build_modeling_dataframe(args.input)
        model, metadata = train_v1_model(modeling_df)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=args.output.parent,
                prefix=f".{args.output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
            save_artifact(temporary_path, model, metadata)
            if args.force:
                os.replace(temporary_path, args.output)
                temporary_path = None
            else:
                os.link(temporary_path, args.output)
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
    except FileExistsError:
        parser.error(
            f"output artifact already exists: {args.output} "
            "(use --force to overwrite)"
        )
    except ValueError as error:
        parser.error(str(error))

    if args.verbose:
        print(f"output: {args.output}")
        print(f"training rows: {metadata['training_row_count']}")
        print(f"training range: {metadata['training_start']} to {metadata['training_end']}")
        print(f"features: {metadata['feature_columns']}")
        print(f"target: {metadata['target_column']}")
        print(f"horizon: {metadata['forecast_horizon_hours']}h")
        print(f"trained at: {metadata['trained_at_utc']}")

    return args.output


if __name__ == "__main__":
    main()
