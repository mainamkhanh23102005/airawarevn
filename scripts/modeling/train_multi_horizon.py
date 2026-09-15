from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from scripts.modeling.features import V1_FEATURE_COLUMNS, build_v1_features
from scripts.modeling.forecast_bundle import HORIZONS, build_bundle, save_bundle, target_column_for_horizon
from scripts.modeling.train_cli import load_frozen_pm25_dataframe

DEFAULT_OUTPUT_ARTIFACT = Path(".artifacts/models/airaware-mh-v1.joblib")


def _parse_cutoff(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() != pd.Timedelta(0):
        raise ValueError("training cutoff must be an aware UTC timestamp")
    timestamp = timestamp.tz_convert("UTC")
    if timestamp.minute or timestamp.second or timestamp.microsecond or timestamp.nanosecond:
        raise ValueError("training cutoff must align to an hour boundary")
    return timestamp


def _validate_intervals(raw):
    if not isinstance(raw, dict) or not isinstance(raw.get("normalized_records"), list):
        raise ValueError("input artifact must contain normalized_records")
    for record in raw["normalized_records"]:
        if not isinstance(record, dict):
            raise ValueError("normalized record must be an object")
        try:
            event_time = pd.Timestamp(record["event_time"])
            period_end = pd.Timestamp(record["period_end_utc"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("normalized records require hourly period_end_utc") from error
        if any(value.tzinfo is None or value.utcoffset() != pd.Timedelta(0) for value in (event_time, period_end)):
            raise ValueError("normalized record intervals must be aware UTC")
        if period_end.tz_convert("UTC") != event_time.tz_convert("UTC") + pd.Timedelta(hours=1):
            raise ValueError("normalized record period_end_utc must be one hour after event_time")


def build_horizon_dataframe(raw, horizon, training_cutoff=None):
    target_column = target_column_for_horizon(horizon)
    features = build_v1_features(raw)
    targets = raw[["event_time", "pm25"]].copy()
    targets["event_time"] = pd.to_datetime(targets["event_time"], utc=True) - pd.Timedelta(hours=horizon)
    targets = targets.rename(columns={"pm25": target_column})
    result = features.merge(targets, on="event_time", how="left", validate="one_to_one")
    result = result.dropna(subset=[*V1_FEATURE_COLUMNS, target_column])
    if training_cutoff is not None:
        target_end = pd.to_datetime(result["event_time"], utc=True) + pd.Timedelta(hours=horizon + 1)
        result = result.loc[target_end <= training_cutoff]
    return result.reset_index(drop=True)


def _cohort_sha256(data, horizon):
    values = []
    for origin in pd.to_datetime(data["event_time"], utc=True):
        values.append(f"{origin.isoformat()}|{(origin + pd.Timedelta(hours=horizon)).isoformat()}")
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def train_bundle(input_bytes, training_cutoff, model_version):
    raw_artifact = json.loads(input_bytes)
    _validate_intervals(raw_artifact)
    with tempfile.TemporaryDirectory() as directory:
        snapshot = Path(directory) / "input.json"
        snapshot.write_bytes(input_bytes)
        raw = load_frozen_pm25_dataframe(snapshot)
    models = {}
    horizon_metadata = {}
    for horizon in HORIZONS:
        data = build_horizon_dataframe(raw, horizon, training_cutoff)
        if data.empty:
            raise ValueError(f"no eligible training rows for horizon {horizon}")
        model = LinearRegression().fit(data[V1_FEATURE_COLUMNS], data[target_column_for_horizon(horizon)])
        models[horizon] = model
        origins = pd.to_datetime(data["event_time"], utc=True)
        horizon_metadata[horizon] = {
            "training_row_count": len(data),
            "training_origin_start": origins.min().isoformat(),
            "training_origin_end": origins.max().isoformat(),
            "target_interval_start": (origins.min() + pd.Timedelta(hours=horizon)).isoformat(),
            "target_interval_end": (origins.max() + pd.Timedelta(hours=horizon + 1)).isoformat(),
            "cohort_sha256": _cohort_sha256(data, horizon),
        }
    return build_bundle(models, model_version=model_version, training_input_sha256=hashlib.sha256(input_bytes).hexdigest(), training_cutoff=training_cutoff.isoformat(), horizon_metadata=horizon_metadata)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m scripts.modeling.train_multi_horizon")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ARTIFACT)
    parser.add_argument("--training-cutoff", required=True)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"input artifact not found: {args.input}")
    if args.output.exists() and not args.force:
        parser.error(f"output artifact already exists: {args.output} (use --force to overwrite)")
    try:
        cutoff = _parse_cutoff(args.training_cutoff)
        bundle = train_bundle(args.input.read_bytes(), cutoff, args.model_version)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=args.output.parent, prefix=f".{args.output.name}.", suffix=".tmp", delete=False) as temporary:
            temporary_path = Path(temporary.name)
        try:
            save_bundle(temporary_path, bundle)
            if args.force:
                os.replace(temporary_path, args.output)
            else:
                os.link(temporary_path, args.output)
        finally:
            temporary_path.unlink(missing_ok=True)
    except FileExistsError:
        parser.error(f"output artifact already exists: {args.output} (use --force to overwrite)")
    except (ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    return args.output


if __name__ == "__main__":
    main()
