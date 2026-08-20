import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler


PM25_PATH = Path(
    ".artifacts/data_spike/coverage/"
    "openaq_normalized_sensor_13502151_20250731T170000+0000.json"
)
WEATHER_PATH = Path(
    ".artifacts/data_spike/weather/openmeteo_normalized.json"
)

feature_columns = [
    "pm25_lag_1h",
    "pm25_lag_3h",
    "pm25_lag_6h",
    "pm25_lag_12h",
    "pm25_lag_24h",
    "pm25_rolling_mean_6h",
    "pm25_rolling_mean_12h",
    "pm25_rolling_mean_24h",
    "hour",
    "day_of_week",
    "month",
    "is_weekend",
]
experimental_feature_columns = [
    "pm25_change_1h_3h",
    "pm25_change_1h_6h",
    "pm25_change_1h_12h",
    "pm25_change_1h_24h",
    "pm25_rolling_std_6h",
    "pm25_rolling_std_12h",
    "pm25_rolling_std_24h",
]
expanded_feature_columns = [*feature_columns, *experimental_feature_columns]
target_column = "target_pm25_t_plus_6"
weather_columns = {
    "temperature_2m",
    "relative_humidity_2m",
    "wind_speed_10m",
    "precipitation",
    "surface_pressure",
}


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_modeling_dataframe():
    pm25_data = load_json(PM25_PATH)
    weather_data = load_json(WEATHER_PATH)

    pm25 = pd.DataFrame(pm25_data["normalized_records"])
    weather = pd.DataFrame(weather_data["records"])

    pm25["event_time"] = pd.to_datetime(pm25["event_time"], utc=True)
    weather["event_time"] = pd.to_datetime(weather["event_time"], utc=True)

    start = pd.to_datetime(
        pm25_data["frozen_candidate"]["start_utc"],
        utc=True,
    )
    end = pd.to_datetime(
        pm25_data["frozen_candidate"]["end_utc"],
        utc=True,
    )

    pm25_frozen = pm25[
        (pm25["event_time"] >= start)
        & (pm25["event_time"] < end)
    ].copy()
    weather_frozen = weather[
        (weather["event_time"] >= start)
        & (weather["event_time"] < end)
    ].copy()

    modeling_df = pd.DataFrame(
        {
            "event_time": pd.date_range(
                start=start,
                end=end,
                freq="1h",
                inclusive="left",
            )
        }
    )
    modeling_df = modeling_df.merge(
        pm25_frozen[["event_time", "value"]],
        on="event_time",
        how="left",
    ).rename(columns={"value": "pm25"})
    modeling_df = modeling_df.merge(
        weather_frozen,
        on="event_time",
        how="left",
    )

    modeling_df[target_column] = modeling_df["pm25"].shift(-6)

    for lag in [1, 3, 6, 12, 24]:
        modeling_df[f"pm25_lag_{lag}h"] = modeling_df["pm25"].shift(lag)

    modeling_df["pm25_change_1h_3h"] = (
        modeling_df["pm25_lag_1h"] - modeling_df["pm25_lag_3h"]
    )
    modeling_df["pm25_change_1h_6h"] = (
        modeling_df["pm25_lag_1h"] - modeling_df["pm25_lag_6h"]
    )
    modeling_df["pm25_change_1h_12h"] = (
        modeling_df["pm25_lag_1h"] - modeling_df["pm25_lag_12h"]
    )
    modeling_df["pm25_change_1h_24h"] = (
        modeling_df["pm25_lag_1h"] - modeling_df["pm25_lag_24h"]
    )

    historical_pm25 = modeling_df["pm25"].shift(1)
    for window in [6, 12, 24]:
        historical_window = historical_pm25.rolling(
            window=window,
            min_periods=window,
        )
        modeling_df[f"pm25_rolling_mean_{window}h"] = (
            historical_window.mean()
        )
        modeling_df[f"pm25_rolling_std_{window}h"] = historical_window.std()

    local_time = modeling_df["event_time"].dt.tz_convert(
        "Asia/Ho_Chi_Minh"
    )
    modeling_df["hour"] = local_time.dt.hour
    modeling_df["day_of_week"] = local_time.dt.dayofweek
    modeling_df["month"] = local_time.dt.month
    modeling_df["is_weekend"] = (
        modeling_df["day_of_week"] >= 5
    ).astype(int)

    return modeling_df


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    return mae, rmse


def print_improvement(name, baseline, model):
    absolute_improvement = baseline - model
    percentage_improvement = absolute_improvement / baseline * 100
    print(
        f"{name}: {absolute_improvement:.4f} "
        f"({percentage_improvement:.2f}%)"
    )


def print_split(name, dataframe):
    print(
        f"{name}: {len(dataframe)} rows, "
        f"{dataframe['event_time'].min()} to "
        f"{dataframe['event_time'].max()}"
    )


def main():
    assert "pm25" not in feature_columns
    assert "pm25" not in expanded_feature_columns
    assert weather_columns.isdisjoint(feature_columns)
    assert weather_columns.isdisjoint(expanded_feature_columns)

    modeling_df = build_modeling_dataframe()
    training_columns = [
        "event_time",
        *expanded_feature_columns,
        target_column,
    ]

    target_complete_df = modeling_df.dropna(
        subset=[target_column]
    )[training_columns]
    trainable_df = target_complete_df.dropna(
        subset=expanded_feature_columns
    ).sort_values("event_time").reset_index(drop=True)

    total_rows = len(modeling_df)
    target_complete_rows = len(target_complete_df)
    trainable_rows = len(trainable_df)

    print("=== TRAINABLE DATASET ===")
    print(f"Total modeling rows: {total_rows}")
    print(f"Rows after target filtering: {target_complete_rows}")
    print(f"Rows after complete-feature filtering: {trainable_rows}")
    print(
        "Rows removed because of incomplete history: "
        f"{target_complete_rows - trainable_rows}"
    )

    train_end = int(trainable_rows * 0.70)
    validation_end = int(trainable_rows * 0.85)

    train_df = trainable_df.iloc[:train_end]
    validation_df = trainable_df.iloc[train_end:validation_end]
    test_df = trainable_df.iloc[validation_end:]

    assert train_df["event_time"].max() < validation_df["event_time"].min()
    assert validation_df["event_time"].max() < test_df["event_time"].min()

    print("\n=== CHRONOLOGICAL SPLIT ===")
    print_split("Train", train_df)
    print_split("Validation", validation_df)
    print_split("Test", test_df)

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]
    X_train_expanded = train_df[expanded_feature_columns]
    X_validation_expanded = validation_df[expanded_feature_columns]

    for X, y in [
        (X_train, y_train),
        (X_validation, y_validation),
        (X_test, y_test),
    ]:
        assert len(X) == len(y)
        assert not X.isna().any().any()
        assert not y.isna().any()
    assert len(X_train_expanded) == len(y_train)
    assert len(X_validation_expanded) == len(y_validation)
    assert not X_train_expanded.isna().any().any()
    assert not X_validation_expanded.isna().any().any()

    persistence_validation_predictions = validation_df["pm25_lag_1h"]
    persistence_test_predictions = test_df["pm25_lag_1h"]
    persistence_validation_mae, persistence_validation_rmse = (
        calculate_metrics(
            y_validation,
            persistence_validation_predictions,
        )
    )
    persistence_test_mae, persistence_test_rmse = calculate_metrics(
        y_test,
        persistence_test_predictions,
    )

    print("\n=== PERSISTENCE BASELINE ===")
    print(f"Validation MAE: {persistence_validation_mae:.4f}")
    print(f"Validation RMSE: {persistence_validation_rmse:.4f}")
    print(f"Test MAE: {persistence_test_mae:.4f}")
    print(f"Test RMSE: {persistence_test_rmse:.4f}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    validation_predictions = model.predict(X_validation)
    test_predictions = model.predict(X_test)
    validation_mae, validation_rmse = calculate_metrics(
        y_validation,
        validation_predictions,
    )
    test_mae, test_rmse = calculate_metrics(y_test, test_predictions)

    print("\n=== LINEAR REGRESSION ===")
    print(f"Validation MAE: {validation_mae:.4f}")
    print(f"Validation RMSE: {validation_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    expanded_model = LinearRegression()
    expanded_model.fit(X_train_expanded, y_train)
    assert expanded_model.feature_names_in_.tolist() == (
        expanded_feature_columns
    )
    expanded_validation_predictions = expanded_model.predict(
        X_validation_expanded
    )
    expanded_validation_mae, expanded_validation_rmse = calculate_metrics(
        y_validation,
        expanded_validation_predictions,
    )
    expanded_mae_improvement = validation_mae - expanded_validation_mae
    expanded_rmse_improvement = validation_rmse - expanded_validation_rmse
    expanded_mae_percentage_improvement = (
        expanded_mae_improvement / validation_mae * 100
    )
    expanded_rmse_percentage_improvement = (
        expanded_rmse_improvement / validation_rmse * 100
    )

    print("\n=== ORIGINAL LINEAR REGRESSION VALIDATION ===")
    print(f"MAE: {validation_mae:.4f}")
    print(f"RMSE: {validation_rmse:.4f}")
    print("\n=== EXPANDED FEATURE LINEAR REGRESSION VALIDATION ===")
    print(f"MAE: {expanded_validation_mae:.4f}")
    print(f"RMSE: {expanded_validation_rmse:.4f}")
    print("\n=== VALIDATION IMPROVEMENT ===")
    print(f"MAE absolute improvement: {expanded_mae_improvement:.4f}")
    print(
        "MAE percentage improvement: "
        f"{expanded_mae_percentage_improvement:.2f}%"
    )
    print(f"RMSE absolute improvement: {expanded_rmse_improvement:.4f}")
    print(
        "RMSE percentage improvement: "
        f"{expanded_rmse_percentage_improvement:.2f}%"
    )

    expanded_validation_diagnostics = pd.DataFrame(
        {
            "actual_pm25": y_validation.to_numpy(),
            "prediction": expanded_validation_predictions,
        }
    )
    expanded_validation_diagnostics["signed_error"] = (
        expanded_validation_diagnostics["prediction"]
        - expanded_validation_diagnostics["actual_pm25"]
    )
    expanded_validation_diagnostics["absolute_error"] = (
        expanded_validation_diagnostics["signed_error"].abs()
    )
    expanded_validation_diagnostics["actual_pm25_range"] = pd.cut(
        expanded_validation_diagnostics["actual_pm25"],
        bins=[float("-inf"), 15, 35, 55, float("inf")],
        labels=["< 15", "15–35", "35–55", ">= 55"],
        right=False,
    )
    expanded_range_summary = (
        expanded_validation_diagnostics.groupby(
            "actual_pm25_range",
            observed=False,
            sort=True,
        ).agg(
            Count=("actual_pm25", "size"),
            MAE=("absolute_error", "mean"),
            Mean_signed_error=("signed_error", "mean"),
        ).reset_index()
    )
    assert not expanded_validation_diagnostics.isna().any().any()

    print("\n=== EXPANDED VALIDATION ERROR BY ACTUAL PM2.5 ===")
    print(
        expanded_range_summary.to_string(
            index=False,
            float_format="%.4f",
        )
    )

    test_diagnostics = pd.DataFrame(
        {
            "event_time": test_df["event_time"].to_numpy(),
            "actual_pm25": y_test.to_numpy(),
            "linear_prediction": test_predictions,
            "persistence_prediction": (
                persistence_test_predictions.to_numpy()
            ),
        }
    )
    test_diagnostics["linear_signed_error"] = (
        test_diagnostics["linear_prediction"]
        - test_diagnostics["actual_pm25"]
    )
    test_diagnostics["linear_absolute_error"] = test_diagnostics[
        "linear_signed_error"
    ].abs()
    test_diagnostics["persistence_signed_error"] = (
        test_diagnostics["persistence_prediction"]
        - test_diagnostics["actual_pm25"]
    )
    test_diagnostics["persistence_absolute_error"] = test_diagnostics[
        "persistence_signed_error"
    ].abs()
    assert len(test_diagnostics) == len(test_df)
    assert not test_diagnostics.isna().any().any()

    mean_signed_error = test_diagnostics["linear_signed_error"].mean()
    median_signed_error = test_diagnostics["linear_signed_error"].median()
    underprediction_count = (
        test_diagnostics["linear_signed_error"] < 0
    ).sum()
    overprediction_count = (
        test_diagnostics["linear_signed_error"] > 0
    ).sum()
    underprediction_percentage = underprediction_count / len(test_df) * 100
    overprediction_percentage = overprediction_count / len(test_df) * 100

    print("\n=== LINEAR REGRESSION TEST BIAS ===")
    print(f"Mean signed error: {mean_signed_error:.4f}")
    print(f"Median signed error: {median_signed_error:.4f}")
    print(
        f"Underpredictions: {underprediction_count} "
        f"({underprediction_percentage:.2f}%)"
    )
    print(
        f"Overpredictions: {overprediction_count} "
        f"({overprediction_percentage:.2f}%)"
    )

    test_diagnostics["actual_pm25_range"] = pd.cut(
        test_diagnostics["actual_pm25"],
        bins=[float("-inf"), 15, 35, 55, float("inf")],
        labels=["< 15", "15–35", "35–55", ">= 55"],
        right=False,
    )
    range_rows = []
    for pm25_range, group in test_diagnostics.groupby(
        "actual_pm25_range",
        observed=False,
        sort=True,
    ):
        range_mae, range_rmse = calculate_metrics(
            group["actual_pm25"],
            group["linear_prediction"],
        )
        range_rows.append(
            {
                "Actual PM2.5 range": pm25_range,
                "Count": len(group),
                "MAE": range_mae,
                "RMSE": range_rmse,
                "Mean signed error": group["linear_signed_error"].mean(),
            }
        )
    range_summary = pd.DataFrame(range_rows)

    print("\n=== LINEAR REGRESSION TEST ERROR BY ACTUAL PM2.5 ===")
    print(range_summary.to_string(index=False, float_format="%.4f"))

    high_pollution = test_diagnostics[
        test_diagnostics["actual_pm25"] >= 35
    ]
    high_linear_mae = high_pollution["linear_absolute_error"].mean()
    high_persistence_mae = high_pollution[
        "persistence_absolute_error"
    ].mean()
    high_linear_signed_error = high_pollution[
        "linear_signed_error"
    ].mean()
    high_persistence_signed_error = high_pollution[
        "persistence_signed_error"
    ].mean()
    elevated_underestimation = high_linear_signed_error < 0

    print("\n=== HIGH-POLLUTION TEST BEHAVIOR (ACTUAL PM2.5 >= 35) ===")
    print(f"Rows: {len(high_pollution)}")
    print(f"Linear Regression MAE: {high_linear_mae:.4f}")
    print(f"Persistence MAE: {high_persistence_mae:.4f}")
    print(
        "Linear Regression mean signed error: "
        f"{high_linear_signed_error:.4f}"
    )
    print(
        "Persistence mean signed error: "
        f"{high_persistence_signed_error:.4f}"
    )
    print(
        "Systematic Linear Regression underestimation: "
        f"{'Yes' if elevated_underestimation else 'No'}"
    )

    local_time = test_diagnostics["event_time"].dt.tz_convert(
        "Asia/Ho_Chi_Minh"
    )
    test_diagnostics["local_hour"] = local_time.dt.hour
    test_diagnostics["local_month"] = local_time.dt.month
    hourly_summary = test_diagnostics.groupby("local_hour").agg(
        Count=("actual_pm25", "size"),
        Linear_MAE=("linear_absolute_error", "mean"),
        Persistence_MAE=("persistence_absolute_error", "mean"),
    ).reset_index().sort_values("local_hour")
    worst_hours = hourly_summary.nlargest(5, "Linear_MAE")
    monthly_summary = test_diagnostics.groupby("local_month").agg(
        Count=("actual_pm25", "size"),
        Linear_MAE=("linear_absolute_error", "mean"),
        Persistence_MAE=("persistence_absolute_error", "mean"),
    ).reset_index().sort_values("local_month")

    print("\n=== TEST ERROR BY LOCAL HOUR (ASIA/HO_CHI_MINH) ===")
    print(hourly_summary.to_string(index=False, float_format="%.4f"))
    print("\n=== 5 WORST LOCAL HOURS BY LINEAR REGRESSION MAE ===")
    print(worst_hours.to_string(index=False, float_format="%.4f"))
    print("\n=== TEST ERROR BY LOCAL MONTH (ASIA/HO_CHI_MINH) ===")
    print(monthly_summary.to_string(index=False, float_format="%.4f"))

    biggest_misses = test_diagnostics.nlargest(
        10,
        "linear_absolute_error",
    )[
        [
            "event_time",
            "actual_pm25",
            "linear_prediction",
            "persistence_prediction",
            "linear_absolute_error",
            "linear_signed_error",
        ]
    ]
    print("\n=== 10 LARGEST LINEAR REGRESSION TEST MISSES ===")
    print(biggest_misses.to_string(index=False, float_format="%.4f"))

    print("\n=== COMPARISON TO PERSISTENCE ===")
    print_improvement(
        "Validation MAE improvement",
        persistence_validation_mae,
        validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        persistence_validation_rmse,
        validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        persistence_test_mae,
        test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        persistence_test_rmse,
        test_rmse,
    )

    coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": model.coef_,
        }
    )
    coefficients["absolute_coefficient"] = coefficients[
        "coefficient"
    ].abs()
    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )[["feature", "coefficient"]]

    print("\n=== LINEAR REGRESSION COEFFICIENTS ===")
    print(f"Intercept: {model.intercept_:.6f}")
    print(coefficients.to_string(index=False))

    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    random_forest.fit(X_train, y_train)

    random_forest_train_predictions = random_forest.predict(X_train)
    random_forest_validation_predictions = random_forest.predict(X_validation)
    random_forest_test_predictions = random_forest.predict(X_test)
    random_forest_train_mae, random_forest_train_rmse = calculate_metrics(
        y_train,
        random_forest_train_predictions,
    )
    random_forest_validation_mae, random_forest_validation_rmse = (
        calculate_metrics(
            y_validation,
            random_forest_validation_predictions,
        )
    )
    random_forest_test_mae, random_forest_test_rmse = calculate_metrics(
        y_test,
        random_forest_test_predictions,
    )

    print("\n=== RANDOM FOREST ===")
    print(f"Train MAE: {random_forest_train_mae:.4f}")
    print(f"Train RMSE: {random_forest_train_rmse:.4f}")
    print(f"Validation MAE: {random_forest_validation_mae:.4f}")
    print(f"Validation RMSE: {random_forest_validation_rmse:.4f}")
    print(f"Test MAE: {random_forest_test_mae:.4f}")
    print(f"Test RMSE: {random_forest_test_rmse:.4f}")

    gradient_boosting = HistGradientBoostingRegressor(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )
    gradient_boosting.fit(X_train, y_train)
    assert gradient_boosting.feature_names_in_.tolist() == feature_columns
    assert gradient_boosting.n_features_in_ == X_train.shape[1]

    gradient_boosting_train_predictions = gradient_boosting.predict(X_train)
    gradient_boosting_validation_predictions = gradient_boosting.predict(
        X_validation
    )
    gradient_boosting_test_predictions = gradient_boosting.predict(X_test)
    gradient_boosting_train_mae, gradient_boosting_train_rmse = (
        calculate_metrics(y_train, gradient_boosting_train_predictions)
    )
    (
        gradient_boosting_validation_mae,
        gradient_boosting_validation_rmse,
    ) = calculate_metrics(
        y_validation,
        gradient_boosting_validation_predictions,
    )
    gradient_boosting_test_mae, gradient_boosting_test_rmse = (
        calculate_metrics(y_test, gradient_boosting_test_predictions)
    )

    print("\n=== HISTOGRAM GRADIENT BOOSTING ===")
    print(f"Train MAE: {gradient_boosting_train_mae:.4f}")
    print(f"Train RMSE: {gradient_boosting_train_rmse:.4f}")
    print(f"Validation MAE: {gradient_boosting_validation_mae:.4f}")
    print(f"Validation RMSE: {gradient_boosting_validation_rmse:.4f}")
    print(f"Test MAE: {gradient_boosting_test_mae:.4f}")
    print(f"Test RMSE: {gradient_boosting_test_rmse:.4f}")

    validation_mae_gap = (
        gradient_boosting_validation_mae - gradient_boosting_train_mae
    )
    test_mae_gap = gradient_boosting_test_mae - gradient_boosting_train_mae
    substantial_overfitting = (
        gradient_boosting_validation_mae
        > gradient_boosting_train_mae * 1.5
        or gradient_boosting_test_mae > gradient_boosting_train_mae * 1.5
    )
    print("\n=== HISTOGRAM GRADIENT BOOSTING OVERFITTING CHECK ===")
    print(f"Validation-train MAE gap: {validation_mae_gap:.4f}")
    print(f"Test-train MAE gap: {test_mae_gap:.4f}")
    print(
        "Evidence of substantial overfitting: "
        f"{'Yes' if substantial_overfitting else 'No'}"
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    assert scaler.n_samples_seen_ == len(X_train)
    assert pd.Series(
        scaler.mean_,
        index=feature_columns,
    ).equals(X_train.mean())

    X_train_scaled = scaler.transform(X_train)
    X_validation_scaled = scaler.transform(X_validation)
    X_test_scaled = scaler.transform(X_test)
    assert not pd.isna(X_train_scaled).any()
    assert not pd.isna(X_validation_scaled).any()
    assert not pd.isna(X_test_scaled).any()

    alphas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ridge_results = []

    print("\n=== RIDGE VALIDATION SEARCH ===")
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_validation_predictions = ridge.predict(X_validation_scaled)
        ridge_validation_mae, ridge_validation_rmse = calculate_metrics(
            y_validation,
            ridge_validation_predictions,
        )
        ridge_results.append(
            (
                ridge_validation_mae,
                alpha,
                ridge_validation_rmse,
                ridge,
            )
        )
        print(f"alpha={alpha}")
        print(f"Validation MAE: {ridge_validation_mae:.4f}")
        print(f"Validation RMSE: {ridge_validation_rmse:.4f}")

    (
        ridge_validation_mae,
        best_alpha,
        ridge_validation_rmse,
        best_ridge,
    ) = min(ridge_results, key=lambda result: result[0])
    ridge_test_predictions = best_ridge.predict(X_test_scaled)
    ridge_test_mae, ridge_test_rmse = calculate_metrics(
        y_test,
        ridge_test_predictions,
    )

    print("\n=== BEST RIDGE ===")
    print(f"Best alpha: {best_alpha}")
    print(f"Validation MAE: {ridge_validation_mae:.4f}")
    print(f"Validation RMSE: {ridge_validation_rmse:.4f}")
    print(f"Test MAE: {ridge_test_mae:.4f}")
    print(f"Test RMSE: {ridge_test_rmse:.4f}")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Persistence",
                "Linear Regression",
                "Ridge Regression",
                "Random Forest",
                "Histogram Gradient Boosting",
            ],
            "Validation MAE": [
                persistence_validation_mae,
                validation_mae,
                ridge_validation_mae,
                random_forest_validation_mae,
                gradient_boosting_validation_mae,
            ],
            "Validation RMSE": [
                persistence_validation_rmse,
                validation_rmse,
                ridge_validation_rmse,
                random_forest_validation_rmse,
                gradient_boosting_validation_rmse,
            ],
            "Test MAE": [
                persistence_test_mae,
                test_mae,
                ridge_test_mae,
                random_forest_test_mae,
                gradient_boosting_test_mae,
            ],
            "Test RMSE": [
                persistence_test_rmse,
                test_rmse,
                ridge_test_rmse,
                random_forest_test_rmse,
                gradient_boosting_test_rmse,
            ],
        }
    )

    print("\n=== MODEL COMPARISON ===")
    print(comparison.to_string(index=False, float_format="%.4f"))

    print("\n=== RANDOM FOREST IMPROVEMENT VS PERSISTENCE ===")
    print_improvement(
        "Validation MAE improvement",
        persistence_validation_mae,
        random_forest_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        persistence_validation_rmse,
        random_forest_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        persistence_test_mae,
        random_forest_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        persistence_test_rmse,
        random_forest_test_rmse,
    )

    print("\n=== RANDOM FOREST IMPROVEMENT VS LINEAR REGRESSION ===")
    print_improvement(
        "Validation MAE improvement",
        validation_mae,
        random_forest_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        validation_rmse,
        random_forest_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        test_mae,
        random_forest_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        test_rmse,
        random_forest_test_rmse,
    )

    print("\n=== RIDGE IMPROVEMENT VS PERSISTENCE ===")
    print_improvement(
        "Validation MAE improvement",
        persistence_validation_mae,
        ridge_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        persistence_validation_rmse,
        ridge_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        persistence_test_mae,
        ridge_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        persistence_test_rmse,
        ridge_test_rmse,
    )

    print("\n=== RIDGE IMPROVEMENT VS LINEAR REGRESSION ===")
    print_improvement(
        "Validation MAE improvement",
        validation_mae,
        ridge_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        validation_rmse,
        ridge_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        test_mae,
        ridge_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        test_rmse,
        ridge_test_rmse,
    )

    print("\n=== HISTOGRAM GRADIENT BOOSTING VS PERSISTENCE ===")
    print_improvement(
        "Validation MAE improvement",
        persistence_validation_mae,
        gradient_boosting_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        persistence_validation_rmse,
        gradient_boosting_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        persistence_test_mae,
        gradient_boosting_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        persistence_test_rmse,
        gradient_boosting_test_rmse,
    )

    print("\n=== HISTOGRAM GRADIENT BOOSTING VS LINEAR REGRESSION ===")
    print_improvement(
        "Validation MAE improvement",
        validation_mae,
        gradient_boosting_validation_mae,
    )
    print_improvement(
        "Validation RMSE improvement",
        validation_rmse,
        gradient_boosting_validation_rmse,
    )
    print_improvement(
        "Test MAE improvement",
        test_mae,
        gradient_boosting_test_mae,
    )
    print_improvement(
        "Test RMSE improvement",
        test_rmse,
        gradient_boosting_test_rmse,
    )

    ridge_coefficients = pd.DataFrame(
        {
            "feature": feature_columns,
            "coefficient": best_ridge.coef_,
        }
    )
    ridge_coefficients["absolute_coefficient"] = ridge_coefficients[
        "coefficient"
    ].abs()
    ridge_coefficients = ridge_coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )[["feature", "coefficient"]]

    print("\n=== RIDGE COEFFICIENTS (STANDARDIZED FEATURES) ===")
    print(f"Intercept: {best_ridge.intercept_:.6f}")
    print(ridge_coefficients.to_string(index=False))
    print("Coefficients are associations, not causal effects.")

    feature_importances = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": random_forest.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    print("\n=== RANDOM FOREST FEATURE IMPORTANCE ===")
    print(feature_importances.to_string(index=False))


if __name__ == "__main__":
    main()
