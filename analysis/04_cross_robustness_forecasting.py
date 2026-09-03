# ==============================================================
# 04_cross_crypto_robustness_forecasting.py
#
# Dissertation:
# Does Social Media Sentiment Signals Improve the Prediction of
# Cryptocurrency Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum
#
# Purpose:
# Estimate expanding-window out-of-sample forecasting models
# and test whether lagged returns of the other major
# cryptocurrency improve forecasting performance.
#
# BTC robustness:
#   Baseline BTC model + ETH_Lagged_Return
#
# ETH robustness:
#   Baseline ETH model + BTC_Lagged_Return
#
# Performance measures:
#   RMSE
#   MAE
#   Out-of-sample R-squared
#   Directional accuracy
#
# IMPORTANT:
# All predictors are lagged / predetermined variables.
# No contemporaneous market information is used for forecasting.
# ==============================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ==============================================================
# 1. PROJECT PATHS
# ==============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

DATA_PROCESSED = PROJECT_ROOT / "data_processed"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------
# IMPORTANT:
# This should be the forecasting dataset you already created
# after correcting the traditional-market calendar alignment.
# --------------------------------------------------------------

INPUT_FILE = DATA_PROCESSED / "forecast_structure.csv"


# Output files

FORECAST_OUTPUT = (
    RESULTS_DIR / "cross_crypto_robustness_forecasts.csv"
)

SUMMARY_OUTPUT = (
    RESULTS_DIR / "cross_crypto_robustness_forecasting_summary.csv"
)

COMPARISON_OUTPUT = (
    RESULTS_DIR / "cross_crypto_robustness_forecasting_comparison.csv"
)


# ==============================================================
# 2. FORECAST SETTINGS
# ==============================================================

# Use the same out-of-sample start date as your existing
# baseline forecasting exercise.

FORECAST_START = pd.Timestamp("2024-01-02")

FORECAST_END = pd.Timestamp("2025-12-31")


# ==============================================================
# 3. VARIABLE SPECIFICATIONS
# ==============================================================

# --------------------------------------------------------------
# BTC BASELINE
# --------------------------------------------------------------

BTC_BASELINE_PREDICTORS = [
    "BTC_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------------------
# BTC CROSS-CRYPTO ROBUSTNESS
#
# Adds yesterday's ETH return.
# --------------------------------------------------------------

BTC_CROSS_PREDICTORS = [
    "BTC_Lagged_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------------------
# ETH BASELINE
# --------------------------------------------------------------

ETH_BASELINE_PREDICTORS = [
    "ETH_Lagged_Return",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------------------
# ETH CROSS-CRYPTO ROBUSTNESS
#
# Adds yesterday's BTC return.
# --------------------------------------------------------------

ETH_CROSS_PREDICTORS = [
    "ETH_Lagged_Return",
    "BTC_Lagged_Return",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# ==============================================================
# 4. PRINT HEADER
# ==============================================================

print("=" * 80)
print("CROSS-CRYPTO ROBUSTNESS FORECASTING MODELS")
print("=" * 80)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())


if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nForecasting dataset not found:\n{INPUT_FILE}\n\n"
        "Check the INPUT_FILE name at the top of the script."
    )


# ==============================================================
# 5. LOAD DATA
# ==============================================================

print("\n" + "=" * 80)
print("IMPORTING FORECASTING DATASET")
print("=" * 80)

df = pd.read_csv(INPUT_FILE)

print("\nImported shape:")
print(df.shape)

print("\nColumns found:")
print(df.columns.tolist())


# ==============================================================
# 6. DATE CLEANING
# ==============================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates were found in the forecasting dataset."
    )

df = df.sort_values("Date").reset_index(drop=True)

print("\nDataset date range:")
print(df["Date"].min(), "to", df["Date"].max())


# ==============================================================
# 7. CHECK REQUIRED VARIABLES
# ==============================================================

print("\n" + "=" * 80)
print("CHECKING REQUIRED VARIABLES")
print("=" * 80)

required_columns = list(
    set(
        ["Date", "BTC_Return", "ETH_Return"]
        + BTC_BASELINE_PREDICTORS
        + BTC_CROSS_PREDICTORS
        + ETH_BASELINE_PREDICTORS
        + ETH_CROSS_PREDICTORS
    )
)

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    print("\nMissing required columns:")

    for column in missing_columns:
        print(column)

    raise KeyError(
        "\nOne or more required variables are missing.\n"
        "Check the column names in forecast_structure.csv."
    )

else:
    print("\nAll required forecasting variables are present.")


# ==============================================================
# 8. DISPLAY MODEL SPECIFICATIONS
# ==============================================================

print("\n" + "=" * 80)
print("MODEL SPECIFICATIONS")
print("=" * 80)

print("\nBTC BASELINE MODEL:")
print("Dependent variable: BTC_Return")

for variable in BTC_BASELINE_PREDICTORS:
    print("  ", variable)


print("\nBTC CROSS-CRYPTO ROBUSTNESS MODEL:")
print("Dependent variable: BTC_Return")

for variable in BTC_CROSS_PREDICTORS:
    print("  ", variable)


print("\nETH BASELINE MODEL:")
print("Dependent variable: ETH_Return")

for variable in ETH_BASELINE_PREDICTORS:
    print("  ", variable)


print("\nETH CROSS-CRYPTO ROBUSTNESS MODEL:")
print("Dependent variable: ETH_Return")

for variable in ETH_CROSS_PREDICTORS:
    print("  ", variable)


# ==============================================================
# 9. EXPANDING-WINDOW FORECAST FUNCTION
# ==============================================================

def expanding_window_forecast(
    data,
    target,
    predictors,
    asset,
    model_name,
    forecast_start,
    forecast_end
):

    """
    Produce one-day-ahead expanding-window forecasts.

    For every forecast date t:

    1. Estimate the OLS model using observations strictly
       before date t.

    2. Use the predictor values available for date t.

    3. Generate a forecast for the cryptocurrency return
       on date t.

    The estimation window therefore expands through time.
    """

    model_columns = (
        ["Date", target]
        + predictors
    )

    model_data = data[model_columns].copy()

    # Convert numerical variables explicitly
    numerical_columns = [target] + predictors

    for column in numerical_columns:
        model_data[column] = pd.to_numeric(
            model_data[column],
            errors="coerce"
        )

    # Restrict forecast dates
    forecast_dates = model_data.loc[
        (model_data["Date"] >= forecast_start)
        & (model_data["Date"] <= forecast_end),
        "Date"
    ].drop_duplicates().sort_values()

    forecasts = []

    print("\n" + "-" * 80)
    print(asset, "-", model_name)
    print("-" * 80)

    print("\nPotential forecast dates:")
    print(len(forecast_dates))

    for forecast_date in forecast_dates:

        # ------------------------------------------------------
        # Training sample:
        # all information strictly BEFORE forecast date
        # ------------------------------------------------------

        train = model_data[
            model_data["Date"] < forecast_date
        ].copy()

        train = train.dropna(
            subset=[target] + predictors
        )

        # ------------------------------------------------------
        # Test observation:
        # current forecast date
        # ------------------------------------------------------

        test = model_data[
            model_data["Date"] == forecast_date
        ].copy()

        test = test.dropna(
            subset=[target] + predictors
        )

        if test.empty:
            continue

        # Need enough training observations relative to
        # number of predictors.
        if len(train) <= len(predictors) + 1:
            continue

        # ------------------------------------------------------
        # Construct training matrices
        # ------------------------------------------------------

        X_train = train[predictors].astype(float)
        y_train = train[target].astype(float)

        X_train = sm.add_constant(
            X_train,
            has_constant="add"
        )

        # ------------------------------------------------------
        # Estimate OLS model
        #
        # HAC is NOT required for point forecasts.
        # HAC changes inference / standard errors, not the
        # OLS coefficient estimates used for prediction.
        # ------------------------------------------------------

        model = sm.OLS(
            y_train,
            X_train
        ).fit()

        # ------------------------------------------------------
        # Construct test predictor matrix
        # ------------------------------------------------------

        X_test = test[predictors].astype(float)

        X_test = sm.add_constant(
            X_test,
            has_constant="add"
        )

        # Ensure exact same columns/order as training matrix
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=1.0
        )

        # ------------------------------------------------------
        # Generate forecast
        # ------------------------------------------------------

        predicted_return = float(
            model.predict(X_test).iloc[0]
        )

        actual_return = float(
            test[target].iloc[0]
        )

        forecast_error = (
            actual_return - predicted_return
        )

        forecasts.append(
            {
                "Asset": asset,
                "Model": model_name,
                "Date": forecast_date,
                "Actual_Return": actual_return,
                "Predicted_Return": predicted_return,
                "Forecast_Error": forecast_error,
                "Squared_Error": forecast_error ** 2,
                "Absolute_Error": abs(forecast_error),
                "Actual_Direction": int(actual_return > 0),
                "Predicted_Direction": int(predicted_return > 0),
                "Direction_Correct": int(
                    (actual_return > 0)
                    == (predicted_return > 0)
                ),
                "Training_Observations": len(train),
            }
        )

    forecast_df = pd.DataFrame(forecasts)

    if forecast_df.empty:
        raise ValueError(
            f"No forecasts were generated for "
            f"{asset} - {model_name}."
        )

    print("\nForecasts generated:")
    print(len(forecast_df))

    print("\nFirst forecast date:")
    print(forecast_df["Date"].min())

    print("\nLast forecast date:")
    print(forecast_df["Date"].max())

    return forecast_df


# ==============================================================
# 10. GENERATE BTC BASELINE FORECASTS
# ==============================================================

print("\n" + "=" * 80)
print("BTC BASELINE FORECASTING")
print("=" * 80)

btc_baseline_forecasts = expanding_window_forecast(
    data=df,
    target="BTC_Return",
    predictors=BTC_BASELINE_PREDICTORS,
    asset="BTC",
    model_name="Baseline",
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
)


# ==============================================================
# 11. GENERATE BTC CROSS-CRYPTO FORECASTS
# ==============================================================

print("\n" + "=" * 80)
print("BTC CROSS-CRYPTO ROBUSTNESS FORECASTING")
print("=" * 80)

btc_cross_forecasts = expanding_window_forecast(
    data=df,
    target="BTC_Return",
    predictors=BTC_CROSS_PREDICTORS,
    asset="BTC",
    model_name="Cross_Crypto",
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
)


# ==============================================================
# 12. GENERATE ETH BASELINE FORECASTS
# ==============================================================

print("\n" + "=" * 80)
print("ETH BASELINE FORECASTING")
print("=" * 80)

eth_baseline_forecasts = expanding_window_forecast(
    data=df,
    target="ETH_Return",
    predictors=ETH_BASELINE_PREDICTORS,
    asset="ETH",
    model_name="Baseline",
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
)


# ==============================================================
# 13. GENERATE ETH CROSS-CRYPTO FORECASTS
# ==============================================================

print("\n" + "=" * 80)
print("ETH CROSS-CRYPTO ROBUSTNESS FORECASTING")
print("=" * 80)

eth_cross_forecasts = expanding_window_forecast(
    data=df,
    target="ETH_Return",
    predictors=ETH_CROSS_PREDICTORS,
    asset="ETH",
    model_name="Cross_Crypto",
    forecast_start=FORECAST_START,
    forecast_end=FORECAST_END,
)


# ==============================================================
# 14. ALIGN BASELINE AND CROSS-CRYPTO FORECAST DATES
# ==============================================================

print("\n" + "=" * 80)
print("ALIGNING FORECAST DATES FOR FAIR COMPARISON")
print("=" * 80)


def align_forecasts(baseline, cross):

    common_dates = sorted(
        set(baseline["Date"])
        .intersection(set(cross["Date"]))
    )

    baseline_aligned = baseline[
        baseline["Date"].isin(common_dates)
    ].copy()

    cross_aligned = cross[
        cross["Date"].isin(common_dates)
    ].copy()

    baseline_aligned = (
        baseline_aligned
        .sort_values("Date")
        .reset_index(drop=True)
    )

    cross_aligned = (
        cross_aligned
        .sort_values("Date")
        .reset_index(drop=True)
    )

    return baseline_aligned, cross_aligned


btc_baseline_aligned, btc_cross_aligned = align_forecasts(
    btc_baseline_forecasts,
    btc_cross_forecasts
)

eth_baseline_aligned, eth_cross_aligned = align_forecasts(
    eth_baseline_forecasts,
    eth_cross_forecasts
)


print("\nBTC common forecast observations:")
print(len(btc_baseline_aligned))

print("\nETH common forecast observations:")
print(len(eth_baseline_aligned))


# ==============================================================
# 15. FORECAST PERFORMANCE FUNCTION
# ==============================================================

def calculate_performance(forecast_df):

    actual = forecast_df["Actual_Return"].to_numpy()
    predicted = forecast_df["Predicted_Return"].to_numpy()

    errors = actual - predicted

    # ----------------------------------------------------------
    # RMSE
    # ----------------------------------------------------------

    rmse = np.sqrt(
        np.mean(errors ** 2)
    )

    # ----------------------------------------------------------
    # MAE
    # ----------------------------------------------------------

    mae = np.mean(
        np.abs(errors)
    )

    # ----------------------------------------------------------
    # OUT-OF-SAMPLE R-SQUARED
    #
    # Benchmark = historical mean return available BEFORE
    # each forecast date.
    #
    # We therefore construct a recursive historical-mean
    # forecast rather than using the full-sample mean.
    # ----------------------------------------------------------

    historical_mean_forecasts = []

    for _, row in forecast_df.iterrows():

        forecast_date = row["Date"]

        asset = row["Asset"]

        if asset == "BTC":
            target = "BTC_Return"
        else:
            target = "ETH_Return"

        historical_data = df[
            df["Date"] < forecast_date
        ][target].dropna()

        historical_mean = historical_data.mean()

        historical_mean_forecasts.append(
            historical_mean
        )

    historical_mean_forecasts = np.array(
        historical_mean_forecasts
    )

    benchmark_errors = (
        actual - historical_mean_forecasts
    )

    model_sse = np.sum(
        errors ** 2
    )

    benchmark_sse = np.sum(
        benchmark_errors ** 2
    )

    if benchmark_sse == 0:
        oos_r2 = np.nan
    else:
        oos_r2 = (
            1 - model_sse / benchmark_sse
        )

    # ----------------------------------------------------------
    # DIRECTIONAL ACCURACY
    # ----------------------------------------------------------

    directional_accuracy = np.mean(
        np.sign(actual)
        == np.sign(predicted)
    )

    return {
        "N_Forecasts": len(forecast_df),
        "Forecast_Start": forecast_df["Date"].min(),
        "Forecast_End": forecast_df["Date"].max(),
        "RMSE": rmse,
        "MAE": mae,
        "OOS_R2": oos_r2,
        "Directional_Accuracy": directional_accuracy,
    }


# ==============================================================
# 16. CALCULATE PERFORMANCE
# ==============================================================

print("\n" + "=" * 80)
print("CALCULATING FORECAST PERFORMANCE")
print("=" * 80)


performance_rows = []


for forecast_data in [
    btc_baseline_aligned,
    btc_cross_aligned,
    eth_baseline_aligned,
    eth_cross_aligned,
]:

    performance = calculate_performance(
        forecast_data
    )

    performance_rows.append(
        {
            "Asset": forecast_data["Asset"].iloc[0],
            "Model": forecast_data["Model"].iloc[0],
            **performance,
        }
    )


performance_df = pd.DataFrame(
    performance_rows
)


print("\nForecast performance:")
print(
    performance_df.to_string(
        index=False
    )
)


# ==============================================================
# 17. COMPARE BASELINE VS CROSS-CRYPTO
# ==============================================================

print("\n" + "=" * 80)
print("BASELINE VS CROSS-CRYPTO COMPARISON")
print("=" * 80)


comparison_rows = []


for asset in ["BTC", "ETH"]:

    asset_results = performance_df[
        performance_df["Asset"] == asset
    ]

    baseline = asset_results[
        asset_results["Model"] == "Baseline"
    ].iloc[0]

    cross = asset_results[
        asset_results["Model"] == "Cross_Crypto"
    ].iloc[0]

    rmse_change = (
        cross["RMSE"]
        - baseline["RMSE"]
    )

    mae_change = (
        cross["MAE"]
        - baseline["MAE"]
    )

    oos_r2_change = (
        cross["OOS_R2"]
        - baseline["OOS_R2"]
    )

    directional_change = (
        cross["Directional_Accuracy"]
        - baseline["Directional_Accuracy"]
    )

    # Percentage RMSE improvement
    # Positive = improvement

    rmse_improvement_pct = (
        (
            baseline["RMSE"]
            - cross["RMSE"]
        )
        / baseline["RMSE"]
    ) * 100

    # Percentage MAE improvement
    # Positive = improvement

    mae_improvement_pct = (
        (
            baseline["MAE"]
            - cross["MAE"]
        )
        / baseline["MAE"]
    ) * 100

    comparison_rows.append(
        {
            "Asset": asset,

            "Baseline_RMSE": baseline["RMSE"],
            "Cross_Crypto_RMSE": cross["RMSE"],
            "RMSE_Change": rmse_change,
            "RMSE_Improvement_Percent":
                rmse_improvement_pct,

            "Baseline_MAE": baseline["MAE"],
            "Cross_Crypto_MAE": cross["MAE"],
            "MAE_Change": mae_change,
            "MAE_Improvement_Percent":
                mae_improvement_pct,

            "Baseline_OOS_R2": baseline["OOS_R2"],
            "Cross_Crypto_OOS_R2": cross["OOS_R2"],
            "OOS_R2_Change": oos_r2_change,

            "Baseline_Directional_Accuracy":
                baseline["Directional_Accuracy"],

            "Cross_Crypto_Directional_Accuracy":
                cross["Directional_Accuracy"],

            "Directional_Accuracy_Change":
                directional_change,
        }
    )


comparison_df = pd.DataFrame(
    comparison_rows
)


print(
    "\n",
    comparison_df.to_string(
        index=False
    )
)


# ==============================================================
# 18. INTERPRETATION CHECK
# ==============================================================

print("\n" + "=" * 80)
print("AUTOMATIC INTERPRETATION CHECK")
print("=" * 80)


for _, row in comparison_df.iterrows():

    asset = row["Asset"]

    print("\n" + asset)
    print("-" * 40)

    # RMSE

    if row["Cross_Crypto_RMSE"] < row["Baseline_RMSE"]:

        print(
            "RMSE: IMPROVED after adding "
            "the cross-crypto lag."
        )

    else:

        print(
            "RMSE: DID NOT IMPROVE after adding "
            "the cross-crypto lag."
        )

    # MAE

    if row["Cross_Crypto_MAE"] < row["Baseline_MAE"]:

        print(
            "MAE: IMPROVED after adding "
            "the cross-crypto lag."
        )

    else:

        print(
            "MAE: DID NOT IMPROVE after adding "
            "the cross-crypto lag."
        )

    # OOS R2

    if row["Cross_Crypto_OOS_R2"] > row["Baseline_OOS_R2"]:

        print(
            "OOS R-squared: IMPROVED."
        )

    else:

        print(
            "OOS R-squared: DID NOT IMPROVE."
        )

    # Direction

    if (
        row["Cross_Crypto_Directional_Accuracy"]
        >
        row["Baseline_Directional_Accuracy"]
    ):

        print(
            "Directional accuracy: IMPROVED."
        )

    elif (
        row["Cross_Crypto_Directional_Accuracy"]
        ==
        row["Baseline_Directional_Accuracy"]
    ):

        print(
            "Directional accuracy: UNCHANGED."
        )

    else:

        print(
            "Directional accuracy: DID NOT IMPROVE."
        )


# ==============================================================
# 19. COMBINE INDIVIDUAL FORECASTS
# ==============================================================

all_forecasts = pd.concat(
    [
        btc_baseline_aligned,
        btc_cross_aligned,
        eth_baseline_aligned,
        eth_cross_aligned,
    ],
    ignore_index=True
)


# ==============================================================
# 20. SAVE RESULTS
# ==============================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)


all_forecasts.to_csv(
    FORECAST_OUTPUT,
    index=False
)

performance_df.to_csv(
    SUMMARY_OUTPUT,
    index=False
)

comparison_df.to_csv(
    COMPARISON_OUTPUT,
    index=False
)


print("\nIndividual forecasts saved to:")
print(FORECAST_OUTPUT)

print("\nPerformance summary saved to:")
print(SUMMARY_OUTPUT)

print("\nBaseline vs cross-crypto comparison saved to:")
print(COMPARISON_OUTPUT)


# ==============================================================
# 21. FINAL CHECK
# ==============================================================

print("\n" + "=" * 80)
print("FINAL CHECK")
print("=" * 80)

print("\nTotal individual forecast rows:")
print(len(all_forecasts))

print("\nPerformance table:")
print(performance_df)

print("\nComparison table:")
print(comparison_df)


print("\n" + "=" * 80)
print("CROSS-CRYPTO ROBUSTNESS FORECASTING COMPLETE")
print("=" * 80)