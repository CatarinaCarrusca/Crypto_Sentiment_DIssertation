# ======================================================================
# 02_baseline_forecasting_models.py
#
# BASELINE OUT-OF-SAMPLE FORECASTING MODELS
# Crypto Sentiment Dissertation
#
# PURPOSE
# ----------------------------------------------------------------------
# Estimate genuine out-of-sample baseline forecasts for:
#
#   1. BTC daily returns
#   2. ETH daily returns
#
# These are BASELINE models: NO Reddit variables are included yet.
#
# Later, the exact same forecasting framework can be used for:
#
#   Controls only
#   Controls + Reddit activity
#   Controls + Reddit sentiment
#   Controls + Reddit activity + Reddit sentiment
#
# FORECASTING DESIGN
# ----------------------------------------------------------------------
# - Chronological forecasting
# - Expanding estimation window
# - One-step-ahead forecasts
# - No random train/test split
# - No future observations used in model estimation
# - Predictors must already be information-aligned
#
# PERFORMANCE MEASURES
# ----------------------------------------------------------------------
# - RMSE
# - MAE
# - Out-of-sample R-squared
# - Directional accuracy
#
# ======================================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

warnings.filterwarnings("ignore")


# ======================================================================
# 1. PROJECT PATHS
# ======================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "final_forecast_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "baseline_forecasting"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ======================================================================
# 2. PRINTING HELPER
# ======================================================================

def heading(text):

    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


# ======================================================================
# 3. START
# ======================================================================

heading("BASELINE OUT-OF-SAMPLE FORECASTING MODELS")

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nForecasting dataset not found:\n{INPUT_FILE}"
    )


# ======================================================================
# 4. IMPORT DATA
# ======================================================================

heading("IMPORTING FINAL FORECAST DATASET")

df = pd.read_csv(INPUT_FILE)

print("\nImported shape:")
print(df.shape)

print("\nColumns found:")

for col in df.columns:
    print(f" - {col}")


# ======================================================================
# 5. DATE VALIDATION
# ======================================================================

heading("VALIDATING DATES")

if "Date" not in df.columns:

    raise KeyError(
        "The dataset must contain a column called 'Date'."
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates detected. "
        "Fix these before forecasting."
    )


duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)


if duplicate_dates > 0:

    raise ValueError(
        "Duplicate dates detected. "
        "Fix these before forecasting."
    )


df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\nNumber of observations:")
print(len(df))

print("\nDate range:")
print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# ======================================================================
# 6. CHECK DEPENDENT VARIABLES
# ======================================================================

heading("CHECKING DEPENDENT VARIABLES")

DEPENDENT_VARIABLES = [
    "BTC_Return",
    "ETH_Return"
]


for variable in DEPENDENT_VARIABLES:

    if variable not in df.columns:

        raise KeyError(
            f"Required dependent variable missing: {variable}"
        )

    print(f"\n{variable}: FOUND")

    print("Missing observations:")
    print(df[variable].isna().sum())


# ======================================================================
# 7. DEFINE BASELINE PREDICTORS
# ======================================================================

heading("DEFINING BASELINE PREDICTORS")


# ----------------------------------------------------------------------
# BTC BENCHMARK MODEL
#
# BTC_Return_t =
#
#     BTC_Return_(t-1)
#   + lagged BTC volume
#   + lagged / information-aligned S&P 500 return
#   + lagged / information-aligned VIX change
#   + lagged / information-aligned Gold return
#   + lagged / information-aligned DXY return
#   + lagged / information-aligned US10Y change
#
# ----------------------------------------------------------------------

BTC_PREDICTORS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume",

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"
]


# ----------------------------------------------------------------------
# ETH BENCHMARK MODEL
# ----------------------------------------------------------------------

ETH_PREDICTORS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume",

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"
]


# ======================================================================
# 8. VALIDATE PREDICTOR NAMES
# ======================================================================

def check_predictors(
    data,
    predictors,
    asset
):

    print(f"\n{asset} predictors:")

    missing = []

    for variable in predictors:

        if variable in data.columns:

            print(
                f"  [FOUND]   {variable}"
            )

        else:

            print(
                f"  [MISSING] {variable}"
            )

            missing.append(variable)


    if len(missing) > 0:

        print("\n" + "-" * 70)

        print(
            f"{asset}: required predictor names "
            "do not exactly match the dataset."
        )

        print("\nMissing variables:")

        for variable in missing:
            print(" -", variable)


        print(
            "\nRelevant columns actually present "
            "in final_forecast_dataset.csv:"
        )

        keywords = [

            "btc",
            "eth",
            "volume",
            "sp500",
            "vix",
            "gold",
            "dxy",
            "us10",
            "treasury"
        ]

        for col in data.columns:

            if any(
                keyword in col.lower()
                for keyword in keywords
            ):

                print(" -", col)


        raise KeyError(
            "\nSTOPPING intentionally.\n"
            "Do NOT automatically substitute variables.\n"
            "The forecasting model must use the exact "
            "information-aligned predictors."
        )


check_predictors(
    df,
    BTC_PREDICTORS,
    "BTC"
)

check_predictors(
    df,
    ETH_PREDICTORS,
    "ETH"
)


# ======================================================================
# 9. DISPLAY MODEL SPECIFICATIONS
# ======================================================================

heading("BASELINE MODEL SPECIFICATIONS")


print("\nBTC BASELINE MODEL")

print("\nDependent variable:")
print("BTC_Return")

print("\nPredictors:")

for variable in BTC_PREDICTORS:
    print(" -", variable)


print("\nETH BASELINE MODEL")

print("\nDependent variable:")
print("ETH_Return")

print("\nPredictors:")

for variable in ETH_PREDICTORS:
    print(" -", variable)


# ======================================================================
# 10. IDENTIFY FORECAST PERIOD
# ======================================================================

heading("IDENTIFYING OUT-OF-SAMPLE FORECAST PERIOD")


# ----------------------------------------------------------------------
# First try to use an existing split created by your forecast structure.
# ----------------------------------------------------------------------

possible_split_columns = [

    "Sample",
    "Forecast_Sample",
    "Forecast_Set",
    "Data_Split",
    "Split",
    "Set"
]


split_column = None


for column in possible_split_columns:

    if column in df.columns:

        split_column = column
        break


if split_column is not None:

    print("\nExisting split column found:")
    print(split_column)

    print("\nSplit values:")
    print(
        df[split_column]
        .value_counts(dropna=False)
    )


# ----------------------------------------------------------------------
# Construct OOS indicator
# ----------------------------------------------------------------------

df["Is_Forecast_Period"] = False

existing_split_used = False


if split_column is not None:

    split_values = (
        df[split_column]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    forecast_labels = [

        "test",
        "forecast",
        "out-of-sample",
        "out_of_sample",
        "oos"
    ]

    forecast_indicator = (
        split_values.isin(forecast_labels)
    )


    if forecast_indicator.sum() > 0:

        df["Is_Forecast_Period"] = (
            forecast_indicator
        )

        existing_split_used = True

        print(
            "\nExisting forecasting split will be used."
        )


# ======================================================================
# 11. FALLBACK FORECAST PERIOD
# ======================================================================

# If the forecasting dataset does not contain an explicit usable
# train/test indicator, use 2025 as the holdout period.
#
# Initial estimation period:
#     observations before 2025-01-01
#
# Out-of-sample evaluation:
#     2025-01-01 onward
#
# Because an EXPANDING WINDOW is used, observations become part of
# the estimation sample only after they have occurred.

if not existing_split_used:

    FORECAST_START_DATE = pd.Timestamp(
        "2025-01-01"
    )

    print(
        "\nNo usable existing split detected."
    )

    print(
        "\nUsing forecast start date:"
    )

    print(
        FORECAST_START_DATE.date()
    )

    df["Is_Forecast_Period"] = (
        df["Date"]
        >= FORECAST_START_DATE
    )


# ======================================================================
# 12. VALIDATE FORECAST PERIOD
# ======================================================================

training_count = (
    ~df["Is_Forecast_Period"]
).sum()

forecast_count = (
    df["Is_Forecast_Period"]
).sum()


print("\nInitial estimation observations:")
print(training_count)

print("\nPotential forecast observations:")
print(forecast_count)


if training_count == 0:

    raise ValueError(
        "No initial estimation observations found."
    )


if forecast_count == 0:

    raise ValueError(
        "No out-of-sample forecast observations found."
    )


forecast_dates = df.loc[
    df["Is_Forecast_Period"],
    "Date"
]


print("\nOut-of-sample evaluation period:")

print(
    forecast_dates.min(),
    "to",
    forecast_dates.max()
)


# ======================================================================
# 13. EXPANDING-WINDOW FORECAST FUNCTION
# ======================================================================

def expanding_window_forecast(
    data,
    dependent,
    predictors,
    asset
):

    heading(
        f"{asset} EXPANDING-WINDOW FORECAST"
    )


    required_columns = (

        [
            "Date",
            dependent,
            "Is_Forecast_Period"
        ]

        + predictors
    )


    model_data = (
        data[required_columns]
        .copy()
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    forecast_indices = model_data.index[
        model_data["Is_Forecast_Period"]
    ].tolist()


    print(
        "\nPotential forecast dates:"
    )

    print(
        len(forecast_indices)
    )


    forecast_results = []

    skipped_missing_predictors = 0

    skipped_missing_actual = 0

    skipped_small_training_sample = 0


    # ==============================================================
    # EXPANDING-WINDOW LOOP
    # ==============================================================

    for forecast_index in forecast_indices:


        forecast_row = (
            model_data.loc[
                forecast_index
            ]
        )


        # ----------------------------------------------------------
        # Actual return must exist for evaluation
        # ----------------------------------------------------------

        if pd.isna(
            forecast_row[dependent]
        ):

            skipped_missing_actual += 1

            continue


        # ----------------------------------------------------------
        # Predictor information for forecast date must exist
        # ----------------------------------------------------------

        if (
            forecast_row[predictors]
            .isna()
            .any()
        ):

            skipped_missing_predictors += 1

            continue


        # ----------------------------------------------------------
        # CRITICAL FORECASTING RULE
        #
        # Use ONLY observations strictly before the forecast date.
        #
        # Therefore:
        #
        # forecast t
        #     uses estimation data through t-1
        #
        # No future information enters the regression.
        # ----------------------------------------------------------

        training_data = (
            model_data.loc[
                model_data.index
                < forecast_index
            ]
            .copy()
        )


        training_data = (
            training_data
            .dropna(
                subset=[
                    dependent
                ] + predictors
            )
        )


        # ----------------------------------------------------------
        # Require reasonable initial estimation sample
        # ----------------------------------------------------------

        if (
            len(training_data)
            <= len(predictors) + 20
        ):

            skipped_small_training_sample += 1

            continue


        # ==========================================================
        # BASELINE REGRESSION
        # ==========================================================

        X_train = (
            training_data[
                predictors
            ]
            .astype(float)
        )


        y_train = (
            training_data[
                dependent
            ]
            .astype(float)
        )


        X_train = sm.add_constant(
            X_train,
            has_constant="add"
        )


        model = sm.OLS(
            y_train,
            X_train
        ).fit()


        # ==========================================================
        # ONE-STEP-AHEAD PREDICTOR VECTOR
        # ==========================================================

        X_forecast = pd.DataFrame(

            [
                forecast_row[
                    predictors
                ].astype(float).values
            ],

            columns=predictors
        )


        X_forecast = sm.add_constant(
            X_forecast,
            has_constant="add"
        )


        # Ensure identical column order

        X_forecast = (
            X_forecast[
                X_train.columns
            ]
        )


        # ==========================================================
        # BASELINE FORECAST
        # ==========================================================

        baseline_forecast = float(

            model.predict(
                X_forecast
            ).iloc[0]
        )


        # ==========================================================
        # HISTORICAL-MEAN BENCHMARK
        #
        # Also constructed recursively.
        #
        # Only returns observed before t are used.
        # ==========================================================

        historical_mean_forecast = float(
            y_train.mean()
        )


        actual_return = float(
            forecast_row[dependent]
        )


        # ==========================================================
        # STORE RESULT
        # ==========================================================

        forecast_results.append({

            "Date":
                forecast_row["Date"],

            "Actual_Return":
                actual_return,

            "Baseline_Forecast":
                baseline_forecast,

            "Historical_Mean_Forecast":
                historical_mean_forecast,

            "Baseline_Error":
                actual_return
                - baseline_forecast,

            "Historical_Mean_Error":
                actual_return
                - historical_mean_forecast,

            "Training_Observations":
                len(training_data)
        })


    # ==============================================================
    # CONVERT RESULTS
    # ==============================================================

    results = pd.DataFrame(
        forecast_results
    )


    print("\nSuccessful forecasts:")
    print(len(results))


    print(
        "\nSkipped because predictors were missing:"
    )

    print(
        skipped_missing_predictors
    )


    print(
        "\nSkipped because actual return was missing:"
    )

    print(
        skipped_missing_actual
    )


    print(
        "\nSkipped because training sample was too small:"
    )

    print(
        skipped_small_training_sample
    )


    if results.empty:

        raise ValueError(
            f"No valid {asset} forecasts were generated."
        )


    print("\nSuccessful forecast period:")

    print(
        results["Date"].min(),
        "to",
        results["Date"].max()
    )


    return results


# ======================================================================
# 14. BTC BASELINE FORECASTS
# ======================================================================

btc_forecasts = expanding_window_forecast(

    data=df,

    dependent="BTC_Return",

    predictors=BTC_PREDICTORS,

    asset="BTC"
)


# ======================================================================
# 15. ETH BASELINE FORECASTS
# ======================================================================

eth_forecasts = expanding_window_forecast(

    data=df,

    dependent="ETH_Return",

    predictors=ETH_PREDICTORS,

    asset="ETH"
)


# ======================================================================
# 16. FORECAST EVALUATION FUNCTION
# ======================================================================

def evaluate_forecasts(
    forecasts,
    asset
):


    actual = (
        forecasts[
            "Actual_Return"
        ].astype(float)
    )


    baseline_forecast = (
        forecasts[
            "Baseline_Forecast"
        ].astype(float)
    )


    historical_forecast = (
        forecasts[
            "Historical_Mean_Forecast"
        ].astype(float)
    )


    # ==============================================================
    # FORECAST ERRORS
    # ==============================================================

    baseline_errors = (
        actual
        - baseline_forecast
    )


    historical_errors = (
        actual
        - historical_forecast
    )


    # ==============================================================
    # RMSE
    # ==============================================================

    baseline_rmse = np.sqrt(
        np.mean(
            baseline_errors ** 2
        )
    )


    historical_rmse = np.sqrt(
        np.mean(
            historical_errors ** 2
        )
    )


    # ==============================================================
    # MAE
    # ==============================================================

    baseline_mae = np.mean(
        np.abs(
            baseline_errors
        )
    )


    historical_mae = np.mean(
        np.abs(
            historical_errors
        )
    )


    # ==============================================================
    # OUT-OF-SAMPLE R-SQUARED
    #
    # R2_OS =
    #
    # 1 -
    #
    # sum((actual - model forecast)^2)
    # --------------------------------
    # sum((actual - historical mean forecast)^2)
    #
    #
    # Positive:
    #     model beats historical mean.
    #
    # Negative:
    #     historical mean beats model.
    # ==============================================================

    model_sse = np.sum(
        baseline_errors ** 2
    )


    benchmark_sse = np.sum(
        historical_errors ** 2
    )


    if benchmark_sse == 0:

        oos_r_squared = np.nan

    else:

        oos_r_squared = (
            1
            - (
                model_sse
                / benchmark_sse
            )
        )


    # ==============================================================
    # DIRECTIONAL ACCURACY
    # ==============================================================

    actual_direction = np.sign(
        actual
    )


    forecast_direction = np.sign(
        baseline_forecast
    )


    directional_accuracy = np.mean(
        actual_direction
        == forecast_direction
    )


    # ==============================================================
    # RMSE IMPROVEMENT
    # ==============================================================

    rmse_improvement = (

        historical_rmse
        - baseline_rmse

    )


    rmse_improvement_percent = (

        (
            historical_rmse
            - baseline_rmse
        )

        / historical_rmse

        * 100

        if historical_rmse != 0

        else np.nan
    )


    # ==============================================================
    # RESULT
    # ==============================================================

    metrics = {

        "Asset":
            asset,

        "N_Forecasts":
            len(forecasts),

        "Forecast_Start":
            forecasts[
                "Date"
            ].min(),

        "Forecast_End":
            forecasts[
                "Date"
            ].max(),

        "Baseline_RMSE":
            baseline_rmse,

        "Historical_Mean_RMSE":
            historical_rmse,

        "RMSE_Improvement":
            rmse_improvement,

        "RMSE_Improvement_Percent":
            rmse_improvement_percent,

        "Baseline_MAE":
            baseline_mae,

        "Historical_Mean_MAE":
            historical_mae,

        "OOS_R2":
            oos_r_squared,

        "Directional_Accuracy":
            directional_accuracy
    }


    return metrics


# ======================================================================
# 17. EVALUATE FORECASTS
# ======================================================================

heading("EVALUATING OUT-OF-SAMPLE FORECAST PERFORMANCE")


btc_metrics = evaluate_forecasts(

    btc_forecasts,

    "BTC"
)


eth_metrics = evaluate_forecasts(

    eth_forecasts,

    "ETH"
)


metrics_df = pd.DataFrame(

    [
        btc_metrics,
        eth_metrics
    ]
)


print("\nForecast performance:")

print(
    metrics_df.to_string(
        index=False
    )
)


# ======================================================================
# 18. INTERPRET OOS R-SQUARED
# ======================================================================

heading("OUT-OF-SAMPLE R-SQUARED INTERPRETATION")


for _, row in metrics_df.iterrows():


    asset = row["Asset"]

    r2 = row["OOS_R2"]


    print(
        f"\n{asset} OOS R-squared:"
    )

    print(
        f"{r2:.6f}"
    )


    if pd.isna(r2):

        print(
            "Could not calculate OOS R-squared."
        )


    elif r2 > 0:

        print(
            "The baseline control model outperforms "
            "the recursive historical-mean benchmark "
            "in squared-error terms."
        )


    elif r2 < 0:

        print(
            "The baseline control model underperforms "
            "the recursive historical-mean benchmark "
            "in squared-error terms."
        )


    else:

        print(
            "The baseline control model and historical "
            "mean benchmark have equal squared-error performance."
        )


# ======================================================================
# 19. SAVE FORECASTS
# ======================================================================

heading("SAVING BASELINE FORECASTING RESULTS")


BTC_FORECAST_FILE = (

    OUTPUT_DIR
    / "btc_baseline_forecasts.csv"
)


ETH_FORECAST_FILE = (

    OUTPUT_DIR
    / "eth_baseline_forecasts.csv"
)


METRICS_FILE = (

    OUTPUT_DIR
    / "baseline_forecast_metrics.csv"
)


btc_forecasts.to_csv(

    BTC_FORECAST_FILE,

    index=False
)


eth_forecasts.to_csv(

    ETH_FORECAST_FILE,

    index=False
)


metrics_df.to_csv(

    METRICS_FILE,

    index=False
)


print("\nBTC forecast file:")
print(BTC_FORECAST_FILE)


print("\nETH forecast file:")
print(ETH_FORECAST_FILE)


print("\nForecast metrics file:")
print(METRICS_FILE)


# ======================================================================
# 20. FINAL DATA CHECKS
# ======================================================================

heading("FINAL FORECAST VALIDATION")


print("\nBTC forecast observations:")
print(len(btc_forecasts))


print("\nETH forecast observations:")
print(len(eth_forecasts))


print("\nBTC missing values:")

print(

    btc_forecasts[
        [
            "Actual_Return",
            "Baseline_Forecast",
            "Historical_Mean_Forecast"
        ]
    ]
    .isna()
    .sum()
)


print("\nETH missing values:")

print(

    eth_forecasts[
        [
            "Actual_Return",
            "Baseline_Forecast",
            "Historical_Mean_Forecast"
        ]
    ]
    .isna()
    .sum()
)


# ======================================================================
# 21. DISPLAY SAMPLE FORECASTS
# ======================================================================

heading("BTC FIRST FIVE FORECASTS")

print(

    btc_forecasts
    .head()
    .to_string(
        index=False
    )
)


heading("BTC LAST FIVE FORECASTS")

print(

    btc_forecasts
    .tail()
    .to_string(
        index=False
    )
)


heading("ETH FIRST FIVE FORECASTS")

print(

    eth_forecasts
    .head()
    .to_string(
        index=False
    )
)


heading("ETH LAST FIVE FORECASTS")

print(

    eth_forecasts
    .tail()
    .to_string(
        index=False
    )
)


# ======================================================================
# 22. METHODOLOGICAL SUMMARY
# ======================================================================

heading("METHODOLOGICAL SUMMARY")


print(
    """
BASELINE FORECASTING DESIGN

The forecasting exercise uses an expanding estimation window.

For each forecast date t:

    1. Only observations occurring before t are used for estimation.

    2. The baseline OLS model is re-estimated.

    3. Information-aligned predictor values are used to produce
       a one-step-ahead return forecast.

    4. A recursive historical-mean forecast is also produced using
       only information available before t.

    5. The forecast is compared with the subsequently realised
       cryptocurrency return.


This separates OUT-OF-SAMPLE PREDICTIVE PERFORMANCE from the
IN-SAMPLE EXPLANATORY regressions estimated elsewhere in the project.


IMPORTANT:

No Reddit sentiment or Reddit activity variables are included in these
models.

These results establish the benchmark against which the later Reddit
forecasting models will be compared.
"""
)


# ======================================================================
# 23. NEXT REDDIT SPECIFICATIONS
# ======================================================================

heading("LATER REDDIT FORECASTING SPECIFICATIONS")


print(
    """
When the cleaned Reddit data become available, retain the SAME
forecasting framework and compare:

MODEL 1
Controls only

MODEL 2
Controls + lagged Reddit activity

MODEL 3
Controls + lagged Reddit sentiment

MODEL 4
Controls + lagged Reddit activity + lagged Reddit sentiment


Reddit activity should be transformed as:

    Log_Reddit_Activity = log(1 + post count)

and appropriately lagged/information-aligned before forecasting.


The central H3/H4 forecasting comparison will then examine whether
adding lagged Reddit sentiment improves out-of-sample forecasting
performance relative to the otherwise identical baseline model.


Primary metrics:

    RMSE
    MAE
    Out-of-sample R-squared

In-sample R-squared and coefficient significance must NOT be treated
as evidence of forecasting performance.
"""
)


heading("BASELINE FORECASTING MODELS COMPLETE")