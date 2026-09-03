# =====================================================================
# 09_forecast_evaluation_infrastructure.py
#
# FORECAST EVALUATION INFRASTRUCTURE
#
# Dissertation:
# Do Social Media Sentiment Signals Improve the Prediction of
# Cryptocurrency Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum
#
# =====================================================================
#
# PURPOSE
# ---------------------------------------------------------------------
# Prepare a reusable out-of-sample forecast-evaluation framework for
# H3 and H4.
#
# BEFORE REDDIT ARRIVES:
#
#   - Generate corrected benchmark forecasts directly from
#     final_forecast_dataset.csv.
#   - Use information-aligned predictors.
#   - Use genuine expanding-window one-day-ahead forecasts.
#   - Calculate benchmark RMSE, MAE, OOS R2 and directional accuracy.
#   - Save the benchmark forecasts.
#   - Save ready-to-use sentiment forecast templates.
#   - Validate the OOS infrastructure.
#
# AFTER REDDIT ARRIVES:
#
#   - Add the sentiment-augmented forecast file.
#   - The script automatically switches into formal comparison mode.
#   - Benchmark and augmented models are compared on EXACTLY the same
#     OOS dates.
#   - RMSE, MAE, OOS R2, directional accuracy and loss differences
#     are calculated.
#   - Diebold-Mariano-type HAC forecast-accuracy tests are performed.
#
# ---------------------------------------------------------------------
# IMPORTANT
# ---------------------------------------------------------------------
#
# This script DOES NOT create fake sentiment forecasts.
#
# If Reddit/sentiment forecasts do not yet exist, the script finishes
# successfully with:
#
#       INFRASTRUCTURE STATUS: READY - WAITING FOR REDDIT
#
# rather than raising FileNotFoundError.
#
# =====================================================================


from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats


warnings.filterwarnings("ignore")


# =====================================================================
# 1. PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

DATA_PROCESSED = (
    PROJECT_ROOT
    / "data_processed"
)

ANALYSIS_OUTPUTS = (
    PROJECT_ROOT
    / "analysis_outputs"
)

OUTPUT_DIR = (
    ANALYSIS_OUTPUTS
    / "forecast_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_DATASET_FILE = (
    DATA_PROCESSED
    / "final_forecast_dataset.csv"
)


# =====================================================================
# 2. SETTINGS
# =====================================================================

OOS_START = pd.Timestamp(
    "2024-01-02"
)

OOS_END = pd.Timestamp(
    "2025-12-31"
)

DM_HAC_MAXLAGS = 7

SIGNIFICANCE_LEVEL = 0.05

ACTUAL_MATCH_TOLERANCE = 1e-12


# =====================================================================
# 3. ASSETS
# =====================================================================
#
# Run BOTH BTC and ETH automatically.
# =====================================================================

ASSETS = [
    "BTC",
    "ETH"
]


# =====================================================================
# 4. CORRECT INFORMATION-ALIGNED MODEL SPECIFICATIONS
# =====================================================================

MODEL_SPECS = {

    "BTC": {

        "dependent":
            "BTC_Return",

        "predictors": [

            "BTC_Lagged_Return",

            "Lagged_Log_BTC_Volume",

            "Lagged_SP500_Return_Aligned",

            "Lagged_VIX_Change_Aligned",

            "Lagged_Gold_Return_Aligned",

            "Lagged_DXY_Return_Aligned",

            "Lagged_US10Y_Change_Aligned"

        ]

    },

    "ETH": {

        "dependent":
            "ETH_Return",

        "predictors": [

            "ETH_Lagged_Return",

            "Lagged_Log_ETH_Volume",

            "Lagged_SP500_Return_Aligned",

            "Lagged_VIX_Change_Aligned",

            "Lagged_Gold_Return_Aligned",

            "Lagged_DXY_Return_Aligned",

            "Lagged_US10Y_Change_Aligned"

        ]

    }

}


# =====================================================================
# 5. OLD NON-ALIGNED VARIABLES - MUST NEVER BE USED
# =====================================================================

OLD_NONALIGNED_CONTROLS = [

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"

]


# =====================================================================
# 6. SENTIMENT FORECAST FILES
# =====================================================================
#
# These files do NOT need to exist yet.
#
# Once Reddit forecasting models are available, save:
#
# BTC:
#
# analysis_outputs/forecast_evaluation/
# btc_sentiment_augmented_forecasts.csv
#
# ETH:
#
# analysis_outputs/forecast_evaluation/
# eth_sentiment_augmented_forecasts.csv
#
# Required columns:
#
#       Date
#       Actual
#       Forecast
#
# =====================================================================

SENTIMENT_FORECAST_FILES = {

    "BTC":
        OUTPUT_DIR
        / "btc_sentiment_augmented_forecasts.csv",

    "ETH":
        OUTPUT_DIR
        / "eth_sentiment_augmented_forecasts.csv"

}


# =====================================================================
# 7. HELPER FUNCTIONS
# =====================================================================


def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):

    try:

        return float(value)

    except Exception:

        return np.nan


def significance_label(p_value):

    if pd.isna(p_value):

        return "NA"

    if p_value < 0.01:

        return "***"

    if p_value < 0.05:

        return "**"

    if p_value < 0.10:

        return "*"

    return ""


# =====================================================================
# 8. LOAD FINAL DATASET
# =====================================================================

section(
    "FORECAST EVALUATION INFRASTRUCTURE"
)

print(
    "\nFinal dataset:"
)

print(
    FINAL_DATASET_FILE
)

print(
    "\nOOS period:"
)

print(
    OOS_START,
    "to",
    OOS_END
)

print(
    "\nForecasting method:"
)

print(
    "Expanding-window one-day-ahead OLS"
)

print(
    "\nDM-type HAC maximum lag:"
)

print(
    DM_HAC_MAXLAGS
)


if not FINAL_DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nRequired dataset not found:\n"
        f"{FINAL_DATASET_FILE}"
    )


df = pd.read_csv(
    FINAL_DATASET_FILE
)


print(
    "\nDataset shape:"
)

print(
    df.shape
)


# =====================================================================
# 9. DATE VALIDATION
# =====================================================================

section(
    "DATE VALIDATION"
)


if "Date" not in df.columns:

    raise KeyError(
        "Date column is missing."
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


invalid_dates = int(
    df["Date"]
    .isna()
    .sum()
)


duplicate_dates = int(
    df["Date"]
    .duplicated()
    .sum()
)


print(
    "\nInvalid dates:"
)

print(
    invalid_dates
)


print(
    "\nDuplicate dates:"
)

print(
    duplicate_dates
)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates found."
    )


if duplicate_dates > 0:

    raise ValueError(
        "Duplicate dates found."
    )


df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print(
    "\nDate range:"
)

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =====================================================================
# 10. CALENDAR VALIDATION
# =====================================================================

section(
    "DAILY CALENDAR VALIDATION"
)


date_differences = (
    df["Date"]
    .diff()
)


calendar_gaps = int(
    (
        date_differences.notna()
        &
        (
            date_differences
            !=
            pd.Timedelta(days=1)
        )
    )
    .sum()
)


weekend_observations = int(
    (
        df["Date"]
        .dt.dayofweek
        >= 5
    )
    .sum()
)


print(
    "\nObservations:"
)

print(
    len(df)
)


print(
    "\nCalendar gaps:"
)

print(
    calendar_gaps
)


print(
    "\nWeekend observations:"
)

print(
    weekend_observations
)


if calendar_gaps > 0:

    raise ValueError(
        "Final dataset is not a continuous daily calendar."
    )


print(
    "\nPASS: Continuous cryptocurrency calendar confirmed."
)


# =====================================================================
# 11. MODEL VARIABLE VALIDATION
# =====================================================================

section(
    "MODEL SPECIFICATION VALIDATION"
)


required_variables = [
    "Date"
]


for asset in ASSETS:

    required_variables.append(
        MODEL_SPECS[
            asset
        ][
            "dependent"
        ]
    )

    required_variables.extend(
        MODEL_SPECS[
            asset
        ][
            "predictors"
        ]
    )


required_variables = list(
    dict.fromkeys(
        required_variables
    )
)


missing_variables = [

    variable

    for variable in required_variables

    if variable not in df.columns

]


if missing_variables:

    raise KeyError(
        "\nMissing required variables:\n"
        +
        "\n".join(
            missing_variables
        )
    )


print(
    "\nPASS: All required variables are available."
)


# =====================================================================
# 12. INFORMATION-ALIGNMENT SAFETY CHECK
# =====================================================================

section(
    "INFORMATION-ALIGNMENT SAFETY CHECK"
)


all_used_predictors = []


for asset in ASSETS:

    all_used_predictors.extend(
        MODEL_SPECS[
            asset
        ][
            "predictors"
        ]
    )


old_controls_used = [

    variable

    for variable in OLD_NONALIGNED_CONTROLS

    if variable in all_used_predictors

]


print(
    "\nOld non-aligned controls used:"
)

print(
    old_controls_used
)


if old_controls_used:

    raise ValueError(
        "Old non-aligned traditional-market controls "
        "must not be used."
    )


aligned_controls = [

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


for variable in aligned_controls:

    print(
        " -",
        variable
    )


print(
    "\nPASS: Forecast models use only information-aligned "
    "traditional-market controls."
)


# =====================================================================
# 13. NUMERIC CONVERSION
# =====================================================================

section(
    "NUMERIC CONVERSION"
)


numeric_variables = [

    variable

    for variable in required_variables

    if variable != "Date"

]


for variable in numeric_variables:

    df[
        variable
    ] = (
        pd.to_numeric(
            df[
                variable
            ],
            errors="coerce"
        )
        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )
    )


print(
    "\nPASS: Required model variables converted to numeric."
)


# =====================================================================
# 14. EXPANDING-WINDOW BENCHMARK FORECAST FUNCTION
# =====================================================================


def generate_expanding_forecasts(
    dataframe,
    asset,
    dependent,
    predictors
):

    section(
        f"{asset}: GENERATING EXPANDING-WINDOW BENCHMARK FORECASTS"
    )


    model_variables = (
        [
            "Date",
            dependent
        ]
        +
        predictors
    )


    model_data = (
        dataframe[
            model_variables
        ]
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    # ---------------------------------------------------------
    # Potential OOS target dates
    # ---------------------------------------------------------

    oos_dates = (
        model_data.loc[
            (
                model_data["Date"]
                >=
                OOS_START
            )
            &
            (
                model_data["Date"]
                <=
                OOS_END
            ),
            "Date"
        ]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )


    print(
        "\nPotential OOS dates:"
    )

    print(
        len(
            oos_dates
        )
    )


    forecast_rows = []


    for forecast_number, forecast_date in enumerate(
        oos_dates,
        start=1
    ):

        # -----------------------------------------------------
        # Forecast row
        # -----------------------------------------------------

        forecast_row = (
            model_data.loc[
                model_data["Date"]
                ==
                forecast_date
            ]
            .copy()
        )


        if forecast_row.empty:

            continue


        # -----------------------------------------------------
        # Require actual return and all predictors on forecast
        # date.
        # -----------------------------------------------------

        required_forecast_values = (
            [
                dependent
            ]
            +
            predictors
        )


        if (
            forecast_row[
                required_forecast_values
            ]
            .isna()
            .any()
            .any()
        ):

            continue


        # -----------------------------------------------------
        # Expanding training sample:
        #
        # STRICTLY BEFORE forecast date.
        # -----------------------------------------------------

        training_sample = (
            model_data.loc[
                model_data["Date"]
                <
                forecast_date
            ]
            .dropna(
                subset=[
                    dependent
                ]
                +
                predictors
            )
            .copy()
        )


        if training_sample.empty:

            continue


        if len(
            training_sample
        ) <= (
            len(
                predictors
            )
            +
            5
        ):

            continue


        # -----------------------------------------------------
        # Estimate benchmark OLS
        # -----------------------------------------------------

        y_train = (
            training_sample[
                dependent
            ]
            .astype(float)
        )


        X_train = (
            training_sample[
                predictors
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


        # -----------------------------------------------------
        # Construct forecast predictor row
        # -----------------------------------------------------

        X_forecast = (
            forecast_row[
                predictors
            ]
            .astype(float)
            .copy()
        )


        X_forecast = sm.add_constant(
            X_forecast,
            has_constant="add"
        )


        # Ensure exact same column order as training matrix.

        X_forecast = X_forecast[
            X_train.columns
        ]


        forecast_value = safe_float(
            model.predict(
                X_forecast
            )
            .iloc[0]
        )


        actual_value = safe_float(
            forecast_row[
                dependent
            ]
            .iloc[0]
        )


        # -----------------------------------------------------
        # Historical-mean forecast
        #
        # IMPORTANT:
        #
        # Use the SAME training sample as the model.
        #
        # Therefore the benchmark denominator is constructed
        # from observations genuinely available before t and
        # from the same usable estimation history.
        # -----------------------------------------------------

        historical_mean_forecast = safe_float(
            y_train.mean()
        )


        training_start = (
            training_sample[
                "Date"
            ]
            .min()
        )


        training_end = (
            training_sample[
                "Date"
            ]
            .max()
        )


        forecast_rows.append(
            {

                "Asset":
                    asset,

                "Forecast_Number":
                    int(
                        forecast_number
                    ),

                "Date":
                    forecast_date,

                "Training_Start":
                    training_start,

                "Training_End":
                    training_end,

                "Training_N":
                    int(
                        len(
                            training_sample
                        )
                    ),

                "Actual":
                    actual_value,

                "Forecast":
                    forecast_value,

                "Historical_Mean_Forecast":
                    historical_mean_forecast,

                "Weekend":
                    bool(
                        forecast_date.dayofweek
                        >= 5
                    )

            }
        )


    forecasts = pd.DataFrame(
        forecast_rows
    )


    if forecasts.empty:

        raise ValueError(
            f"{asset}: no benchmark forecasts generated."
        )


    forecasts = (
        forecasts
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\nGenerated benchmark forecasts:"
    )

    print(
        len(
            forecasts
        )
    )


    print(
        "\nForecast period:"
    )

    print(
        forecasts[
            "Date"
        ].min(),
        "to",
        forecasts[
            "Date"
        ].max()
    )


    print(
        "\nWeekend forecasts:"
    )

    print(
        int(
            forecasts[
                "Weekend"
            ]
            .sum()
        )
    )


    print(
        "\nFirst training sample:"
    )

    print(
        forecasts[
            "Training_Start"
        ].iloc[0],
        "to",
        forecasts[
            "Training_End"
        ].iloc[0]
    )


    print(
        "\nFirst training N:"
    )

    print(
        int(
            forecasts[
                "Training_N"
            ].iloc[0]
        )
    )


    print(
        "\nFinal training N:"
    )

    print(
        int(
            forecasts[
                "Training_N"
            ].iloc[-1]
        )
    )


    return forecasts


# =====================================================================
# 15. FORECAST METRICS
# =====================================================================


def calculate_metrics(
    actual,
    forecast,
    historical_mean,
    model_name
):

    actual = np.asarray(
        actual,
        dtype=float
    )


    forecast = np.asarray(
        forecast,
        dtype=float
    )


    historical_mean = np.asarray(
        historical_mean,
        dtype=float
    )


    error = (
        actual
        -
        forecast
    )


    squared_error = (
        error
        ** 2
    )


    absolute_error = np.abs(
        error
    )


    rmse = math.sqrt(
        np.mean(
            squared_error
        )
    )


    mae = np.mean(
        absolute_error
    )


    historical_error = (
        actual
        -
        historical_mean
    )


    historical_squared_error = (
        historical_error
        ** 2
    )


    model_sse = safe_float(
        np.sum(
            squared_error
        )
    )


    historical_sse = safe_float(
        np.sum(
            historical_squared_error
        )
    )


    if historical_sse > 0:

        oos_r2 = (
            1.0
            -
            (
                model_sse
                /
                historical_sse
            )
        )

    else:

        oos_r2 = np.nan


    actual_direction = np.sign(
        actual
    )


    forecast_direction = np.sign(
        forecast
    )


    direction_correct = (
        actual_direction
        ==
        forecast_direction
    )


    directional_accuracy = safe_float(
        np.mean(
            direction_correct
        )
    )


    return {

        "Model":
            model_name,

        "N":
            int(
                len(
                    actual
                )
            ),

        "RMSE":
            safe_float(
                rmse
            ),

        "MAE":
            safe_float(
                mae
            ),

        "OOS_R_Squared":
            safe_float(
                oos_r2
            ),

        "Directional_Accuracy":
            directional_accuracy,

        "Directional_Accuracy_Percent":
            safe_float(
                100.0
                *
                directional_accuracy
            ),

        "SSE":
            model_sse,

        "Historical_Mean_SSE":
            historical_sse

    }


# =====================================================================
# 16. NEWEY-WEST LONG-RUN VARIANCE
# =====================================================================


def newey_west_long_run_variance(
    values,
    maxlags=7
):

    values = np.asarray(
        values,
        dtype=float
    )


    values = values[
        np.isfinite(
            values
        )
    ]


    n = len(
        values
    )


    if n < 2:

        return np.nan


    centered = (
        values
        -
        np.mean(
            values
        )
    )


    gamma_zero = (
        np.dot(
            centered,
            centered
        )
        /
        n
    )


    long_run_variance = (
        gamma_zero
    )


    effective_lags = min(
        int(
            maxlags
        ),
        n - 1
    )


    for lag in range(
        1,
        effective_lags + 1
    ):

        covariance = (
            np.dot(
                centered[
                    lag:
                ],
                centered[
                    :-lag
                ]
            )
            /
            n
        )


        weight = (
            1.0
            -
            (
                lag
                /
                (
                    effective_lags
                    +
                    1.0
                )
            )
        )


        long_run_variance += (
            2.0
            *
            weight
            *
            covariance
        )


    return safe_float(
        long_run_variance
    )


# =====================================================================
# 17. DIEBOLD-MARIANO-TYPE TEST
# =====================================================================


def diebold_mariano_type_test(
    loss_a,
    loss_b,
    maxlags=7
):

    """
    Loss differential:

        d_t = loss_A - loss_B

    Positive mean differential:
        Model B has lower average loss.

    Negative mean differential:
        Model A has lower average loss.
    """


    loss_a = np.asarray(
        loss_a,
        dtype=float
    )


    loss_b = np.asarray(
        loss_b,
        dtype=float
    )


    valid = (
        np.isfinite(
            loss_a
        )
        &
        np.isfinite(
            loss_b
        )
    )


    loss_a = loss_a[
        valid
    ]


    loss_b = loss_b[
        valid
    ]


    differential = (
        loss_a
        -
        loss_b
    )


    n = len(
        differential
    )


    if n < 2:

        return {

            "N":
                n,

            "Mean_Loss_Differential_A_minus_B":
                np.nan,

            "DM_Statistic":
                np.nan,

            "P_Value_Two_Sided":
                np.nan,

            "P_Value_One_Sided_B_Better":
                np.nan,

            "P_Value_One_Sided_A_Better":
                np.nan,

            "HAC_Long_Run_Variance":
                np.nan,

            "SE_Mean_Loss_Differential":
                np.nan,

            "HAC_Maxlags":
                maxlags

        }


    mean_differential = safe_float(
        np.mean(
            differential
        )
    )


    long_run_variance = (
        newey_west_long_run_variance(
            differential,
            maxlags=maxlags
        )
    )


    if (
        pd.isna(
            long_run_variance
        )
        or
        long_run_variance <= 0
    ):

        return {

            "N":
                n,

            "Mean_Loss_Differential_A_minus_B":
                mean_differential,

            "DM_Statistic":
                np.nan,

            "P_Value_Two_Sided":
                np.nan,

            "P_Value_One_Sided_B_Better":
                np.nan,

            "P_Value_One_Sided_A_Better":
                np.nan,

            "HAC_Long_Run_Variance":
                long_run_variance,

            "SE_Mean_Loss_Differential":
                np.nan,

            "HAC_Maxlags":
                maxlags

        }


    standard_error = math.sqrt(
        long_run_variance
        /
        n
    )


    dm_statistic = (
        mean_differential
        /
        standard_error
    )


    p_two_sided = (
        2.0
        *
        stats.norm.sf(
            abs(
                dm_statistic
            )
        )
    )


    # Alternative:
    #
    # E(loss_A - loss_B) > 0
    #
    # i.e. Model B is better.

    p_b_better = stats.norm.sf(
        dm_statistic
    )


    # Alternative:
    #
    # E(loss_A - loss_B) < 0
    #
    # i.e. Model A is better.

    p_a_better = stats.norm.cdf(
        dm_statistic
    )


    return {

        "N":
            int(
                n
            ),

        "Mean_Loss_Differential_A_minus_B":
            mean_differential,

        "DM_Statistic":
            safe_float(
                dm_statistic
            ),

        "P_Value_Two_Sided":
            safe_float(
                p_two_sided
            ),

        "P_Value_One_Sided_B_Better":
            safe_float(
                p_b_better
            ),

        "P_Value_One_Sided_A_Better":
            safe_float(
                p_a_better
            ),

        "HAC_Long_Run_Variance":
            safe_float(
                long_run_variance
            ),

        "SE_Mean_Loss_Differential":
            safe_float(
                standard_error
            ),

        "HAC_Maxlags":
            int(
                maxlags
            )

    }


# =====================================================================
# 18. GENERATE CORRECTED BENCHMARK FORECASTS
# =====================================================================

all_benchmark_metrics = []

benchmark_forecasts_by_asset = {}


for asset in ASSETS:

    spec = MODEL_SPECS[
        asset
    ]


    forecasts = generate_expanding_forecasts(

        dataframe=
            df,

        asset=
            asset,

        dependent=
            spec[
                "dependent"
            ],

        predictors=
            spec[
                "predictors"
            ]

    )


    benchmark_forecasts_by_asset[
        asset
    ] = forecasts


    metrics = calculate_metrics(

        actual=
            forecasts[
                "Actual"
            ],

        forecast=
            forecasts[
                "Forecast"
            ],

        historical_mean=
            forecasts[
                "Historical_Mean_Forecast"
            ],

        model_name=
            "Traditional_Market_Benchmark"

    )


    metrics[
        "Asset"
    ] = asset


    metrics[
        "OOS_Start"
    ] = forecasts[
        "Date"
    ].min()


    metrics[
        "OOS_End"
    ] = forecasts[
        "Date"
    ].max()


    all_benchmark_metrics.append(
        metrics
    )


# =====================================================================
# 19. BENCHMARK RESULTS
# =====================================================================

section(
    "CORRECTED BENCHMARK FORECAST PERFORMANCE"
)


benchmark_metrics_df = pd.DataFrame(
    all_benchmark_metrics
)


benchmark_metrics_df = benchmark_metrics_df[
    [

        "Asset",

        "Model",

        "N",

        "OOS_Start",

        "OOS_End",

        "RMSE",

        "MAE",

        "OOS_R_Squared",

        "Directional_Accuracy",

        "Directional_Accuracy_Percent",

        "SSE",

        "Historical_Mean_SSE"

    ]
]


print(
    "\n",
    benchmark_metrics_df.to_string(
        index=False
    )
)


# =====================================================================
# 20. SAVE BENCHMARK FORECASTS
# =====================================================================

section(
    "SAVING CORRECTED BENCHMARK FORECASTS"
)


for asset in ASSETS:

    output_file = (
        OUTPUT_DIR
        /
        f"{asset.lower()}_benchmark_forecasts.csv"
    )


    benchmark_forecasts_by_asset[
        asset
    ].to_csv(
        output_file,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        output_file
    )


benchmark_metrics_file = (
    OUTPUT_DIR
    /
    "benchmark_forecast_metrics.csv"
)


benchmark_metrics_df.to_csv(
    benchmark_metrics_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    benchmark_metrics_file
)


# =====================================================================
# 21. BENCHMARK OOS VALIDATION
# =====================================================================

section(
    "BENCHMARK OOS VALIDATION"
)


benchmark_validation_rows = []


for asset in ASSETS:

    forecasts = (
        benchmark_forecasts_by_asset[
            asset
        ]
    )


    unique_dates = bool(
        forecasts[
            "Date"
        ]
        .is_unique
    )


    sorted_dates = bool(
        forecasts[
            "Date"
        ]
        .is_monotonic_increasing
    )


    complete_values = bool(
        forecasts[
            [
                "Actual",
                "Forecast",
                "Historical_Mean_Forecast"
            ]
        ]
        .notna()
        .all()
        .all()
    )


    training_strictly_prior = bool(
        (
            forecasts[
                "Training_End"
            ]
            <
            forecasts[
                "Date"
            ]
        )
        .all()
    )


    start_pass = bool(
        forecasts[
            "Date"
        ].min()
        >=
        OOS_START
    )


    end_pass = bool(
        forecasts[
            "Date"
        ].max()
        <=
        OOS_END
    )


    weekend_count = int(
        forecasts[
            "Weekend"
        ]
        .sum()
    )


    benchmark_validation_rows.append(
        {

            "Asset":
                asset,

            "N":
                int(
                    len(
                        forecasts
                    )
                ),

            "OOS_Start":
                forecasts[
                    "Date"
                ].min(),

            "OOS_End":
                forecasts[
                    "Date"
                ].max(),

            "Weekend_Observations":
                weekend_count,

            "Dates_Unique":
                unique_dates,

            "Dates_Sorted":
                sorted_dates,

            "Forecasts_Complete":
                complete_values,

            "Training_Strictly_Prior":
                training_strictly_prior,

            "OOS_Start_Valid":
                start_pass,

            "OOS_End_Valid":
                end_pass

        }
    )


benchmark_validation_df = pd.DataFrame(
    benchmark_validation_rows
)


print(
    "\n",
    benchmark_validation_df.to_string(
        index=False
    )
)


benchmark_validation_file = (
    OUTPUT_DIR
    /
    "benchmark_forecast_validation.csv"
)


benchmark_validation_df.to_csv(
    benchmark_validation_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    benchmark_validation_file
)


# =====================================================================
# 22. CREATE FUTURE SENTIMENT FORECAST TEMPLATES
# =====================================================================

section(
    "CREATING FUTURE SENTIMENT FORECAST TEMPLATES"
)


for asset in ASSETS:

    benchmark = (
        benchmark_forecasts_by_asset[
            asset
        ]
        .copy()
    )


    template = benchmark[
        [
            "Date",
            "Actual"
        ]
    ].copy()


    template[
        "Forecast"
    ] = np.nan


    template_file = (
        OUTPUT_DIR
        /
        f"{asset.lower()}_sentiment_forecast_template.csv"
    )


    template.to_csv(
        template_file,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        template_file
    )


# =====================================================================
# 23. CHECK WHETHER SENTIMENT FORECASTS EXIST
# =====================================================================

section(
    "CHECKING FOR SENTIMENT-AUGMENTED FORECASTS"
)


sentiment_files_available = {}


for asset in ASSETS:

    sentiment_file = (
        SENTIMENT_FORECAST_FILES[
            asset
        ]
    )


    exists = (
        sentiment_file.exists()
    )


    sentiment_files_available[
        asset
    ] = exists


    print(
        f"\n{asset}:"
    )

    print(
        sentiment_file
    )

    print(
        "AVAILABLE"
        if exists
        else "NOT YET AVAILABLE"
    )


# =====================================================================
# 24. FORMAL COMPARISON FUNCTION
# =====================================================================


def compare_forecasts(
    asset,
    benchmark_forecasts,
    sentiment_file
):

    section(
        f"{asset}: FORMAL BENCHMARK VS SENTIMENT FORECAST COMPARISON"
    )


    sentiment = pd.read_csv(
        sentiment_file
    )


    required_sentiment_columns = [

        "Date",

        "Actual",

        "Forecast"

    ]


    missing_columns = [

        variable

        for variable in required_sentiment_columns

        if variable not in sentiment.columns

    ]


    if missing_columns:

        raise KeyError(
            f"\n{asset} sentiment forecast file is missing:\n"
            +
            "\n".join(
                missing_columns
            )
        )


    sentiment[
        "Date"
    ] = pd.to_datetime(
        sentiment[
            "Date"
        ],
        errors="coerce"
    )


    sentiment[
        "Actual"
    ] = pd.to_numeric(
        sentiment[
            "Actual"
        ],
        errors="coerce"
    )


    sentiment[
        "Forecast"
    ] = pd.to_numeric(
        sentiment[
            "Forecast"
        ],
        errors="coerce"
    )


    if (
        sentiment[
            "Date"
        ]
        .duplicated()
        .any()
    ):

        raise ValueError(
            f"{asset}: duplicate dates in sentiment forecasts."
        )


    benchmark = (
        benchmark_forecasts[
            [
                "Date",
                "Actual",
                "Forecast",
                "Historical_Mean_Forecast"
            ]
        ]
        .rename(
            columns={

                "Actual":
                    "Actual_Benchmark",

                "Forecast":
                    "Forecast_Benchmark"

            }
        )
        .copy()
    )


    sentiment = (
        sentiment[
            [
                "Date",
                "Actual",
                "Forecast"
            ]
        ]
        .rename(
            columns={

                "Actual":
                    "Actual_Sentiment",

                "Forecast":
                    "Forecast_Sentiment"

            }
        )
    )


    # ---------------------------------------------------------
    # Exact common sample
    # ---------------------------------------------------------

    common = (
        benchmark
        .merge(
            sentiment,
            on="Date",
            how="inner",
            validate="one_to_one"
        )
        .dropna(
            subset=[

                "Actual_Benchmark",

                "Forecast_Benchmark",

                "Historical_Mean_Forecast",

                "Actual_Sentiment",

                "Forecast_Sentiment"

            ]
        )
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    if common.empty:

        raise ValueError(
            f"{asset}: no common forecast dates."
        )


    # ---------------------------------------------------------
    # Actual-return consistency
    # ---------------------------------------------------------

    actual_difference = np.abs(
        common[
            "Actual_Benchmark"
        ]
        -
        common[
            "Actual_Sentiment"
        ]
    )


    actual_match = bool(
        (
            actual_difference
            <=
            ACTUAL_MATCH_TOLERANCE
        )
        .all()
    )


    print(
        "\nCommon OOS observations:"
    )

    print(
        len(
            common
        )
    )


    print(
        "\nCommon OOS period:"
    )

    print(
        common[
            "Date"
        ].min(),
        "to",
        common[
            "Date"
        ].max()
    )


    print(
        "\nMaximum actual-return difference:"
    )

    print(
        safe_float(
            actual_difference.max()
        )
    )


    print(
        "\nActual-return match:"
    )

    print(
        "PASS"
        if actual_match
        else "FAIL"
    )


    if not actual_match:

        raise ValueError(
            f"{asset}: actual returns differ between "
            "benchmark and sentiment forecast files."
        )


    common[
        "Actual"
    ] = common[
        "Actual_Benchmark"
    ]


    # ---------------------------------------------------------
    # Forecast errors
    # ---------------------------------------------------------

    common[
        "Error_Benchmark"
    ] = (
        common[
            "Actual"
        ]
        -
        common[
            "Forecast_Benchmark"
        ]
    )


    common[
        "Error_Sentiment"
    ] = (
        common[
            "Actual"
        ]
        -
        common[
            "Forecast_Sentiment"
        ]
    )


    common[
        "Squared_Error_Benchmark"
    ] = (
        common[
            "Error_Benchmark"
        ]
        ** 2
    )


    common[
        "Squared_Error_Sentiment"
    ] = (
        common[
            "Error_Sentiment"
        ]
        ** 2
    )


    common[
        "Absolute_Error_Benchmark"
    ] = np.abs(
        common[
            "Error_Benchmark"
        ]
    )


    common[
        "Absolute_Error_Sentiment"
    ] = np.abs(
        common[
            "Error_Sentiment"
        ]
    )


    # ---------------------------------------------------------
    # Loss differential:
    #
    # benchmark - sentiment
    #
    # Positive = sentiment is better.
    # ---------------------------------------------------------

    common[
        "Squared_Loss_Difference_Benchmark_minus_Sentiment"
    ] = (
        common[
            "Squared_Error_Benchmark"
        ]
        -
        common[
            "Squared_Error_Sentiment"
        ]
    )


    common[
        "Absolute_Loss_Difference_Benchmark_minus_Sentiment"
    ] = (
        common[
            "Absolute_Error_Benchmark"
        ]
        -
        common[
            "Absolute_Error_Sentiment"
        ]
    )


    common[
        "Raw_Error_Difference_Benchmark_minus_Sentiment"
    ] = (
        common[
            "Error_Benchmark"
        ]
        -
        common[
            "Error_Sentiment"
        ]
    )


    # ---------------------------------------------------------
    # Directional indicators
    # ---------------------------------------------------------

    common[
        "Actual_Direction"
    ] = np.sign(
        common[
            "Actual"
        ]
    )


    common[
        "Benchmark_Direction"
    ] = np.sign(
        common[
            "Forecast_Benchmark"
        ]
    )


    common[
        "Sentiment_Direction"
    ] = np.sign(
        common[
            "Forecast_Sentiment"
        ]
    )


    common[
        "Benchmark_Direction_Correct"
    ] = (
        common[
            "Actual_Direction"
        ]
        ==
        common[
            "Benchmark_Direction"
        ]
    )


    common[
        "Sentiment_Direction_Correct"
    ] = (
        common[
            "Actual_Direction"
        ]
        ==
        common[
            "Sentiment_Direction"
        ]
    )


    # ---------------------------------------------------------
    # Metrics on EXACTLY the same dates
    # ---------------------------------------------------------

    benchmark_metrics = calculate_metrics(

        actual=
            common[
                "Actual"
            ],

        forecast=
            common[
                "Forecast_Benchmark"
            ],

        historical_mean=
            common[
                "Historical_Mean_Forecast"
            ],

        model_name=
            "Benchmark"

    )


    sentiment_metrics = calculate_metrics(

        actual=
            common[
                "Actual"
            ],

        forecast=
            common[
                "Forecast_Sentiment"
            ],

        historical_mean=
            common[
                "Historical_Mean_Forecast"
            ],

        model_name=
            "Sentiment_Augmented"

    )


    metrics_df = pd.DataFrame(
        [
            benchmark_metrics,
            sentiment_metrics
        ]
    )


    metrics_df.insert(
        0,
        "Asset",
        asset
    )


    print(
        "\nForecast metrics:"
    )


    print(
        metrics_df[
            [

                "Asset",

                "Model",

                "N",

                "RMSE",

                "MAE",

                "OOS_R_Squared",

                "Directional_Accuracy",

                "Directional_Accuracy_Percent"

            ]
        ]
        .to_string(
            index=False
        )
    )


    # ---------------------------------------------------------
    # Incremental performance
    # ---------------------------------------------------------

    rmse_difference = (
        benchmark_metrics[
            "RMSE"
        ]
        -
        sentiment_metrics[
            "RMSE"
        ]
    )


    mae_difference = (
        benchmark_metrics[
            "MAE"
        ]
        -
        sentiment_metrics[
            "MAE"
        ]
    )


    oos_r2_difference = (
        sentiment_metrics[
            "OOS_R_Squared"
        ]
        -
        benchmark_metrics[
            "OOS_R_Squared"
        ]
    )


    directional_difference = (
        sentiment_metrics[
            "Directional_Accuracy"
        ]
        -
        benchmark_metrics[
            "Directional_Accuracy"
        ]
    )


    if (
        benchmark_metrics[
            "RMSE"
        ]
        != 0
    ):

        rmse_percent_improvement = (
            100.0
            *
            rmse_difference
            /
            benchmark_metrics[
                "RMSE"
            ]
        )

    else:

        rmse_percent_improvement = np.nan


    if (
        benchmark_metrics[
            "MAE"
        ]
        != 0
    ):

        mae_percent_improvement = (
            100.0
            *
            mae_difference
            /
            benchmark_metrics[
                "MAE"
            ]
        )

    else:

        mae_percent_improvement = np.nan


    incremental = pd.DataFrame(
        [
            {

                "Asset":
                    asset,

                "N_Common_OOS":
                    int(
                        len(
                            common
                        )
                    ),

                "OOS_Start":
                    common[
                        "Date"
                    ].min(),

                "OOS_End":
                    common[
                        "Date"
                    ].max(),

                "Benchmark_RMSE":
                    benchmark_metrics[
                        "RMSE"
                    ],

                "Sentiment_RMSE":
                    sentiment_metrics[
                        "RMSE"
                    ],

                "RMSE_Benchmark_minus_Sentiment":
                    safe_float(
                        rmse_difference
                    ),

                "RMSE_Percent_Improvement_Sentiment":
                    safe_float(
                        rmse_percent_improvement
                    ),

                "Benchmark_MAE":
                    benchmark_metrics[
                        "MAE"
                    ],

                "Sentiment_MAE":
                    sentiment_metrics[
                        "MAE"
                    ],

                "MAE_Benchmark_minus_Sentiment":
                    safe_float(
                        mae_difference
                    ),

                "MAE_Percent_Improvement_Sentiment":
                    safe_float(
                        mae_percent_improvement
                    ),

                "Benchmark_OOS_R2":
                    benchmark_metrics[
                        "OOS_R_Squared"
                    ],

                "Sentiment_OOS_R2":
                    sentiment_metrics[
                        "OOS_R_Squared"
                    ],

                "OOS_R2_Sentiment_minus_Benchmark":
                    safe_float(
                        oos_r2_difference
                    ),

                "Benchmark_Directional_Accuracy":
                    benchmark_metrics[
                        "Directional_Accuracy"
                    ],

                "Sentiment_Directional_Accuracy":
                    sentiment_metrics[
                        "Directional_Accuracy"
                    ],

                "Directional_Accuracy_Sentiment_minus_Benchmark":
                    safe_float(
                        directional_difference
                    )

            }
        ]
    )


    # ---------------------------------------------------------
    # DM-type test - squared error
    # ---------------------------------------------------------

    dm_squared = diebold_mariano_type_test(

        common[
            "Squared_Error_Benchmark"
        ],

        common[
            "Squared_Error_Sentiment"
        ],

        maxlags=
            DM_HAC_MAXLAGS

    )


    # ---------------------------------------------------------
    # DM-type test - absolute error
    # ---------------------------------------------------------

    dm_absolute = diebold_mariano_type_test(

        common[
            "Absolute_Error_Benchmark"
        ],

        common[
            "Absolute_Error_Sentiment"
        ],

        maxlags=
            DM_HAC_MAXLAGS

    )


    dm_results = pd.DataFrame(
        [

            {

                "Asset":
                    asset,

                "Loss_Function":
                    "Squared_Error",

                **dm_squared

            },

            {

                "Asset":
                    asset,

                "Loss_Function":
                    "Absolute_Error",

                **dm_absolute

            }

        ]
    )


    dm_results[
        "Significance_Two_Sided"
    ] = (
        dm_results[
            "P_Value_Two_Sided"
        ]
        .apply(
            significance_label
        )
    )


    dm_results[
        "Significance_Sentiment_Better_One_Sided"
    ] = (
        dm_results[
            "P_Value_One_Sided_B_Better"
        ]
        .apply(
            significance_label
        )
    )


    print(
        "\nDM-type forecast comparison:"
    )


    print(
        dm_results.to_string(
            index=False
        )
    )


    # ---------------------------------------------------------
    # H3/H4 decision-support output
    # ---------------------------------------------------------

    squared_mean_diff = (
        dm_squared[
            "Mean_Loss_Differential_A_minus_B"
        ]
    )


    squared_one_sided_p = (
        dm_squared[
            "P_Value_One_Sided_B_Better"
        ]
    )


    formal_improvement = False


    if (
        not pd.isna(
            squared_mean_diff
        )
        and
        not pd.isna(
            squared_one_sided_p
        )
    ):

        formal_improvement = bool(
            (
                squared_mean_diff
                >
                0
            )
            and
            (
                squared_one_sided_p
                <
                SIGNIFICANCE_LEVEL
            )
        )


    decision_support = pd.DataFrame(
        [
            {

                "Asset":
                    asset,

                "Lower_RMSE_Sentiment":
                    bool(
                        sentiment_metrics[
                            "RMSE"
                        ]
                        <
                        benchmark_metrics[
                            "RMSE"
                        ]
                    ),

                "Lower_MAE_Sentiment":
                    bool(
                        sentiment_metrics[
                            "MAE"
                        ]
                        <
                        benchmark_metrics[
                            "MAE"
                        ]
                    ),

                "Higher_OOS_R2_Sentiment":
                    bool(
                        sentiment_metrics[
                            "OOS_R_Squared"
                        ]
                        >
                        benchmark_metrics[
                            "OOS_R_Squared"
                        ]
                    ),

                "Higher_Directional_Accuracy_Sentiment":
                    bool(
                        sentiment_metrics[
                            "Directional_Accuracy"
                        ]
                        >
                        benchmark_metrics[
                            "Directional_Accuracy"
                        ]
                    ),

                "Squared_Loss_Mean_Differential":
                    squared_mean_diff,

                "Squared_Loss_DM_Two_Sided_P":
                    dm_squared[
                        "P_Value_Two_Sided"
                    ],

                "Squared_Loss_DM_One_Sided_Sentiment_Better_P":
                    squared_one_sided_p,

                "Formal_Squared_Loss_Improvement_5pct":
                    formal_improvement

            }
        ]
    )


    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    unique_dates_pass = bool(
        common[
            "Date"
        ]
        .is_unique
    )


    sorted_dates_pass = bool(
        common[
            "Date"
        ]
        .is_monotonic_increasing
    )


    complete_pass = bool(
        common[
            [

                "Actual",

                "Forecast_Benchmark",

                "Forecast_Sentiment",

                "Historical_Mean_Forecast"

            ]
        ]
        .notna()
        .all()
        .all()
    )


    same_denominator_pass = bool(
        np.isclose(

            benchmark_metrics[
                "Historical_Mean_SSE"
            ],

            sentiment_metrics[
                "Historical_Mean_SSE"
            ],

            rtol=1e-12,

            atol=1e-15

        )
    )


    squared_diff_validation = bool(
        np.isclose(

            common[
                "Squared_Loss_Difference_Benchmark_minus_Sentiment"
            ]
            .mean(),

            dm_squared[
                "Mean_Loss_Differential_A_minus_B"
            ],

            rtol=1e-12,

            atol=1e-15

        )
    )


    absolute_diff_validation = bool(
        np.isclose(

            common[
                "Absolute_Loss_Difference_Benchmark_minus_Sentiment"
            ]
            .mean(),

            dm_absolute[
                "Mean_Loss_Differential_A_minus_B"
            ],

            rtol=1e-12,

            atol=1e-15

        )
    )


    validation = pd.DataFrame(
        [
            {

                "Asset":
                    asset,

                "Common_OOS_N":
                    int(
                        len(
                            common
                        )
                    ),

                "Actual_Returns_Match":
                    actual_match,

                "Dates_Unique":
                    unique_dates_pass,

                "Dates_Sorted":
                    sorted_dates_pass,

                "Forecasts_Complete":
                    complete_pass,

                "Identical_OOS_R2_Denominator":
                    same_denominator_pass,

                "Squared_Loss_Differential_Validated":
                    squared_diff_validation,

                "Absolute_Loss_Differential_Validated":
                    absolute_diff_validation

            }
        ]
    )


    overall_pass = bool(
        actual_match
        and
        unique_dates_pass
        and
        sorted_dates_pass
        and
        complete_pass
        and
        same_denominator_pass
        and
        squared_diff_validation
        and
        absolute_diff_validation
    )


    validation[
        "Overall_Validation"
    ] = (
        "PASS"
        if overall_pass
        else "FAIL"
    )


    # ---------------------------------------------------------
    # Save comparison outputs
    # ---------------------------------------------------------

    asset_dir = (
        OUTPUT_DIR
        /
        asset.lower()
    )


    asset_dir.mkdir(
        parents=True,
        exist_ok=True
    )


    common.to_csv(

        asset_dir
        /
        "common_oos_forecast_errors.csv",

        index=False

    )


    metrics_df.to_csv(

        asset_dir
        /
        "forecast_performance_metrics.csv",

        index=False

    )


    incremental.to_csv(

        asset_dir
        /
        "incremental_forecast_performance.csv",

        index=False

    )


    dm_results.to_csv(

        asset_dir
        /
        "diebold_mariano_type_tests.csv",

        index=False

    )


    decision_support.to_csv(

        asset_dir
        /
        "hypothesis_decision_support.csv",

        index=False

    )


    validation.to_csv(

        asset_dir
        /
        "forecast_comparison_validation.csv",

        index=False

    )


    print(
        "\nFormal comparison validation:"
    )


    print(
        "PASS"
        if overall_pass
        else "FAIL"
    )


    if not overall_pass:

        raise ValueError(
            f"{asset}: formal forecast comparison "
            "validation failed."
        )


    return {

        "common":
            common,

        "metrics":
            metrics_df,

        "incremental":
            incremental,

        "dm":
            dm_results,

        "decision":
            decision_support,

        "validation":
            validation

    }


# =====================================================================
# 25. RUN FORMAL COMPARISONS IF SENTIMENT FILES EXIST
# =====================================================================

section(
    "FORMAL FORECAST COMPARISON STATUS"
)


comparison_results = {}


for asset in ASSETS:

    if sentiment_files_available[
        asset
    ]:

        comparison_results[
            asset
        ] = compare_forecasts(

            asset=
                asset,

            benchmark_forecasts=
                benchmark_forecasts_by_asset[
                    asset
                ],

            sentiment_file=
                SENTIMENT_FORECAST_FILES[
                    asset
                ]

        )

    else:

        print(
            f"\n{asset}: sentiment forecast file is not yet "
            "available."
        )

        print(
            "Benchmark infrastructure is ready."
        )

        print(
            "Formal H3/H4 comparison will run automatically "
            "once the sentiment forecast file is added."
        )


# =====================================================================
# 26. SAVE MODEL SPECIFICATIONS
# =====================================================================

section(
    "SAVING MODEL SPECIFICATIONS"
)


specification_rows = []


for asset in ASSETS:

    dependent = (
        MODEL_SPECS[
            asset
        ][
            "dependent"
        ]
    )


    predictors = (
        MODEL_SPECS[
            asset
        ][
            "predictors"
        ]
    )


    for predictor_number, predictor in enumerate(
        predictors,
        start=1
    ):

        specification_rows.append(
            {

                "Asset":
                    asset,

                "Dependent_Variable":
                    dependent,

                "Predictor_Number":
                    predictor_number,

                "Predictor":
                    predictor

            }
        )


specification_df = pd.DataFrame(
    specification_rows
)


specification_file = (
    OUTPUT_DIR
    /
    "forecast_model_specifications.csv"
)


specification_df.to_csv(
    specification_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    specification_file
)


# =====================================================================
# 27. SAVE METHODOLOGY NOTE
# =====================================================================

section(
    "SAVING METHODOLOGY NOTE"
)


methodology_file = (
    OUTPUT_DIR
    /
    "forecast_evaluation_methodology_note.txt"
)


methodology_note = f"""
FORECAST EVALUATION INFRASTRUCTURE

PURPOSE
-------
This script establishes the out-of-sample forecast-evaluation
infrastructure required for H3 and H4.

FORECAST DESIGN
---------------
Forecasts are one-day-ahead daily cryptocurrency return forecasts.

The benchmark models are estimated recursively using an expanding
window.

For each forecast date t, only observations dated strictly before t
are used to estimate the model.

OOS PERIOD
----------
{OOS_START.date()} to {OOS_END.date()}

ASSETS
------
Bitcoin
Ethereum

BENCHMARK MODEL
---------------
The Bitcoin benchmark contains:

BTC_Lagged_Return
Lagged_Log_BTC_Volume
Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

The Ethereum benchmark contains:

ETH_Lagged_Return
Lagged_Log_ETH_Volume
Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

INFORMATION ALIGNMENT
---------------------
Only information-aligned traditional-market controls are used.

The old non-aligned lagged traditional-market variables are not used.

HISTORICAL-MEAN FORECAST
------------------------
For each forecast date, the historical-mean forecast is calculated
from the same expanding training sample used to estimate the
benchmark model.

No observation dated on or after the forecast date enters this
historical mean.

OOS R-SQUARED
-------------
OOS R-squared is calculated as:

R2_OOS = 1 - SSE_model / SSE_historical_mean

A negative OOS R-squared means that the model forecasts perform worse
than the expanding historical-mean forecast over the same OOS sample.

FORECAST PERFORMANCE
--------------------
The evaluation framework reports:

1. RMSE
2. MAE
3. OOS R-squared
4. directional accuracy
5. date-level forecast errors
6. squared-error loss
7. absolute-error loss
8. forecast-loss differences

COMMON OOS SAMPLE
-----------------
When sentiment forecasts become available, the benchmark and
sentiment-augmented models are merged by date.

Formal comparisons are performed only on dates for which both models
have valid forecasts and the realised return is identical.

Both models therefore use exactly the same OOS dates, actual returns
and historical-mean OOS R-squared denominator.

DIEBOLD-MARIANO-TYPE TEST
-------------------------
The formal forecast comparison uses the loss differential:

d_t = Loss_Benchmark,t - Loss_Sentiment,t

Therefore:

d_t > 0
means the sentiment model has lower forecast loss.

d_t < 0
means the benchmark has lower forecast loss.

The long-run variance of the loss differential is estimated using a
Bartlett-weighted HAC/Newey-West estimator with maximum lag:

{DM_HAC_MAXLAGS}

Formal comparisons are reported for:

1. squared-error loss
2. absolute-error loss

The squared-error comparison is the primary formal forecast-accuracy
test because RMSE is a primary forecasting metric.

Both two-sided and one-sided p-values are reported.

H3 AND H4
---------
H3 and H4 should not be evaluated solely from a small difference in
RMSE.

Evidence should consider:

1. whether the sentiment model has lower RMSE;
2. whether it has lower MAE;
3. whether it has higher OOS R-squared;
4. whether it has higher directional accuracy;
5. the mean forecast-loss differential; and
6. the formal DM-type forecast-accuracy comparison.

PRE-REDDIT STATUS
-----------------
Before sentiment forecasts exist, the script generates and validates
the traditional-market benchmark forecasts and saves templates for
the future sentiment forecasts.

It does not create artificial sentiment forecasts.

The absence of a sentiment forecast file therefore does not represent
a methodological or software failure.

Once the sentiment forecast files are supplied, the formal comparison
runs automatically.
""".strip()


with open(
    methodology_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        methodology_note
    )


print(
    "\nSaved:"
)

print(
    methodology_file
)


# =====================================================================
# 28. FINAL INFRASTRUCTURE VALIDATION
# =====================================================================

section(
    "FINAL FORECAST INFRASTRUCTURE VALIDATION"
)


validation_checks = {

    "Final dataset exists":
        FINAL_DATASET_FILE.exists(),

    "No invalid dates":
        (
            invalid_dates
            ==
            0
        ),

    "No duplicate dates":
        (
            duplicate_dates
            ==
            0
        ),

    "Continuous daily calendar":
        (
            calendar_gaps
            ==
            0
        ),

    "No old non-aligned controls used":
        (
            len(
                old_controls_used
            )
            ==
            0
        ),

    "BTC benchmark forecasts generated":
        (
            len(
                benchmark_forecasts_by_asset[
                    "BTC"
                ]
            )
            >
            0
        ),

    "ETH benchmark forecasts generated":
        (
            len(
                benchmark_forecasts_by_asset[
                    "ETH"
                ]
            )
            >
            0
        ),

    "BTC training strictly prior to forecast date":
        bool(
            (
                benchmark_forecasts_by_asset[
                    "BTC"
                ][
                    "Training_End"
                ]
                <
                benchmark_forecasts_by_asset[
                    "BTC"
                ][
                    "Date"
                ]
            )
            .all()
        ),

    "ETH training strictly prior to forecast date":
        bool(
            (
                benchmark_forecasts_by_asset[
                    "ETH"
                ][
                    "Training_End"
                ]
                <
                benchmark_forecasts_by_asset[
                    "ETH"
                ][
                    "Date"
                ]
            )
            .all()
        ),

    "BTC benchmark forecasts complete":
        bool(
            benchmark_forecasts_by_asset[
                "BTC"
            ][
                [
                    "Actual",
                    "Forecast",
                    "Historical_Mean_Forecast"
                ]
            ]
            .notna()
            .all()
            .all()
        ),

    "ETH benchmark forecasts complete":
        bool(
            benchmark_forecasts_by_asset[
                "ETH"
            ][
                [
                    "Actual",
                    "Forecast",
                    "Historical_Mean_Forecast"
                ]
            ]
            .notna()
            .all()
            .all()
        )

}


for check_name, condition in validation_checks.items():

    print(
        f"\n{check_name}: "
        f"{'PASS' if condition else 'FAIL'}"
    )


overall_infrastructure_pass = all(
    validation_checks.values()
)


print(
    "\n" + "-" * 80
)


print(
    "\nOVERALL FORECAST INFRASTRUCTURE VALIDATION:"
)


print(
    "PASS"
    if overall_infrastructure_pass
    else "FAIL"
)


if not overall_infrastructure_pass:

    failed_checks = [

        name

        for name, condition
        in validation_checks.items()

        if not condition

    ]


    print(
        "\nFailed checks:"
    )


    for check in failed_checks:

        print(
            " -",
            check
        )


    raise ValueError(
        "Forecast infrastructure validation failed."
    )


# =====================================================================
# 29. STATUS
# =====================================================================

section(
    "FORECAST EVALUATION STATUS"
)


btc_sentiment_available = (
    sentiment_files_available[
        "BTC"
    ]
)


eth_sentiment_available = (
    sentiment_files_available[
        "ETH"
    ]
)


if (
    btc_sentiment_available
    and
    eth_sentiment_available
):

    status = (
        "COMPLETE - BTC AND ETH FORMAL "
        "SENTIMENT COMPARISONS RUN"
    )


elif (
    btc_sentiment_available
    or
    eth_sentiment_available
):

    status = (
        "PARTIALLY COMPLETE - ONE SENTIMENT "
        "FORECAST SERIES AVAILABLE"
    )


else:

    status = (
        "READY - WAITING FOR REDDIT"
    )


print(
    "\nINFRASTRUCTURE STATUS:"
)

print(
    status
)


# =====================================================================
# 30. INTERPRETATION REMINDERS
# =====================================================================

section(
    "INTERPRETATION REMINDERS"
)


print(
    "\n1. These are genuine expanding-window "
    "one-day-ahead forecasts."
)


print(
    "\n2. Training observations are strictly earlier "
    "than each forecast date."
)


print(
    "\n3. The benchmark uses the corrected information-aligned "
    "traditional-market controls."
)


print(
    "\n4. The old non-aligned traditional-market variables "
    "are not used."
)


print(
    "\n5. OOS R-squared is evaluated against an expanding "
    "historical-mean forecast."
)


print(
    "\n6. Negative OOS R-squared is possible and should not "
    "be hidden or treated as a coding failure."
)


print(
    "\n7. Once sentiment forecasts exist, both models are "
    "evaluated only on their exact common OOS dates."
)


print(
    "\n8. Positive benchmark-minus-sentiment loss differences "
    "mean the sentiment model has lower loss."
)


print(
    "\n9. H3/H4 must not be supported solely because the "
    "sentiment model has a marginally lower RMSE."
)


print(
    "\n10. Formal forecast-loss comparison should be interpreted "
    "alongside RMSE, MAE, OOS R2 and directional accuracy."
)


# =====================================================================
# 31. COMPLETE
# =====================================================================

section(
    "FORECAST EVALUATION INFRASTRUCTURE COMPLETE"
)


print(
    "\nBenchmark forecasting:"
)

print(
    "COMPLETE"
)


print(
    "\nFormal sentiment comparison:"
)


if (
    btc_sentiment_available
    or
    eth_sentiment_available
):

    print(
        "RUN WHERE SENTIMENT FORECAST FILES WERE AVAILABLE"
    )

else:

    print(
        "PENDING REDDIT DATA - THIS IS EXPECTED"
    )


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nOverall infrastructure validation:"
)

print(
    "PASS"
)


print(
    "\nFinal status:"
)

print(
    status
)