# =============================================================================
# 10_construct_forecast_structure.py
#
# FINAL CHRONOLOGICAL FORECASTING / TRAIN-TEST STRUCTURE
#
# Purpose:
#   Prepare the final time-series structure required for genuine
#   out-of-sample forecasting of BTC and ETH daily returns.
#
# IMPORTANT:
#   This script DOES NOT estimate regressions or produce forecasts.
#
#   It:
#       1. Reads the INFORMATION-ALIGNED dataset.
#       2. Preserves chronological ordering.
#       3. Defines the initial estimation period.
#       4. Defines the out-of-sample evaluation period.
#       5. Creates one-day-ahead expanding-window forecast origins.
#       6. Checks that training observations always precede forecast dates.
#       7. Defines the final BTC and ETH benchmark predictor sets.
#       8. Uses information-aligned traditional-market predictors.
#       9. Defines cross-crypto robustness controls.
#      10. Performs look-ahead and information-alignment checks.
#
# Forecast design:
#
#   Initial estimation sample:
#       2021-01-01 to 2023-12-31
#
#   Out-of-sample evaluation:
#       2024-01-01 to 2025-12-31
#
#   Forecast horizon:
#       One calendar day ahead
#
#   Estimation window:
#       Expanding
#
# IMPORTANT INFORMATION-TIMING RULE:
#
#   Crypto markets trade 24/7, while traditional financial markets do not.
#
#   Traditional-market controls have already been aligned so that, for each
#   crypto date t, they contain the most recent transformed traditional-market
#   observation available STRICTLY BEFORE date t.
#
#   Therefore the predictive specification uses:
#
#       Lagged_SP500_Return_Aligned
#       Lagged_VIX_Change_Aligned
#       Lagged_Gold_Return_Aligned
#       Lagged_DXY_Return_Aligned
#       Lagged_US10Y_Change_Aligned
#
#   Same-day traditional-market variables are NOT used in the forecasting
#   specification.
#
#   No traditional-market missing values are replaced with zero.
#
# =============================================================================


from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

PROCESSED_DIR = PROJECT_ROOT / "data_processed"


# IMPORTANT:
# Use the information-aligned dataset, NOT master_aligned_dataset.csv.

INPUT_FILE = (
    PROCESSED_DIR
    / "information_aligned_dataset.csv"
)


# Final forecasting dataset

OUTPUT_STRUCTURE_FILE = (
    PROCESSED_DIR
    / "final_forecast_dataset.csv"
)


# Separate table containing the expanding-window forecast origins

OUTPUT_ORIGINS_FILE = (
    PROCESSED_DIR
    / "forecast_origins.csv"
)


# =============================================================================
# 2. FORECAST DESIGN SETTINGS
# =============================================================================

SAMPLE_START = pd.Timestamp("2021-01-01")

INITIAL_ESTIMATION_END = pd.Timestamp("2023-12-31")

OOS_START = pd.Timestamp("2024-01-01")

OOS_END = pd.Timestamp("2025-12-31")

FORECAST_HORIZON_DAYS = 1


# =============================================================================
# 3. MODEL VARIABLE DEFINITIONS
# =============================================================================

# -----------------------------------------------------------------------------
# Dependent variables
# -----------------------------------------------------------------------------

BTC_DEPENDENT = "BTC_Return"

ETH_DEPENDENT = "ETH_Return"


# -----------------------------------------------------------------------------
# BTC benchmark forecasting controls
# -----------------------------------------------------------------------------
#
# All predictors below are known before the BTC return on forecast date t.
#
# BTC_Lagged_Return:
#     BTC return from t-1.
#
# Lagged_Log_BTC_Volume:
#     BTC trading volume information from t-1.
#
# Traditional-market variables:
#     Most recent transformed observation strictly before crypto date t.
# -----------------------------------------------------------------------------

BTC_BENCHMARK_CONTROLS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume",

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned",
]


# -----------------------------------------------------------------------------
# ETH benchmark forecasting controls
# -----------------------------------------------------------------------------

ETH_BENCHMARK_CONTROLS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume",

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned",
]


# -----------------------------------------------------------------------------
# Cross-crypto robustness controls
# -----------------------------------------------------------------------------
#
# These are NOT part of the main benchmark model.
#
# They will be added in robustness specifications to address the tutor's
# omitted-variable concern regarding the close BTC/ETH relationship.
# -----------------------------------------------------------------------------

BTC_CROSS_CRYPTO_CONTROL = "ETH_Lagged_Return"

ETH_CROSS_CRYPTO_CONTROL = "BTC_Lagged_Return"


# -----------------------------------------------------------------------------
# Source-date variables retained for auditability
# -----------------------------------------------------------------------------
#
# These allow us to verify that traditional-market information used on crypto
# date t genuinely comes from a date before t.
# -----------------------------------------------------------------------------

SOURCE_DATE_COLUMNS = [

    "SP500_Source_Date",

    "VIX_Source_Date",

    "Gold_Source_Date",

    "DXY_Source_Date",

    "US10Y_Source_Date",
]


# Mapping between each aligned predictor and its source-date column.

ALIGNED_SOURCE_MAP = {

    "Lagged_SP500_Return_Aligned":
        "SP500_Source_Date",

    "Lagged_VIX_Change_Aligned":
        "VIX_Source_Date",

    "Lagged_Gold_Return_Aligned":
        "Gold_Source_Date",

    "Lagged_DXY_Return_Aligned":
        "DXY_Source_Date",

    "Lagged_US10Y_Change_Aligned":
        "US10Y_Source_Date",
}


# =============================================================================
# 4. HELPER FUNCTION
# =============================================================================

def print_section(title):

    print("\n" + "=" * 70)

    print(title)

    print("=" * 70)


# =============================================================================
# 5. START
# =============================================================================

print_section(
    "FINAL CHRONOLOGICAL FORECASTING / TRAIN-TEST STRUCTURE"
)


# =============================================================================
# 6. CHECK INPUT FILE
# =============================================================================

print("\nInput file:")

print(INPUT_FILE)


print("\nDoes input file exist?")

print(INPUT_FILE.exists())


if not INPUT_FILE.exists():

    raise FileNotFoundError(

        "\nInformation-aligned dataset was not found:\n"
        f"{INPUT_FILE}"
    )


# =============================================================================
# 7. IMPORT INFORMATION-ALIGNED DATASET
# =============================================================================

print_section(
    "IMPORTING INFORMATION-ALIGNED DATASET"
)


df = pd.read_csv(
    INPUT_FILE
)


print("\nImported shape:")

print(df.shape)


print("\nColumns found:")

for column in df.columns:

    print(column)


# =============================================================================
# 8. VALIDATE DATE VARIABLE
# =============================================================================

print_section(
    "VALIDATING DATE VARIABLE"
)


if "Date" not in df.columns:

    raise KeyError(
        "Required column 'Date' was not found."
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


invalid_dates = (
    df["Date"]
    .isna()
    .sum()
)


print("\nInvalid dates:")

print(invalid_dates)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates detected."
    )


# =============================================================================
# 9. CONVERT SOURCE-DATE VARIABLES
# =============================================================================

print_section(
    "VALIDATING TRADITIONAL-MARKET SOURCE DATES"
)


existing_source_columns = [

    column

    for column in SOURCE_DATE_COLUMNS

    if column in df.columns
]


print("\nSource-date columns found:")

for column in existing_source_columns:

    print(column)


for column in existing_source_columns:

    df[column] = pd.to_datetime(
        df[column],
        errors="coerce"
    )


# =============================================================================
# 10. SORT CHRONOLOGICALLY
# =============================================================================

print_section(
    "SORTING DATA CHRONOLOGICALLY"
)


df = (

    df
    .sort_values("Date")
    .reset_index(drop=True)

)


print("\nFirst date:")

print(df["Date"].min())


print("\nLast date:")

print(df["Date"].max())


# =============================================================================
# 11. CHECK DUPLICATE DATES
# =============================================================================

print_section(
    "CHECKING DUPLICATE DATES"
)


duplicate_dates = (
    df["Date"]
    .duplicated()
    .sum()
)


print("\nDuplicate dates:")

print(duplicate_dates)


if duplicate_dates > 0:

    raise ValueError(
        "Duplicate dates detected."
    )


# =============================================================================
# 12. RESTRICT TO STUDY PERIOD
# =============================================================================

print_section(
    "RESTRICTING TO STUDY PERIOD"
)


df = df.loc[

    (df["Date"] >= SAMPLE_START)

    &

    (df["Date"] <= OOS_END)

].copy()


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


# =============================================================================
# 13. CHECK CRYPTO CALENDAR CONTINUITY
# =============================================================================

print_section(
    "CHECKING CRYPTO CALENDAR CONTINUITY"
)


date_differences = (
    df["Date"]
    .diff()
)


non_consecutive = (
    date_differences
    .dropna()
    .ne(pd.Timedelta(days=1))
    .sum()
)


print(
    "\nNon-consecutive calendar observations:"
)

print(non_consecutive)


if non_consecutive == 0:

    print(
        "\nPASS: Crypto forecasting calendar "
        "contains consecutive calendar days."
    )

else:

    raise ValueError(
        "Crypto calendar is not continuous."
    )


# =============================================================================
# 14. VALIDATE REQUIRED MODEL VARIABLES
# =============================================================================

print_section(
    "VALIDATING REQUIRED MODEL VARIABLES"
)


required_variables = [

    BTC_DEPENDENT,

    ETH_DEPENDENT,

] + BTC_BENCHMARK_CONTROLS + ETH_BENCHMARK_CONTROLS + [

    BTC_CROSS_CRYPTO_CONTROL,

    ETH_CROSS_CRYPTO_CONTROL,
]


# Remove duplicates while preserving order.

required_variables = list(
    dict.fromkeys(required_variables)
)


missing_required_variables = [

    variable

    for variable in required_variables

    if variable not in df.columns
]


if missing_required_variables:

    print(
        "\nMissing required variables:"
    )

    for variable in missing_required_variables:

        print(variable)

    raise KeyError(
        "Required forecasting variables are missing."
    )


print(
    "\nAll required forecasting variables are present."
)


# =============================================================================
# 15. DISPLAY FINAL BTC BENCHMARK SPECIFICATION
# =============================================================================

print_section(
    "BTC BENCHMARK FORECASTING SPECIFICATION"
)


print(
    "\nDependent variable:"
)

print(BTC_DEPENDENT)


print(
    "\nPredictors:"
)

for variable in BTC_BENCHMARK_CONTROLS:

    print(variable)


print(
    "\nCross-crypto robustness control:"
)

print(BTC_CROSS_CRYPTO_CONTROL)


# =============================================================================
# 16. DISPLAY FINAL ETH BENCHMARK SPECIFICATION
# =============================================================================

print_section(
    "ETH BENCHMARK FORECASTING SPECIFICATION"
)


print(
    "\nDependent variable:"
)

print(ETH_DEPENDENT)


print(
    "\nPredictors:"
)

for variable in ETH_BENCHMARK_CONTROLS:

    print(variable)


print(
    "\nCross-crypto robustness control:"
)

print(ETH_CROSS_CRYPTO_CONTROL)


# =============================================================================
# 17. VERIFY TRADITIONAL-MARKET INFORMATION TIMING
# =============================================================================

print_section(
    "VERIFYING TRADITIONAL-MARKET INFORMATION TIMING"
)


total_timing_errors = 0


for aligned_variable, source_column in ALIGNED_SOURCE_MAP.items():

    print(
        f"\n{aligned_variable}"
    )

    if source_column not in df.columns:

        print(
            f"WARNING: {source_column} was not found. "
            "Timing audit cannot be performed for this variable."
        )

        continue


    # Only inspect observations for which the aligned variable is available.

    valid_rows = (

        df[aligned_variable].notna()

        &

        df[source_column].notna()
    )


    timing_errors = (

        df.loc[
            valid_rows,
            source_column
        ]

        >=

        df.loc[
            valid_rows,
            "Date"
        ]

    ).sum()


    print(
        "Observations with source date >= crypto date:"
    )

    print(timing_errors)


    total_timing_errors += timing_errors


print(
    "\nTotal traditional-market timing errors:"
)

print(total_timing_errors)


if total_timing_errors == 0:

    print(
        "\nPASS: All available traditional-market "
        "information comes from dates strictly "
        "before the crypto forecast date."
    )

else:

    raise ValueError(
        "Traditional-market look-ahead bias detected."
    )


# =============================================================================
# 18. CHECK WEEKEND INFORMATION AVAILABILITY
# =============================================================================

print_section(
    "CHECKING WEEKEND INFORMATION AVAILABILITY"
)


df["Day_of_Week"] = (
    df["Date"]
    .dt.day_name()
)


df["Weekend"] = (
    df["Date"]
    .dt.dayofweek
    .isin([5, 6])
    .astype(int)
)


weekend_observations = (
    df["Weekend"] == 1
)


print(
    "\nNumber of weekend crypto observations:"
)

print(
    weekend_observations.sum()
)


aligned_market_variables = list(
    ALIGNED_SOURCE_MAP.keys()
)


for variable in aligned_market_variables:

    weekend_available = (

        df.loc[
            weekend_observations,
            variable
        ]
        .notna()
        .sum()
    )


    weekend_missing = (

        df.loc[
            weekend_observations,
            variable
        ]
        .isna()
        .sum()
    )


    print(
        f"\n{variable}"
    )

    print(
        "Weekend observations available:"
    )

    print(
        weekend_available
    )

    print(
        "Weekend observations missing:"
    )

    print(
        weekend_missing
    )


# =============================================================================
# 19. CREATE FORECAST SAMPLE INDICATOR
# =============================================================================

print_section(
    "CREATING TRAIN / TEST INDICATORS"
)


df["Forecast_Sample"] = np.select(

    [

        (
            (df["Date"] >= SAMPLE_START)

            &

            (
                df["Date"]
                <= INITIAL_ESTIMATION_END
            )
        ),

        (
            (df["Date"] >= OOS_START)

            &

            (df["Date"] <= OOS_END)
        ),

    ],

    [

        "Initial_Estimation",

        "Out_of_Sample",

    ],

    default="Outside_Sample"
)


# =============================================================================
# 20. CREATE OOS FORECAST INDICATOR
# =============================================================================

df["OOS_Forecast"] = (

    (
        (df["Date"] >= OOS_START)

        &

        (df["Date"] <= OOS_END)
    )

    .astype(int)

)


# =============================================================================
# 21. CREATE FORECAST NUMBER
# =============================================================================

df["Forecast_Number"] = pd.Series(
    pd.NA,
    index=df.index,
    dtype="Int64"
)


oos_mask = (
    df["OOS_Forecast"] == 1
)


df.loc[
    oos_mask,
    "Forecast_Number"
] = np.arange(
    1,
    oos_mask.sum() + 1
)


# =============================================================================
# 22. CHECK TRAIN / TEST COUNTS
# =============================================================================

print_section(
    "TRAIN / TEST SAMPLE COUNTS"
)


print(
    "\nForecast sample counts:"
)

print(
    df["Forecast_Sample"]
    .value_counts()
    .sort_index()
)


print(
    "\nNumber of OOS forecast dates:"
)

print(
    df["OOS_Forecast"]
    .sum()
)


# =============================================================================
# 23. CHECK TRAIN / TEST BOUNDARY
# =============================================================================

print_section(
    "CHECKING TRAIN / TEST BOUNDARY"
)


boundary_check = df.loc[

    (
        df["Date"]
        >= INITIAL_ESTIMATION_END
        - pd.Timedelta(days=4)
    )

    &

    (
        df["Date"]
        <= OOS_START
        + pd.Timedelta(days=4)
    ),

    [

        "Date",

        "Forecast_Sample",

        "OOS_Forecast",

        "Forecast_Number",

        "BTC_Return",

        "ETH_Return",
    ]

]


print(
    boundary_check
    .to_string(index=False)
)


# =============================================================================
# 24. CREATE EXPANDING-WINDOW FORECAST ORIGINS
# =============================================================================

print_section(
    "CREATING EXPANDING-WINDOW FORECAST ORIGINS"
)


forecast_dates = (

    df.loc[
        df["OOS_Forecast"] == 1,
        "Date"
    ]

    .sort_values()

    .reset_index(drop=True)
)


forecast_records = []


for forecast_number, forecast_date in enumerate(
    forecast_dates,
    start=1
):

    # -------------------------------------------------------------------------
    # Training observations must occur STRICTLY before forecast date.
    # -------------------------------------------------------------------------

    training_dates = df.loc[

        (
            df["Date"] >= SAMPLE_START
        )

        &

        (
            df["Date"] < forecast_date
        ),

        "Date"

    ]


    if training_dates.empty:

        raise ValueError(
            f"No training observations available "
            f"for forecast date {forecast_date}."
        )


    train_start = (
        training_dates.min()
    )


    train_end = (
        training_dates.max()
    )


    training_n = (
        len(training_dates)
    )


    # -------------------------------------------------------------------------
    # Look-ahead check
    # -------------------------------------------------------------------------

    if train_end >= forecast_date:

        raise ValueError(

            "LOOK-AHEAD ERROR: "
            "training data include the forecast date "
            "or a future observation."
        )


    forecast_records.append(

        {

            "Forecast_Number":
                forecast_number,

            "Forecast_Date":
                forecast_date,

            "Training_Start":
                train_start,

            "Training_End":
                train_end,

            "Training_N":
                training_n,

            "Forecast_Horizon_Days":
                FORECAST_HORIZON_DAYS,

            "Window_Type":
                "Expanding",
        }
    )


forecast_origins = pd.DataFrame(
    forecast_records
)


# =============================================================================
# 25. CHECK FIRST FORECAST ORIGINS
# =============================================================================

print_section(
    "FIRST 10 FORECAST ORIGINS"
)


print(

    forecast_origins
    .head(10)
    .to_string(index=False)

)


# =============================================================================
# 26. CHECK LAST FORECAST ORIGINS
# =============================================================================

print_section(
    "LAST 10 FORECAST ORIGINS"
)


print(

    forecast_origins
    .tail(10)
    .to_string(index=False)

)


# =============================================================================
# 27. LOOK-AHEAD BIAS CHECK
# =============================================================================

print_section(
    "LOOK-AHEAD BIAS CHECK"
)


lookahead_errors = (

    forecast_origins[
        "Training_End"
    ]

    >=

    forecast_origins[
        "Forecast_Date"
    ]

).sum()


print(
    "\nForecast origins with "
    "training end >= forecast date:"
)

print(
    lookahead_errors
)


if lookahead_errors == 0:

    print(
        "\nPASS: Every training window ends "
        "before its forecast date."
    )

else:

    raise ValueError(
        "Look-ahead bias detected in forecast structure."
    )


# =============================================================================
# 28. CHECK TRAINING WINDOW EXPANSION
# =============================================================================

print_section(
    "CHECKING EXPANDING WINDOW"
)


training_n_difference = (

    forecast_origins[
        "Training_N"
    ]

    .diff()

    .dropna()
)


non_expanding = (

    training_n_difference
    <= 0

).sum()


print(
    "\nCases where training sample "
    "failed to expand:"
)

print(
    non_expanding
)


if non_expanding == 0:

    print(
        "\nPASS: Training sample expands "
        "through the OOS period."
    )

else:

    raise ValueError(
        "Expanding-window construction error."
    )


# =============================================================================
# 29. CHECK BENCHMARK MISSINGNESS
# =============================================================================

print_section(
    "BTC BENCHMARK VARIABLE MISSINGNESS"
)


btc_variables = [

    BTC_DEPENDENT

] + BTC_BENCHMARK_CONTROLS


print(

    df[
        btc_variables
    ]

    .isna()

    .sum()

)


print_section(
    "ETH BENCHMARK VARIABLE MISSINGNESS"
)


eth_variables = [

    ETH_DEPENDENT

] + ETH_BENCHMARK_CONTROLS


print(

    df[
        eth_variables
    ]

    .isna()

    .sum()

)


# =============================================================================
# 30. CREATE COMPLETE-CASE FLAGS
# =============================================================================

print_section(
    "CREATING BENCHMARK COMPLETE-CASE FLAGS"
)


df["BTC_Benchmark_Complete"] = (

    df[
        btc_variables
    ]

    .notna()

    .all(axis=1)

    .astype(int)
)


df["ETH_Benchmark_Complete"] = (

    df[
        eth_variables
    ]

    .notna()

    .all(axis=1)

    .astype(int)
)


print(
    "\nBTC benchmark-complete observations:"
)

print(
    df["BTC_Benchmark_Complete"]
    .sum()
)


print(
    "\nETH benchmark-complete observations:"
)

print(
    df["ETH_Benchmark_Complete"]
    .sum()
)


# =============================================================================
# 31. COMPLETE OBSERVATIONS BY FORECAST SAMPLE
# =============================================================================

print_section(
    "COMPLETE OBSERVATIONS BY FORECAST SAMPLE"
)


completion_summary = (

    df.groupby(
        "Forecast_Sample"
    )[
        [
            "BTC_Benchmark_Complete",
            "ETH_Benchmark_Complete"
        ]
    ]

    .agg(
        ["sum", "count"]
    )
)


print(
    completion_summary
)


# =============================================================================
# 32. CHECK OOS COMPLETENESS
# =============================================================================

print_section(
    "OUT-OF-SAMPLE BENCHMARK COMPLETENESS"
)


oos_df = df.loc[
    df["OOS_Forecast"] == 1
]


btc_oos_complete = (
    oos_df[
        "BTC_Benchmark_Complete"
    ]
    .sum()
)


eth_oos_complete = (
    oos_df[
        "ETH_Benchmark_Complete"
    ]
    .sum()
)


total_oos = len(
    oos_df
)


print(
    "\nTotal OOS dates:"
)

print(
    total_oos
)


print(
    "\nBTC complete OOS dates:"
)

print(
    btc_oos_complete
)


print(
    "\nETH complete OOS dates:"
)

print(
    eth_oos_complete
)


# =============================================================================
# 33. INFORMATION-ALIGNMENT CONFIRMATION
# =============================================================================

print_section(
    "TRADITIONAL-MARKET INFORMATION ALIGNMENT"
)


print(
    """
Traditional-market predictors have already been aligned to the
7-day cryptocurrency calendar using a predetermined information-
availability convention.

For each cryptocurrency date t, the aligned traditional-market
variable contains the most recent transformed traditional-market
observation available strictly before date t.

Therefore:

    - Same-day traditional-market information is not used.
    - Weekend crypto observations use the most recently available
      prior traditional-market information.
    - Market-closed days are not interpreted as zero returns.
    - No future traditional-market observations are used.
    - No look-ahead information is introduced.

The predictive traditional-market variables are:

    Lagged_SP500_Return_Aligned
    Lagged_VIX_Change_Aligned
    Lagged_Gold_Return_Aligned
    Lagged_DXY_Return_Aligned
    Lagged_US10Y_Change_Aligned
"""
)


# =============================================================================
# 34. MODEL DESIGN SUMMARY
# =============================================================================

print_section(
    "FINAL MODEL DESIGN"
)


print(
    """
MAIN BTC BENCHMARK:

    BTC_Return(t)

        <- BTC_Lagged_Return(t)
        <- Lagged_Log_BTC_Volume(t)
        <- Lagged_SP500_Return_Aligned(t)
        <- Lagged_VIX_Change_Aligned(t)
        <- Lagged_Gold_Return_Aligned(t)
        <- Lagged_DXY_Return_Aligned(t)
        <- Lagged_US10Y_Change_Aligned(t)


BTC CROSS-CRYPTO ROBUSTNESS:

    Main BTC benchmark
        +
    ETH_Lagged_Return(t)


MAIN ETH BENCHMARK:

    ETH_Return(t)

        <- ETH_Lagged_Return(t)
        <- Lagged_Log_ETH_Volume(t)
        <- Lagged_SP500_Return_Aligned(t)
        <- Lagged_VIX_Change_Aligned(t)
        <- Lagged_Gold_Return_Aligned(t)
        <- Lagged_DXY_Return_Aligned(t)
        <- Lagged_US10Y_Change_Aligned(t)


ETH CROSS-CRYPTO ROBUSTNESS:

    Main ETH benchmark
        +
    BTC_Lagged_Return(t)
"""
)


# =============================================================================
# 35. SAVE FINAL FORECAST DATASET
# =============================================================================

print_section(
    "SAVING FINAL FORECAST DATASET"
)


df.to_csv(
    OUTPUT_STRUCTURE_FILE,
    index=False
)


print(
    "\nFinal forecast dataset saved to:"
)

print(
    OUTPUT_STRUCTURE_FILE
)


# =============================================================================
# 36. SAVE FORECAST ORIGINS
# =============================================================================

forecast_origins.to_csv(
    OUTPUT_ORIGINS_FILE,
    index=False
)


print(
    "\nForecast origins saved to:"
)

print(
    OUTPUT_ORIGINS_FILE
)


# =============================================================================
# 37. FINAL FORECAST DESIGN SUMMARY
# =============================================================================

print_section(
    "FINAL FORECAST DESIGN SUMMARY"
)


print(
    "\nFull sample:"
)

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


print(
    "\nInitial estimation period:"
)

print(
    SAMPLE_START,
    "to",
    INITIAL_ESTIMATION_END
)


print(
    "\nOut-of-sample evaluation period:"
)

print(
    OOS_START,
    "to",
    OOS_END
)


print(
    "\nForecast horizon:"
)

print(
    f"{FORECAST_HORIZON_DAYS} day ahead"
)


print(
    "\nWindow type:"
)

print(
    "Expanding window"
)


print(
    "\nNumber of OOS forecast dates:"
)

print(
    len(forecast_origins)
)


print(
    "\nFirst training window:"
)

print(
    forecast_origins.iloc[0][
        [
            "Training_Start",
            "Training_End",
            "Training_N"
        ]
    ]
)


print(
    "\nFinal training window:"
)

print(
    forecast_origins.iloc[-1][
        [
            "Training_Start",
            "Training_End",
            "Training_N"
        ]
    ]
)


# =============================================================================
# 38. IMPORTANT NEXT STEP
# =============================================================================

print_section(
    "IMPORTANT NEXT STEP"
)


print(
    """
The chronological forecasting structure and traditional-market
information alignment are now complete.

The next empirical step is to estimate the BASELINE BTC and ETH
models WITHOUT Reddit variables.

Two separate tasks should then be distinguished:

1. IN-SAMPLE EXPLANATORY ANALYSIS
   Estimate the benchmark regressions and use appropriate robust
   inference, including HAC/Newey-West standard errors.

2. OUT-OF-SAMPLE PREDICTIVE ANALYSIS
   Produce genuine one-day-ahead forecasts using the expanding
   training windows defined in forecast_origins.csv.

When Reddit sentiment/activity data become available, the sentiment
models must use exactly the same forecast dates, forecasting horizon,
information-timing convention, and expanding-window procedure.

This will allow direct comparison of:

    Benchmark model
        versus
    Benchmark + Reddit activity
        versus
    Benchmark + Reddit sentiment
        versus
    Benchmark + activity + sentiment

using genuine out-of-sample forecasting measures such as RMSE and
out-of-sample R-squared.
"""
)


print_section(
    "FORECAST STRUCTURE CONSTRUCTION COMPLETE"
)