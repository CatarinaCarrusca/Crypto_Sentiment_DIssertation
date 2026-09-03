# =====================================================================
# 06_verify_trading_volume.py
#
# FINAL TRADING-VOLUME TRANSFORMATION AND TIMING VERIFICATION
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
# Verify the BTC and ETH trading-volume variables already constructed
# for the dissertation.
#
# This is primarily a VERIFICATION exercise rather than a new data
# construction exercise.
#
# The script independently checks:
#
#   1. Raw Yahoo Finance volume data are valid.
#
#   2. Stored transformed volume is numerically equivalent to:
#
#          Log_Volume_t = ln(1 + Volume_t)
#
#      using numpy.log1p().
#
#   3. Stored forecasting volume is numerically equivalent to:
#
#          Lagged_Log_Volume_t = Log_Volume_(t-1)
#
#   4. The crypto-volume files follow a consecutive DAILY calendar,
#      so shift(1) genuinely represents the previous calendar day.
#
#   5. Weekend observations are retained.
#
#   6. The processed BTC/ETH volume files agree with the volume
#      variables contained in final_forecast_dataset.csv.
#
#   7. Raw and transformed skewness are reported to document the
#      motivation for the logarithmic transformation.
#
#   8. Tiny floating-point / CSV serialization differences are treated
#      appropriately using numerical equivalence rather than requiring
#      exact binary equality.
#
# =====================================================================
#
# SOURCE INTERPRETATION
# ---------------------------------------------------------------------
# Source:
# Yahoo Finance via yfinance.
#
# The Volume field is described conservatively as:
#
#   "the daily trading-volume field reported for the downloaded
#    Yahoo Finance cryptocurrency series."
#
# Do NOT describe this automatically as total global BTC or ETH
# trading volume unless separate source documentation establishes
# that interpretation.
#
# =====================================================================
#
# TIMING
# ---------------------------------------------------------------------
# Cryptocurrency markets operate seven days per week.
#
# Therefore, for crypto return date t:
#
#   Forecasting volume information = transformed volume from t-1
#
# where t-1 is the previous CALENDAR day.
#
# =====================================================================
#
# IMPORTANT
# ---------------------------------------------------------------------
# This script does NOT overwrite:
#
#   btc_volume_processed.csv
#   eth_volume_processed.csv
#   final_forecast_dataset.csv
#
# It only verifies them and writes diagnostic outputs.
#
# =====================================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "trading_volume_verification"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# 2. INPUT FILES
# =====================================================================

BTC_VOLUME_FILE = (
    DATA_PROCESSED
    / "btc_volume_processed.csv"
)

ETH_VOLUME_FILE = (
    DATA_PROCESSED
    / "eth_volume_processed.csv"
)

FINAL_DATASET_FILE = (
    DATA_PROCESSED
    / "final_forecast_dataset.csv"
)


# =====================================================================
# 3. NUMERICAL TOLERANCES
# =====================================================================
#
# IMPORTANT:
#
# Values read from CSV can differ from independently reconstructed
# floating-point values by extremely small amounts because decimal
# text must be converted back into binary floating-point numbers.
#
# We therefore test NUMERICAL EQUIVALENCE rather than exact binary
# equality.
#
# These tolerances are many orders of magnitude smaller than the
# economic scale of the transformed volume variables.
# =====================================================================

ABS_TOLERANCE = 1e-9
REL_TOLERANCE = 1e-12


# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def pass_fail(condition):

    return (
        "PASS"
        if bool(condition)
        else "FAIL"
    )


def find_column(
    dataframe,
    candidates,
    description
):

    for candidate in candidates:

        if candidate in dataframe.columns:

            print(
                f"{description}: {candidate}"
            )

            return candidate


    raise KeyError(
        f"\nCould not identify {description}.\n"
        f"Tried the following names:\n"
        +
        "\n".join(candidates)
    )


def numeric_series(
    dataframe,
    column
):

    return (
        pd.to_numeric(
            dataframe[column],
            errors="coerce"
        )
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


def numerical_comparison(
    stored,
    reconstructed
):

    """
    Compare two numeric Series using np.isclose().

    Returns a dictionary containing:
      - number compared
      - exact matches
      - numerical mismatches
      - maximum absolute difference
      - mean absolute difference
      - pass/fail
    """

    comparison = pd.DataFrame(
        {
            "Stored": stored,
            "Reconstructed": reconstructed
        }
    ).dropna()


    if comparison.empty:

        raise ValueError(
            "No non-missing observations are available "
            "for numerical comparison."
        )


    comparison[
        "Absolute_Difference"
    ] = (

        comparison["Stored"]
        -
        comparison["Reconstructed"]

    ).abs()


    comparison[
        "Exact_Match"
    ] = (

        comparison["Stored"]
        ==
        comparison["Reconstructed"]

    )


    comparison[
        "Numerically_Equivalent"
    ] = np.isclose(

        comparison["Stored"],

        comparison["Reconstructed"],

        rtol=REL_TOLERANCE,

        atol=ABS_TOLERANCE,

        equal_nan=False

    )


    n_compared = len(
        comparison
    )


    exact_matches = int(
        comparison[
            "Exact_Match"
        ].sum()
    )


    mismatch_count = int(

        (
            ~comparison[
                "Numerically_Equivalent"
            ]
        )
        .sum()

    )


    max_absolute_difference = (
        comparison[
            "Absolute_Difference"
        ].max()
    )


    mean_absolute_difference = (
        comparison[
            "Absolute_Difference"
        ].mean()
    )


    passed = (
        mismatch_count == 0
    )


    return {

        "comparison":
            comparison,

        "n_compared":
            n_compared,

        "exact_matches":
            exact_matches,

        "mismatch_count":
            mismatch_count,

        "max_absolute_difference":
            max_absolute_difference,

        "mean_absolute_difference":
            mean_absolute_difference,

        "passed":
            passed

    }


# =====================================================================
# 5. START
# =====================================================================

section(
    "TRADING-VOLUME TRANSFORMATION AND TIMING VERIFICATION"
)


print(
    "\nRequired transformation:"
)

print(
    "Log_Volume_t = ln(1 + Volume_t)"
)


print(
    "\nPython implementation:"
)

print(
    "numpy.log1p(Volume)"
)


print(
    "\nRequired forecasting timing:"
)

print(
    "Lagged_Log_Volume_t = Log_Volume_(t-1)"
)


print(
    "\nSource:"
)

print(
    "Yahoo Finance via yfinance"
)


print(
    "\nNumerical absolute tolerance:"
)

print(
    ABS_TOLERANCE
)


print(
    "\nNumerical relative tolerance:"
)

print(
    REL_TOLERANCE
)


# =====================================================================
# 6. CHECK INPUT FILES
# =====================================================================

section(
    "CHECKING INPUT FILES"
)


INPUT_FILES = {

    "BTC processed volume":
        BTC_VOLUME_FILE,

    "ETH processed volume":
        ETH_VOLUME_FILE,

    "Final forecast dataset":
        FINAL_DATASET_FILE

}


for description, filepath in INPUT_FILES.items():

    print(
        f"\n{description}:"
    )

    print(
        filepath
    )

    print(
        "Exists:",
        filepath.exists()
    )


    if not filepath.exists():

        raise FileNotFoundError(
            f"\nRequired file not found:\n{filepath}"
        )


# =====================================================================
# 7. LOAD DATA
# =====================================================================

section(
    "LOADING DATA"
)


btc = pd.read_csv(
    BTC_VOLUME_FILE
)

eth = pd.read_csv(
    ETH_VOLUME_FILE
)

final_df = pd.read_csv(
    FINAL_DATASET_FILE
)


print(
    "\nBTC processed volume shape:"
)

print(
    btc.shape
)


print(
    "\nETH processed volume shape:"
)

print(
    eth.shape
)


print(
    "\nFinal forecast dataset shape:"
)

print(
    final_df.shape
)


print(
    "\nBTC processed volume columns:"
)

for column in btc.columns:

    print(
        " -",
        column
    )


print(
    "\nETH processed volume columns:"
)

for column in eth.columns:

    print(
        " -",
        column
    )


# =====================================================================
# 8. IDENTIFY PROCESSED-FILE COLUMNS
# =====================================================================

section(
    "IDENTIFYING VOLUME COLUMNS"
)


btc_date_column = find_column(

    btc,

    [
        "Date",
        "date",
        "Datetime",
        "datetime"
    ],

    "BTC date column"

)


eth_date_column = find_column(

    eth,

    [
        "Date",
        "date",
        "Datetime",
        "datetime"
    ],

    "ETH date column"

)


btc_raw_volume_column = find_column(

    btc,

    [
        "BTC_Volume",
        "Volume",
        "volume"
    ],

    "BTC raw volume column"

)


eth_raw_volume_column = find_column(

    eth,

    [
        "ETH_Volume",
        "Volume",
        "volume"
    ],

    "ETH raw volume column"

)


btc_log_volume_column = find_column(

    btc,

    [
        "Log_BTC_Volume",
        "BTC_Log_Volume",
        "Log_Volume"
    ],

    "BTC logged-volume column"

)


eth_log_volume_column = find_column(

    eth,

    [
        "Log_ETH_Volume",
        "ETH_Log_Volume",
        "Log_Volume"
    ],

    "ETH logged-volume column"

)


btc_lagged_volume_column = find_column(

    btc,

    [
        "Lagged_Log_BTC_Volume",
        "BTC_Lagged_Log_Volume",
        "Lagged_Log_Volume"
    ],

    "BTC lagged logged-volume column"

)


eth_lagged_volume_column = find_column(

    eth,

    [
        "Lagged_Log_ETH_Volume",
        "ETH_Lagged_Log_Volume",
        "Lagged_Log_Volume"
    ],

    "ETH lagged logged-volume column"

)


# =====================================================================
# 9. PREPARE PROCESSED FILES
# =====================================================================

section(
    "PREPARING PROCESSED VOLUME FILES"
)


def prepare_volume_file(
    dataframe,
    asset,
    date_column,
    raw_column,
    log_column,
    lag_column
):

    data = dataframe.copy()


    data["Date"] = pd.to_datetime(
        data[date_column],
        errors="coerce"
    )


    invalid_dates = int(
        data["Date"]
        .isna()
        .sum()
    )


    duplicate_dates = int(
        data["Date"]
        .duplicated()
        .sum()
    )


    print(
        f"\n{asset} invalid dates:"
    )

    print(
        invalid_dates
    )


    print(
        f"{asset} duplicate dates:"
    )

    print(
        duplicate_dates
    )


    if invalid_dates > 0:

        raise ValueError(
            f"{asset}: invalid dates found."
        )


    if duplicate_dates > 0:

        raise ValueError(
            f"{asset}: duplicate dates found."
        )


    data = (

        data

        .sort_values(
            "Date"
        )

        .reset_index(
            drop=True
        )

    )


    data["Raw_Volume"] = numeric_series(
        data,
        raw_column
    )


    data["Stored_Log_Volume"] = numeric_series(
        data,
        log_column
    )


    data["Stored_Lagged_Log_Volume"] = numeric_series(
        data,
        lag_column
    )


    return data


btc = prepare_volume_file(

    btc,
    "BTC",
    btc_date_column,
    btc_raw_volume_column,
    btc_log_volume_column,
    btc_lagged_volume_column

)


eth = prepare_volume_file(

    eth,
    "ETH",
    eth_date_column,
    eth_raw_volume_column,
    eth_log_volume_column,
    eth_lagged_volume_column

)


# =====================================================================
# 10. CALENDAR DIAGNOSTICS
# =====================================================================

section(
    "DATE AND CALENDAR DIAGNOSTICS"
)


def calendar_diagnostics(
    data,
    asset
):

    date_difference = (
        data["Date"]
        .diff()
    )


    gap_mask = (

        date_difference.notna()

        &

        (
            date_difference
            != pd.Timedelta(days=1)
        )

    )


    calendar_gaps = int(
        gap_mask.sum()
    )


    weekend_mask = (
        data["Date"]
        .dt.dayofweek >= 5
    )


    weekend_observations = int(
        weekend_mask.sum()
    )


    print(
        f"\n{asset} observations:"
    )

    print(
        len(data)
    )


    print(
        f"{asset} date range:"
    )

    print(
        data["Date"].min(),
        "to",
        data["Date"].max()
    )


    print(
        f"{asset} non-consecutive calendar gaps:"
    )

    print(
        calendar_gaps
    )


    print(
        f"{asset} weekend observations:"
    )

    print(
        weekend_observations
    )


    if calendar_gaps > 0:

        gap_table = pd.DataFrame(

            {

                "Previous_Date":
                    data["Date"]
                    .shift(1),

                "Current_Date":
                    data["Date"],

                "Difference":
                    date_difference

            }

        )


        gap_table = gap_table.loc[
            gap_mask
        ]


        print(
            f"\n{asset} calendar gaps:"
        )

        print(
            gap_table.to_string(
                index=False
            )
        )


    return {

        "N":
            len(data),

        "Start_Date":
            data["Date"].min(),

        "End_Date":
            data["Date"].max(),

        "Calendar_Gaps":
            calendar_gaps,

        "Weekend_Observations":
            weekend_observations

    }


btc_calendar = calendar_diagnostics(
    btc,
    "BTC"
)


eth_calendar = calendar_diagnostics(
    eth,
    "ETH"
)


if btc_calendar[
    "Calendar_Gaps"
] != 0:

    raise ValueError(
        "\nBTC volume file does not follow a "
        "continuous daily calendar."
    )


if eth_calendar[
    "Calendar_Gaps"
] != 0:

    raise ValueError(
        "\nETH volume file does not follow a "
        "continuous daily calendar."
    )


print(
    "\nPASS: Both processed volume files follow "
    "a consecutive daily crypto calendar."
)


# =====================================================================
# 11. RAW-VOLUME DIAGNOSTICS
# =====================================================================

section(
    "VERIFYING RAW VOLUME"
)


def raw_volume_diagnostics(
    data,
    asset
):

    volume = (
        data[
            "Raw_Volume"
        ]
    )


    missing = int(
        volume
        .isna()
        .sum()
    )


    negative = int(
        (
            volume < 0
        )
        .sum()
    )


    zero = int(
        (
            volume == 0
        )
        .sum()
    )


    positive = int(
        (
            volume > 0
        )
        .sum()
    )


    print(
        f"\n{asset} raw-volume missing values:"
    )

    print(
        missing
    )


    print(
        f"{asset} negative raw-volume observations:"
    )

    print(
        negative
    )


    print(
        f"{asset} zero-volume observations:"
    )

    print(
        zero
    )


    print(
        f"{asset} positive-volume observations:"
    )

    print(
        positive
    )


    print(
        f"{asset} minimum raw volume:"
    )

    print(
        volume.min()
    )


    print(
        f"{asset} maximum raw volume:"
    )

    print(
        volume.max()
    )


    if missing > 0:

        raise ValueError(
            f"{asset}: missing raw-volume observations found."
        )


    if negative > 0:

        raise ValueError(
            f"{asset}: negative raw-volume observations found."
        )


    return {

        "Missing_Raw_Volume":
            missing,

        "Negative_Raw_Volume":
            negative,

        "Zero_Raw_Volume":
            zero,

        "Positive_Raw_Volume":
            positive,

        "Minimum_Raw_Volume":
            volume.min(),

        "Maximum_Raw_Volume":
            volume.max()

    }


btc_raw = raw_volume_diagnostics(
    btc,
    "BTC"
)


eth_raw = raw_volume_diagnostics(
    eth,
    "ETH"
)


# =====================================================================
# 12. INDEPENDENT RECONSTRUCTION OF log(1 + Volume)
# =====================================================================

section(
    "INDEPENDENTLY RECONSTRUCTING log(1 + Volume)"
)


btc[
    "Reconstructed_Log_Volume"
] = np.log1p(
    btc[
        "Raw_Volume"
    ]
)


eth[
    "Reconstructed_Log_Volume"
] = np.log1p(
    eth[
        "Raw_Volume"
    ]
)


print(
    "\nTransformation:"
)

print(
    "numpy.log1p(Volume)"
)


print(
    "\nMathematically:"
)

print(
    "ln(1 + Volume)"
)


# =====================================================================
# 13. VERIFY STORED TRANSFORMATION
# =====================================================================

section(
    "VERIFYING STORED LOG-VOLUME TRANSFORMATION"
)


def verify_transformation(
    data,
    asset
):

    result = numerical_comparison(

        data[
            "Stored_Log_Volume"
        ],

        data[
            "Reconstructed_Log_Volume"
        ]

    )


    print(
        f"\n{asset} observations compared:"
    )

    print(
        result[
            "n_compared"
        ]
    )


    print(
        f"{asset} exact floating-point matches:"
    )

    print(
        result[
            "exact_matches"
        ]
    )


    print(
        f"{asset} maximum absolute difference:"
    )

    print(
        result[
            "max_absolute_difference"
        ]
    )


    print(
        f"{asset} mean absolute difference:"
    )

    print(
        result[
            "mean_absolute_difference"
        ]
    )


    print(
        f"{asset} absolute tolerance:"
    )

    print(
        ABS_TOLERANCE
    )


    print(
        f"{asset} relative tolerance:"
    )

    print(
        REL_TOLERANCE
    )


    print(
        f"{asset} mismatches outside numerical tolerance:"
    )

    print(
        result[
            "mismatch_count"
        ]
    )


    print(
        f"{asset} log(1 + Volume) verification:"
    )

    print(
        pass_fail(
            result[
                "passed"
            ]
        )
    )


    if not result[
        "passed"
    ]:

        raise ValueError(
            f"\n{asset} logged volume differs materially "
            f"from independently reconstructed "
            f"log(1 + Volume)."
        )


    return result


btc_transformation = verify_transformation(
    btc,
    "BTC"
)


eth_transformation = verify_transformation(
    eth,
    "ETH"
)


# =====================================================================
# 14. SKEWNESS DIAGNOSTICS
# =====================================================================

section(
    "RAW VS TRANSFORMED VOLUME SKEWNESS"
)


def skewness_diagnostics(
    data,
    asset
):

    raw_skewness = (
        data[
            "Raw_Volume"
        ]
        .dropna()
        .skew()
    )


    log_skewness = (
        data[
            "Reconstructed_Log_Volume"
        ]
        .dropna()
        .skew()
    )


    reduction_in_absolute_skewness = (

        abs(
            raw_skewness
        )

        -

        abs(
            log_skewness
        )

    )


    skewness_reduced = (

        abs(
            log_skewness
        )

        <

        abs(
            raw_skewness
        )

    )


    print(
        f"\n{asset} raw-volume skewness:"
    )

    print(
        raw_skewness
    )


    print(
        f"{asset} log-volume skewness:"
    )

    print(
        log_skewness
    )


    print(
        f"{asset} reduction in absolute skewness:"
    )

    print(
        reduction_in_absolute_skewness
    )


    print(
        f"{asset} absolute skewness reduced:"
    )

    print(
        skewness_reduced
    )


    return {

        "Raw_Volume_Skewness":
            raw_skewness,

        "Log_Volume_Skewness":
            log_skewness,

        "Reduction_in_Absolute_Skewness":
            reduction_in_absolute_skewness,

        "Absolute_Skewness_Reduced":
            skewness_reduced

    }


btc_skewness = skewness_diagnostics(
    btc,
    "BTC"
)


eth_skewness = skewness_diagnostics(
    eth,
    "ETH"
)


# =====================================================================
# 15. RECONSTRUCT t-1 CALENDAR-DAY VOLUME LAG
# =====================================================================

section(
    "RECONSTRUCTING ONE-CALENDAR-DAY LOG-VOLUME LAG"
)


# Calendar continuity has already been verified above.
#
# Therefore shift(1) is valid here as the previous calendar day's
# cryptocurrency volume.


btc[
    "Reconstructed_Lagged_Log_Volume"
] = (

    btc[
        "Reconstructed_Log_Volume"
    ]

    .shift(1)

)


eth[
    "Reconstructed_Lagged_Log_Volume"
] = (

    eth[
        "Reconstructed_Log_Volume"
    ]

    .shift(1)

)


print(
    "\nPASS: Daily crypto calendar was verified before shift(1)."
)


print(
    "Therefore shift(1) represents the previous calendar day."
)


# =====================================================================
# 16. VERIFY STORED t-1 VOLUME LAG
# =====================================================================

section(
    "VERIFYING STORED FORECASTING VOLUME LAG"
)


def verify_lag(
    data,
    asset
):

    result = numerical_comparison(

        data[
            "Stored_Lagged_Log_Volume"
        ],

        data[
            "Reconstructed_Lagged_Log_Volume"
        ]

    )


    print(
        f"\n{asset} lag observations compared:"
    )

    print(
        result[
            "n_compared"
        ]
    )


    print(
        f"{asset} exact floating-point lag matches:"
    )

    print(
        result[
            "exact_matches"
        ]
    )


    print(
        f"{asset} maximum absolute lag difference:"
    )

    print(
        result[
            "max_absolute_difference"
        ]
    )


    print(
        f"{asset} mean absolute lag difference:"
    )

    print(
        result[
            "mean_absolute_difference"
        ]
    )


    print(
        f"{asset} lag mismatches outside numerical tolerance:"
    )

    print(
        result[
            "mismatch_count"
        ]
    )


    print(
        f"{asset} t-1 volume timing verification:"
    )

    print(
        pass_fail(
            result[
                "passed"
            ]
        )
    )


    if not result[
        "passed"
    ]:

        raise ValueError(
            f"\n{asset} stored lagged log volume differs "
            f"materially from the independently reconstructed "
            f"previous-calendar-day value."
        )


    return result


btc_lag = verify_lag(
    btc,
    "BTC"
)


eth_lag = verify_lag(
    eth,
    "ETH"
)


# =====================================================================
# 17. DATE-BASED t-1 VERIFICATION
# =====================================================================
#
# This check does not rely solely on shift(1).
#
# For every target date t, it explicitly searches for:
#
#       t - 1 calendar day
#
# and verifies that the stored lag corresponds to that date's
# transformed volume.
# =====================================================================

section(
    "DATE-BASED ONE-CALENDAR-DAY LAG VERIFICATION"
)


def date_based_lag_verification(
    data,
    asset
):

    target = data[
        [
            "Date",
            "Stored_Lagged_Log_Volume"
        ]
    ].copy()


    target[
        "Expected_Source_Date"
    ] = (

        target[
            "Date"
        ]

        -

        pd.Timedelta(
            days=1
        )

    )


    source = data[
        [
            "Date",
            "Reconstructed_Log_Volume"
        ]
    ].copy()


    source = source.rename(

        columns={

            "Date":
                "Expected_Source_Date",

            "Reconstructed_Log_Volume":
                "Expected_Previous_Day_Log_Volume"

        }

    )


    check = target.merge(

        source,

        on="Expected_Source_Date",

        how="left",

        validate="one_to_one"

    )


    result = numerical_comparison(

        check[
            "Stored_Lagged_Log_Volume"
        ],

        check[
            "Expected_Previous_Day_Log_Volume"
        ]

    )


    print(
        f"\n{asset} date-based observations compared:"
    )

    print(
        result[
            "n_compared"
        ]
    )


    print(
        f"{asset} maximum date-based difference:"
    )

    print(
        result[
            "max_absolute_difference"
        ]
    )


    print(
        f"{asset} date-based mismatches:"
    )

    print(
        result[
            "mismatch_count"
        ]
    )


    print(
        f"{asset} explicit t-1 date verification:"
    )

    print(
        pass_fail(
            result[
                "passed"
            ]
        )
    )


    if not result[
        "passed"
    ]:

        raise ValueError(
            f"{asset}: explicit date-based "
            f"t-1 verification failed."
        )


    check[
        "Absolute_Difference"
    ] = (

        check[
            "Stored_Lagged_Log_Volume"
        ]

        -

        check[
            "Expected_Previous_Day_Log_Volume"
        ]

    ).abs()


    return (
        result,
        check
    )


(
    btc_date_result,
    btc_date_check
) = date_based_lag_verification(
    btc,
    "BTC"
)


(
    eth_date_result,
    eth_date_check
) = date_based_lag_verification(
    eth,
    "ETH"
)


# =====================================================================
# 18. WEEKEND VERIFICATION IN PROCESSED FILES
# =====================================================================

section(
    "PROCESSED-FILE WEEKEND TIMING VERIFICATION"
)


def weekend_lag_verification(
    data,
    asset
):

    weekend_mask = (
        data["Date"]
        .dt.dayofweek >= 5
    )


    weekend_data = (
        data.loc[
            weekend_mask
        ]
        .copy()
    )


    result = numerical_comparison(

        weekend_data[
            "Stored_Lagged_Log_Volume"
        ],

        weekend_data[
            "Reconstructed_Lagged_Log_Volume"
        ]

    )


    print(
        f"\n{asset} weekend observations:"
    )

    print(
        len(
            weekend_data
        )
    )


    print(
        f"{asset} weekend observations compared:"
    )

    print(
        result[
            "n_compared"
        ]
    )


    print(
        f"{asset} weekend lag mismatches:"
    )

    print(
        result[
            "mismatch_count"
        ]
    )


    print(
        f"{asset} weekend timing verification:"
    )

    print(
        pass_fail(
            result[
                "passed"
            ]
        )
    )


    if not result[
        "passed"
    ]:

        raise ValueError(
            f"{asset}: weekend volume timing verification failed."
        )


    return result


btc_weekend = weekend_lag_verification(
    btc,
    "BTC"
)


eth_weekend = weekend_lag_verification(
    eth,
    "ETH"
)


# =====================================================================
# 19. PREPARE FINAL FORECAST DATASET
# =====================================================================

section(
    "PREPARING FINAL FORECAST DATASET"
)


if "Date" not in final_df.columns:

    raise KeyError(
        "Date column is missing from final_forecast_dataset.csv."
    )


final_df[
    "Date"
] = pd.to_datetime(

    final_df[
        "Date"
    ],

    errors="coerce"

)


invalid_final_dates = int(
    final_df[
        "Date"
    ]
    .isna()
    .sum()
)


duplicate_final_dates = int(
    final_df[
        "Date"
    ]
    .duplicated()
    .sum()
)


print(
    "\nInvalid final-dataset dates:"
)

print(
    invalid_final_dates
)


print(
    "\nDuplicate final-dataset dates:"
)

print(
    duplicate_final_dates
)


if invalid_final_dates > 0:

    raise ValueError(
        "Invalid dates found in final_forecast_dataset.csv."
    )


if duplicate_final_dates > 0:

    raise ValueError(
        "Duplicate dates found in final_forecast_dataset.csv."
    )


final_df = (

    final_df

    .sort_values(
        "Date"
    )

    .reset_index(
        drop=True
    )

)


FINAL_REQUIRED_VOLUME_COLUMNS = [

    "Log_BTC_Volume",
    "Lagged_Log_BTC_Volume",

    "Log_ETH_Volume",
    "Lagged_Log_ETH_Volume"

]


for variable in FINAL_REQUIRED_VOLUME_COLUMNS:

    if variable not in final_df.columns:

        raise KeyError(
            f"{variable} is missing from "
            f"final_forecast_dataset.csv."
        )


    final_df[
        variable
    ] = numeric_series(
        final_df,
        variable
    )


print(
    "\nPASS: All required volume variables are present "
    "in final_forecast_dataset.csv."
)


# =====================================================================
# 20. FINAL DATASET CALENDAR CHECK
# =====================================================================

section(
    "FINAL DATASET CALENDAR CHECK"
)


final_date_difference = (
    final_df[
        "Date"
    ]
    .diff()
)


final_gap_mask = (

    final_date_difference.notna()

    &

    (
        final_date_difference
        != pd.Timedelta(days=1)
    )

)


final_calendar_gaps = int(
    final_gap_mask.sum()
)


final_weekends = int(

    (
        final_df[
            "Date"
        ]
        .dt.dayofweek >= 5
    )
    .sum()

)


print(
    "\nFinal dataset observations:"
)

print(
    len(
        final_df
    )
)


print(
    "\nFinal dataset date range:"
)

print(
    final_df[
        "Date"
    ].min(),
    "to",
    final_df[
        "Date"
    ].max()
)


print(
    "\nFinal dataset calendar gaps:"
)

print(
    final_calendar_gaps
)


print(
    "\nFinal dataset weekend observations:"
)

print(
    final_weekends
)


if final_calendar_gaps > 0:

    raise ValueError(
        "Final forecast dataset does not follow "
        "a consecutive daily calendar."
    )


# =====================================================================
# 21. PROCESSED FILE VS FINAL DATASET
# =====================================================================

section(
    "PROCESSED FILE VS FINAL DATASET VERIFICATION"
)


btc_final_comparison = final_df[
    [
        "Date",
        "Log_BTC_Volume",
        "Lagged_Log_BTC_Volume"
    ]
].merge(

    btc[
        [
            "Date",
            "Stored_Log_Volume",
            "Stored_Lagged_Log_Volume"
        ]
    ],

    on="Date",

    how="inner",

    validate="one_to_one"

)


eth_final_comparison = final_df[
    [
        "Date",
        "Log_ETH_Volume",
        "Lagged_Log_ETH_Volume"
    ]
].merge(

    eth[
        [
            "Date",
            "Stored_Log_Volume",
            "Stored_Lagged_Log_Volume"
        ]
    ],

    on="Date",

    how="inner",

    validate="one_to_one"

)


def verify_processed_vs_final(
    comparison,
    asset,
    final_log_column,
    final_lag_column
):

    log_result = numerical_comparison(

        comparison[
            final_log_column
        ],

        comparison[
            "Stored_Log_Volume"
        ]

    )


    lag_result = numerical_comparison(

        comparison[
            final_lag_column
        ],

        comparison[
            "Stored_Lagged_Log_Volume"
        ]

    )


    print(
        f"\n{asset} matched dates:"
    )

    print(
        len(
            comparison
        )
    )


    print(
        f"{asset} final logged-volume mismatches:"
    )

    print(
        log_result[
            "mismatch_count"
        ]
    )


    print(
        f"{asset} final lagged-volume mismatches:"
    )

    print(
        lag_result[
            "mismatch_count"
        ]
    )


    overall_pass = (

        log_result[
            "passed"
        ]

        and

        lag_result[
            "passed"
        ]

    )


    print(
        f"{asset} processed-to-final verification:"
    )

    print(
        pass_fail(
            overall_pass
        )
    )


    if not overall_pass:

        raise ValueError(
            f"{asset}: processed volume file and final "
            f"forecast dataset are materially inconsistent."
        )


    return {

        "log_result":
            log_result,

        "lag_result":
            lag_result,

        "passed":
            overall_pass

    }


btc_final_consistency = verify_processed_vs_final(

    btc_final_comparison,

    "BTC",

    "Log_BTC_Volume",

    "Lagged_Log_BTC_Volume"

)


eth_final_consistency = verify_processed_vs_final(

    eth_final_comparison,

    "ETH",

    "Log_ETH_Volume",

    "Lagged_Log_ETH_Volume"

)


# =====================================================================
# 22. DIRECT FINAL-DATASET FORECAST-TIMING CHECK
# =====================================================================

section(
    "DIRECT FINAL-DATASET FORECAST-TIMING CHECK"
)


final_df[
    "Expected_Lagged_Log_BTC_Volume"
] = (

    final_df[
        "Log_BTC_Volume"
    ]

    .shift(1)

)


final_df[
    "Expected_Lagged_Log_ETH_Volume"
] = (

    final_df[
        "Log_ETH_Volume"
    ]

    .shift(1)

)


btc_final_timing = numerical_comparison(

    final_df[
        "Lagged_Log_BTC_Volume"
    ],

    final_df[
        "Expected_Lagged_Log_BTC_Volume"
    ]

)


eth_final_timing = numerical_comparison(

    final_df[
        "Lagged_Log_ETH_Volume"
    ],

    final_df[
        "Expected_Lagged_Log_ETH_Volume"
    ]

)


print(
    "\nBTC final timing observations compared:"
)

print(
    btc_final_timing[
        "n_compared"
    ]
)


print(
    "BTC final timing maximum difference:"
)

print(
    btc_final_timing[
        "max_absolute_difference"
    ]
)


print(
    "BTC final timing mismatches:"
)

print(
    btc_final_timing[
        "mismatch_count"
    ]
)


print(
    "BTC final forecasting timing:"
)

print(
    pass_fail(
        btc_final_timing[
            "passed"
        ]
    )
)


print(
    "\nETH final timing observations compared:"
)

print(
    eth_final_timing[
        "n_compared"
    ]
)


print(
    "ETH final timing maximum difference:"
)

print(
    eth_final_timing[
        "max_absolute_difference"
    ]
)


print(
    "ETH final timing mismatches:"
)

print(
    eth_final_timing[
        "mismatch_count"
    ]
)


print(
    "ETH final forecasting timing:"
)

print(
    pass_fail(
        eth_final_timing[
            "passed"
        ]
    )
)


if not btc_final_timing[
    "passed"
]:

    raise ValueError(
        "BTC final forecasting volume timing failed."
    )


if not eth_final_timing[
    "passed"
]:

    raise ValueError(
        "ETH final forecasting volume timing failed."
    )


# =====================================================================
# 23. FINAL-DATASET WEEKEND TIMING CHECK
# =====================================================================

section(
    "FINAL-DATASET WEEKEND TIMING CHECK"
)


final_df[
    "Is_Weekend"
] = (

    final_df[
        "Date"
    ]
    .dt.dayofweek >= 5

)


weekend_final = final_df.loc[
    final_df[
        "Is_Weekend"
    ]
].copy()


btc_final_weekend = numerical_comparison(

    weekend_final[
        "Lagged_Log_BTC_Volume"
    ],

    weekend_final[
        "Expected_Lagged_Log_BTC_Volume"
    ]

)


eth_final_weekend = numerical_comparison(

    weekend_final[
        "Lagged_Log_ETH_Volume"
    ],

    weekend_final[
        "Expected_Lagged_Log_ETH_Volume"
    ]

)


print(
    "\nWeekend observations in final dataset:"
)

print(
    len(
        weekend_final
    )
)


print(
    "\nBTC weekend timing mismatches:"
)

print(
    btc_final_weekend[
        "mismatch_count"
    ]
)


print(
    "BTC weekend timing:"
)

print(
    pass_fail(
        btc_final_weekend[
            "passed"
        ]
    )
)


print(
    "\nETH weekend timing mismatches:"
)

print(
    eth_final_weekend[
        "mismatch_count"
    ]
)


print(
    "ETH weekend timing:"
)

print(
    pass_fail(
        eth_final_weekend[
            "passed"
        ]
    )
)


if not btc_final_weekend[
    "passed"
]:

    raise ValueError(
        "BTC final-dataset weekend timing failed."
    )


if not eth_final_weekend[
    "passed"
]:

    raise ValueError(
        "ETH final-dataset weekend timing failed."
    )


# =====================================================================
# 24. DESCRIPTIVE STATISTICS
# =====================================================================

section(
    "TRADING-VOLUME DESCRIPTIVE STATISTICS"
)


def create_descriptive_rows(
    data,
    asset
):

    rows = []


    variables = {

        "Raw_Volume":
            data[
                "Raw_Volume"
            ],

        "Log1p_Volume":
            data[
                "Reconstructed_Log_Volume"
            ],

        "Lagged_Log1p_Volume":
            data[
                "Reconstructed_Lagged_Log_Volume"
            ]

    }


    for variable_name, series in variables.items():

        clean = (
            series
            .dropna()
        )


        rows.append(

            {

                "Asset":
                    asset,

                "Variable":
                    variable_name,

                "N":
                    int(
                        clean.count()
                    ),

                "Mean":
                    clean.mean(),

                "Std_Dev":
                    clean.std(),

                "Min":
                    clean.min(),

                "Q1":
                    clean.quantile(
                        0.25
                    ),

                "Median":
                    clean.median(),

                "Q3":
                    clean.quantile(
                        0.75
                    ),

                "Max":
                    clean.max(),

                "Skewness":
                    clean.skew()

            }

        )


    return rows


descriptive_statistics = pd.DataFrame(

    create_descriptive_rows(
        btc,
        "BTC"
    )

    +

    create_descriptive_rows(
        eth,
        "ETH"
    )

)


print(
    "\n",
    descriptive_statistics
    .to_string(
        index=False
    )
)


# =====================================================================
# 25. MASTER VERIFICATION SUMMARY
# =====================================================================

section(
    "MASTER VERIFICATION SUMMARY"
)


verification_summary = pd.DataFrame(

    [

        {

            "Asset":
                "BTC",

            "Source":
                "Yahoo Finance via yfinance",

            "Source_Field":
                "Volume",

            "Source_Field_Interpretation":
                (
                    "Daily trading-volume field reported for "
                    "the downloaded Yahoo Finance "
                    "cryptocurrency series"
                ),

            "Transformation":
                "ln(1 + Volume)",

            "Python_Implementation":
                "numpy.log1p(Volume)",

            "Forecasting_Variable":
                "Lagged_Log_BTC_Volume",

            "Forecasting_Timing":
                "Previous calendar day (t-1)",

            "Observations":
                btc_calendar[
                    "N"
                ],

            "Start_Date":
                btc_calendar[
                    "Start_Date"
                ],

            "End_Date":
                btc_calendar[
                    "End_Date"
                ],

            "Calendar_Gaps":
                btc_calendar[
                    "Calendar_Gaps"
                ],

            "Weekend_Observations":
                btc_calendar[
                    "Weekend_Observations"
                ],

            "Missing_Raw_Volume":
                btc_raw[
                    "Missing_Raw_Volume"
                ],

            "Negative_Raw_Volume":
                btc_raw[
                    "Negative_Raw_Volume"
                ],

            "Zero_Raw_Volume":
                btc_raw[
                    "Zero_Raw_Volume"
                ],

            "Raw_Volume_Skewness":
                btc_skewness[
                    "Raw_Volume_Skewness"
                ],

            "Log_Volume_Skewness":
                btc_skewness[
                    "Log_Volume_Skewness"
                ],

            "Transformation_Max_Abs_Difference":
                btc_transformation[
                    "max_absolute_difference"
                ],

            "Transformation_Pass":
                btc_transformation[
                    "passed"
                ],

            "Processed_Lag_Max_Abs_Difference":
                btc_lag[
                    "max_absolute_difference"
                ],

            "Processed_Lag_Pass":
                btc_lag[
                    "passed"
                ],

            "Date_Based_Lag_Pass":
                btc_date_result[
                    "passed"
                ],

            "Processed_Weekend_Pass":
                btc_weekend[
                    "passed"
                ],

            "Processed_to_Final_Pass":
                btc_final_consistency[
                    "passed"
                ],

            "Final_Timing_Pass":
                btc_final_timing[
                    "passed"
                ],

            "Final_Weekend_Pass":
                btc_final_weekend[
                    "passed"
                ]

        },

        {

            "Asset":
                "ETH",

            "Source":
                "Yahoo Finance via yfinance",

            "Source_Field":
                "Volume",

            "Source_Field_Interpretation":
                (
                    "Daily trading-volume field reported for "
                    "the downloaded Yahoo Finance "
                    "cryptocurrency series"
                ),

            "Transformation":
                "ln(1 + Volume)",

            "Python_Implementation":
                "numpy.log1p(Volume)",

            "Forecasting_Variable":
                "Lagged_Log_ETH_Volume",

            "Forecasting_Timing":
                "Previous calendar day (t-1)",

            "Observations":
                eth_calendar[
                    "N"
                ],

            "Start_Date":
                eth_calendar[
                    "Start_Date"
                ],

            "End_Date":
                eth_calendar[
                    "End_Date"
                ],

            "Calendar_Gaps":
                eth_calendar[
                    "Calendar_Gaps"
                ],

            "Weekend_Observations":
                eth_calendar[
                    "Weekend_Observations"
                ],

            "Missing_Raw_Volume":
                eth_raw[
                    "Missing_Raw_Volume"
                ],

            "Negative_Raw_Volume":
                eth_raw[
                    "Negative_Raw_Volume"
                ],

            "Zero_Raw_Volume":
                eth_raw[
                    "Zero_Raw_Volume"
                ],

            "Raw_Volume_Skewness":
                eth_skewness[
                    "Raw_Volume_Skewness"
                ],

            "Log_Volume_Skewness":
                eth_skewness[
                    "Log_Volume_Skewness"
                ],

            "Transformation_Max_Abs_Difference":
                eth_transformation[
                    "max_absolute_difference"
                ],

            "Transformation_Pass":
                eth_transformation[
                    "passed"
                ],

            "Processed_Lag_Max_Abs_Difference":
                eth_lag[
                    "max_absolute_difference"
                ],

            "Processed_Lag_Pass":
                eth_lag[
                    "passed"
                ],

            "Date_Based_Lag_Pass":
                eth_date_result[
                    "passed"
                ],

            "Processed_Weekend_Pass":
                eth_weekend[
                    "passed"
                ],

            "Processed_to_Final_Pass":
                eth_final_consistency[
                    "passed"
                ],

            "Final_Timing_Pass":
                eth_final_timing[
                    "passed"
                ],

            "Final_Weekend_Pass":
                eth_final_weekend[
                    "passed"
                ]

        }

    ]

)


print(
    "\n",
    verification_summary
    .to_string(
        index=False
    )
)


# =====================================================================
# 26. SAVE OUTPUT FILES
# =====================================================================

section(
    "SAVING VERIFICATION OUTPUTS"
)


summary_file = (
    OUTPUT_DIR
    / "trading_volume_verification_summary.csv"
)


descriptive_file = (
    OUTPUT_DIR
    / "trading_volume_descriptive_statistics.csv"
)


btc_date_file = (
    OUTPUT_DIR
    / "btc_volume_date_lag_verification.csv"
)


eth_date_file = (
    OUTPUT_DIR
    / "eth_volume_date_lag_verification.csv"
)


verification_summary.to_csv(
    summary_file,
    index=False
)


descriptive_statistics.to_csv(
    descriptive_file,
    index=False
)


btc_date_check.to_csv(
    btc_date_file,
    index=False
)


eth_date_check.to_csv(
    eth_date_file,
    index=False
)


OUTPUT_FILES = [

    summary_file,
    descriptive_file,
    btc_date_file,
    eth_date_file

]


for filepath in OUTPUT_FILES:

    print(
        "\nSaved:"
    )

    print(
        filepath
    )


# =====================================================================
# 27. SAVE DISSERTATION METHODOLOGY NOTE
# =====================================================================

section(
    "SAVING METHODOLOGY DOCUMENTATION"
)


methodology_file = (
    OUTPUT_DIR
    / "trading_volume_methodology_note.txt"
)


methodology_note = """
TRADING-VOLUME TRANSFORMATION AND TIMING

Data source
-----------
Bitcoin and Ethereum trading-volume data were obtained from Yahoo
Finance using the yfinance Python package.

The variable corresponds to the daily Volume field reported for the
downloaded Yahoo Finance cryptocurrency series. Accordingly, the
measure is described as Yahoo Finance reported daily trading volume
for the respective cryptocurrency series rather than total global
cryptocurrency-market trading volume.

Transformation
--------------
Trading volume is non-negative and can exhibit substantial positive
skewness. To reduce the influence of extreme observations, the raw
volume measure is transformed as:

    Log_Volume_t = ln(1 + Volume_t)

The transformation is implemented using numpy.log1p(), which computes
ln(1 + x).

The addition of one also ensures that the logarithmic transformation
would remain defined if a zero-volume observation were present.

Timing
------
For predictive specifications, contemporaneous trading volume is not
used. Instead, the transformed volume measure is lagged by one
calendar day:

    Lagged_Log_Volume_t = Log_Volume_(t-1)

Thus, cryptocurrency return on calendar date t is modelled using
volume information from the preceding calendar date.

Because cryptocurrency markets operate continuously, including
weekends, the relevant lag is defined in calendar days rather than
traditional-market trading days.

Verification
------------
The stored transformed variables were independently reconstructed
from the raw Yahoo Finance Volume fields using numpy.log1p().

The stored lagged variables were independently reconstructed using
the preceding calendar day's transformed volume.

Date-level checks were additionally performed to verify explicitly
that each lagged value corresponds to date t-1.

The processed BTC and ETH volume files were then compared with the
volume variables contained in the final forecasting dataset.

Numerical comparison
--------------------
Because values stored in CSV files can exhibit extremely small
floating-point representation differences after serialization and
reloading, numerical equivalence was assessed using both absolute and
relative tolerances rather than requiring exact binary equality.

The verification uses:

    absolute tolerance = 1e-9
    relative tolerance = 1e-12

Differences within these tolerances are treated as numerical
equivalence and have no economically meaningful effect on the
variables or regression analysis.
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
# 28. FINAL OVERALL PASS / FAIL
# =====================================================================

section(
    "FINAL PASS / FAIL"
)


ALL_CHECKS = {

    "BTC raw volume has no missing values":
        (
            btc_raw[
                "Missing_Raw_Volume"
            ]
            == 0
        ),

    "ETH raw volume has no missing values":
        (
            eth_raw[
                "Missing_Raw_Volume"
            ]
            == 0
        ),

    "BTC raw volume has no negative values":
        (
            btc_raw[
                "Negative_Raw_Volume"
            ]
            == 0
        ),

    "ETH raw volume has no negative values":
        (
            eth_raw[
                "Negative_Raw_Volume"
            ]
            == 0
        ),

    "BTC calendar is continuous":
        (
            btc_calendar[
                "Calendar_Gaps"
            ]
            == 0
        ),

    "ETH calendar is continuous":
        (
            eth_calendar[
                "Calendar_Gaps"
            ]
            == 0
        ),

    "BTC transformation is ln(1 + Volume)":
        btc_transformation[
            "passed"
        ],

    "ETH transformation is ln(1 + Volume)":
        eth_transformation[
            "passed"
        ],

    "BTC stored lag is t-1":
        btc_lag[
            "passed"
        ],

    "ETH stored lag is t-1":
        eth_lag[
            "passed"
        ],

    "BTC explicit date-based lag is t-1":
        btc_date_result[
            "passed"
        ],

    "ETH explicit date-based lag is t-1":
        eth_date_result[
            "passed"
        ],

    "BTC weekend timing is correct":
        btc_weekend[
            "passed"
        ],

    "ETH weekend timing is correct":
        eth_weekend[
            "passed"
        ],

    "BTC processed file matches final dataset":
        btc_final_consistency[
            "passed"
        ],

    "ETH processed file matches final dataset":
        eth_final_consistency[
            "passed"
        ],

    "BTC final forecasting timing is correct":
        btc_final_timing[
            "passed"
        ],

    "ETH final forecasting timing is correct":
        eth_final_timing[
            "passed"
        ],

    "BTC final weekend timing is correct":
        btc_final_weekend[
            "passed"
        ],

    "ETH final weekend timing is correct":
        eth_final_weekend[
            "passed"
        ]

}


for check_name, condition in ALL_CHECKS.items():

    print(
        f"\n{check_name}: "
        f"{pass_fail(condition)}"
    )


overall_pass = all(
    ALL_CHECKS.values()
)


print(
    "\n" + "-" * 80
)


print(
    "\nOVERALL TRADING-VOLUME VERIFICATION:"
)

print(
    pass_fail(
        overall_pass
    )
)


if not overall_pass:

    raise ValueError(
        "\nOne or more substantive trading-volume "
        "verification checks failed."
    )


# =====================================================================
# 29. FINAL INTERPRETATION
# =====================================================================

section(
    "FINAL INTERPRETATION"
)


print(
    "\nIf the overall verification above is PASS:"
)


print(
    "\n1. BTC and ETH raw trading-volume data are valid."
)


print(
    "\n2. Stored transformed volume is numerically equivalent to:"
)

print(
    "   ln(1 + Volume)"
)


print(
    "\n3. Stored forecasting volume is numerically equivalent to:"
)

print(
    "   previous-calendar-day transformed volume (t-1)"
)


print(
    "\n4. Weekend crypto-volume observations are retained "
    "and correctly lagged."
)


print(
    "\n5. Processed volume files agree with "
    "final_forecast_dataset.csv."
)


print(
    "\n6. No reconstruction of the processed volume variables "
    "is required."
)


print(
    "\n7. Tiny differences within the numerical tolerance "
    "represent floating-point / CSV precision rather than "
    "economically meaningful data discrepancies."
)


print(
    "\nSource wording for dissertation:"
)

print(
    "Yahoo Finance via yfinance; daily Volume field reported "
    "for the downloaded cryptocurrency series."
)


print(
    "\nTransformation wording:"
)

print(
    "Raw trading volume was transformed as ln(1 + Volume) "
    "to reduce positive skewness."
)


print(
    "\nForecast-timing wording:"
)

print(
    "Predictive specifications used the one-calendar-day lag "
    "of transformed cryptocurrency trading volume."
)


print(
    "\nDo NOT automatically describe Yahoo Finance Volume "
    "as total global cryptocurrency-market volume."
)


section(
    "TRADING-VOLUME VERIFICATION COMPLETE"
)