"""
06_prepare_reddit_forecast.py

SECTION 06 — CALENDAR, MISSING DAYS AND LAGS
Forecast-ready Reddit predictor dataset

Dissertation:
Do Social Media Sentiment Signals Improve the Prediction of
Cryptocurrency Returns Beyond Traditional Market Indicators?
Evidence from Bitcoin and Ethereum

===============================================================================
PURPOSE
===============================================================================

This script prepares the daily Reddit variables created in Section 05 for
subsequent explanatory regressions and genuine out-of-sample forecasting.

It performs the following tasks:

1. Builds a complete DAILY calendar for BTC and ETH from:
       2021-01-01 to 2025-12-31.

2. Merges the validated Section 05 daily Reddit sentiment and activity
   measures onto the complete calendar.

3. Uses the previously validated raw-versus-retained Reddit coverage data
   to distinguish:

       a. OBSERVED_RETAINED_POSTS
          At least one Reddit post survives the cleaning criteria.

       b. RAW_POSTS_ALL_EXCLUDED
          Reddit posts existed in the raw source on that date, but all of
          them were excluded by the documented cleaning rules.

       c. GENUINE_ZERO_OBSERVED_RAW_ACTIVITY
          The date lies inside confirmed Reddit source coverage and the
          validated raw dataset contains zero observed posts.

       d. OUTSIDE_REDDIT_SOURCE_COVERAGE
          The date lies outside confirmed Reddit extraction coverage.

       e. UNRESOLVED_MISSING
          Defensive category. The final validated dataset should contain
          zero such cases.

4. Distinguishes zero analytical activity from missing information.

5. Preserves missing sentiment correctly:
       sentiment = 0 means neutral sentiment
       sentiment = NaN means no sentiment observation exists

6. Defines Reddit activity as:
       log(1 + retained Reddit post count)

7. Creates the PRIMARY Reddit forecasting predictors using t-1.

8. Creates t-2, t-3 and t-7 Reddit predictors for lag-length robustness.

9. Uses CALENDAR-DAY lags, never "previous available Reddit observation".

10. Records the source date associated with every lagged predictor.

11. Explicitly checks that predictor information precedes the target date.

12. Tests for accidental contemporaneous / future-data contamination.

13. Saves a forecast-ready REDDIT-ONLY predictor dataset.

===============================================================================
IMPORTANT FORECASTING PRINCIPLE
===============================================================================

For a cryptocurrency return on target date t:

        Reddit information from t-1
                    |
                    v
              Return on date t

No Reddit information observed during target date t is used as a predictor
of the return for date t.

This timing directly supports the dissertation hypotheses involving
lagged Reddit sentiment and subsequent cryptocurrency returns.

Section 06 does NOT estimate the hypotheses. It constructs the predictors
needed to test them correctly.

===============================================================================
WHAT IS NOT DONE IN SECTION 06
===============================================================================

This script deliberately does NOT merge:

    - BTC returns
    - ETH returns
    - BTC trading volume
    - ETH trading volume
    - S&P 500
    - VIX
    - Gold
    - DXY
    - US 10-year Treasury yield

These variables belong in Section 07.

Traditional-market controls require separate information-availability
alignment because cryptocurrency markets operate 24/7 whereas traditional
financial markets do not.

Cryptocurrency volume must also be lagged before use in forecasting.

===============================================================================
REDDIT VARIABLE DEFINITIONS
===============================================================================

Post-level continuous sentiment:

    S_i = P(Positive_i) - P(Negative_i)

Daily Reddit sentiment:

    Sentiment_(a,t)
        = arithmetic mean of S_i
          across retained posts for asset a on date t.

The daily sentiment measure is UNWEIGHTED.

Reddit scores, upvotes, upvote ratios and comment counts are NOT used as
sentiment weights.

Daily Reddit activity:

    N_(a,t) = number of retained posts

Modelling transformation:

    Activity_(a,t) = log(1 + N_(a,t))

Sentiment and activity are retained separately.

===============================================================================
MISSING-DATA PRINCIPLE
===============================================================================

A missing Section 05 row must NOT automatically be interpreted as:

    sentiment = 0

or:

    activity = 0

The previously validated raw coverage dataset determines why the row is
missing.

If raw source coverage is confirmed and raw_post_count == 0:

    retained analytical post count = 0
    log activity = 0
    sentiment = NaN

If raw posts existed but every post was excluded:

    retained analytical post count = 0
    log activity = 0
    sentiment = NaN

If the date is outside Reddit source coverage:

    activity = NaN
    sentiment = NaN

The Reddit extraction supplied for this dissertation ends on:

    2025-12-30

Therefore 2025-12-31 is OUTSIDE REDDIT SOURCE COVERAGE for both BTC and ETH.
It must NOT be treated as a zero-activity Reddit day.
"""


# =============================================================================
# 01. IMPORTS
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 02. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

REDDIT_DIR = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
)

STAGE05_DIR = (
    REDDIT_DIR
    / "stage05_daily_reddit"
)

# Section 06 outputs are processed Reddit data.
OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "reddit"
    / "stage06_reddit_forecast"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =============================================================================
# 03. INPUT FILES
# =============================================================================

# Validated daily raw-versus-retained Reddit coverage.
COVERAGE_INPUT_FILE = (
    REDDIT_DIR
    / "reddit_daily_coverage_after_cleaning.csv"
)

# Validated Section 05 daily Reddit variables.
STAGE05_INPUT_FILE = (
    STAGE05_DIR
    / "reddit_daily_sentiment_activity.csv"
)


# =============================================================================
# 04. OUTPUT FILES
# =============================================================================

FORECAST_READY_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_ready.csv"
)

MISSING_CLASSIFICATION_FILE = (
    OUTPUT_DIR
    / "reddit_missing_day_classification.csv"
)

COVERAGE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_calendar_coverage.csv"
)

LAG_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_lag_validation.csv"
)

LOOKAHEAD_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_no_lookahead_validation.csv"
)

QC_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_stage06_qc.csv"
)

METHODOLOGY_FILE = (
    OUTPUT_DIR
    / "reddit_forecast_methodology_note.txt"
)


# =============================================================================
# 05. STUDY SETTINGS
# =============================================================================

STUDY_START = pd.Timestamp(
    "2021-01-01"
)

STUDY_END = pd.Timestamp(
    "2025-12-31"
)

# Confirmed Reddit extraction coverage.
REDDIT_SOURCE_START = pd.Timestamp(
    "2021-01-01"
)

REDDIT_SOURCE_END = pd.Timestamp(
    "2025-12-30"
)

ASSETS = [
    "BTC",
    "ETH",
]

PRIMARY_LAG = 1

ROBUSTNESS_LAGS = [
    2,
    3,
    7,
]

ALL_LAGS = [
    PRIMARY_LAG,
    *ROBUSTNESS_LAGS,
]

EXPECTED_CALENDAR_DAYS = 1826

EXPECTED_DATE_ASSET_ROWS = (
    EXPECTED_CALENDAR_DAYS
    * len(ASSETS)
)

EXPECTED_STAGE05_ROWS = 3628

EXPECTED_BTC_STAGE05_DAYS = 1825

EXPECTED_ETH_STAGE05_DAYS = 1803

EXPECTED_UNOBSERVED_ASSET_DAYS = 24


# =============================================================================
# 06. HELPER FUNCTIONS
# =============================================================================

def section(title):
    """Print a clearly separated console section."""
    print()
    print("=" * 88)
    print(title)
    print("=" * 88)


def fail(message):
    """Stop execution when a validation check fails."""
    raise ValueError(
        "\nSECTION 06 VALIDATION FAILED\n"
        + str(message)
    )


def require_file(path):
    """Check that an input file exists and is non-empty."""

    if not path.exists():
        fail(
            "Required input file does not exist:\n"
            f"{path}"
        )

    if path.stat().st_size == 0:
        fail(
            "Required input file is empty:\n"
            f"{path}"
        )


def require_columns(
    dataframe,
    required_columns,
    dataframe_name,
):
    """Check that all required columns are present."""

    missing = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing:
        fail(
            f"{dataframe_name} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(dataframe.columns)}"
        )


def normalise_asset(series):
    """Standardise BTC/ETH asset labels."""

    return (
        series
        .astype(str)
        .str.strip()
        .str.upper()
    )


def normalise_date(series):
    """Parse dates and remove time components."""

    return pd.to_datetime(
        series,
        errors="coerce",
    ).dt.normalize()


def safe_numeric(
    dataframe,
    columns,
):
    """Convert selected columns to numeric safely."""

    result = dataframe.copy()

    for column in columns:

        result[column] = (
            pd.to_numeric(
                result[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

    return result


def values_equal(
    left,
    right,
    atol=1e-12,
    rtol=1e-12,
):
    """
    Compare numeric arrays while treating paired NaNs as equal.
    """

    left = pd.to_numeric(
        left,
        errors="coerce",
    )

    right = pd.to_numeric(
        right,
        errors="coerce",
    )

    return np.isclose(
        left,
        right,
        equal_nan=True,
        atol=atol,
        rtol=rtol,
    )


# =============================================================================
# 07. LOAD INPUT DATA
# =============================================================================

section(
    "SECTION 06 — LOADING INPUT DATA"
)

require_file(
    COVERAGE_INPUT_FILE
)

require_file(
    STAGE05_INPUT_FILE
)

print(
    "Coverage input:\n"
    f"{COVERAGE_INPUT_FILE}"
)

print(
    "\nSection 05 input:\n"
    f"{STAGE05_INPUT_FILE}"
)

coverage_raw = pd.read_csv(
    COVERAGE_INPUT_FILE,
    low_memory=False,
)

daily = pd.read_csv(
    STAGE05_INPUT_FILE,
    low_memory=False,
)

coverage_raw.columns = [
    str(column).strip()
    for column in coverage_raw.columns
]

daily.columns = [
    str(column).strip()
    for column in daily.columns
]

print(
    f"\nCoverage rows loaded: "
    f"{len(coverage_raw):,}"
)

print(
    f"Section 05 rows loaded: "
    f"{len(daily):,}"
)


# =============================================================================
# 08. VALIDATE INPUT SCHEMAS
# =============================================================================

section(
    "VALIDATING INPUT SCHEMAS"
)

require_columns(
    coverage_raw,
    [
        "post_date",
        "asset",
        "raw_post_count",
        "retained_post_count",
        "excluded_post_count",
    ],
    "reddit_daily_coverage_after_cleaning.csv",
)

require_columns(
    daily,
    [
        "post_date",
        "asset",
        "Post_Count",
        "Log_Reddit_Post_Count",
        "Mean_Reddit_Sentiment",
    ],
    "reddit_daily_sentiment_activity.csv",
)

print(
    "Required coverage columns: PASS"
)

print(
    "Required Section 05 columns: PASS"
)


# =============================================================================
# 09. STANDARDISE DATES AND ASSETS
# =============================================================================

section(
    "STANDARDISING DATES AND ASSETS"
)

coverage_raw["post_date"] = normalise_date(
    coverage_raw["post_date"]
)

daily["post_date"] = normalise_date(
    daily["post_date"]
)

if coverage_raw["post_date"].isna().any():
    fail(
        "Invalid dates were found in the coverage dataset."
    )

if daily["post_date"].isna().any():
    fail(
        "Invalid dates were found in the Section 05 dataset."
    )

coverage_raw["asset"] = normalise_asset(
    coverage_raw["asset"]
)

daily["asset"] = normalise_asset(
    daily["asset"]
)

unexpected_coverage_assets = (
    set(
        coverage_raw["asset"].unique()
    )
    - set(ASSETS)
)

unexpected_daily_assets = (
    set(
        daily["asset"].unique()
    )
    - set(ASSETS)
)

if unexpected_coverage_assets:
    fail(
        "Unexpected assets in coverage data:\n"
        f"{sorted(unexpected_coverage_assets)}"
    )

if unexpected_daily_assets:
    fail(
        "Unexpected assets in Section 05 data:\n"
        f"{sorted(unexpected_daily_assets)}"
    )

print(
    "Date parsing: PASS"
)

print(
    "Asset standardisation: PASS"
)


# =============================================================================
# 10. VALIDATE STUDY PERIOD
# =============================================================================

section(
    "VALIDATING STUDY PERIOD"
)

coverage_outside = (
    coverage_raw["post_date"].lt(
        STUDY_START
    )
    |
    coverage_raw["post_date"].gt(
        STUDY_END
    )
)

daily_outside = (
    daily["post_date"].lt(
        STUDY_START
    )
    |
    daily["post_date"].gt(
        STUDY_END
    )
)

if coverage_outside.any():
    fail(
        f"{int(coverage_outside.sum())} coverage rows "
        "fall outside the study period."
    )

if daily_outside.any():
    fail(
        f"{int(daily_outside.sum())} Section 05 rows "
        "fall outside the study period."
    )

print(
    f"Study period: "
    f"{STUDY_START.date()} to "
    f"{STUDY_END.date()}"
)

print(
    "Study-period validation: PASS"
)


# =============================================================================
# 11. VALIDATE DATE × ASSET UNIQUENESS
# =============================================================================

section(
    "VALIDATING DATE × ASSET UNIQUENESS"
)

coverage_duplicates = int(
    coverage_raw.duplicated(
        [
            "post_date",
            "asset",
        ]
    ).sum()
)

daily_duplicates = int(
    daily.duplicated(
        [
            "post_date",
            "asset",
        ]
    ).sum()
)

print(
    f"Coverage duplicate Date × Asset rows: "
    f"{coverage_duplicates}"
)

print(
    f"Section 05 duplicate Date × Asset rows: "
    f"{daily_duplicates}"
)

if coverage_duplicates != 0:
    fail(
        "Duplicate Date × Asset rows exist in "
        "the coverage dataset."
    )

if daily_duplicates != 0:
    fail(
        "Duplicate Date × Asset rows exist in "
        "the Section 05 dataset."
    )

print(
    "Date × Asset uniqueness: PASS"
)


# =============================================================================
# 12. VALIDATE NUMERIC COVERAGE VARIABLES
# =============================================================================

section(
    "VALIDATING NUMERIC COVERAGE VARIABLES"
)

coverage_raw = safe_numeric(
    coverage_raw,
    [
        "raw_post_count",
        "retained_post_count",
        "excluded_post_count",
    ],
)

daily = safe_numeric(
    daily,
    [
        "Post_Count",
        "Log_Reddit_Post_Count",
        "Mean_Reddit_Sentiment",
    ],
)

coverage_count_columns = [
    "raw_post_count",
    "retained_post_count",
    "excluded_post_count",
]

if coverage_raw[
    coverage_count_columns
].isna().any().any():

    fail(
        "Missing or non-numeric raw/retained coverage "
        "counts were found."
    )

for column in coverage_count_columns:

    if (
        coverage_raw[column] < 0
    ).any():

        fail(
            f"Negative values found in {column}."
        )


count_reconciliation = (
    coverage_raw[
        "retained_post_count"
    ]
    +
    coverage_raw[
        "excluded_post_count"
    ]
)

count_mismatch = ~np.isclose(
    coverage_raw[
        "raw_post_count"
    ],
    count_reconciliation,
    atol=0,
    rtol=0,
)

if count_mismatch.any():

    example = coverage_raw.loc[
        count_mismatch,
        [
            "post_date",
            "asset",
            "raw_post_count",
            "retained_post_count",
            "excluded_post_count",
        ],
    ].head(20)

    fail(
        "Raw post count does not equal retained + excluded "
        "for at least one Date × Asset row.\n\n"
        + example.to_string(
            index=False
        )
    )

print(
    "Raw = retained + excluded: PASS"
)


# =============================================================================
# 13. VALIDATE SECTION 05 ACTIVITY TRANSFORMATION
# =============================================================================

section(
    "VALIDATING SECTION 05 ACTIVITY TRANSFORMATION"
)

if (
    daily["Post_Count"].isna()
).any():

    fail(
        "Missing Post_Count values exist in observed "
        "Section 05 rows."
    )

if (
    daily["Post_Count"] <= 0
).any():

    fail(
        "Section 05 contains an observed row with "
        "Post_Count <= 0."
    )

expected_log_activity = np.log1p(
    daily["Post_Count"]
)

activity_difference = (
    daily[
        "Log_Reddit_Post_Count"
    ]
    - expected_log_activity
).abs()

max_activity_difference = float(
    activity_difference.max()
)

print(
    "Maximum |stored log activity "
    "- log(1 + Post_Count)|:"
)

print(
    max_activity_difference
)

if max_activity_difference > 1e-10:

    fail(
        "Section 05 log activity does not equal "
        "log(1 + Post_Count)."
    )


invalid_sentiment_range = (
    daily[
        "Mean_Reddit_Sentiment"
    ].lt(-1)
    |
    daily[
        "Mean_Reddit_Sentiment"
    ].gt(1)
)

if invalid_sentiment_range.any():

    fail(
        "Daily Reddit sentiment falls outside [-1, 1]."
    )

print(
    "Activity transformation: PASS"
)

print(
    "Sentiment range: PASS"
)


# =============================================================================
# 14. VALIDATE FROZEN SECTION 05 COUNTS
# =============================================================================

section(
    "VALIDATING FROZEN SECTION 05 COUNTS"
)

btc_stage05_days = int(
    daily[
        "asset"
    ].eq(
        "BTC"
    ).sum()
)

eth_stage05_days = int(
    daily[
        "asset"
    ].eq(
        "ETH"
    ).sum()
)

print(
    f"Section 05 total rows: "
    f"{len(daily):,}"
)

print(
    f"BTC observed days: "
    f"{btc_stage05_days:,}"
)

print(
    f"ETH observed days: "
    f"{eth_stage05_days:,}"
)

if len(daily) != EXPECTED_STAGE05_ROWS:

    fail(
        "Unexpected Section 05 row count.\n"
        f"Expected: {EXPECTED_STAGE05_ROWS:,}\n"
        f"Found:    {len(daily):,}"
    )

if (
    btc_stage05_days
    != EXPECTED_BTC_STAGE05_DAYS
):

    fail(
        "Unexpected BTC Stage 05 daily count.\n"
        f"Expected: {EXPECTED_BTC_STAGE05_DAYS:,}\n"
        f"Found:    {btc_stage05_days:,}"
    )

if (
    eth_stage05_days
    != EXPECTED_ETH_STAGE05_DAYS
):

    fail(
        "Unexpected ETH Stage 05 daily count.\n"
        f"Expected: {EXPECTED_ETH_STAGE05_DAYS:,}\n"
        f"Found:    {eth_stage05_days:,}"
    )

print(
    "Frozen Section 05 counts: PASS"
)


# =============================================================================
# 15. BUILD COMPLETE BTC/ETH DAILY CALENDAR
# =============================================================================

section(
    "BUILDING COMPLETE BTC/ETH DAILY CALENDAR"
)

calendar_dates = pd.date_range(
    STUDY_START,
    STUDY_END,
    freq="D",
)

if (
    len(calendar_dates)
    != EXPECTED_CALENDAR_DAYS
):

    fail(
        "Unexpected number of calendar days.\n"
        f"Expected: {EXPECTED_CALENDAR_DAYS:,}\n"
        f"Found:    {len(calendar_dates):,}"
    )


calendar = (
    pd.MultiIndex.from_product(
        [
            calendar_dates,
            ASSETS,
        ],
        names=[
            "Date",
            "Asset",
        ],
    )
    .to_frame(
        index=False
    )
)

print(
    f"Calendar days: "
    f"{len(calendar_dates):,}"
)

print(
    f"Date × Asset rows: "
    f"{len(calendar):,}"
)

if (
    len(calendar)
    != EXPECTED_DATE_ASSET_ROWS
):

    fail(
        "Complete calendar does not contain "
        "the expected number of Date × Asset rows."
    )


# =============================================================================
# 16. PREPARE COVERAGE DATA
# =============================================================================

section(
    "PREPARING RAW-VERSUS-RETAINED COVERAGE"
)

coverage = coverage_raw[
    [
        "post_date",
        "asset",
        "raw_post_count",
        "retained_post_count",
        "excluded_post_count",
    ]
].copy()

coverage = coverage.rename(
    columns={
        "post_date":
            "Date",

        "asset":
            "Asset",

        "raw_post_count":
            "Raw_Reddit_Post_Count",

        "retained_post_count":
            "Retained_Reddit_Post_Count_Coverage",

        "excluded_post_count":
            "Excluded_Reddit_Post_Count",
    }
)


calendar = calendar.merge(
    coverage,
    how="left",
    on=[
        "Date",
        "Asset",
    ],
    validate="one_to_one",
)


coverage_columns = [
    "Raw_Reddit_Post_Count",
    "Retained_Reddit_Post_Count_Coverage",
    "Excluded_Reddit_Post_Count",
]

if calendar[
    coverage_columns
].isna().any().any():

    missing_coverage = calendar.loc[
        calendar[
            "Raw_Reddit_Post_Count"
        ].isna(),
        [
            "Date",
            "Asset",
        ],
    ]

    fail(
        "The validated coverage dataset does not span "
        "the complete study calendar.\n\n"
        + missing_coverage.head(
            50
        ).to_string(
            index=False
        )
    )

print(
    "Coverage merged onto complete calendar: PASS"
)


# =============================================================================
# 17. PREPARE SECTION 05 DAILY REDDIT VARIABLES
# =============================================================================

section(
    "PREPARING SECTION 05 DAILY REDDIT VARIABLES"
)

reddit = daily[
    [
        "post_date",
        "asset",
        "Post_Count",
        "Log_Reddit_Post_Count",
        "Mean_Reddit_Sentiment",
    ]
].copy()


reddit = reddit.rename(
    columns={
        "post_date":
            "Date",

        "asset":
            "Asset",

        "Post_Count":
            "Stage05_Reddit_Post_Count",

        "Log_Reddit_Post_Count":
            "Stage05_Log_Reddit_Post_Count",

        "Mean_Reddit_Sentiment":
            "Reddit_Sentiment",
    }
)


forecast = calendar.merge(
    reddit,
    on=[
        "Date",
        "Asset",
    ],
    how="left",
    validate="one_to_one",
)


if (
    len(forecast)
    != EXPECTED_DATE_ASSET_ROWS
):

    fail(
        "Section 05 merge changed the number "
        "of Date × Asset calendar rows."
    )


forecast = (
    forecast
    .sort_values(
        [
            "Asset",
            "Date",
        ]
    )
    .reset_index(
        drop=True
    )
)

print(
    "Section 05 variables merged: PASS"
)


# =============================================================================
# 18. CROSS-CHECK SECTION 05 AGAINST CLEANING COVERAGE
# =============================================================================

section(
    "CROSS-CHECKING SECTION 05 AGAINST CLEANING COVERAGE"
)

stage05_observed = (
    forecast[
        "Stage05_Reddit_Post_Count"
    ].notna()
)

coverage_retained_positive = (
    forecast[
        "Retained_Reddit_Post_Count_Coverage"
    ].gt(0)
)


membership_mismatch = (
    stage05_observed
    != coverage_retained_positive
)


if membership_mismatch.any():

    problem = forecast.loc[
        membership_mismatch,
        [
            "Date",
            "Asset",
            "Raw_Reddit_Post_Count",
            "Retained_Reddit_Post_Count_Coverage",
            "Excluded_Reddit_Post_Count",
            "Stage05_Reddit_Post_Count",
        ],
    ]

    fail(
        "Section 05 observed-date membership does not "
        "match retained-post coverage.\n\n"
        + problem.head(
            50
        ).to_string(
            index=False
        )
    )


count_match = values_equal(
    forecast[
        "Stage05_Reddit_Post_Count"
    ],
    forecast[
        "Retained_Reddit_Post_Count_Coverage"
    ].where(
        coverage_retained_positive
    ),
)


if not count_match.all():

    problem = forecast.loc[
        ~count_match,
        [
            "Date",
            "Asset",
            "Stage05_Reddit_Post_Count",
            "Retained_Reddit_Post_Count_Coverage",
        ],
    ]

    fail(
        "Section 05 post counts do not match "
        "validated retained-post coverage.\n\n"
        + problem.head(
            50
        ).to_string(
            index=False
        )
    )


print(
    "Section 05 membership versus cleaning coverage: PASS"
)

print(
    "Section 05 retained counts versus coverage: PASS"
)


# =============================================================================
# 19. IDENTIFY CONFIRMED REDDIT SOURCE COVERAGE
# =============================================================================

section(
    "IDENTIFYING CONFIRMED REDDIT SOURCE COVERAGE"
)


forecast[
    "Within_Reddit_Source_Coverage"
] = forecast[
    "Date"
].between(
    REDDIT_SOURCE_START,
    REDDIT_SOURCE_END,
    inclusive="both",
)


forecast[
    "Outside_Reddit_Source_Coverage"
] = ~forecast[
    "Within_Reddit_Source_Coverage"
]


outside_rows = forecast.loc[
    forecast[
        "Outside_Reddit_Source_Coverage"
    ],
    [
        "Date",
        "Asset",
    ],
].copy()


print(
    "\nRows outside confirmed Reddit source coverage:"
)

print(
    outside_rows.to_string(
        index=False
    )
)


expected_outside = {
    (
        pd.Timestamp(
            "2025-12-31"
        ),
        "BTC",
    ),
    (
        pd.Timestamp(
            "2025-12-31"
        ),
        "ETH",
    ),
}


actual_outside = set(
    outside_rows.itertuples(
        index=False,
        name=None,
    )
)


if actual_outside != expected_outside:

    fail(
        "Unexpected rows outside confirmed Reddit "
        "source coverage.\n\n"
        f"Expected:\n{expected_outside}\n\n"
        f"Found:\n{actual_outside}"
    )


print(
    "Confirmed source-coverage boundary: PASS"
)


# =============================================================================
# 20. CLASSIFY REDDIT OBSERVATION STATUS
# =============================================================================

section(
    "CLASSIFYING REDDIT OBSERVATION STATUS"
)


forecast[
    "Reddit_Observation_Status"
] = "UNRESOLVED_MISSING"


# -----------------------------------------------------------------------------
# CATEGORY 1:
# At least one cleaned / retained Reddit post exists.
# -----------------------------------------------------------------------------

condition_observed = (
    forecast[
        "Within_Reddit_Source_Coverage"
    ]
    &
    forecast[
        "Retained_Reddit_Post_Count_Coverage"
    ].gt(0)
)


forecast.loc[
    condition_observed,
    "Reddit_Observation_Status",
] = "OBSERVED_RETAINED_POSTS"


# -----------------------------------------------------------------------------
# CATEGORY 2:
# Raw posts existed, but all were excluded.
# -----------------------------------------------------------------------------

condition_all_excluded = (
    forecast[
        "Within_Reddit_Source_Coverage"
    ]
    &
    forecast[
        "Raw_Reddit_Post_Count"
    ].gt(0)
    &
    forecast[
        "Retained_Reddit_Post_Count_Coverage"
    ].eq(0)
    &
    forecast[
        "Excluded_Reddit_Post_Count"
    ].eq(
        forecast[
            "Raw_Reddit_Post_Count"
        ]
    )
)


forecast.loc[
    condition_all_excluded,
    "Reddit_Observation_Status",
] = "RAW_POSTS_ALL_EXCLUDED"


# -----------------------------------------------------------------------------
# CATEGORY 3:
# No raw posts observed inside confirmed extraction coverage.
# -----------------------------------------------------------------------------

condition_zero_raw = (
    forecast[
        "Within_Reddit_Source_Coverage"
    ]
    &
    forecast[
        "Raw_Reddit_Post_Count"
    ].eq(0)
    &
    forecast[
        "Retained_Reddit_Post_Count_Coverage"
    ].eq(0)
    &
    forecast[
        "Excluded_Reddit_Post_Count"
    ].eq(0)
)


forecast.loc[
    condition_zero_raw,
    "Reddit_Observation_Status",
] = "GENUINE_ZERO_OBSERVED_RAW_ACTIVITY"


# -----------------------------------------------------------------------------
# CATEGORY 4:
# Outside confirmed Reddit extraction coverage.
# -----------------------------------------------------------------------------

condition_outside = (
    forecast[
        "Outside_Reddit_Source_Coverage"
    ]
)


forecast.loc[
    condition_outside,
    "Reddit_Observation_Status",
] = "OUTSIDE_REDDIT_SOURCE_COVERAGE"


# =============================================================================
# 21. VERIFY NO UNRESOLVED CLASSIFICATIONS
# =============================================================================

section(
    "VALIDATING REDDIT MISSING-DAY CLASSIFICATION"
)


status_counts = (
    forecast[
        "Reddit_Observation_Status"
    ]
    .value_counts(
        dropna=False
    )
)


print(
    status_counts.to_string()
)


unresolved_count = int(
    forecast[
        "Reddit_Observation_Status"
    ]
    .eq(
        "UNRESOLVED_MISSING"
    )
    .sum()
)


if unresolved_count != 0:

    unresolved = forecast.loc[
        forecast[
            "Reddit_Observation_Status"
        ].eq(
            "UNRESOLVED_MISSING"
        ),
        [
            "Date",
            "Asset",
            "Raw_Reddit_Post_Count",
            "Retained_Reddit_Post_Count_Coverage",
            "Excluded_Reddit_Post_Count",
            "Stage05_Reddit_Post_Count",
        ],
    ]

    fail(
        "Unresolved Reddit observation statuses remain.\n\n"
        + unresolved.to_string(
            index=False
        )
    )


print(
    "Missing-day classification: PASS"
)


# =============================================================================
# 22. VALIDATE THE 24 SECTION 05 UNOBSERVED ASSET-DAYS
# =============================================================================

section(
    "VALIDATING SECTION 05 UNOBSERVED ASSET-DAYS"
)


stage05_unobserved = forecast.loc[
    forecast[
        "Stage05_Reddit_Post_Count"
    ].isna(),
    [
        "Date",
        "Asset",
        "Raw_Reddit_Post_Count",
        "Retained_Reddit_Post_Count_Coverage",
        "Excluded_Reddit_Post_Count",
        "Reddit_Observation_Status",
    ],
].copy()


print(
    f"Section 05 unobserved Date × Asset rows: "
    f"{len(stage05_unobserved):,}"
)

print(
    stage05_unobserved.to_string(
        index=False
    )
)


if (
    len(stage05_unobserved)
    != EXPECTED_UNOBSERVED_ASSET_DAYS
):

    fail(
        "Unexpected number of Section 05 unobserved "
        "asset-days.\n"
        f"Expected: "
        f"{EXPECTED_UNOBSERVED_ASSET_DAYS:,}\n"
        f"Found:    "
        f"{len(stage05_unobserved):,}"
    )


print(
    "Expected Section 05 missing-day count: PASS"
)


# =============================================================================
# 23. VALIDATE KNOWN MISSING-DAY COMPOSITION
# =============================================================================

section(
    "VALIDATING KNOWN MISSING-DAY COMPOSITION"
)


all_excluded_count = int(
    forecast[
        "Reddit_Observation_Status"
    ]
    .eq(
        "RAW_POSTS_ALL_EXCLUDED"
    )
    .sum()
)


zero_raw_count = int(
    forecast[
        "Reddit_Observation_Status"
    ]
    .eq(
        "GENUINE_ZERO_OBSERVED_RAW_ACTIVITY"
    )
    .sum()
)


outside_coverage_count = int(
    forecast[
        "Reddit_Observation_Status"
    ]
    .eq(
        "OUTSIDE_REDDIT_SOURCE_COVERAGE"
    )
    .sum()
)


print(
    f"Raw posts all excluded: "
    f"{all_excluded_count}"
)

print(
    f"Zero raw posts within source coverage: "
    f"{zero_raw_count}"
)

print(
    f"Outside Reddit source coverage: "
    f"{outside_coverage_count}"
)


# Validated earlier in the Reddit cleaning/coverage audit:
#
# 8 ETH asset-days had raw posts but all posts were excluded.
# 14 ETH asset-days had zero raw observed posts within source coverage.
# 2025-12-31 is outside source coverage for both BTC and ETH.

if all_excluded_count != 8:

    fail(
        "Expected exactly 8 Date × Asset rows where "
        "raw posts existed but all were excluded.\n"
        f"Found: {all_excluded_count}"
    )


if zero_raw_count != 14:

    fail(
        "Expected exactly 14 Date × Asset rows with "
        "zero observed raw posts inside confirmed "
        "source coverage.\n"
        f"Found: {zero_raw_count}"
    )


if outside_coverage_count != 2:

    fail(
        "Expected exactly 2 Date × Asset rows outside "
        "Reddit source coverage.\n"
        f"Found: {outside_coverage_count}"
    )


if (
    all_excluded_count
    + zero_raw_count
    + outside_coverage_count
    != EXPECTED_UNOBSERVED_ASSET_DAYS
):

    fail(
        "Missing-day categories do not reconcile "
        "to the expected 24 Section 05 unobserved "
        "asset-days."
    )


print(
    "Known missing-day composition: PASS"
)


# =============================================================================
# 24. VALIDATE THE EIGHT KNOWN ALL-EXCLUDED ETH DATES
# =============================================================================

section(
    "VALIDATING KNOWN ALL-EXCLUDED ETH DATES"
)


expected_all_excluded_eth_dates = {
    pd.Timestamp(
        "2023-04-22"
    ),
    pd.Timestamp(
        "2023-08-19"
    ),
    pd.Timestamp(
        "2023-08-30"
    ),
    pd.Timestamp(
        "2023-09-23"
    ),
    pd.Timestamp(
        "2023-10-09"
    ),
    pd.Timestamp(
        "2024-06-01"
    ),
    pd.Timestamp(
        "2024-07-06"
    ),
    pd.Timestamp(
        "2024-08-21"
    ),
}


actual_all_excluded_eth_dates = set(
    forecast.loc[
        (
            forecast[
                "Asset"
            ].eq(
                "ETH"
            )
            &
            forecast[
                "Reddit_Observation_Status"
            ].eq(
                "RAW_POSTS_ALL_EXCLUDED"
            )
        ),
        "Date",
    ]
)


if (
    actual_all_excluded_eth_dates
    != expected_all_excluded_eth_dates
):

    fail(
        "All-excluded ETH dates do not match the "
        "previously validated coverage audit.\n\n"
        f"Expected:\n"
        f"{sorted(expected_all_excluded_eth_dates)}\n\n"
        f"Found:\n"
        f"{sorted(actual_all_excluded_eth_dates)}"
    )


print(
    "Eight known all-excluded ETH dates: PASS"
)


# =============================================================================
# 25. CONSTRUCT FINAL CONTEMPORANEOUS REDDIT ACTIVITY
# =============================================================================

section(
    "CONSTRUCTING ANALYTICAL REDDIT ACTIVITY"
)


# Start as missing.
forecast[
    "Reddit_Post_Count"
] = np.nan


# -----------------------------------------------------------------------------
# Observed retained-post dates
# -----------------------------------------------------------------------------

observed_mask = (
    forecast[
        "Reddit_Observation_Status"
    ].eq(
        "OBSERVED_RETAINED_POSTS"
    )
)


forecast.loc[
    observed_mask,
    "Reddit_Post_Count",
] = forecast.loc[
    observed_mask,
    "Stage05_Reddit_Post_Count",
]


# -----------------------------------------------------------------------------
# Raw posts existed but all were excluded.
#
# The retained analytical activity count is zero.
# -----------------------------------------------------------------------------

all_excluded_mask = (
    forecast[
        "Reddit_Observation_Status"
    ].eq(
        "RAW_POSTS_ALL_EXCLUDED"
    )
)


forecast.loc[
    all_excluded_mask,
    "Reddit_Post_Count",
] = 0.0


# -----------------------------------------------------------------------------
# Confirmed zero raw observed posts.
#
# The retained analytical activity count is also zero.
# -----------------------------------------------------------------------------

zero_raw_mask = (
    forecast[
        "Reddit_Observation_Status"
    ].eq(
        "GENUINE_ZERO_OBSERVED_RAW_ACTIVITY"
    )
)


forecast.loc[
    zero_raw_mask,
    "Reddit_Post_Count",
] = 0.0


# Outside source coverage remains NaN.


# =============================================================================
# 26. CONSTRUCT FINAL LOG ACTIVITY
# =============================================================================

forecast[
    "Log_Reddit_Post_Count"
] = np.where(
    forecast[
        "Reddit_Post_Count"
    ].notna(),
    np.log1p(
        forecast[
            "Reddit_Post_Count"
        ]
    ),
    np.nan,
)


# Confirm observed Section 05 log activity is unchanged.

observed_activity_match = values_equal(
    forecast.loc[
        observed_mask,
        "Log_Reddit_Post_Count",
    ],
    forecast.loc[
        observed_mask,
        "Stage05_Log_Reddit_Post_Count",
    ],
)


if not observed_activity_match.all():

    fail(
        "Observed Section 05 log activity changed "
        "during Section 06 construction."
    )


print(
    "Analytical Reddit activity construction: PASS"
)


# =============================================================================
# 27. PRESERVE SENTIMENT MISSINGNESS
# =============================================================================

section(
    "VALIDATING SENTIMENT MISSINGNESS"
)


# Sentiment must only exist where retained posts exist.

invalid_sentiment_nonobserved = (
    ~forecast[
        "Reddit_Observation_Status"
    ].eq(
        "OBSERVED_RETAINED_POSTS"
    )
    &
    forecast[
        "Reddit_Sentiment"
    ].notna()
)


if invalid_sentiment_nonobserved.any():

    problem = forecast.loc[
        invalid_sentiment_nonobserved,
        [
            "Date",
            "Asset",
            "Reddit_Observation_Status",
            "Reddit_Sentiment",
        ],
    ]

    fail(
        "Sentiment was found on a date with no retained "
        "Reddit posts.\n\n"
        + problem.to_string(
            index=False
        )
    )


invalid_observed_missing_sentiment = (
    forecast[
        "Reddit_Observation_Status"
    ].eq(
        "OBSERVED_RETAINED_POSTS"
    )
    &
    forecast[
        "Reddit_Sentiment"
    ].isna()
)


if invalid_observed_missing_sentiment.any():

    fail(
        "An observed retained-post day has missing "
        "daily sentiment."
    )


print(
    "No sentiment = 0 imputation performed: PASS"
)

print(
    "Sentiment missingness correctly preserved: PASS"
)


# =============================================================================
# 28. CREATE AVAILABILITY INDICATORS
# =============================================================================

section(
    "CREATING REDDIT AVAILABILITY INDICATORS"
)


forecast[
    "Reddit_Activity_Available"
] = forecast[
    "Reddit_Post_Count"
].notna()


forecast[
    "Reddit_Sentiment_Available"
] = forecast[
    "Reddit_Sentiment"
].notna()


forecast[
    "Reddit_Has_Retained_Posts"
] = forecast[
    "Reddit_Post_Count"
].gt(0)


forecast[
    "Reddit_Is_Zero_Analytical_Activity"
] = forecast[
    "Reddit_Post_Count"
].eq(0)


print(
    "\nActivity availability:"
)

print(
    forecast[
        "Reddit_Activity_Available"
    ]
    .value_counts()
    .to_string()
)


print(
    "\nSentiment availability:"
)

print(
    forecast[
        "Reddit_Sentiment_Available"
    ]
    .value_counts()
    .to_string()
)


# =============================================================================
# 29. CREATE CALENDAR-DAY LAGGED REDDIT VARIABLES
# =============================================================================

section(
    "CREATING CALENDAR-DAY REDDIT LAGS"
)


# IMPORTANT:
#
# The complete calendar was constructed BEFORE shift().
#
# Therefore:
#
#     shift(1) = exactly one calendar day
#     shift(2) = exactly two calendar days
#     shift(3) = exactly three calendar days
#     shift(7) = exactly seven calendar days
#
# This prevents "previous available Reddit observation" from being
# incorrectly treated as t-1.


for lag in ALL_LAGS:

    # -------------------------------------------------------------------------
    # Lagged retained activity
    # -------------------------------------------------------------------------

    forecast[
        f"Reddit_Post_Count_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Reddit_Post_Count"
        ]
        .shift(
            lag
        )
    )


    forecast[
        f"Log_Reddit_Post_Count_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Log_Reddit_Post_Count"
        ]
        .shift(
            lag
        )
    )


    # -------------------------------------------------------------------------
    # Lagged sentiment
    # -------------------------------------------------------------------------

    forecast[
        f"Reddit_Sentiment_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Reddit_Sentiment"
        ]
        .shift(
            lag
        )
    )


    # -------------------------------------------------------------------------
    # Lagged observation status
    # -------------------------------------------------------------------------

    forecast[
        f"Reddit_Observation_Status_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Reddit_Observation_Status"
        ]
        .shift(
            lag
        )
    )


    # -------------------------------------------------------------------------
    # Lagged availability indicators
    # -------------------------------------------------------------------------

    forecast[
        f"Reddit_Activity_Available_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Reddit_Activity_Available"
        ]
        .shift(
            lag
        )
    )


    forecast[
        f"Reddit_Sentiment_Available_Lag_{lag}"
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Reddit_Sentiment_Available"
        ]
        .shift(
            lag
        )
    )


print(
    "Calendar-day lag construction completed."
)


# =============================================================================
# 30. CREATE EXPLICIT SOURCE DATES FOR EVERY LAG
# =============================================================================

section(
    "CREATING PREDICTOR SOURCE DATES"
)


for lag in ALL_LAGS:

    source_date_column = (
        f"Reddit_Source_Date_Lag_{lag}"
    )

    forecast[
        source_date_column
    ] = (
        forecast
        .groupby(
            "Asset",
            sort=False,
        )[
            "Date"
        ]
        .shift(
            lag
        )
    )


print(
    "Predictor source dates created."
)


# =============================================================================
# 31. CREATE PRIMARY GENERIC T-1 FORECASTING VARIABLES
# =============================================================================

section(
    "CREATING PRIMARY T-1 FORECASTING VARIABLES"
)


forecast[
    "Lagged_Reddit_Post_Count"
] = forecast[
    "Reddit_Post_Count_Lag_1"
]


forecast[
    "Lagged_Log_Reddit_Post_Count"
] = forecast[
    "Log_Reddit_Post_Count_Lag_1"
]


forecast[
    "Lagged_Reddit_Sentiment"
] = forecast[
    "Reddit_Sentiment_Lag_1"
]


forecast[
    "Lagged_Reddit_Source_Date"
] = forecast[
    "Reddit_Source_Date_Lag_1"
]


# =============================================================================
# 32. CREATE ASSET-SPECIFIC PRIMARY VARIABLE NAMES
# =============================================================================

section(
    "CREATING ASSET-SPECIFIC PRIMARY PREDICTORS"
)


forecast[
    "Lagged_Log_BTC_Reddit_Post_Count"
] = np.nan

forecast[
    "Lagged_BTC_Reddit_Sentiment"
] = np.nan

forecast[
    "Lagged_Log_ETH_Reddit_Post_Count"
] = np.nan

forecast[
    "Lagged_ETH_Reddit_Sentiment"
] = np.nan


btc_mask = forecast[
    "Asset"
].eq(
    "BTC"
)

eth_mask = forecast[
    "Asset"
].eq(
    "ETH"
)


forecast.loc[
    btc_mask,
    "Lagged_Log_BTC_Reddit_Post_Count",
] = forecast.loc[
    btc_mask,
    "Log_Reddit_Post_Count_Lag_1",
]


forecast.loc[
    btc_mask,
    "Lagged_BTC_Reddit_Sentiment",
] = forecast.loc[
    btc_mask,
    "Reddit_Sentiment_Lag_1",
]


forecast.loc[
    eth_mask,
    "Lagged_Log_ETH_Reddit_Post_Count",
] = forecast.loc[
    eth_mask,
    "Log_Reddit_Post_Count_Lag_1",
]


forecast.loc[
    eth_mask,
    "Lagged_ETH_Reddit_Sentiment",
] = forecast.loc[
    eth_mask,
    "Reddit_Sentiment_Lag_1",
]


print(
    "Primary BTC/ETH Reddit predictors created."
)


# =============================================================================
# 33. CREATE T-2 / T-3 / T-7 ROBUSTNESS VARIABLES
# =============================================================================

section(
    "CREATING T-2 / T-3 / T-7 ROBUSTNESS VARIABLES"
)


for lag in ROBUSTNESS_LAGS:

    # BTC activity
    forecast[
        f"Lagged_{lag}_Log_BTC_Reddit_Post_Count"
    ] = np.nan

    # BTC sentiment
    forecast[
        f"Lagged_{lag}_BTC_Reddit_Sentiment"
    ] = np.nan

    # ETH activity
    forecast[
        f"Lagged_{lag}_Log_ETH_Reddit_Post_Count"
    ] = np.nan

    # ETH sentiment
    forecast[
        f"Lagged_{lag}_ETH_Reddit_Sentiment"
    ] = np.nan


    forecast.loc[
        btc_mask,
        f"Lagged_{lag}_Log_BTC_Reddit_Post_Count",
    ] = forecast.loc[
        btc_mask,
        f"Log_Reddit_Post_Count_Lag_{lag}",
    ]


    forecast.loc[
        btc_mask,
        f"Lagged_{lag}_BTC_Reddit_Sentiment",
    ] = forecast.loc[
        btc_mask,
        f"Reddit_Sentiment_Lag_{lag}",
    ]


    forecast.loc[
        eth_mask,
        f"Lagged_{lag}_Log_ETH_Reddit_Post_Count",
    ] = forecast.loc[
        eth_mask,
        f"Log_Reddit_Post_Count_Lag_{lag}",
    ]


    forecast.loc[
        eth_mask,
        f"Lagged_{lag}_ETH_Reddit_Sentiment",
    ] = forecast.loc[
        eth_mask,
        f"Reddit_Sentiment_Lag_{lag}",
    ]


print(
    "Robustness predictors created."
)


# =============================================================================
# 34. INDEPENDENT LAG RECONSTRUCTION TEST
# =============================================================================

section(
    "INDEPENDENTLY RECONSTRUCTING REDDIT LAGS"
)


lag_validation_rows = []


for asset in ASSETS:

    asset_data = (
        forecast.loc[
            forecast[
                "Asset"
            ].eq(
                asset
            )
        ]
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    for lag in ALL_LAGS:

        expected_count = (
            asset_data[
                "Reddit_Post_Count"
            ]
            .shift(
                lag
            )
        )


        expected_log_activity = (
            asset_data[
                "Log_Reddit_Post_Count"
            ]
            .shift(
                lag
            )
        )


        expected_sentiment = (
            asset_data[
                "Reddit_Sentiment"
            ]
            .shift(
                lag
            )
        )


        expected_source_date = (
            asset_data[
                "Date"
            ]
            .shift(
                lag
            )
        )


        stored_count = (
            asset_data[
                f"Reddit_Post_Count_Lag_{lag}"
            ]
        )


        stored_log_activity = (
            asset_data[
                f"Log_Reddit_Post_Count_Lag_{lag}"
            ]
        )


        stored_sentiment = (
            asset_data[
                f"Reddit_Sentiment_Lag_{lag}"
            ]
        )


        stored_source_date = (
            asset_data[
                f"Reddit_Source_Date_Lag_{lag}"
            ]
        )


        count_match = values_equal(
            stored_count,
            expected_count,
        )


        activity_match = values_equal(
            stored_log_activity,
            expected_log_activity,
        )


        sentiment_match = values_equal(
            stored_sentiment,
            expected_sentiment,
        )


        source_date_match = (
            (
                stored_source_date
                == expected_source_date
            )
            |
            (
                stored_source_date.isna()
                &
                expected_source_date.isna()
            )
        )


        count_failures = int(
            (~count_match).sum()
        )


        activity_failures = int(
            (~activity_match).sum()
        )


        sentiment_failures = int(
            (~sentiment_match).sum()
        )


        source_date_failures = int(
            (~source_date_match).sum()
        )


        lag_validation_rows.append(
            {
                "Asset":
                    asset,

                "Lag":
                    lag,

                "Count_Mismatches":
                    count_failures,

                "Activity_Mismatches":
                    activity_failures,

                "Sentiment_Mismatches":
                    sentiment_failures,

                "Source_Date_Mismatches":
                    source_date_failures,

                "Pass":
                    (
                        count_failures == 0
                        and
                        activity_failures == 0
                        and
                        sentiment_failures == 0
                        and
                        source_date_failures == 0
                    ),
            }
        )


lag_validation = pd.DataFrame(
    lag_validation_rows
)


print(
    lag_validation.to_string(
        index=False
    )
)


if not lag_validation[
    "Pass"
].all():

    fail(
        "One or more independently reconstructed "
        "Reddit lags do not match stored values."
    )


print(
    "Independent lag reconstruction: PASS"
)


# =============================================================================
# 35. FORMAL NO-LOOK-AHEAD TEST
# =============================================================================

section(
    "FORMAL NO-LOOK-AHEAD VALIDATION"
)


lookahead_rows = []


for lag in ALL_LAGS:

    source_column = (
        f"Reddit_Source_Date_Lag_{lag}"
    )


    valid_source = (
        forecast[
            source_column
        ].notna()
    )


    # Source date must always be earlier than target date.
    future_or_same_day = (
        valid_source
        &
        (
            forecast[
                source_column
            ]
            >= forecast[
                "Date"
            ]
        )
    )


    actual_difference = (
        forecast[
            "Date"
        ]
        - forecast[
            source_column
        ]
    )


    # Exact calendar-day lag must equal requested lag.
    incorrect_calendar_gap = (
        valid_source
        &
        (
            actual_difference
            != pd.Timedelta(
                days=lag
            )
        )
    )


    lookahead_rows.append(
        {
            "Lag":
                lag,

            "Rows_With_Source_Date":
                int(
                    valid_source.sum()
                ),

            "Same_Day_Or_Future_Source":
                int(
                    future_or_same_day.sum()
                ),

            "Incorrect_Calendar_Gap":
                int(
                    incorrect_calendar_gap.sum()
                ),

            "Pass":
                (
                    int(
                        future_or_same_day.sum()
                    ) == 0
                    and
                    int(
                        incorrect_calendar_gap.sum()
                    ) == 0
                ),
        }
    )


lookahead_validation = pd.DataFrame(
    lookahead_rows
)


print(
    lookahead_validation.to_string(
        index=False
    )
)


if not lookahead_validation[
    "Pass"
].all():

    fail(
        "Accidental look-ahead was detected in "
        "one or more Reddit predictor lags."
    )


print(
    "All Reddit predictor source dates precede "
    "their forecast target dates: PASS"
)


# =============================================================================
# 36. EXPLICIT PRIMARY T-1 TIMING TEST
# =============================================================================

section(
    "VALIDATING PRIMARY T-1 FORECAST TIMING"
)


primary_source_available = (
    forecast[
        "Lagged_Reddit_Source_Date"
    ].notna()
)


primary_timing_difference = (
    forecast[
        "Date"
    ]
    - forecast[
        "Lagged_Reddit_Source_Date"
    ]
)


primary_timing_failure = (
    primary_source_available
    &
    (
        primary_timing_difference
        != pd.Timedelta(
            days=1
        )
    )
)


primary_timing_failures = int(
    primary_timing_failure.sum()
)


print(
    f"Primary t-1 timing failures: "
    f"{primary_timing_failures}"
)


if primary_timing_failures != 0:

    problem = forecast.loc[
        primary_timing_failure,
        [
            "Date",
            "Asset",
            "Lagged_Reddit_Source_Date",
        ],
    ]

    fail(
        "Primary Reddit predictor is not exactly "
        "one calendar day before target date.\n\n"
        + problem.head(
            50
        ).to_string(
            index=False
        )
    )


print(
    "Primary predictor = exact calendar t-1: PASS"
)


# =============================================================================
# 37. TEST FOR ACCIDENTAL CONTEMPORANEOUS INFORMATION
# =============================================================================

section(
    "TESTING FOR ACCIDENTAL CONTEMPORANEOUS INFORMATION"
)


same_day_source = (
    forecast[
        "Lagged_Reddit_Source_Date"
    ].notna()
    &
    forecast[
        "Lagged_Reddit_Source_Date"
    ].eq(
        forecast[
            "Date"
        ]
    )
)


same_day_source_count = int(
    same_day_source.sum()
)


print(
    "Primary predictor rows whose source date "
    "equals forecast target date:"
)

print(
    same_day_source_count
)


if same_day_source_count != 0:

    fail(
        "Contemporaneous Reddit information was "
        "mistakenly stored as a lagged predictor."
    )


print(
    "No accidental contemporaneous source dates: PASS"
)


# =============================================================================
# 38. TEST FOR FUTURE-INFORMATION CONTAMINATION
# =============================================================================

section(
    "TESTING FOR FUTURE-INFORMATION CONTAMINATION"
)


future_information_failures = 0


for lag in ALL_LAGS:

    source_col = (
        f"Reddit_Source_Date_Lag_{lag}"
    )


    failure = (
        forecast[
            source_col
        ].notna()
        &
        (
            forecast[
                source_col
            ]
            > forecast[
                "Date"
            ]
        )
    )


    count = int(
        failure.sum()
    )


    print(
        f"Lag {lag}: "
        f"future-information rows = {count}"
    )


    future_information_failures += count


if future_information_failures != 0:

    fail(
        "Future Reddit information enters one or "
        "more forecasting predictors."
    )


print(
    "No future Reddit information: PASS"
)


# =============================================================================
# 39. VALIDATE FIRST-DATE LAG MISSINGNESS
# =============================================================================

section(
    "VALIDATING START-OF-SAMPLE LAG MISSINGNESS"
)


first_day = forecast[
    "Date"
].eq(
    STUDY_START
)


if forecast.loc[
    first_day,
    "Lagged_Reddit_Source_Date",
].notna().any():

    fail(
        "A t-1 source date exists for 2021-01-01, "
        "which is outside the study's Reddit history."
    )


if forecast.loc[
    first_day,
    "Lagged_Log_Reddit_Post_Count",
].notna().any():

    fail(
        "A lagged activity predictor exists on "
        "the first study date."
    )


if forecast.loc[
    first_day,
    "Lagged_Reddit_Sentiment",
].notna().any():

    fail(
        "A lagged sentiment predictor exists on "
        "the first study date."
    )


print(
    "First-study-date lag missingness: PASS"
)


# =============================================================================
# 40. VALIDATE END-OF-SAMPLE FORECASTING LOGIC
# =============================================================================

section(
    "VALIDATING 2025-12-31 FORECASTING LOGIC"
)


# Contemporaneous Reddit information is unavailable on 2025-12-31,
# because source coverage ends on 2025-12-30.
#
# However, the 2025-12-31 target may use t-1 Reddit information
# from 2025-12-30.


final_day = forecast[
    "Date"
].eq(
    STUDY_END
)


if not forecast.loc[
    final_day,
    "Reddit_Observation_Status",
].eq(
    "OUTSIDE_REDDIT_SOURCE_COVERAGE"
).all():

    fail(
        "2025-12-31 must be outside contemporaneous "
        "Reddit source coverage for both assets."
    )


expected_final_source_date = pd.Timestamp(
    "2025-12-30"
)


if not forecast.loc[
    final_day,
    "Lagged_Reddit_Source_Date",
].eq(
    expected_final_source_date
).all():

    fail(
        "2025-12-31 t-1 predictor source date must "
        "be 2025-12-30."
    )


print(
    "2025-12-31 correctly uses 2025-12-30 "
    "as the t-1 Reddit source date: PASS"
)


# =============================================================================
# 41. CREATE FORECAST-ELIGIBILITY FLAGS
# =============================================================================

section(
    "CREATING FORECAST-ELIGIBILITY FLAGS"
)


forecast[
    "Primary_Reddit_Activity_Predictor_Available"
] = forecast[
    "Lagged_Log_Reddit_Post_Count"
].notna()


forecast[
    "Primary_Reddit_Sentiment_Predictor_Available"
] = forecast[
    "Lagged_Reddit_Sentiment"
].notna()


forecast[
    "Primary_Reddit_Both_Predictors_Available"
] = (
    forecast[
        "Primary_Reddit_Activity_Predictor_Available"
    ]
    &
    forecast[
        "Primary_Reddit_Sentiment_Predictor_Available"
    ]
)


# IMPORTANT:
#
# Do NOT permanently drop dates here.
#
# Section 08 should construct common model-specific samples so:
#
# controls only
# controls + activity
# controls + sentiment
# controls + both
#
# are compared on identical dates.


# =============================================================================
# 42. BUILD COVERAGE SUMMARY
# =============================================================================

section(
    "CREATING SECTION 06 COVERAGE SUMMARY"
)


coverage_rows = []


for asset in ASSETS:

    asset_data = forecast.loc[
        forecast[
            "Asset"
        ].eq(
            asset
        )
    ]


    status = asset_data[
        "Reddit_Observation_Status"
    ]


    coverage_rows.append(
        {
            "Asset":
                asset,

            "Calendar_Days":
                len(
                    asset_data
                ),

            "Days_With_Retained_Posts":
                int(
                    status.eq(
                        "OBSERVED_RETAINED_POSTS"
                    ).sum()
                ),

            "Raw_Posts_All_Excluded_Days":
                int(
                    status.eq(
                        "RAW_POSTS_ALL_EXCLUDED"
                    ).sum()
                ),

            "Zero_Raw_Observed_Days":
                int(
                    status.eq(
                        "GENUINE_ZERO_OBSERVED_RAW_ACTIVITY"
                    ).sum()
                ),

            "Outside_Source_Coverage_Days":
                int(
                    status.eq(
                        "OUTSIDE_REDDIT_SOURCE_COVERAGE"
                    ).sum()
                ),

            "Analytical_Activity_Available_Days":
                int(
                    asset_data[
                        "Reddit_Activity_Available"
                    ].sum()
                ),

            "Sentiment_Available_Days":
                int(
                    asset_data[
                        "Reddit_Sentiment_Available"
                    ].sum()
                ),

            "T1_Activity_Predictor_Available_Days":
                int(
                    asset_data[
                        "Primary_Reddit_Activity_Predictor_Available"
                    ].sum()
                ),

            "T1_Sentiment_Predictor_Available_Days":
                int(
                    asset_data[
                        "Primary_Reddit_Sentiment_Predictor_Available"
                    ].sum()
                ),

            "T1_Both_Predictors_Available_Days":
                int(
                    asset_data[
                        "Primary_Reddit_Both_Predictors_Available"
                    ].sum()
                ),
        }
    )


coverage_summary = pd.DataFrame(
    coverage_rows
)


print(
    coverage_summary.to_string(
        index=False
    )
)


# =============================================================================
# 43. CREATE MISSING-DAY AUDIT DATASET
# =============================================================================

section(
    "CREATING MISSING-DAY AUDIT DATASET"
)


missing_day_audit = forecast.loc[
    ~forecast[
        "Reddit_Observation_Status"
    ].eq(
        "OBSERVED_RETAINED_POSTS"
    ),
    [
        "Date",
        "Asset",
        "Raw_Reddit_Post_Count",
        "Retained_Reddit_Post_Count_Coverage",
        "Excluded_Reddit_Post_Count",
        "Reddit_Observation_Status",
        "Within_Reddit_Source_Coverage",
        "Reddit_Post_Count",
        "Log_Reddit_Post_Count",
        "Reddit_Sentiment",
    ],
].copy()


print(
    missing_day_audit.to_string(
        index=False
    )
)


# =============================================================================
# 44. FINAL STRUCTURAL VALIDATION
# =============================================================================

section(
    "FINAL STRUCTURAL VALIDATION"
)


if (
    len(forecast)
    != EXPECTED_DATE_ASSET_ROWS
):

    fail(
        "Forecast-ready dataset does not contain "
        f"{EXPECTED_DATE_ASSET_ROWS:,} rows."
    )


if forecast.duplicated(
    [
        "Date",
        "Asset",
    ]
).any():

    fail(
        "Duplicate Date × Asset rows remain."
    )


for asset in ASSETS:

    asset_dates = (
        forecast.loc[
            forecast[
                "Asset"
            ].eq(
                asset
            ),
            "Date",
        ]
        .sort_values()
        .reset_index(
            drop=True
        )
    )


    if len(
        asset_dates
    ) != EXPECTED_CALENDAR_DAYS:

        fail(
            f"{asset} does not contain "
            f"{EXPECTED_CALENDAR_DAYS:,} calendar days."
        )


    if asset_dates.iloc[
        0
    ] != STUDY_START:

        fail(
            f"{asset} calendar does not start on "
            f"{STUDY_START.date()}."
        )


    if asset_dates.iloc[
        -1
    ] != STUDY_END:

        fail(
            f"{asset} calendar does not end on "
            f"{STUDY_END.date()}."
        )


    day_gaps = (
        asset_dates
        .diff()
        .dropna()
    )


    if not (
        day_gaps
        == pd.Timedelta(
            days=1
        )
    ).all():

        fail(
            f"{asset} calendar is not consecutive "
            "by one calendar day."
        )


print(
    "Complete BTC calendar: PASS"
)

print(
    "Complete ETH calendar: PASS"
)

print(
    "Unique Date × Asset rows: PASS"
)

print(
    "Consecutive daily dates: PASS"
)

print(
    "Calendar-day lag construction: PASS"
)

print(
    "No-look-ahead validation: PASS"
)


# =============================================================================
# 45. CREATE QC SUMMARY
# =============================================================================

section(
    "CREATING SECTION 06 QC SUMMARY"
)


qc_rows = [

    {
        "Check":
            "Calendar_Days",

        "Expected":
            EXPECTED_CALENDAR_DAYS,

        "Actual":
            len(
                calendar_dates
            ),

        "Pass":
            len(
                calendar_dates
            )
            == EXPECTED_CALENDAR_DAYS,
    },


    {
        "Check":
            "Date_Asset_Rows",

        "Expected":
            EXPECTED_DATE_ASSET_ROWS,

        "Actual":
            len(
                forecast
            ),

        "Pass":
            len(
                forecast
            )
            == EXPECTED_DATE_ASSET_ROWS,
    },


    {
        "Check":
            "Stage05_Input_Rows",

        "Expected":
            EXPECTED_STAGE05_ROWS,

        "Actual":
            len(
                daily
            ),

        "Pass":
            len(
                daily
            )
            == EXPECTED_STAGE05_ROWS,
    },


    {
        "Check":
            "BTC_Stage05_Observed_Days",

        "Expected":
            EXPECTED_BTC_STAGE05_DAYS,

        "Actual":
            btc_stage05_days,

        "Pass":
            btc_stage05_days
            == EXPECTED_BTC_STAGE05_DAYS,
    },


    {
        "Check":
            "ETH_Stage05_Observed_Days",

        "Expected":
            EXPECTED_ETH_STAGE05_DAYS,

        "Actual":
            eth_stage05_days,

        "Pass":
            eth_stage05_days
            == EXPECTED_ETH_STAGE05_DAYS,
    },


    {
        "Check":
            "Stage05_Unobserved_Asset_Days",

        "Expected":
            EXPECTED_UNOBSERVED_ASSET_DAYS,

        "Actual":
            len(
                stage05_unobserved
            ),

        "Pass":
            len(
                stage05_unobserved
            )
            == EXPECTED_UNOBSERVED_ASSET_DAYS,
    },


    {
        "Check":
            "Raw_Posts_All_Excluded_Days",

        "Expected":
            8,

        "Actual":
            all_excluded_count,

        "Pass":
            all_excluded_count
            == 8,
    },


    {
        "Check":
            "Zero_Raw_Within_Source_Days",

        "Expected":
            14,

        "Actual":
            zero_raw_count,

        "Pass":
            zero_raw_count
            == 14,
    },


    {
        "Check":
            "Outside_Source_Coverage_Days",

        "Expected":
            2,

        "Actual":
            outside_coverage_count,

        "Pass":
            outside_coverage_count
            == 2,
    },


    {
        "Check":
            "Unresolved_Missing_Statuses",

        "Expected":
            0,

        "Actual":
            unresolved_count,

        "Pass":
            unresolved_count
            == 0,
    },


    {
        "Check":
            "Lag_Reconstruction_Tests",

        "Expected":
            True,

        "Actual":
            bool(
                lag_validation[
                    "Pass"
                ].all()
            ),

        "Pass":
            bool(
                lag_validation[
                    "Pass"
                ].all()
            ),
    },


    {
        "Check":
            "No_Lookahead_Tests",

        "Expected":
            True,

        "Actual":
            bool(
                lookahead_validation[
                    "Pass"
                ].all()
            ),

        "Pass":
            bool(
                lookahead_validation[
                    "Pass"
                ].all()
            ),
    },


    {
        "Check":
            "Primary_T1_Timing_Failures",

        "Expected":
            0,

        "Actual":
            primary_timing_failures,

        "Pass":
            primary_timing_failures
            == 0,
    },


    {
        "Check":
            "Future_Information_Failures",

        "Expected":
            0,

        "Actual":
            future_information_failures,

        "Pass":
            future_information_failures
            == 0,
    },


    {
        "Check":
            "Same_Day_Source_Failures",

        "Expected":
            0,

        "Actual":
            same_day_source_count,

        "Pass":
            same_day_source_count
            == 0,
    },
]


qc = pd.DataFrame(
    qc_rows
)


print(
    qc.to_string(
        index=False
    )
)


if not qc[
    "Pass"
].all():

    fail(
        "At least one final Section 06 QC "
        "check failed."
    )


# =============================================================================
# 46. ORDER IMPORTANT OUTPUT COLUMNS
# =============================================================================

section(
    "ORDERING FORECAST-READY OUTPUT"
)


leading_columns = [

    "Date",
    "Asset",

    # Source coverage
    "Within_Reddit_Source_Coverage",
    "Outside_Reddit_Source_Coverage",
    "Reddit_Observation_Status",

    # Raw / retained coverage
    "Raw_Reddit_Post_Count",
    "Retained_Reddit_Post_Count_Coverage",
    "Excluded_Reddit_Post_Count",

    # Final contemporaneous analytical Reddit variables
    "Reddit_Post_Count",
    "Log_Reddit_Post_Count",
    "Reddit_Sentiment",

    # Availability
    "Reddit_Activity_Available",
    "Reddit_Sentiment_Available",
    "Reddit_Has_Retained_Posts",
    "Reddit_Is_Zero_Analytical_Activity",

    # Primary source date
    "Lagged_Reddit_Source_Date",

    # Primary generic t-1 predictors
    "Lagged_Reddit_Post_Count",
    "Lagged_Log_Reddit_Post_Count",
    "Lagged_Reddit_Sentiment",

    # Primary asset-specific predictors
    "Lagged_Log_BTC_Reddit_Post_Count",
    "Lagged_BTC_Reddit_Sentiment",
    "Lagged_Log_ETH_Reddit_Post_Count",
    "Lagged_ETH_Reddit_Sentiment",

    # Predictor availability
    "Primary_Reddit_Activity_Predictor_Available",
    "Primary_Reddit_Sentiment_Predictor_Available",
    "Primary_Reddit_Both_Predictors_Available",
]


remaining_columns = [
    column
    for column in forecast.columns
    if column not in leading_columns
]


forecast = forecast[
    leading_columns
    + remaining_columns
]


# =============================================================================
# 47. SAVE OUTPUT FILES
# =============================================================================

section(
    "SAVING SECTION 06 OUTPUTS"
)


forecast.to_csv(
    FORECAST_READY_FILE,
    index=False,
)


missing_day_audit.to_csv(
    MISSING_CLASSIFICATION_FILE,
    index=False,
)


coverage_summary.to_csv(
    COVERAGE_SUMMARY_FILE,
    index=False,
)


lag_validation.to_csv(
    LAG_VALIDATION_FILE,
    index=False,
)


lookahead_validation.to_csv(
    LOOKAHEAD_VALIDATION_FILE,
    index=False,
)


qc.to_csv(
    QC_FILE,
    index=False,
)


# =============================================================================
# 48. SAVE METHODOLOGY NOTE
# =============================================================================

methodology = f"""
SECTION 06 — CALENDAR, MISSING DAYS AND LAGS
=============================================

Study period
------------
{STUDY_START.date()} to {STUDY_END.date()}

Assets
------
Bitcoin (BTC)
Ethereum (ETH)

Study calendar
--------------
A complete calendar containing every calendar day between the start and
end of the study period is constructed separately for BTC and ETH.

The resulting dataset therefore contains:

    {EXPECTED_CALENDAR_DAYS:,} calendar days
    {EXPECTED_DATE_ASSET_ROWS:,} Date × Asset observations

Constructing the complete calendar before generating lags ensures that
t-1 represents exactly the preceding calendar day rather than the
previous available Reddit observation.

Reddit sentiment interpretation
--------------------------------
The Reddit sentiment measures represent textual sentiment among the
selected Reddit communities used in the dissertation. They should not
be described as general investor sentiment.

Daily sentiment
---------------
The post-level continuous sentiment score is:

    S_i = P(Positive_i) - P(Negative_i)

For each Date × Asset, daily sentiment is the unweighted arithmetic mean
of the post-level continuous sentiment scores across retained posts.

Reddit scores, upvotes, upvote ratios and comment counts are not used
as sentiment weights.

Daily activity
--------------
Daily Reddit activity is defined using the retained analytical post count:

    N_(a,t)

The modelling transformation is:

    Activity_(a,t) = log(1 + N_(a,t))

Sentiment and activity remain separate predictors.

Missing versus zero activity
----------------------------
Missing Reddit observations are not automatically assigned zero.

The validated raw-versus-retained coverage information is used to
distinguish four situations.

1. OBSERVED_RETAINED_POSTS

At least one Reddit post survives the analytical cleaning criteria.

Daily sentiment and retained-post activity are observed.

2. RAW_POSTS_ALL_EXCLUDED

Raw Reddit posts existed but every raw post was excluded by the
documented analytical cleaning rules.

For retained analytical activity:

    retained post count = 0
    log(1 + retained post count) = 0

Daily sentiment remains missing because no eligible post remains from
which sentiment can be calculated.

3. GENUINE_ZERO_OBSERVED_RAW_ACTIVITY

The date lies inside confirmed Reddit source coverage and the validated
raw extraction contains zero observed posts for the selected asset
communities.

For retained analytical activity:

    retained post count = 0
    log(1 + retained post count) = 0

Sentiment remains missing.

This should be interpreted as no raw posts being observed among the
selected Reddit communities in the extraction, rather than as proof of
no social-media discussion more generally.

4. OUTSIDE_REDDIT_SOURCE_COVERAGE

The analytical study period ends on 2025-12-31, but confirmed Reddit
source coverage ends on {REDDIT_SOURCE_END.date()}.

Consequently, 2025-12-31 is treated as unavailable for contemporaneous
Reddit sentiment and activity for BTC and ETH rather than as a
zero-activity day.

Forecast timing
---------------
The primary Reddit predictor uses information from calendar day t-1 to
predict the cryptocurrency return on calendar day t.

Therefore:

    Predictor_Source_Date = Target_Date - 1 calendar day

Reddit information from target day t is not used as a predictor of the
return on target day t.

Robustness lags
---------------
In addition to t-1, the following lag lengths are constructed:

    t-2
    t-3
    t-7

These provide lag-length robustness tests.

No-look-ahead validation
------------------------
Each lagged Reddit predictor is independently reconstructed after the
complete calendar is created.

For every available lagged predictor, Section 06 verifies:

    Source_Date < Target_Date

and:

    Target_Date - Source_Date = requested calendar lag

The script also tests explicitly for same-day and future-information
contamination.

Traditional financial-market variables
--------------------------------------
Traditional controls are not merged in Section 06.

S&P 500 returns, VIX changes, gold returns, DXY returns and US 10-year
Treasury yield changes operate on market calendars that differ from
the 24/7 cryptocurrency calendar.

They must therefore be aligned in Section 07 using only information
available before the cryptocurrency forecast target.

Cryptocurrency trading volume
-----------------------------
Cryptocurrency trading volume is also not incorporated in Section 06.

The dissertation uses Yahoo Finance-reported daily trading volume,
transformed using log(1 + volume).

For predictive specifications, lagged cryptocurrency volume should be
used rather than contemporaneous target-day volume.

Future model specifications
---------------------------
The forecast-ready Reddit variables support separate specifications:

    1. controls only
    2. controls + Reddit activity
    3. controls + Reddit sentiment
    4. controls + Reddit activity + Reddit sentiment

This separation allows the contribution of Reddit attention/activity
to be distinguished from the contribution of sentiment tone.

Explanatory versus predictive analysis
--------------------------------------
Section 06 does not estimate regressions or forecasting performance.

Later explanatory regressions should use appropriate robust inference,
including HAC/Newey-West standard errors.

Predictive performance must be assessed using genuinely held-out
observations and one-step-ahead forecasting.

Forecasting comparisons should use identical forecast dates for the
benchmark and Reddit-extended models and should report predictive
metrics such as RMSE, MAE and out-of-sample R-squared.

BTC versus ETH coefficient comparison
--------------------------------------
A later comparison of BTC and ETH sentiment effects should use a formal
coefficient-difference test, such as an asset × sentiment interaction
and Wald test.

A significant BTC coefficient and insignificant ETH coefficient do not
by themselves establish that the two coefficients differ statistically.
"""


METHODOLOGY_FILE.write_text(
    methodology.strip(),
    encoding="utf-8",
)


# =============================================================================
# 49. FINAL OUTPUT RELOAD VALIDATION
# =============================================================================

section(
    "FINAL OUTPUT RELOAD VALIDATION"
)


reloaded = pd.read_csv(
    FORECAST_READY_FILE,
    low_memory=False,
)


if (
    len(reloaded)
    != EXPECTED_DATE_ASSET_ROWS
):

    fail(
        "Reloaded forecast-ready file has an "
        "unexpected row count."
    )


if reloaded.duplicated(
    [
        "Date",
        "Asset",
    ]
).any():

    fail(
        "Reloaded forecast-ready file contains "
        "duplicate Date × Asset rows."
    )


print(
    f"Reloaded rows: "
    f"{len(reloaded):,}"
)

print(
    "Final reload validation: PASS"
)


# =============================================================================
# 50. FINAL SUMMARY
# =============================================================================

section(
    "SECTION 06 COMPLETE — FORECAST-READY REDDIT PREDICTORS"
)


print(
    f"""
Study period:
    {STUDY_START.date()} to {STUDY_END.date()}

Confirmed Reddit source period:
    {REDDIT_SOURCE_START.date()} to {REDDIT_SOURCE_END.date()}

Calendar days:
    {EXPECTED_CALENDAR_DAYS:,}

Assets:
    BTC
    ETH

Date × Asset rows:
    {len(forecast):,}

Section 05 observed rows:
    {len(daily):,}

BTC Section 05 observed days:
    {btc_stage05_days:,}

ETH Section 05 observed days:
    {eth_stage05_days:,}

Section 05 unobserved asset-days:
    {len(stage05_unobserved):,}

Missing-day classification:
    OBSERVED_RETAINED_POSTS
    RAW_POSTS_ALL_EXCLUDED
    GENUINE_ZERO_OBSERVED_RAW_ACTIVITY
    OUTSIDE_REDDIT_SOURCE_COVERAGE

All-excluded Date × Asset rows:
    {all_excluded_count:,}

Zero-raw-activity Date × Asset rows:
    {zero_raw_count:,}

Outside-source-coverage rows:
    {outside_coverage_count:,}

Primary forecasting lag:
    t-1 calendar day

Robustness lags:
    t-2
    t-3
    t-7

Primary BTC predictors:
    Lagged_BTC_Reddit_Sentiment
    Lagged_Log_BTC_Reddit_Post_Count

Primary ETH predictors:
    Lagged_ETH_Reddit_Sentiment
    Lagged_Log_ETH_Reddit_Post_Count

No-look-ahead validation:
    PASS

Future-information validation:
    PASS

Contemporaneous-source validation:
    PASS

Section 06 output directory:
    {OUTPUT_DIR}

Main forecast-ready output:
    {FORECAST_READY_FILE}

Next stage:
    Section 07 — merge these forecast-safe Reddit predictors with the
    information-aligned cryptocurrency and traditional-market dataset.

IMPORTANT:
    Section 07 must preserve the forecasting timing rule.

    Same-day Reddit variables must not be used as evidence of
    one-day-ahead predictive information.

    Same-day cryptocurrency volume must not be used as a predictive
    control.

    Traditional-market variables must use information that was actually
    available before the cryptocurrency return forecast target.
"""
)


print(
    "SECTION 06: PASS"
)