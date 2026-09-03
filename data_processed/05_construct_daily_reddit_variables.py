"""
===============================================================================
05 — DAILY REDDIT VARIABLE CONSTRUCTION
===============================================================================

Purpose
-------
Convert the final Stage 04 post-level Reddit sentiment dataset into daily
BTC/ETH Reddit sentiment and activity variables.

Study design
------------
Study period:
    2021-01-01 to 2025-12-31

Assets:
    BTC and ETH

Unit of aggregation:
    date × asset

Primary daily sentiment measure:
    Arithmetic mean of post-level continuous sentiment, where:

        post sentiment = P(Positive) - P(Negative)

Reddit activity:
    1. Daily retained post count
    2. log(1 + daily retained post count)

Important methodological decisions
----------------------------------
1. Sentiment and activity are constructed separately.
2. Sentiment is NOT weighted by Reddit score, upvotes or comments.
3. Raw post count is retained descriptively.
4. log(1 + post count) is the primary modelling activity variable.
5. No cryptocurrency or traditional-market variables are merged here.
6. No forecasting lags are created here.
7. Days without observed Reddit posts are NOT automatically assigned
   sentiment = 0.
8. Stage 05 identifies unobserved asset-days but does not yet decide whether
   each represents a genuine zero-post day or unavailable source coverage.
9. Calendar completion and source-coverage classification belong to Stage 06.
10. Aggregation is independently validated against the Stage 04 post-level data.
11. Stage 05 does not modify Stage 04 post-level sentiment scores.

Expected Stage 04 input
-----------------------
Rows:
    136,019

BTC:
    117,086

ETH:
    18,933

Expected input file
-------------------
data_clean/reddit/stage04_sentiment/reddit_post_level_sentiment.csv

Outputs
-------
data_clean/reddit/stage05_daily_reddit/

    reddit_daily_sentiment_activity.csv
    reddit_daily_sentiment_activity_wide.csv
    reddit_daily_sentiment_by_asset.csv
    reddit_daily_sentiment_by_year.csv
    reddit_daily_sentiment_by_year_asset.csv
    reddit_daily_activity_distribution.csv
    reddit_daily_aggregation_validation.csv
    reddit_daily_coverage.csv
    reddit_daily_unobserved_asset_days.csv
    reddit_daily_stage05_qc.csv
    reddit_daily_methodology_note.txt

===============================================================================
"""

from pathlib import Path

import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

# This script is intended to live in:
#
# data_processed/05_construct_daily_reddit_variables.py

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
    / "stage04_sentiment"
    / "reddit_post_level_sentiment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
    / "stage05_daily_reddit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =============================================================================
# 2. STUDY DESIGN AND FROZEN STAGE 04 EXPECTATIONS
# =============================================================================

STUDY_START = pd.Timestamp("2021-01-01")
STUDY_END = pd.Timestamp("2025-12-31")

FULL_CALENDAR = pd.date_range(
    STUDY_START,
    STUDY_END,
    freq="D"
)

EXPECTED_CALENDAR_DAYS = 1826

EXPECTED_ASSETS = {
    "BTC",
    "ETH"
}

EXPECTED_INPUT_ROWS = 136_019

EXPECTED_POSTS_BY_ASSET = {
    "BTC": 117_086,
    "ETH": 18_933
}

EXPECTED_SENTIMENT_LABELS = {
    "negative",
    "neutral",
    "positive"
}

PRIMARY_SENTIMENT_DEFINITION = (
    "Mean post-level continuous sentiment, where "
    "post-level sentiment = P(Positive) - P(Negative)."
)

ACTIVITY_DEFINITION = (
    "log(1 + daily retained Reddit post count)"
)


# =============================================================================
# 3. HELPER FUNCTIONS
# =============================================================================

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def fail(message):
    raise ValueError(
        "\nERROR:\n" + message
    )


def find_column(df, candidates, description):
    """
    Find a required column using an ordered list of acceptable names.

    The first matching candidate is used.
    """

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    fail(
        f"Could not identify {description}.\n"
        f"Accepted column names:\n{candidates}\n\n"
        f"Available columns:\n{list(df.columns)}"
    )


def safe_float(series):
    """
    Convert a Series to numeric.

    Invalid values become NaN and are subsequently caught by explicit
    validation.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def check_missing(series, description):
    """
    Fail if a required source column contains missing values.

    This check is deliberately performed before converting categorical
    columns to strings so that genuine NaN values are not converted to
    the literal string 'nan'.
    """

    missing_count = int(
        series.isna().sum()
    )

    print(
        f"{description} missing values: "
        f"{missing_count:,}"
    )

    if missing_count != 0:
        fail(
            f"{missing_count:,} missing values found in "
            f"{description}."
        )


# =============================================================================
# 4. START
# =============================================================================

section(
    "05 — DAILY REDDIT VARIABLE CONSTRUCTION"
)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nInput file:")
print(INPUT_FILE)

print("\nOutput directory:")
print(OUTPUT_DIR)


# =============================================================================
# 5. CHECK INPUT FILE
# =============================================================================

section(
    "CHECKING INPUT FILE"
)

if not INPUT_FILE.exists():
    fail(
        "The final Stage 04 post-level sentiment file does not exist:\n"
        f"{INPUT_FILE}"
    )

print("Input exists: True")


# =============================================================================
# 6. IMPORT FINAL STAGE 04 POST-LEVEL DATA
# =============================================================================

section(
    "IMPORTING STAGE 04 POST-LEVEL SENTIMENT DATA"
)

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nRows loaded:")
print(f"{len(df):,}")

print("\nColumns:")
print(list(df.columns))


# =============================================================================
# 7. HARD INPUT-ROW VALIDATION
# =============================================================================

section(
    "VALIDATING FROZEN STAGE 04 INPUT SIZE"
)

print(
    f"\nExpected Stage 04 rows: "
    f"{EXPECTED_INPUT_ROWS:,}"
)

print(
    f"Observed Stage 04 rows: "
    f"{len(df):,}"
)

if len(df) != EXPECTED_INPUT_ROWS:
    fail(
        "Unexpected Stage 04 input size.\n\n"
        f"Expected: {EXPECTED_INPUT_ROWS:,}\n"
        f"Found:    {len(df):,}\n\n"
        "Stage 05 must use the frozen final Stage 04 "
        "post-level sentiment dataset."
    )

print("\nStage 04 input-row validation: PASS")


# =============================================================================
# 8. IDENTIFY REQUIRED COLUMNS
# =============================================================================

section(
    "IDENTIFYING REQUIRED COLUMNS"
)

post_id_col = find_column(
    df,
    [
        "post_id",
        "id"
    ],
    "post ID"
)

date_col = find_column(
    df,
    [
        "post_date",
        "date",
        "Date"
    ],
    "post date"
)

asset_col = find_column(
    df,
    [
        "asset",
        "Asset"
    ],
    "asset"
)

subreddit_col = find_column(
    df,
    [
        "subreddit",
        "Subreddit"
    ],
    "subreddit"
)

sentiment_col = find_column(
    df,
    [
        "continuous_sentiment",
        "sentiment_score",
        "sentiment_continuous",
        "Continuous_Sentiment",
        "Sentiment_Score"
    ],
    "continuous sentiment score"
)

label_col = find_column(
    df,
    [
        "sentiment_label",
        "model_label",
        "predicted_label",
        "label",
        "Sentiment_Label"
    ],
    "sentiment label"
)

print("\nIdentified columns:")
print(f"Post ID:              {post_id_col}")
print(f"Date:                 {date_col}")
print(f"Asset:                {asset_col}")
print(f"Subreddit:            {subreddit_col}")
print(f"Continuous sentiment: {sentiment_col}")
print(f"Sentiment label:      {label_col}")


# =============================================================================
# 9. VALIDATE REQUIRED SOURCE VALUES BEFORE STRING CONVERSION
# =============================================================================

section(
    "VALIDATING REQUIRED SOURCE VALUES"
)

check_missing(
    df[post_id_col],
    "post ID"
)

check_missing(
    df[date_col],
    "post date"
)

check_missing(
    df[asset_col],
    "asset"
)

check_missing(
    df[subreddit_col],
    "subreddit"
)

check_missing(
    df[sentiment_col],
    "continuous sentiment"
)

check_missing(
    df[label_col],
    "sentiment label"
)

print(
    "\nRequired source-value validation: PASS"
)


# =============================================================================
# 10. STANDARDISE WORKING COLUMNS
# =============================================================================

section(
    "STANDARDISING WORKING COLUMNS"
)

working = pd.DataFrame(
    {
        "post_id":
            df[post_id_col],

        "post_date":
            df[date_col],

        "asset":
            df[asset_col],

        "subreddit":
            df[subreddit_col],

        "continuous_sentiment":
            safe_float(
                df[sentiment_col]
            ),

        "sentiment_label":
            (
                df[label_col]
                .astype(str)
                .str.strip()
                .str.lower()
            )
    }
)


# =============================================================================
# 11. DATE VALIDATION
# =============================================================================

section(
    "VALIDATING POST DATES"
)

working["post_date"] = pd.to_datetime(
    working["post_date"],
    errors="coerce"
)

invalid_dates = int(
    working["post_date"]
    .isna()
    .sum()
)

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates != 0:
    fail(
        f"{invalid_dates:,} rows have invalid post dates."
    )

working["post_date"] = (
    working["post_date"]
    .dt.normalize()
)

outside_period = (
    (working["post_date"] < STUDY_START)
    |
    (working["post_date"] > STUDY_END)
)

outside_count = int(
    outside_period.sum()
)

print("\nObservations outside study period:")
print(outside_count)

if outside_count != 0:
    fail(
        f"{outside_count:,} observations fall outside "
        f"{STUDY_START.date()} to {STUDY_END.date()}."
    )

if len(FULL_CALENDAR) != EXPECTED_CALENDAR_DAYS:
    fail(
        "Unexpected calendar length.\n"
        f"Expected: {EXPECTED_CALENDAR_DAYS:,}\n"
        f"Found:    {len(FULL_CALENDAR):,}"
    )

print(
    f"\nStudy calendar contains "
    f"{len(FULL_CALENDAR):,} days."
)

print("\nDate validation: PASS")


# =============================================================================
# 12. POST ID VALIDATION
# =============================================================================

section(
    "VALIDATING POST IDENTIFIERS"
)

missing_post_ids = int(
    working["post_id"]
    .isna()
    .sum()
)

duplicate_post_ids = int(
    working["post_id"]
    .duplicated()
    .sum()
)

print("\nMissing post IDs:")
print(missing_post_ids)

print("\nDuplicate post IDs:")
print(duplicate_post_ids)

if missing_post_ids != 0:
    fail(
        "Missing post IDs found."
    )

if duplicate_post_ids != 0:
    fail(
        "Duplicate post IDs found."
    )

print("\nPost-ID validation: PASS")


# =============================================================================
# 13. ASSET VALIDATION
# =============================================================================

section(
    "VALIDATING BTC/ETH ASSET CLASSIFICATION"
)

working["asset"] = (
    working["asset"]
    .astype(str)
    .str.strip()
    .str.upper()
)

asset_values = set(
    working["asset"]
    .unique()
)

print("\nAssets found:")
print(sorted(asset_values))

unexpected_assets = (
    asset_values
    - EXPECTED_ASSETS
)

missing_expected_assets = (
    EXPECTED_ASSETS
    - asset_values
)

print("\nUnexpected assets:")
print(sorted(unexpected_assets))

print("\nMissing expected assets:")
print(sorted(missing_expected_assets))

if unexpected_assets:
    fail(
        f"Unexpected asset values found: "
        f"{sorted(unexpected_assets)}"
    )

if missing_expected_assets:
    fail(
        f"Expected asset values are missing: "
        f"{sorted(missing_expected_assets)}"
    )


# =============================================================================
# 14. VALIDATE FROZEN BTC/ETH POST COUNTS
# =============================================================================

section(
    "VALIDATING STAGE 04 BTC/ETH POST COUNTS"
)

actual_posts_by_asset = (
    working["asset"]
    .value_counts()
    .to_dict()
)

for asset in ["BTC", "ETH"]:

    actual_count = int(
        actual_posts_by_asset.get(
            asset,
            0
        )
    )

    expected_count = int(
        EXPECTED_POSTS_BY_ASSET[asset]
    )

    print(
        f"\n{asset}: "
        f"{actual_count:,} observed / "
        f"{expected_count:,} expected"
    )

    if actual_count != expected_count:
        fail(
            f"Unexpected Stage 04 post count for {asset}.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {actual_count:,}"
        )

if sum(actual_posts_by_asset.values()) != EXPECTED_INPUT_ROWS:
    fail(
        "Asset counts do not reconstruct the "
        "expected Stage 04 total."
    )

print(
    "\nFrozen Stage 04 asset-count validation: PASS"
)


# =============================================================================
# 15. SUBREDDIT VALIDATION
# =============================================================================

section(
    "VALIDATING SUBREDDIT INFORMATION"
)

working["subreddit"] = (
    working["subreddit"]
    .astype(str)
    .str.strip()
)

empty_subreddits = int(
    working["subreddit"]
    .eq("")
    .sum()
)

print("\nEmpty subreddit strings:")
print(empty_subreddits)

if empty_subreddits != 0:
    fail(
        f"{empty_subreddits:,} empty subreddit values found."
    )

print("\nSubreddits:")
print(
    working["subreddit"]
    .value_counts()
    .to_string()
)


# =============================================================================
# 16. SENTIMENT SCORE VALIDATION
# =============================================================================

section(
    "VALIDATING CONTINUOUS SENTIMENT"
)

missing_sentiment = int(
    working["continuous_sentiment"]
    .isna()
    .sum()
)

print("\nMissing sentiment scores:")
print(missing_sentiment)

if missing_sentiment != 0:
    fail(
        "Missing or non-numeric continuous sentiment scores found."
    )

sentiment_min = float(
    working["continuous_sentiment"]
    .min()
)

sentiment_max = float(
    working["continuous_sentiment"]
    .max()
)

print("\nMinimum sentiment:")
print(sentiment_min)

print("\nMaximum sentiment:")
print(sentiment_max)

if sentiment_min < -1 - 1e-12:
    fail(
        "Continuous sentiment contains values below -1."
    )

if sentiment_max > 1 + 1e-12:
    fail(
        "Continuous sentiment contains values above +1."
    )

print("\nContinuous sentiment validation: PASS")


# =============================================================================
# 17. SENTIMENT LABEL VALIDATION
# =============================================================================

section(
    "VALIDATING SENTIMENT LABELS"
)

observed_labels = set(
    working["sentiment_label"]
    .unique()
)

print("\nLabels found:")
print(sorted(observed_labels))

unexpected_labels = (
    observed_labels
    - EXPECTED_SENTIMENT_LABELS
)

missing_labels = (
    EXPECTED_SENTIMENT_LABELS
    - observed_labels
)

if unexpected_labels:
    fail(
        f"Unexpected sentiment labels found: "
        f"{sorted(unexpected_labels)}"
    )

if missing_labels:
    fail(
        f"Expected sentiment labels are absent: "
        f"{sorted(missing_labels)}"
    )

print("\nSentiment-label validation: PASS")


# =============================================================================
# 18. VERIFY PRIMARY SENTIMENT DEFINITION
# =============================================================================

section(
    "VERIFYING SENTIMENT MEASURE DEFINITION"
)

print("\nPrimary sentiment measure:")
print(PRIMARY_SENTIMENT_DEFINITION)

print(
    "\nDaily sentiment will be the unweighted arithmetic "
    "mean of post-level continuous sentiment scores."
)

print(
    "\nNo Reddit score, upvote ratio or comment count "
    "is used as a sentiment weight."
)


# =============================================================================
# 19. CREATE CATEGORICAL INDICATORS
# =============================================================================

section(
    "CREATING SENTIMENT CATEGORY INDICATORS"
)

working["is_positive"] = (
    working["sentiment_label"]
    == "positive"
).astype(int)

working["is_neutral"] = (
    working["sentiment_label"]
    == "neutral"
).astype(int)

working["is_negative"] = (
    working["sentiment_label"]
    == "negative"
).astype(int)


# =============================================================================
# 20. POST-LEVEL LABEL DISTRIBUTION
# =============================================================================

section(
    "POST-LEVEL SENTIMENT LABEL DISTRIBUTION"
)

label_distribution = (
    working["sentiment_label"]
    .value_counts()
    .rename_axis("Sentiment_Label")
    .reset_index(
        name="Post_Count"
    )
)

label_distribution["Post_Share"] = (
    label_distribution["Post_Count"]
    / len(working)
)

print(
    label_distribution.to_string(
        index=False
    )
)


# =============================================================================
# 21. DAILY × ASSET AGGREGATION
# =============================================================================

section(
    "AGGREGATING REDDIT DATA BY DATE × ASSET"
)

daily = (
    working
    .groupby(
        [
            "post_date",
            "asset"
        ],
        as_index=False
    )
    .agg(
        Post_Count=(
            "post_id",
            "count"
        ),

        Mean_Reddit_Sentiment=(
            "continuous_sentiment",
            "mean"
        ),

        Median_Reddit_Sentiment=(
            "continuous_sentiment",
            "median"
        ),

        Std_Reddit_Sentiment=(
            "continuous_sentiment",
            "std"
        ),

        Positive_Post_Count=(
            "is_positive",
            "sum"
        ),

        Neutral_Post_Count=(
            "is_neutral",
            "sum"
        ),

        Negative_Post_Count=(
            "is_negative",
            "sum"
        )
    )
)

print(
    f"\nObserved date × asset rows created: "
    f"{len(daily):,}"
)


# =============================================================================
# 22. DAILY SENTIMENT SHARES
# =============================================================================

section(
    "CALCULATING DAILY SENTIMENT SHARES"
)

daily["Positive_Share"] = (
    daily["Positive_Post_Count"]
    / daily["Post_Count"]
)

daily["Neutral_Share"] = (
    daily["Neutral_Post_Count"]
    / daily["Post_Count"]
)

daily["Negative_Share"] = (
    daily["Negative_Post_Count"]
    / daily["Post_Count"]
)


# =============================================================================
# 23. DAILY REDDIT ACTIVITY
# =============================================================================

section(
    "CONSTRUCTING DAILY REDDIT ACTIVITY"
)

daily["Log_Reddit_Post_Count"] = np.log1p(
    daily["Post_Count"]
)

print("\nActivity definition:")
print(ACTIVITY_DEFINITION)


# =============================================================================
# 24. VALIDATE DAILY SENTIMENT SHARES
# =============================================================================

section(
    "VALIDATING DAILY SENTIMENT SHARES"
)

share_sum = (
    daily["Positive_Share"]
    + daily["Neutral_Share"]
    + daily["Negative_Share"]
)

share_error = float(
    (share_sum - 1.0)
    .abs()
    .max()
)

print("\nMaximum share-sum error:")
print(share_error)

if share_error > 1e-12:
    fail(
        "Daily sentiment shares do not sum to one."
    )

print("\nDaily sentiment-share validation: PASS")


# =============================================================================
# 25. VALIDATE DAILY POST COUNT RECONSTRUCTION
# =============================================================================

section(
    "VALIDATING DAILY POST COUNT RECONSTRUCTION"
)

reconstructed_daily_count = (
    daily["Positive_Post_Count"]
    + daily["Neutral_Post_Count"]
    + daily["Negative_Post_Count"]
)

count_mismatches = int(
    (
        reconstructed_daily_count
        != daily["Post_Count"]
    )
    .sum()
)

print("\nDaily count mismatches:")
print(count_mismatches)

if count_mismatches != 0:
    fail(
        "Sentiment category counts do not reconstruct "
        "the total daily post count."
    )

print("\nDaily category-count reconstruction: PASS")


# =============================================================================
# 26. VALIDATE LOG TRANSFORMATION
# =============================================================================

section(
    "VALIDATING log(1 + POST COUNT)"
)

expected_log_activity = np.log1p(
    daily["Post_Count"]
)

log_difference = float(
    (
        daily["Log_Reddit_Post_Count"]
        - expected_log_activity
    )
    .abs()
    .max()
)

print("\nMaximum transformation difference:")
print(log_difference)

if log_difference > 1e-12:
    fail(
        "log(1 + Post_Count) transformation failed."
    )

print("\nLog-activity validation: PASS")


# =============================================================================
# 27. SORT DAILY DATA
# =============================================================================

daily = (
    daily
    .sort_values(
        [
            "post_date",
            "asset"
        ]
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 28. REORDER DAILY COLUMNS
# =============================================================================

daily = daily[
    [
        "post_date",
        "asset",
        "Post_Count",
        "Log_Reddit_Post_Count",
        "Mean_Reddit_Sentiment",
        "Median_Reddit_Sentiment",
        "Std_Reddit_Sentiment",
        "Positive_Post_Count",
        "Neutral_Post_Count",
        "Negative_Post_Count",
        "Positive_Share",
        "Neutral_Share",
        "Negative_Share"
    ]
]


# =============================================================================
# 29. GLOBAL POST-LEVEL → DAILY RECONSTRUCTION
# =============================================================================

section(
    "VALIDATING GLOBAL POST-LEVEL → DAILY RECONSTRUCTION"
)

total_daily_posts = int(
    daily["Post_Count"]
    .sum()
)

print(
    f"\nPost-level rows: "
    f"{len(working):,}"
)

print(
    f"Posts represented by daily aggregation: "
    f"{total_daily_posts:,}"
)

if total_daily_posts != EXPECTED_INPUT_ROWS:
    fail(
        "Global aggregation count mismatch.\n"
        f"Expected:   {EXPECTED_INPUT_ROWS:,}\n"
        f"Aggregated: {total_daily_posts:,}"
    )

if total_daily_posts != len(working):
    fail(
        "Daily aggregation does not reconstruct "
        "the post-level working dataset."
    )

print(
    f"\nGlobal post-count reconstruction: PASS "
    f"({total_daily_posts:,}/{EXPECTED_INPUT_ROWS:,})"
)


# =============================================================================
# 30. ASSET-SPECIFIC POST-LEVEL → DAILY VALIDATION
# =============================================================================

section(
    "VALIDATING AGGREGATION AGAINST POST-LEVEL DATA"
)

validation_rows = []

for asset in ["BTC", "ETH"]:

    post_subset = (
        working[
            working["asset"] == asset
        ]
    )

    daily_subset = (
        daily[
            daily["asset"] == asset
        ]
    )

    post_count_total = len(
        post_subset
    )

    daily_count_total = int(
        daily_subset["Post_Count"]
        .sum()
    )

    post_days = int(
        post_subset["post_date"]
        .nunique()
    )

    daily_days = int(
        daily_subset["post_date"]
        .nunique()
    )

    overall_post_mean = float(
        post_subset[
            "continuous_sentiment"
        ]
        .mean()
    )

    weighted_daily_mean = float(
        (
            daily_subset[
                "Mean_Reddit_Sentiment"
            ]
            * daily_subset[
                "Post_Count"
            ]
        )
        .sum()
        / daily_count_total
    )

    count_difference = (
        daily_count_total
        - post_count_total
    )

    mean_difference = (
        weighted_daily_mean
        - overall_post_mean
    )

    expected_asset_count = (
        EXPECTED_POSTS_BY_ASSET[asset]
    )

    validation_pass = (
        count_difference == 0
        and abs(mean_difference) < 1e-10
        and post_days == daily_days
        and post_count_total == expected_asset_count
    )

    validation_rows.append(
        {
            "Asset":
                asset,

            "Expected_Post_Count":
                expected_asset_count,

            "Post_Level_Count":
                post_count_total,

            "Daily_Aggregated_Count":
                daily_count_total,

            "Count_Difference":
                count_difference,

            "Post_Level_Days":
                post_days,

            "Daily_Aggregated_Days":
                daily_days,

            "Post_Level_Overall_Mean_Sentiment":
                overall_post_mean,

            "Reconstructed_Overall_Mean_From_Daily":
                weighted_daily_mean,

            "Mean_Difference":
                mean_difference,

            "Validation_Pass":
                validation_pass
        }
    )

aggregation_validation = pd.DataFrame(
    validation_rows
)

print(
    aggregation_validation.to_string(
        index=False
    )
)

if not aggregation_validation[
    "Validation_Pass"
].all():
    fail(
        "Post-level to daily aggregation validation failed."
    )

print(
    "\nAsset-specific aggregation validation: PASS"
)


# =============================================================================
# 31. DAILY COVERAGE
# =============================================================================

section(
    "CONSTRUCTING DAILY REDDIT COVERAGE"
)

coverage_rows = []

for asset in ["BTC", "ETH"]:

    asset_daily = (
        daily[
            daily["asset"] == asset
        ]
    )

    observed_dates = set(
        asset_daily["post_date"]
    )

    unobserved_dates = [
        date
        for date in FULL_CALENDAR
        if date not in observed_dates
    ]

    coverage_rows.append(
        {
            "Asset":
                asset,

            "Study_Start":
                STUDY_START,

            "Study_End":
                STUDY_END,

            "Calendar_Days":
                len(FULL_CALENDAR),

            "Days_With_Observed_Reddit_Posts":
                len(observed_dates),

            "Days_Without_Observed_Reddit_Posts":
                len(unobserved_dates),

            "Coverage_Rate":
                (
                    len(observed_dates)
                    / len(FULL_CALENDAR)
                )
        }
    )

coverage = pd.DataFrame(
    coverage_rows
)

print(
    coverage.to_string(
        index=False
    )
)


# =============================================================================
# 32. IDENTIFY UNOBSERVED ASSET-DAYS
# =============================================================================

section(
    "IDENTIFYING UNOBSERVED REDDIT ASSET-DAYS"
)

unobserved_asset_day_rows = []

for asset in ["BTC", "ETH"]:

    observed_dates = set(
        daily.loc[
            daily["asset"] == asset,
            "post_date"
        ]
    )

    for date in FULL_CALENDAR:

        if date not in observed_dates:

            unobserved_asset_day_rows.append(
                {
                    "Date":
                        date,

                    "Asset":
                        asset,

                    "Stage05_Status":
                        "NO_OBSERVED_REDDIT_POST"
                }
            )

unobserved_asset_days = pd.DataFrame(
    unobserved_asset_day_rows,
    columns=[
        "Date",
        "Asset",
        "Stage05_Status"
    ]
)

print(
    "\nNumber of unobserved asset-days:"
)

print(
    len(unobserved_asset_days)
)

if not unobserved_asset_days.empty:

    print(
        "\nUnobserved asset-days:"
    )

    print(
        unobserved_asset_days
        .to_string(
            index=False
        )
    )


# =============================================================================
# 33. IMPORTANT COVERAGE INTERPRETATION
# =============================================================================

section(
    "COVERAGE INTERPRETATION"
)

print(
    """
Stage 05 does NOT assume that every unobserved date × asset observation
represents a genuine zero-post day.

At this stage:

    observed post(s)
        = Reddit information is observed.

    no observed post
        = the date × asset is absent from the retained post-level dataset.

The reason for absence is not inferred here.

In particular, Stage 06 must distinguish genuine zero-post dates from dates
that fall outside available Reddit source/extraction coverage.

Therefore:

    - Post_Count is NOT automatically set to zero for unobserved asset-days.
    - Log_Reddit_Post_Count is NOT automatically set to zero.
    - Sentiment is NOT automatically set to zero.
    - Sentiment = 0 continues to mean neutral measured sentiment.

Calendar completion and missing-day classification belong to Stage 06.
"""
)


# =============================================================================
# 34. CREATE WIDE DAILY DATASET
# =============================================================================

section(
    "CREATING WIDE DAILY REDDIT DATASET"
)

btc_daily = (
    daily[
        daily["asset"] == "BTC"
    ]
    .drop(
        columns="asset"
    )
    .rename(
        columns={
            "post_date":
                "Date",

            "Post_Count":
                "BTC_Reddit_Post_Count",

            "Log_Reddit_Post_Count":
                "Log_BTC_Reddit_Post_Count",

            "Mean_Reddit_Sentiment":
                "BTC_Reddit_Sentiment",

            "Median_Reddit_Sentiment":
                "BTC_Reddit_Sentiment_Median",

            "Std_Reddit_Sentiment":
                "BTC_Reddit_Sentiment_SD",

            "Positive_Post_Count":
                "BTC_Reddit_Positive_Post_Count",

            "Neutral_Post_Count":
                "BTC_Reddit_Neutral_Post_Count",

            "Negative_Post_Count":
                "BTC_Reddit_Negative_Post_Count",

            "Positive_Share":
                "BTC_Reddit_Positive_Share",

            "Neutral_Share":
                "BTC_Reddit_Neutral_Share",

            "Negative_Share":
                "BTC_Reddit_Negative_Share"
        }
    )
)

eth_daily = (
    daily[
        daily["asset"] == "ETH"
    ]
    .drop(
        columns="asset"
    )
    .rename(
        columns={
            "post_date":
                "Date",

            "Post_Count":
                "ETH_Reddit_Post_Count",

            "Log_Reddit_Post_Count":
                "Log_ETH_Reddit_Post_Count",

            "Mean_Reddit_Sentiment":
                "ETH_Reddit_Sentiment",

            "Median_Reddit_Sentiment":
                "ETH_Reddit_Sentiment_Median",

            "Std_Reddit_Sentiment":
                "ETH_Reddit_Sentiment_SD",

            "Positive_Post_Count":
                "ETH_Reddit_Positive_Post_Count",

            "Neutral_Post_Count":
                "ETH_Reddit_Neutral_Post_Count",

            "Negative_Post_Count":
                "ETH_Reddit_Negative_Post_Count",

            "Positive_Share":
                "ETH_Reddit_Positive_Share",

            "Neutral_Share":
                "ETH_Reddit_Neutral_Share",

            "Negative_Share":
                "ETH_Reddit_Negative_Share"
        }
    )
)

wide = pd.merge(
    btc_daily,
    eth_daily,
    on="Date",
    how="outer",
    validate="one_to_one"
)

wide = (
    wide
    .sort_values(
        "Date"
    )
    .reset_index(
        drop=True
    )
)


# =============================================================================
# 35. DO NOT CREATE FORECASTING LAGS HERE
# =============================================================================

section(
    "TIMING CHECK"
)

print(
    """
Section 05 creates contemporaneous daily Reddit variables only.

No t-1, t-2, t-3 or t-7 variables are created here.

Lagged Reddit variables will be constructed in the later
information-alignment stage.

Primary forecasting timing:

        Reddit information at t-1
                    ↓
              predicts return at t

This separation prevents accidental look-ahead in Stage 05.
"""
)


# =============================================================================
# 36. DAILY BY-ASSET DATA
# =============================================================================

section(
    "CREATING DAILY BY-ASSET OUTPUT"
)

daily_by_asset = (
    daily[
        [
            "post_date",
            "asset",
            "Post_Count",
            "Log_Reddit_Post_Count",
            "Mean_Reddit_Sentiment",
            "Median_Reddit_Sentiment",
            "Std_Reddit_Sentiment",
            "Positive_Share",
            "Neutral_Share",
            "Negative_Share"
        ]
    ]
    .copy()
)


# =============================================================================
# 37. YEAR SUMMARY
# =============================================================================

section(
    "CREATING YEARLY REDDIT SUMMARY"
)

daily_with_year = (
    daily.copy()
)

daily_with_year["Year"] = (
    daily_with_year["post_date"]
    .dt.year
)

year_summary = (
    daily_with_year
    .groupby(
        "Year",
        as_index=False
    )
    .agg(
        Asset_Days=(
            "asset",
            "count"
        ),

        Total_Reddit_Posts=(
            "Post_Count",
            "sum"
        ),

        Mean_Daily_Sentiment=(
            "Mean_Reddit_Sentiment",
            "mean"
        ),

        Median_Daily_Sentiment=(
            "Mean_Reddit_Sentiment",
            "median"
        ),

        Mean_Daily_Log_Activity=(
            "Log_Reddit_Post_Count",
            "mean"
        )
    )
)


# =============================================================================
# 38. YEAR × ASSET SUMMARY
# =============================================================================

section(
    "CREATING YEAR × ASSET REDDIT SUMMARY"
)

year_asset_summary = (
    daily_with_year
    .groupby(
        [
            "Year",
            "asset"
        ],
        as_index=False
    )
    .agg(
        Asset_Days=(
            "post_date",
            "count"
        ),

        Total_Reddit_Posts=(
            "Post_Count",
            "sum"
        ),

        Mean_Daily_Sentiment=(
            "Mean_Reddit_Sentiment",
            "mean"
        ),

        Median_Daily_Sentiment=(
            "Mean_Reddit_Sentiment",
            "median"
        ),

        Mean_Daily_Log_Activity=(
            "Log_Reddit_Post_Count",
            "mean"
        )
    )
)


# =============================================================================
# 39. DAILY ACTIVITY DISTRIBUTION
# =============================================================================

section(
    "CREATING DAILY ACTIVITY DISTRIBUTION"
)

activity_distribution_rows = []

for asset in ["BTC", "ETH"]:

    asset_activity = (
        daily.loc[
            daily["asset"] == asset,
            "Post_Count"
        ]
    )

    asset_log_activity = (
        daily.loc[
            daily["asset"] == asset,
            "Log_Reddit_Post_Count"
        ]
    )

    activity_distribution_rows.append(
        {
            "Asset":
                asset,

            "Daily_Observations":
                len(asset_activity),

            "Mean_Post_Count":
                asset_activity.mean(),

            "Median_Post_Count":
                asset_activity.median(),

            "SD_Post_Count":
                asset_activity.std(),

            "Min_Post_Count":
                asset_activity.min(),

            "Max_Post_Count":
                asset_activity.max(),

            "Raw_Post_Count_Skewness":
                asset_activity.skew(),

            "Mean_Log_Post_Count":
                asset_log_activity.mean(),

            "Median_Log_Post_Count":
                asset_log_activity.median(),

            "SD_Log_Post_Count":
                asset_log_activity.std(),

            "Log_Post_Count_Skewness":
                asset_log_activity.skew()
        }
    )

activity_distribution = pd.DataFrame(
    activity_distribution_rows
)

print(
    activity_distribution.to_string(
        index=False
    )
)


# =============================================================================
# 40. FINAL DAILY DATA VALIDATION
# =============================================================================

section(
    "FINAL DAILY DATA VALIDATION"
)

duplicate_daily_rows = int(
    daily
    .duplicated(
        subset=[
            "post_date",
            "asset"
        ]
    )
    .sum()
)

print("\nDuplicate date × asset rows:")
print(duplicate_daily_rows)

if duplicate_daily_rows != 0:
    fail(
        "Duplicate date × asset rows found."
    )


nonpositive_counts = int(
    (
        daily["Post_Count"]
        <= 0
    )
    .sum()
)

print("\nNon-positive observed daily post counts:")
print(nonpositive_counts)

if nonpositive_counts != 0:
    fail(
        "Non-positive post counts found among observed "
        "date × asset rows."
    )


daily_sentiment_min = float(
    daily["Mean_Reddit_Sentiment"]
    .min()
)

daily_sentiment_max = float(
    daily["Mean_Reddit_Sentiment"]
    .max()
)

print("\nDaily sentiment minimum:")
print(daily_sentiment_min)

print("\nDaily sentiment maximum:")
print(daily_sentiment_max)

if daily_sentiment_min < -1 - 1e-12:
    fail(
        "Daily sentiment below -1."
    )

if daily_sentiment_max > 1 + 1e-12:
    fail(
        "Daily sentiment above +1."
    )


if daily["Mean_Reddit_Sentiment"].isna().any():
    fail(
        "Missing mean sentiment found among observed "
        "date × asset rows."
    )


if daily["Log_Reddit_Post_Count"].isna().any():
    fail(
        "Missing log activity found among observed "
        "date × asset rows."
    )


if len(daily) != (
    daily[
        [
            "post_date",
            "asset"
        ]
    ]
    .drop_duplicates()
    .shape[0]
):
    fail(
        "Daily dataset is not unique by date × asset."
    )

print("\nFinal daily-data validation: PASS")


# =============================================================================
# 41. STAGE 05 QC SUMMARY
# =============================================================================

section(
    "CREATING STAGE 05 QC SUMMARY"
)

btc_daily_rows = int(
    (
        daily["asset"]
        == "BTC"
    )
    .sum()
)

eth_daily_rows = int(
    (
        daily["asset"]
        == "ETH"
    )
    .sum()
)

btc_unobserved_days = int(
    (
        unobserved_asset_days["Asset"]
        == "BTC"
    )
    .sum()
)

eth_unobserved_days = int(
    (
        unobserved_asset_days["Asset"]
        == "ETH"
    )
    .sum()
)

qc_rows = [
    {
        "Check":
            "Stage04_Input_Rows",

        "Expected":
            EXPECTED_INPUT_ROWS,

        "Observed":
            len(working),

        "Pass":
            len(working) == EXPECTED_INPUT_ROWS
    },

    {
        "Check":
            "BTC_Post_Count",

        "Expected":
            EXPECTED_POSTS_BY_ASSET["BTC"],

        "Observed":
            actual_posts_by_asset["BTC"],

        "Pass":
            (
                actual_posts_by_asset["BTC"]
                == EXPECTED_POSTS_BY_ASSET["BTC"]
            )
    },

    {
        "Check":
            "ETH_Post_Count",

        "Expected":
            EXPECTED_POSTS_BY_ASSET["ETH"],

        "Observed":
            actual_posts_by_asset["ETH"],

        "Pass":
            (
                actual_posts_by_asset["ETH"]
                == EXPECTED_POSTS_BY_ASSET["ETH"]
            )
    },

    {
        "Check":
            "Global_Daily_Post_Reconstruction",

        "Expected":
            EXPECTED_INPUT_ROWS,

        "Observed":
            total_daily_posts,

        "Pass":
            total_daily_posts == EXPECTED_INPUT_ROWS
    },

    {
        "Check":
            "Calendar_Days",

        "Expected":
            EXPECTED_CALENDAR_DAYS,

        "Observed":
            len(FULL_CALENDAR),

        "Pass":
            (
                len(FULL_CALENDAR)
                == EXPECTED_CALENDAR_DAYS
            )
    },

    {
        "Check":
            "Duplicate_Date_Asset_Rows",

        "Expected":
            0,

        "Observed":
            duplicate_daily_rows,

        "Pass":
            duplicate_daily_rows == 0
    },

    {
        "Check":
            "Daily_Category_Count_Mismatches",

        "Expected":
            0,

        "Observed":
            count_mismatches,

        "Pass":
            count_mismatches == 0
    },

    {
        "Check":
            "BTC_Observed_Daily_Rows",

        "Expected":
            "diagnostic",

        "Observed":
            btc_daily_rows,

        "Pass":
            True
    },

    {
        "Check":
            "ETH_Observed_Daily_Rows",

        "Expected":
            "diagnostic",

        "Observed":
            eth_daily_rows,

        "Pass":
            True
    },

    {
        "Check":
            "BTC_Unobserved_Asset_Days",

        "Expected":
            "diagnostic",

        "Observed":
            btc_unobserved_days,

        "Pass":
            True
    },

    {
        "Check":
            "ETH_Unobserved_Asset_Days",

        "Expected":
            "diagnostic",

        "Observed":
            eth_unobserved_days,

        "Pass":
            True
    }
]

stage05_qc = pd.DataFrame(
    qc_rows
)

print(
    stage05_qc.to_string(
        index=False
    )
)

hard_qc = stage05_qc[
    stage05_qc["Expected"]
    != "diagnostic"
]

if not hard_qc["Pass"].all():
    fail(
        "One or more Stage 05 hard QC checks failed."
    )

print("\nAll Stage 05 hard QC checks: PASS")


# =============================================================================
# 42. SAVE OUTPUTS
# =============================================================================

section(
    "SAVING STAGE 05 OUTPUTS"
)

files_to_save = {

    "reddit_daily_sentiment_activity.csv":
        daily,

    "reddit_daily_sentiment_activity_wide.csv":
        wide,

    "reddit_daily_sentiment_by_asset.csv":
        daily_by_asset,

    "reddit_daily_sentiment_by_year.csv":
        year_summary,

    "reddit_daily_sentiment_by_year_asset.csv":
        year_asset_summary,

    "reddit_daily_activity_distribution.csv":
        activity_distribution,

    "reddit_daily_aggregation_validation.csv":
        aggregation_validation,

    "reddit_daily_coverage.csv":
        coverage,

    "reddit_daily_unobserved_asset_days.csv":
        unobserved_asset_days,

    "reddit_daily_stage05_qc.csv":
        stage05_qc
}


for filename, dataframe in files_to_save.items():

    output_path = (
        OUTPUT_DIR
        / filename
    )

    dataframe.to_csv(
        output_path,
        index=False
    )

    print(
        f"Saved: {output_path}"
    )


# =============================================================================
# 43. REMOVE OBSOLETE ZERO-COVERAGE OUTPUT IF PRESENT
# =============================================================================

# The previous Stage 05 version called absent asset-days "zero coverage".
# That terminology is potentially too strong because absence may reflect
# genuine no-post activity OR unavailable source coverage.
#
# The improved output is:
#
# reddit_daily_unobserved_asset_days.csv
#
# Remove the obsolete file so that it cannot accidentally be used later.

obsolete_zero_coverage_file = (
    OUTPUT_DIR
    / "reddit_daily_zero_coverage.csv"
)

if obsolete_zero_coverage_file.exists():

    obsolete_zero_coverage_file.unlink()

    print(
        "\nRemoved obsolete output:"
    )

    print(
        obsolete_zero_coverage_file
    )


# =============================================================================
# 44. METHODOLOGY NOTE
# =============================================================================

section(
    "WRITING METHODOLOGY NOTE"
)

methodology_text = f"""
SECTION 05 — DAILY REDDIT VARIABLE CONSTRUCTION
================================================

Study period
------------
{STUDY_START.date()} to {STUDY_END.date()}

Assets
------
Bitcoin (BTC) and Ethereum (ETH)

Frozen Stage 04 input
---------------------
Total retained and sentiment-scored Reddit posts:
    {EXPECTED_INPUT_ROWS:,}

BTC:
    {EXPECTED_POSTS_BY_ASSET["BTC"]:,}

ETH:
    {EXPECTED_POSTS_BY_ASSET["ETH"]:,}

Unit of aggregation
-------------------
Date × asset.

Post-level sentiment
--------------------
Each retained Reddit post has a continuous sentiment score:

    S_i = P(Positive_i) - P(Negative_i)

where P(Positive_i) and P(Negative_i) are probabilities assigned by the
Stage 04 sentiment model.

Daily Reddit sentiment
----------------------
For each asset a and date t:

    Sentiment_(a,t)
        = (1 / N_(a,t)) * SUM(S_i)

where N_(a,t) is the number of retained Reddit posts observed for asset a
on date t.

The primary daily sentiment measure is therefore the unweighted arithmetic
mean of post-level continuous sentiment.

Reddit scores, upvotes, upvote ratios and comment counts are not used as
sentiment weights.

Reddit activity
---------------
Observed daily Reddit activity is measured using the number of retained
posts:

    N_(a,t)

For later econometric modelling, the activity variable is:

    Activity_(a,t) = log(1 + N_(a,t))

The raw count is retained for descriptive reporting.

Sentiment versus activity
-------------------------
Sentiment measures the estimated tone of Reddit discussion.

Activity measures the observed quantity or intensity of Reddit discussion.

The variables are retained separately so later models can distinguish
whether incremental explanatory or predictive information is associated
with sentiment, activity, or both.

Timing
------
Stage 05 contains contemporaneous daily Reddit variables only.

No t-1, t-2, t-3 or t-7 forecasting variables are created here.

Forecasting lags are constructed only at the subsequent information-
alignment stage so that information available at t-1 can be used to
predict cryptocurrency returns at t.

Unobserved asset-days
---------------------
A date × asset combination absent from the Stage 04 post-level data is
identified in Stage 05 as an unobserved Reddit asset-day.

Stage 05 does NOT assume that every such date is a genuine zero-post day.

An absent asset-day may represent:

    1. a genuine date with no retained Reddit posts; or
    2. unavailable/incomplete Reddit source coverage.

The distinction is made in Stage 06 using the available source-coverage
information.

Consequently, Stage 05 does not automatically assign:

    Post_Count = 0
    Log_Reddit_Post_Count = 0
    Sentiment = 0

to unobserved asset-days.

This is particularly important because a sentiment value of zero means
measured neutral sentiment and must not be used as a generic missing-value
code.

Aggregation validation
----------------------
Daily aggregation is independently validated against Stage 04 by checking:

    - frozen total Stage 04 input rows;
    - frozen BTC and ETH post counts;
    - unique post identifiers;
    - valid study dates;
    - valid BTC/ETH asset values;
    - valid sentiment values and labels;
    - total post-count reconstruction;
    - asset-specific post-count reconstruction;
    - observed-date reconstruction;
    - reconstruction of the overall post-level sentiment mean from daily
      means weighted by daily post counts;
    - daily sentiment-share reconstruction;
    - log(1 + post count) construction;
    - uniqueness of date × asset rows;
    - daily sentiment bounds.

Stage boundary
--------------
No cryptocurrency returns, trading volume or traditional-market variables
are merged in Stage 05.

Calendar completion, classification of missing Reddit days, forecasting
lags and information alignment are performed in later stages.
"""

methodology_path = (
    OUTPUT_DIR
    / "reddit_daily_methodology_note.txt"
)

methodology_path.write_text(
    methodology_text.strip(),
    encoding="utf-8"
)

print(
    f"Saved: {methodology_path}"
)


# =============================================================================
# 45. FINAL RELOAD VALIDATION
# =============================================================================

section(
    "FINAL OUTPUT RELOAD VALIDATION"
)

main_output_path = (
    OUTPUT_DIR
    / "reddit_daily_sentiment_activity.csv"
)

reloaded_daily = pd.read_csv(
    main_output_path,
    low_memory=False
)

print(
    f"\nReloaded daily rows: "
    f"{len(reloaded_daily):,}"
)

if len(reloaded_daily) != len(daily):
    fail(
        "Reloaded Stage 05 daily output has an "
        "unexpected number of rows."
    )

reloaded_post_total = int(
    reloaded_daily["Post_Count"]
    .sum()
)

if reloaded_post_total != EXPECTED_INPUT_ROWS:
    fail(
        "Reloaded Stage 05 output does not reconstruct "
        "the frozen Stage 04 input count."
    )

reloaded_duplicates = int(
    reloaded_daily
    .duplicated(
        subset=[
            "post_date",
            "asset"
        ]
    )
    .sum()
)

if reloaded_duplicates != 0:
    fail(
        "Reloaded Stage 05 output contains duplicate "
        "date × asset rows."
    )

print(
    "\nFinal output reload validation: PASS"
)


# =============================================================================
# 46. FINAL SUMMARY
# =============================================================================

section(
    "SECTION 05 COMPLETE"
)

print(
    f"""
Study period:
    {STUDY_START.date()} to {STUDY_END.date()}

Calendar days:
    {len(FULL_CALENDAR):,}

Stage 04 post-level observations processed:
    {len(working):,}

Expected Stage 04 observations:
    {EXPECTED_INPUT_ROWS:,}

BTC post-level observations:
    {actual_posts_by_asset["BTC"]:,}

ETH post-level observations:
    {actual_posts_by_asset["ETH"]:,}

Observed daily date × asset observations:
    {len(daily):,}

BTC observed daily observations:
    {btc_daily_rows:,}

ETH observed daily observations:
    {eth_daily_rows:,}

Unobserved asset-days:
    {len(unobserved_asset_days):,}

BTC unobserved asset-days:
    {btc_unobserved_days:,}

ETH unobserved asset-days:
    {eth_unobserved_days:,}

Primary sentiment:
    Mean post-level P(Positive) - P(Negative)

Activity:
    log(1 + daily retained post count)

Primary forecasting lag:
    NOT created here

Traditional market variables:
    NOT included in Stage 05

Cryptocurrency returns:
    NOT included in Stage 05

Trading volume:
    NOT included in Stage 05

Frozen input validation:
    PASS

Global post-count reconstruction:
    PASS

Asset-specific aggregation validation:
    PASS

Daily category reconstruction:
    PASS

Final daily duplicate check:
    PASS

Final output reload validation:
    PASS

Stage 05 hard QC:
    PASS
"""
)

print(
    "\nSection 05 finished successfully."
)