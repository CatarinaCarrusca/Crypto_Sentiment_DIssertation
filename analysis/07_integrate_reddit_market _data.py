"""
===============================================================================
SECTION 07 — REDDIT–MARKET DATA INTEGRATION
===============================================================================

Dissertation:
Do Social Media Sentiment Signals Improve the Prediction of Cryptocurrency
Returns Beyond Traditional Market Indicators? Evidence from Bitcoin and
Ethereum

Purpose
-------
This script integrates:

1. The forecast-safe Reddit variables produced by Section 06; and
2. The information-aligned cryptocurrency / traditional-market dataset.

The resulting dataset preserves the 24/7 BTC/ETH target calendar and uses
only explanatory information available strictly before the target return
date for predictive specifications.

IMPORTANT TIMING PRINCIPLES
---------------------------
For target cryptocurrency return on calendar day t:

    Reddit:
        primary predictor source = t - 1 calendar day

    Cryptocurrency own return:
        lagged return only

    Cryptocurrency trading volume:
        lagged log(1 + Yahoo Finance-reported volume) only

    Traditional controls:
        latest completed transformed observation strictly before t,
        as already constructed in information_aligned_dataset.csv

Traditional controls:
    S&P 500 return
    VIX change
    Gold return
    DXY return
    US 10-year Treasury yield change

This script does NOT estimate regressions or forecasts.

It performs:
    - schema validation
    - identifier harmonisation
    - date harmonisation
    - target-preserving BTC/ETH calendar construction
    - Reddit + market integration
    - unmatched-date checks
    - duplicate checks
    - row-count reconciliation
    - independent market lag checks
    - independent traditional-control alignment checks
    - no-look-ahead checks
    - creation of asset-specific modelling variables
    - output reload validation
    - QC reporting
    - methodology documentation

Main output:
    combined_market_reddit_dataset.csv

===============================================================================
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REDDIT_INPUT = (
    PROJECT_ROOT
    / "data_processed"
    / "reddit"
    / "stage06_reddit_forecast"
    / "reddit_forecast_ready.csv"
)

MARKET_INPUT = (
    PROJECT_ROOT
    / "data_processed"
    / "information_aligned_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "stage07_reddit_market_integration"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAIN_OUTPUT = (
    OUTPUT_DIR
    / "combined_market_reddit_dataset.csv"
)

QC_OUTPUT = (
    OUTPUT_DIR
    / "stage07_integration_qc.csv"
)

UNMATCHED_OUTPUT = (
    OUTPUT_DIR
    / "stage07_unmatched_dates.csv"
)

TIMING_OUTPUT = (
    OUTPUT_DIR
    / "stage07_information_timing_validation.csv"
)

COVERAGE_OUTPUT = (
    OUTPUT_DIR
    / "stage07_asset_coverage_summary.csv"
)

VARIABLE_OUTPUT = (
    OUTPUT_DIR
    / "stage07_variable_dictionary.csv"
)

METHODOLOGY_OUTPUT = (
    OUTPUT_DIR
    / "stage07_methodology_note.txt"
)


# =============================================================================
# 2. STUDY DESIGN
# =============================================================================

STUDY_START = pd.Timestamp("2021-01-01")
STUDY_END = pd.Timestamp("2025-12-31")

ASSETS = ["BTC", "ETH"]

EXPECTED_CALENDAR_DAYS = 1826
EXPECTED_DATE_ASSET_ROWS = 3652
EXPECTED_REDDIT_ROWS = 3652
EXPECTED_MARKET_ROWS = 1826

TOL = 1e-12


# =============================================================================
# 3. REQUIRED MARKET VARIABLES
# =============================================================================

MARKET_REQUIRED = [
    "Date",

    # Cryptocurrency targets / lagged predictors
    "BTC_Return",
    "BTC_Lagged_Return",
    "Lagged_Log_BTC_Volume",

    "ETH_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_ETH_Volume",

    # Original transformed traditional-market series
    # retained only for validation / audit
    "SP500_Return",
    "VIX_Change",
    "Gold_Return",
    "DXY_Return",
    "US10Y_Change",

    # Forecast-safe aligned traditional controls
    "Lagged_SP500_Return_Aligned",
    "SP500_Source_Date",

    "Lagged_VIX_Change_Aligned",
    "VIX_Source_Date",

    "Lagged_Gold_Return_Aligned",
    "Gold_Source_Date",

    "Lagged_DXY_Return_Aligned",
    "DXY_Source_Date",

    "Lagged_US10Y_Change_Aligned",
    "US10Y_Source_Date",

    # Information-age audit fields
    "SP500_Information_Age_Days",
    "VIX_Information_Age_Days",
    "Gold_Information_Age_Days",
    "DXY_Information_Age_Days",
    "US10Y_Information_Age_Days",
]


# =============================================================================
# 4. REQUIRED REDDIT VARIABLES
# =============================================================================

REDDIT_PRIMARY_REQUIRED = [
    "Lagged_BTC_Reddit_Sentiment",
    "Lagged_Log_BTC_Reddit_Post_Count",
    "Lagged_ETH_Reddit_Sentiment",
    "Lagged_Log_ETH_Reddit_Post_Count",
]


# =============================================================================
# 5. TRADITIONAL-CONTROL ALIGNMENT MAP
# =============================================================================

TRADITIONAL_ALIGNMENT = {
    "SP500": {
        "raw": "SP500_Return",
        "aligned": "Lagged_SP500_Return_Aligned",
        "source": "SP500_Source_Date",
        "age": "SP500_Information_Age_Days",
    },
    "VIX": {
        "raw": "VIX_Change",
        "aligned": "Lagged_VIX_Change_Aligned",
        "source": "VIX_Source_Date",
        "age": "VIX_Information_Age_Days",
    },
    "Gold": {
        "raw": "Gold_Return",
        "aligned": "Lagged_Gold_Return_Aligned",
        "source": "Gold_Source_Date",
        "age": "Gold_Information_Age_Days",
    },
    "DXY": {
        "raw": "DXY_Return",
        "aligned": "Lagged_DXY_Return_Aligned",
        "source": "DXY_Source_Date",
        "age": "DXY_Information_Age_Days",
    },
    "US10Y": {
        "raw": "US10Y_Change",
        "aligned": "Lagged_US10Y_Change_Aligned",
        "source": "US10Y_Source_Date",
        "age": "US10Y_Information_Age_Days",
    },
}


# =============================================================================
# 6. HELPERS
# =============================================================================

def section(title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def fail(message):
    raise ValueError(
        "\nSECTION 07 ERROR\n"
        "----------------\n"
        + str(message)
    )


def require_file(path, label):
    if not path.exists():
        fail(
            f"{label} does not exist:\n{path}"
        )


def require_columns(df, required, label):
    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        fail(
            f"{label} is missing required columns:\n"
            f"{missing}\n\n"
            f"Available columns:\n"
            f"{list(df.columns)}"
        )


def parse_date_column(series, label):
    parsed = pd.to_datetime(
        series,
        errors="coerce"
    ).dt.normalize()

    bad = int(parsed.isna().sum())

    if bad > 0:
        fail(
            f"{label} contains {bad:,} unparseable dates."
        )

    return parsed


def standardise_asset(series):
    mapping = {
        "BTC": "BTC",
        "BITCOIN": "BTC",
        "XBT": "BTC",
        "ETH": "ETH",
        "ETHEREUM": "ETH",
    }

    clean = (
        series
        .astype(str)
        .str.strip()
        .str.upper()
        .map(mapping)
    )

    if clean.isna().any():
        bad_values = (
            series.loc[clean.isna()]
            .astype(str)
            .drop_duplicates()
            .tolist()
        )

        fail(
            "Could not standardise all Reddit asset identifiers.\n"
            f"Unrecognised values: {bad_values}"
        )

    return clean


def values_equal_with_nan(a, b, tol=TOL):
    """
    Elementwise comparison where NaN == NaN for validation purposes.
    """

    a = pd.to_numeric(
        a,
        errors="coerce"
    )

    b = pd.to_numeric(
        b,
        errors="coerce"
    )

    both_nan = a.isna() & b.isna()

    both_present_close = (
        a.notna()
        & b.notna()
        & np.isclose(
            a,
            b,
            atol=tol,
            rtol=tol,
            equal_nan=False
        )
    )

    return both_nan | both_present_close


def add_qc(qc_rows, check, expected, actual, passed):
    qc_rows.append(
        {
            "Check": check,
            "Expected": expected,
            "Actual": actual,
            "Pass": bool(passed),
        }
    )


def safe_reddit_columns(df):
    """
    Retain forecasting-safe lagged Reddit variables and useful audit metadata.

    We deliberately do NOT carry arbitrary contemporaneous Reddit sentiment
    or activity variables into the Section 07 modelling dataset.

    Stage 06 itself remains the authoritative source for contemporaneous
    Reddit audit information.
    """

    mandatory = {
        "Date",
        "Asset",
        *REDDIT_PRIMARY_REQUIRED,
    }

    metadata_exact = {
        "Reddit_Observation_Status",
        "Within_Reddit_Source_Coverage",
        "Activity_Available",
        "Sentiment_Available",
        "Reddit_Activity_Available",
        "Reddit_Sentiment_Available",
    }

    selected = []

    for col in df.columns:

        low = col.lower()

        keep = False

        if col in mandatory:
            keep = True

        elif col in metadata_exact:
            keep = True

        # Forecast-safe lagged variables
        elif "lagged" in low:
            keep = True

        elif "source_date" in low:
            keep = True

        elif "_lag_" in low:
            keep = True

        elif low.endswith("_lag"):
            keep = True

        elif any(
            token in low
            for token in [
                "_t1",
                "_t2",
                "_t3",
                "_t7",
                "lag1",
                "lag2",
                "lag3",
                "lag7",
            ]
        ):
            keep = True

        if keep:
            selected.append(col)

    # Preserve order and remove accidental duplicates.
    selected = list(dict.fromkeys(selected))

    return selected


# =============================================================================
# 7. START
# =============================================================================

section(
    "SECTION 07 — REDDIT–MARKET DATA INTEGRATION"
)

print("\nProject root:")
print(PROJECT_ROOT)

print("\nReddit input:")
print(REDDIT_INPUT)

print("\nMarket input:")
print(MARKET_INPUT)

print("\nOutput directory:")
print(OUTPUT_DIR)


# =============================================================================
# 8. CHECK INPUT FILES
# =============================================================================

section(
    "CHECKING SECTION 07 INPUT FILES"
)

require_file(
    REDDIT_INPUT,
    "Section 06 Reddit forecast-ready dataset"
)

require_file(
    MARKET_INPUT,
    "Information-aligned market dataset"
)

print("Reddit input exists: PASS")
print("Market input exists: PASS")


# =============================================================================
# 9. LOAD DATA
# =============================================================================

section(
    "LOADING INPUT DATA"
)

reddit = pd.read_csv(
    REDDIT_INPUT,
    low_memory=False
)

market = pd.read_csv(
    MARKET_INPUT,
    low_memory=False
)

print(f"Reddit rows loaded: {len(reddit):,}")
print(f"Market rows loaded: {len(market):,}")

print("\nReddit columns:")
print(list(reddit.columns))

print("\nMarket columns:")
print(list(market.columns))


# =============================================================================
# 10. IDENTIFY / STANDARDISE REDDIT KEYS
# =============================================================================

section(
    "STANDARDISING REDDIT DATE AND ASSET IDENTIFIERS"
)

reddit_date_candidates = [
    "Date",
    "date",
    "Target_Date",
    "target_date",
    "post_date",
]

reddit_asset_candidates = [
    "Asset",
    "asset",
    "Crypto",
    "crypto",
]

reddit_date_col = next(
    (
        col for col in reddit_date_candidates
        if col in reddit.columns
    ),
    None
)

reddit_asset_col = next(
    (
        col for col in reddit_asset_candidates
        if col in reddit.columns
    ),
    None
)

if reddit_date_col is None:
    fail(
        "Could not identify the Reddit date column."
    )

if reddit_asset_col is None:
    fail(
        "Could not identify the Reddit asset column."
    )

if reddit_date_col != "Date":
    reddit = reddit.rename(
        columns={
            reddit_date_col: "Date"
        }
    )

if reddit_asset_col != "Asset":
    reddit = reddit.rename(
        columns={
            reddit_asset_col: "Asset"
        }
    )

reddit["Date"] = parse_date_column(
    reddit["Date"],
    "Reddit Date"
)

reddit["Asset"] = standardise_asset(
    reddit["Asset"]
)

market["Date"] = parse_date_column(
    market["Date"],
    "Market Date"
)

for cfg in TRADITIONAL_ALIGNMENT.values():
    market[cfg["source"]] = pd.to_datetime(
        market[cfg["source"]],
        errors="coerce"
    ).dt.normalize()

print("Date parsing: PASS")
print("Asset standardisation: PASS")


# =============================================================================
# 11. VALIDATE INPUT SCHEMAS
# =============================================================================

section(
    "VALIDATING INPUT SCHEMAS"
)

require_columns(
    reddit,
    [
        "Date",
        "Asset",
        *REDDIT_PRIMARY_REQUIRED,
    ],
    "Section 06 Reddit input"
)

require_columns(
    market,
    MARKET_REQUIRED,
    "Information-aligned market input"
)

print("Required Reddit columns: PASS")
print("Required market columns: PASS")


# =============================================================================
# 12. VALIDATE STUDY PERIOD
# =============================================================================

section(
    "VALIDATING STUDY PERIOD"
)

reddit_min = reddit["Date"].min()
reddit_max = reddit["Date"].max()

market_min = market["Date"].min()
market_max = market["Date"].max()

print(
    f"Reddit period: {reddit_min.date()} to {reddit_max.date()}"
)

print(
    f"Market period: {market_min.date()} to {market_max.date()}"
)

if reddit_min != STUDY_START:
    fail(
        f"Unexpected Reddit start date: {reddit_min}"
    )

if reddit_max != STUDY_END:
    fail(
        f"Unexpected Reddit end date: {reddit_max}"
    )

if market_min != STUDY_START:
    fail(
        f"Unexpected market start date: {market_min}"
    )

if market_max != STUDY_END:
    fail(
        f"Unexpected market end date: {market_max}"
    )

print("Study-period validation: PASS")


# =============================================================================
# 13. VALIDATE INPUT ROW COUNTS
# =============================================================================

section(
    "VALIDATING INPUT ROW COUNTS"
)

if len(reddit) != EXPECTED_REDDIT_ROWS:
    fail(
        f"Expected {EXPECTED_REDDIT_ROWS:,} Reddit rows, "
        f"found {len(reddit):,}."
    )

if len(market) != EXPECTED_MARKET_ROWS:
    fail(
        f"Expected {EXPECTED_MARKET_ROWS:,} market rows, "
        f"found {len(market):,}."
    )

print(
    f"Reddit rows: {len(reddit):,} — PASS"
)

print(
    f"Market rows: {len(market):,} — PASS"
)


# =============================================================================
# 14. VALIDATE UNIQUE KEYS
# =============================================================================

section(
    "VALIDATING UNIQUE INPUT KEYS"
)

reddit_dup = int(
    reddit.duplicated(
        ["Date", "Asset"]
    ).sum()
)

market_dup = int(
    market.duplicated(
        ["Date"]
    ).sum()
)

print(
    f"Reddit duplicate Date × Asset rows: {reddit_dup}"
)

print(
    f"Market duplicate Date rows: {market_dup}"
)

if reddit_dup != 0:
    fail(
        "Section 06 Reddit input contains duplicate "
        "Date × Asset observations."
    )

if market_dup != 0:
    fail(
        "Market input contains duplicate Date observations."
    )

print("Input uniqueness validation: PASS")


# =============================================================================
# 15. VALIDATE COMPLETE MARKET TARGET CALENDAR
# =============================================================================

section(
    "VALIDATING 24/7 MARKET TARGET CALENDAR"
)

expected_dates = pd.date_range(
    STUDY_START,
    STUDY_END,
    freq="D"
)

actual_market_dates = pd.DatetimeIndex(
    market["Date"].sort_values()
)

if len(expected_dates) != EXPECTED_CALENDAR_DAYS:
    fail(
        "Internal expected-calendar calculation is incorrect."
    )

missing_market_calendar_dates = (
    expected_dates
    .difference(actual_market_dates)
)

extra_market_dates = (
    actual_market_dates
    .difference(expected_dates)
)

print(
    f"Expected calendar days: {len(expected_dates):,}"
)

print(
    f"Market calendar days: {market['Date'].nunique():,}"
)

print(
    f"Missing market calendar dates: "
    f"{len(missing_market_calendar_dates):,}"
)

print(
    f"Unexpected market dates: "
    f"{len(extra_market_dates):,}"
)

if len(missing_market_calendar_dates) != 0:
    fail(
        "Market target calendar is incomplete."
    )

if len(extra_market_dates) != 0:
    fail(
        "Market dataset contains dates outside the study calendar."
    )

print("Complete 24/7 market target calendar: PASS")


# =============================================================================
# 16. INDEPENDENT CRYPTO LAG VALIDATION
# =============================================================================

section(
    "VALIDATING CRYPTOCURRENCY CALENDAR LAGS"
)

market = market.sort_values(
    "Date"
).reset_index(drop=True)

btc_return_expected = (
    market["BTC_Return"].shift(1)
)

eth_return_expected = (
    market["ETH_Return"].shift(1)
)

btc_return_match = values_equal_with_nan(
    market["BTC_Lagged_Return"],
    btc_return_expected
)

eth_return_match = values_equal_with_nan(
    market["ETH_Lagged_Return"],
    eth_return_expected
)

btc_volume_expected = (
    market["Log_BTC_Volume"].shift(1)
    if "Log_BTC_Volume" in market.columns
    else None
)

eth_volume_expected = (
    market["Log_ETH_Volume"].shift(1)
    if "Log_ETH_Volume" in market.columns
    else None
)

if btc_volume_expected is None:
    fail(
        "Log_BTC_Volume is required for independent "
        "lagged-volume reconstruction."
    )

if eth_volume_expected is None:
    fail(
        "Log_ETH_Volume is required for independent "
        "lagged-volume reconstruction."
    )

btc_volume_match = values_equal_with_nan(
    market["Lagged_Log_BTC_Volume"],
    btc_volume_expected
)

eth_volume_match = values_equal_with_nan(
    market["Lagged_Log_ETH_Volume"],
    eth_volume_expected
)

btc_ret_mismatch = int(
    (~btc_return_match).sum()
)

eth_ret_mismatch = int(
    (~eth_return_match).sum()
)

btc_vol_mismatch = int(
    (~btc_volume_match).sum()
)

eth_vol_mismatch = int(
    (~eth_volume_match).sum()
)

print(
    f"BTC return-lag mismatches: {btc_ret_mismatch}"
)

print(
    f"ETH return-lag mismatches: {eth_ret_mismatch}"
)

print(
    f"BTC volume-lag mismatches: {btc_vol_mismatch}"
)

print(
    f"ETH volume-lag mismatches: {eth_vol_mismatch}"
)

if any(
    value != 0
    for value in [
        btc_ret_mismatch,
        eth_ret_mismatch,
        btc_vol_mismatch,
        eth_vol_mismatch,
    ]
):
    fail(
        "One or more cryptocurrency lagged variables "
        "failed independent reconstruction."
    )

print("Cryptocurrency lag validation: PASS")


# =============================================================================
# 17. VALIDATE TRADITIONAL-MARKET INFORMATION ALIGNMENT
# =============================================================================

section(
    "VALIDATING TRADITIONAL-MARKET INFORMATION ALIGNMENT"
)

timing_rows = []

for name, cfg in TRADITIONAL_ALIGNMENT.items():

    aligned_col = cfg["aligned"]
    source_col = cfg["source"]
    raw_col = cfg["raw"]
    age_col = cfg["age"]

    source_present = (
        market[source_col].notna()
    )

    aligned_present = (
        market[aligned_col].notna()
    )

    # -------------------------------------------------------------------------
    # Strict pre-target timing
    # -------------------------------------------------------------------------

    bad_same_or_future = (
        source_present
        & (
            market[source_col]
            >= market["Date"]
        )
    )

    same_or_future_count = int(
        bad_same_or_future.sum()
    )

    # -------------------------------------------------------------------------
    # Information age must equal Target_Date - Source_Date
    # -------------------------------------------------------------------------

    calculated_age = (
        market["Date"]
        - market[source_col]
    ).dt.days

    age_match = (
        market[age_col].isna()
        & calculated_age.isna()
    ) | (
        market[age_col].notna()
        & calculated_age.notna()
        & np.isclose(
            pd.to_numeric(
                market[age_col],
                errors="coerce"
            ),
            calculated_age,
            atol=TOL,
            rtol=TOL,
            equal_nan=False
        )
    )

    age_mismatch_count = int(
        (~age_match).sum()
    )

    # -------------------------------------------------------------------------
    # Independently reconstruct aligned value from source date
    # -------------------------------------------------------------------------

    lookup = (
        market[
            ["Date", raw_col]
        ]
        .drop_duplicates("Date")
        .set_index("Date")[raw_col]
    )

    reconstructed = (
        market[source_col]
        .map(lookup)
    )

    reconstructed_match = (
        values_equal_with_nan(
            market[aligned_col],
            reconstructed
        )
    )

    reconstructed_mismatch_count = int(
        (~reconstructed_match).sum()
    )

    passed = (
        same_or_future_count == 0
        and age_mismatch_count == 0
        and reconstructed_mismatch_count == 0
    )

    timing_rows.append(
        {
            "Variable": name,
            "Aligned_Column": aligned_col,
            "Source_Date_Column": source_col,
            "Rows_With_Source_Date": int(
                source_present.sum()
            ),
            "Rows_With_Aligned_Value": int(
                aligned_present.sum()
            ),
            "Same_Day_Or_Future_Source_Failures":
                same_or_future_count,
            "Information_Age_Mismatches":
                age_mismatch_count,
            "Aligned_Value_Reconstruction_Mismatches":
                reconstructed_mismatch_count,
            "Pass": passed,
        }
    )

    print(
        f"\n{name}"
    )

    print(
        f"  rows with source date: "
        f"{int(source_present.sum()):,}"
    )

    print(
        f"  same-day/future failures: "
        f"{same_or_future_count:,}"
    )

    print(
        f"  age mismatches: "
        f"{age_mismatch_count:,}"
    )

    print(
        f"  aligned-value mismatches: "
        f"{reconstructed_mismatch_count:,}"
    )

    print(
        f"  PASS: {passed}"
    )

timing_df = pd.DataFrame(
    timing_rows
)

if not timing_df["Pass"].all():
    fail(
        "Traditional-market information alignment "
        "failed one or more no-look-ahead checks."
    )

print(
    "\nTraditional-market information alignment: PASS"
)


# =============================================================================
# 18. CHECK REDDIT / MARKET DATE COVERAGE BEFORE MERGE
# =============================================================================

section(
    "CHECKING REDDIT / MARKET DATE COVERAGE"
)

reddit_dates = pd.DatetimeIndex(
    reddit["Date"].drop_duplicates()
)

market_dates = pd.DatetimeIndex(
    market["Date"].drop_duplicates()
)

market_not_in_reddit = (
    market_dates
    .difference(reddit_dates)
)

reddit_not_in_market = (
    reddit_dates
    .difference(market_dates)
)

unmatched_rows = []

for d in market_not_in_reddit:
    unmatched_rows.append(
        {
            "Date": d,
            "Issue": "MARKET_DATE_NOT_IN_REDDIT_CALENDAR",
        }
    )

for d in reddit_not_in_market:
    unmatched_rows.append(
        {
            "Date": d,
            "Issue": "REDDIT_DATE_NOT_IN_MARKET_CALENDAR",
        }
    )

unmatched_df = pd.DataFrame(
    unmatched_rows,
    columns=[
        "Date",
        "Issue",
    ]
)

print(
    f"Market dates absent from Reddit calendar: "
    f"{len(market_not_in_reddit):,}"
)

print(
    f"Reddit dates absent from market calendar: "
    f"{len(reddit_not_in_market):,}"
)

if (
    len(market_not_in_reddit) != 0
    or len(reddit_not_in_market) != 0
):
    fail(
        "Reddit and market target calendars do not match."
    )

print("Date coverage reconciliation: PASS")


# =============================================================================
# 19. SELECT FORECAST-SAFE REDDIT FIELDS
# =============================================================================

section(
    "SELECTING FORECAST-SAFE REDDIT FIELDS"
)

reddit_safe_cols = safe_reddit_columns(
    reddit
)

require_columns(
    reddit,
    [
        "Date",
        "Asset",
        *REDDIT_PRIMARY_REQUIRED,
    ],
    "Forecast-safe Reddit selection"
)

print(
    f"Selected Reddit columns: "
    f"{len(reddit_safe_cols):,}"
)

for col in reddit_safe_cols:
    print(f"  {col}")

reddit_safe = (
    reddit[
        reddit_safe_cols
    ]
    .copy()
)


# =============================================================================
# 20. BUILD MARKET-MASTER DATE × ASSET TARGET GRID
# =============================================================================

section(
    "BUILDING MARKET-MASTER DATE × ASSET GRID"
)

market_target_grid = (
    pd.MultiIndex.from_product(
        [
            market["Date"],
            ASSETS,
        ],
        names=[
            "Date",
            "Asset",
        ]
    )
    .to_frame(
        index=False
    )
)

print(
    f"Market calendar days: "
    f"{market['Date'].nunique():,}"
)

print(
    f"Assets: {ASSETS}"
)

print(
    f"Expected Date × Asset rows: "
    f"{EXPECTED_DATE_ASSET_ROWS:,}"
)

print(
    f"Constructed Date × Asset rows: "
    f"{len(market_target_grid):,}"
)

if len(market_target_grid) != EXPECTED_DATE_ASSET_ROWS:
    fail(
        "Market target Date × Asset grid has "
        "an unexpected number of rows."
    )

print("Market-master target grid: PASS")


# =============================================================================
# 21. MERGE REDDIT ONTO MARKET-MASTER TARGET GRID
# =============================================================================

section(
    "MERGING SECTION 06 REDDIT DATA"
)

combined = market_target_grid.merge(
    reddit_safe,
    on=[
        "Date",
        "Asset",
    ],
    how="left",
    validate="one_to_one",
    indicator="_reddit_merge"
)

reddit_merge_counts = (
    combined["_reddit_merge"]
    .value_counts(dropna=False)
)

print(reddit_merge_counts)

reddit_unmatched_target_rows = int(
    (
        combined["_reddit_merge"]
        != "both"
    ).sum()
)

if reddit_unmatched_target_rows != 0:
    fail(
        f"{reddit_unmatched_target_rows:,} market target rows "
        "failed to match a Section 06 Reddit calendar row."
    )

combined = combined.drop(
    columns="_reddit_merge"
)

print("Reddit target-grid merge: PASS")


# =============================================================================
# 22. MERGE MARKET INFORMATION
# =============================================================================

section(
    "MERGING INFORMATION-ALIGNED MARKET DATA"
)

combined = combined.merge(
    market,
    on="Date",
    how="left",
    validate="many_to_one",
    indicator="_market_merge"
)

market_merge_counts = (
    combined["_market_merge"]
    .value_counts(dropna=False)
)

print(market_merge_counts)

market_unmatched_rows = int(
    (
        combined["_market_merge"]
        != "both"
    ).sum()
)

if market_unmatched_rows != 0:
    fail(
        f"{market_unmatched_rows:,} target rows failed "
        "to match the market dataset."
    )

combined = combined.drop(
    columns="_market_merge"
)

print("Market merge: PASS")


# =============================================================================
# 23. CREATE ASSET-SPECIFIC TARGET / CONTROL VARIABLES
# =============================================================================

section(
    "CREATING ASSET-SPECIFIC MODELLING VARIABLES"
)

btc_mask = (
    combined["Asset"] == "BTC"
)

eth_mask = (
    combined["Asset"] == "ETH"
)

# -------------------------------------------------------------------------
# Dependent variable
# -------------------------------------------------------------------------

combined["Target_Return"] = np.where(
    btc_mask,
    combined["BTC_Return"],
    combined["ETH_Return"],
)

# -------------------------------------------------------------------------
# Own lagged cryptocurrency return
# -------------------------------------------------------------------------

combined["Own_Lagged_Return"] = np.where(
    btc_mask,
    combined["BTC_Lagged_Return"],
    combined["ETH_Lagged_Return"],
)

# -------------------------------------------------------------------------
# Cross-cryptocurrency lagged return for robustness
# -------------------------------------------------------------------------

combined["Cross_Crypto_Lagged_Return"] = np.where(
    btc_mask,
    combined["ETH_Lagged_Return"],
    combined["BTC_Lagged_Return"],
)

# -------------------------------------------------------------------------
# Lagged log cryptocurrency volume
# -------------------------------------------------------------------------

combined["Lagged_Log_Crypto_Volume"] = np.where(
    btc_mask,
    combined["Lagged_Log_BTC_Volume"],
    combined["Lagged_Log_ETH_Volume"],
)

# -------------------------------------------------------------------------
# Source dates for calendar-t−1 crypto controls
# -------------------------------------------------------------------------

calendar_t1_date = (
    combined["Date"]
    - pd.Timedelta(days=1)
)

combined["Own_Return_Source_Date"] = (
    calendar_t1_date
)

combined["Cross_Crypto_Return_Source_Date"] = (
    calendar_t1_date
)

combined["Crypto_Volume_Source_Date"] = (
    calendar_t1_date
)

# Source dates outside sample are not observed.
first_date_mask = (
    combined["Date"] == STUDY_START
)

combined.loc[
    first_date_mask,
    [
        "Own_Return_Source_Date",
        "Cross_Crypto_Return_Source_Date",
        "Crypto_Volume_Source_Date",
    ]
] = pd.NaT


# -------------------------------------------------------------------------
# Unified primary Reddit variables
# -------------------------------------------------------------------------

combined["Lagged_Reddit_Sentiment"] = np.where(
    btc_mask,
    combined["Lagged_BTC_Reddit_Sentiment"],
    combined["Lagged_ETH_Reddit_Sentiment"],
)

combined["Lagged_Log_Reddit_Post_Count"] = np.where(
    btc_mask,
    combined["Lagged_Log_BTC_Reddit_Post_Count"],
    combined["Lagged_Log_ETH_Reddit_Post_Count"],
)

combined["Reddit_Primary_Source_Date"] = (
    calendar_t1_date
)

combined.loc[
    first_date_mask,
    "Reddit_Primary_Source_Date"
] = pd.NaT

print("Target_Return created: PASS")
print("Own_Lagged_Return created: PASS")
print("Cross_Crypto_Lagged_Return created: PASS")
print("Lagged_Log_Crypto_Volume created: PASS")
print("Unified lagged Reddit sentiment created: PASS")
print("Unified lagged Reddit activity created: PASS")


# =============================================================================
# 24. REMOVE CONTEMPORANEOUS MARKET PREDICTORS FROM MODELLING OUTPUT
# =============================================================================

section(
    "ISOLATING FORECAST-SAFE MARKET PREDICTORS"
)

# These columns were needed above for validation and target construction.
# They are removed from the final Section 07 modelling-facing dataset to
# reduce the risk that future scripts accidentally use contemporaneous
# traditional controls or same-day crypto volume as predictive regressors.

DROP_FROM_FINAL = [
    # Same-day crypto volume
    "Log_BTC_Volume",
    "Log_ETH_Volume",

    # Wide target/lag fields already converted into asset-specific columns
    "BTC_Return",
    "BTC_Lagged_Return",
    "ETH_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_Log_ETH_Volume",

    # Contemporaneous traditional-market transformations
    "SP500_Return",
    "VIX_Change",
    "Gold_Return",
    "DXY_Return",
    "US10Y_Change",

    # Old ordinary trading-row lag fields are not the forecast-safe
    # 24/7-aligned controls and should not be used downstream.
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]

drop_existing = [
    col for col in DROP_FROM_FINAL
    if col in combined.columns
]

combined = combined.drop(
    columns=drop_existing
)

print(
    f"Removed {len(drop_existing):,} "
    "non-modelling / potentially confusing market columns."
)

print(
    "Forecast-safe aligned traditional controls retained: PASS"
)


# =============================================================================
# 25. EXPLICIT NO-LOOK-AHEAD CHECKS AFTER INTEGRATION
# =============================================================================

section(
    "RUNNING INTEGRATED NO-LOOK-AHEAD CHECKS"
)

integrated_timing_rows = []

# -------------------------------------------------------------------------
# Reddit t-1
# -------------------------------------------------------------------------

reddit_source_present = (
    combined["Reddit_Primary_Source_Date"].notna()
)

reddit_bad = (
    reddit_source_present
    & (
        combined["Reddit_Primary_Source_Date"]
        >= combined["Date"]
    )
)

reddit_bad_gap = (
    reddit_source_present
    & (
        (
            combined["Date"]
            - combined["Reddit_Primary_Source_Date"]
        ).dt.days
        != 1
    )
)

integrated_timing_rows.append(
    {
        "Variable_Group": "Reddit_Primary_t1",
        "Rows_Checked": int(
            reddit_source_present.sum()
        ),
        "Same_Day_Or_Future_Failures": int(
            reddit_bad.sum()
        ),
        "Incorrect_Gap_Failures": int(
            reddit_bad_gap.sum()
        ),
        "Pass": (
            int(reddit_bad.sum()) == 0
            and int(reddit_bad_gap.sum()) == 0
        ),
    }
)

# -------------------------------------------------------------------------
# Own-return t-1
# -------------------------------------------------------------------------

own_source_present = (
    combined["Own_Return_Source_Date"].notna()
)

own_bad = (
    own_source_present
    & (
        combined["Own_Return_Source_Date"]
        >= combined["Date"]
    )
)

own_bad_gap = (
    own_source_present
    & (
        (
            combined["Date"]
            - combined["Own_Return_Source_Date"]
        ).dt.days
        != 1
    )
)

integrated_timing_rows.append(
    {
        "Variable_Group": "Own_Crypto_Return_t1",
        "Rows_Checked": int(
            own_source_present.sum()
        ),
        "Same_Day_Or_Future_Failures": int(
            own_bad.sum()
        ),
        "Incorrect_Gap_Failures": int(
            own_bad_gap.sum()
        ),
        "Pass": (
            int(own_bad.sum()) == 0
            and int(own_bad_gap.sum()) == 0
        ),
    }
)

# -------------------------------------------------------------------------
# Crypto volume t-1
# -------------------------------------------------------------------------

volume_source_present = (
    combined["Crypto_Volume_Source_Date"].notna()
)

volume_bad = (
    volume_source_present
    & (
        combined["Crypto_Volume_Source_Date"]
        >= combined["Date"]
    )
)

volume_bad_gap = (
    volume_source_present
    & (
        (
            combined["Date"]
            - combined["Crypto_Volume_Source_Date"]
        ).dt.days
        != 1
    )
)

integrated_timing_rows.append(
    {
        "Variable_Group": "Crypto_Volume_t1",
        "Rows_Checked": int(
            volume_source_present.sum()
        ),
        "Same_Day_Or_Future_Failures": int(
            volume_bad.sum()
        ),
        "Incorrect_Gap_Failures": int(
            volume_bad_gap.sum()
        ),
        "Pass": (
            int(volume_bad.sum()) == 0
            and int(volume_bad_gap.sum()) == 0
        ),
    }
)

# -------------------------------------------------------------------------
# Traditional controls
# -------------------------------------------------------------------------

for name, cfg in TRADITIONAL_ALIGNMENT.items():

    source_col = cfg["source"]

    source_present = (
        combined[source_col].notna()
    )

    bad = (
        source_present
        & (
            combined[source_col]
            >= combined["Date"]
        )
    )

    integrated_timing_rows.append(
        {
            "Variable_Group":
                f"{name}_Strict_PreTarget",
            "Rows_Checked": int(
                source_present.sum()
            ),
            "Same_Day_Or_Future_Failures": int(
                bad.sum()
            ),
            "Incorrect_Gap_Failures": 0,
            "Pass": int(bad.sum()) == 0,
        }
    )

integrated_timing_df = pd.DataFrame(
    integrated_timing_rows
)

print(
    integrated_timing_df.to_string(
        index=False
    )
)

if not integrated_timing_df["Pass"].all():
    fail(
        "Integrated no-look-ahead validation failed."
    )

print(
    "\nIntegrated no-look-ahead validation: PASS"
)


# =============================================================================
# 26. DUPLICATE / ROW-COUNT VALIDATION AFTER MERGE
# =============================================================================

section(
    "VALIDATING FINAL DATE × ASSET STRUCTURE"
)

final_dup = int(
    combined.duplicated(
        ["Date", "Asset"]
    ).sum()
)

final_rows = len(combined)

final_dates = combined["Date"].nunique()

btc_rows = int(
    (combined["Asset"] == "BTC").sum()
)

eth_rows = int(
    (combined["Asset"] == "ETH").sum()
)

print(
    f"Final rows: {final_rows:,}"
)

print(
    f"Unique dates: {final_dates:,}"
)

print(
    f"BTC rows: {btc_rows:,}"
)

print(
    f"ETH rows: {eth_rows:,}"
)

print(
    f"Duplicate Date × Asset rows: {final_dup:,}"
)

if final_rows != EXPECTED_DATE_ASSET_ROWS:
    fail(
        "Final row count differs from expected "
        "3,652 Date × Asset observations."
    )

if final_dates != EXPECTED_CALENDAR_DAYS:
    fail(
        "Final calendar does not contain 1,826 dates."
    )

if btc_rows != EXPECTED_CALENDAR_DAYS:
    fail(
        "BTC target rows do not equal 1,826."
    )

if eth_rows != EXPECTED_CALENDAR_DAYS:
    fail(
        "ETH target rows do not equal 1,826."
    )

if final_dup != 0:
    fail(
        "Final dataset contains duplicate Date × Asset rows."
    )

print("Final structural validation: PASS")


# =============================================================================
# 27. TARGET PRESERVATION VALIDATION
# =============================================================================

section(
    "VALIDATING MARKET TARGET PRESERVATION"
)

btc_original = (
    market[
        ["Date", "BTC_Return"]
    ]
    .rename(
        columns={
            "BTC_Return":
                "Expected_Target_Return"
        }
    )
)

eth_original = (
    market[
        ["Date", "ETH_Return"]
    ]
    .rename(
        columns={
            "ETH_Return":
                "Expected_Target_Return"
        }
    )
)

btc_final = (
    combined.loc[
        combined["Asset"] == "BTC",
        [
            "Date",
            "Target_Return",
        ]
    ]
    .merge(
        btc_original,
        on="Date",
        how="left",
        validate="one_to_one"
    )
)

eth_final = (
    combined.loc[
        combined["Asset"] == "ETH",
        [
            "Date",
            "Target_Return",
        ]
    ]
    .merge(
        eth_original,
        on="Date",
        how="left",
        validate="one_to_one"
    )
)

btc_target_match = (
    values_equal_with_nan(
        btc_final["Target_Return"],
        btc_final["Expected_Target_Return"]
    )
)

eth_target_match = (
    values_equal_with_nan(
        eth_final["Target_Return"],
        eth_final["Expected_Target_Return"]
    )
)

btc_target_mismatch = int(
    (~btc_target_match).sum()
)

eth_target_mismatch = int(
    (~eth_target_match).sum()
)

print(
    f"BTC target mismatches: "
    f"{btc_target_mismatch}"
)

print(
    f"ETH target mismatches: "
    f"{eth_target_mismatch}"
)

if btc_target_mismatch != 0:
    fail(
        "BTC market targets changed during integration."
    )

if eth_target_mismatch != 0:
    fail(
        "ETH market targets changed during integration."
    )

print("Market target preservation: PASS")


# =============================================================================
# 28. PRIMARY REDDIT PREDICTOR COVERAGE
# =============================================================================

section(
    "SUMMARISING PRIMARY REDDIT PREDICTOR COVERAGE"
)

coverage_rows = []

for asset in ASSETS:

    sub = combined.loc[
        combined["Asset"] == asset
    ]

    coverage_rows.append(
        {
            "Asset": asset,
            "Calendar_Rows": len(sub),
            "Target_Return_Available":
                int(
                    sub["Target_Return"]
                    .notna()
                    .sum()
                ),
            "Own_Lagged_Return_Available":
                int(
                    sub["Own_Lagged_Return"]
                    .notna()
                    .sum()
                ),
            "Lagged_Crypto_Volume_Available":
                int(
                    sub["Lagged_Log_Crypto_Volume"]
                    .notna()
                    .sum()
                ),
            "Lagged_Reddit_Activity_Available":
                int(
                    sub[
                        "Lagged_Log_Reddit_Post_Count"
                    ]
                    .notna()
                    .sum()
                ),
            "Lagged_Reddit_Sentiment_Available":
                int(
                    sub[
                        "Lagged_Reddit_Sentiment"
                    ]
                    .notna()
                    .sum()
                ),
            "SP500_Aligned_Available":
                int(
                    sub[
                        "Lagged_SP500_Return_Aligned"
                    ]
                    .notna()
                    .sum()
                ),
            "VIX_Aligned_Available":
                int(
                    sub[
                        "Lagged_VIX_Change_Aligned"
                    ]
                    .notna()
                    .sum()
                ),
            "Gold_Aligned_Available":
                int(
                    sub[
                        "Lagged_Gold_Return_Aligned"
                    ]
                    .notna()
                    .sum()
                ),
            "DXY_Aligned_Available":
                int(
                    sub[
                        "Lagged_DXY_Return_Aligned"
                    ]
                    .notna()
                    .sum()
                ),
            "US10Y_Aligned_Available":
                int(
                    sub[
                        "Lagged_US10Y_Change_Aligned"
                    ]
                    .notna()
                    .sum()
                ),
        }
    )

coverage_df = pd.DataFrame(
    coverage_rows
)

print(
    coverage_df.to_string(
        index=False
    )
)


# =============================================================================
# 29. BUILD HARD QC TABLE
# =============================================================================

section(
    "BUILDING SECTION 07 HARD QC"
)

qc_rows = []

add_qc(
    qc_rows,
    "Market_Input_Rows",
    EXPECTED_MARKET_ROWS,
    len(market),
    len(market) == EXPECTED_MARKET_ROWS
)

add_qc(
    qc_rows,
    "Reddit_Input_Rows",
    EXPECTED_REDDIT_ROWS,
    len(reddit),
    len(reddit) == EXPECTED_REDDIT_ROWS
)

add_qc(
    qc_rows,
    "Calendar_Days",
    EXPECTED_CALENDAR_DAYS,
    final_dates,
    final_dates == EXPECTED_CALENDAR_DAYS
)

add_qc(
    qc_rows,
    "Date_Asset_Rows",
    EXPECTED_DATE_ASSET_ROWS,
    final_rows,
    final_rows == EXPECTED_DATE_ASSET_ROWS
)

add_qc(
    qc_rows,
    "BTC_Rows",
    EXPECTED_CALENDAR_DAYS,
    btc_rows,
    btc_rows == EXPECTED_CALENDAR_DAYS
)

add_qc(
    qc_rows,
    "ETH_Rows",
    EXPECTED_CALENDAR_DAYS,
    eth_rows,
    eth_rows == EXPECTED_CALENDAR_DAYS
)

add_qc(
    qc_rows,
    "Duplicate_Date_Asset_Rows",
    0,
    final_dup,
    final_dup == 0
)

add_qc(
    qc_rows,
    "Market_Dates_Unmatched_To_Reddit",
    0,
    len(market_not_in_reddit),
    len(market_not_in_reddit) == 0
)

add_qc(
    qc_rows,
    "Reddit_Dates_Unmatched_To_Market",
    0,
    len(reddit_not_in_market),
    len(reddit_not_in_market) == 0
)

add_qc(
    qc_rows,
    "Market_Target_Rows_Unmatched_To_Reddit",
    0,
    reddit_unmatched_target_rows,
    reddit_unmatched_target_rows == 0
)

add_qc(
    qc_rows,
    "Market_Rows_Unmatched_After_Merge",
    0,
    market_unmatched_rows,
    market_unmatched_rows == 0
)

add_qc(
    qc_rows,
    "BTC_Target_Mismatches",
    0,
    btc_target_mismatch,
    btc_target_mismatch == 0
)

add_qc(
    qc_rows,
    "ETH_Target_Mismatches",
    0,
    eth_target_mismatch,
    eth_target_mismatch == 0
)

add_qc(
    qc_rows,
    "BTC_Return_Lag_Reconstruction_Mismatches",
    0,
    btc_ret_mismatch,
    btc_ret_mismatch == 0
)

add_qc(
    qc_rows,
    "ETH_Return_Lag_Reconstruction_Mismatches",
    0,
    eth_ret_mismatch,
    eth_ret_mismatch == 0
)

add_qc(
    qc_rows,
    "BTC_Volume_Lag_Reconstruction_Mismatches",
    0,
    btc_vol_mismatch,
    btc_vol_mismatch == 0
)

add_qc(
    qc_rows,
    "ETH_Volume_Lag_Reconstruction_Mismatches",
    0,
    eth_vol_mismatch,
    eth_vol_mismatch == 0
)

add_qc(
    qc_rows,
    "Traditional_Market_Alignment_Tests",
    True,
    bool(timing_df["Pass"].all()),
    bool(timing_df["Pass"].all())
)

add_qc(
    qc_rows,
    "Integrated_No_Lookahead_Tests",
    True,
    bool(
        integrated_timing_df["Pass"].all()
    ),
    bool(
        integrated_timing_df["Pass"].all()
    )
)

qc_df = pd.DataFrame(
    qc_rows
)

print(
    qc_df.to_string(
        index=False
    )
)

if not qc_df["Pass"].all():
    fail(
        "One or more Section 07 hard QC tests failed."
    )

print(
    "\nSection 07 hard QC: PASS"
)


# =============================================================================
# 30. ORDER FINAL OUTPUT
# =============================================================================

section(
    "ORDERING COMBINED MARKET + REDDIT DATASET"
)

preferred_front = [
    "Date",
    "Asset",

    # Dependent variable
    "Target_Return",

    # Core benchmark crypto predictors
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",

    # Cross-crypto robustness control
    "Cross_Crypto_Lagged_Return",

    # Primary Reddit predictors
    "Lagged_Reddit_Sentiment",
    "Lagged_Log_Reddit_Post_Count",

    # Aligned traditional controls
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",

    # Predictor source dates
    "Reddit_Primary_Source_Date",
    "Own_Return_Source_Date",
    "Crypto_Volume_Source_Date",
    "Cross_Crypto_Return_Source_Date",

    "SP500_Source_Date",
    "VIX_Source_Date",
    "Gold_Source_Date",
    "DXY_Source_Date",
    "US10Y_Source_Date",

    # Traditional information ages
    "SP500_Information_Age_Days",
    "VIX_Information_Age_Days",
    "Gold_Information_Age_Days",
    "DXY_Information_Age_Days",
    "US10Y_Information_Age_Days",
]

preferred_front = [
    col for col in preferred_front
    if col in combined.columns
]

remaining = [
    col for col in combined.columns
    if col not in preferred_front
]

combined = combined[
    preferred_front
    + remaining
]

combined = combined.sort_values(
    [
        "Date",
        "Asset",
    ]
).reset_index(drop=True)

print(
    f"Final columns: {len(combined.columns):,}"
)


# =============================================================================
# 31. VARIABLE DICTIONARY
# =============================================================================

section(
    "CREATING SECTION 07 VARIABLE DICTIONARY"
)

variable_dictionary = [
    {
        "Variable": "Target_Return",
        "Role": "Dependent variable",
        "Timing":
            "Return on target calendar day t",
        "Description":
            "BTC return for BTC rows and ETH return for ETH rows.",
    },
    {
        "Variable": "Own_Lagged_Return",
        "Role": "Benchmark predictor",
        "Timing":
            "Previous calendar-day cryptocurrency return",
        "Description":
            "Asset's own lagged return.",
    },
    {
        "Variable": "Lagged_Log_Crypto_Volume",
        "Role": "Benchmark predictor",
        "Timing":
            "Previous calendar-day information",
        "Description":
            "Lagged log-transformed Yahoo Finance-reported "
            "cryptocurrency volume.",
    },
    {
        "Variable": "Cross_Crypto_Lagged_Return",
        "Role": "Robustness predictor",
        "Timing":
            "Previous calendar-day information",
        "Description":
            "Lagged ETH return in BTC rows and lagged BTC return "
            "in ETH rows.",
    },
    {
        "Variable": "Lagged_Reddit_Sentiment",
        "Role": "Primary sentiment predictor",
        "Timing":
            "t-1 calendar day",
        "Description":
            "Lagged Reddit textual sentiment among the selected "
            "asset-related Reddit communities.",
    },
    {
        "Variable": "Lagged_Log_Reddit_Post_Count",
        "Role": "Reddit activity predictor",
        "Timing":
            "t-1 calendar day",
        "Description":
            "log(1 + retained Reddit post count), lagged one "
            "calendar day.",
    },
    {
        "Variable": "Lagged_SP500_Return_Aligned",
        "Role": "Traditional control",
        "Timing":
            "Latest completed observation strictly before target",
        "Description":
            "Forecast-safe aligned S&P 500 return.",
    },
    {
        "Variable": "Lagged_VIX_Change_Aligned",
        "Role": "Traditional control",
        "Timing":
            "Latest completed observation strictly before target",
        "Description":
            "Forecast-safe aligned daily VIX change.",
    },
    {
        "Variable": "Lagged_Gold_Return_Aligned",
        "Role": "Traditional control",
        "Timing":
            "Latest completed observation strictly before target",
        "Description":
            "Forecast-safe aligned gold return.",
    },
    {
        "Variable": "Lagged_DXY_Return_Aligned",
        "Role": "Traditional control",
        "Timing":
            "Latest completed observation strictly before target",
        "Description":
            "Forecast-safe aligned DXY return.",
    },
    {
        "Variable": "Lagged_US10Y_Change_Aligned",
        "Role": "Traditional control",
        "Timing":
            "Latest completed observation strictly before target",
        "Description":
            "Forecast-safe aligned US 10-year Treasury yield change.",
    },
]

variable_df = pd.DataFrame(
    variable_dictionary
)


# =============================================================================
# 32. METHODOLOGY NOTE
# =============================================================================

section(
    "WRITING SECTION 07 METHODOLOGY NOTE"
)

methodology_note = """
SECTION 07 — REDDIT–MARKET DATA INTEGRATION
===========================================

Purpose
-------
Section 07 integrates the forecast-ready Reddit variables produced in
Section 06 with the information-aligned cryptocurrency and traditional
financial-market dataset.

The integration preserves the cryptocurrency target calendar rather than
restricting the sample to conventional financial-market trading days.

Study period
------------
2021-01-01 to 2025-12-31.

Assets
------
Bitcoin (BTC)
Ethereum (ETH)

Target calendar
---------------
Cryptocurrency markets operate continuously. The market-side master calendar
therefore contains every calendar day in the study period.

The final integrated structure contains:

    1,826 calendar dates
    2 assets
    3,652 Date × Asset observations

The market target calendar is treated as the master structure. Reddit data
are merged onto that structure rather than using an inner merge that could
remove target observations.

Dependent variable
------------------
For each asset row:

    Target_Return

equals the BTC daily return for BTC observations and the ETH daily return for
ETH observations.

The target return is not altered during the Reddit merge.

Cryptocurrency autoregressive control
-------------------------------------
Predictive specifications use the cryptocurrency's own lagged return rather
than its target-day return as an explanatory variable.

The previous calendar-day return is represented as:

    Own_Lagged_Return

Cross-cryptocurrency robustness
-------------------------------
A robustness variable is retained:

    Cross_Crypto_Lagged_Return

For BTC observations this contains the lagged ETH return.
For ETH observations this contains the lagged BTC return.

This permits later robustness analysis of common cryptocurrency-market
conditions without using contemporaneous cross-asset returns.

Cryptocurrency trading volume
-----------------------------
Trading volume comes from the existing Yahoo Finance-based cryptocurrency
volume series.

The modelling transformation is:

    log(1 + volume)

Predictive specifications use:

    Lagged_Log_Crypto_Volume

rather than target-day volume.

This reflects the information-timing requirement that a predictor used to
forecast the target-day return must be known before that return is realised.

Reddit predictors
-----------------
The primary Reddit variables are:

    Lagged_Reddit_Sentiment
    Lagged_Log_Reddit_Post_Count

They refer to calendar day t-1 when predicting the cryptocurrency return on
calendar day t.

Reddit sentiment represents textual sentiment among the selected Reddit
communities included in the dissertation and should not be described as
general investor sentiment.

Reddit activity and sentiment remain separate predictors so that later models
can distinguish discussion intensity from the tone of discussion.

Traditional-market controls
---------------------------
The traditional financial controls are:

    S&P 500 return
    VIX change
    Gold return
    DXY return
    US 10-year Treasury yield change

Because conventional financial markets do not operate on the same 24/7
calendar as cryptocurrencies, Section 07 does not merge same-date traditional
market observations as forecasting predictors.

Instead, it uses the information-aligned variables previously constructed
from the latest completed transformed observation strictly before the
cryptocurrency target date.

The predictive control variables are:

    Lagged_SP500_Return_Aligned
    Lagged_VIX_Change_Aligned
    Lagged_Gold_Return_Aligned
    Lagged_DXY_Return_Aligned
    Lagged_US10Y_Change_Aligned

Each aligned control is accompanied by its actual source date and information
age.

No-look-ahead protection
------------------------
For every traditional-market source date, Section 07 verifies:

    Source_Date < Target_Date

It independently reconstructs each aligned traditional control from the
transformed value recorded on the stated source date.

It also verifies that the stored information age equals:

    Target_Date - Source_Date

Reddit, cryptocurrency returns and cryptocurrency volume use calendar t-1
timing in the primary predictive design.

Same-day traditional-market controls and same-day cryptocurrency volume are
excluded from the modelling-facing Section 07 output to reduce the risk of
their accidental use in predictive specifications.

Explanatory versus predictive interpretation
---------------------------------------------
The integrated dataset supports both explanatory and predictive analyses,
but the two should not be interpreted interchangeably.

Statistical significance or increased in-sample R-squared may demonstrate
association or explanatory contribution. They do not establish improved
forecasting performance.

Out-of-sample forecasting must later compare models on identical held-out
forecast dates using metrics such as RMSE, MAE and out-of-sample R-squared.

Later model specifications
--------------------------
The integrated dataset permits the following main specification sequence:

    1. controls only
    2. controls + Reddit activity
    3. controls + Reddit sentiment
    4. controls + Reddit activity + Reddit sentiment

This allows the incremental contribution of textual sentiment to be evaluated
separately from the contribution of Reddit activity.

The dataset also preserves cross-cryptocurrency lagged returns for robustness
analysis and the Section 06 alternative Reddit lag structures for lag-length
robustness.

Section 07 does not estimate H1-H5.

Later inference and forecasting
--------------------------------
Explanatory regressions should use appropriate robust inference such as
HAC/Newey-West standard errors.

Predictive hypotheses should be evaluated using genuinely held-out
one-step-ahead forecasts.

BTC-versus-ETH sentiment differences should be assessed using a formal
coefficient-difference test rather than by comparing significance levels from
two independent regressions.

Structural stability, multicollinearity, alternative lag lengths and economic
significance are later modelling and robustness tasks.

SECTION 07 END
"""

METHODOLOGY_OUTPUT.write_text(
    methodology_note,
    encoding="utf-8"
)

print("Methodology note written: PASS")


# =============================================================================
# 33. SAVE OUTPUTS
# =============================================================================

section(
    "SAVING SECTION 07 OUTPUTS"
)

combined.to_csv(
    MAIN_OUTPUT,
    index=False
)

qc_df.to_csv(
    QC_OUTPUT,
    index=False
)

unmatched_df.to_csv(
    UNMATCHED_OUTPUT,
    index=False
)

timing_combined_output = pd.concat(
    [
        timing_df.assign(
            Validation_Type=
                "Traditional_Alignment_Reconstruction"
        ),
        integrated_timing_df.rename(
            columns={
                "Variable_Group":
                    "Variable"
            }
        ).assign(
            Validation_Type=
                "Integrated_No_Lookahead"
        ),
    ],
    ignore_index=True,
    sort=False
)

timing_combined_output.to_csv(
    TIMING_OUTPUT,
    index=False
)

coverage_df.to_csv(
    COVERAGE_OUTPUT,
    index=False
)

variable_df.to_csv(
    VARIABLE_OUTPUT,
    index=False
)

print("Saved:")
print(MAIN_OUTPUT)
print(QC_OUTPUT)
print(UNMATCHED_OUTPUT)
print(TIMING_OUTPUT)
print(COVERAGE_OUTPUT)
print(VARIABLE_OUTPUT)
print(METHODOLOGY_OUTPUT)


# =============================================================================
# 34. FINAL RELOAD VALIDATION
# =============================================================================

section(
    "FINAL OUTPUT RELOAD VALIDATION"
)

reloaded = pd.read_csv(
    MAIN_OUTPUT,
    low_memory=False
)

reload_rows = len(reloaded)

reload_dup = int(
    reloaded.duplicated(
        ["Date", "Asset"]
    ).sum()
)

print(
    f"Reloaded rows: {reload_rows:,}"
)

print(
    f"Reloaded duplicate Date × Asset rows: "
    f"{reload_dup:,}"
)

if reload_rows != EXPECTED_DATE_ASSET_ROWS:
    fail(
        "Reloaded final output has an unexpected row count."
    )

if reload_dup != 0:
    fail(
        "Reloaded final output contains duplicate "
        "Date × Asset observations."
    )

reload_required = [
    "Date",
    "Asset",
    "Target_Return",
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",
    "Lagged_Reddit_Sentiment",
    "Lagged_Log_Reddit_Post_Count",
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
]

require_columns(
    reloaded,
    reload_required,
    "Reloaded Section 07 output"
)

print("Final reload validation: PASS")


# =============================================================================
# 35. FINAL SUMMARY
# =============================================================================

section(
    "SECTION 07 COMPLETE — COMBINED MARKET + REDDIT DATASET"
)

print("\nStudy period:")
print(
    f"    {STUDY_START.date()} to "
    f"{STUDY_END.date()}"
)

print("\nCalendar days:")
print(
    f"    {EXPECTED_CALENDAR_DAYS:,}"
)

print("\nAssets:")
print("    BTC")
print("    ETH")

print("\nFinal Date × Asset rows:")
print(
    f"    {len(combined):,}"
)

print("\nPrimary dependent variable:")
print("    Target_Return")

print("\nPrimary benchmark predictors:")
print("    Own_Lagged_Return")
print("    Lagged_Log_Crypto_Volume")
print("    Lagged_SP500_Return_Aligned")
print("    Lagged_VIX_Change_Aligned")
print("    Lagged_Gold_Return_Aligned")
print("    Lagged_DXY_Return_Aligned")
print("    Lagged_US10Y_Change_Aligned")

print("\nPrimary Reddit activity predictor:")
print("    Lagged_Log_Reddit_Post_Count")

print("\nPrimary Reddit sentiment predictor:")
print("    Lagged_Reddit_Sentiment")

print("\nCross-crypto robustness predictor:")
print("    Cross_Crypto_Lagged_Return")

print("\nInformation timing:")
print("    Reddit: calendar t-1")
print("    Crypto return: calendar t-1")
print("    Crypto volume: calendar t-1")
print(
    "    Traditional controls: latest completed "
    "observation strictly before target"
)

print("\nHard QC:")
print("    PASS")

print("\nNo-look-ahead validation:")
print("    PASS")

print("\nMarket target preservation:")
print("    PASS")

print("\nMain combined output:")
print(f"    {MAIN_OUTPUT}")

print("\nNext stage:")
print(
    "    Section 08 — Final Modelling Dataset Validation"
)

print(
    "\nIMPORTANT:"
)

print(
    "    Section 08 should determine common complete-case "
    "samples for each model comparison."
)

print(
    "    Benchmark and Reddit-extended forecasting models "
    "must use identical OOS forecast dates."
)

print(
    "    Do not infer predictive improvement from "
    "in-sample statistical significance or R-squared."
)

print("\nSECTION 07: PASS")

sys.exit(0)