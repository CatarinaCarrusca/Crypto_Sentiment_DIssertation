# ============================================================
# 08_construct_cross_crypto_returns.py
#
# Purpose:
# Combine the previously constructed BTC and ETH return
# variables into one dataset for cross-cryptocurrency
# robustness analysis.
#
# IMPORTANT:
# Returns and lagged returns have ALREADY been constructed in:
#
#   btc_returns.csv
#   eth_returns.csv
#
# Therefore, this script does NOT calculate additional lags.
#
# Robustness specifications:
#
# BTC model:
#   Dependent variable: BTC_Return
#   Own lag:           BTC_Lagged_Return
#   Cross-crypto lag:  ETH_Lagged_Return
#
# ETH model:
#   Dependent variable: ETH_Return
#   Own lag:           ETH_Lagged_Return
#   Cross-crypto lag:  BTC_Lagged_Return
#
# This addresses the potential omitted-variable issue arising
# from the close relationship between BTC and ETH.
# ============================================================


from pathlib import Path
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

BTC_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "btc_returns.csv"
)

ETH_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "eth_returns.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "cross_crypto_returns.csv"
)


print("=" * 70)
print("CROSS-CRYPTO RETURN DATASET CONSTRUCTION")
print("=" * 70)


# ============================================================
# 2. CHECK INPUT FILES
# ============================================================

print("\nBTC input file:")
print(BTC_FILE)

print("\nDoes BTC input file exist?")
print(BTC_FILE.exists())

print("\nETH input file:")
print(ETH_FILE)

print("\nDoes ETH input file exist?")
print(ETH_FILE.exists())


if not BTC_FILE.exists():
    raise FileNotFoundError(
        f"BTC return file not found:\n{BTC_FILE}"
    )

if not ETH_FILE.exists():
    raise FileNotFoundError(
        f"ETH return file not found:\n{ETH_FILE}"
    )


# ============================================================
# 3. IMPORT PROCESSED RETURN DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING PROCESSED BTC AND ETH RETURN DATA")
print("=" * 70)

btc = pd.read_csv(BTC_FILE)
eth = pd.read_csv(ETH_FILE)


print("\nBTC imported shape:")
print(btc.shape)

print("\nBTC columns:")
print(btc.columns.tolist())


print("\nETH imported shape:")
print(eth.shape)

print("\nETH columns:")
print(eth.columns.tolist())


# ============================================================
# 4. CHECK REQUIRED COLUMNS
# ============================================================

btc_required = [
    "Date",
    "BTC_Return",
    "BTC_Lagged_Return"
]

eth_required = [
    "Date",
    "ETH_Return",
    "ETH_Lagged_Return"
]


btc_missing_columns = [
    column
    for column in btc_required
    if column not in btc.columns
]

eth_missing_columns = [
    column
    for column in eth_required
    if column not in eth.columns
]


if btc_missing_columns:
    raise ValueError(
        "Missing required BTC columns: "
        f"{btc_missing_columns}"
    )

if eth_missing_columns:
    raise ValueError(
        "Missing required ETH columns: "
        f"{eth_missing_columns}"
    )


print("\nAll required BTC and ETH columns are present.")


# ============================================================
# 5. CONVERT DATES
# ============================================================

print("\n" + "=" * 70)
print("VALIDATING DATES")
print("=" * 70)


btc["Date"] = pd.to_datetime(
    btc["Date"],
    errors="coerce"
)

eth["Date"] = pd.to_datetime(
    eth["Date"],
    errors="coerce"
)


btc_invalid_dates = btc["Date"].isna().sum()
eth_invalid_dates = eth["Date"].isna().sum()


print("\nInvalid BTC dates:")
print(btc_invalid_dates)

print("\nInvalid ETH dates:")
print(eth_invalid_dates)


if btc_invalid_dates > 0:
    raise ValueError(
        "Invalid dates found in BTC return dataset."
    )

if eth_invalid_dates > 0:
    raise ValueError(
        "Invalid dates found in ETH return dataset."
    )


# ============================================================
# 6. CHECK DUPLICATE DATES
# ============================================================

btc_duplicates = btc["Date"].duplicated().sum()
eth_duplicates = eth["Date"].duplicated().sum()


print("\nDuplicate BTC dates:")
print(btc_duplicates)

print("\nDuplicate ETH dates:")
print(eth_duplicates)


if btc_duplicates > 0:
    raise ValueError(
        "Duplicate dates found in BTC return dataset."
    )

if eth_duplicates > 0:
    raise ValueError(
        "Duplicate dates found in ETH return dataset."
    )


# ============================================================
# 7. SORT DATA CHRONOLOGICALLY
# ============================================================

btc = (
    btc
    .sort_values("Date")
    .reset_index(drop=True)
)

eth = (
    eth
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# 8. CHECK DATE RANGES BEFORE MERGE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATE RANGES BEFORE MERGE")
print("=" * 70)


print("\nBTC observations:")
print(len(btc))

print("\nBTC date range:")
print(
    btc["Date"].min(),
    "to",
    btc["Date"].max()
)


print("\nETH observations:")
print(len(eth))

print("\nETH date range:")
print(
    eth["Date"].min(),
    "to",
    eth["Date"].max()
)


# ============================================================
# 9. KEEP ONLY REQUIRED RETURN VARIABLES
# ============================================================

btc = btc[
    [
        "Date",
        "BTC_Return",
        "BTC_Lagged_Return"
    ]
].copy()


eth = eth[
    [
        "Date",
        "ETH_Return",
        "ETH_Lagged_Return"
    ]
].copy()


# ============================================================
# 10. MERGE BTC AND ETH DATA BY DATE
# ============================================================

print("\n" + "=" * 70)
print("MERGING BTC AND ETH RETURN DATA")
print("=" * 70)


cross_crypto = pd.merge(
    btc,
    eth,
    on="Date",
    how="inner",
    validate="one_to_one"
)


cross_crypto = (
    cross_crypto
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\nMerge completed successfully.")

print("\nMerged dataset shape:")
print(cross_crypto.shape)


print("\nMerged date range:")
print(
    cross_crypto["Date"].min(),
    "to",
    cross_crypto["Date"].max()
)


# ============================================================
# 11. CHECK WHETHER ANY DATES WERE LOST
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATE MATCHING")
print("=" * 70)


btc_dates = set(btc["Date"])
eth_dates = set(eth["Date"])


btc_only_dates = btc_dates - eth_dates
eth_only_dates = eth_dates - btc_dates


print("\nDates present in BTC but not ETH:")
print(len(btc_only_dates))

print("\nDates present in ETH but not BTC:")
print(len(eth_only_dates))


if len(btc_only_dates) == 0 and len(eth_only_dates) == 0:
    print(
        "\nBTC and ETH datasets contain exactly "
        "the same dates."
    )
else:
    print(
        "\nWARNING: BTC and ETH date coverage differs."
    )


# ============================================================
# 12. CHECK CALENDAR CONTINUITY
# ============================================================

print("\n" + "=" * 70)
print("CHECKING CALENDAR CONTINUITY")
print("=" * 70)


date_differences = (
    cross_crypto["Date"]
    .diff()
    .dt.days
)

non_consecutive = (
    date_differences
    .dropna()
    .ne(1)
    .sum()
)


print("\nNon-consecutive calendar observations:")
print(non_consecutive)


if non_consecutive == 0:
    print(
        "\nAll BTC/ETH observations are consecutive "
        "calendar days."
    )
else:
    print(
        "\nWARNING: Gaps exist in the merged "
        "crypto dataset."
    )


# ============================================================
# 13. VERIFY EXISTING BTC LAG
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING BTC LAG CONSTRUCTION")
print("=" * 70)


btc_expected_lag = (
    cross_crypto["BTC_Return"]
    .shift(1)
)


btc_lag_difference = (
    cross_crypto["BTC_Lagged_Return"]
    - btc_expected_lag
)


btc_lag_mismatches = (
    btc_lag_difference
    .dropna()
    .abs()
    .gt(1e-12)
    .sum()
)


print("\nBTC lag mismatches:")
print(btc_lag_mismatches)


if btc_lag_mismatches == 0:
    print(
        "BTC_Lagged_Return is correctly aligned "
        "with BTC_Return(t-1)."
    )
else:
    raise ValueError(
        "BTC_Lagged_Return does not correctly match "
        "BTC_Return(t-1)."
    )


# ============================================================
# 14. VERIFY EXISTING ETH LAG
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING ETH LAG CONSTRUCTION")
print("=" * 70)


eth_expected_lag = (
    cross_crypto["ETH_Return"]
    .shift(1)
)


eth_lag_difference = (
    cross_crypto["ETH_Lagged_Return"]
    - eth_expected_lag
)


eth_lag_mismatches = (
    eth_lag_difference
    .dropna()
    .abs()
    .gt(1e-12)
    .sum()
)


print("\nETH lag mismatches:")
print(eth_lag_mismatches)


if eth_lag_mismatches == 0:
    print(
        "ETH_Lagged_Return is correctly aligned "
        "with ETH_Return(t-1)."
    )
else:
    raise ValueError(
        "ETH_Lagged_Return does not correctly match "
        "ETH_Return(t-1)."
    )


# ============================================================
# 15. CHECK FIRST 10 OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)


print(
    cross_crypto
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 16. CHECK LAST 10 OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LAST 10 OBSERVATIONS")
print("=" * 70)


print(
    cross_crypto
    .tail(10)
    .to_string(index=False)
)


# ============================================================
# 17. CHECK MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("CHECKING MISSING VALUES")
print("=" * 70)


print("\nMissing values:")
print(
    cross_crypto.isna().sum()
)


# Expected:
#
# BTC_Return             1
# BTC_Lagged_Return      2
# ETH_Return             1
# ETH_Lagged_Return      2
#
# These are expected because:
#
# Return(t) requires Price(t-1)
# Lagged_Return(t) requires Return(t-1)


# ============================================================
# 18. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)


print(
    cross_crypto[
        [
            "BTC_Return",
            "BTC_Lagged_Return",
            "ETH_Return",
            "ETH_Lagged_Return"
        ]
    ].describe()
)


# ============================================================
# 19. BTC-ETH RETURN CORRELATION
# ============================================================

print("\n" + "=" * 70)
print("BTC-ETH RETURN CORRELATION")
print("=" * 70)


return_correlation = (
    cross_crypto[
        [
            "BTC_Return",
            "ETH_Return"
        ]
    ]
    .corr()
)


print(return_correlation)


# ============================================================
# 20. SAVE CROSS-CRYPTO DATASET
# ============================================================

print("\n" + "=" * 70)
print("SAVING CROSS-CRYPTO RETURN DATASET")
print("=" * 70)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


cross_crypto.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFile saved successfully:")
print(OUTPUT_FILE)


# ============================================================
# 21. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)


print("\nNumber of observations:")
print(len(cross_crypto))


print("\nDate range:")
print(
    cross_crypto["Date"].min(),
    "to",
    cross_crypto["Date"].max()
)


print("\nFinal variables:")
print(
    cross_crypto.columns.tolist()
)


print("\nMissing values:")
print(
    cross_crypto.isna().sum()
)


# ============================================================
# 22. REGRESSION USE
# ============================================================

print("\n" + "=" * 70)
print("REGRESSION USE")
print("=" * 70)


print(
    "\nBTC robustness model:"
)

print(
    "Dependent variable: BTC_Return"
)

print(
    "Own lag control: BTC_Lagged_Return"
)

print(
    "Cross-crypto robustness control: "
    "ETH_Lagged_Return"
)


print(
    "\nETH robustness model:"
)

print(
    "Dependent variable: ETH_Return"
)

print(
    "Own lag control: ETH_Lagged_Return"
)

print(
    "Cross-crypto robustness control: "
    "BTC_Lagged_Return"
)


print(
    "\nIMPORTANT: No additional lag has been created "
    "in this script."
)

print(
    "BTC_Lagged_Return and ETH_Lagged_Return were "
    "already constructed as t-1 variables."
)


print("\n" + "=" * 70)
print("CROSS-CRYPTO RETURN DATASET CONSTRUCTION COMPLETE")
print("=" * 70)