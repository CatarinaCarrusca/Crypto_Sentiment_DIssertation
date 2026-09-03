# ============================================================
# 07_construct_us10y_change.py
#
# Purpose:
# Construct the US 10-Year Treasury yield variables used in
# the dissertation regression analysis.
#
# Original variable:
#     Treasury_10Y
#
# Constructed variables:
#     US10Y_Change
#     Lagged_US10Y_Change
#
# Main predictive regression variable:
#     Lagged_US10Y_Change
#
# US10Y_Change_t = Treasury_10Y_t - Treasury_10Y_(t-1)
#
# Lagged_US10Y_Change_t = US10Y_Change_(t-1)
# ============================================================


from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data_clean"
    / "treasury10y_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "us10y_change_processed.csv"
)


print("=" * 70)
print("US 10-YEAR TREASURY YIELD VARIABLE CONSTRUCTION")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())


# Stop immediately if the cleaned file cannot be found
if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ============================================================
# 2. IMPORT CLEANED US10Y DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED US10Y DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nRaw imported shape:")
print(df.shape)

print("\nColumns found:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["Date", "Treasury_10Y"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing from dataset: "
        f"{missing_columns}"
    )


# ============================================================
# 4. CLEAN AND VALIDATE DATE
# ============================================================

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates were found in the cleaned US10Y dataset."
    )


# Sort chronologically before constructing differences/lags
df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# 5. VALIDATE TREASURY YIELD
# ============================================================

df["Treasury_10Y"] = pd.to_numeric(
    df["Treasury_10Y"],
    errors="coerce"
)

missing_values = df["Treasury_10Y"].isna().sum()

print("\nMissing Treasury_10Y observations:")
print(missing_values)

if missing_values > 0:
    raise ValueError(
        "Missing or non-numeric Treasury_10Y observations "
        "were found."
    )


duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

if duplicate_dates > 0:
    raise ValueError(
        "Duplicate dates were found in the US10Y dataset."
    )


# ============================================================
# 6. CHECK DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DATE RANGE")
print("=" * 70)

print("\nNumber of observations:")
print(len(df))

print("\nFirst date:")
print(df["Date"].min())

print("\nLast date:")
print(df["Date"].max())


# ============================================================
# 7. CONSTRUCT US10Y_CHANGE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING US10Y_CHANGE")
print("=" * 70)

# Daily change in the Treasury yield:
#
# US10Y_Change_t =
# Treasury_10Y_t - Treasury_10Y_(t-1)
#
# Example:
# 4.20 -> 4.30
# change = +0.10 percentage points
#        = +10 basis points

df["US10Y_Change"] = df["Treasury_10Y"].diff()


print("\nUS10Y_Change constructed successfully.")

print("\nMissing US10Y_Change observations:")
print(df["US10Y_Change"].isna().sum())

print("\nInfinite US10Y_Change observations:")
print(
    np.isinf(
        df["US10Y_Change"].dropna()
    ).sum()
)


# ============================================================
# 8. CONSTRUCT LAGGED_US10Y_CHANGE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_US10Y_CHANGE")
print("=" * 70)

# Previous available Treasury observation's yield change
df["Lagged_US10Y_Change"] = (
    df["US10Y_Change"].shift(1)
)


print(
    "\nLagged_US10Y_Change constructed successfully."
)

print("\nMissing Lagged_US10Y_Change observations:")
print(
    df["Lagged_US10Y_Change"].isna().sum()
)

print("\nInfinite Lagged_US10Y_Change observations:")
print(
    np.isinf(
        df["Lagged_US10Y_Change"].dropna()
    ).sum()
)


# ============================================================
# 9. CHECK TIMING OF OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING OBSERVATION TIMING")
print("=" * 70)

date_gaps = df["Date"].diff().dt.days

non_consecutive = (date_gaps.dropna() != 1).sum()

print("\nNumber of gaps greater than one calendar day:")
print(non_consecutive)

print(
    "\nNOTE: Treasury yields are normally observed on "
    "business/trading days rather than every calendar day."
)

print(
    "Therefore, shift(1) represents the previous AVAILABLE "
    "Treasury observation, not necessarily the previous "
    "calendar day."
)


# ============================================================
# 10. CHECK FIRST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "Treasury_10Y",
            "US10Y_Change",
            "Lagged_US10Y_Change"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# 11. CHECK LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LAST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "Treasury_10Y",
            "US10Y_Change",
            "Lagged_US10Y_Change"
        ]
    ]
    .tail(10)
    .to_string(index=False)
)


# ============================================================
# 12. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    df[
        [
            "Treasury_10Y",
            "US10Y_Change",
            "Lagged_US10Y_Change"
        ]
    ].describe()
)


# ============================================================
# 13. IDENTIFY LARGEST YIELD MOVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("LARGEST ABSOLUTE US10Y CHANGES")
print("=" * 70)

largest_changes = (
    df.loc[
        df["US10Y_Change"].notna(),
        [
            "Date",
            "Treasury_10Y",
            "US10Y_Change"
        ]
    ]
    .assign(
        Absolute_Change=lambda x:
        x["US10Y_Change"].abs()
    )
    .sort_values(
        "Absolute_Change",
        ascending=False
    )
    .head(10)
)

print(
    largest_changes.to_string(index=False)
)


# ============================================================
# 14. SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED US10Y DATA")
print("=" * 70)

# Ensure output directory exists
OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_columns = [
    "Date",
    "Treasury_10Y",
    "US10Y_Change",
    "Lagged_US10Y_Change"
]

df = df[final_columns]

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFile saved successfully:")
print(OUTPUT_FILE)


# ============================================================
# 15. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print("\nNumber of observations:")
print(len(df))

print("\nDate range:")
print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print("\nFinal variables:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isna().sum())


# Expected:
#
# Treasury_10Y               0
# US10Y_Change               1
# Lagged_US10Y_Change        2


# ============================================================
# 16. INTERPRETATION NOTE
# ============================================================

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)

print(
    "\nTreasury_10Y is expressed in percentage points."
)

print(
    "Therefore, US10Y_Change is also measured in "
    "percentage-point changes."
)

print(
    "For example, a yield increase from 4.20 to 4.30 "
    "produces US10Y_Change = 0.10, equivalent to "
    "a 10-basis-point increase."
)

print(
    "\nThe main predictive regression variable is:"
)

print("Lagged_US10Y_Change")


print("\n" + "=" * 70)
print("US 10-YEAR TREASURY YIELD CONSTRUCTION COMPLETE")
print("=" * 70)