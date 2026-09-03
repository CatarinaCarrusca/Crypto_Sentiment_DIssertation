from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

input_file = project_folder / "data_clean" / "vix_clean.csv"
output_folder = project_folder / "data_processed"
output_file = output_folder / "vix_change.csv"

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("VIX CHANGE VARIABLE CONSTRUCTION")
print("=" * 70)

print("\nInput file:")
print(input_file)

print("\nDoes input file exist?")
print(input_file.exists())

if not input_file.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{input_file}"
    )


# ============================================================
# IMPORT CLEANED VIX DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED VIX DATA")
print("=" * 70)

df = pd.read_csv(input_file)

print("\nRaw imported shape:")
print(df.shape)

print("\nColumns found:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(df.tail(10).to_string(index=False))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = ["Date", "VIX"]

missing_columns = [
    col for col in required_columns
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Required columns missing: {missing_columns}"
    )


# ============================================================
# STANDARDISE DATE
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
        "Invalid dates were found in the cleaned VIX dataset."
    )


# ============================================================
# STANDARDISE VIX
# ============================================================

df["VIX"] = pd.to_numeric(
    df["VIX"],
    errors="coerce"
)

missing_vix = df["VIX"].isna().sum()

print("\nMissing VIX observations:")
print(missing_vix)

if missing_vix > 0:
    raise ValueError(
        "Missing/non-numeric VIX observations were found."
    )


# ============================================================
# SORT DATA
# ============================================================

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

if duplicate_dates > 0:
    raise ValueError(
        "Duplicate dates were found in the VIX dataset."
    )


# ============================================================
# NON-POSITIVE VALUE CHECK
# ============================================================

non_positive = (df["VIX"] <= 0).sum()

print("\nZero or negative VIX observations:")
print(non_positive)

if non_positive > 0:
    print(
        "\nWARNING: Zero or negative VIX values were found. "
        "These observations should be investigated."
    )


# ============================================================
# DATE RANGE CHECK
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
# CONSTRUCT VIX_CHANGE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING VIX_CHANGE")
print("=" * 70)

# First difference of VIX:
#
# VIX_Change_t = VIX_t - VIX_(t-1)
#
# Example:
# Previous VIX = 20
# Current VIX  = 22
# VIX_Change   = 22 - 20 = 2

df["VIX_Change"] = df["VIX"].diff()

print("\nVIX_Change constructed successfully.")

print("\nMissing VIX_Change observations:")
print(df["VIX_Change"].isna().sum())

print("\nInfinite VIX_Change observations:")
print(
    np.isinf(df["VIX_Change"]).sum()
)


# ============================================================
# CONSTRUCT LAGGED VIX CHANGE
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_VIX_CHANGE")
print("=" * 70)

# Previous observed VIX change:
#
# Lagged_VIX_Change_t = VIX_Change_(t-1)
#
# This is the variable intended for the main predictive model.

df["Lagged_VIX_Change"] = (
    df["VIX_Change"].shift(1)
)

print("\nLagged_VIX_Change constructed successfully.")

print("\nMissing Lagged_VIX_Change observations:")
print(
    df["Lagged_VIX_Change"].isna().sum()
)


# ============================================================
# CHECK TRADING-DAY GAPS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING VIX DATE GAPS")
print("=" * 70)

df["Calendar_Days_Since_Previous"] = (
    df["Date"].diff().dt.days
)

print("\nDistribution of calendar-day gaps:")
print(
    df["Calendar_Days_Since_Previous"]
    .value_counts()
    .sort_index()
)

print(
    "\nNOTE: Gaps in calendar dates can occur because VIX "
    "is not observed on the same continuous 24/7 calendar "
    "as Bitcoin and Ethereum."
)


# ============================================================
# CHECK FIRST 15 OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 15 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "VIX",
            "VIX_Change",
            "Lagged_VIX_Change",
            "Calendar_Days_Since_Previous",
        ]
    ]
    .head(15)
    .to_string(index=False)
)


# ============================================================
# CHECK LAST 15 OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LAST 15 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "VIX",
            "VIX_Change",
            "Lagged_VIX_Change",
            "Calendar_Days_Since_Previous",
        ]
    ]
    .tail(15)
    .to_string(index=False)
)


# ============================================================
# SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    df[
        [
            "VIX",
            "VIX_Change",
            "Lagged_VIX_Change",
        ]
    ].describe()
)


# ============================================================
# EXTREME CHANGE CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXTREME VIX CHANGE CHECK")
print("=" * 70)

# This is only a diagnostic.
# Large VIX changes should NOT automatically be deleted.

extreme_changes = df[
    df["VIX_Change"].abs() > 10
][
    [
        "Date",
        "VIX",
        "VIX_Change",
    ]
]

print("\nNumber of absolute VIX changes > 10 points:")
print(len(extreme_changes))

if len(extreme_changes) > 0:
    print("\nLarge VIX changes identified:")
    print(
        extreme_changes.to_string(index=False)
    )

print(
    "\nNOTE: Large VIX changes are flagged for validation only. "
    "They are NOT automatically deleted, winsorised, "
    "interpolated, or replaced."
)


# ============================================================
# REMOVE TEMPORARY CHECK COLUMN
# ============================================================

df = df.drop(
    columns=["Calendar_Days_Since_Previous"]
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED VIX DATA")
print("=" * 70)

df.to_csv(
    output_file,
    index=False
)

print("\nFile saved successfully:")
print(output_file)


# ============================================================
# FINAL CHECK
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
print(
    df[
        [
            "VIX",
            "VIX_Change",
            "Lagged_VIX_Change",
        ]
    ].isna().sum()
)


print("\n" + "=" * 70)
print("VIX CHANGE VARIABLE CONSTRUCTION COMPLETE")
print("=" * 70)