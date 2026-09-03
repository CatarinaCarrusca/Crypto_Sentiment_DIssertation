from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

input_file = project_folder / "data_clean" / "dxy_clean.csv"
output_folder = project_folder / "data_processed"
output_file = output_folder / "dxy_returns.csv"

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("DXY RETURN VARIABLE CONSTRUCTION")
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
# IMPORT CLEANED DXY DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED DXY DATA")
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

required_columns = ["Date", "DXY"]

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
        "Invalid dates were found in the cleaned DXY dataset."
    )


# ============================================================
# STANDARDISE DXY
# ============================================================

df["DXY"] = pd.to_numeric(
    df["DXY"],
    errors="coerce"
)

missing_values = df["DXY"].isna().sum()

print("\nMissing DXY observations:")
print(missing_values)

if missing_values > 0:
    raise ValueError(
        "Missing/non-numeric DXY observations were found."
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
        "Duplicate dates were found in the DXY dataset."
    )


# ============================================================
# NON-POSITIVE VALUE CHECK
# ============================================================

non_positive = (df["DXY"] <= 0).sum()

print("\nZero or negative DXY observations:")
print(non_positive)

if non_positive > 0:
    raise ValueError(
        "DXY observations must be positive before "
        "log returns can be calculated."
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
# CONSTRUCT DXY_RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING DXY_RETURN")
print("=" * 70)

# Daily log return:
#
# DXY_Return_t = ln(DXY_t / DXY_(t-1))

df["DXY_Return"] = np.log(
    df["DXY"] / df["DXY"].shift(1)
)

print("\nDXY_Return constructed successfully.")

print("\nMissing DXY_Return observations:")
print(df["DXY_Return"].isna().sum())

print("\nInfinite DXY_Return observations:")
print(
    np.isinf(df["DXY_Return"]).sum()
)


# ============================================================
# CONSTRUCT LAGGED DXY RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_DXY_RETURN")
print("=" * 70)

# Previous available DXY return:
#
# Lagged_DXY_Return_t = DXY_Return_(t-1)

df["Lagged_DXY_Return"] = (
    df["DXY_Return"].shift(1)
)

print("\nLagged_DXY_Return constructed successfully.")

print("\nMissing Lagged_DXY_Return observations:")
print(
    df["Lagged_DXY_Return"].isna().sum()
)


# ============================================================
# CHECK DATE GAPS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DXY DATE GAPS")
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

non_consecutive = (
    df["Calendar_Days_Since_Previous"] > 1
).sum()

print("\nNumber of observations following a calendar gap:")
print(non_consecutive)

print(
    "\nNOTE: Calendar gaps are expected because DXY does not "
    "follow the continuous 7-day cryptocurrency trading "
    "calendar. Returns are calculated between consecutive "
    "available DXY observations."
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
            "DXY",
            "DXY_Return",
            "Lagged_DXY_Return",
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
            "DXY",
            "DXY_Return",
            "Lagged_DXY_Return",
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
            "DXY",
            "DXY_Return",
            "Lagged_DXY_Return",
        ]
    ].describe()
)


# ============================================================
# EXTREME RETURN CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXTREME DXY RETURN CHECK")
print("=" * 70)

# Diagnostic check only.
# Large observations are NOT automatically removed.

extreme_returns = df[
    df["DXY_Return"].abs() > 0.05
][
    [
        "Date",
        "DXY",
        "DXY_Return",
    ]
]

print("\nNumber of absolute DXY log returns > 5%:")
print(len(extreme_returns))

if len(extreme_returns) > 0:
    print("\nLarge DXY returns identified:")
    print(
        extreme_returns.to_string(index=False)
    )

print(
    "\nNOTE: Extreme observations are flagged for validation only. "
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
print("SAVING PROCESSED DXY DATA")
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
            "DXY",
            "DXY_Return",
            "Lagged_DXY_Return",
        ]
    ].isna().sum()
)


print("\n" + "=" * 70)
print("DXY RETURN CONSTRUCTION COMPLETE")
print("=" * 70)