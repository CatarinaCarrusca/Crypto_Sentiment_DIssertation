from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

input_file = project_folder / "data_clean" / "gold_clean.csv"
output_folder = project_folder / "data_processed"
output_file = output_folder / "gold_returns.csv"

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("GOLD RETURN VARIABLE CONSTRUCTION")
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
# IMPORT CLEANED GOLD PRICE DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED GOLD PRICE DATA")
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

required_columns = ["Date", "Gold_Price"]

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
        "Invalid dates were found in the cleaned Gold dataset."
    )


# ============================================================
# STANDARDISE GOLD PRICE
# ============================================================

df["Gold_Price"] = pd.to_numeric(
    df["Gold_Price"],
    errors="coerce"
)

missing_prices = df["Gold_Price"].isna().sum()

print("\nMissing Gold prices:")
print(missing_prices)

if missing_prices > 0:
    raise ValueError(
        "Missing/non-numeric Gold prices were found."
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
        "Duplicate dates were found in the Gold dataset."
    )


# ============================================================
# NON-POSITIVE PRICE CHECK
# ============================================================

non_positive = (df["Gold_Price"] <= 0).sum()

print("\nZero or negative Gold prices:")
print(non_positive)

if non_positive > 0:
    raise ValueError(
        "Gold prices must be positive before log returns "
        "can be calculated."
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
# CONSTRUCT GOLD_RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING GOLD_RETURN")
print("=" * 70)

# Daily log return:
#
# Gold_Return_t = ln(Gold_Price_t / Gold_Price_(t-1))

df["Gold_Return"] = np.log(
    df["Gold_Price"] / df["Gold_Price"].shift(1)
)

print("\nGold_Return constructed successfully.")

print("\nMissing Gold_Return observations:")
print(df["Gold_Return"].isna().sum())

print("\nInfinite Gold_Return observations:")
print(
    np.isinf(df["Gold_Return"]).sum()
)


# ============================================================
# CONSTRUCT LAGGED GOLD RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_GOLD_RETURN")
print("=" * 70)

# Previous available Gold return:
#
# Lagged_Gold_Return_t = Gold_Return_(t-1)

df["Lagged_Gold_Return"] = (
    df["Gold_Return"].shift(1)
)

print("\nLagged_Gold_Return constructed successfully.")

print("\nMissing Lagged_Gold_Return observations:")
print(
    df["Lagged_Gold_Return"].isna().sum()
)


# ============================================================
# CHECK DATE GAPS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING GOLD DATE GAPS")
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
    "\nNOTE: Calendar gaps are expected because Gold does not "
    "follow the same continuous 7-day trading calendar as "
    "Bitcoin and Ethereum. Returns are calculated between "
    "consecutive available Gold observations."
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
            "Gold_Price",
            "Gold_Return",
            "Lagged_Gold_Return",
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
            "Gold_Price",
            "Gold_Return",
            "Lagged_Gold_Return",
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
            "Gold_Price",
            "Gold_Return",
            "Lagged_Gold_Return",
        ]
    ].describe()
)


# ============================================================
# EXTREME RETURN CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXTREME GOLD RETURN CHECK")
print("=" * 70)

# Diagnostic only.
# Large returns should not automatically be removed.

extreme_returns = df[
    df["Gold_Return"].abs() > 0.10
][
    [
        "Date",
        "Gold_Price",
        "Gold_Return",
    ]
]

print("\nNumber of absolute Gold log returns > 10%:")
print(len(extreme_returns))

if len(extreme_returns) > 0:
    print("\nLarge Gold returns identified:")
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
print("SAVING PROCESSED GOLD DATA")
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
            "Gold_Price",
            "Gold_Return",
            "Lagged_Gold_Return",
        ]
    ].isna().sum()
)


print("\n" + "=" * 70)
print("GOLD RETURN CONSTRUCTION COMPLETE")
print("=" * 70)