from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

input_file = project_folder / "data_clean" / "sp500_clean.csv"
output_folder = project_folder / "data_processed"
output_file = output_folder / "sp500_returns.csv"

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("S&P 500 RETURN VARIABLE CONSTRUCTION")
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
# IMPORT CLEANED S&P 500 DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED S&P 500 DATA")
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

required_columns = ["Date", "SP500"]

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
        "Invalid dates were found in the cleaned S&P 500 dataset."
    )


# ============================================================
# STANDARDISE S&P 500 LEVEL
# ============================================================

df["SP500"] = pd.to_numeric(
    df["SP500"],
    errors="coerce"
)

missing_values = df["SP500"].isna().sum()

print("\nMissing S&P 500 observations:")
print(missing_values)

if missing_values > 0:
    raise ValueError(
        "Missing/non-numeric S&P 500 observations were found."
    )


# ============================================================
# SORT BY DATE
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
        "Duplicate dates were found in the S&P 500 dataset."
    )


# ============================================================
# CHECK NON-POSITIVE VALUES
# ============================================================

non_positive = (df["SP500"] <= 0).sum()

print("\nZero or negative S&P 500 observations:")
print(non_positive)

if non_positive > 0:
    raise ValueError(
        "S&P 500 contains zero or negative values. "
        "Log returns cannot be calculated safely."
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
# CONSTRUCT SP500_RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING SP500_RETURN")
print("=" * 70)

# Daily S&P 500 log return:
#
# SP500_Return_t = ln(SP500_t / SP500_(t-1))
#
# IMPORTANT:
# Because the S&P 500 trades only on market days,
# the previous observation is the previous S&P 500 trading day.

df["SP500_Return"] = np.log(
    df["SP500"] / df["SP500"].shift(1)
)

print("\nSP500_Return constructed successfully.")

print("\nMissing SP500_Return observations:")
print(df["SP500_Return"].isna().sum())

print("\nInfinite SP500_Return observations:")
print(
    np.isinf(df["SP500_Return"]).sum()
)


# ============================================================
# CONSTRUCT LAGGED SP500 RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_SP500_RETURN")
print("=" * 70)

# Previous S&P 500 trading-day return.
#
# This is the variable intended for the main predictive model.

df["Lagged_SP500_Return"] = (
    df["SP500_Return"].shift(1)
)

print("\nLagged_SP500_Return constructed successfully.")

print("\nMissing Lagged_SP500_Return observations:")
print(
    df["Lagged_SP500_Return"].isna().sum()
)


# ============================================================
# CHECK GAPS BETWEEN MARKET OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING S&P 500 TRADING-DAY GAPS")
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
    "\nNOTE: Gaps greater than one calendar day are expected "
    "because the S&P 500 does not trade on weekends and "
    "market holidays."
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
            "SP500",
            "SP500_Return",
            "Lagged_SP500_Return",
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
            "SP500",
            "SP500_Return",
            "Lagged_SP500_Return",
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
            "SP500",
            "SP500_Return",
            "Lagged_SP500_Return",
        ]
    ].describe()
)


# ============================================================
# EXTREME RETURN CHECK
# ============================================================

print("\n" + "=" * 70)
print("EXTREME S&P 500 RETURN CHECK")
print("=" * 70)

# Validation only.
# Do NOT automatically delete large market movements.

extreme_returns = df[
    df["SP500_Return"].abs() > 0.05
][
    [
        "Date",
        "SP500",
        "SP500_Return",
    ]
]

print("\nNumber of absolute S&P 500 log returns > 5%:")
print(len(extreme_returns))

if len(extreme_returns) > 0:
    print("\nExtreme returns identified:")
    print(
        extreme_returns.to_string(index=False)
    )

print(
    "\nNOTE: Extreme observations are flagged for validation only. "
    "They are NOT automatically removed, winsorised, "
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
print("SAVING PROCESSED S&P 500 DATA")
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
            "SP500",
            "SP500_Return",
            "Lagged_SP500_Return",
        ]
    ].isna().sum()
)


print("\n" + "=" * 70)
print("S&P 500 RETURN VARIABLE CONSTRUCTION COMPLETE")
print("=" * 70)