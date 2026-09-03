from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

input_file = project_folder / "data_clean" / "eth_volume_clean.csv"
output_folder = project_folder / "data_processed"
output_file = output_folder / "eth_volume_processed.csv"

output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ETH TRADING VOLUME VARIABLE CONSTRUCTION")
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
# IMPORT CLEANED ETH VOLUME DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED ETH VOLUME DATA")
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

required_columns = ["Date", "ETH_Volume"]

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
        "Invalid dates were found in the cleaned ETH volume dataset."
    )


# ============================================================
# STANDARDISE ETH VOLUME
# ============================================================

df["ETH_Volume"] = pd.to_numeric(
    df["ETH_Volume"],
    errors="coerce"
)

missing_volume = df["ETH_Volume"].isna().sum()

print("\nMissing ETH volume observations:")
print(missing_volume)

if missing_volume > 0:
    raise ValueError(
        "Missing/non-numeric ETH volume values were found."
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
        "Duplicate dates were found in the ETH volume dataset."
    )


# ============================================================
# CHECK ZERO OR NEGATIVE VOLUME
# ============================================================

non_positive_volume = (df["ETH_Volume"] <= 0).sum()

print("\nZero or negative ETH volume observations:")
print(non_positive_volume)

if non_positive_volume > 0:
    raise ValueError(
        "ETH_Volume contains zero or negative values. "
        "Natural logarithms cannot be calculated safely."
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
# CONSTRUCT LOG_ETH_VOLUME
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LOG_ETH_VOLUME")
print("=" * 70)

# Natural logarithm:
# Log_ETH_Volume_t = ln(ETH_Volume_t)

df["Log_ETH_Volume"] = np.log(df["ETH_Volume"])

print("\nLog_ETH_Volume constructed successfully.")

print("\nMissing Log_ETH_Volume observations:")
print(df["Log_ETH_Volume"].isna().sum())

print("\nInfinite Log_ETH_Volume observations:")
print(
    np.isinf(df["Log_ETH_Volume"]).sum()
)


# ============================================================
# CONSTRUCT LAGGED LOG ETH VOLUME
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_LOG_ETH_VOLUME")
print("=" * 70)

# Yesterday's logged ETH volume:
# Lagged_Log_ETH_Volume_t = Log_ETH_Volume_(t-1)

df["Lagged_Log_ETH_Volume"] = (
    df["Log_ETH_Volume"].shift(1)
)

print("\nLagged_Log_ETH_Volume constructed successfully.")

print("\nMissing Lagged_Log_ETH_Volume observations:")
print(
    df["Lagged_Log_ETH_Volume"].isna().sum()
)


# ============================================================
# CHECK WHETHER DATES ARE CONSECUTIVE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LAG CONSTRUCTION")
print("=" * 70)

df["Date_Difference"] = df["Date"].diff().dt.days

non_consecutive = (
    df["Date_Difference"].dropna() != 1
).sum()

print("\nNon-consecutive calendar observations found:")
print(non_consecutive)

if non_consecutive == 0:
    print(
        "\nAll observations are consecutive calendar days. "
        "The lag therefore represents the previous day's ETH volume."
    )
else:
    print(
        "\nWARNING: Some calendar dates are missing. "
        "For those observations, shift(1) represents the previous "
        "available observation rather than necessarily the previous "
        "calendar day."
    )

df = df.drop(columns=["Date_Difference"])


# ============================================================
# FIRST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "ETH_Volume",
            "Log_ETH_Volume",
            "Lagged_Log_ETH_Volume",
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ============================================================
# LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING LAST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "ETH_Volume",
            "Log_ETH_Volume",
            "Lagged_Log_ETH_Volume",
        ]
    ]
    .tail(10)
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
            "ETH_Volume",
            "Log_ETH_Volume",
            "Lagged_Log_ETH_Volume",
        ]
    ].describe()
)


# ============================================================
# SKEWNESS CHECK
# ============================================================

print("\n" + "=" * 70)
print("SKEWNESS CHECK")
print("=" * 70)

raw_skewness = df["ETH_Volume"].skew()
log_skewness = df["Log_ETH_Volume"].skew()

print("\nETH_Volume skewness:")
print(raw_skewness)

print("\nLog_ETH_Volume skewness:")
print(log_skewness)

print(
    "\nThe log transformation is intended to reduce the strong "
    "positive skew typically present in trading-volume data."
)


# ============================================================
# SAVE PROCESSED DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED ETH VOLUME DATA")
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
            "ETH_Volume",
            "Log_ETH_Volume",
            "Lagged_Log_ETH_Volume",
        ]
    ].isna().sum()
)


print("\n" + "=" * 70)
print("ETH TRADING VOLUME VARIABLE CONSTRUCTION COMPLETE")
print("=" * 70)