from pathlib import Path
import pandas as pd
import numpy as np


# ======================================================================
# FILE PATHS
# ======================================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "data_clean" / "btc_volume_clean.csv"
OUTPUT_FILE = PROJECT_ROOT / "data_processed" / "btc_volume_processed.csv"


print("=" * 70)
print("BTC TRADING VOLUME VARIABLE CONSTRUCTION")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCleaned BTC volume file was not found:\n{INPUT_FILE}"
    )


# ======================================================================
# IMPORT CLEANED BTC VOLUME DATA
# ======================================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED BTC VOLUME DATA")
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


# ======================================================================
# CHECK REQUIRED VARIABLES
# ======================================================================

required_columns = ["Date", "BTC_Volume"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nRequired columns are missing: {missing_columns}\n"
        f"Columns actually found: {df.columns.tolist()}"
    )


# ======================================================================
# STANDARDISE DATE
# ======================================================================

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates were found in the cleaned BTC volume dataset."
    )


# ======================================================================
# STANDARDISE BTC VOLUME
# ======================================================================

df["BTC_Volume"] = pd.to_numeric(
    df["BTC_Volume"],
    errors="coerce"
)

missing_volume = df["BTC_Volume"].isna().sum()

print("\nMissing BTC volume observations:")
print(missing_volume)

if missing_volume > 0:
    raise ValueError(
        "Missing/non-numeric BTC volume observations were found."
    )


# ======================================================================
# SORT DATA CHRONOLOGICALLY
# ======================================================================

df = df.sort_values("Date").reset_index(drop=True)


# ======================================================================
# BASIC VALIDATION
# ======================================================================

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

if duplicate_dates > 0:
    raise ValueError(
        "Duplicate dates were found in the cleaned BTC volume dataset."
    )


non_positive_volume = (df["BTC_Volume"] <= 0).sum()

print("\nZero or negative BTC volume observations:")
print(non_positive_volume)

if non_positive_volume > 0:
    raise ValueError(
        "BTC volume contains zero or negative observations. "
        "The natural logarithm cannot be calculated safely."
    )


# ======================================================================
# DATE RANGE
# ======================================================================

print("\n" + "=" * 70)
print("CHECKING DATE RANGE")
print("=" * 70)

print("\nNumber of observations:")
print(len(df))

print("\nFirst date:")
print(df["Date"].min())

print("\nLast date:")
print(df["Date"].max())


# ======================================================================
# CONSTRUCT LOG_BTC_VOLUME
# ======================================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LOG_BTC_VOLUME")
print("=" * 70)

# Natural logarithm of BTC trading volume
df["Log_BTC_Volume"] = np.log(df["BTC_Volume"])

print("\nLog_BTC_Volume constructed successfully.")

print("\nMissing Log_BTC_Volume observations:")
print(df["Log_BTC_Volume"].isna().sum())

print("\nInfinite Log_BTC_Volume observations:")
print(np.isinf(df["Log_BTC_Volume"]).sum())


# ======================================================================
# CONSTRUCT LAGGED_LOG_BTC_VOLUME
# ======================================================================

print("\n" + "=" * 70)
print("CONSTRUCTING LAGGED_LOG_BTC_VOLUME")
print("=" * 70)

# One-calendar-observation lag.
# Because BTC trades every day, the previous observation should normally
# correspond to the previous calendar day.
df["Lagged_Log_BTC_Volume"] = df["Log_BTC_Volume"].shift(1)

print("\nLagged_Log_BTC_Volume constructed successfully.")

print("\nMissing Lagged_Log_BTC_Volume observations:")
print(df["Lagged_Log_BTC_Volume"].isna().sum())


# ======================================================================
# CHECK THAT THE LAG REALLY IS THE PREVIOUS DAY
# ======================================================================

print("\n" + "=" * 70)
print("CHECKING LAG CONSTRUCTION")
print("=" * 70)

df["Previous_Date"] = df["Date"].shift(1)
df["Days_From_Previous_Observation"] = (
    df["Date"] - df["Previous_Date"]
).dt.days

gaps = df.loc[
    df["Days_From_Previous_Observation"].notna()
    & (df["Days_From_Previous_Observation"] != 1),
    ["Previous_Date", "Date", "Days_From_Previous_Observation"]
]

print("\nNon-consecutive calendar observations found:")
print(len(gaps))

if len(gaps) > 0:
    print("\nWARNING:")
    print(
        "The following observations are not exactly one calendar day "
        "after the preceding BTC volume observation:"
    )
    print(gaps.to_string(index=False))

    print(
        "\nThis matters because shift(1) means previous OBSERVATION, "
        "not necessarily previous CALENDAR DAY."
    )
else:
    print(
        "\nAll observations are consecutive calendar days. "
        "The lag therefore represents the previous day's BTC volume."
    )


# ======================================================================
# CHECK FIRST OBSERVATIONS
# ======================================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "BTC_Volume",
            "Log_BTC_Volume",
            "Lagged_Log_BTC_Volume"
        ]
    ]
    .head(10)
    .to_string(index=False)
)


# ======================================================================
# CHECK LAST OBSERVATIONS
# ======================================================================

print("\n" + "=" * 70)
print("CHECKING LAST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "BTC_Volume",
            "Log_BTC_Volume",
            "Lagged_Log_BTC_Volume"
        ]
    ]
    .tail(10)
    .to_string(index=False)
)


# ======================================================================
# SUMMARY STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    df[
        [
            "BTC_Volume",
            "Log_BTC_Volume",
            "Lagged_Log_BTC_Volume"
        ]
    ].describe()
)


# ======================================================================
# COMPARE SKEWNESS BEFORE AND AFTER LOG TRANSFORMATION
# ======================================================================

print("\n" + "=" * 70)
print("SKEWNESS CHECK")
print("=" * 70)

raw_skewness = df["BTC_Volume"].skew()
log_skewness = df["Log_BTC_Volume"].skew()

print("\nBTC_Volume skewness:")
print(raw_skewness)

print("\nLog_BTC_Volume skewness:")
print(log_skewness)

print(
    "\nThe log transformation is intended to reduce the strong "
    "positive skew typically present in trading-volume data."
)


# ======================================================================
# REMOVE TEMPORARY CHECKING VARIABLES
# ======================================================================

df = df.drop(
    columns=[
        "Previous_Date",
        "Days_From_Previous_Observation"
    ]
)


# ======================================================================
# SAVE PROCESSED DATA
# ======================================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED BTC VOLUME DATA")
print("=" * 70)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFile saved successfully:")
print(OUTPUT_FILE)


# ======================================================================
# FINAL CHECK
# ======================================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print("\nNumber of observations:")
print(len(df))

print("\nDate range:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nFinal variables:")
print(df.columns.tolist())

print("\nMissing values:")
print(
    df[
        [
            "BTC_Volume",
            "Log_BTC_Volume",
            "Lagged_Log_BTC_Volume"
        ]
    ].isna().sum()
)

print("\n" + "=" * 70)
print("BTC TRADING VOLUME VARIABLE CONSTRUCTION COMPLETE")
print("=" * 70)