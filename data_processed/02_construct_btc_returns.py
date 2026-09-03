from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
# BTC RETURN VARIABLE CONSTRUCTION
# ============================================================

print("=" * 70)
print("BTC RETURN VARIABLE CONSTRUCTION")
print("=" * 70)

# ------------------------------------------------------------
# 1. DEFINE PROJECT PATHS
# ------------------------------------------------------------

project_root = Path(__file__).resolve().parents[1]

input_file = project_root / "data_clean" / "btc_price_clean.csv"
output_file = project_root / "data_processed" / "btc_returns.csv"

print("\nInput file:")
print(input_file)

print("\nDoes input file exist?")
print(input_file.exists())

if not input_file.exists():
    raise FileNotFoundError(
        f"\nCould not find cleaned BTC price file:\n{input_file}"
    )

# ------------------------------------------------------------
# 2. IMPORT CLEANED BTC PRICE DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("IMPORTING CLEANED BTC PRICE DATA")
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

# ------------------------------------------------------------
# 3. CHECK REQUIRED COLUMNS
# ------------------------------------------------------------

required_columns = ["Date", "BTC_Price"]

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nRequired columns missing from input file: {missing_columns}"
    )

# ------------------------------------------------------------
# 4. STANDARDISE DATE
# ------------------------------------------------------------

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates found in cleaned BTC price dataset."
    )

# ------------------------------------------------------------
# 5. STANDARDISE BTC PRICE
# ------------------------------------------------------------

df["BTC_Price"] = pd.to_numeric(
    df["BTC_Price"],
    errors="coerce"
)

missing_prices = df["BTC_Price"].isna().sum()

print("\nMissing BTC prices:")
print(missing_prices)

if missing_prices > 0:
    raise ValueError(
        "Missing BTC prices found in cleaned BTC price dataset."
    )

non_positive_prices = (df["BTC_Price"] <= 0).sum()

print("\nZero or negative BTC prices:")
print(non_positive_prices)

if non_positive_prices > 0:
    raise ValueError(
        "BTC price contains zero or negative observations."
    )

# ------------------------------------------------------------
# 6. SORT DATA CHRONOLOGICALLY
# ------------------------------------------------------------

df = (
    df.sort_values("Date")
    .reset_index(drop=True)
)

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

if duplicate_dates > 0:
    raise ValueError(
        "Duplicate dates found in cleaned BTC price dataset."
    )

# ------------------------------------------------------------
# 7. CHECK STUDY PERIOD
# ------------------------------------------------------------

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2025-12-31")

outside_period = (
    (df["Date"] < start_date) |
    (df["Date"] > end_date)
).sum()

print("\nObservations outside study period:")
print(outside_period)

if outside_period > 0:
    raise ValueError(
        "Observations outside 2021-01-01 to 2025-12-31 were found."
    )

# ------------------------------------------------------------
# 8. CHECK CALENDAR COVERAGE
# ------------------------------------------------------------

expected_dates = pd.date_range(
    start=start_date,
    end=end_date,
    freq="D"
)

actual_dates = pd.DatetimeIndex(df["Date"])

missing_dates = expected_dates.difference(actual_dates)

print("\nExpected calendar days:")
print(len(expected_dates))

print("\nActual BTC price observations:")
print(len(df))

print("\nMissing calendar dates:")
print(len(missing_dates))

if len(missing_dates) > 0:
    print("\nMissing dates:")
    for date in missing_dates:
        print(date.strftime("%Y-%m-%d"))

# ------------------------------------------------------------
# 9. CHECK THE KNOWN REFINITIV OBSERVATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING REFINITIV OBSERVATION: 2023-07-30")
print("=" * 70)

check_start = pd.Timestamp("2023-07-27")
check_end = pd.Timestamp("2023-08-02")

btc_check = df.loc[
    (df["Date"] >= check_start) &
    (df["Date"] <= check_end),
    ["Date", "BTC_Price"]
]

print("\nBTC prices surrounding 30 July 2023:")
print(btc_check.to_string(index=False))

target_date = pd.Timestamp("2023-07-30")

target_row = df.loc[
    df["Date"] == target_date,
    "BTC_Price"
]

if target_row.empty:
    print("\nWARNING: 2023-07-30 was not found.")
else:
    target_price = target_row.iloc[0]

    print("\nBTC price stored for 2023-07-30:")
    print(target_price)

    if np.isclose(target_price, 20269.50):
        print(
            "\nNOTE: The cleaned dataset contains the previously "
            "identified Refinitiv value of 20,269.50."
        )
        print(
            "This value will be retained as supplied by Refinitiv."
        )
        print(
            "Returns around this date should be treated as a "
            "documented source-data sensitivity issue."
        )

# ------------------------------------------------------------
# 10. CONSTRUCT BTC LOG RETURN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONSTRUCTING BTC_RETURN")
print("=" * 70)

df["BTC_Return"] = np.log(
    df["BTC_Price"] / df["BTC_Price"].shift(1)
)

print("\nBTC_Return constructed successfully.")

print("\nMissing BTC_Return observations:")
print(df["BTC_Return"].isna().sum())

# ------------------------------------------------------------
# 11. CONSTRUCT ONE-DAY LAGGED BTC RETURN
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CONSTRUCTING BTC_LAGGED_RETURN")
print("=" * 70)

df["BTC_Lagged_Return"] = df["BTC_Return"].shift(1)

print("\nBTC_Lagged_Return constructed successfully.")

print("\nMissing BTC_Lagged_Return observations:")
print(df["BTC_Lagged_Return"].isna().sum())

# ------------------------------------------------------------
# 12. CHECK FIRST OBSERVATIONS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)

print(
    df[
        [
            "Date",
            "BTC_Price",
            "BTC_Return",
            "BTC_Lagged_Return"
        ]
    ].head(10).to_string(index=False)
)

# ------------------------------------------------------------
# 13. SPECIFICALLY INSPECT RETURNS AROUND 30 JULY 2023
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("CHECKING RETURNS AROUND 30 JULY 2023")
print("=" * 70)

anomaly_check = df.loc[
    (df["Date"] >= pd.Timestamp("2023-07-27")) &
    (df["Date"] <= pd.Timestamp("2023-08-03")),
    [
        "Date",
        "BTC_Price",
        "BTC_Return",
        "BTC_Lagged_Return"
    ]
]

print(anomaly_check.to_string(index=False))

# ------------------------------------------------------------
# 14. SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    df[
        [
            "BTC_Price",
            "BTC_Return",
            "BTC_Lagged_Return"
        ]
    ].describe()
)

# ------------------------------------------------------------
# 15. CHECK INFINITE VALUES
# ------------------------------------------------------------

infinite_returns = np.isinf(df["BTC_Return"]).sum()
infinite_lagged_returns = np.isinf(
    df["BTC_Lagged_Return"]
).sum()

print("\nInfinite BTC_Return values:")
print(infinite_returns)

print("\nInfinite BTC_Lagged_Return values:")
print(infinite_lagged_returns)

if infinite_returns > 0 or infinite_lagged_returns > 0:
    raise ValueError(
        "Infinite return values detected."
    )

# ------------------------------------------------------------
# 16. IDENTIFY EXTREME RETURNS FOR VALIDATION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("EXTREME BTC RETURN CHECK")
print("=" * 70)

# Flag absolute daily log returns greater than 20%.
# These are NOT automatically deleted.
extreme_returns = df.loc[
    df["BTC_Return"].abs() > 0.20,
    [
        "Date",
        "BTC_Price",
        "BTC_Return"
    ]
]

print("\nNumber of absolute BTC log returns > 20%:")
print(len(extreme_returns))

if len(extreme_returns) > 0:
    print("\nExtreme returns identified:")
    print(extreme_returns.to_string(index=False))

print(
    "\nNOTE: Extreme observations are flagged for validation only."
)
print(
    "They are NOT automatically deleted, winsorised, interpolated, "
    "or replaced."
)

# ------------------------------------------------------------
# 17. SAVE PROCESSED DATA
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SAVING PROCESSED BTC DATA")
print("=" * 70)

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

final_df = df[
    [
        "Date",
        "BTC_Price",
        "BTC_Return",
        "BTC_Lagged_Return"
    ]
].copy()

final_df.to_csv(
    output_file,
    index=False
)

print("\nFile saved successfully:")
print(output_file)

# ------------------------------------------------------------
# 18. FINAL CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print("\nNumber of observations:")
print(len(final_df))

print("\nDate range:")
print(
    final_df["Date"].min(),
    "to",
    final_df["Date"].max()
)

print("\nFinal variables:")
print(final_df.columns.tolist())

print("\nMissing values:")
print(final_df.isna().sum())

print("\n" + "=" * 70)
print("BTC RETURN CONSTRUCTION COMPLETE")
print("=" * 70)