from pathlib import Path
import pandas as pd

# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

EXCEL_FILE = PROJECT_DIR / "data_raw" / "Dissertation Data.xlsx"
OUTPUT_FILE = PROJECT_DIR / "data_clean" / "sp500_clean.csv"

SHEET_NAME = " S&P500(FRED)"

START_DATE = pd.Timestamp("2021-01-01")
END_DATE = pd.Timestamp("2025-12-31")


print("=" * 70)
print("S&P 500 DATA CLEANING")
print("=" * 70)

print("\nLooking for Excel file:")
print(EXCEL_FILE)

print("\nDoes file exist?")
print(EXCEL_FILE.exists())

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"\nExcel file could not be found:\n{EXCEL_FILE}"
    )

print("\nExcel file found successfully.")


# ============================================================
# 1. IMPORT S&P 500 SHEET
# ============================================================

print("\nImporting S&P 500 sheet...")

df = pd.read_excel(
    EXCEL_FILE,
    sheet_name=SHEET_NAME
)

print("S&P 500 sheet imported successfully.")

print("\nRaw shape:")
print(df.shape)

print("\nRaw columns:")
print(df.columns.tolist())

print("\nFirst 10 raw observations:")
print(df.head(10).to_string(index=False))


# ============================================================
# 2. STANDARDISE COLUMN NAMES
# ============================================================

# Remove accidental spaces from column names
df.columns = df.columns.astype(str).str.strip()

print("\n" + "=" * 70)
print("STANDARDISING COLUMNS")
print("=" * 70)

print("\nColumns after removing spaces:")
print(df.columns.tolist())

# Rename FRED variables
df = df.rename(
    columns={
        "observation_date": "Date",
        "SP500": "SP500"
    }
)

required_columns = ["Date", "SP500"]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"\nRequired columns not found: {missing_columns}\n"
        f"Available columns: {df.columns.tolist()}"
    )

# Keep only required variables
df = df[["Date", "SP500"]].copy()


# ============================================================
# 3. CLEAN DATE VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATE VARIABLE")
print("=" * 70)

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce",
    dayfirst=True
)

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates found:")
print(invalid_dates)

if invalid_dates > 0:
    print("\nRows with invalid dates:")
    print(df[df["Date"].isna()].to_string(index=False))

# Remove invalid dates
df = df.dropna(subset=["Date"]).copy()


# ============================================================
# 4. RESTRICT SAMPLE TO 2021-2025
# ============================================================

before_date_filter = len(df)

df = df[
    (df["Date"] >= START_DATE) &
    (df["Date"] <= END_DATE)
].copy()

outside_period = before_date_filter - len(df)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# 5. CLEAN S&P 500 VALUES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING S&P 500 VALUES")
print("=" * 70)

df["SP500"] = pd.to_numeric(
    df["SP500"],
    errors="coerce"
)

missing_sp500 = df["SP500"].isna().sum()

print("\nNumber of missing S&P 500 observations:")
print(missing_sp500)

if missing_sp500 > 0:

    print("\nDates with missing S&P 500 values:")

    print(
        df.loc[
            df["SP500"].isna(),
            ["Date", "SP500"]
        ].to_string(index=False)
    )


# ============================================================
# 6. REMOVE MISSING S&P 500 OBSERVATIONS
# ============================================================

before_missing_removal = len(df)

df = df.dropna(subset=["SP500"]).copy()

removed_missing = before_missing_removal - len(df)

print("\nMissing S&P 500 observations removed:")
print(removed_missing)

print("\nMissing S&P 500 observations remaining:")
print(df["SP500"].isna().sum())

print("\nObservations remaining after removing missing values:")
print(len(df))


# ============================================================
# 7. CHECK DUPLICATE DATES
# ============================================================

duplicate_dates = df.duplicated(
    subset=["Date"],
    keep=False
)

number_duplicates = duplicate_dates.sum()

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

print("\nDuplicate dates found:")
print(number_duplicates)

if number_duplicates > 0:

    print("\nDuplicate observations:")

    print(
        df.loc[
            duplicate_dates
        ].sort_values("Date").to_string(index=False)
    )

    # Keep first observation if exact/duplicate dates exist
    df = df.drop_duplicates(
        subset=["Date"],
        keep="first"
    ).copy()


# ============================================================
# 8. CHECK ZERO OR NEGATIVE VALUES
# ============================================================

non_positive = df["SP500"] <= 0

print("\n" + "=" * 70)
print("NON-POSITIVE VALUE CHECK")
print("=" * 70)

print("\nZero or negative S&P 500 values found:")
print(non_positive.sum())

if non_positive.sum() > 0:

    print("\nNon-positive observations:")

    print(
        df.loc[
            non_positive,
            ["Date", "SP500"]
        ].to_string(index=False)
    )


# ============================================================
# 9. SORT DATA
# ============================================================

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# 10. VALIDATION-ONLY DAILY CHANGES
# ============================================================

# These are temporary checks.
# They are NOT being saved as final regression variables.

df["Validation_Log_Return"] = (
    df["SP500"] / df["SP500"].shift(1)
).apply(
    lambda x: pd.NA if pd.isna(x) or x <= 0 else x
)

df["Validation_Log_Return"] = pd.to_numeric(
    df["Validation_Log_Return"],
    errors="coerce"
)

import numpy as np

df["Validation_Log_Return"] = np.log(
    df["SP500"] / df["SP500"].shift(1)
)

df["Validation_Return_Pct"] = (
    (df["SP500"] / df["SP500"].shift(1)) - 1
) * 100


# ============================================================
# 11. SHOW LARGEST DAILY MOVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("10 LARGEST ABSOLUTE S&P 500 DAILY MOVEMENTS - VALIDATION ONLY")
print("=" * 70)

largest_moves = (
    df.dropna(subset=["Validation_Log_Return"])
    .assign(
        Abs_Return=lambda x:
        x["Validation_Log_Return"].abs()
    )
    .sort_values(
        "Abs_Return",
        ascending=False
    )
    .head(10)
)

print(
    largest_moves[
        [
            "Date",
            "SP500",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. FLAG VERY LARGE MOVEMENTS
# ============================================================

large_moves = df[
    df["Validation_Return_Pct"].abs() > 5
].copy()

print("\n" + "=" * 70)
print("DAILY S&P 500 MOVEMENTS GREATER THAN 5%")
print("=" * 70)

print("\nNumber of absolute daily movements >5%:")
print(len(large_moves))

if len(large_moves) > 0:

    print(
        large_moves[
            [
                "Date",
                "SP500",
                "Validation_Log_Return",
                "Validation_Return_Pct"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 13. FINAL CLEANING CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(df[["Date", "SP500"]].isna().sum())

print("\nDuplicate dates:")
print(df.duplicated(subset=["Date"]).sum())

print("\nZero or negative S&P 500 values:")
print((df["SP500"] <= 0).sum())


# ============================================================
# 14. CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean observations:")
print(len(df))

print("\nFirst date:")
print(df["Date"].min().date())

print("\nLast date:")
print(df["Date"].max().date())

print("\nMissing S&P 500 values:")
print(df["SP500"].isna().sum())

print("\nDuplicate dates:")
print(df.duplicated(subset=["Date"]).sum())

print("\nMinimum S&P 500:")
print(df["SP500"].min())

print("\nMaximum S&P 500:")
print(df["SP500"].max())


# ============================================================
# 15. DISPLAY FIRST/LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN S&P 500 OBSERVATIONS")
print("=" * 70)

print(
    df[
        ["Date", "SP500"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("LAST 10 CLEAN S&P 500 OBSERVATIONS")
print("=" * 70)

print(
    df[
        ["Date", "SP500"]
    ].tail(10).to_string(index=False)
)


# ============================================================
# 16. SAVE CLEAN DATA
# ============================================================

# Save ONLY the clean price data.
# Validation variables are deliberately excluded.

clean_df = df[
    ["Date", "SP500"]
].copy()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

clean_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 17. SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean S&P 500 data saved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(clean_df.columns.tolist())

print("\nFinal shape:")
print(clean_df.shape)

print("\nS&P 500 cleaning completed successfully.")

print(
    "\nIMPORTANT: Missing S&P 500 observations were removed, "
    "not interpolated or replaced with zero."
)

print(
    "S&P500_Return has NOT yet been created as the final "
    "regression variable."
)