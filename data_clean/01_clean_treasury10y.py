import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# FILE PATHS
# ============================================================

project_path = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

excel_file = project_path / "data_raw" / "Dissertation Data.xlsx"
output_file = project_path / "data_clean" / "treasury10y_clean.csv"

# IMPORTANT:
# Change this only if the sheet in your Excel workbook has a different name.
sheet_name = "10 YEAR TREASURY YIELD(FRED)"


# ============================================================
# TITLE
# ============================================================

print("=" * 70)
print("10-YEAR TREASURY YIELD DATA CLEANING")
print("=" * 70)

print("\nLooking for Excel file:")
print(excel_file)

print("\nDoes file exist?")
print(excel_file.exists())

if not excel_file.exists():
    raise FileNotFoundError(
        f"\nExcel file could not be found:\n{excel_file}"
    )

print("\nExcel file found successfully.")


# ============================================================
# IMPORT DATA
# ============================================================

print("\nImporting 10-Year Treasury Yield sheet...")

try:
    df = pd.read_excel(
        excel_file,
        sheet_name=sheet_name
    )

except ValueError:
    print("\nERROR: The sheet name was not found.")
    print(f"Sheet requested: {sheet_name}")

    excel = pd.ExcelFile(excel_file)

    print("\nAvailable sheets:")
    for sheet in excel.sheet_names:
        print("-", sheet)

    raise

print("10-Year Treasury Yield sheet imported successfully.")

print("\nRaw shape:")
print(df.shape)

print("\nRaw columns:")
print(df.columns.tolist())

print("\nFirst 10 raw observations:")
print(df.head(10).to_string(index=False))


# ============================================================
# STANDARDISE COLUMN NAMES
# ============================================================

print("\n" + "=" * 70)
print("STANDARDISING COLUMNS")
print("=" * 70)

df.columns = (
    df.columns
    .astype(str)
    .str.strip()
)

print("\nColumns after removing spaces:")
print(df.columns.tolist())


# ============================================================
# IDENTIFY DATE AND YIELD COLUMNS
# ============================================================

date_candidates = [
    "observation_date",
    "Observation Date",
    "Date",
    "date"
]

yield_candidates = [
    "DGS10",
    "10Y",
    "Treasury10Y",
    "Treasury_10Y"
]

date_column = None
yield_column = None

for col in date_candidates:
    if col in df.columns:
        date_column = col
        break

for col in yield_candidates:
    if col in df.columns:
        yield_column = col
        break

if date_column is None:
    raise ValueError(
        "\nCould not identify the date column."
    )

if yield_column is None:
    raise ValueError(
        "\nCould not identify the 10-Year Treasury Yield column."
    )

print("\nDate column identified:")
print(date_column)

print("\n10-Year Treasury Yield column identified:")
print(yield_column)


# ============================================================
# KEEP REQUIRED VARIABLES
# ============================================================

df = df[[date_column, yield_column]].copy()

df = df.rename(
    columns={
        date_column: "Date",
        yield_column: "Treasury_10Y"
    }
)


# ============================================================
# CLEAN DATE VARIABLE
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

df = df.dropna(subset=["Date"]).copy()


# ============================================================
# RESTRICT SAMPLE TO 2021-2025
# ============================================================

before_period_filter = len(df)

df = df[
    (df["Date"] >= "2021-01-01") &
    (df["Date"] <= "2025-12-31")
].copy()

outside_period = before_period_filter - len(df)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# CLEAN TREASURY YIELD
# ============================================================

print("\n" + "=" * 70)
print("CLEANING 10-YEAR TREASURY YIELD")
print("=" * 70)

df["Treasury_10Y"] = pd.to_numeric(
    df["Treasury_10Y"],
    errors="coerce"
)

missing_count = df["Treasury_10Y"].isna().sum()

print("\nNumber of missing Treasury Yield observations:")
print(missing_count)

if missing_count > 0:

    print("\nDates with missing Treasury Yield values:")

    print(
        df.loc[
            df["Treasury_10Y"].isna(),
            ["Date", "Treasury_10Y"]
        ].to_string(index=False)
    )


# ============================================================
# REMOVE MISSING VALUES
# ============================================================

before_missing_removal = len(df)

df = df.dropna(
    subset=["Treasury_10Y"]
).copy()

removed_missing = before_missing_removal - len(df)

print("\nMissing Treasury Yield observations removed:")
print(removed_missing)

print("\nMissing Treasury Yield observations remaining:")
print(df["Treasury_10Y"].isna().sum())

print("\nObservations remaining after removing missing values:")
print(len(df))


# ============================================================
# SORT BY DATE
# ============================================================

df = df.sort_values("Date").reset_index(drop=True)


# ============================================================
# DUPLICATE DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates found:")
print(duplicate_dates)

if duplicate_dates > 0:

    print("\nDuplicate observations:")

    print(
        df.loc[
            df["Date"].duplicated(keep=False)
        ].to_string(index=False)
    )

    # Keep first observation for each date
    df = df.drop_duplicates(
        subset="Date",
        keep="first"
    ).copy()

    print("\nDuplicate dates removed.")


# ============================================================
# NON-POSITIVE VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("NON-POSITIVE VALUE CHECK")
print("=" * 70)

non_positive = (df["Treasury_10Y"] <= 0).sum()

print("\nZero or negative Treasury Yield values found:")
print(non_positive)

if non_positive > 0:

    print(
        df.loc[
            df["Treasury_10Y"] <= 0,
            ["Date", "Treasury_10Y"]
        ].to_string(index=False)
    )


# ============================================================
# VALIDATION: DAILY CHANGE IN YIELD
# ============================================================
#
# IMPORTANT:
# Treasury yields are already expressed as percentages.
#
# Therefore, for validation it is more useful to inspect
# changes in the yield itself rather than calculate a log return.
#
# Example:
# 4.20 -> 4.30 = +0.10 percentage points = +10 basis points.
#

df["Validation_Yield_Change"] = (
    df["Treasury_10Y"].diff()
)

df["Validation_Change_BPS"] = (
    df["Validation_Yield_Change"] * 100
)


# ============================================================
# LARGEST DAILY YIELD CHANGES
# ============================================================

print("\n" + "=" * 70)
print("10 LARGEST ABSOLUTE DAILY YIELD CHANGES - VALIDATION ONLY")
print("=" * 70)

largest_changes = (
    df.dropna(subset=["Validation_Yield_Change"])
      .assign(
          Absolute_Change=lambda x:
          x["Validation_Yield_Change"].abs()
      )
      .sort_values(
          "Absolute_Change",
          ascending=False
      )
      .head(10)
)

print(
    largest_changes[
        [
            "Date",
            "Treasury_10Y",
            "Validation_Yield_Change",
            "Validation_Change_BPS"
        ]
    ].to_string(index=False)
)


# ============================================================
# FLAG LARGE DAILY MOVEMENTS
# ============================================================
#
# Flag changes of 25 basis points or more for inspection.
# These are NOT automatically deleted.
#

print("\n" + "=" * 70)
print("DAILY TREASURY YIELD CHANGES OF 25 BASIS POINTS OR MORE")
print("=" * 70)

large_changes = df[
    df["Validation_Change_BPS"].abs() >= 25
].copy()

print("\nNumber of absolute daily changes >= 25 basis points:")
print(len(large_changes))

if len(large_changes) > 0:

    print(
        large_changes[
            [
                "Date",
                "Treasury_10Y",
                "Validation_Yield_Change",
                "Validation_Change_BPS"
            ]
        ].to_string(index=False)
    )

else:

    print("\nNone found.")


# ============================================================
# CONSECUTIVE IDENTICAL YIELD CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONSECUTIVE IDENTICAL YIELD CHECK")
print("=" * 70)

df["Previous_Yield"] = df["Treasury_10Y"].shift(1)

identical = df[
    df["Treasury_10Y"] == df["Previous_Yield"]
].copy()

print("\nNumber of consecutive identical Treasury Yield values:")
print(len(identical))

if len(identical) > 0:

    print(
        identical[
            [
                "Date",
                "Previous_Yield",
                "Treasury_10Y"
            ]
        ].to_string(index=False)
    )


# ============================================================
# FINAL CLEANING CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(
    df[
        ["Date", "Treasury_10Y"]
    ].isna().sum()
)

print("\nDuplicate dates:")
print(df["Date"].duplicated().sum())

print("\nZero or negative Treasury Yield values:")
print((df["Treasury_10Y"] <= 0).sum())


# ============================================================
# CLEANING SUMMARY
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

print("\nMissing Treasury Yield values:")
print(df["Treasury_10Y"].isna().sum())

print("\nDuplicate dates:")
print(df["Date"].duplicated().sum())

print("\nMinimum 10-Year Treasury Yield:")
print(df["Treasury_10Y"].min())

print("\nMaximum 10-Year Treasury Yield:")
print(df["Treasury_10Y"].max())


# ============================================================
# DISPLAY FIRST AND LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN TREASURY YIELD OBSERVATIONS")
print("=" * 70)

print(
    df[
        ["Date", "Treasury_10Y"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("LAST 10 CLEAN TREASURY YIELD OBSERVATIONS")
print("=" * 70)

print(
    df[
        ["Date", "Treasury_10Y"]
    ].tail(10).to_string(index=False)
)


# ============================================================
# REMOVE TEMPORARY VALIDATION VARIABLES
# ============================================================

df = df[
    [
        "Date",
        "Treasury_10Y"
    ]
].copy()


# ============================================================
# SAVE CLEAN DATA
# ============================================================

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean 10-Year Treasury Yield data saved to:")
print(output_file)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal shape:")
print(df.shape)

print("\n10-Year Treasury Yield cleaning completed successfully.")

print(
    "\nIMPORTANT: Missing Treasury Yield observations were removed, "
    "not interpolated or replaced with zero."
)

print(
    "The daily yield changes and basis-point changes above "
    "were calculated for validation only."
)

print(
    "The final Treasury yield change variable has NOT yet "
    "been created."
)