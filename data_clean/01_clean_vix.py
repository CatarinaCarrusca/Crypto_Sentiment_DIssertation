import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# VIX DATA CLEANING
# Dissertation: Crypto Sentiment and Returns
# Sample period: 01-Jan-2021 to 31-Dec-2025
# ============================================================

print("=" * 70)
print("VIX DATA CLEANING")
print("=" * 70)


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_DIR = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

EXCEL_FILE = PROJECT_DIR / "data_raw" / "Dissertation Data.xlsx"

OUTPUT_DIR = PROJECT_DIR / "data_clean"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "vix_clean.csv"


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
# 2. IMPORT RAW VIX SHEET
# ============================================================

print("\nImporting VIX sheet...")

raw_vix = pd.read_excel(
    EXCEL_FILE,
    sheet_name="VIX(wrds)",
    header=None
)

print("VIX sheet imported successfully.")

print("\nRaw shape:")
print(raw_vix.shape)

print("\nFirst 10 raw rows:")
print(raw_vix.head(10).to_string(index=False))


# ============================================================
# 3. IDENTIFY AND PARSE VIX DATA
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING VIX DATA")
print("=" * 70)

# The WRDS data has been imported into Excel as one
# comma-separated column, e.g.:
#
# 2021-01-04,26.97
#
# Therefore, identify the column containing these observations.

data_column = None

for col in raw_vix.columns:

    sample = raw_vix[col].dropna().astype(str)

    if sample.str.contains(",", regex=False).any():
        data_column = col
        break


if data_column is None:
    raise ValueError(
        "Could not identify the column containing the VIX observations."
    )

print(f"\nVIX data found in raw column: {data_column}")


# Convert column to string
vix_strings = raw_vix[data_column].astype(str).str.strip()


# Keep rows beginning with YYYY-MM-DD
date_pattern = r"^\d{4}-\d{2}-\d{2},"

mask = vix_strings.str.match(date_pattern, na=False)

vix_strings = vix_strings[mask].copy()

print("\nRows identified as VIX observations:")
print(len(vix_strings))


# ============================================================
# 4. SPLIT DATE AND VIX VALUE
# ============================================================

split_data = vix_strings.str.split(",", n=1, expand=True)

vix = pd.DataFrame({
    "Date": split_data[0],
    "VIX": split_data[1]
})


# ============================================================
# 5. CONVERT DATA TYPES
# ============================================================

vix["Date"] = pd.to_datetime(
    vix["Date"],
    errors="coerce"
)

vix["VIX"] = pd.to_numeric(
    vix["VIX"],
    errors="coerce"
)


# ============================================================
# 6. CHECK INVALID DATES
# ============================================================

invalid_dates = vix["Date"].isna().sum()

print("\nInvalid dates found:")
print(invalid_dates)

if invalid_dates > 0:

    print("\nRows with invalid dates:")
    print(
        vix[vix["Date"].isna()].to_string(index=False)
    )

    # Invalid dates cannot be used
    vix = vix.dropna(subset=["Date"]).copy()


# ============================================================
# 7. KEEP DISSERTATION SAMPLE: 2021-2025
# ============================================================

before_period_filter = len(vix)

vix = vix[
    (vix["Date"] >= "2021-01-01") &
    (vix["Date"] <= "2025-12-31")
].copy()

outside_period = before_period_filter - len(vix)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# 8. SORT BY DATE
# ============================================================

vix = vix.sort_values("Date").reset_index(drop=True)


# ============================================================
# 9. CHECK DUPLICATE DATES
# ============================================================

duplicate_mask = vix["Date"].duplicated(keep=False)

duplicate_count = duplicate_mask.sum()

print("\nDuplicate dates found:")
print(duplicate_count)

if duplicate_count > 0:

    print("\nDuplicate observations:")
    print(
        vix.loc[
            duplicate_mask,
            ["Date", "VIX"]
        ].to_string(index=False)
    )

    # Keep the first observation for each date
    vix = (
        vix
        .drop_duplicates(
            subset="Date",
            keep="first"
        )
        .copy()
    )


# ============================================================
# 10. CHECK MISSING VIX VALUES
# ============================================================

print("\n" + "=" * 70)
print("MISSING VIX VALUES")
print("=" * 70)

missing_vix = vix["VIX"].isna().sum()

print("\nNumber of missing VIX observations:")
print(missing_vix)

if missing_vix > 0:

    print("\nDates with missing VIX values:")

    print(
        vix.loc[
            vix["VIX"].isna(),
            ["Date", "VIX"]
        ].to_string(index=False)
    )


# ============================================================
# 11. REMOVE MISSING VIX OBSERVATIONS
# ============================================================

# IMPORTANT:
# Missing VIX values are NOT replaced with zero.
# They are NOT interpolated.
# Only observations where the VIX value itself is missing
# are removed.

missing_before_removal = vix["VIX"].isna().sum()

vix = vix.dropna(subset=["VIX"]).copy()

print("\nMissing VIX observations removed:")
print(missing_before_removal)

print("\nMissing VIX observations remaining:")
print(vix["VIX"].isna().sum())

print("\nObservations remaining after removing missing VIX:")
print(len(vix))


# ============================================================
# 12. CHECK ZERO OR NEGATIVE VALUES
# ============================================================

non_positive = vix[vix["VIX"] <= 0]

print("\nZero or negative VIX values found:")
print(len(non_positive))

if len(non_positive) > 0:

    print(non_positive.to_string(index=False))


# ============================================================
# 13. VALIDATE HIGH AND LOW OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("10 HIGHEST VIX OBSERVATIONS")
print("=" * 70)

print(
    vix.nlargest(
        10,
        "VIX"
    )[["Date", "VIX"]].to_string(index=False)
)


print("\n" + "=" * 70)
print("10 LOWEST VIX OBSERVATIONS")
print("=" * 70)

print(
    vix.nsmallest(
        10,
        "VIX"
    )[["Date", "VIX"]].to_string(index=False)
)


# ============================================================
# 14. TEMPORARY VIX CHANGE FOR VALIDATION
# ============================================================

# This is ONLY used to identify suspicious observations.
# It is NOT yet the final regression variable.

vix["Validation_VIX_Change"] = vix["VIX"].diff()

vix["Abs_Validation_VIX_Change"] = (
    vix["Validation_VIX_Change"].abs()
)


print("\n" + "=" * 70)
print("10 LARGEST ABSOLUTE VIX CHANGES - VALIDATION ONLY")
print("=" * 70)

print(
    vix.nlargest(
        10,
        "Abs_Validation_VIX_Change"
    )[
        [
            "Date",
            "VIX",
            "Validation_VIX_Change"
        ]
    ].to_string(index=False)
)


# ============================================================
# 15. FINAL CLEAN DATASET
# ============================================================

clean_vix = vix[
    [
        "Date",
        "VIX"
    ]
].copy()


# ============================================================
# 16. FINAL VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(clean_vix.isna().sum())

print("\nDuplicate dates:")
print(clean_vix["Date"].duplicated().sum())

print("\nZero or negative VIX values:")
print((clean_vix["VIX"] <= 0).sum())


# ============================================================
# 17. CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean observations:")
print(len(clean_vix))

print("\nFirst date:")
print(clean_vix["Date"].min().date())

print("\nLast date:")
print(clean_vix["Date"].max().date())

print("\nMissing VIX values:")
print(clean_vix["VIX"].isna().sum())

print("\nDuplicate dates:")
print(clean_vix["Date"].duplicated().sum())

print("\nMinimum VIX:")
print(clean_vix["VIX"].min())

print("\nMaximum VIX:")
print(clean_vix["VIX"].max())


# ============================================================
# 18. DISPLAY FIRST AND LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN VIX OBSERVATIONS")
print("=" * 70)

print(
    clean_vix.head(10).to_string(index=False)
)


print("\n" + "=" * 70)
print("LAST 10 CLEAN VIX OBSERVATIONS")
print("=" * 70)

print(
    clean_vix.tail(10).to_string(index=False)
)


# ============================================================
# 19. SAVE CLEAN DATA
# ============================================================

clean_vix.to_csv(
    OUTPUT_FILE,
    index=False,
    date_format="%Y-%m-%d"
)


# ============================================================
# 20. SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean VIX data saved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(clean_vix.columns.tolist())

print("\nFinal shape:")
print(clean_vix.shape)

print("\nVIX cleaning completed successfully.")

print(
    "\nIMPORTANT: Missing VIX observations were removed, "
    "not interpolated or replaced with zero."
)

print(
    "VIX_Change has NOT yet been created as the final "
    "regression variable."
)