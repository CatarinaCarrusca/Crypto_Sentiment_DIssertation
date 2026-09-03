import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path("/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation")

excel_file = BASE_DIR / "data_raw" / "Dissertation Data.xlsx"
output_file = BASE_DIR / "data_clean" / "dxy_clean.csv"

sheet_name = "US DOLLAR INDEX(REFINITIV)"

print("=" * 70)
print("US DOLLAR INDEX (DXY) DATA CLEANING")
print("=" * 70)

print("\nLooking for Excel file:")
print(excel_file)

print("\nDoes file exist?")
print(excel_file.exists())

if not excel_file.exists():
    raise FileNotFoundError(f"Excel file not found: {excel_file}")

print("\nExcel file found successfully.")


# ============================================================
# 1. IMPORT RAW DXY SHEET
# ============================================================

print("\nImporting US Dollar Index sheet...")

raw = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    header=None
)

print("US Dollar Index sheet imported successfully.")

print("\nRaw shape:")
print(raw.shape)

print("\nFirst 20 raw rows:")
print(raw.head(20).to_string(index=False))


# ============================================================
# 2. FIND HISTORICAL DATA HEADER
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING HISTORICAL DXY DATA")
print("=" * 70)

header_row = None

for i in range(len(raw)):
    row_values = raw.iloc[i].astype(str).str.strip().tolist()

    if "Exchange Date" in row_values and "Trade Price" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError(
        "Could not find the historical data header containing "
        "'Exchange Date' and 'Trade Price'."
    )

print("\nHistorical data header found at row:")
print(header_row)


# ============================================================
# 3. EXTRACT DATE AND TRADE PRICE
# ============================================================

historical = raw.iloc[header_row + 1:, :2].copy()

historical.columns = [
    "Date",
    "DXY"
]

print("\nHistorical rows extracted:")
print(len(historical))

print("\nFirst 10 extracted observations:")
print(historical.head(10).to_string(index=False))


# ============================================================
# 4. CLEAN DATE VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATE VARIABLE")
print("=" * 70)

historical["Date"] = pd.to_datetime(
    historical["Date"],
    errors="coerce",
    dayfirst=True
)

invalid_dates = historical["Date"].isna().sum()

print("\nInvalid/non-date rows found:")
print(invalid_dates)

# Remove rows that are not actual historical observations
historical = historical.dropna(subset=["Date"]).copy()


# ============================================================
# 5. KEEP SAMPLE PERIOD 2021-2025
# ============================================================

before_period_filter = len(historical)

historical = historical[
    (historical["Date"] >= "2021-01-01") &
    (historical["Date"] <= "2025-12-31")
].copy()

outside_period = before_period_filter - len(historical)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# 6. CLEAN DXY VALUES
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DXY VALUES")
print("=" * 70)

historical["DXY"] = pd.to_numeric(
    historical["DXY"],
    errors="coerce"
)

missing_dxy = historical["DXY"].isna().sum()

print("\nNumber of missing DXY observations:")
print(missing_dxy)

if missing_dxy > 0:
    print("\nDates with missing DXY values:")
    print(
        historical.loc[
            historical["DXY"].isna(),
            ["Date", "DXY"]
        ].to_string(index=False)
    )

# Remove missing DXY observations
before_missing_removal = len(historical)

historical = historical.dropna(subset=["DXY"]).copy()

removed_missing = before_missing_removal - len(historical)

print("\nMissing DXY observations removed:")
print(removed_missing)

print("\nMissing DXY observations remaining:")
print(historical["DXY"].isna().sum())


# ============================================================
# 7. SORT BY DATE
# ============================================================

historical = historical.sort_values("Date").reset_index(drop=True)


# ============================================================
# 8. DUPLICATE DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

duplicate_dates = historical["Date"].duplicated().sum()

print("\nDuplicate dates found:")
print(duplicate_dates)

if duplicate_dates > 0:
    print("\nDuplicate observations:")
    print(
        historical[
            historical["Date"].duplicated(keep=False)
        ].to_string(index=False)
    )

    # Keep first observation for each date
    historical = historical.drop_duplicates(
        subset="Date",
        keep="first"
    ).copy()


# ============================================================
# 9. CHECK ZERO OR NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("NON-POSITIVE VALUE CHECK")
print("=" * 70)

non_positive = (historical["DXY"] <= 0).sum()

print("\nZero or negative DXY values found:")
print(non_positive)

if non_positive > 0:
    print(
        historical.loc[
            historical["DXY"] <= 0
        ].to_string(index=False)
    )


# ============================================================
# 10. TEMPORARY RETURN FOR VALIDATION ONLY
# ============================================================

historical["Validation_Log_Return"] = np.log(
    historical["DXY"] / historical["DXY"].shift(1)
)

historical["Validation_Return_Pct"] = (
    historical["DXY"].pct_change(fill_method=None) * 100
)


# ============================================================
# 11. LARGEST DAILY MOVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("10 LARGEST ABSOLUTE DXY DAILY MOVEMENTS - VALIDATION ONLY")
print("=" * 70)

largest_moves = (
    historical
    .dropna(subset=["Validation_Log_Return"])
    .assign(
        Abs_Return=lambda x:
        x["Validation_Log_Return"].abs()
    )
    .sort_values("Abs_Return", ascending=False)
    .head(10)
)

print(
    largest_moves[
        [
            "Date",
            "DXY",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)


# ============================================================
# 12. FLAG MOVEMENTS GREATER THAN 2%
# ============================================================

print("\n" + "=" * 70)
print("DAILY DXY MOVEMENTS GREATER THAN 2%")
print("=" * 70)

large_moves = historical[
    historical["Validation_Return_Pct"].abs() > 2
].copy()

print("\nNumber of absolute daily movements >2%:")
print(len(large_moves))

if len(large_moves) > 0:
    print(
        large_moves[
            [
                "Date",
                "DXY",
                "Validation_Log_Return",
                "Validation_Return_Pct"
            ]
        ].to_string(index=False)
    )
else:
    print("None.")


# ============================================================
# 13. CONSECUTIVE IDENTICAL VALUE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONSECUTIVE IDENTICAL DXY CHECK")
print("=" * 70)

historical["Previous_DXY"] = historical["DXY"].shift(1)

identical = historical[
    historical["DXY"] == historical["Previous_DXY"]
].copy()

print("\nNumber of consecutive identical DXY values:")
print(len(identical))

if len(identical) > 0:
    print(
        identical[
            [
                "Date",
                "Previous_DXY",
                "DXY"
            ]
        ].to_string(index=False)
    )
else:
    print("None.")


# ============================================================
# 14. FINAL CLEAN DATASET
# ============================================================

dxy_clean = historical[
    [
        "Date",
        "DXY"
    ]
].copy()


# ============================================================
# 15. FINAL CLEANING CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(dxy_clean.isna().sum())

print("\nDuplicate dates:")
print(dxy_clean["Date"].duplicated().sum())

print("\nZero or negative DXY values:")
print((dxy_clean["DXY"] <= 0).sum())


# ============================================================
# 16. CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean observations:")
print(len(dxy_clean))

print("\nFirst date:")
print(dxy_clean["Date"].min().date())

print("\nLast date:")
print(dxy_clean["Date"].max().date())

print("\nMissing DXY values:")
print(dxy_clean["DXY"].isna().sum())

print("\nDuplicate dates:")
print(dxy_clean["Date"].duplicated().sum())

print("\nMinimum DXY:")
print(dxy_clean["DXY"].min())

print("\nMaximum DXY:")
print(dxy_clean["DXY"].max())


# ============================================================
# 17. DISPLAY FIRST AND LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN DXY OBSERVATIONS")
print("=" * 70)

print(
    dxy_clean.head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("LAST 10 CLEAN DXY OBSERVATIONS")
print("=" * 70)

print(
    dxy_clean.tail(10).to_string(index=False)
)


# ============================================================
# 18. SAVE CLEAN DATA
# ============================================================

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

dxy_clean.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean US Dollar Index data saved to:")
print(output_file)

print("\nFinal columns:")
print(dxy_clean.columns.tolist())

print("\nFinal shape:")
print(dxy_clean.shape)

print("\nUS Dollar Index cleaning completed successfully.")

print(
    "\nIMPORTANT: DXY_Return has NOT yet been created as the "
    "final regression variable."
)

print(
    "The daily movement calculations above are for validation only."
)