import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# SETTINGS
# ============================================================

BASE_DIR = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

EXCEL_FILE = BASE_DIR / "data_raw" / "Dissertation Data.xlsx"
OUTPUT_FILE = BASE_DIR / "data_clean" / "gold_clean.csv"

SHEET_NAME = "Gold Retruns (REFINITIV)"


print("=" * 70)
print("GOLD DATA CLEANING")
print("=" * 70)

print("\nLooking for Excel file:")
print(EXCEL_FILE)

print("\nDoes file exist?")
print(EXCEL_FILE.exists())

if not EXCEL_FILE.exists():
    raise FileNotFoundError(f"Excel file not found:\n{EXCEL_FILE}")

print("\nExcel file found successfully.")


# ============================================================
# 1. IMPORT GOLD SHEET
# ============================================================

print("\nImporting Gold sheet...")

gold_raw = pd.read_excel(
    EXCEL_FILE,
    sheet_name=SHEET_NAME,
    header=None
)

print("Gold sheet imported successfully.")

print("\nRaw shape:")
print(gold_raw.shape)

print("\nFirst 10 raw rows:")
print(gold_raw.head(10).to_string(index=False))


# ============================================================
# 2. IDENTIFY THE COLUMN CONTAINING DATE,PRICE
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING GOLD DATA")
print("=" * 70)

gold_column = None

for col in gold_raw.columns:

    sample = gold_raw[col].dropna().astype(str)

    matches = sample.str.match(
        r"^\s*\d{4}-\d{2}-\d{2}\s*,"
    )

    if matches.sum() > 10:
        gold_column = col
        break


if gold_column is None:
    raise ValueError(
        "Could not automatically identify the Gold date/price column."
    )

print("\nGold data found in raw column:")
print(gold_column)


# ============================================================
# 3. KEEP ONLY VALID DATE,PRICE OBSERVATIONS
# ============================================================

gold = gold_raw[[gold_column]].copy()

gold.columns = ["Raw"]

gold["Raw"] = gold["Raw"].astype(str).str.strip()

valid_pattern = gold["Raw"].str.match(
    r"^\d{4}-\d{2}-\d{2}\s*,"
)

gold = gold[valid_pattern].copy()

print("\nRows identified as Gold observations:")
print(len(gold))


# ============================================================
# 4. SPLIT DATE AND GOLD PRICE
# ============================================================

split_data = gold["Raw"].str.split(",", n=1, expand=True)

gold["Date"] = split_data[0].str.strip()
gold["Gold_Price"] = split_data[1].str.strip()

gold = gold[["Date", "Gold_Price"]]


# ============================================================
# 5. CLEAN DATE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATE VARIABLE")
print("=" * 70)

gold["Date"] = pd.to_datetime(
    gold["Date"],
    errors="coerce"
)

invalid_dates = gold["Date"].isna().sum()

print("\nInvalid dates found:")
print(invalid_dates)

if invalid_dates > 0:
    gold = gold.dropna(subset=["Date"])


# ============================================================
# 6. RESTRICT SAMPLE TO 2021-2025
# ============================================================

before_period_filter = len(gold)

gold = gold[
    (gold["Date"] >= "2021-01-01") &
    (gold["Date"] <= "2025-12-31")
].copy()

outside_period = before_period_filter - len(gold)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# 7. CLEAN GOLD PRICE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING GOLD PRICE")
print("=" * 70)

gold["Gold_Price"] = pd.to_numeric(
    gold["Gold_Price"],
    errors="coerce"
)

missing_gold = gold["Gold_Price"].isna().sum()

print("\nNumber of missing Gold observations:")
print(missing_gold)

if missing_gold > 0:

    print("\nDates with missing Gold values:")

    print(
        gold.loc[
            gold["Gold_Price"].isna(),
            ["Date", "Gold_Price"]
        ].to_string(index=False)
    )

    gold = gold.dropna(subset=["Gold_Price"])

    print("\nMissing Gold observations removed:")
    print(missing_gold)


print("\nMissing Gold observations remaining:")
print(gold["Gold_Price"].isna().sum())


# ============================================================
# 8. SORT BY DATE
# ============================================================

gold = gold.sort_values("Date").reset_index(drop=True)


# ============================================================
# 9. DUPLICATE DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

duplicate_dates = gold["Date"].duplicated().sum()

print("\nDuplicate dates found:")
print(duplicate_dates)

if duplicate_dates > 0:

    print("\nDuplicate observations:")

    print(
        gold[
            gold["Date"].duplicated(keep=False)
        ].to_string(index=False)
    )

    # Keep first occurrence
    gold = gold.drop_duplicates(
        subset="Date",
        keep="first"
    ).reset_index(drop=True)

    print("\nDuplicate dates removed.")


# ============================================================
# 10. CHECK ZERO OR NEGATIVE PRICES
# ============================================================

print("\n" + "=" * 70)
print("NON-POSITIVE VALUE CHECK")
print("=" * 70)

non_positive = (gold["Gold_Price"] <= 0).sum()

print("\nZero or negative Gold prices found:")
print(non_positive)

if non_positive > 0:

    print(
        gold[
            gold["Gold_Price"] <= 0
        ].to_string(index=False)
    )

    raise ValueError(
        "Non-positive Gold prices detected. "
        "Review these observations before continuing."
    )


# ============================================================
# 11. TEMPORARY RETURN CALCULATION FOR VALIDATION
# ============================================================

gold["Validation_Log_Return"] = np.log(
    gold["Gold_Price"] /
    gold["Gold_Price"].shift(1)
)

gold["Validation_Return_Pct"] = (
    gold["Gold_Price"].pct_change(fill_method=None) * 100
)


# ============================================================
# 12. SHOW LARGEST GOLD PRICE MOVEMENTS
# ============================================================

print("\n" + "=" * 70)
print("10 LARGEST ABSOLUTE GOLD DAILY MOVEMENTS - VALIDATION ONLY")
print("=" * 70)

largest_movements = (
    gold.dropna(subset=["Validation_Log_Return"])
        .assign(
            Absolute_Move=
            gold["Validation_Log_Return"].abs()
        )
        .sort_values(
            "Absolute_Move",
            ascending=False
        )
        .head(10)
)

print(
    largest_movements[
        [
            "Date",
            "Gold_Price",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)


# ============================================================
# 13. FLAG MOVEMENTS GREATER THAN 5%
# ============================================================

print("\n" + "=" * 70)
print("DAILY GOLD MOVEMENTS GREATER THAN 5%")
print("=" * 70)

large_moves = gold[
    gold["Validation_Return_Pct"].abs() > 5
].copy()

print("\nNumber of absolute daily movements >5%:")
print(len(large_moves))

if len(large_moves) > 0:

    print(
        large_moves[
            [
                "Date",
                "Gold_Price",
                "Validation_Log_Return",
                "Validation_Return_Pct"
            ]
        ].to_string(index=False)
    )

else:

    print("None.")


# ============================================================
# 14. CHECK CONSECUTIVE IDENTICAL PRICES
# ============================================================

print("\n" + "=" * 70)
print("CONSECUTIVE IDENTICAL PRICE CHECK")
print("=" * 70)

gold["Previous_Price"] = gold["Gold_Price"].shift(1)

identical_prices = gold[
    gold["Gold_Price"] == gold["Previous_Price"]
].copy()

print("\nNumber of consecutive identical Gold prices:")
print(len(identical_prices))

if len(identical_prices) > 0:

    print(
        identical_prices[
            [
                "Date",
                "Previous_Price",
                "Gold_Price"
            ]
        ].to_string(index=False)
    )

else:

    print("None.")


# ============================================================
# 15. FINAL CLEANING CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(
    gold[
        ["Date", "Gold_Price"]
    ].isna().sum()
)

print("\nDuplicate dates:")
print(gold["Date"].duplicated().sum())

print("\nZero or negative Gold prices:")
print((gold["Gold_Price"] <= 0).sum())


# ============================================================
# 16. CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean observations:")
print(len(gold))

print("\nFirst date:")
print(gold["Date"].min().date())

print("\nLast date:")
print(gold["Date"].max().date())

print("\nMissing Gold values:")
print(gold["Gold_Price"].isna().sum())

print("\nDuplicate dates:")
print(gold["Date"].duplicated().sum())

print("\nMinimum Gold price:")
print(gold["Gold_Price"].min())

print("\nMaximum Gold price:")
print(gold["Gold_Price"].max())


# ============================================================
# 17. SHOW FIRST AND LAST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN GOLD OBSERVATIONS")
print("=" * 70)

print(
    gold[
        ["Date", "Gold_Price"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 70)
print("LAST 10 CLEAN GOLD OBSERVATIONS")
print("=" * 70)

print(
    gold[
        ["Date", "Gold_Price"]
    ].tail(10).to_string(index=False)
)


# ============================================================
# 18. SAVE CLEAN DATA
# ============================================================

gold_final = gold[
    ["Date", "Gold_Price"]
].copy()

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

gold_final.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# 19. SUCCESS MESSAGE
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean Gold data saved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(gold_final.columns.tolist())

print("\nFinal shape:")
print(gold_final.shape)

print("\nGold cleaning completed successfully.")

print(
    "\nIMPORTANT: Gold_Return has NOT yet been created "
    "as the final regression variable."
)

print(
    "The large-movement calculation above is for "
    "validation only."
)