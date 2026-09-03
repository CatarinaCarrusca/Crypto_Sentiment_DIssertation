import pandas as pd
from pathlib import Path


# ============================================================
# BTC DAILY PRICE CLEANING
# Source: Refinitiv
# Period: 01-Jan-2021 to 31-Dec-2025
#
# Objective:
# Keep only:
#   Date
#   BTC_Price (Refinitiv Mid Price)
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

excel_file = (
    project_folder
    / "data_raw"
    / "Dissertation Data.xlsx"
)

output_folder = (
    project_folder
    / "data_clean"
)

output_file = (
    output_folder
    / "btc_price_clean.csv"
)

# Make sure output folder exists
output_folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CHECK EXCEL FILE EXISTS
# ============================================================

print("=" * 60)
print("BTC DAILY PRICE CLEANING")
print("=" * 60)

print("\nLooking for Excel file:")
print(excel_file)

if not excel_file.exists():
    raise FileNotFoundError(
        f"\nExcel file not found:\n{excel_file}"
    )

print("\nExcel file found successfully.")


# ============================================================
# 3. IMPORT BTC SHEET WITHOUT ASSUMING A HEADER ROW
# ============================================================

sheet_name = "BTC Daily retruns(REFINITIV) "

print("\nImporting BTC Refinitiv sheet...")

btc_raw = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    header=None
)

print("BTC sheet imported successfully.")

print("\nRaw shape:")
print(btc_raw.shape)


# ============================================================
# 4. FIND THE ACTUAL HISTORICAL DATA HEADER
# ============================================================

# Refinitiv includes metadata and statistics before the actual
# daily price history.
#
# Instead of assuming a fixed row number, we search for the row
# containing "Exchange Date".

header_row = None

for index, row in btc_raw.iterrows():

    row_values = (
        row.astype(str)
        .str.strip()
        .str.lower()
        .tolist()
    )

    if "exchange date" in row_values:
        header_row = index
        break


if header_row is None:
    raise ValueError(
        "\nCould not find the 'Exchange Date' header "
        "in the BTC Refinitiv sheet."
    )

print("\nHistorical data header found at Excel/Pandas row:")
print(header_row)


# ============================================================
# 5. CREATE THE HISTORICAL PRICE TABLE
# ============================================================

# Use the discovered row as the column names

btc_history = btc_raw.iloc[header_row + 1:].copy()

btc_history.columns = (
    btc_raw.iloc[header_row]
    .astype(str)
    .str.strip()
)


print("\nHistorical columns found:")

for column in btc_history.columns:
    print("-", column)


# ============================================================
# 6. CHECK REQUIRED COLUMNS EXIST
# ============================================================

required_columns = [
    "Exchange Date",
    "Mid Price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in btc_history.columns
]

if missing_columns:

    raise ValueError(
        "\nRequired BTC columns were not found:\n"
        + str(missing_columns)
        + "\n\nColumns available:\n"
        + str(list(btc_history.columns))
    )


# ============================================================
# 7. KEEP ONLY DATE AND MID PRICE
# ============================================================

btc = btc_history[
    [
        "Exchange Date",
        "Mid Price"
    ]
].copy()


# ============================================================
# 8. RENAME VARIABLES
# ============================================================

btc = btc.rename(
    columns={
        "Exchange Date": "Date",
        "Mid Price": "BTC_Price"
    }
)


# ============================================================
# 9. CONVERT DATE
# ============================================================

btc["Date"] = pd.to_datetime(
    btc["Date"],
    errors="coerce"
)


# ============================================================
# 10. CONVERT BTC PRICE TO NUMERIC
# ============================================================

# Remove commas in case Refinitiv stores values such as
# "105,000.50" as text.

btc["BTC_Price"] = (
    btc["BTC_Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

btc["BTC_Price"] = pd.to_numeric(
    btc["BTC_Price"],
    errors="coerce"
)


# ============================================================
# 11. REMOVE ROWS THAT ARE NOT ACTUAL DAILY OBSERVATIONS
# ============================================================

# Any Refinitiv footer/statistics rows should fail the date
# conversion and become NaT.

before_invalid_removal = len(btc)

btc = btc.dropna(
    subset=["Date", "BTC_Price"]
).copy()

removed_invalid = (
    before_invalid_removal
    - len(btc)
)

print(
    f"\nInvalid/non-price rows removed: "
    f"{removed_invalid}"
)


# ============================================================
# 12. KEEP ONLY THE DISSERTATION SAMPLE
# ============================================================

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2025-12-31")

before_date_filter = len(btc)

btc = btc[
    (btc["Date"] >= start_date)
    &
    (btc["Date"] <= end_date)
].copy()

removed_outside_period = (
    before_date_filter
    - len(btc)
)

print(
    f"Rows outside 2021-2025 removed: "
    f"{removed_outside_period}"
)


# ============================================================
# 13. SORT CHRONOLOGICALLY
# ============================================================

# This is essential before calculating log returns later.

btc = btc.sort_values(
    by="Date",
    ascending=True
).reset_index(drop=True)


# ============================================================
# 14. CHECK FOR DUPLICATE DATES
# ============================================================

duplicate_count = btc["Date"].duplicated().sum()

print(
    f"\nDuplicate dates found: "
    f"{duplicate_count}"
)

if duplicate_count > 0:

    print("\nDuplicate observations:")

    print(
        btc[
            btc["Date"].duplicated(
                keep=False
            )
        ].to_string(index=False)
    )

    # Do NOT silently average duplicate prices.
    # Keep the first observation for now and report what happened.

    btc = btc.drop_duplicates(
        subset="Date",
        keep="first"
    ).copy()

    print(
        "\nDuplicate dates removed using "
        "the first observation."
    )


# ============================================================
# 15. CHECK MISSING VALUES
# ============================================================

print("\nMissing values:")

print(
    btc[
        [
            "Date",
            "BTC_Price"
        ]
    ].isna().sum()
)


# ============================================================
# 16. CHECK FOR INVALID BTC PRICES
# ============================================================

invalid_prices = btc[
    btc["BTC_Price"] <= 0
]

print(
    f"\nZero or negative BTC prices found: "
    f"{len(invalid_prices)}"
)

if len(invalid_prices) > 0:

    print(invalid_prices.to_string(index=False))

    raise ValueError(
        "\nInvalid BTC prices were found. "
        "Review them before continuing."
    )


# ============================================================
# 17. FINAL VALIDATION
# ============================================================

if btc.empty:
    raise ValueError(
        "\nBTC dataset is empty after cleaning."
    )


print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(
    f"\nNumber of clean BTC observations: "
    f"{len(btc):,}"
)

print(
    "\nFirst date:"
)

print(
    btc["Date"].min().date()
)

print(
    "\nLast date:"
)

print(
    btc["Date"].max().date()
)

print(
    "\nMinimum BTC price:"
)

print(
    btc["BTC_Price"].min()
)

print(
    "\nMaximum BTC price:"
)

print(
    btc["BTC_Price"].max()
)


# ============================================================
# 18. DISPLAY FIRST 10 CLEAN OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("FIRST 10 CLEAN BTC OBSERVATIONS")
print("=" * 60)

print(
    btc.head(10).to_string(
        index=False
    )
)


# ============================================================
# 19. DISPLAY LAST 10 CLEAN OBSERVATIONS
# ============================================================

print("\n" + "=" * 60)
print("LAST 10 CLEAN BTC OBSERVATIONS")
print("=" * 60)

print(
    btc.tail(10).to_string(
        index=False
    )
)


# ============================================================
# 20. SAVE CLEAN BTC PRICE DATA
# ============================================================

btc.to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)


# ============================================================
# 21. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nClean BTC price data saved to:")

print(output_file)

print("\nFinal columns:")

print(
    btc.columns.tolist()
)

print("\nFinal shape:")

print(
    btc.shape
)

print(
    "\nBTC price cleaning completed successfully."
)

print(
    "\nIMPORTANT: BTC_Return has NOT been calculated yet."
)

print(
    "The next step is to validate this cleaned price series "
    "and then calculate the daily BTC log return."
)