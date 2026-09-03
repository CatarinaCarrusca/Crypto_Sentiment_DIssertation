import pandas as pd
from pathlib import Path

# ============================================================
# ETH DAILY PRICE CLEANING
# ============================================================

print("=" * 60)
print("ETH DAILY PRICE CLEANING")
print("=" * 60)

# ------------------------------------------------------------
# 1. FILE PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

excel_file = PROJECT_ROOT / "data_raw" / "Dissertation Data.xlsx"
output_file = PROJECT_ROOT / "data_clean" / "eth_price_clean.csv"

sheet_name = "ETH Daily returns(REFINITIV)"

print("\nLooking for Excel file:")
print(excel_file)

if not excel_file.exists():
    raise FileNotFoundError(
        f"\nExcel file not found:\n{excel_file}"
    )

print("\nExcel file found successfully.")


# ------------------------------------------------------------
# 2. IMPORT ETH SHEET WITHOUT ASSUMING A HEADER
# ------------------------------------------------------------

print("\nImporting ETH Refinitiv sheet...")

raw = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    header=None
)

print("ETH sheet imported successfully.")

print("\nRaw shape:")
print(raw.shape)


# ------------------------------------------------------------
# 3. FIND THE HISTORICAL DATA HEADER
# ------------------------------------------------------------

header_row = None

for i in range(len(raw)):
    row_values = raw.iloc[i].astype(str).str.strip().tolist()

    if "Exchange Date" in row_values and "Mid Price" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError(
        "\nCould not find the historical ETH data header "
        "containing 'Exchange Date' and 'Mid Price'."
    )

print("\nHistorical data header found at Excel/Pandas row:")
print(header_row)


# ------------------------------------------------------------
# 4. READ THE ACTUAL COLUMN NAMES
# ------------------------------------------------------------

historical_columns = raw.iloc[header_row].tolist()

print("\nHistorical columns found:")

for col in historical_columns:
    print(f"- {col}")


# ------------------------------------------------------------
# 5. EXTRACT ONLY THE HISTORICAL DATA BELOW THE HEADER
# ------------------------------------------------------------

eth = raw.iloc[header_row + 1:].copy()

eth.columns = historical_columns


# ------------------------------------------------------------
# 6. KEEP ONLY THE VARIABLES NEEDED
# ------------------------------------------------------------

required_columns = ["Exchange Date", "Mid Price"]

for column in required_columns:
    if column not in eth.columns:
        raise ValueError(
            f"\nRequired column '{column}' was not found."
        )

eth = eth[required_columns].copy()


# ------------------------------------------------------------
# 7. RENAME VARIABLES
# ------------------------------------------------------------

eth = eth.rename(
    columns={
        "Exchange Date": "Date",
        "Mid Price": "ETH_Price"
    }
)


# ------------------------------------------------------------
# 8. CONVERT DATE COLUMN
# ------------------------------------------------------------

eth["Date"] = pd.to_datetime(
    eth["Date"],
    errors="coerce",
    dayfirst=True
)


# ------------------------------------------------------------
# 9. CLEAN ETH PRICE COLUMN
# ------------------------------------------------------------

# Convert values such as "3,020.73" into numeric values.
eth["ETH_Price"] = (
    eth["ETH_Price"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.strip()
)

eth["ETH_Price"] = pd.to_numeric(
    eth["ETH_Price"],
    errors="coerce"
)


# ------------------------------------------------------------
# 10. REMOVE NON-HISTORICAL / INVALID ROWS
# ------------------------------------------------------------

rows_before = len(eth)

eth = eth.dropna(
    subset=["Date", "ETH_Price"]
).copy()

rows_after = len(eth)

print(
    f"\nInvalid/non-price rows removed: "
    f"{rows_before - rows_after}"
)


# ------------------------------------------------------------
# 11. KEEP ONLY THE DISSERTATION SAMPLE
# ------------------------------------------------------------

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2025-12-31")

rows_before_period_filter = len(eth)

eth = eth[
    (eth["Date"] >= start_date) &
    (eth["Date"] <= end_date)
].copy()

rows_after_period_filter = len(eth)

print(
    f"Rows outside 2021-2025 removed: "
    f"{rows_before_period_filter - rows_after_period_filter}"
)


# ------------------------------------------------------------
# 12. SORT CHRONOLOGICALLY
# ------------------------------------------------------------

eth = eth.sort_values("Date").reset_index(drop=True)


# ------------------------------------------------------------
# 13. CHECK DUPLICATE DATES
# ------------------------------------------------------------

duplicate_count = eth["Date"].duplicated().sum()

print(f"\nDuplicate dates found: {duplicate_count}")

if duplicate_count > 0:

    print("\nDuplicate observations:")
    print(
        eth[
            eth["Date"].duplicated(keep=False)
        ].to_string(index=False)
    )

    # Keep the first observation for each date.
    eth = eth.drop_duplicates(
        subset="Date",
        keep="first"
    ).reset_index(drop=True)


# ------------------------------------------------------------
# 14. CHECK MISSING VALUES
# ------------------------------------------------------------

print("\nMissing values:")

print(
    eth[
        ["Date", "ETH_Price"]
    ].isna().sum()
)


# ------------------------------------------------------------
# 15. CHECK FOR ZERO OR NEGATIVE PRICES
# ------------------------------------------------------------

invalid_prices = eth[
    eth["ETH_Price"] <= 0
]

print(
    f"\nZero or negative ETH prices found: "
    f"{len(invalid_prices)}"
)

if len(invalid_prices) > 0:

    print("\nInvalid price observations:")

    print(
        invalid_prices.to_string(index=False)
    )


# ------------------------------------------------------------
# 16. BASIC CLEANING SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("CLEANING SUMMARY")
print("=" * 60)

print(
    f"\nNumber of clean ETH observations: "
    f"{len(eth):,}"
)

if not eth.empty:

    print("\nFirst date:")
    print(eth["Date"].min().date())

    print("\nLast date:")
    print(eth["Date"].max().date())

    print("\nMinimum ETH price:")
    print(eth["ETH_Price"].min())

    print("\nMaximum ETH price:")
    print(eth["ETH_Price"].max())


# ------------------------------------------------------------
# 17. DISPLAY FIRST AND LAST OBSERVATIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("FIRST 10 CLEAN ETH OBSERVATIONS")
print("=" * 60)

print(
    eth.head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("LAST 10 CLEAN ETH OBSERVATIONS")
print("=" * 60)

print(
    eth.tail(10).to_string(index=False)
)


# ------------------------------------------------------------
# 18. SAVE CLEAN DATA
# ------------------------------------------------------------

output_file.parent.mkdir(
    parents=True,
    exist_ok=True
)

eth.to_csv(
    output_file,
    index=False
)


# ------------------------------------------------------------
# 19. FINAL MESSAGE
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("SUCCESS")
print("=" * 60)

print("\nClean ETH price data saved to:")
print(output_file)

print("\nFinal columns:")
print(eth.columns.tolist())

print("\nFinal shape:")
print(eth.shape)

print("\nETH price cleaning completed successfully.")

print(
    "\nIMPORTANT: ETH_Return has NOT been calculated yet."
)

print(
    "The next step is to validate this cleaned price series "
    "and then calculate the daily ETH log return."
)