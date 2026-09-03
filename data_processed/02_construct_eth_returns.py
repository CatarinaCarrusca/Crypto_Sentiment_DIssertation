from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# ETH RETURN VARIABLE CONSTRUCTION
# ============================================================

print("=" * 70)
print("ETH RETURN VARIABLE CONSTRUCTION")
print("=" * 70)


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

CLEAN_DIR = PROJECT_ROOT / "data_clean"
PROCESSED_DIR = PROJECT_ROOT / "data_processed"

# Create processed-data folder if it does not already exist
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# IMPORTANT:
# Change this filename ONLY if your cleaned ETH file has a different name.
INPUT_FILE = CLEAN_DIR / "eth_price_clean.csv"

OUTPUT_FILE = PROCESSED_DIR / "eth_returns.csv"


print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}\n\n"
        "Check the exact filename of your cleaned ETH price CSV."
    )


# ============================================================
# 2. IMPORT CLEANED ETH PRICE DATA
# ============================================================

print("\n" + "=" * 70)
print("IMPORTING CLEANED ETH PRICE DATA")
print("=" * 70)

eth = pd.read_csv(INPUT_FILE)

print("\nRaw imported shape:")
print(eth.shape)

print("\nColumns found:")
print(eth.columns.tolist())

print("\nFirst 10 rows:")
print(eth.head(10).to_string(index=False))

print("\nLast 10 rows:")
print(eth.tail(10).to_string(index=False))


# ============================================================
# 3. STANDARDISE DATE
# ============================================================

if "Date" not in eth.columns:
    raise ValueError(
        "The cleaned ETH dataset does not contain a column called 'Date'."
    )

eth["Date"] = pd.to_datetime(
    eth["Date"],
    errors="coerce"
)

invalid_dates = eth["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates were found in the cleaned ETH dataset."
    )


# ============================================================
# 4. CHECK ETH PRICE VARIABLE
# ============================================================

if "ETH_Price" not in eth.columns:
    raise ValueError(
        "The cleaned ETH dataset does not contain a column called 'ETH_Price'."
    )

eth["ETH_Price"] = pd.to_numeric(
    eth["ETH_Price"],
    errors="coerce"
)

missing_prices = eth["ETH_Price"].isna().sum()
non_positive_prices = (eth["ETH_Price"] <= 0).sum()

print("\nMissing ETH prices:")
print(missing_prices)

print("\nZero or negative ETH prices:")
print(non_positive_prices)

if missing_prices > 0:
    raise ValueError(
        "Missing ETH prices were found."
    )

if non_positive_prices > 0:
    raise ValueError(
        "Zero or negative ETH prices were found."
    )


# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

eth = (
    eth
    .sort_values("Date")
    .reset_index(drop=True)
)

duplicate_dates = eth["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

if duplicate_dates > 0:
    raise ValueError(
        "Duplicate dates were found. "
        "Returns should not be calculated until these are resolved."
    )


# ============================================================
# 6. CONSTRUCT DAILY ETH LOG RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING ETH_RETURN")
print("=" * 70)

# ETH_Return_t = ln(ETH_Price_t / ETH_Price_(t-1))

eth["ETH_Return"] = np.log(
    eth["ETH_Price"] / eth["ETH_Price"].shift(1)
)

print("\nETH_Return constructed successfully.")

print("\nMissing ETH_Return observations:")
print(eth["ETH_Return"].isna().sum())


# ============================================================
# 7. CONSTRUCT ONE-DAY LAGGED ETH RETURN
# ============================================================

print("\n" + "=" * 70)
print("CONSTRUCTING ETH_LAGGED_RETURN")
print("=" * 70)

# ETH_Lagged_Return_t = ETH_Return_(t-1)

eth["ETH_Lagged_Return"] = eth["ETH_Return"].shift(1)

print("\nETH_Lagged_Return constructed successfully.")

print("\nMissing ETH_Lagged_Return observations:")
print(eth["ETH_Lagged_Return"].isna().sum())


# ============================================================
# 8. INSPECT FIRST OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FIRST 10 OBSERVATIONS")
print("=" * 70)

print(
    eth[
        [
            "Date",
            "ETH_Price",
            "ETH_Return",
            "ETH_Lagged_Return"
        ]
    ].head(10).to_string(index=False)
)


# ============================================================
# 9. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY STATISTICS")
print("=" * 70)

print(
    eth[
        [
            "ETH_Price",
            "ETH_Return",
            "ETH_Lagged_Return"
        ]
    ].describe()
)


# ============================================================
# 10. CHECK FOR INFINITE VALUES
# ============================================================

return_inf = np.isinf(eth["ETH_Return"]).sum()
lagged_inf = np.isinf(eth["ETH_Lagged_Return"]).sum()

print("\nInfinite ETH_Return values:")
print(return_inf)

print("\nInfinite ETH_Lagged_Return values:")
print(lagged_inf)

if return_inf > 0 or lagged_inf > 0:
    raise ValueError(
        "Infinite return values were detected."
    )


# ============================================================
# 11. SAVE PROCESSED ETH DATA
# ============================================================

print("\n" + "=" * 70)
print("SAVING PROCESSED ETH DATA")
print("=" * 70)

eth.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nFile saved successfully:")
print(OUTPUT_FILE)


# ============================================================
# 12. FINAL CHECK
# ============================================================

print("\n" + "=" * 70)
print("FINAL CHECK")
print("=" * 70)

print("\nNumber of observations:")
print(len(eth))

print("\nDate range:")
print(eth["Date"].min(), "to", eth["Date"].max())

print("\nFinal variables:")
print(eth.columns.tolist())

print("\nMissing values:")
print(
    eth[
        [
            "ETH_Price",
            "ETH_Return",
            "ETH_Lagged_Return"
        ]
    ].isna().sum()
)

print("\n" + "=" * 70)
print("ETH RETURN CONSTRUCTION COMPLETE")
print("=" * 70)