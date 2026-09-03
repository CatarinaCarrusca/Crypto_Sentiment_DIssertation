import yfinance as yf
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. Set dissertation period
# --------------------------------------------------

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"   # yfinance end date is exclusive

# --------------------------------------------------
# 2. Download Gold Futures data from Yahoo Finance
# --------------------------------------------------

gold = yf.download(
    "GC=F",
    start=START_DATE,
    end=END_DATE,
    interval="1d",
    auto_adjust=False,
    progress=True
)

# --------------------------------------------------
# 3. Check that data were downloaded
# --------------------------------------------------

if gold.empty:
    raise ValueError("No gold data were downloaded.")

print("\nDownloaded data:")
print(gold)

# --------------------------------------------------
# 4. Keep only the closing price
# --------------------------------------------------

gold = gold[["Close"]].copy()

# Handle yfinance MultiIndex columns
if isinstance(gold.columns, pd.MultiIndex):
    gold.columns = gold.columns.get_level_values(0)

gold = gold.reset_index()

gold = gold.rename(columns={
    "Date": "date",
    "Close": "gold_price"
})

# --------------------------------------------------
# 5. Keep dissertation period only
# --------------------------------------------------

gold["date"] = pd.to_datetime(gold["date"])

gold = gold[
    (gold["date"] >= "2021-01-01") &
    (gold["date"] <= "2025-12-31")
]

# --------------------------------------------------
# 6. Check the final data
# --------------------------------------------------

print("\nClean Gold price data:")
print(gold)

print("\nFirst observations:")
print(gold.head())

print("\nLast observations:")
print(gold.tail())

print("\nNumber of observations:", len(gold))

print("\nDate range:")
print(gold["date"].min(), "to", gold["date"].max())

# --------------------------------------------------
# 7. Save CSV in data_raw
# --------------------------------------------------

output_path = Path(__file__).parent / "gold_price_2021_2025.csv"

gold.to_csv(output_path, index=False)

print("\nCSV successfully created:")
print(output_path)