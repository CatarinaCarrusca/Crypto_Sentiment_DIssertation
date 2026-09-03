from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)
DATA = PROJECT_ROOT / "data_processed"

files = [
    "btc_returns.csv",
    "btc_volume_processed.csv",
    "eth_returns.csv",
    "eth_volume_processed.csv",
    "sp500_returns.csv",
    "vix_change.csv",
    "gold_returns.csv",
    "dxy_returns.csv",
    "us10y_change_processed.csv",
    "cross_crypto_returns.csv",
    "master_aligned_dataset.csv",
    "forecast_structure.csv",
    "information_aligned_dataset.csv",
    "final_forecast_dataset.csv",
]

for filename in files:
    path = DATA / filename

    print("\n" + "=" * 90)
    print(filename)
    print("=" * 90)

    if not path.exists():
        print("FILE NOT FOUND")
        continue

    df = pd.read_csv(path)

    print("Shape:", df.shape)

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nLast 3 rows:")
    print(df.tail(3).to_string(index=False))