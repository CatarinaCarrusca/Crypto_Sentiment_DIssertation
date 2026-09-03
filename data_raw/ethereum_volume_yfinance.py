import yfinance as yf

eth = yf.download(
    "ETH-USD",
    start="2021-01-01",
    end="2026-01-01",
    interval="1d"
)

print(eth)
eth.to_csv("/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation/data_raw/ethereum_volume_2021_2025.csv")