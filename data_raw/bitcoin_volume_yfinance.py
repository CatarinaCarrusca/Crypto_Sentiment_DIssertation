import yfinance as yf

btc = yf.download(
    "BTC-USD",
    start="2021-01-01",
    end="2025-12-31",
    interval="1d"
)

print(btc)
btc.to_csv("/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation/data_raw/bitcoin_volume_2021_2025.csv")