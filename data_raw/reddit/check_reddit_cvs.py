import pandas as pd

path = (
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation/data_raw/reddit/"
    "reddit_posts_raw_minimized.csv"
)

df = pd.read_csv(path, low_memory=False)

print("LAST 10 ROWS:")
print(
    df[
        ["post_id", "post_date", "subreddit", "title"]
    ].tail(10).to_string(index=False)
)