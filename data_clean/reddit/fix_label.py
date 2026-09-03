import pandas as pd
from pathlib import Path

FILE = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation/"
    "data_clean/reddit/stage04_sentiment/"
    "reddit_sentiment_manual_validation_blinded.csv"
)

df = pd.read_csv(FILE)

# Find the Swell post
mask = df["analysis_text"].str.contains(
    "Swell: A Bull Case",
    case=False,
    na=False
)

print("Rows found:", mask.sum())

# Show what it currently contains
print(df.loc[mask, ["analysis_text", "human_label"]])

# Correct the human label
df.loc[mask, "human_label"] = "positive"

# Save
df.to_csv(FILE, index=False)

print("\nCORRECTION SAVED")
print(df.loc[mask, ["analysis_text", "human_label"]])