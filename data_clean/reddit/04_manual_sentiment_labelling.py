import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

FILE = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
    / "stage04_sentiment"
    / "reddit_sentiment_manual_validation_blinded.csv"
)

# Load the 300-post validation sample
df = pd.read_csv(FILE)

if "human_label" not in df.columns:
    raise ValueError("ERROR: human_label column not found.")

if "analysis_text" not in df.columns:
    raise ValueError("ERROR: analysis_text column not found.")

# Convert empty/NULL labels to blank strings
df["human_label"] = (
    df["human_label"]
    .fillna("")
    .astype(str)
    .str.strip()
    .str.lower()
)

valid_labels = {"negative", "neutral", "positive"}

print("=" * 80)
print("MANUAL REDDIT SENTIMENT VALIDATION")
print("=" * 80)
print()
print("1 = NEGATIVE")
print("2 = NEUTRAL")
print("3 = POSITIVE")
print("q = SAVE AND QUIT")
print()
print("Your answer will be saved after EVERY post.")
print("=" * 80)


for idx, row in df.iterrows():

    # Skip posts already labelled
    if row["human_label"] in valid_labels:
        continue

    print("\n\n" + "=" * 80)
    print(f"POST {idx + 1} OF {len(df)}")
    print("=" * 80)

    if "asset" in df.columns:
        print(f"Asset: {row['asset']}")

    if "post_date" in df.columns:
        print(f"Date: {row['post_date']}")

    if "subreddit" in df.columns:
        print(f"Subreddit: {row['subreddit']}")

    print("\nPOST:")
    print("-" * 80)
    print(row["analysis_text"])
    print("-" * 80)

    print("\n1 = Negative")
    print("2 = Neutral")
    print("3 = Positive")
    print("q = Save and quit")

    while True:

        choice = input("\nYOUR LABEL: ").strip().lower()

        if choice == "1":
            label = "negative"
            break

        elif choice == "2":
            label = "neutral"
            break

        elif choice == "3":
            label = "positive"
            break

        elif choice == "q":

            df.to_csv(FILE, index=False)

            completed = df["human_label"].isin(valid_labels).sum()

            print("\nProgress saved successfully.")
            print(f"Completed: {completed}/{len(df)}")
            print(f"Remaining: {len(df) - completed}")

            raise SystemExit

        else:
            print("Please type only 1, 2, 3, or q.")

    # Save your label
    df.at[idx, "human_label"] = label

    # SAVE AFTER EVERY POST
    df.to_csv(FILE, index=False)

    completed = df["human_label"].isin(valid_labels).sum()

    print(f"\n✓ Saved: {label.upper()}")
    print(f"Progress: {completed}/{len(df)}")


print("\n\n" + "=" * 80)
print("MANUAL VALIDATION COMPLETE")
print("=" * 80)

print(f"\nAll {len(df)} posts have been labelled.")
print("\nSaved successfully to:")
print(FILE)

print("\nYou can now rerun:")
print("04_post_level_sentiment_analysis.py")