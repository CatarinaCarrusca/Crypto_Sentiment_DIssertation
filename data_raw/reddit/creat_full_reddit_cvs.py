from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

RAW_FILE = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation/data_raw/reddit/"
    "reddit_posts_raw_minimized.csv"
)

OUTPUT_FILE = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation/data_raw/reddit/"
    "reddit_posts_2021_2025_FULL.csv"
)

# ============================================================
# EXPECTED DATA
# ============================================================

EXPECTED_ROWS = 141_560

EXPECTED_YEAR_COUNTS = {
    2021: 54_502,
    2022: 25_197,
    2023: 17_692,
    2024: 22_418,
    2025: 21_751,
}

# ============================================================
# LOAD ORIGINAL RAW CSV
# ============================================================

print("=" * 70)
print("CREATING FULL REDDIT CSV: 2021-2025")
print("=" * 70)

print("\nReading:")
print(RAW_FILE)

if not RAW_FILE.exists():
    raise FileNotFoundError(f"Raw CSV not found:\n{RAW_FILE}")

df = pd.read_csv(
    RAW_FILE,
    low_memory=False,
)

print(f"\nRows loaded: {len(df):,}")
print(f"Columns loaded: {len(df.columns)}")

# ============================================================
# CHECK REQUIRED DATE COLUMN
# ============================================================

if "post_date" not in df.columns:
    raise ValueError(
        "The CSV does not contain the required 'post_date' column."
    )

df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="raise"
)

# ============================================================
# KEEP ONLY 2021-01-01 TO 2025-12-31
# ============================================================

start_date = pd.Timestamp("2021-01-01")
end_date = pd.Timestamp("2025-12-31")

df = df.loc[
    (df["post_date"] >= start_date)
    & (df["post_date"] <= end_date)
].copy()

# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

# created_at gives the correct order within each day
if "created_at" in df.columns:
    df["created_at"] = pd.to_datetime(
        df["created_at"],
        errors="coerce",
        utc=True
    )

    df = df.sort_values(
        ["post_date", "created_at", "post_id"],
        kind="stable"
    )
else:
    df = df.sort_values(
        ["post_date", "post_id"],
        kind="stable"
    )

df = df.reset_index(drop=True)

# ============================================================
# VALIDATE FULL DATASET
# ============================================================

print("\n" + "=" * 70)
print("VALIDATION")
print("=" * 70)

print(f"\nTotal rows: {len(df):,}")
print(f"Earliest date: {df['post_date'].min().date()}")
print(f"Latest date:   {df['post_date'].max().date()}")

year_counts = (
    df["post_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

print("\nPOSTS BY YEAR:")
for year, count in year_counts.items():
    print(f"{year}: {count:,}")

# Check total rows
if len(df) != EXPECTED_ROWS:
    raise ValueError(
        f"\nERROR: Expected {EXPECTED_ROWS:,} rows "
        f"but found {len(df):,}."
    )

# Check every year individually
for year, expected_count in EXPECTED_YEAR_COUNTS.items():

    actual_count = int(year_counts.get(year, 0))

    if actual_count != expected_count:
        raise ValueError(
            f"\nERROR FOR {year}: "
            f"expected {expected_count:,} posts, "
            f"found {actual_count:,}."
        )

print("\nALL YEAR COUNTS PASS.")

# ============================================================
# SAVE AS A NEW FULL CSV
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)

print("\n" + "=" * 70)
print("FULL CSV SAVED")
print("=" * 70)

print("\nSaved to:")
print(OUTPUT_FILE)

print(
    f"\nFile size: "
    f"{OUTPUT_FILE.stat().st_size / 1024**2:.2f} MB"
)

# ============================================================
# RELOAD THE SAVED FILE
# ============================================================

print("\nReloading the NEW CSV to verify it...")

check = pd.read_csv(
    OUTPUT_FILE,
    low_memory=False
)

check["post_date"] = pd.to_datetime(
    check["post_date"],
    errors="raise"
)

print(f"\nReloaded rows: {len(check):,}")
print(f"Earliest: {check['post_date'].min().date()}")
print(f"Latest:   {check['post_date'].max().date()}")

print("\nRELOADED POSTS BY YEAR:")

reloaded_year_counts = (
    check["post_date"]
    .dt.year
    .value_counts()
    .sort_index()
)

for year, count in reloaded_year_counts.items():
    print(f"{year}: {count:,}")

# Final safety check
if len(check) != EXPECTED_ROWS:
    raise ValueError(
        "ERROR: Saved CSV does not contain the expected number of rows."
    )

for year, expected_count in EXPECTED_YEAR_COUNTS.items():

    actual_count = int(reloaded_year_counts.get(year, 0))

    if actual_count != expected_count:
        raise ValueError(
            f"ERROR: Saved CSV failed validation for {year}."
        )

# ============================================================
# SHOW FIRST AND LAST ROWS
# ============================================================

print("\nFIRST 5 POSTS:")
print(
    check[
        ["post_id", "post_date", "subreddit", "title"]
    ]
    .head(5)
    .to_string(index=False)
)

print("\nLAST 5 POSTS:")
print(
    check[
        ["post_id", "post_date", "subreddit", "title"]
    ]
    .tail(5)
    .to_string(index=False)
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print(
    "\nreddit_posts_2021_2025_FULL.csv contains "
    "all 141,560 extracted posts from the 2021-2025 dataset."
)