# ============================================================
# 02_clean_reddit_posts.py
# Stage 02: Reddit cleaning and audit
# ============================================================

from pathlib import Path
import re
import html

import numpy as np
import pandas as pd


# ============================================================
# SECTION 1 — PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data_raw"
    / "reddit"
    / "reddit_posts_2021_2025_FULL.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# OUTPUT CSV FILES
# ============================================================

CLEAN_FULL_FILE = (
    OUTPUT_DIR
    / "reddit_posts_cleaned_full.csv"
)

PRIMARY_FILE = (
    OUTPUT_DIR
    / "reddit_posts_primary_sentiment_sample.csv"
)

AUDIT_FILE = (
    OUTPUT_DIR
    / "reddit_cleaning_audit.csv"
)

YEAR_ASSET_FILE = (
    OUTPUT_DIR
    / "reddit_cleaning_by_year_asset.csv"
)

SUBREDDIT_FILE = (
    OUTPUT_DIR
    / "reddit_cleaning_by_subreddit.csv"
)

DAILY_FILE = (
    OUTPUT_DIR
    / "reddit_daily_coverage_after_cleaning.csv"
)

EXAMPLES_FILE = (
    OUTPUT_DIR
    / "reddit_exclusion_examples.csv"
)


# ============================================================
# EXPECTED RAW DATA
# ============================================================

EXPECTED_ROWS = 141_560

EXPECTED_YEAR_COUNTS = {
    2021: 54_502,
    2022: 25_197,
    2023: 17_692,
    2024: 22_418,
    2025: 21_751,
}

EXPECTED_SUBREDDIT_COUNTS = {
    "Bitcoin": 118_491,
    "BitcoinMarkets": 2_660,
    "ethereum": 20_409,
}

REQUIRED_COLUMNS = [
    "post_id",
    "created_at",
    "post_date",
    "subreddit",
    "title",
    "body",
    "score",
    "upvote_ratio",
    "num_comments",
    "crosspost_parent_id",
    "url",
    "permalink",
]


# ============================================================
# BASIC PRINTING FUNCTION
# ============================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
# ============================================================
# SECTION 2 — TEXT CLEANING AND HELPER FUNCTIONS
# ============================================================


def safe_text(value):
    """
    Convert values safely to strings.

    Missing values become empty strings.
    """

    if pd.isna(value):
        return ""

    return str(value)


# ============================================================
# CLEAN TEXT FOR LATER SENTIMENT ANALYSIS
# ============================================================

def clean_text(value):
    """
    Conservative Reddit text cleaning.

    IMPORTANT:
    We preserve:
    - capitalization
    - punctuation
    - negation
    - emojis

    These may contain useful sentiment information.

    We remove:
    - URLs
    - HTML
    - Markdown formatting
    - invisible characters
    - excessive whitespace
    """

    text = safe_text(value)

    # Decode HTML entities.
    # Example:
    # Bitcoin &amp; Ethereum
    # becomes
    # Bitcoin & Ethereum

    text = html.unescape(text)


    # Remove zero-width / invisible characters

    text = re.sub(
        r"[\u200B-\u200D\uFEFF]",
        "",
        text
    )


    # Remove fenced code blocks

    text = re.sub(
        r"```.*?```",
        " ",
        text,
        flags=re.DOTALL
    )


    # Markdown links:
    # [Bitcoin website](https://...)
    # becomes:
    # Bitcoin website

    text = re.sub(
        r"\[([^\]]+)\]\([^)]+\)",
        r"\1",
        text
    )


    # Remove normal URLs

    text = re.sub(
        r"https?://\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )


    # Remove www URLs

    text = re.sub(
        r"www\.\S+",
        " ",
        text,
        flags=re.IGNORECASE
    )


    # Remove HTML tags

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # Remove common Markdown formatting characters

    text = re.sub(
        r"[*_~`]+",
        "",
        text
    )


    # Remove Reddit quote marker at beginning of lines

    text = re.sub(
        r"(?m)^\s*>\s?",
        "",
        text
    )


    # Normalize multiple spaces/newlines/tabs

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# NORMALIZATION FOR DUPLICATE DETECTION ONLY
# ============================================================

def normalize_for_matching(value):
    """
    Create a simplified version of the text for detecting
    exact duplicate/repeated content.

    IMPORTANT:
    This normalized version is NOT used for sentiment scoring.
    """

    text = safe_text(value)

    # Lowercase ONLY for matching

    text = text.lower()


    # Decode HTML

    text = html.unescape(text)


    # Remove URLs

    text = re.sub(
        r"https?://\S+",
        " ",
        text
    )

    text = re.sub(
        r"www\.\S+",
        " ",
        text
    )


    # Keep letters, numbers and spaces only

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )


    # Normalize whitespace

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# COUNT ALPHABETIC CHARACTERS
# ============================================================

def alpha_count(value):
    """
    Count alphabetic characters.

    Used to identify genuinely empty / unusable / link-only
    posts after cleaning.
    """

    text = safe_text(value)

    return sum(
        character.isalpha()
        for character in text
    )


# ============================================================
# DETECT REDDIT DELETED / REMOVED PLACEHOLDERS
# ============================================================

def is_deleted_removed(value):
    """
    Identify Reddit deleted/removed placeholders.
    """

    text = (
        safe_text(value)
        .strip()
        .lower()
    )

    placeholders = {
        "[deleted]",
        "[removed]",
        "deleted",
        "removed",
    }

    return text in placeholders


# ============================================================
# MATCH A LIST OF REGEX PATTERNS
# ============================================================

def matches_any_pattern(text, patterns):
    """
    Return True if the text matches at least one supplied
    regular-expression pattern.
    """

    text = safe_text(text)

    for pattern in patterns:

        if re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        ):
            return True

    return False
# ============================================================
# SECTION 3A — CLEANING RULES
# ============================================================


# Conservative promotional / spam-like patterns

PROMO_PATTERNS = [

    r"\breferral\s+(?:link|code)\b",

    r"\bref\s+code\b",

    r"\bpromo\s+code\b",

    r"\baffiliate\s+link\b",

    r"\bjoin\s+(?:my|our)\s+telegram\b",

    r"\bjoin\s+(?:my|our)\s+discord\b",

    r"\bguaranteed\s+(?:profit|return|returns)\b",

    r"\bdouble\s+your\s+(?:btc|bitcoin|eth|ethereum|crypto)\b",

]


# Recurring discussion / template patterns

TEMPLATE_PATTERNS = [

    r"^\s*daily\s+discussion\b",

    r"^\s*daily\s+discussion\s+thread\b",

    r"^\s*bitcoin\s+daily\s+discussion\b",

    r"^\s*daily\s+bitcoin\s+discussion\b",

    r"^\s*daily\s+altcoin\s+discussion\b",

    r"^\s*altcoin\s+discussion\b",

]
# ============================================================
# SECTION 3B — LOAD INPUT CSV AND CHECK REQUIRED COLUMNS
# ============================================================


section("LOADING REDDIT DATA")


# Check that the input CSV exists

if not INPUT_FILE.exists():

    raise FileNotFoundError(

        "\nThe Reddit input CSV was not found:\n"
        f"{INPUT_FILE}\n\n"
        "Expected file:\n"
        "reddit_posts_2021_2025_FULL.csv"

    )


# Load the full Reddit dataset

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)


print("\nInput file loaded successfully.")

print(
    f"Rows loaded: {len(df):,}"
)

print(
    f"Columns loaded: {len(df.columns):,}"
)


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

missing_columns = [

    column
    for column in REQUIRED_COLUMNS
    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "\nThe input CSV is missing required columns:\n"
        f"{missing_columns}"

    )


print("\nRequired columns: PASS")
# ============================================================
# SECTION 3C — VALIDATE ROW COUNT AND POST IDs
# ============================================================


section("VALIDATING ROW COUNT AND POST IDs")


# ============================================================
# CHECK TOTAL ROW COUNT
# ============================================================

actual_rows = len(df)


if actual_rows != EXPECTED_ROWS:

    raise ValueError(

        f"\nUnexpected number of Reddit posts.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {actual_rows:,}"

    )


print(
    f"\nRow count: PASS "
    f"({actual_rows:,} rows)"
)


# ============================================================
# CHECK FOR MISSING POST IDs
# ============================================================

missing_post_ids = int(
    df["post_id"]
    .isna()
    .sum()
)


if missing_post_ids != 0:

    raise ValueError(

        f"\nFound {missing_post_ids:,} "
        f"missing post_id values."

    )


print(
    f"Missing post IDs: PASS "
    f"({missing_post_ids:,})"
)


# ============================================================
# CHECK FOR DUPLICATE POST IDs
# ============================================================

duplicate_post_ids = int(
    df["post_id"]
    .duplicated()
    .sum()
)


if duplicate_post_ids != 0:

    raise ValueError(

        f"\nFound {duplicate_post_ids:,} "
        f"duplicate post_id values."

    )


print(
    f"Duplicate post IDs: PASS "
    f"({duplicate_post_ids:,})"
)


# ============================================================
# CONFIRM UNIQUE POST COUNT
# ============================================================

unique_post_ids = int(
    df["post_id"]
    .nunique()
)


if unique_post_ids != EXPECTED_ROWS:

    raise ValueError(

        f"\nExpected {EXPECTED_ROWS:,} unique post IDs, "
        f"but found {unique_post_ids:,}."

    )


print(
    f"Unique post IDs: PASS "
    f"({unique_post_ids:,})"
)
# ============================================================
# SECTION 3D — VALIDATE DATES AND YEAR COUNTS
# ============================================================


section("VALIDATING DATES AND YEAR COUNTS")


# ============================================================
# PARSE POST DATE
# ============================================================

df["post_date"] = pd.to_datetime(
    df["post_date"],
    errors="raise"
)


# ============================================================
# PARSE CREATION TIMESTAMP
# ============================================================

df["created_at"] = pd.to_datetime(
    df["created_at"],
    format="mixed",
    errors="raise",
    utc=True
)


print("\nDate parsing: PASS")

# ============================================================
# CHECK EARLIEST AND LATEST POST DATES
# ============================================================

earliest_date = df["post_date"].min()

latest_date = df["post_date"].max()


print(
    f"Earliest post date: "
    f"{earliest_date.date()}"
)

print(
    f"Latest post date: "
    f"{latest_date.date()}"
)


if earliest_date != pd.Timestamp("2021-01-01"):

    raise ValueError(

        "\nUnexpected earliest post date.\n"
        f"Expected: 2021-01-01\n"
        f"Found:    {earliest_date.date()}"

    )


if latest_date != pd.Timestamp("2025-12-30"):

    raise ValueError(

        "\nUnexpected latest post date.\n"
        f"Expected: 2025-12-30\n"
        f"Found:    {latest_date.date()}"

    )


print("\nDate range: PASS")


# ============================================================
# CREATE TEMPORARY YEAR SERIES
# ============================================================

year_counts = (
    df["post_date"]
    .dt.year
    .value_counts()
    .sort_index()
)


# ============================================================
# DISPLAY YEAR COUNTS
# ============================================================

print("\nPosts by year:")


for year in sorted(EXPECTED_YEAR_COUNTS):

    actual_count = int(
        year_counts.get(
            year,
            0
        )
    )

    print(
        f"{year}: "
        f"{actual_count:,}"
    )


# ============================================================
# VALIDATE EXACT YEAR COUNTS
# ============================================================

for year, expected_count in EXPECTED_YEAR_COUNTS.items():

    actual_count = int(
        year_counts.get(
            year,
            0
        )
    )


    if actual_count != expected_count:

        raise ValueError(

            f"\nUnexpected post count for {year}.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {actual_count:,}"

        )


print("\nYear counts: PASS")


# ============================================================
# CHECK THAT ONLY 2021-2025 ARE PRESENT
# ============================================================

actual_years = set(
    df["post_date"]
    .dt.year
    .unique()
)


expected_years = set(
    EXPECTED_YEAR_COUNTS.keys()
)


if actual_years != expected_years:

    raise ValueError(

        "\nUnexpected years found in the Reddit dataset.\n"
        f"Expected: {sorted(expected_years)}\n"
        f"Found:    {sorted(actual_years)}"

    )


print("Year range 2021-2025 only: PASS")
# ============================================================
# SECTION 3E — VALIDATE SUBREDDITS AND CREATE ASSET VARIABLE
# ============================================================


section("VALIDATING SUBREDDITS AND CREATING ASSET VARIABLE")


# ============================================================
# COUNT POSTS BY SUBREDDIT
# ============================================================

subreddit_counts = (
    df["subreddit"]
    .value_counts()
)


print("\nPosts by subreddit:")


for subreddit in EXPECTED_SUBREDDIT_COUNTS:

    actual_count = int(
        subreddit_counts.get(
            subreddit,
            0
        )
    )

    print(
        f"{subreddit}: "
        f"{actual_count:,}"
    )


# ============================================================
# VALIDATE EXACT SUBREDDIT COUNTS
# ============================================================

for subreddit, expected_count in EXPECTED_SUBREDDIT_COUNTS.items():

    actual_count = int(
        subreddit_counts.get(
            subreddit,
            0
        )
    )


    if actual_count != expected_count:

        raise ValueError(

            f"\nUnexpected post count for r/{subreddit}.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {actual_count:,}"

        )


print("\nSubreddit counts: PASS")


# ============================================================
# CHECK FOR UNEXPECTED SUBREDDITS
# ============================================================

actual_subreddits = set(
    df["subreddit"]
    .dropna()
    .unique()
)


expected_subreddits = set(
    EXPECTED_SUBREDDIT_COUNTS.keys()
)


unexpected_subreddits = (
    actual_subreddits
    - expected_subreddits
)


if unexpected_subreddits:

    raise ValueError(

        "\nUnexpected subreddits found:\n"
        f"{sorted(unexpected_subreddits)}"

    )


print("No unexpected subreddits: PASS")


# ============================================================
# CREATE ASSET MAPPING
# ============================================================

ASSET_MAP = {

    "Bitcoin": "BTC",

    "BitcoinMarkets": "BTC",

    "ethereum": "ETH",

}


df["asset"] = (
    df["subreddit"]
    .map(ASSET_MAP)
)


# ============================================================
# CHECK ASSET MAPPING
# ============================================================

missing_asset = int(
    df["asset"]
    .isna()
    .sum()
)


if missing_asset != 0:

    raise ValueError(

        f"\nAsset mapping failed for "
        f"{missing_asset:,} posts."

    )


print("\nAsset mapping: PASS")


# ============================================================
# DISPLAY ASSET COUNTS
# ============================================================

asset_counts = (
    df["asset"]
    .value_counts()
)


print("\nPosts by asset:")


for asset in ["BTC", "ETH"]:

    count = int(
        asset_counts.get(
            asset,
            0
        )
    )

    print(
        f"{asset}: "
        f"{count:,}"
    )


# ============================================================
# FINAL INPUT VALIDATION MESSAGE
# ============================================================

print("\nAll raw Reddit input validation checks: PASS")
# ============================================================
# SECTION 4A — CLEAN TITLE/BODY AND CREATE ANALYSIS TEXT
# ============================================================


section("CLEANING REDDIT TEXT")


# ============================================================
# PRESERVE ORIGINAL TITLE AND BODY
# ============================================================

# Missing title/body values are converted to empty strings.
#
# IMPORTANT:
# A missing body by itself is NOT an exclusion reason.
# Many legitimate Reddit posts contain only a title.

df["title_original"] = (
    df["title"]
    .fillna("")
    .astype(str)
)


df["body_original"] = (
    df["body"]
    .fillna("")
    .astype(str)
)


print("\nOriginal title/body preserved: PASS")


# ============================================================
# CLEAN TITLE
# ============================================================

df["title_clean"] = (
    df["title_original"]
    .map(clean_text)
)


# ============================================================
# CLEAN BODY
# ============================================================

df["body_clean"] = (
    df["body_original"]
    .map(clean_text)
)


print("Title and body cleaning: PASS")


# ============================================================
# IDENTIFY WHEN BOTH TITLE AND BODY ARE PRESENT
# ============================================================

both_title_and_body = (

    df["title_clean"]
    .str.len()
    .gt(0)

    &

    df["body_clean"]
    .str.len()
    .gt(0)

)


# ============================================================
# CREATE ANALYSIS TEXT
# ============================================================

# If both title and body exist:
#
#     Title. Body
#
# If only one exists:
#
#     use whichever text exists.
#
# This preserves useful punctuation/capitalization for
# later sentiment scoring.

df["analysis_text"] = np.where(

    both_title_and_body,

    df["title_clean"]
    + ". "
    + df["body_clean"],

    df["title_clean"]
    + df["body_clean"]

)


# ============================================================
# FINAL WHITESPACE NORMALIZATION
# ============================================================

df["analysis_text"] = (

    df["analysis_text"]
    .str.replace(
        r"\s+",
        " ",
        regex=True
    )
    .str.strip()

)


# ============================================================
# CREATE NORMALIZED TEXT FOR MATCHING ONLY
# ============================================================

# IMPORTANT:
#
# normalized_text is ONLY for:
# - duplicate detection
# - repeated-text diagnostics
#
# It will NOT be used for sentiment scoring.

df["normalized_text"] = (
    df["analysis_text"]
    .map(normalize_for_matching)
)


# ============================================================
# COUNT ALPHABETIC CHARACTERS
# ============================================================

df["analysis_alpha_count"] = (
    df["analysis_text"]
    .map(alpha_count)
)


# ============================================================
# BASIC TEXT DIAGNOSTICS
# ============================================================

non_empty_analysis = int(

    df["analysis_text"]
    .str.len()
    .gt(0)
    .sum()

)


empty_analysis = int(

    df["analysis_text"]
    .str.len()
    .eq(0)
    .sum()

)


missing_body_count = int(

    df["body_clean"]
    .str.len()
    .eq(0)
    .sum()

)


print("\nText diagnostics:")

print(
    f"Posts with non-empty analysis_text: "
    f"{non_empty_analysis:,}"
)

print(
    f"Posts with empty analysis_text: "
    f"{empty_analysis:,}"
)

print(
    f"Posts with empty/missing cleaned body: "
    f"{missing_body_count:,}"
)


# ============================================================
# IMPORTANT CHECK
# ============================================================

# Missing body alone is NOT marked for exclusion here.
# Exclusion flags are created separately in the next sections.

print(
    "\nMissing body alone is NOT treated as an exclusion."
)

print("Analysis text construction: PASS")
# ============================================================
# SECTION 4B — SORT POSTS AND CREATE BASIC CLEANING FLAGS
# ============================================================


section("CREATING BASIC CLEANING FLAGS")


# ============================================================
# SORT POSTS CHRONOLOGICALLY
# ============================================================

# Sorting before duplicate detection is important because
# later we want to keep the EARLIEST occurrence of duplicated
# text and flag later occurrences.

df = (

    df.sort_values(

        by=[
            "created_at",
            "post_id",
        ],

        kind="stable"

    )

    .reset_index(drop=True)

)


print("\nChronological sorting: PASS")


# ============================================================
# FLAG 1 — EMPTY ANALYSIS TEXT
# ============================================================

df["flag_empty_text"] = (

    df["analysis_text"]
    .str.strip()
    .eq("")

)


empty_text_count = int(
    df["flag_empty_text"].sum()
)


print(
    f"Empty analysis text: "
    f"{empty_text_count:,}"
)


# ============================================================
# FLAG 2 — DELETED / REMOVED CONTENT
# ============================================================

# We check the original title and original body for Reddit
# placeholders such as:
#
# [deleted]
# [removed]
#
# This is kept separate from ordinary missing body values.

title_deleted_removed = (

    df["title_original"]
    .map(is_deleted_removed)

)


body_deleted_removed = (

    df["body_original"]
    .map(is_deleted_removed)

)


df["flag_deleted_removed"] = (

    title_deleted_removed
    |

    body_deleted_removed

)


deleted_removed_count = int(
    df["flag_deleted_removed"].sum()
)


print(
    f"Deleted/removed placeholders: "
    f"{deleted_removed_count:,}"
)


# ============================================================
# FLAG 3 — TOO LITTLE USABLE TEXT
# ============================================================

# Posts with fewer than 3 alphabetic characters after
# cleaning do not contain enough textual information for
# meaningful post-level sentiment analysis.

df["flag_too_short"] = (

    df["analysis_alpha_count"]
    .lt(3)

)


too_short_count = int(
    df["flag_too_short"].sum()
)


print(
    f"Posts with fewer than 3 alphabetic characters: "
    f"{too_short_count:,}"
)


# ============================================================
# COMBINED UNUSABLE-TEXT FLAG
# ============================================================

df["flag_unusable_text"] = (

    df["flag_empty_text"]

    |

    df["flag_deleted_removed"]

    |

    df["flag_too_short"]

)


unusable_count = int(
    df["flag_unusable_text"].sum()
)


print(
    f"Combined unusable-text flag: "
    f"{unusable_count:,}"
)


# ============================================================
# FLAG 4 — CROSSPOST
# ============================================================

# crosspost_parent_id identifies Reddit crossposts.
#
# Convert missing values to an empty string first so that
# NaN values are not incorrectly treated as crossposts.

crosspost_value = (

    df["crosspost_parent_id"]
    .fillna("")
    .astype(str)
    .str.strip()

)


df["flag_crosspost"] = (

    crosspost_value.ne("")

    &

    crosspost_value.str.lower().ne("nan")

    &

    crosspost_value.str.lower().ne("none")

)


crosspost_count = int(
    df["flag_crosspost"].sum()
)


print(
    f"Crossposts flagged: "
    f"{crosspost_count:,}"
)


# ============================================================
# CHECK MISSING BODY IS NOT AUTOMATICALLY EXCLUDED
# ============================================================

missing_body_only = (

    df["body_clean"]
    .str.len()
    .eq(0)

    &

    df["title_clean"]
    .str.len()
    .gt(0)

)


missing_body_only_count = int(
    missing_body_only.sum()
)


print(
    f"Title-only posts: "
    f"{missing_body_only_count:,}"
)


print(
    "\nTitle-only posts are NOT automatically excluded."
)

print("Basic cleaning flags: PASS")
# ============================================================
# SECTION 4C — ADDITIONAL CLEANING FLAGS
# ============================================================


section("CREATING ADDITIONAL CLEANING FLAGS")


# ============================================================
# FLAG 5 — PROMOTIONAL / SPAM-LIKE CONTENT
# ============================================================

# Uses the conservative PROMO_PATTERNS created in Section 3A.
#
# This is content-based only.
# We are NOT making any author-level or bot-level claims.

df["flag_promotional"] = (

    df["analysis_text"]
    .map(
        lambda text: matches_any_pattern(
            text,
            PROMO_PATTERNS
        )
    )

)


promotional_count = int(
    df["flag_promotional"].sum()
)


print(
    f"\nPromotional/spam-like posts flagged: "
    f"{promotional_count:,}"
)


# ============================================================
# FLAG 6 — RECURRING DISCUSSION TEMPLATE
# ============================================================

# We apply the recurring-template rules to the cleaned title.
#
# These are intended to identify recurring discussion threads
# rather than ordinary posts that happen to mention the word
# "discussion".

df["flag_template"] = (

    df["title_clean"]
    .map(
        lambda text: matches_any_pattern(
            text,
            TEMPLATE_PATTERNS
        )
    )

)


template_count = int(
    df["flag_template"].sum()
)


print(
    f"Recurring-template posts flagged: "
    f"{template_count:,}"
)


# ============================================================
# FLAG 7 — EXACT DUPLICATE NORMALIZED TEXT WITHIN ASSET
# ============================================================

# Duplicate detection is performed separately within BTC/ETH.
#
# Because the dataframe was sorted chronologically in
# Section 4B, keep="first" means:
#
# KEEP the earliest occurrence.
# FLAG later identical occurrences.
#
# Empty normalized text is not treated as a duplicate here
# because unusable text has its own flag.

valid_normalized_text = (

    df["normalized_text"]
    .str.len()
    .gt(0)

)


df["flag_exact_duplicate"] = (

    valid_normalized_text

    &

    df.duplicated(
        subset=[
            "asset",
            "normalized_text",
        ],
        keep="first"
    )

)


exact_duplicate_count = int(
    df["flag_exact_duplicate"].sum()
)


print(
    f"Later exact duplicate posts flagged: "
    f"{exact_duplicate_count:,}"
)


# ============================================================
# REPEATED-TEXT FREQUENCY
# ============================================================

# Count how often each normalized text occurs within each asset.
#
# This variable is useful for diagnosing highly repeated
# content.

text_frequency = (

    df.groupby(
        [
            "asset",
            "normalized_text",
        ],
        dropna=False
    )["post_id"]
    .transform("size")

)


df["normalized_text_frequency"] = np.where(

    valid_normalized_text,

    text_frequency,

    0

)


df["normalized_text_frequency"] = (

    df["normalized_text_frequency"]
    .astype(int)

)


# ============================================================
# DIAGNOSTIC FLAG — TEXT REPEATED AT LEAST 5 TIMES
# ============================================================

# IMPORTANT:
#
# This is a DIAGNOSTIC flag only.
#
# A post is NOT excluded merely because its text occurs
# five or more times.
#
# Exact duplicate handling is already performed separately
# above.

df["flag_repeated_text_5plus"] = (

    df["normalized_text_frequency"]
    .ge(5)

)


repeated_5plus_count = int(
    df["flag_repeated_text_5plus"].sum()
)


print(
    f"Posts belonging to text repeated >=5 times: "
    f"{repeated_5plus_count:,}"
)

print(
    "Repeated >=5 is diagnostic only."
)


# ============================================================
# FLAG 8 — GENUINELY LINK-ONLY POSTS
# ============================================================

# A post is considered link-only only when:
#
# 1. the original Reddit record contains a URL, AND
# 2. fewer than 3 alphabetic characters remain in
#    analysis_text after cleaning.
#
# This prevents substantive linked posts with meaningful
# titles from being incorrectly removed.

url_value = (

    df["url"]
    .fillna("")
    .astype(str)
    .str.strip()

)


url_present = (

    url_value.ne("")

    &

    url_value.str.lower().ne("nan")

    &

    url_value.str.lower().ne("none")

)


df["flag_link_only"] = (

    url_present

    &

    df["analysis_alpha_count"]
    .lt(3)

)


link_only_count = int(
    df["flag_link_only"].sum()
)


print(
    f"Genuinely link-only posts flagged: "
    f"{link_only_count:,}"
)


# ============================================================
# DISPLAY FLAG COUNTS
# ============================================================

print("\nAdditional cleaning flag counts:")

print(
    f"Promotional/spam-like: "
    f"{promotional_count:,}"
)

print(
    f"Recurring templates: "
    f"{template_count:,}"
)

print(
    f"Later exact duplicates: "
    f"{exact_duplicate_count:,}"
)

print(
    f"Repeated text >=5 diagnostic: "
    f"{repeated_5plus_count:,}"
)

print(
    f"Link-only: "
    f"{link_only_count:,}"
)


print("\nAdditional cleaning flags: PASS")
# ============================================================
# SECTION 4D — CREATE PRIMARY EXCLUSION DECISION
# ============================================================


section("CREATING PRIMARY EXCLUSION DECISION")


# ============================================================
# DEFINE PRIMARY EXCLUSION FLAGS
# ============================================================

# A post is excluded from the PRIMARY sentiment sample if
# ANY of the following is True:
#
# 1. unusable text
# 2. crosspost
# 3. promotional / spam-like content
# 4. recurring discussion template
# 5. later exact duplicate
# 6. genuinely link-only
#
# IMPORTANT:
# flag_repeated_text_5plus is NOT included here because it is
# diagnostic only.

PRIMARY_EXCLUSION_FLAGS = [

    "flag_unusable_text",

    "flag_crosspost",

    "flag_promotional",

    "flag_template",

    "flag_exact_duplicate",

    "flag_link_only",

]


# ============================================================
# CREATE EXCLUDE_PRIMARY
# ============================================================

df["exclude_primary"] = (

    df[PRIMARY_EXCLUSION_FLAGS]
    .any(axis=1)

)


# ============================================================
# CREATE RETAIN_PRIMARY
# ============================================================

df["retain_primary"] = (

    ~df["exclude_primary"]

)


# ============================================================
# CREATE READABLE EXCLUSION REASON
# ============================================================

def build_exclusion_reason(row):

    """
    Return all primary exclusion reasons for a Reddit post.

    If no primary exclusion flag is present, return 'retained'.
    """

    reasons = []


    if row["flag_unusable_text"]:

        reasons.append(
            "unusable_text"
        )


    if row["flag_crosspost"]:

        reasons.append(
            "crosspost"
        )


    if row["flag_promotional"]:

        reasons.append(
            "promotional"
        )


    if row["flag_template"]:

        reasons.append(
            "recurring_template"
        )


    if row["flag_exact_duplicate"]:

        reasons.append(
            "exact_duplicate"
        )


    if row["flag_link_only"]:

        reasons.append(
            "link_only"
        )


    if len(reasons) == 0:

        return "retained"


    return ";".join(reasons)


# Apply the function to every row

df["exclusion_reason"] = (

    df.apply(
        build_exclusion_reason,
        axis=1
    )

)


# ============================================================
# COUNT RETAINED AND EXCLUDED POSTS
# ============================================================

excluded_primary_count = int(

    df["exclude_primary"]
    .sum()

)


retained_primary_count = int(

    df["retain_primary"]
    .sum()

)


# ============================================================
# SAFETY CHECK
# ============================================================

if (
    excluded_primary_count
    + retained_primary_count
    != len(df)
):

    raise ValueError(

        "\nRetained + excluded posts do not equal "
        "the total number of input posts."

    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\nPrimary cleaning decision:")


print(
    f"Input posts: "
    f"{len(df):,}"
)


print(
    f"Excluded from primary sample: "
    f"{excluded_primary_count:,}"
)


print(
    f"Retained for primary sample: "
    f"{retained_primary_count:,}"
)


retained_percentage = (

    retained_primary_count
    / len(df)
    * 100

)


excluded_percentage = (

    excluded_primary_count
    / len(df)
    * 100

)


print(
    f"Retained percentage: "
    f"{retained_percentage:.2f}%"
)


print(
    f"Excluded percentage: "
    f"{excluded_percentage:.2f}%"
)


# ============================================================
# CHECK EXCLUSION REASON
# ============================================================

missing_reason_count = int(

    df["exclusion_reason"]
    .isna()
    .sum()

)


if missing_reason_count != 0:

    raise ValueError(

        f"\nFound {missing_reason_count:,} posts "
        "without an exclusion_reason."

    )


print(
    "\nEvery post has an exclusion decision: PASS"
)


# ============================================================
# CONFIRM REPEATED >=5 IS DIAGNOSTIC ONLY
# ============================================================

print(
    "Repeated-text >=5 remains diagnostic only "
    "and is not a standalone exclusion rule."
)


print("\nPrimary exclusion decision: PASS")

# ============================================================
# SECTION 5A — CREATE CLEANING AUDIT TABLE
# ============================================================


section("CREATING CLEANING AUDIT TABLE")


# ============================================================
# CALCULATE COUNTS FOR EACH CLEANING RULE
# ============================================================

audit_rows = [

    {
        "measure": "Input posts",
        "count": len(df),
    },

    {
        "measure": "Empty analysis text",
        "count": int(
            df["flag_empty_text"].sum()
        ),
    },

    {
        "measure": "Deleted/removed content",
        "count": int(
            df["flag_deleted_removed"].sum()
        ),
    },

    {
        "measure": "Too little usable text",
        "count": int(
            df["flag_too_short"].sum()
        ),
    },

    {
        "measure": "Unusable text - combined",
        "count": int(
            df["flag_unusable_text"].sum()
        ),
    },

    {
        "measure": "Crossposts",
        "count": int(
            df["flag_crosspost"].sum()
        ),
    },

    {
        "measure": "Promotional/spam-like",
        "count": int(
            df["flag_promotional"].sum()
        ),
    },

    {
        "measure": "Recurring templates",
        "count": int(
            df["flag_template"].sum()
        ),
    },

    {
        "measure": "Later exact duplicates",
        "count": int(
            df["flag_exact_duplicate"].sum()
        ),
    },

    {
        "measure": "Link-only",
        "count": int(
            df["flag_link_only"].sum()
        ),
    },

    {
        "measure": "Repeated text >=5 - diagnostic only",
        "count": int(
            df["flag_repeated_text_5plus"].sum()
        ),
    },

    {
        "measure": "Excluded from primary sample",
        "count": int(
            df["exclude_primary"].sum()
        ),
    },

    {
        "measure": "Retained for primary sample",
        "count": int(
            df["retain_primary"].sum()
        ),
    },

]


# ============================================================
# CREATE AUDIT DATAFRAME
# ============================================================

audit = pd.DataFrame(
    audit_rows
)


# ============================================================
# ADD PERCENTAGE OF ORIGINAL INPUT
# ============================================================

audit["percent_of_input"] = (

    audit["count"]
    / len(df)
    * 100

)


# Round percentages for readability

audit["percent_of_input"] = (

    audit["percent_of_input"]
    .round(4)

)


# ============================================================
# ADD RULE TYPE
# ============================================================

rule_type_map = {

    "Input posts":
        "input",

    "Empty analysis text":
        "component_flag",

    "Deleted/removed content":
        "component_flag",

    "Too little usable text":
        "component_flag",

    "Unusable text - combined":
        "primary_exclusion_rule",

    "Crossposts":
        "primary_exclusion_rule",

    "Promotional/spam-like":
        "primary_exclusion_rule",

    "Recurring templates":
        "primary_exclusion_rule",

    "Later exact duplicates":
        "primary_exclusion_rule",

    "Link-only":
        "primary_exclusion_rule",

    "Repeated text >=5 - diagnostic only":
        "diagnostic_only",

    "Excluded from primary sample":
        "final_decision",

    "Retained for primary sample":
        "final_decision",

}


audit["rule_type"] = (

    audit["measure"]
    .map(rule_type_map)

)


# ============================================================
# REORDER COLUMNS
# ============================================================

audit = audit[

    [
        "measure",
        "rule_type",
        "count",
        "percent_of_input",
    ]

]


# ============================================================
# DISPLAY AUDIT TABLE
# ============================================================

print("\nCleaning audit:")

print(
    audit.to_string(
        index=False
    )
)


# ============================================================
# BASIC AUDIT VALIDATION
# ============================================================

audit_retained = int(

    audit.loc[
        audit["measure"]
        == "Retained for primary sample",
        "count"
    ].iloc[0]

)


audit_excluded = int(

    audit.loc[
        audit["measure"]
        == "Excluded from primary sample",
        "count"
    ].iloc[0]

)


if audit_retained != retained_primary_count:

    raise ValueError(

        "\nAudit retained count does not match "
        "the main retained count."

    )


if audit_excluded != excluded_primary_count:

    raise ValueError(

        "\nAudit excluded count does not match "
        "the main excluded count."

    )


if (
    audit_retained
    + audit_excluded
    != EXPECTED_ROWS
):

    raise ValueError(

        "\nAudit retained + excluded counts "
        "do not equal the original input size."

    )


print("\nCleaning audit validation: PASS")
# ============================================================
# SECTION 5B — CLEANING SUMMARY BY YEAR AND ASSET
# ============================================================


section("CREATING CLEANING SUMMARY BY YEAR AND ASSET")


# ============================================================
# CREATE YEAR VARIABLE
# ============================================================

df["year"] = (
    df["post_date"]
    .dt.year
)
# ============================================================
# TEMPORARY QC — ETH 2023 EXCLUSIONS
# ============================================================

eth_2023 = df.loc[
    (df["year"] == 2023)
    & (df["asset"] == "ETH")
].copy()

eth_2023_excluded = eth_2023.loc[
    eth_2023["exclude_primary"]
].copy()

print("\nETH 2023 QC")
print(f"Total ETH 2023 posts: {len(eth_2023):,}")
print(
    f"Excluded ETH 2023 posts: "
    f"{len(eth_2023_excluded):,}"
)

print("\nETH 2023 exclusion flags:")
print(
    eth_2023_excluded[
        [
            "flag_unusable_text",
            "flag_crosspost",
            "flag_promotional",
            "flag_template",
            "flag_exact_duplicate",
            "flag_link_only",
        ]
    ]
    .sum()
)

print("\nRandom ETH 2023 excluded examples:")

print(
    eth_2023_excluded[
        [
            "post_id",
            "subreddit",
            "analysis_text",
            "exclusion_reason",
        ]
    ]
    .sample(
        n=min(20, len(eth_2023_excluded)),
        random_state=42,
    )
    .to_string(index=False)
)


# ============================================================
# GROUP BY YEAR AND ASSET
# ============================================================

year_asset = (

    df.groupby(
        [
            "year",
            "asset",
        ],
        observed=True
    )

    .agg(

        input_posts=(
            "post_id",
            "size"
        ),

        retained_posts=(
            "retain_primary",
            "sum"
        ),

        excluded_posts=(
            "exclude_primary",
            "sum"
        ),

        unusable_posts=(
            "flag_unusable_text",
            "sum"
        ),

        crossposts=(
            "flag_crosspost",
            "sum"
        ),

        promotional_posts=(
            "flag_promotional",
            "sum"
        ),

        template_posts=(
            "flag_template",
            "sum"
        ),

        exact_duplicate_posts=(
            "flag_exact_duplicate",
            "sum"
        ),

        link_only_posts=(
            "flag_link_only",
            "sum"
        ),

        repeated_text_5plus_posts=(
            "flag_repeated_text_5plus",
            "sum"
        ),

    )

    .reset_index()

)


# ============================================================
# CONVERT COUNT COLUMNS TO INTEGERS
# ============================================================

count_columns = [

    "input_posts",

    "retained_posts",

    "excluded_posts",

    "unusable_posts",

    "crossposts",

    "promotional_posts",

    "template_posts",

    "exact_duplicate_posts",

    "link_only_posts",

    "repeated_text_5plus_posts",

]


for column in count_columns:

    year_asset[column] = (
        year_asset[column]
        .astype(int)
    )


# ============================================================
# CALCULATE RETAINED PERCENTAGE
# ============================================================

year_asset["retained_percent"] = (

    year_asset["retained_posts"]

    / year_asset["input_posts"]

    * 100

)


year_asset["retained_percent"] = (

    year_asset["retained_percent"]
    .round(4)

)


# ============================================================
# CALCULATE EXCLUDED PERCENTAGE
# ============================================================

year_asset["excluded_percent"] = (

    year_asset["excluded_posts"]

    / year_asset["input_posts"]

    * 100

)


year_asset["excluded_percent"] = (

    year_asset["excluded_percent"]
    .round(4)

)


# ============================================================
# SORT SUMMARY
# ============================================================

year_asset = (

    year_asset.sort_values(
        [
            "year",
            "asset",
        ]
    )

    .reset_index(drop=True)

)


# ============================================================
# VALIDATE NUMBER OF YEAR-ASSET GROUPS
# ============================================================

# 5 years x 2 assets = 10 expected groups

expected_year_asset_groups = 10


if len(year_asset) != expected_year_asset_groups:

    raise ValueError(

        "\nUnexpected number of year-asset groups.\n"
        f"Expected: {expected_year_asset_groups}\n"
        f"Found:    {len(year_asset)}"

    )


# ============================================================
# VALIDATE TOTAL INPUT POSTS
# ============================================================

year_asset_input_total = int(
    year_asset["input_posts"].sum()
)


if year_asset_input_total != EXPECTED_ROWS:

    raise ValueError(

        "\nYear-asset input total does not match "
        "the original Reddit dataset.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {year_asset_input_total:,}"

    )


# ============================================================
# VALIDATE TOTAL RETAINED POSTS
# ============================================================

year_asset_retained_total = int(
    year_asset["retained_posts"].sum()
)


if year_asset_retained_total != retained_primary_count:

    raise ValueError(

        "\nYear-asset retained total does not match "
        "the primary retained count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {year_asset_retained_total:,}"

    )


# ============================================================
# VALIDATE TOTAL EXCLUDED POSTS
# ============================================================

year_asset_excluded_total = int(
    year_asset["excluded_posts"].sum()
)


if year_asset_excluded_total != excluded_primary_count:

    raise ValueError(

        "\nYear-asset excluded total does not match "
        "the primary excluded count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {year_asset_excluded_total:,}"

    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\nCleaning summary by year and asset:")


print(
    year_asset.to_string(
        index=False
    )
)


print(
    "\nYear-asset groups: "
    f"{len(year_asset)}"
)


print(
    "Year-asset input total: "
    f"{year_asset_input_total:,}"
)


print(
    "Year-asset retained total: "
    f"{year_asset_retained_total:,}"
)


print(
    "Year-asset excluded total: "
    f"{year_asset_excluded_total:,}"
)


print("\nYear and asset cleaning summary: PASS")
# ============================================================
# SECTION 5C — CLEANING SUMMARY BY SUBREDDIT
# ============================================================


section("CREATING CLEANING SUMMARY BY SUBREDDIT")


# ============================================================
# GROUP BY SUBREDDIT
# ============================================================

subreddit_summary = (

    df.groupby(
        "subreddit",
        observed=True
    )

    .agg(

        input_posts=(
            "post_id",
            "size"
        ),

        retained_posts=(
            "retain_primary",
            "sum"
        ),

        excluded_posts=(
            "exclude_primary",
            "sum"
        ),

        unusable_posts=(
            "flag_unusable_text",
            "sum"
        ),

        crossposts=(
            "flag_crosspost",
            "sum"
        ),

        promotional_posts=(
            "flag_promotional",
            "sum"
        ),

        template_posts=(
            "flag_template",
            "sum"
        ),

        exact_duplicate_posts=(
            "flag_exact_duplicate",
            "sum"
        ),

        link_only_posts=(
            "flag_link_only",
            "sum"
        ),

        repeated_text_5plus_posts=(
            "flag_repeated_text_5plus",
            "sum"
        ),

    )

    .reset_index()

)


# ============================================================
# CONVERT COUNT COLUMNS TO INTEGERS
# ============================================================

subreddit_count_columns = [

    "input_posts",

    "retained_posts",

    "excluded_posts",

    "unusable_posts",

    "crossposts",

    "promotional_posts",

    "template_posts",

    "exact_duplicate_posts",

    "link_only_posts",

    "repeated_text_5plus_posts",

]


for column in subreddit_count_columns:

    subreddit_summary[column] = (
        subreddit_summary[column]
        .astype(int)
    )


# ============================================================
# CALCULATE RETAINED PERCENTAGE
# ============================================================

subreddit_summary["retained_percent"] = (

    subreddit_summary["retained_posts"]

    / subreddit_summary["input_posts"]

    * 100

)


subreddit_summary["retained_percent"] = (

    subreddit_summary["retained_percent"]
    .round(4)

)


# ============================================================
# CALCULATE EXCLUDED PERCENTAGE
# ============================================================

subreddit_summary["excluded_percent"] = (

    subreddit_summary["excluded_posts"]

    / subreddit_summary["input_posts"]

    * 100

)


subreddit_summary["excluded_percent"] = (

    subreddit_summary["excluded_percent"]
    .round(4)

)


# ============================================================
# SORT SUBREDDITS
# ============================================================

subreddit_order = {

    "Bitcoin": 1,

    "BitcoinMarkets": 2,

    "ethereum": 3,

}


subreddit_summary["sort_order"] = (

    subreddit_summary["subreddit"]
    .map(subreddit_order)

)


subreddit_summary = (

    subreddit_summary.sort_values(
        "sort_order"
    )

    .drop(
        columns="sort_order"
    )

    .reset_index(drop=True)

)


# ============================================================
# VALIDATE NUMBER OF SUBREDDITS
# ============================================================

expected_subreddit_groups = 3


if len(subreddit_summary) != expected_subreddit_groups:

    raise ValueError(

        "\nUnexpected number of subreddit groups.\n"
        f"Expected: {expected_subreddit_groups}\n"
        f"Found:    {len(subreddit_summary)}"

    )


# ============================================================
# VALIDATE TOTAL INPUT POSTS
# ============================================================

subreddit_input_total = int(

    subreddit_summary[
        "input_posts"
    ].sum()

)


if subreddit_input_total != EXPECTED_ROWS:

    raise ValueError(

        "\nSubreddit input total does not match "
        "the original dataset.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {subreddit_input_total:,}"

    )


# ============================================================
# VALIDATE RETAINED TOTAL
# ============================================================

subreddit_retained_total = int(

    subreddit_summary[
        "retained_posts"
    ].sum()

)


if subreddit_retained_total != retained_primary_count:

    raise ValueError(

        "\nSubreddit retained total does not match "
        "the primary retained count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {subreddit_retained_total:,}"

    )


# ============================================================
# VALIDATE EXCLUDED TOTAL
# ============================================================

subreddit_excluded_total = int(

    subreddit_summary[
        "excluded_posts"
    ].sum()

)


if subreddit_excluded_total != excluded_primary_count:

    raise ValueError(

        "\nSubreddit excluded total does not match "
        "the primary excluded count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {subreddit_excluded_total:,}"

    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print("\nCleaning summary by subreddit:")


print(
    subreddit_summary.to_string(
        index=False
    )
)


print(
    "\nSubreddit input total: "
    f"{subreddit_input_total:,}"
)


print(
    "Subreddit retained total: "
    f"{subreddit_retained_total:,}"
)


print(
    "Subreddit excluded total: "
    f"{subreddit_excluded_total:,}"
)


print("\nSubreddit cleaning summary: PASS")
# ============================================================
# SECTION 5D — CREATE DAILY BTC/ETH COVERAGE
# ============================================================


section("CREATING DAILY BTC/ETH COVERAGE")


# ============================================================
# CREATE COMPLETE DATE CALENDAR
# ============================================================

# The dissertation period is:
#
# 2021-01-01 through 2025-12-31
#
# We deliberately include every calendar day, even when
# Reddit contains zero posts on that day.

calendar = pd.DataFrame(

    {
        "post_date": pd.date_range(
            start="2021-01-01",
            end="2025-12-31",
            freq="D"
        )
    }

)


# ============================================================
# CREATE ASSET TABLE
# ============================================================

assets = pd.DataFrame(

    {
        "asset": [
            "BTC",
            "ETH",
        ]
    }

)


# ============================================================
# CREATE DATE x ASSET GRID
# ============================================================

calendar["join_key"] = 1

assets["join_key"] = 1


complete_calendar = (

    calendar.merge(
        assets,
        on="join_key",
        how="inner"
    )

    .drop(
        columns="join_key"
    )

)


# ============================================================
# VALIDATE COMPLETE CALENDAR SIZE
# ============================================================

# 2021-01-01 through 2025-12-31 contains 1,826 days.
#
# 1,826 days x 2 assets = 3,652 rows.

expected_calendar_days = 1826

expected_daily_rows = 3652


actual_calendar_days = int(
    calendar["post_date"].nunique()
)


actual_daily_rows = len(
    complete_calendar
)


if actual_calendar_days != expected_calendar_days:

    raise ValueError(

        "\nUnexpected number of calendar days.\n"
        f"Expected: {expected_calendar_days:,}\n"
        f"Found:    {actual_calendar_days:,}"

    )


if actual_daily_rows != expected_daily_rows:

    raise ValueError(

        "\nUnexpected number of date-asset rows.\n"
        f"Expected: {expected_daily_rows:,}\n"
        f"Found:    {actual_daily_rows:,}"

    )


# ============================================================
# CALCULATE OBSERVED DAILY COUNTS
# ============================================================

daily_counts = (

    df.groupby(
        [
            "post_date",
            "asset",
        ],
        observed=True
    )

    .agg(

        raw_post_count=(
            "post_id",
            "size"
        ),

        retained_post_count=(
            "retain_primary",
            "sum"
        ),

        excluded_post_count=(
            "exclude_primary",
            "sum"
        ),

    )

    .reset_index()

)


# ============================================================
# MERGE OBSERVED COUNTS ONTO COMPLETE CALENDAR
# ============================================================

daily_coverage = (

    complete_calendar.merge(

        daily_counts,

        on=[
            "post_date",
            "asset",
        ],

        how="left"

    )

)


# ============================================================
# REPLACE MISSING COUNTS WITH ZERO
# ============================================================

daily_count_columns = [

    "raw_post_count",

    "retained_post_count",

    "excluded_post_count",

]


daily_coverage[daily_count_columns] = (

    daily_coverage[daily_count_columns]

    .fillna(0)

    .astype(int)

)


# ============================================================
# CREATE DAILY COVERAGE INDICATORS
# ============================================================

daily_coverage["has_raw_posts"] = (

    daily_coverage["raw_post_count"]
    .gt(0)

)


daily_coverage["has_retained_posts"] = (

    daily_coverage["retained_post_count"]
    .gt(0)

)


# ============================================================
# ADD YEAR
# ============================================================

daily_coverage["year"] = (

    daily_coverage["post_date"]
    .dt.year

)


# ============================================================
# CHECK DAILY ACCOUNTING
# ============================================================

daily_coverage["count_check"] = (

    daily_coverage["retained_post_count"]

    +

    daily_coverage["excluded_post_count"]

)


invalid_daily_accounting = int(

    (
        daily_coverage["count_check"]
        != daily_coverage["raw_post_count"]
    )
    .sum()

)


if invalid_daily_accounting != 0:

    raise ValueError(

        "\nDaily retained + excluded counts do not equal "
        "raw counts for all date-asset rows.\n"
        f"Problem rows: {invalid_daily_accounting:,}"

    )


daily_coverage = daily_coverage.drop(
    columns="count_check"
)


# ============================================================
# SORT DAILY DATA
# ============================================================

daily_coverage = (

    daily_coverage.sort_values(
        [
            "post_date",
            "asset",
        ]
    )

    .reset_index(drop=True)

)


# ============================================================
# VALIDATE FINAL DAILY DATASET
# ============================================================

if len(daily_coverage) != expected_daily_rows:

    raise ValueError(

        "\nFinal daily coverage dataset has "
        "an unexpected number of rows.\n"
        f"Expected: {expected_daily_rows:,}\n"
        f"Found:    {len(daily_coverage):,}"

    )


if daily_coverage["post_date"].min() != pd.Timestamp("2021-01-01"):

    raise ValueError(
        "\nDaily coverage does not start on 2021-01-01."
    )


if daily_coverage["post_date"].max() != pd.Timestamp("2025-12-31"):

    raise ValueError(
        "\nDaily coverage does not end on 2025-12-31."
    )


# ============================================================
# DISPLAY DAILY COVERAGE DIAGNOSTICS
# ============================================================

print(
    f"\nComplete calendar days: "
    f"{actual_calendar_days:,}"
)

print(
    f"Complete date-asset rows: "
    f"{len(daily_coverage):,}"
)


for asset in ["BTC", "ETH"]:

    asset_daily = daily_coverage.loc[
        daily_coverage["asset"] == asset
    ]


    days_with_raw_posts = int(

        asset_daily[
            "has_raw_posts"
        ].sum()

    )


    days_with_retained_posts = int(

        asset_daily[
            "has_retained_posts"
        ].sum()

    )


    days_without_raw_posts = int(

        (
            ~asset_daily[
                "has_raw_posts"
            ]
        ).sum()

    )


    days_without_retained_posts = int(

        (
            ~asset_daily[
                "has_retained_posts"
            ]
        ).sum()

    )


    print(
        f"\n{asset} daily coverage:"
    )

    print(
        f"Days in calendar: "
        f"{len(asset_daily):,}"
    )

    print(
        f"Days with raw Reddit posts: "
        f"{days_with_raw_posts:,}"
    )

    print(
        f"Days without raw Reddit posts: "
        f"{days_without_raw_posts:,}"
    )

    print(
        f"Days with retained Reddit posts: "
        f"{days_with_retained_posts:,}"
    )

    print(
        f"Days without retained Reddit posts: "
        f"{days_without_retained_posts:,}"
    )


print("\nDaily BTC/ETH coverage: PASS")
# ============================================================
# SECTION 5E — CREATE MANUAL-REVIEW EXAMPLES
# ============================================================


section("CREATING MANUAL-REVIEW EXAMPLES")


# ============================================================
# DEFINE FLAGS TO REVIEW
# ============================================================

# We save examples from each important cleaning category.
#
# This allows us to manually inspect whether the rules are
# behaving sensibly before sentiment scoring.

review_flags = {

    "unusable_text":
        "flag_unusable_text",

    "crosspost":
        "flag_crosspost",

    "promotional":
        "flag_promotional",

    "recurring_template":
        "flag_template",

    "exact_duplicate":
        "flag_exact_duplicate",

    "link_only":
        "flag_link_only",

    "repeated_text_5plus_diagnostic":
        "flag_repeated_text_5plus",

}


# ============================================================
# NUMBER OF EXAMPLES PER CATEGORY
# ============================================================

EXAMPLES_PER_CATEGORY = 25


# ============================================================
# COLUMNS TO KEEP IN REVIEW FILE
# ============================================================

example_columns = [

    "post_id",

    "post_date",

    "subreddit",

    "asset",

    "title_original",

    "body_original",

    "analysis_text",

    "normalized_text_frequency",

    "exclusion_reason",

]


# ============================================================
# COLLECT EXAMPLES
# ============================================================

example_frames = []


for review_category, flag_column in review_flags.items():

    # Select posts carrying this flag.

    flagged_posts = df.loc[
        df[flag_column],
        example_columns
    ].copy()


    # Keep a manageable number for manual inspection.

    sample = flagged_posts.head(
        EXAMPLES_PER_CATEGORY
    ).copy()


    # Add the category that caused the post to appear
    # in the review file.

    sample.insert(
        0,
        "review_category",
        review_category
    )


    # Add total number of posts carrying this flag.

    sample.insert(
        1,
        "total_posts_with_flag",
        len(flagged_posts)
    )


    # Add only non-empty samples.

    if not sample.empty:

        example_frames.append(
            sample
        )


    print(
        f"{review_category}: "
        f"{len(flagged_posts):,} flagged posts; "
        f"{len(sample):,} examples selected"
    )


# ============================================================
# COMBINE EXAMPLES
# ============================================================

if example_frames:

    examples = pd.concat(
        example_frames,
        ignore_index=True
    )


else:

    examples = pd.DataFrame(

        columns=[

            "review_category",

            "total_posts_with_flag",

        ]
        + example_columns

    )


# ============================================================
# ADD A SHORT PREVIEW COLUMN
# ============================================================

# The full analysis_text remains available.
# This preview makes the CSV easier to inspect quickly.

examples["analysis_text_preview"] = (

    examples["analysis_text"]
    .fillna("")
    .astype(str)
    .str.slice(
        0,
        500
    )

)


# ============================================================
# REORDER COLUMNS
# ============================================================

final_example_columns = [

    "review_category",

    "total_posts_with_flag",

    "post_id",

    "post_date",

    "subreddit",

    "asset",

    "title_original",

    "body_original",

    "analysis_text",

    "analysis_text_preview",

    "normalized_text_frequency",

    "exclusion_reason",

]


examples = examples[
    final_example_columns
]


# ============================================================
# VALIDATE REVIEW CATEGORIES
# ============================================================

if not examples.empty:

    unexpected_review_categories = (

        set(
            examples[
                "review_category"
            ].unique()
        )

        -

        set(
            review_flags.keys()
        )

    )


    if unexpected_review_categories:

        raise ValueError(

            "\nUnexpected review categories found:\n"
            f"{sorted(unexpected_review_categories)}"

        )


# ============================================================
# CHECK MAXIMUM NUMBER OF EXAMPLES
# ============================================================

if not examples.empty:

    examples_per_group = (

        examples.groupby(
            "review_category"
        )
        .size()

    )


    too_many_examples = (

        examples_per_group
        > EXAMPLES_PER_CATEGORY

    )


    if too_many_examples.any():

        raise ValueError(

            "\nMore than the permitted number of "
            "manual-review examples was selected "
            "for at least one category."

        )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    f"\nTotal manual-review rows created: "
    f"{len(examples):,}"
)


if not examples.empty:

    print(
        "\nExamples by review category:"
    )


    print(

        examples[
            "review_category"
        ]
        .value_counts()
        .to_string()

    )


print("\nManual-review examples: PASS")
# ============================================================
# SECTION 6A — CREATE PRIMARY SENTIMENT SAMPLE
# ============================================================


section("CREATING PRIMARY SENTIMENT SAMPLE")


# ============================================================
# SELECT ONLY RETAINED POSTS
# ============================================================

# Only posts that passed all PRIMARY cleaning rules enter
# the dataset that will later be used for sentiment scoring.
#
# IMPORTANT:
# No sentiment is calculated in this script.

primary = (

    df.loc[
        df["retain_primary"]
    ]

    .copy()

)


# ============================================================
# DEFINE COLUMNS FOR PRIMARY SENTIMENT SAMPLE
# ============================================================

# Keep only variables needed for:
#
# - post identification
# - date/time
# - subreddit
# - BTC/ETH asset
# - cleaned text
# - basic Reddit engagement variables
#
# We deliberately DO NOT include:
#
# - url
# - permalink
# - crosspost_parent_id
#
# These are not required for sentiment scoring.

primary_columns = [

    "post_id",

    "created_at",

    "post_date",

    "year",

    "subreddit",

    "asset",

    "title_clean",

    "body_clean",

    "analysis_text",

    "score",

    "upvote_ratio",

    "num_comments",

]


# ============================================================
# CHECK THAT ALL REQUIRED PRIMARY COLUMNS EXIST
# ============================================================

missing_primary_columns = [

    column

    for column in primary_columns

    if column not in primary.columns

]


if missing_primary_columns:

    raise ValueError(

        "\nThe primary sentiment sample is missing "
        "required columns:\n"
        f"{missing_primary_columns}"

    )


# ============================================================
# KEEP ONLY REQUIRED PRIMARY COLUMNS
# ============================================================

primary = (

    primary[
        primary_columns
    ]

    .copy()

)


# ============================================================
# SORT PRIMARY SAMPLE CHRONOLOGICALLY
# ============================================================

primary = (

    primary.sort_values(

        by=[
            "post_date",
            "created_at",
            "post_id",
        ],

        kind="stable"

    )

    .reset_index(drop=True)

)


# ============================================================
# VALIDATE PRIMARY ROW COUNT
# ============================================================

primary_row_count = len(
    primary
)


if primary_row_count != retained_primary_count:

    raise ValueError(

        "\nPrimary sentiment sample row count "
        "does not match retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {primary_row_count:,}"

    )


# ============================================================
# CHECK FOR MISSING POST IDs
# ============================================================

primary_missing_ids = int(

    primary["post_id"]
    .isna()
    .sum()

)


if primary_missing_ids != 0:

    raise ValueError(

        "\nMissing post IDs found in the "
        "primary sentiment sample.\n"
        f"Missing IDs: {primary_missing_ids:,}"

    )


# ============================================================
# CHECK FOR DUPLICATE POST IDs
# ============================================================

primary_duplicate_ids = int(

    primary["post_id"]
    .duplicated()
    .sum()

)


if primary_duplicate_ids != 0:

    raise ValueError(

        "\nDuplicate post IDs found in the "
        "primary sentiment sample.\n"
        f"Duplicate IDs: {primary_duplicate_ids:,}"

    )


# ============================================================
# CHECK FOR EMPTY ANALYSIS TEXT
# ============================================================

primary_empty_text = int(

    primary["analysis_text"]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()

)


if primary_empty_text != 0:

    raise ValueError(

        "\nEmpty analysis_text values remain in "
        "the primary sentiment sample.\n"
        f"Empty rows: {primary_empty_text:,}"

    )


# ============================================================
# CHECK FOR TOO-SHORT ANALYSIS TEXT
# ============================================================

# Every retained post should contain at least
# 3 alphabetic characters.

primary_alpha_counts = (

    primary["analysis_text"]
    .map(alpha_count)

)


primary_too_short = int(

    primary_alpha_counts
    .lt(3)
    .sum()

)


if primary_too_short != 0:

    raise ValueError(

        "\nPosts with fewer than 3 alphabetic characters "
        "remain in the primary sentiment sample.\n"
        f"Problem rows: {primary_too_short:,}"

    )


# ============================================================
# CHECK FOR DELETED / REMOVED PLACEHOLDERS
# ============================================================

# These should already have been excluded by the
# unusable-text rule.

primary_deleted_titles = int(

    primary["title_clean"]
    .map(is_deleted_removed)
    .sum()

)


primary_deleted_bodies = int(

    primary["body_clean"]
    .map(is_deleted_removed)
    .sum()

)


if (
    primary_deleted_titles != 0
    or primary_deleted_bodies != 0
):

    raise ValueError(

        "\nDeleted/removed placeholders remain in "
        "the primary sentiment sample.\n"
        f"Deleted/removed titles: {primary_deleted_titles:,}\n"
        f"Deleted/removed bodies: {primary_deleted_bodies:,}"

    )


# ============================================================
# CHECK EXPECTED ASSETS
# ============================================================

primary_assets = set(

    primary["asset"]
    .dropna()
    .unique()

)


expected_primary_assets = {
    "BTC",
    "ETH",
}


if primary_assets != expected_primary_assets:

    raise ValueError(

        "\nUnexpected assets in the primary "
        "sentiment sample.\n"
        f"Expected: {sorted(expected_primary_assets)}\n"
        f"Found:    {sorted(primary_assets)}"

    )


# ============================================================
# CHECK EXPECTED SUBREDDITS
# ============================================================

primary_subreddits = set(

    primary["subreddit"]
    .dropna()
    .unique()

)


unexpected_primary_subreddits = (

    primary_subreddits

    -

    set(
        EXPECTED_SUBREDDIT_COUNTS.keys()
    )

)


if unexpected_primary_subreddits:

    raise ValueError(

        "\nUnexpected subreddits found in the "
        "primary sentiment sample:\n"
        f"{sorted(unexpected_primary_subreddits)}"

    )


# ============================================================
# CHECK DATE RANGE
# ============================================================

primary_earliest_date = (

    primary["post_date"]
    .min()

)


primary_latest_date = (

    primary["post_date"]
    .max()

)


if primary_earliest_date < pd.Timestamp("2021-01-01"):

    raise ValueError(

        "\nPrimary sentiment sample contains "
        "a date before 2021-01-01."

    )


if primary_latest_date > pd.Timestamp("2025-12-31"):

    raise ValueError(

        "\nPrimary sentiment sample contains "
        "a date after 2025-12-31."

    )


# ============================================================
# DATA-MINIMIZATION CHECK
# ============================================================

# URL/permalink/crosspost identifiers must not enter
# the primary sentiment-scoring dataset.

forbidden_primary_columns = {

    "url",

    "permalink",

    "crosspost_parent_id",

}


present_forbidden_columns = (

    forbidden_primary_columns

    &

    set(
        primary.columns
    )

)


if present_forbidden_columns:

    raise ValueError(

        "\nData-minimization check failed.\n"
        "The primary sentiment sample contains:\n"
        f"{sorted(present_forbidden_columns)}"

    )


# ============================================================
# CHECK THAT PRIMARY SAMPLE HAS EXACTLY EXPECTED COLUMNS
# ============================================================

if list(primary.columns) != primary_columns:

    raise ValueError(

        "\nPrimary sentiment sample columns "
        "do not match the expected column list."

    )


# ============================================================
# DISPLAY PRIMARY SAMPLE COUNTS BY ASSET
# ============================================================

primary_asset_counts = (

    primary["asset"]
    .value_counts()
    .sort_index()

)


print("\nPrimary sentiment sample by asset:")


for asset in ["BTC", "ETH"]:

    asset_count = int(

        primary_asset_counts.get(
            asset,
            0
        )

    )

    print(
        f"{asset}: "
        f"{asset_count:,}"
    )


# ============================================================
# DISPLAY PRIMARY SAMPLE COUNTS BY SUBREDDIT
# ============================================================

primary_subreddit_counts = (

    primary["subreddit"]
    .value_counts()

)


print("\nPrimary sentiment sample by subreddit:")


for subreddit in EXPECTED_SUBREDDIT_COUNTS:

    subreddit_count = int(

        primary_subreddit_counts.get(
            subreddit,
            0
        )

    )

    print(
        f"{subreddit}: "
        f"{subreddit_count:,}"
    )


# ============================================================
# DISPLAY PRIMARY SAMPLE COUNTS BY YEAR
# ============================================================

primary_year_counts = (

    primary["year"]
    .value_counts()
    .sort_index()

)


print("\nPrimary sentiment sample by year:")


for year in sorted(
    EXPECTED_YEAR_COUNTS.keys()
):

    year_count = int(

        primary_year_counts.get(
            year,
            0
        )

    )

    print(
        f"{year}: "
        f"{year_count:,}"
    )


# ============================================================
# FINAL SECTION 6A SUMMARY
# ============================================================

print(
    f"\nPrimary sentiment sample rows: "
    f"{primary_row_count:,}"
)


print(
    f"Missing post IDs: "
    f"{primary_missing_ids:,}"
)


print(
    f"Duplicate post IDs: "
    f"{primary_duplicate_ids:,}"
)


print(
    f"Empty analysis_text rows: "
    f"{primary_empty_text:,}"
)


print(
    f"Posts with fewer than 3 alphabetic characters: "
    f"{primary_too_short:,}"
)


print(
    "URL/permalink/crosspost-ID minimization check: PASS"
)


print("\nPrimary sentiment sample creation: PASS")
# ============================================================
# SECTION 6B — PREPARE FULL CLEANED / AUDITED DATASET
# ============================================================


section("PREPARING FULL CLEANED AND AUDITED DATASET")


# ============================================================
# CREATE A SEPARATE OUTPUT COPY
# ============================================================

# df remains the working dataframe.
#
# clean_full will become:
#
# reddit_posts_cleaned_full.csv
#
# It contains ALL original posts, including posts that were
# excluded from the primary sentiment sample.
#
# This is important because the cleaning decisions remain
# transparent and auditable.

clean_full = df.copy()


# ============================================================
# REMOVE UNNECESSARY LINK / CROSSPOST IDENTIFIER COLUMNS
# ============================================================

# The raw source file remains unchanged.
#
# These columns are not required in the derived cleaned file
# because we have already created:
#
# flag_crosspost
# flag_link_only
#
# Removing the original URL/permalink/crosspost fields from
# the derived dataset improves data minimization.

columns_to_remove_from_clean_full = [

    "url",

    "permalink",

    "crosspost_parent_id",

]


existing_columns_to_remove = [

    column

    for column in columns_to_remove_from_clean_full

    if column in clean_full.columns

]


if existing_columns_to_remove:

    clean_full = clean_full.drop(
        columns=existing_columns_to_remove
    )


# ============================================================
# DEFINE FINAL CLEANED DATASET COLUMNS
# ============================================================

clean_full_columns = [

    # --------------------------------
    # Post identification / timing
    # --------------------------------

    "post_id",

    "created_at",

    "post_date",

    "year",


    # --------------------------------
    # Source / asset
    # --------------------------------

    "subreddit",

    "asset",


    # --------------------------------
    # Original text
    # --------------------------------

    "title_original",

    "body_original",


    # --------------------------------
    # Cleaned text
    # --------------------------------

    "title_clean",

    "body_clean",

    "analysis_text",


    # --------------------------------
    # Reddit engagement variables
    # --------------------------------

    "score",

    "upvote_ratio",

    "num_comments",


    # --------------------------------
    # Text diagnostics
    # --------------------------------

    "analysis_alpha_count",

    "normalized_text_frequency",


    # --------------------------------
    # Component cleaning flags
    # --------------------------------

    "flag_empty_text",

    "flag_deleted_removed",

    "flag_too_short",


    # --------------------------------
    # Primary cleaning-rule flags
    # --------------------------------

    "flag_unusable_text",

    "flag_crosspost",

    "flag_promotional",

    "flag_template",

    "flag_exact_duplicate",

    "flag_link_only",


    # --------------------------------
    # Diagnostic-only flag
    # --------------------------------

    "flag_repeated_text_5plus",


    # --------------------------------
    # Final primary decision
    # --------------------------------

    "exclude_primary",

    "retain_primary",

    "exclusion_reason",

]


# ============================================================
# CHECK ALL EXPECTED COLUMNS EXIST
# ============================================================

missing_clean_full_columns = [

    column

    for column in clean_full_columns

    if column not in clean_full.columns

]


if missing_clean_full_columns:

    raise ValueError(

        "\nThe full cleaned dataset is missing "
        "required columns:\n"
        f"{missing_clean_full_columns}"

    )


# ============================================================
# KEEP ONLY THE FINAL CLEANED COLUMNS
# ============================================================

clean_full = (

    clean_full[
        clean_full_columns
    ]

    .copy()

)


# ============================================================
# SORT FULL CLEANED DATASET CHRONOLOGICALLY
# ============================================================

clean_full = (

    clean_full.sort_values(

        by=[
            "post_date",
            "created_at",
            "post_id",
        ],

        kind="stable"

    )

    .reset_index(drop=True)

)


# ============================================================
# VALIDATE ROW COUNT
# ============================================================

clean_full_row_count = len(
    clean_full
)


if clean_full_row_count != EXPECTED_ROWS:

    raise ValueError(

        "\nFull cleaned dataset has an unexpected "
        "number of rows.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {clean_full_row_count:,}"

    )


# ============================================================
# VALIDATE POST IDs
# ============================================================

clean_full_missing_ids = int(

    clean_full["post_id"]
    .isna()
    .sum()

)


if clean_full_missing_ids != 0:

    raise ValueError(

        "\nMissing post IDs found in the "
        "full cleaned dataset.\n"
        f"Missing IDs: {clean_full_missing_ids:,}"

    )


clean_full_duplicate_ids = int(

    clean_full["post_id"]
    .duplicated()
    .sum()

)


if clean_full_duplicate_ids != 0:

    raise ValueError(

        "\nDuplicate post IDs found in the "
        "full cleaned dataset.\n"
        f"Duplicate IDs: {clean_full_duplicate_ids:,}"

    )


# ============================================================
# VALIDATE RETAINED / EXCLUDED ACCOUNTING
# ============================================================

clean_full_retained = int(

    clean_full["retain_primary"]
    .sum()

)


clean_full_excluded = int(

    clean_full["exclude_primary"]
    .sum()

)


if clean_full_retained != retained_primary_count:

    raise ValueError(

        "\nFull cleaned retained count does not match "
        "the previously calculated retained count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {clean_full_retained:,}"

    )


if clean_full_excluded != excluded_primary_count:

    raise ValueError(

        "\nFull cleaned excluded count does not match "
        "the previously calculated excluded count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {clean_full_excluded:,}"

    )


if (
    clean_full_retained
    + clean_full_excluded
    != EXPECTED_ROWS
):

    raise ValueError(

        "\nFull cleaned retained + excluded counts "
        "do not equal the original dataset size."

    )


# ============================================================
# CHECK FINAL EXCLUSION DECISIONS
# ============================================================

invalid_decision_rows = int(

    (
        clean_full["retain_primary"]
        == clean_full["exclude_primary"]
    )
    .sum()

)


if invalid_decision_rows != 0:

    raise ValueError(

        "\nInvalid primary cleaning decisions found.\n"
        "Each post must be either retained or excluded, "
        "but not both.\n"
        f"Problem rows: {invalid_decision_rows:,}"

    )


# ============================================================
# CHECK EXCLUSION REASONS
# ============================================================

missing_exclusion_reasons = int(

    clean_full["exclusion_reason"]
    .isna()
    .sum()

)


if missing_exclusion_reasons != 0:

    raise ValueError(

        "\nMissing exclusion_reason values found in "
        "the full cleaned dataset.\n"
        f"Missing reasons: {missing_exclusion_reasons:,}"

    )


# ============================================================
# CHECK RETAINED POSTS SAY "retained"
# ============================================================

invalid_retained_reasons = int(

    (
        clean_full["retain_primary"]

        &

        clean_full["exclusion_reason"]
        .ne("retained")
    )
    .sum()

)


if invalid_retained_reasons != 0:

    raise ValueError(

        "\nSome retained posts have an unexpected "
        "exclusion_reason.\n"
        f"Problem rows: {invalid_retained_reasons:,}"

    )


# ============================================================
# CHECK EXCLUDED POSTS HAVE AN EXCLUSION REASON
# ============================================================

invalid_excluded_reasons = int(

    (
        clean_full["exclude_primary"]

        &

        clean_full["exclusion_reason"]
        .eq("retained")
    )
    .sum()

)


if invalid_excluded_reasons != 0:

    raise ValueError(

        "\nSome excluded posts are incorrectly labelled "
        "as retained in exclusion_reason.\n"
        f"Problem rows: {invalid_excluded_reasons:,}"

    )


# ============================================================
# DATA-MINIMIZATION CHECK
# ============================================================

forbidden_clean_full_columns = {

    "url",

    "permalink",

    "crosspost_parent_id",

}


present_forbidden_clean_columns = (

    forbidden_clean_full_columns

    &

    set(
        clean_full.columns
    )

)


if present_forbidden_clean_columns:

    raise ValueError(

        "\nData-minimization check failed for "
        "the full cleaned dataset.\n"
        "Unexpected columns:\n"
        f"{sorted(present_forbidden_clean_columns)}"

    )


# ============================================================
# CHECK NORMALIZED TEXT IS NOT BEING SAVED
# ============================================================

# normalized_text was required internally for duplicate
# detection, but we do not need to save the complete
# normalized text in the final derived dataset.
#
# normalized_text_frequency is retained because it is useful
# for auditing repeated content.

if "normalized_text" in clean_full.columns:

    raise ValueError(

        "\nnormalized_text should not be present in "
        "the final clean_full dataset."

    )


# ============================================================
# DISPLAY SUMMARY
# ============================================================

print(
    f"\nFull cleaned dataset rows: "
    f"{clean_full_row_count:,}"
)


print(
    f"Retained posts: "
    f"{clean_full_retained:,}"
)


print(
    f"Excluded posts: "
    f"{clean_full_excluded:,}"
)


print(
    f"Missing post IDs: "
    f"{clean_full_missing_ids:,}"
)


print(
    f"Duplicate post IDs: "
    f"{clean_full_duplicate_ids:,}"
)


print(
    "URL/permalink/crosspost-ID minimization check: PASS"
)


print(
    "Normalized matching text excluded from saved "
    "cleaned dataset: PASS"
)


print("\nFull cleaned/audited dataset preparation: PASS")
# ============================================================
# SECTION 7A — SAVE ALL STAGE 02 OUTPUT FILES
# ============================================================


section("SAVING STAGE 02 CLEANING OUTPUTS")


# ============================================================
# PRE-SAVE VALIDATION
# ============================================================

# This section saves the seven Stage 02 outputs:
#
# 1. Full cleaned/audited Reddit dataset
# 2. Primary sentiment sample
# 3. Overall cleaning audit
# 4. Cleaning summary by year and asset
# 5. Cleaning summary by subreddit
# 6. Daily BTC/ETH coverage
# 7. Manual-review exclusion examples
#
# Before saving, perform final consistency checks.


# ============================================================
# CHECK FULL CLEANED DATASET
# ============================================================

if len(clean_full) != EXPECTED_ROWS:

    raise ValueError(
        "\nCannot save clean_full.\n"
        f"Expected rows: {EXPECTED_ROWS:,}\n"
        f"Found rows:    {len(clean_full):,}"
    )


# ============================================================
# CHECK PRIMARY SENTIMENT SAMPLE
# ============================================================

if len(primary) != retained_primary_count:

    raise ValueError(
        "\nCannot save primary sentiment sample.\n"
        f"Expected rows: {retained_primary_count:,}\n"
        f"Found rows:    {len(primary):,}"
    )


# ============================================================
# CHECK AUDIT TABLE
# ============================================================

if audit.empty:

    raise ValueError(
        "\nCannot save cleaning audit because "
        "the audit table is empty."
    )


required_audit_measures = {

    "Input posts",

    "Excluded from primary sample",

    "Retained for primary sample",

}


actual_audit_measures = set(
    audit["measure"]
)


missing_audit_measures = (

    required_audit_measures
    -
    actual_audit_measures

)


if missing_audit_measures:

    raise ValueError(
        "\nCleaning audit is missing required measures:\n"
        f"{sorted(missing_audit_measures)}"
    )


# ============================================================
# CHECK YEAR / ASSET SUMMARY
# ============================================================

if len(year_asset) != 10:

    raise ValueError(
        "\nYear/asset summary should contain exactly "
        "10 rows: 5 years x 2 assets.\n"
        f"Found: {len(year_asset):,}"
    )


if int(
    year_asset["input_posts"].sum()
) != EXPECTED_ROWS:

    raise ValueError(
        "\nYear/asset summary input total does not "
        "match the original dataset."
    )


if int(
    year_asset["retained_posts"].sum()
) != retained_primary_count:

    raise ValueError(
        "\nYear/asset retained total does not match "
        "retained_primary_count."
    )


if int(
    year_asset["excluded_posts"].sum()
) != excluded_primary_count:

    raise ValueError(
        "\nYear/asset excluded total does not match "
        "excluded_primary_count."
    )


# ============================================================
# CHECK SUBREDDIT SUMMARY
# ============================================================

if len(subreddit_summary) != 3:

    raise ValueError(
        "\nSubreddit summary should contain exactly "
        "3 rows.\n"
        f"Found: {len(subreddit_summary):,}"
    )


if int(
    subreddit_summary["input_posts"].sum()
) != EXPECTED_ROWS:

    raise ValueError(
        "\nSubreddit summary input total does not "
        "match the original dataset."
    )


if int(
    subreddit_summary["retained_posts"].sum()
) != retained_primary_count:

    raise ValueError(
        "\nSubreddit retained total does not match "
        "retained_primary_count."
    )


if int(
    subreddit_summary["excluded_posts"].sum()
) != excluded_primary_count:

    raise ValueError(
        "\nSubreddit excluded total does not match "
        "excluded_primary_count."
    )


# ============================================================
# CHECK DAILY COVERAGE TABLE
# ============================================================

EXPECTED_CALENDAR_DAYS = 1826

EXPECTED_DAILY_ROWS = (
    EXPECTED_CALENDAR_DAYS * 2
)


if len(daily_coverage) != EXPECTED_DAILY_ROWS:

    raise ValueError(
        "\nDaily coverage table has an unexpected "
        "number of rows.\n"
        f"Expected: {EXPECTED_DAILY_ROWS:,}\n"
        f"Found:    {len(daily_coverage):,}"
    )


if int(
    daily_coverage["raw_post_count"].sum()
) != EXPECTED_ROWS:

    raise ValueError(
        "\nDaily coverage raw-post total does not "
        "match the original dataset."
    )


if int(
    daily_coverage["retained_post_count"].sum()
) != retained_primary_count:

    raise ValueError(
        "\nDaily coverage retained-post total does not "
        "match retained_primary_count."
    )


if int(
    daily_coverage["excluded_post_count"].sum()
) != excluded_primary_count:

    raise ValueError(
        "\nDaily coverage excluded-post total does not "
        "match excluded_primary_count."
    )


# ============================================================
# CHECK MANUAL-REVIEW EXAMPLES
# ============================================================

required_example_columns = {

    "review_category",

    "total_posts_with_flag",

    "post_id",

    "post_date",

    "subreddit",

    "asset",

    "title_original",

    "body_original",

    "analysis_text",

    "analysis_text_preview",

    "normalized_text_frequency",

    "exclusion_reason",

}


missing_example_columns = (

    required_example_columns
    -
    set(examples.columns)

)


if missing_example_columns:

    raise ValueError(
        "\nManual-review examples are missing "
        "required columns:\n"
        f"{sorted(missing_example_columns)}"
    )


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN RAW LINK FIELDS
# ============================================================

forbidden_primary_save_columns = {

    "url",

    "permalink",

    "crosspost_parent_id",

}


unexpected_primary_save_columns = (

    forbidden_primary_save_columns
    &
    set(primary.columns)

)


if unexpected_primary_save_columns:

    raise ValueError(
        "\nPrimary sentiment sample contains fields "
        "that should not be saved:\n"
        f"{sorted(unexpected_primary_save_columns)}"
    )


# ============================================================
# CHECK CLEAN FULL DOES NOT CONTAIN RAW LINK FIELDS
# ============================================================

forbidden_clean_save_columns = {

    "url",

    "permalink",

    "crosspost_parent_id",

    "normalized_text",

}


unexpected_clean_save_columns = (

    forbidden_clean_save_columns
    &
    set(clean_full.columns)

)


if unexpected_clean_save_columns:

    raise ValueError(
        "\nFull cleaned dataset contains fields "
        "that should not be saved:\n"
        f"{sorted(unexpected_clean_save_columns)}"
    )


# ============================================================
# PRINT PRE-SAVE STATUS
# ============================================================

print(
    "\nAll pre-save validation checks: PASS"
)


# ============================================================
# SAVE FULL CLEANED / AUDITED DATASET
# ============================================================

clean_full.to_csv(
    CLEAN_FULL_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE PRIMARY SENTIMENT SAMPLE
# ============================================================

primary.to_csv(
    PRIMARY_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE OVERALL CLEANING AUDIT
# ============================================================

audit.to_csv(
    AUDIT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE YEAR / ASSET CLEANING SUMMARY
# ============================================================

year_asset.to_csv(
    YEAR_ASSET_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE SUBREDDIT CLEANING SUMMARY
# ============================================================

subreddit_summary.to_csv(
    SUBREDDIT_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE DAILY COVERAGE
# ============================================================

daily_coverage.to_csv(
    DAILY_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# SAVE MANUAL-REVIEW EXAMPLES
# ============================================================

examples.to_csv(
    EXAMPLES_FILE,
    index=False,
    encoding="utf-8"
)


# ============================================================
# VERIFY THAT ALL SEVEN FILES NOW EXIST
# ============================================================

output_files = {

    "Full cleaned/audited dataset":
        CLEAN_FULL_FILE,

    "Primary sentiment sample":
        PRIMARY_FILE,

    "Overall cleaning audit":
        AUDIT_FILE,

    "Year/asset cleaning summary":
        YEAR_ASSET_FILE,

    "Subreddit cleaning summary":
        SUBREDDIT_FILE,

    "Daily BTC/ETH coverage":
        DAILY_FILE,

    "Manual-review examples":
        EXAMPLES_FILE,

}


missing_output_files = [

    str(file_path)

    for file_path in output_files.values()

    if not file_path.exists()

]


if missing_output_files:

    raise FileNotFoundError(
        "\nOne or more Stage 02 output files "
        "were not created:\n"
        + "\n".join(missing_output_files)
    )


# ============================================================
# CHECK THAT SAVED FILES ARE NOT ZERO BYTES
# ============================================================

empty_output_files = [

    str(file_path)

    for file_path in output_files.values()

    if file_path.stat().st_size == 0

]


if empty_output_files:

    raise ValueError(
        "\nOne or more saved output files are empty:\n"
        + "\n".join(empty_output_files)
    )


# ============================================================
# DISPLAY SAVED FILES
# ============================================================

print(
    "\nStage 02 output files successfully written:"
)


for label, file_path in output_files.items():

    file_size_mb = (
        file_path.stat().st_size
        /
        (1024 ** 2)
    )

    print(
        f"\n{label}:"
    )

    print(
        f"  {file_path}"
    )

    print(
        f"  Size: {file_size_mb:.2f} MB"
    )


# ============================================================
# DISPLAY SAVED ROW COUNTS
# ============================================================

print(
    "\nSaved dataset row counts:"
)


print(
    f"  Full cleaned/audited dataset: "
    f"{len(clean_full):,}"
)


print(
    f"  Primary sentiment sample: "
    f"{len(primary):,}"
)


print(
    f"  Overall audit rows: "
    f"{len(audit):,}"
)


print(
    f"  Year/asset summary rows: "
    f"{len(year_asset):,}"
)


print(
    f"  Subreddit summary rows: "
    f"{len(subreddit_summary):,}"
)


print(
    f"  Daily coverage rows: "
    f"{len(daily_coverage):,}"
)


print(
    f"  Manual-review example rows: "
    f"{len(examples):,}"
)


# ============================================================
# IMPORTANT STAGE BOUNDARY
# ============================================================

# Stage 02 stops at cleaning/filtering/auditing.
#
# We deliberately DO NOT:
#
# - calculate sentiment
# - aggregate daily sentiment
# - download cryptocurrency prices
# - calculate returns
# - merge Reddit data with market data
# - run regressions
# - run forecasting models
#
# Those steps belong to later stages and should only begin
# after the cleaning outputs have been inspected.


print(
    "\nNo sentiment or market analysis has been performed."
)


print("\nStage 02 output saving: PASS")


# ============================================================
# END OF SECTION 7A
# ============================================================
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 1 — RELOAD SAVED OUTPUT FILES
# ============================================================


section("FINAL RELOAD AND INTEGRITY VALIDATION")


# ============================================================
# RELOAD ALL SEVEN STAGE 02 OUTPUT FILES
# ============================================================

# We deliberately reload the CSV files from disk.
#
# This means the final validation checks the files that were
# actually saved, rather than only checking the DataFrames
# currently held in memory.


check_clean = pd.read_csv(
    CLEAN_FULL_FILE,
    low_memory=False
)


check_primary = pd.read_csv(
    PRIMARY_FILE,
    low_memory=False
)


check_audit = pd.read_csv(
    AUDIT_FILE,
    low_memory=False
)


check_year_asset = pd.read_csv(
    YEAR_ASSET_FILE,
    low_memory=False
)


check_subreddit = pd.read_csv(
    SUBREDDIT_FILE,
    low_memory=False
)


check_daily = pd.read_csv(
    DAILY_FILE,
    low_memory=False
)


check_examples = pd.read_csv(
    EXAMPLES_FILE,
    low_memory=False
)


# ============================================================
# CONFIRM RELOAD
# ============================================================

print(
    "\nAll seven Stage 02 output CSV files "
    "reloaded successfully."
)


print(
    f"\nFull cleaned rows: "
    f"{len(check_clean):,}"
)


print(
    f"Primary sentiment sample rows: "
    f"{len(check_primary):,}"
)


print(
    f"Audit rows: "
    f"{len(check_audit):,}"
)


print(
    f"Year/asset summary rows: "
    f"{len(check_year_asset):,}"
)


print(
    f"Subreddit summary rows: "
    f"{len(check_subreddit):,}"
)


print(
    f"Daily coverage rows: "
    f"{len(check_daily):,}"
)


print(
    f"Manual-review example rows: "
    f"{len(check_examples):,}"
)


print("\nSection 7B Part 1 — file reload: PASS")
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 2 — ROW COUNTS AND POST IDs
# ============================================================


# ============================================================
# VALIDATE FULL CLEANED DATASET ROW COUNT
# ============================================================

if len(check_clean) != EXPECTED_ROWS:

    raise ValueError(

        "\nReloaded full cleaned dataset has an "
        "unexpected row count.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {len(check_clean):,}"

    )


# ============================================================
# VALIDATE PRIMARY SAMPLE ROW COUNT
# ============================================================

if len(check_primary) != retained_primary_count:

    raise ValueError(

        "\nReloaded primary sentiment sample has an "
        "unexpected row count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {len(check_primary):,}"

    )


# ============================================================
# VALIDATE YEAR / ASSET SUMMARY ROW COUNT
# ============================================================

if len(check_year_asset) != 10:

    raise ValueError(

        "\nReloaded year/asset summary should contain "
        "exactly 10 rows.\n"
        f"Expected: 10\n"
        f"Found:    {len(check_year_asset):,}"

    )


# ============================================================
# VALIDATE SUBREDDIT SUMMARY ROW COUNT
# ============================================================

if len(check_subreddit) != 3:

    raise ValueError(

        "\nReloaded subreddit summary should contain "
        "exactly 3 rows.\n"
        f"Expected: 3\n"
        f"Found:    {len(check_subreddit):,}"

    )


# ============================================================
# VALIDATE DAILY COVERAGE ROW COUNT
# ============================================================

# 2021-01-01 through 2025-12-31 contains 1,826 days.
#
# We require one row per day for each of:
#
# BTC
# ETH
#
# Therefore:
#
# 1,826 x 2 = 3,652 rows.

EXPECTED_DAILY_ROWS_FINAL = 3652


if len(check_daily) != EXPECTED_DAILY_ROWS_FINAL:

    raise ValueError(

        "\nReloaded daily coverage table has an "
        "unexpected row count.\n"
        f"Expected: {EXPECTED_DAILY_ROWS_FINAL:,}\n"
        f"Found:    {len(check_daily):,}"

    )


print(
    "\nReloaded output row-count checks: PASS"
)


# ============================================================
# CHECK MISSING POST IDs — FULL CLEANED DATASET
# ============================================================

clean_missing_ids_final = int(

    check_clean["post_id"]
    .isna()
    .sum()

)


if clean_missing_ids_final != 0:

    raise ValueError(

        "\nReloaded full cleaned dataset contains "
        "missing post IDs.\n"
        f"Missing IDs: {clean_missing_ids_final:,}"

    )


# ============================================================
# CHECK DUPLICATE POST IDs — FULL CLEANED DATASET
# ============================================================

clean_duplicate_ids_final = int(

    check_clean["post_id"]
    .duplicated()
    .sum()

)


if clean_duplicate_ids_final != 0:

    raise ValueError(

        "\nReloaded full cleaned dataset contains "
        "duplicate post IDs.\n"
        f"Duplicate IDs: {clean_duplicate_ids_final:,}"

    )


# ============================================================
# CHECK MISSING POST IDs — PRIMARY SAMPLE
# ============================================================

primary_missing_ids_final = int(

    check_primary["post_id"]
    .isna()
    .sum()

)


if primary_missing_ids_final != 0:

    raise ValueError(

        "\nReloaded primary sentiment sample contains "
        "missing post IDs.\n"
        f"Missing IDs: {primary_missing_ids_final:,}"

    )


# ============================================================
# CHECK DUPLICATE POST IDs — PRIMARY SAMPLE
# ============================================================

primary_duplicate_ids_final = int(

    check_primary["post_id"]
    .duplicated()
    .sum()

)


if primary_duplicate_ids_final != 0:

    raise ValueError(

        "\nReloaded primary sentiment sample contains "
        "duplicate post IDs.\n"
        f"Duplicate IDs: {primary_duplicate_ids_final:,}"

    )


# ============================================================
# DISPLAY POST-ID RESULTS
# ============================================================

print(
    f"\nFull cleaned missing post IDs: "
    f"{clean_missing_ids_final:,}"
)


print(
    f"Full cleaned duplicate post IDs: "
    f"{clean_duplicate_ids_final:,}"
)


print(
    f"Primary sample missing post IDs: "
    f"{primary_missing_ids_final:,}"
)


print(
    f"Primary sample duplicate post IDs: "
    f"{primary_duplicate_ids_final:,}"
)


print(
    "\nSection 7B Part 2 — row counts and post IDs: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 3 — RETAINED / EXCLUDED ACCOUNTING
# ============================================================


# ============================================================
# ENSURE DECISION COLUMNS ARE BOOLEAN
# ============================================================

# CSV reload normally preserves True/False values correctly.
#
# However, we explicitly handle the possibility that pandas
# reloads them as strings.

decision_columns = [

    "retain_primary",

    "exclude_primary",

]


for column in decision_columns:

    if check_clean[column].dtype != bool:

        converted_column = (

            check_clean[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                }
            )

        )


        if converted_column.isna().any():

            raise ValueError(

                f"\nCould not convert {column} "
                "to boolean after CSV reload."

            )


        check_clean[column] = converted_column


# ============================================================
# CALCULATE SAVED RETAINED / EXCLUDED COUNTS
# ============================================================

saved_retained_count = int(

    check_clean[
        "retain_primary"
    ].sum()

)


saved_excluded_count = int(

    check_clean[
        "exclude_primary"
    ].sum()

)


# ============================================================
# CHECK RETAINED COUNT
# ============================================================

if saved_retained_count != retained_primary_count:

    raise ValueError(

        "\nReloaded retained-post count does not match "
        "the original cleaning decision.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {saved_retained_count:,}"

    )


# ============================================================
# CHECK EXCLUDED COUNT
# ============================================================

if saved_excluded_count != excluded_primary_count:

    raise ValueError(

        "\nReloaded excluded-post count does not match "
        "the original cleaning decision.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {saved_excluded_count:,}"

    )


# ============================================================
# CHECK TOTAL ACCOUNTING
# ============================================================

if (

    saved_retained_count
    +
    saved_excluded_count

    !=

    EXPECTED_ROWS

):

    raise ValueError(

        "\nReloaded retained + excluded counts "
        "do not equal the original input size.\n"
        f"Retained: {saved_retained_count:,}\n"
        f"Excluded: {saved_excluded_count:,}\n"
        f"Total:    "
        f"{saved_retained_count + saved_excluded_count:,}\n"
        f"Expected: {EXPECTED_ROWS:,}"

    )


# ============================================================
# CHECK EACH POST HAS EXACTLY ONE DECISION
# ============================================================

# A post must be:
#
# retained = True, excluded = False
#
# OR
#
# retained = False, excluded = True
#
# It must never be both True or both False.

invalid_decision_mask = (

    check_clean[
        "retain_primary"
    ]

    ==

    check_clean[
        "exclude_primary"
    ]

)


invalid_decision_count = int(

    invalid_decision_mask.sum()

)


if invalid_decision_count != 0:

    raise ValueError(

        "\nInvalid retain/exclude decisions found "
        "after CSV reload.\n"
        f"Problem rows: {invalid_decision_count:,}"

    )


# ============================================================
# CHECK EXCLUSION REASON EXISTS
# ============================================================

missing_reason_count = int(

    check_clean[
        "exclusion_reason"
    ]
    .isna()
    .sum()

)


if missing_reason_count != 0:

    raise ValueError(

        "\nMissing exclusion_reason values found "
        "after CSV reload.\n"
        f"Missing reasons: {missing_reason_count:,}"

    )


# ============================================================
# CHECK RETAINED POSTS HAVE "retained" AS THEIR REASON
# ============================================================

invalid_retained_reason_count = int(

    (

        check_clean[
            "retain_primary"
        ]

        &

        check_clean[
            "exclusion_reason"
        ].ne("retained")

    ).sum()

)


if invalid_retained_reason_count != 0:

    raise ValueError(

        "\nSome retained posts do not have "
        "exclusion_reason='retained'.\n"
        f"Problem rows: "
        f"{invalid_retained_reason_count:,}"

    )


# ============================================================
# CHECK EXCLUDED POSTS ARE NOT LABELLED "retained"
# ============================================================

invalid_excluded_reason_count = int(

    (

        check_clean[
            "exclude_primary"
        ]

        &

        check_clean[
            "exclusion_reason"
        ].eq("retained")

    ).sum()

)


if invalid_excluded_reason_count != 0:

    raise ValueError(

        "\nSome excluded posts are incorrectly labelled "
        "with exclusion_reason='retained'.\n"
        f"Problem rows: "
        f"{invalid_excluded_reason_count:,}"

    )


# ============================================================
# DISPLAY ACCOUNTING RESULTS
# ============================================================

print(
    f"\nReloaded retained posts: "
    f"{saved_retained_count:,}"
)


print(
    f"Reloaded excluded posts: "
    f"{saved_excluded_count:,}"
)


print(
    f"Retained + excluded: "
    f"{saved_retained_count + saved_excluded_count:,}"
)


print(
    f"Invalid retain/exclude decisions: "
    f"{invalid_decision_count:,}"
)


print(
    f"Missing exclusion reasons: "
    f"{missing_reason_count:,}"
)


print(
    "\nSection 7B Part 3 — "
    "retained/excluded accounting: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 4 — PRIMARY SAMPLE VS RETAINED POSTS
# ============================================================


# ============================================================
# CREATE POST-ID SET FOR RETAINED POSTS
# ============================================================

# From the full cleaned/audited dataset, select every post
# that was marked retain_primary = True.

retained_post_ids = set(

    check_clean.loc[

        check_clean[
            "retain_primary"
        ],

        "post_id"

    ]
    .astype(str)

)


# ============================================================
# CREATE POST-ID SET FOR PRIMARY SENTIMENT SAMPLE
# ============================================================

primary_post_ids = set(

    check_primary[
        "post_id"
    ]
    .astype(str)

)


# ============================================================
# CHECK SET SIZES
# ============================================================

if len(retained_post_ids) != saved_retained_count:

    raise ValueError(

        "\nUnique retained post-ID count does not match "
        "saved_retained_count.\n"
        f"Unique retained IDs: "
        f"{len(retained_post_ids):,}\n"
        f"Expected: "
        f"{saved_retained_count:,}"

    )


if len(primary_post_ids) != len(check_primary):

    raise ValueError(

        "\nUnique primary post-ID count does not match "
        "the number of rows in the primary sample.\n"
        f"Unique primary IDs: "
        f"{len(primary_post_ids):,}\n"
        f"Primary rows: "
        f"{len(check_primary):,}"

    )


# ============================================================
# FIND RETAINED POSTS MISSING FROM PRIMARY SAMPLE
# ============================================================

missing_from_primary = (

    retained_post_ids

    -

    primary_post_ids

)


# ============================================================
# FIND UNEXPECTED POSTS IN PRIMARY SAMPLE
# ============================================================

unexpected_in_primary = (

    primary_post_ids

    -

    retained_post_ids

)


# ============================================================
# VALIDATE EXACT MATCH
# ============================================================

if missing_from_primary:

    raise ValueError(

        "\nSome posts marked retain_primary=True "
        "are missing from the saved primary sample.\n"
        f"Missing posts: "
        f"{len(missing_from_primary):,}"

    )


if unexpected_in_primary:

    raise ValueError(

        "\nThe saved primary sample contains posts "
        "that were not marked retain_primary=True.\n"
        f"Unexpected posts: "
        f"{len(unexpected_in_primary):,}"

    )


if primary_post_ids != retained_post_ids:

    raise ValueError(

        "\nPrimary post IDs do not exactly match "
        "the retained post IDs."

    )


# ============================================================
# VERIFY NO EXCLUDED POST ENTERED PRIMARY SAMPLE
# ============================================================

excluded_post_ids = set(

    check_clean.loc[

        check_clean[
            "exclude_primary"
        ],

        "post_id"

    ]
    .astype(str)

)


excluded_ids_in_primary = (

    excluded_post_ids

    &

    primary_post_ids

)


if excluded_ids_in_primary:

    raise ValueError(

        "\nExcluded posts were found in the primary "
        "sentiment sample.\n"
        f"Problem posts: "
        f"{len(excluded_ids_in_primary):,}"

    )


# ============================================================
# CHECK RETAINED AND EXCLUDED SETS DO NOT OVERLAP
# ============================================================

retained_excluded_overlap = (

    retained_post_ids

    &

    excluded_post_ids

)


if retained_excluded_overlap:

    raise ValueError(

        "\nSome post IDs appear in both the retained "
        "and excluded sets.\n"
        f"Overlapping posts: "
        f"{len(retained_excluded_overlap):,}"

    )


# ============================================================
# CHECK RETAINED + EXCLUDED IDS COVER FULL DATASET
# ============================================================

all_clean_post_ids = set(

    check_clean[
        "post_id"
    ]
    .astype(str)

)


decision_post_ids = (

    retained_post_ids

    |

    excluded_post_ids

)


if decision_post_ids != all_clean_post_ids:

    raise ValueError(

        "\nRetained and excluded post-ID sets do not "
        "fully cover the cleaned dataset."

    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nUnique retained post IDs: "
    f"{len(retained_post_ids):,}"
)


print(
    f"Unique primary post IDs: "
    f"{len(primary_post_ids):,}"
)


print(
    f"Retained posts missing from primary: "
    f"{len(missing_from_primary):,}"
)


print(
    f"Unexpected posts in primary: "
    f"{len(unexpected_in_primary):,}"
)


print(
    f"Excluded posts found in primary: "
    f"{len(excluded_ids_in_primary):,}"
)


print(
    f"Retained/excluded ID overlap: "
    f"{len(retained_excluded_overlap):,}"
)


print(
    "\nPrimary sample exactly matches "
    "retain_primary=True posts: PASS"
)


print(
    "\nSection 7B Part 4 — "
    "primary sample integrity: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 5 — TEXT, ASSETS, SUBREDDITS, AND DATES
# ============================================================


# ============================================================
# CHECK PRIMARY ANALYSIS TEXT IS NOT MISSING
# ============================================================

primary_missing_text_final = int(

    check_primary[
        "analysis_text"
    ]
    .isna()
    .sum()

)


if primary_missing_text_final != 0:

    raise ValueError(

        "\nMissing analysis_text values found in "
        "the reloaded primary sentiment sample.\n"
        f"Missing values: "
        f"{primary_missing_text_final:,}"

    )


# ============================================================
# CHECK PRIMARY ANALYSIS TEXT IS NOT EMPTY
# ============================================================

primary_empty_text_final = int(

    check_primary[
        "analysis_text"
    ]
    .fillna("")
    .astype(str)
    .str.strip()
    .eq("")
    .sum()

)


if primary_empty_text_final != 0:

    raise ValueError(

        "\nEmpty analysis_text values found in "
        "the reloaded primary sentiment sample.\n"
        f"Empty rows: "
        f"{primary_empty_text_final:,}"

    )


# ============================================================
# CHECK PRIMARY TEXT HAS AT LEAST 3 ALPHABETIC CHARACTERS
# ============================================================

primary_alpha_counts_final = (

    check_primary[
        "analysis_text"
    ]
    .map(alpha_count)

)


primary_too_short_final = int(

    primary_alpha_counts_final
    .lt(3)
    .sum()

)


if primary_too_short_final != 0:

    raise ValueError(

        "\nPosts with fewer than 3 alphabetic characters "
        "remain in the reloaded primary sample.\n"
        f"Problem rows: "
        f"{primary_too_short_final:,}"

    )


# ============================================================
# CHECK EXPECTED ASSETS
# ============================================================

primary_assets_final = set(

    check_primary[
        "asset"
    ]
    .dropna()
    .unique()

)


expected_assets_final = {

    "BTC",

    "ETH",

}


if primary_assets_final != expected_assets_final:

    raise ValueError(

        "\nUnexpected asset values found in the "
        "reloaded primary sample.\n"
        f"Expected: {sorted(expected_assets_final)}\n"
        f"Found:    {sorted(primary_assets_final)}"

    )


# ============================================================
# CHECK EXPECTED SUBREDDITS
# ============================================================

primary_subreddits_final = set(

    check_primary[
        "subreddit"
    ]
    .dropna()
    .unique()

)


expected_subreddits_final = set(

    EXPECTED_SUBREDDIT_COUNTS.keys()

)


unexpected_subreddits_final = (

    primary_subreddits_final

    -

    expected_subreddits_final

)


if unexpected_subreddits_final:

    raise ValueError(

        "\nUnexpected subreddits found in the "
        "reloaded primary sample:\n"
        f"{sorted(unexpected_subreddits_final)}"

    )


# ============================================================
# CHECK ASSET MAPPING BY SUBREDDIT
# ============================================================

invalid_btc_mapping = int(

    (

        check_primary[
            "subreddit"
        ]
        .isin(
            [
                "Bitcoin",
                "BitcoinMarkets",
            ]
        )

        &

        check_primary[
            "asset"
        ]
        .ne("BTC")

    ).sum()

)


if invalid_btc_mapping != 0:

    raise ValueError(

        "\nBitcoin / BitcoinMarkets posts with an "
        "incorrect asset mapping were found.\n"
        f"Problem rows: "
        f"{invalid_btc_mapping:,}"

    )


invalid_eth_mapping = int(

    (

        check_primary[
            "subreddit"
        ]
        .eq("ethereum")

        &

        check_primary[
            "asset"
        ]
        .ne("ETH")

    ).sum()

)


if invalid_eth_mapping != 0:

    raise ValueError(

        "\nEthereum posts with an incorrect asset "
        "mapping were found.\n"
        f"Problem rows: "
        f"{invalid_eth_mapping:,}"

    )


# ============================================================
# PARSE SAVED POST DATES
# ============================================================

primary_dates_final = pd.to_datetime(

    check_primary[
        "post_date"
    ],

    errors="raise"

)


# ============================================================
# CHECK PRIMARY DATE RANGE
# ============================================================

primary_min_date_final = (

    primary_dates_final
    .min()

)


primary_max_date_final = (

    primary_dates_final
    .max()

)


if (

    primary_min_date_final

    <

    pd.Timestamp("2021-01-01")

):

    raise ValueError(

        "\nPrimary sample contains a date "
        "before 2021-01-01."

    )


if (

    primary_max_date_final

    >

    pd.Timestamp("2025-12-31")

):

    raise ValueError(

        "\nPrimary sample contains a date "
        "after 2025-12-31."

    )


# ============================================================
# CHECK PRIMARY YEARS
# ============================================================

primary_years_final = set(

    primary_dates_final
    .dt.year
    .unique()

)


expected_years_final = {

    2021,

    2022,

    2023,

    2024,

    2025,

}


if primary_years_final != expected_years_final:

    raise ValueError(

        "\nUnexpected year coverage in the "
        "reloaded primary sample.\n"
        f"Expected: {sorted(expected_years_final)}\n"
        f"Found:    {sorted(primary_years_final)}"

    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nMissing primary analysis_text: "
    f"{primary_missing_text_final:,}"
)


print(
    f"Empty primary analysis_text: "
    f"{primary_empty_text_final:,}"
)


print(
    f"Primary posts with <3 alphabetic characters: "
    f"{primary_too_short_final:,}"
)


print(
    f"Primary assets: "
    f"{sorted(primary_assets_final)}"
)


print(
    f"Primary subreddits: "
    f"{sorted(primary_subreddits_final)}"
)


print(
    f"Invalid BTC mappings: "
    f"{invalid_btc_mapping:,}"
)


print(
    f"Invalid ETH mappings: "
    f"{invalid_eth_mapping:,}"
)


print(
    f"Primary earliest date: "
    f"{primary_min_date_final.date()}"
)


print(
    f"Primary latest date: "
    f"{primary_max_date_final.date()}"
)


print(
    "\nSection 7B Part 5 — "
    "text, asset, subreddit, and date checks: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6A — AUDIT TABLE VALIDATION
# ============================================================


# ============================================================
# CHECK AUDIT TABLE IS NOT EMPTY
# ============================================================

if check_audit.empty:

    raise ValueError(

        "\nReloaded cleaning audit is empty."

    )


# ============================================================
# CHECK REQUIRED AUDIT COLUMNS
# ============================================================

required_audit_columns_final = {

    "measure",

    "rule_type",

    "count",

    "percent_of_input",

}


missing_audit_columns_final = (

    required_audit_columns_final

    -

    set(
        check_audit.columns
    )

)


if missing_audit_columns_final:

    raise ValueError(

        "\nReloaded cleaning audit is missing "
        "required columns:\n"
        f"{sorted(missing_audit_columns_final)}"

    )


# ============================================================
# CHECK REQUIRED AUDIT MEASURES
# ============================================================

required_audit_measures_final = {

    "Input posts",

    "Excluded from primary sample",

    "Retained for primary sample",

}


actual_audit_measures_final = set(

    check_audit[
        "measure"
    ]
    .dropna()
    .astype(str)

)


missing_audit_measures_final = (

    required_audit_measures_final

    -

    actual_audit_measures_final

)


if missing_audit_measures_final:

    raise ValueError(

        "\nReloaded cleaning audit is missing "
        "required measures:\n"
        f"{sorted(missing_audit_measures_final)}"

    )


# ============================================================
# EXTRACT INPUT COUNT FROM AUDIT
# ============================================================

audit_input_rows_final = check_audit.loc[

    check_audit[
        "measure"
    ].eq("Input posts"),

    "count"

]


if len(audit_input_rows_final) != 1:

    raise ValueError(

        "\nExpected exactly one 'Input posts' row "
        "in the cleaning audit.\n"
        f"Found: {len(audit_input_rows_final):,}"

    )


audit_input_count_final = int(

    audit_input_rows_final.iloc[0]

)


# ============================================================
# EXTRACT RETAINED COUNT FROM AUDIT
# ============================================================

audit_retained_rows_final = check_audit.loc[

    check_audit[
        "measure"
    ].eq("Retained for primary sample"),

    "count"

]


if len(audit_retained_rows_final) != 1:

    raise ValueError(

        "\nExpected exactly one "
        "'Retained for primary sample' row "
        "in the cleaning audit.\n"
        f"Found: {len(audit_retained_rows_final):,}"

    )


audit_retained_count_final = int(

    audit_retained_rows_final.iloc[0]

)


# ============================================================
# EXTRACT EXCLUDED COUNT FROM AUDIT
# ============================================================

audit_excluded_rows_final = check_audit.loc[

    check_audit[
        "measure"
    ].eq("Excluded from primary sample"),

    "count"

]


if len(audit_excluded_rows_final) != 1:

    raise ValueError(

        "\nExpected exactly one "
        "'Excluded from primary sample' row "
        "in the cleaning audit.\n"
        f"Found: {len(audit_excluded_rows_final):,}"

    )


audit_excluded_count_final = int(

    audit_excluded_rows_final.iloc[0]

)


# ============================================================
# VALIDATE AUDIT INPUT COUNT
# ============================================================

if audit_input_count_final != EXPECTED_ROWS:

    raise ValueError(

        "\nAudit input count does not match "
        "the expected raw Reddit row count.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {audit_input_count_final:,}"

    )


# ============================================================
# VALIDATE AUDIT RETAINED COUNT
# ============================================================

if audit_retained_count_final != retained_primary_count:

    raise ValueError(

        "\nAudit retained count does not match "
        "retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {audit_retained_count_final:,}"

    )


# ============================================================
# VALIDATE AUDIT EXCLUDED COUNT
# ============================================================

if audit_excluded_count_final != excluded_primary_count:

    raise ValueError(

        "\nAudit excluded count does not match "
        "excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {audit_excluded_count_final:,}"

    )


# ============================================================
# VALIDATE AUDIT ACCOUNTING
# ============================================================

if (

    audit_retained_count_final
    +
    audit_excluded_count_final

    !=

    audit_input_count_final

):

    raise ValueError(

        "\nAudit retained + excluded counts do not "
        "equal the audit input count."

    )


# ============================================================
# DISPLAY AUDIT RESULTS
# ============================================================

print(
    f"\nAudit input posts: "
    f"{audit_input_count_final:,}"
)


print(
    f"Audit retained posts: "
    f"{audit_retained_count_final:,}"
)


print(
    f"Audit excluded posts: "
    f"{audit_excluded_count_final:,}"
)


print(
    "\nSection 7B Part 6A — audit table validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6B — YEAR / ASSET SUMMARY VALIDATION
# ============================================================


# ============================================================
# CHECK REQUIRED YEAR / ASSET COLUMNS
# ============================================================

required_year_asset_columns_final = {

    "year",

    "asset",

    "input_posts",

    "retained_posts",

    "excluded_posts",

    "unusable_posts",

    "crossposts",

    "promotional_posts",

    "template_posts",

    "exact_duplicate_posts",

    "link_only_posts",

    "repeated_text_5plus_posts",

    "retained_percent",

    "excluded_percent",

}


missing_year_asset_columns_final = (

    required_year_asset_columns_final

    -

    set(
        check_year_asset.columns
    )

)


if missing_year_asset_columns_final:

    raise ValueError(

        "\nReloaded year/asset summary is missing "
        "required columns:\n"
        f"{sorted(missing_year_asset_columns_final)}"

    )


# ============================================================
# CHECK EXPECTED NUMBER OF ROWS
# ============================================================

# Five years x two assets = 10 rows.

if len(check_year_asset) != 10:

    raise ValueError(

        "\nYear/asset summary should contain "
        "exactly 10 rows.\n"
        f"Expected: 10\n"
        f"Found:    {len(check_year_asset):,}"

    )


# ============================================================
# CHECK EXPECTED YEARS
# ============================================================

year_asset_years_final = set(

    pd.to_numeric(
        check_year_asset["year"],
        errors="raise"
    )
    .astype(int)
    .unique()

)


expected_year_asset_years_final = {

    2021,

    2022,

    2023,

    2024,

    2025,

}


if (
    year_asset_years_final
    !=
    expected_year_asset_years_final
):

    raise ValueError(

        "\nUnexpected years in the reloaded "
        "year/asset summary.\n"
        f"Expected: "
        f"{sorted(expected_year_asset_years_final)}\n"
        f"Found:    "
        f"{sorted(year_asset_years_final)}"

    )


# ============================================================
# CHECK EXPECTED ASSETS
# ============================================================

year_asset_assets_final = set(

    check_year_asset[
        "asset"
    ]
    .dropna()
    .astype(str)
    .unique()

)


expected_year_asset_assets_final = {

    "BTC",

    "ETH",

}


if (
    year_asset_assets_final
    !=
    expected_year_asset_assets_final
):

    raise ValueError(

        "\nUnexpected assets in the reloaded "
        "year/asset summary.\n"
        f"Expected: "
        f"{sorted(expected_year_asset_assets_final)}\n"
        f"Found:    "
        f"{sorted(year_asset_assets_final)}"

    )


# ============================================================
# CHECK EACH YEAR / ASSET COMBINATION IS UNIQUE
# ============================================================

duplicate_year_asset_rows_final = int(

    check_year_asset
    .duplicated(
        subset=[
            "year",
            "asset",
        ]
    )
    .sum()

)


if duplicate_year_asset_rows_final != 0:

    raise ValueError(

        "\nDuplicate year/asset combinations found "
        "in the reloaded summary.\n"
        f"Duplicate rows: "
        f"{duplicate_year_asset_rows_final:,}"

    )


# ============================================================
# CHECK ALL 10 EXPECTED YEAR / ASSET COMBINATIONS EXIST
# ============================================================

expected_year_asset_pairs_final = {

    (year, asset)

    for year in [
        2021,
        2022,
        2023,
        2024,
        2025,
    ]

    for asset in [
        "BTC",
        "ETH",
    ]

}


actual_year_asset_pairs_final = set(

    zip(

        pd.to_numeric(
            check_year_asset["year"],
            errors="raise"
        ).astype(int),

        check_year_asset[
            "asset"
        ].astype(str),

    )

)


if (
    actual_year_asset_pairs_final
    !=
    expected_year_asset_pairs_final
):

    raise ValueError(

        "\nThe reloaded year/asset summary does not "
        "contain exactly the expected 10 "
        "year/asset combinations."

    )


# ============================================================
# CHECK INPUT TOTAL
# ============================================================

year_asset_input_total_final = int(

    check_year_asset[
        "input_posts"
    ].sum()

)


if year_asset_input_total_final != EXPECTED_ROWS:

    raise ValueError(

        "\nYear/asset input total does not match "
        "the original Reddit dataset.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {year_asset_input_total_final:,}"

    )


# ============================================================
# CHECK RETAINED TOTAL
# ============================================================

year_asset_retained_total_final = int(

    check_year_asset[
        "retained_posts"
    ].sum()

)


if (
    year_asset_retained_total_final
    !=
    retained_primary_count
):

    raise ValueError(

        "\nYear/asset retained total does not match "
        "retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {year_asset_retained_total_final:,}"

    )


# ============================================================
# CHECK EXCLUDED TOTAL
# ============================================================

year_asset_excluded_total_final = int(

    check_year_asset[
        "excluded_posts"
    ].sum()

)


if (
    year_asset_excluded_total_final
    !=
    excluded_primary_count
):

    raise ValueError(

        "\nYear/asset excluded total does not match "
        "excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {year_asset_excluded_total_final:,}"

    )


# ============================================================
# CHECK ROW-LEVEL ACCOUNTING
# ============================================================

invalid_year_asset_accounting_final = int(

    (

        check_year_asset[
            "retained_posts"
        ]

        +

        check_year_asset[
            "excluded_posts"
        ]

        !=

        check_year_asset[
            "input_posts"
        ]

    ).sum()

)


if invalid_year_asset_accounting_final != 0:

    raise ValueError(

        "\nSome year/asset rows do not satisfy:\n"
        "retained_posts + excluded_posts = input_posts\n"
        f"Problem rows: "
        f"{invalid_year_asset_accounting_final:,}"

    )


# ============================================================
# CHECK ORIGINAL YEAR TOTALS
# ============================================================

saved_year_totals_final = (

    check_year_asset

    .groupby(
        "year"
    )[
        "input_posts"
    ]

    .sum()

)


for year, expected_count in EXPECTED_YEAR_COUNTS.items():

    actual_count = int(

        saved_year_totals_final.get(
            year,
            0
        )

    )


    if actual_count != expected_count:

        raise ValueError(

            f"\nYear {year} input count does not match "
            "the expected raw-data count.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {actual_count:,}"

        )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nYear/asset rows: "
    f"{len(check_year_asset):,}"
)


print(
    f"Year/asset input total: "
    f"{year_asset_input_total_final:,}"
)


print(
    f"Year/asset retained total: "
    f"{year_asset_retained_total_final:,}"
)


print(
    f"Year/asset excluded total: "
    f"{year_asset_excluded_total_final:,}"
)


print(
    f"Duplicate year/asset combinations: "
    f"{duplicate_year_asset_rows_final:,}"
)


print(
    "\nSection 7B Part 6B — "
    "year/asset summary validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6C — SUBREDDIT SUMMARY VALIDATION
# ============================================================


# ============================================================
# CHECK REQUIRED SUBREDDIT SUMMARY COLUMNS
# ============================================================

required_subreddit_columns_final = {

    "subreddit",

    "input_posts",

    "retained_posts",

    "excluded_posts",

    "unusable_posts",

    "crossposts",

    "promotional_posts",

    "template_posts",

    "exact_duplicate_posts",

    "link_only_posts",

    "repeated_text_5plus_posts",

    "retained_percent",

    "excluded_percent",

}


missing_subreddit_columns_final = (

    required_subreddit_columns_final

    -

    set(
        check_subreddit.columns
    )

)


if missing_subreddit_columns_final:

    raise ValueError(

        "\nReloaded subreddit summary is missing "
        "required columns:\n"
        f"{sorted(missing_subreddit_columns_final)}"

    )


# ============================================================
# CHECK EXPECTED NUMBER OF SUBREDDIT ROWS
# ============================================================

if len(check_subreddit) != 3:

    raise ValueError(

        "\nSubreddit summary should contain exactly "
        "3 rows.\n"
        f"Expected: 3\n"
        f"Found:    {len(check_subreddit):,}"

    )


# ============================================================
# CHECK EXPECTED SUBREDDITS
# ============================================================

saved_subreddits_summary_final = set(

    check_subreddit[
        "subreddit"
    ]
    .dropna()
    .astype(str)
    .unique()

)


expected_subreddits_summary_final = set(

    EXPECTED_SUBREDDIT_COUNTS.keys()

)


if (
    saved_subreddits_summary_final
    !=
    expected_subreddits_summary_final
):

    raise ValueError(

        "\nUnexpected subreddit coverage in the "
        "reloaded subreddit summary.\n"
        f"Expected: "
        f"{sorted(expected_subreddits_summary_final)}\n"
        f"Found:    "
        f"{sorted(saved_subreddits_summary_final)}"

    )


# ============================================================
# CHECK EACH SUBREDDIT APPEARS EXACTLY ONCE
# ============================================================

duplicate_subreddit_rows_final = int(

    check_subreddit[
        "subreddit"
    ]
    .duplicated()
    .sum()

)


if duplicate_subreddit_rows_final != 0:

    raise ValueError(

        "\nDuplicate subreddit rows found in the "
        "reloaded subreddit summary.\n"
        f"Duplicate rows: "
        f"{duplicate_subreddit_rows_final:,}"

    )


# ============================================================
# CHECK TOTAL INPUT POSTS
# ============================================================

subreddit_input_total_final = int(

    check_subreddit[
        "input_posts"
    ]
    .sum()

)


if subreddit_input_total_final != EXPECTED_ROWS:

    raise ValueError(

        "\nSubreddit summary input total does not "
        "match the original Reddit dataset.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {subreddit_input_total_final:,}"

    )


# ============================================================
# CHECK TOTAL RETAINED POSTS
# ============================================================

subreddit_retained_total_final = int(

    check_subreddit[
        "retained_posts"
    ]
    .sum()

)


if (
    subreddit_retained_total_final
    !=
    retained_primary_count
):

    raise ValueError(

        "\nSubreddit summary retained total does not "
        "match retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {subreddit_retained_total_final:,}"

    )


# ============================================================
# CHECK TOTAL EXCLUDED POSTS
# ============================================================

subreddit_excluded_total_final = int(

    check_subreddit[
        "excluded_posts"
    ]
    .sum()

)


if (
    subreddit_excluded_total_final
    !=
    excluded_primary_count
):

    raise ValueError(

        "\nSubreddit summary excluded total does not "
        "match excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {subreddit_excluded_total_final:,}"

    )


# ============================================================
# CHECK ROW-LEVEL ACCOUNTING
# ============================================================

invalid_subreddit_accounting_final = int(

    (

        check_subreddit[
            "retained_posts"
        ]

        +

        check_subreddit[
            "excluded_posts"
        ]

        !=

        check_subreddit[
            "input_posts"
        ]

    ).sum()

)


if invalid_subreddit_accounting_final != 0:

    raise ValueError(

        "\nSome subreddit rows do not satisfy:\n"
        "retained_posts + excluded_posts = input_posts\n"
        f"Problem rows: "
        f"{invalid_subreddit_accounting_final:,}"

    )


# ============================================================
# CHECK ORIGINAL RAW SUBREDDIT COUNTS
# ============================================================

for subreddit, expected_count in (
    EXPECTED_SUBREDDIT_COUNTS.items()
):

    matching_rows = check_subreddit.loc[

        check_subreddit[
            "subreddit"
        ].eq(subreddit),

        "input_posts"

    ]


    if len(matching_rows) != 1:

        raise ValueError(

            f"\nExpected exactly one summary row for "
            f"r/{subreddit}.\n"
            f"Found: {len(matching_rows):,}"

        )


    actual_count = int(
        matching_rows.iloc[0]
    )


    if actual_count != expected_count:

        raise ValueError(

            f"\nInput count for r/{subreddit} does not "
            "match the expected raw-data count.\n"
            f"Expected: {expected_count:,}\n"
            f"Found:    {actual_count:,}"

        )


# ============================================================
# CHECK PERCENTAGES ARE WITHIN VALID RANGE
# ============================================================

invalid_retained_percent_final = int(

    (

        check_subreddit[
            "retained_percent"
        ].lt(0)

        |

        check_subreddit[
            "retained_percent"
        ].gt(100)

    ).sum()

)


invalid_excluded_percent_final = int(

    (

        check_subreddit[
            "excluded_percent"
        ].lt(0)

        |

        check_subreddit[
            "excluded_percent"
        ].gt(100)

    ).sum()

)


if invalid_retained_percent_final != 0:

    raise ValueError(

        "\nInvalid retained_percent values found "
        "in the subreddit summary."

    )


if invalid_excluded_percent_final != 0:

    raise ValueError(

        "\nInvalid excluded_percent values found "
        "in the subreddit summary."

    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

print(
    f"\nSubreddit summary rows: "
    f"{len(check_subreddit):,}"
)


print(
    f"Subreddit input total: "
    f"{subreddit_input_total_final:,}"
)


print(
    f"Subreddit retained total: "
    f"{subreddit_retained_total_final:,}"
)


print(
    f"Subreddit excluded total: "
    f"{subreddit_excluded_total_final:,}"
)


print(
    f"Duplicate subreddit rows: "
    f"{duplicate_subreddit_rows_final:,}"
)


print(
    "\nSection 7B Part 6C — "
    "subreddit summary validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6D — DAILY BTC / ETH COVERAGE VALIDATION
# ============================================================


# ============================================================
# CHECK REQUIRED DAILY COVERAGE COLUMNS
# ============================================================

required_daily_columns_final = {

    "post_date",

    "asset",

    "raw_post_count",

    "retained_post_count",

    "excluded_post_count",

    "has_raw_posts",

    "has_retained_posts",

    "year",

}


missing_daily_columns_final = (

    required_daily_columns_final

    -

    set(
        check_daily.columns
    )

)


if missing_daily_columns_final:

    raise ValueError(

        "\nReloaded daily coverage file is missing "
        "required columns:\n"
        f"{sorted(missing_daily_columns_final)}"

    )


# ============================================================
# PARSE DAILY DATES
# ============================================================

check_daily["post_date"] = pd.to_datetime(

    check_daily[
        "post_date"
    ],

    errors="raise"

)


# ============================================================
# CHECK EXPECTED NUMBER OF ROWS
# ============================================================

# 2021-01-01 through 2025-12-31 = 1,826 calendar days.
#
# Two assets are required for every day:
#
# BTC
# ETH
#
# Therefore:
#
# 1,826 x 2 = 3,652 date-asset rows.

expected_calendar_days_final = 1826

expected_daily_rows_final = (

    expected_calendar_days_final
    *
    2

)


if len(check_daily) != expected_daily_rows_final:

    raise ValueError(

        "\nDaily coverage file has an unexpected "
        "number of rows.\n"
        f"Expected: {expected_daily_rows_final:,}\n"
        f"Found:    {len(check_daily):,}"

    )


# ============================================================
# CHECK DATE RANGE
# ============================================================

daily_min_date_final = (

    check_daily[
        "post_date"
    ]
    .min()

)


daily_max_date_final = (

    check_daily[
        "post_date"
    ]
    .max()

)


if daily_min_date_final != pd.Timestamp("2021-01-01"):

    raise ValueError(

        "\nDaily coverage does not begin on "
        "2021-01-01.\n"
        f"Found: {daily_min_date_final.date()}"

    )


if daily_max_date_final != pd.Timestamp("2025-12-31"):

    raise ValueError(

        "\nDaily coverage does not end on "
        "2025-12-31.\n"
        f"Found: {daily_max_date_final.date()}"

    )


# ============================================================
# CHECK NUMBER OF UNIQUE CALENDAR DAYS
# ============================================================

unique_daily_dates_final = int(

    check_daily[
        "post_date"
    ]
    .nunique()

)


if unique_daily_dates_final != expected_calendar_days_final:

    raise ValueError(

        "\nDaily coverage does not contain exactly "
        "1,826 unique calendar dates.\n"
        f"Expected: {expected_calendar_days_final:,}\n"
        f"Found:    {unique_daily_dates_final:,}"

    )


# ============================================================
# CHECK EXPECTED ASSETS
# ============================================================

daily_assets_final = set(

    check_daily[
        "asset"
    ]
    .dropna()
    .astype(str)
    .unique()

)


if daily_assets_final != {"BTC", "ETH"}:

    raise ValueError(

        "\nUnexpected assets in daily coverage.\n"
        f"Expected: ['BTC', 'ETH']\n"
        f"Found:    {sorted(daily_assets_final)}"

    )


# ============================================================
# CHECK EACH DATE / ASSET PAIR IS UNIQUE
# ============================================================

duplicate_daily_pairs_final = int(

    check_daily
    .duplicated(
        subset=[
            "post_date",
            "asset",
        ]
    )
    .sum()

)


if duplicate_daily_pairs_final != 0:

    raise ValueError(

        "\nDuplicate date/asset rows found in "
        "daily coverage.\n"
        f"Duplicate rows: "
        f"{duplicate_daily_pairs_final:,}"

    )


# ============================================================
# CHECK EACH DATE HAS BOTH BTC AND ETH
# ============================================================

assets_per_date_final = (

    check_daily

    .groupby(
        "post_date"
    )[
        "asset"
    ]

    .nunique()

)


dates_without_two_assets_final = int(

    assets_per_date_final
    .ne(2)
    .sum()

)


if dates_without_two_assets_final != 0:

    raise ValueError(

        "\nSome calendar dates do not contain both "
        "BTC and ETH rows.\n"
        f"Problem dates: "
        f"{dates_without_two_assets_final:,}"

    )


# ============================================================
# CHECK COUNT COLUMNS ARE NON-NEGATIVE
# ============================================================

daily_count_columns_final = [

    "raw_post_count",

    "retained_post_count",

    "excluded_post_count",

]


for column in daily_count_columns_final:

    negative_count = int(

        check_daily[
            column
        ]
        .lt(0)
        .sum()

    )


    if negative_count != 0:

        raise ValueError(

            f"\nNegative values found in {column}.\n"
            f"Problem rows: {negative_count:,}"

        )


# ============================================================
# CHECK DAILY ACCOUNTING
# ============================================================

# For every date and asset:
#
# retained + excluded = raw

invalid_daily_accounting_final = int(

    (

        check_daily[
            "retained_post_count"
        ]

        +

        check_daily[
            "excluded_post_count"
        ]

        !=

        check_daily[
            "raw_post_count"
        ]

    ).sum()

)


if invalid_daily_accounting_final != 0:

    raise ValueError(

        "\nSome daily rows do not satisfy:\n"
        "retained_post_count + excluded_post_count "
        "= raw_post_count\n"
        f"Problem rows: "
        f"{invalid_daily_accounting_final:,}"

    )


# ============================================================
# CHECK RAW POST TOTAL
# ============================================================

daily_raw_total_final = int(

    check_daily[
        "raw_post_count"
    ]
    .sum()

)


if daily_raw_total_final != EXPECTED_ROWS:

    raise ValueError(

        "\nDaily raw-post total does not match "
        "the original Reddit dataset.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {daily_raw_total_final:,}"

    )

# ============================================================
# CHECK RETAINED POST TOTAL
# ============================================================

daily_retained_total_final = int(

    check_daily[
        "retained_post_count"
    ]
    .sum()

)


if daily_retained_total_final != retained_primary_count:

    raise ValueError(

        "\nDaily retained-post total does not match "
        "retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {daily_retained_total_final:,}"

    )


# ============================================================
# CHECK EXCLUDED POST TOTAL
# ============================================================

daily_excluded_total_final = int(

    check_daily[
        "excluded_post_count"
    ]
    .sum()

)


if daily_excluded_total_final != excluded_primary_count:

    raise ValueError(

        "\nDaily excluded-post total does not match "
        "excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {daily_excluded_total_final:,}"

    )


# ============================================================
# CHECK DAILY YEAR COLUMN
# ============================================================

calculated_daily_year_final = (

    check_daily[
        "post_date"
    ]
    .dt.year

)


saved_daily_year_final = (

    pd.to_numeric(
        check_daily[
            "year"
        ],
        errors="raise"
    )
    .astype(int)

)


invalid_daily_year_rows_final = int(

    (
        calculated_daily_year_final
        !=
        saved_daily_year_final
    )
    .sum()

)


if invalid_daily_year_rows_final != 0:

    raise ValueError(

        "\nSome saved daily year values do not match "
        "post_date.\n"
        f"Problem rows: "
        f"{invalid_daily_year_rows_final:,}"

    )


# ============================================================
# DISPLAY DAILY COVERAGE RESULTS
# ============================================================

print(
    f"\nDaily coverage rows: "
    f"{len(check_daily):,}"
)


print(
    f"Unique calendar dates: "
    f"{unique_daily_dates_final:,}"
)


print(
    f"Daily raw-post total: "
    f"{daily_raw_total_final:,}"
)


print(
    f"Daily retained-post total: "
    f"{daily_retained_total_final:,}"
)


print(
    f"Daily excluded-post total: "
    f"{daily_excluded_total_final:,}"
)


print(
    f"Duplicate date/asset rows: "
    f"{duplicate_daily_pairs_final:,}"
)


print(
    f"Dates missing BTC or ETH: "
    f"{dates_without_two_assets_final:,}"
)


print(
    f"Invalid daily accounting rows: "
    f"{invalid_daily_accounting_final:,}"
)


print(
    "\nSection 7B Part 6D — "
    "daily BTC/ETH coverage validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6E — MANUAL-REVIEW EXAMPLES VALIDATION
# CHUNK 1
# ============================================================


# ============================================================
# CHECK REQUIRED EXAMPLE COLUMNS
# ============================================================

required_example_columns_final = {

    "review_category",

    "total_posts_with_flag",

    "post_id",

    "post_date",

    "subreddit",

    "asset",

    "title_original",

    "body_original",

    "analysis_text_preview",

    "normalized_text_frequency",

    "exclusion_reason",

}


missing_example_columns_final = (

    required_example_columns_final

    -

    set(
        check_examples.columns
    )

)


if missing_example_columns_final:

    raise ValueError(

        "\nReloaded manual-review examples file is "
        "missing required columns:\n"
        f"{sorted(missing_example_columns_final)}"

    )


# ============================================================
# CHECK EXPECTED REVIEW CATEGORIES
# ============================================================

expected_review_categories_final = {

    "unusable_text",

    "crosspost",

    "promotional",

    "recurring_template",

    "exact_duplicate",

    "link_only",

    "repeated_text_5plus_diagnostic",

}


actual_review_categories_final = set(

    check_examples[
        "review_category"
    ]
    .dropna()
    .astype(str)
    .unique()

)


unexpected_review_categories_final = (

    actual_review_categories_final

    -

    expected_review_categories_final

)


if unexpected_review_categories_final:

    raise ValueError(

        "\nUnexpected manual-review categories found:\n"
        f"{sorted(unexpected_review_categories_final)}"

    )


# ============================================================
# CHECK MAXIMUM 25 EXAMPLES PER CATEGORY
# ============================================================

example_counts_by_category_final = (

    check_examples

    .groupby(
        "review_category"
    )

    .size()

)


too_many_examples_final = (

    example_counts_by_category_final[

        example_counts_by_category_final > 25

    ]

)


if not too_many_examples_final.empty:

    raise ValueError(

        "\nMore than 25 manual-review examples were "
        "saved for at least one category:\n"
        f"{too_many_examples_final.to_dict()}"

    )


# ============================================================
# CHECK POST IDs ARE PRESENT
# ============================================================

missing_example_post_ids_final = int(

    check_examples[
        "post_id"
    ]
    .isna()
    .sum()

)


if missing_example_post_ids_final != 0:

    raise ValueError(

        "\nManual-review examples contain missing "
        "post IDs.\n"
        f"Missing IDs: "
        f"{missing_example_post_ids_final:,}"

    )


# ============================================================
# CHECK EXAMPLE POSTS EXIST IN FULL CLEANED DATA
# ============================================================

clean_post_id_set_final = set(

    check_clean[
        "post_id"
    ]
    .astype(str)

)


example_post_id_set_final = set(

    check_examples[
        "post_id"
    ]
    .dropna()
    .astype(str)

)


unknown_example_post_ids_final = (

    example_post_id_set_final

    -

    clean_post_id_set_final

)


if unknown_example_post_ids_final:

    raise ValueError(

        "\nSome manual-review example post IDs do "
        "not exist in the full cleaned dataset.\n"
        f"Unknown IDs: "
        f"{len(unknown_example_post_ids_final):,}"

    )


# ============================================================
# MAP REVIEW CATEGORIES TO CLEANING FLAGS
# ============================================================

review_category_flag_map_final = {

    "unusable_text":
        "flag_unusable_text",

    "crosspost":
        "flag_crosspost",

    "promotional":
        "flag_promotional",

    "recurring_template":
        "flag_template",

    "exact_duplicate":
        "flag_exact_duplicate",

    "link_only":
        "flag_link_only",

    "repeated_text_5plus_diagnostic":
        "flag_repeated_text_5plus",

}


# ============================================================
# BUILD FLAG LOOKUP FROM FULL CLEANED DATA
# ============================================================

example_flag_lookup_final = (

    check_clean[
        [
            "post_id",
            *review_category_flag_map_final.values(),
        ]
    ]

    .copy()

)


example_flag_lookup_final[
    "post_id"
] = (

    example_flag_lookup_final[
        "post_id"
    ]
    .astype(str)

)


# ============================================================
# NORMALISE FLAG COLUMNS TO BOOLEAN
# ============================================================

for flag_column in review_category_flag_map_final.values():

    if (
        example_flag_lookup_final[
            flag_column
        ].dtype
        != bool
    ):

        example_flag_lookup_final[
            flag_column
        ] = (

            example_flag_lookup_final[
                flag_column
            ]

            .astype(str)

            .str.strip()

            .str.lower()

            .map(
                {
                    "true": True,
                    "false": False,
                }
            )

        )


        if (
            example_flag_lookup_final[
                flag_column
            ]
            .isna()
            .any()
        ):

            raise ValueError(

                f"\nCould not convert {flag_column} "
                "to Boolean during final validation."

            )


print(
    "\nSection 7B Part 6E — Chunk 1 complete"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6E — MANUAL-REVIEW EXAMPLES VALIDATION
# CHUNK 2
# ============================================================


# ============================================================
# VALIDATE EACH REVIEW CATEGORY AGAINST ITS FLAG
# ============================================================

for (
    review_category,
    flag_column
) in review_category_flag_map_final.items():

    category_examples_final = (

        check_examples.loc[

            check_examples[
                "review_category"
            ].eq(review_category),

            [
                "post_id",
                "total_posts_with_flag",
            ]

        ]

        .copy()

    )


    # A category may legitimately have zero saved examples
    # if no posts received that flag.

    if category_examples_final.empty:

        continue


    category_examples_final[
        "post_id"
    ] = (

        category_examples_final[
            "post_id"
        ]
        .astype(str)

    )


    category_validation_final = (

        category_examples_final

        .merge(

            example_flag_lookup_final[
                [
                    "post_id",
                    flag_column,
                ]
            ],

            on="post_id",

            how="left",

            validate="many_to_one",

        )

    )


    # ========================================================
    # CHECK ALL EXAMPLES MATCH BACK TO CLEANED DATA
    # ========================================================

    missing_flag_matches_final = int(

        category_validation_final[
            flag_column
        ]
        .isna()
        .sum()

    )


    if missing_flag_matches_final != 0:

        raise ValueError(

            f"\nCould not match some "
            f"'{review_category}' examples back to "
            "the full cleaned dataset.\n"
            f"Problem rows: "
            f"{missing_flag_matches_final:,}"

        )


    # ========================================================
    # CHECK CORRECT FLAG IS TRUE
    # ========================================================

    invalid_category_examples_final = int(

        (

            ~category_validation_final[
                flag_column
            ]

        ).sum()

    )


    if invalid_category_examples_final != 0:

        raise ValueError(

            f"\nSome examples labelled "
            f"'{review_category}' do not have "
            f"{flag_column}=True in the full "
            "cleaned dataset.\n"
            f"Problem rows: "
            f"{invalid_category_examples_final:,}"

        )


    # ========================================================
    # CHECK SAVED TOTAL_POSTS_WITH_FLAG VALUE
    # ========================================================

    expected_flag_total_final = int(

        example_flag_lookup_final[
            flag_column
        ]
        .sum()

    )


    saved_flag_totals_final = (

        pd.to_numeric(

            category_examples_final[
                "total_posts_with_flag"
            ],

            errors="raise",

        )
        .astype(int)
        .unique()

    )


    if len(saved_flag_totals_final) != 1:

        raise ValueError(

            f"\nExamples for '{review_category}' "
            "contain inconsistent values in "
            "total_posts_with_flag."

        )


    saved_flag_total_final = int(

        saved_flag_totals_final[0]

    )


    if (
        saved_flag_total_final
        !=
        expected_flag_total_final
    ):

        raise ValueError(

            f"\nSaved total_posts_with_flag for "
            f"'{review_category}' does not match "
            "the full cleaned dataset.\n"
            f"Expected: {expected_flag_total_final:,}\n"
            f"Found:    {saved_flag_total_final:,}"

        )


# ============================================================
# CHECK EXAMPLE ASSETS
# ============================================================

example_assets_final = set(

    check_examples[
        "asset"
    ]
    .dropna()
    .astype(str)
    .unique()

)


unexpected_example_assets_final = (

    example_assets_final

    -

    {
        "BTC",
        "ETH",
    }

)


if unexpected_example_assets_final:

    raise ValueError(

        "\nUnexpected assets found in manual-review "
        "examples:\n"
        f"{sorted(unexpected_example_assets_final)}"

    )


# ============================================================
# CHECK EXAMPLE SUBREDDITS
# ============================================================

example_subreddits_final = set(

    check_examples[
        "subreddit"
    ]
    .dropna()
    .astype(str)
    .unique()

)


unexpected_example_subreddits_final = (

    example_subreddits_final

    -

    set(
        EXPECTED_SUBREDDIT_COUNTS.keys()
    )

)


if unexpected_example_subreddits_final:

    raise ValueError(

        "\nUnexpected subreddits found in "
        "manual-review examples:\n"
        f"{sorted(unexpected_example_subreddits_final)}"

    )


# ============================================================
# CHECK EXAMPLE DATES
# ============================================================

example_dates_final = pd.to_datetime(

    check_examples[
        "post_date"
    ],

    errors="raise",

)


invalid_example_dates_final = int(

    (

        example_dates_final.lt(
            pd.Timestamp("2021-01-01")
        )

        |

        example_dates_final.gt(
            pd.Timestamp("2025-12-31")
        )

    ).sum()

)


if invalid_example_dates_final != 0:

    raise ValueError(

        "\nManual-review examples contain dates "
        "outside the study period.\n"
        f"Problem rows: "
        f"{invalid_example_dates_final:,}"

    )


# ============================================================
# DISPLAY MANUAL-REVIEW VALIDATION RESULTS
# ============================================================

print(
    f"\nManual-review example rows: "
    f"{len(check_examples):,}"
)


print(
    f"Review categories represented: "
    f"{len(actual_review_categories_final):,}"
)


print(
    f"Unique example post IDs: "
    f"{len(example_post_id_set_final):,}"
)


print(
    f"Unknown example post IDs: "
    f"{len(unknown_example_post_ids_final):,}"
)


print(
    "\nExamples per review category:"
)


print(
    example_counts_by_category_final
    .sort_index()
    .to_string()
)


print(
    "\nSection 7B Part 6E — "
    "manual-review examples validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6F — OUTPUT SCHEMA AND DATA-MINIMISATION VALIDATION
# CHUNK 1
# ============================================================


# ============================================================
# DEFINE EXPECTED FULL CLEANED DATASET COLUMNS
# ============================================================

expected_clean_full_columns_final = [

    "post_id",
    "created_at",
    "post_date",
    "year",
    "subreddit",
    "asset",

    "title_original",
    "body_original",

    "title_clean",
    "body_clean",
    "analysis_text",

    "score",
    "upvote_ratio",
    "num_comments",

    "analysis_alpha_count",
    "normalized_text_frequency",

    "flag_empty_text",
    "flag_deleted_removed",
    "flag_too_short",
    "flag_unusable_text",
    "flag_crosspost",
    "flag_promotional",
    "flag_template",
    "flag_exact_duplicate",
    "flag_link_only",
    "flag_repeated_text_5plus",

    "exclude_primary",
    "retain_primary",
    "exclusion_reason",

]


# ============================================================
# DEFINE EXPECTED PRIMARY SAMPLE COLUMNS
# ============================================================

expected_primary_columns_final = [

    "post_id",
    "created_at",
    "post_date",
    "year",
    "subreddit",
    "asset",

    "title_clean",
    "body_clean",
    "analysis_text",

    "score",
    "upvote_ratio",
    "num_comments",

]


# ============================================================
# CHECK FULL CLEANED DATASET SCHEMA
# ============================================================

actual_clean_full_columns_final = list(
    check_clean.columns
)


if (
    actual_clean_full_columns_final
    !=
    expected_clean_full_columns_final
):

    missing_clean_columns_final = [

        column

        for column in expected_clean_full_columns_final

        if column not in actual_clean_full_columns_final

    ]


    unexpected_clean_columns_final = [

        column

        for column in actual_clean_full_columns_final

        if column not in expected_clean_full_columns_final

    ]


    raise ValueError(

        "\nFull cleaned dataset schema does not match "
        "the expected final schema.\n"

        f"\nMissing columns:\n"
        f"{missing_clean_columns_final}\n"

        f"\nUnexpected columns:\n"
        f"{unexpected_clean_columns_final}\n"

        f"\nExpected order:\n"
        f"{expected_clean_full_columns_final}\n"

        f"\nActual order:\n"
        f"{actual_clean_full_columns_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE SCHEMA
# ============================================================

actual_primary_columns_final = list(
    check_primary.columns
)


if (
    actual_primary_columns_final
    !=
    expected_primary_columns_final
):

    missing_primary_columns_final = [

        column

        for column in expected_primary_columns_final

        if column not in actual_primary_columns_final

    ]


    unexpected_primary_columns_final = [

        column

        for column in actual_primary_columns_final

        if column not in expected_primary_columns_final

    ]


    raise ValueError(

        "\nPrimary sentiment sample schema does not "
        "match the expected final schema.\n"

        f"\nMissing columns:\n"
        f"{missing_primary_columns_final}\n"

        f"\nUnexpected columns:\n"
        f"{unexpected_primary_columns_final}\n"

        f"\nExpected order:\n"
        f"{expected_primary_columns_final}\n"

        f"\nActual order:\n"
        f"{actual_primary_columns_final}"

    )


# ============================================================
# DEFINE FORBIDDEN OUTPUT COLUMNS
# ============================================================

forbidden_output_columns_final = {

    "url",
    "permalink",
    "crosspost_parent_id",

    "author",
    "author_id",
    "author_name",
    "username",
    "user",
    "user_id",
    "account_id",

}


# ============================================================
# CHECK FULL CLEANED DATASET FOR FORBIDDEN COLUMNS
# ============================================================

forbidden_clean_columns_found_final = sorted(

    forbidden_output_columns_final.intersection(
        set(check_clean.columns)
    )

)


if forbidden_clean_columns_found_final:

    raise ValueError(

        "\nForbidden source/account columns remain "
        "in the full cleaned dataset:\n"
        f"{forbidden_clean_columns_found_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE FOR FORBIDDEN COLUMNS
# ============================================================

forbidden_primary_columns_found_final = sorted(

    forbidden_output_columns_final.intersection(
        set(check_primary.columns)
    )

)


if forbidden_primary_columns_found_final:

    raise ValueError(

        "\nForbidden source/account columns remain "
        "in the primary sentiment sample:\n"
        f"{forbidden_primary_columns_found_final}"

    )


# ============================================================
# CHECK NORMALIZED TEXT WAS NOT SAVED
# ============================================================

if "normalized_text" in check_clean.columns:

    raise ValueError(

        "\nnormalized_text should not be present "
        "in the final full cleaned dataset."

    )


if "normalized_text" in check_primary.columns:

    raise ValueError(

        "\nnormalized_text should not be present "
        "in the primary sentiment sample."

    )


print(
    "\nSection 7B Part 6F — Chunk 1 complete"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6F — OUTPUT SCHEMA AND DATA-MINIMISATION VALIDATION
# CHUNK 2
# ============================================================


# ============================================================
# CHECK FULL CLEANED DATASET ROW COUNT
# ============================================================

if len(check_clean) != EXPECTED_ROWS:

    raise ValueError(

        "\nFull cleaned dataset row count changed "
        "during save/reload.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {len(check_clean):,}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE ROW COUNT
# ============================================================

if len(check_primary) != retained_primary_count:

    raise ValueError(

        "\nPrimary sentiment sample row count does "
        "not match retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {len(check_primary):,}"

    )


# ============================================================
# CHECK MISSING POST IDs
# ============================================================

clean_missing_ids_final = int(

    check_clean[
        "post_id"
    ]
    .isna()
    .sum()

)


primary_missing_ids_final = int(

    check_primary[
        "post_id"
    ]
    .isna()
    .sum()

)


if clean_missing_ids_final != 0:

    raise ValueError(

        "\nMissing post IDs found in the full "
        "cleaned dataset.\n"
        f"Missing IDs: {clean_missing_ids_final:,}"

    )


if primary_missing_ids_final != 0:

    raise ValueError(

        "\nMissing post IDs found in the primary "
        "sentiment sample.\n"
        f"Missing IDs: {primary_missing_ids_final:,}"

    )


# ============================================================
# CHECK DUPLICATE POST IDs
# ============================================================

clean_duplicate_ids_final = int(

    check_clean[
        "post_id"
    ]
    .duplicated()
    .sum()

)


primary_duplicate_ids_final = int(

    check_primary[
        "post_id"
    ]
    .duplicated()
    .sum()

)


if clean_duplicate_ids_final != 0:

    raise ValueError(

        "\nDuplicate post IDs found in the full "
        "cleaned dataset.\n"
        f"Duplicate IDs: {clean_duplicate_ids_final:,}"

    )


if primary_duplicate_ids_final != 0:

    raise ValueError(

        "\nDuplicate post IDs found in the primary "
        "sentiment sample.\n"
        f"Duplicate IDs: {primary_duplicate_ids_final:,}"

    )


# ============================================================
# CHECK PRIMARY ANALYSIS TEXT
# ============================================================

primary_analysis_text_final = (

    check_primary[
        "analysis_text"
    ]
    .fillna("")
    .astype(str)
    .str.strip()

)


empty_primary_text_final = int(

    primary_analysis_text_final
    .eq("")
    .sum()

)


if empty_primary_text_final != 0:

    raise ValueError(

        "\nEmpty analysis_text values found in the "
        "primary sentiment sample.\n"
        f"Problem rows: {empty_primary_text_final:,}"

    )


# ============================================================
# CHECK PRIMARY TEXT HAS AT LEAST 3 ALPHABETIC CHARACTERS
# ============================================================

primary_alpha_counts_final = (

    primary_analysis_text_final
    .map(alpha_count)

)


too_short_primary_text_final = int(

    primary_alpha_counts_final
    .lt(3)
    .sum()

)


if too_short_primary_text_final != 0:

    raise ValueError(

        "\nPrimary sentiment sample contains text "
        "with fewer than 3 alphabetic characters.\n"
        f"Problem rows: "
        f"{too_short_primary_text_final:,}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE ASSETS
# ============================================================

primary_assets_final = set(

    check_primary[
        "asset"
    ]
    .dropna()
    .astype(str)
    .unique()

)


if primary_assets_final != {"BTC", "ETH"}:

    raise ValueError(

        "\nPrimary sentiment sample does not contain "
        "exactly the expected assets.\n"
        "Expected: ['BTC', 'ETH']\n"
        f"Found:    {sorted(primary_assets_final)}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE SUBREDDITS
# ============================================================

primary_subreddits_final = set(

    check_primary[
        "subreddit"
    ]
    .dropna()
    .astype(str)
    .unique()

)


unexpected_primary_subreddits_final = (

    primary_subreddits_final

    -

    set(
        EXPECTED_SUBREDDIT_COUNTS.keys()
    )

)


if unexpected_primary_subreddits_final:

    raise ValueError(

        "\nUnexpected subreddits found in the "
        "primary sentiment sample:\n"
        f"{sorted(unexpected_primary_subreddits_final)}"

    )


# ============================================================
# CHECK BITCOIN SUBREDDIT -> BTC MAPPING
# ============================================================

invalid_btc_mapping_final = int(

    (

        check_primary[
            "subreddit"
        ]
        .isin(
            [
                "Bitcoin",
                "BitcoinMarkets",
            ]
        )

        &

        check_primary[
            "asset"
        ]
        .ne("BTC")

    ).sum()

)


if invalid_btc_mapping_final != 0:

    raise ValueError(

        "\nBitcoin/BitcoinMarkets posts with an "
        "incorrect asset mapping were found.\n"
        f"Problem rows: "
        f"{invalid_btc_mapping_final:,}"

    )


# ============================================================
# CHECK ETHEREUM SUBREDDIT -> ETH MAPPING
# ============================================================

invalid_eth_mapping_final = int(

    (

        check_primary[
            "subreddit"
        ]
        .eq("ethereum")

        &

        check_primary[
            "asset"
        ]
        .ne("ETH")

    ).sum()

)


if invalid_eth_mapping_final != 0:

    raise ValueError(

        "\nEthereum posts with an incorrect asset "
        "mapping were found.\n"
        f"Problem rows: "
        f"{invalid_eth_mapping_final:,}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE DATE RANGE
# ============================================================

primary_dates_final = pd.to_datetime(

    check_primary[
        "post_date"
    ],

    errors="raise",

)


invalid_primary_dates_final = int(

    (

        primary_dates_final.lt(
            pd.Timestamp("2021-01-01")
        )

        |

        primary_dates_final.gt(
            pd.Timestamp("2025-12-31")
        )

    ).sum()

)


if invalid_primary_dates_final != 0:

    raise ValueError(

        "\nPrimary sentiment sample contains dates "
        "outside the study period.\n"
        f"Problem rows: "
        f"{invalid_primary_dates_final:,}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE YEARS
# ============================================================

primary_years_final = set(

    pd.to_numeric(

        check_primary[
            "year"
        ],

        errors="raise",

    )
    .astype(int)
    .unique()

)


if primary_years_final != {
    2021,
    2022,
    2023,
    2024,
    2025,
}:

    raise ValueError(

        "\nPrimary sentiment sample does not contain "
        "the expected study years.\n"
        f"Found: {sorted(primary_years_final)}"

    )


print(
    "\nSection 7B Part 6F — Chunk 2 complete"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6F — OUTPUT SCHEMA AND DATA-MINIMISATION VALIDATION
# CHUNK 3
# ============================================================


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN ORIGINAL RAW TEXT
# ============================================================

raw_text_columns_final = {

    "title_original",
    "body_original",

}


raw_text_in_primary_final = sorted(

    raw_text_columns_final.intersection(
        set(check_primary.columns)
    )

)


if raw_text_in_primary_final:

    raise ValueError(

        "\nOriginal raw-text columns unexpectedly "
        "remain in the primary sentiment sample:\n"
        f"{raw_text_in_primary_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN CLEANING FLAGS
# ============================================================

primary_flag_columns_final = [

    column

    for column in check_primary.columns

    if str(column).startswith("flag_")

]


if primary_flag_columns_final:

    raise ValueError(

        "\nCleaning flag columns unexpectedly remain "
        "in the primary sentiment sample:\n"
        f"{primary_flag_columns_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN DECISION COLUMNS
# ============================================================

decision_columns_not_for_primary_final = {

    "exclude_primary",
    "retain_primary",
    "exclusion_reason",

}


decision_columns_in_primary_final = sorted(

    decision_columns_not_for_primary_final.intersection(
        set(check_primary.columns)
    )

)


if decision_columns_in_primary_final:

    raise ValueError(

        "\nCleaning decision columns unexpectedly "
        "remain in the primary sentiment sample:\n"
        f"{decision_columns_in_primary_final}"

    )


# ============================================================
# CHECK FULL CLEANED DATASET RETAINS AUDIT FLAGS
# ============================================================

required_cleaning_flags_final = {

    "flag_empty_text",
    "flag_deleted_removed",
    "flag_too_short",
    "flag_unusable_text",
    "flag_crosspost",
    "flag_promotional",
    "flag_template",
    "flag_exact_duplicate",
    "flag_link_only",
    "flag_repeated_text_5plus",

}


missing_cleaning_flags_final = (

    required_cleaning_flags_final

    -

    set(check_clean.columns)

)


if missing_cleaning_flags_final:

    raise ValueError(

        "\nFull cleaned dataset is missing required "
        "cleaning/audit flags:\n"
        f"{sorted(missing_cleaning_flags_final)}"

    )


# ============================================================
# CHECK FULL CLEANED DATASET RETAINS DECISION FIELDS
# ============================================================

required_decision_columns_final = {

    "exclude_primary",
    "retain_primary",
    "exclusion_reason",

}


missing_decision_columns_final = (

    required_decision_columns_final

    -

    set(check_clean.columns)

)


if missing_decision_columns_final:

    raise ValueError(

        "\nFull cleaned dataset is missing required "
        "primary-sample decision fields:\n"
        f"{sorted(missing_decision_columns_final)}"

    )


# ============================================================
# CHECK NO AUTHOR / ACCOUNT-LIKE COLUMN NAMES REMAIN
# ============================================================

# This is an additional defensive check.
# It catches obvious identifier-like column names even if
# they were not included in the explicit forbidden set.

identifier_terms_final = {

    "author",
    "username",
    "user_id",
    "account_id",
    "author_id",
    "author_name",

}


identifier_like_clean_columns_final = [

    column

    for column in check_clean.columns

    if str(column).strip().lower()
    in identifier_terms_final

]


identifier_like_primary_columns_final = [

    column

    for column in check_primary.columns

    if str(column).strip().lower()
    in identifier_terms_final

]


if identifier_like_clean_columns_final:

    raise ValueError(

        "\nAuthor/account identifier-like columns "
        "remain in the full cleaned dataset:\n"
        f"{identifier_like_clean_columns_final}"

    )


if identifier_like_primary_columns_final:

    raise ValueError(

        "\nAuthor/account identifier-like columns "
        "remain in the primary sentiment sample:\n"
        f"{identifier_like_primary_columns_final}"

    )


# ============================================================
# CHECK RETAINED PRIMARY IDS MATCH PRIMARY FILE
# ============================================================

retained_ids_schema_final = set(

    check_clean.loc[

        check_clean[
            "retain_primary"
        ].eq(True),

        "post_id"

    ]
    .astype(str)

)


primary_ids_schema_final = set(

    check_primary[
        "post_id"
    ]
    .astype(str)

)


missing_primary_ids_schema_final = (

    retained_ids_schema_final

    -

    primary_ids_schema_final

)


unexpected_primary_ids_schema_final = (

    primary_ids_schema_final

    -

    retained_ids_schema_final

)


if missing_primary_ids_schema_final:

    raise ValueError(

        "\nSome retained posts from the full cleaned "
        "dataset are missing from the primary "
        "sentiment sample.\n"
        f"Missing posts: "
        f"{len(missing_primary_ids_schema_final):,}"

    )


if unexpected_primary_ids_schema_final:

    raise ValueError(

        "\nThe primary sentiment sample contains "
        "posts that are not marked retain_primary "
        "in the full cleaned dataset.\n"
        f"Unexpected posts: "
        f"{len(unexpected_primary_ids_schema_final):,}"

    )


# ============================================================
# CHECK EXCLUDED POSTS ARE ABSENT FROM PRIMARY FILE
# ============================================================

excluded_ids_schema_final = set(

    check_clean.loc[

        check_clean[
            "exclude_primary"
        ].eq(True),

        "post_id"

    ]
    .astype(str)

)


excluded_in_primary_schema_final = (

    excluded_ids_schema_final

    &

    primary_ids_schema_final

)


if excluded_in_primary_schema_final:

    raise ValueError(

        "\nExcluded posts were found in the primary "
        "sentiment sample.\n"
        f"Problem posts: "
        f"{len(excluded_in_primary_schema_final):,}"

    )


# ============================================================
# CHECK FINAL RETAINED / EXCLUDED ACCOUNTING
# ============================================================

final_retained_schema_count = len(
    retained_ids_schema_final
)


final_excluded_schema_count = len(
    excluded_ids_schema_final
)


if final_retained_schema_count != retained_primary_count:

    raise ValueError(

        "\nFinal retained ID count does not match "
        "retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {final_retained_schema_count:,}"

    )


if final_excluded_schema_count != excluded_primary_count:

    raise ValueError(

        "\nFinal excluded ID count does not match "
        "excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {final_excluded_schema_count:,}"

    )


if (

    final_retained_schema_count
    +
    final_excluded_schema_count

    !=

    EXPECTED_ROWS

):

    raise ValueError(

        "\nFinal retained + excluded ID counts do "
        "not equal the original dataset size."

    )


# ============================================================
# DISPLAY FINAL SCHEMA / MINIMISATION RESULTS
# ============================================================

print(
    f"\nFull cleaned columns: "
    f"{len(check_clean.columns):,}"
)


print(
    f"Primary sample columns: "
    f"{len(check_primary.columns):,}"
)


print(
    f"Forbidden columns in full cleaned file: "
    f"{len(forbidden_clean_columns_found_final):,}"
)


print(
    f"Forbidden columns in primary file: "
    f"{len(forbidden_primary_columns_found_final):,}"
)


print(
    f"Final retained IDs: "
    f"{final_retained_schema_count:,}"
)


print(
    f"Final excluded IDs: "
    f"{final_excluded_schema_count:,}"
)


print(
    f"Excluded IDs found in primary sample: "
    f"{len(excluded_in_primary_schema_final):,}"
)


print(
    "\nSection 7B Part 6F — "
    "output schema and data-minimisation validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 6F — OUTPUT SCHEMA AND DATA-MINIMISATION VALIDATION
# CHUNK 3
# ============================================================


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN ORIGINAL RAW TEXT
# ============================================================

raw_text_columns_final = {

    "title_original",
    "body_original",

}


raw_text_in_primary_final = sorted(

    raw_text_columns_final.intersection(
        set(check_primary.columns)
    )

)


if raw_text_in_primary_final:

    raise ValueError(

        "\nOriginal raw-text columns unexpectedly "
        "remain in the primary sentiment sample:\n"
        f"{raw_text_in_primary_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN CLEANING FLAGS
# ============================================================

primary_flag_columns_final = [

    column

    for column in check_primary.columns

    if str(column).startswith("flag_")

]


if primary_flag_columns_final:

    raise ValueError(

        "\nCleaning flag columns unexpectedly remain "
        "in the primary sentiment sample:\n"
        f"{primary_flag_columns_final}"

    )


# ============================================================
# CHECK PRIMARY SAMPLE DOES NOT CONTAIN DECISION COLUMNS
# ============================================================

decision_columns_not_for_primary_final = {

    "exclude_primary",
    "retain_primary",
    "exclusion_reason",

}


decision_columns_in_primary_final = sorted(

    decision_columns_not_for_primary_final.intersection(
        set(check_primary.columns)
    )

)


if decision_columns_in_primary_final:

    raise ValueError(

        "\nCleaning decision columns unexpectedly "
        "remain in the primary sentiment sample:\n"
        f"{decision_columns_in_primary_final}"

    )


# ============================================================
# CHECK FULL CLEANED DATASET RETAINS AUDIT FLAGS
# ============================================================

required_cleaning_flags_final = {

    "flag_empty_text",
    "flag_deleted_removed",
    "flag_too_short",
    "flag_unusable_text",
    "flag_crosspost",
    "flag_promotional",
    "flag_template",
    "flag_exact_duplicate",
    "flag_link_only",
    "flag_repeated_text_5plus",

}


missing_cleaning_flags_final = (

    required_cleaning_flags_final

    -

    set(check_clean.columns)

)


if missing_cleaning_flags_final:

    raise ValueError(

        "\nFull cleaned dataset is missing required "
        "cleaning/audit flags:\n"
        f"{sorted(missing_cleaning_flags_final)}"

    )


# ============================================================
# CHECK FULL CLEANED DATASET RETAINS DECISION FIELDS
# ============================================================

required_decision_columns_final = {

    "exclude_primary",
    "retain_primary",
    "exclusion_reason",

}


missing_decision_columns_final = (

    required_decision_columns_final

    -

    set(check_clean.columns)

)


if missing_decision_columns_final:

    raise ValueError(

        "\nFull cleaned dataset is missing required "
        "primary-sample decision fields:\n"
        f"{sorted(missing_decision_columns_final)}"

    )


# ============================================================
# CHECK NO AUTHOR / ACCOUNT-LIKE COLUMN NAMES REMAIN
# ============================================================

# This is an additional defensive check.
# It catches obvious identifier-like column names even if
# they were not included in the explicit forbidden set.

identifier_terms_final = {

    "author",
    "username",
    "user_id",
    "account_id",
    "author_id",
    "author_name",

}


identifier_like_clean_columns_final = [

    column

    for column in check_clean.columns

    if str(column).strip().lower()
    in identifier_terms_final

]


identifier_like_primary_columns_final = [

    column

    for column in check_primary.columns

    if str(column).strip().lower()
    in identifier_terms_final

]


if identifier_like_clean_columns_final:

    raise ValueError(

        "\nAuthor/account identifier-like columns "
        "remain in the full cleaned dataset:\n"
        f"{identifier_like_clean_columns_final}"

    )


if identifier_like_primary_columns_final:

    raise ValueError(

        "\nAuthor/account identifier-like columns "
        "remain in the primary sentiment sample:\n"
        f"{identifier_like_primary_columns_final}"

    )


# ============================================================
# CHECK RETAINED PRIMARY IDS MATCH PRIMARY FILE
# ============================================================

retained_ids_schema_final = set(

    check_clean.loc[

        check_clean[
            "retain_primary"
        ].eq(True),

        "post_id"

    ]
    .astype(str)

)


primary_ids_schema_final = set(

    check_primary[
        "post_id"
    ]
    .astype(str)

)


missing_primary_ids_schema_final = (

    retained_ids_schema_final

    -

    primary_ids_schema_final

)


unexpected_primary_ids_schema_final = (

    primary_ids_schema_final

    -

    retained_ids_schema_final

)


if missing_primary_ids_schema_final:

    raise ValueError(

        "\nSome retained posts from the full cleaned "
        "dataset are missing from the primary "
        "sentiment sample.\n"
        f"Missing posts: "
        f"{len(missing_primary_ids_schema_final):,}"

    )


if unexpected_primary_ids_schema_final:

    raise ValueError(

        "\nThe primary sentiment sample contains "
        "posts that are not marked retain_primary "
        "in the full cleaned dataset.\n"
        f"Unexpected posts: "
        f"{len(unexpected_primary_ids_schema_final):,}"

    )


# ============================================================
# CHECK EXCLUDED POSTS ARE ABSENT FROM PRIMARY FILE
# ============================================================

excluded_ids_schema_final = set(

    check_clean.loc[

        check_clean[
            "exclude_primary"
        ].eq(True),

        "post_id"

    ]
    .astype(str)

)


excluded_in_primary_schema_final = (

    excluded_ids_schema_final

    &

    primary_ids_schema_final

)


if excluded_in_primary_schema_final:

    raise ValueError(

        "\nExcluded posts were found in the primary "
        "sentiment sample.\n"
        f"Problem posts: "
        f"{len(excluded_in_primary_schema_final):,}"

    )


# ============================================================
# CHECK FINAL RETAINED / EXCLUDED ACCOUNTING
# ============================================================

final_retained_schema_count = len(
    retained_ids_schema_final
)


final_excluded_schema_count = len(
    excluded_ids_schema_final
)


if final_retained_schema_count != retained_primary_count:

    raise ValueError(

        "\nFinal retained ID count does not match "
        "retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {final_retained_schema_count:,}"

    )


if final_excluded_schema_count != excluded_primary_count:

    raise ValueError(

        "\nFinal excluded ID count does not match "
        "excluded_primary_count.\n"
        f"Expected: {excluded_primary_count:,}\n"
        f"Found:    {final_excluded_schema_count:,}"

    )


if (

    final_retained_schema_count
    +
    final_excluded_schema_count

    !=

    EXPECTED_ROWS

):

    raise ValueError(

        "\nFinal retained + excluded ID counts do "
        "not equal the original dataset size."

    )


# ============================================================
# DISPLAY FINAL SCHEMA / MINIMISATION RESULTS
# ============================================================

print(
    f"\nFull cleaned columns: "
    f"{len(check_clean.columns):,}"
)


print(
    f"Primary sample columns: "
    f"{len(check_primary.columns):,}"
)


print(
    f"Forbidden columns in full cleaned file: "
    f"{len(forbidden_clean_columns_found_final):,}"
)


print(
    f"Forbidden columns in primary file: "
    f"{len(forbidden_primary_columns_found_final):,}"
)


print(
    f"Final retained IDs: "
    f"{final_retained_schema_count:,}"
)


print(
    f"Final excluded IDs: "
    f"{final_excluded_schema_count:,}"
)


print(
    f"Excluded IDs found in primary sample: "
    f"{len(excluded_in_primary_schema_final):,}"
)


print(
    "\nSection 7B Part 6F — "
    "output schema and data-minimisation validation: PASS"
)
# ============================================================
# SECTION 7B — FINAL RELOAD AND INTEGRITY VALIDATION
# PART 7 — FINAL STAGE 02 COMPLETION SUMMARY
# ============================================================


section("STAGE 02 — FINAL COMPLETION SUMMARY")


# ============================================================
# FINAL OUTPUT FILE EXISTENCE CHECK
# ============================================================

final_output_files = [

    CLEAN_FULL_FILE,
    PRIMARY_FILE,
    AUDIT_FILE,
    YEAR_ASSET_FILE,
    SUBREDDIT_FILE,
    DAILY_FILE,
    EXAMPLES_FILE,

]


missing_final_output_files = [

    str(file_path)

    for file_path in final_output_files

    if not file_path.exists()

]


if missing_final_output_files:

    raise FileNotFoundError(

        "\nOne or more required Stage 02 output files "
        "are missing:\n"
        + "\n".join(missing_final_output_files)

    )


# ============================================================
# FINAL OUTPUT FILE SIZE CHECK
# ============================================================

empty_final_output_files = [

    str(file_path)

    for file_path in final_output_files

    if file_path.stat().st_size == 0

]


if empty_final_output_files:

    raise ValueError(

        "\nOne or more required Stage 02 output files "
        "are empty:\n"
        + "\n".join(empty_final_output_files)

    )


# ============================================================
# FINAL CORE ROW-COUNT CHECKS
# ============================================================

if len(check_clean) != EXPECTED_ROWS:

    raise ValueError(

        "\nFinal full cleaned dataset row count "
        "does not match EXPECTED_ROWS.\n"
        f"Expected: {EXPECTED_ROWS:,}\n"
        f"Found:    {len(check_clean):,}"

    )


if len(check_primary) != retained_primary_count:

    raise ValueError(

        "\nFinal primary sample row count does not "
        "match retained_primary_count.\n"
        f"Expected: {retained_primary_count:,}\n"
        f"Found:    {len(check_primary):,}"

    )


if (

    retained_primary_count
    +
    excluded_primary_count

    !=

    EXPECTED_ROWS

):

    raise ValueError(

        "\nFinal retained + excluded counts do not "
        "equal the original Reddit dataset size."

    )


# ============================================================
# FINAL DAILY COVERAGE CHECK
# ============================================================

if len(check_daily) != 3652:

    raise ValueError(

        "\nFinal daily coverage file should contain "
        "3,652 BTC/ETH date-asset rows.\n"
        f"Found: {len(check_daily):,}"

    )


# ============================================================
# FINAL STUDY PERIOD CHECK
# ============================================================

final_clean_dates = pd.to_datetime(

    check_clean[
        "post_date"
    ],

    errors="raise",

)


final_earliest_date = final_clean_dates.min()

final_latest_observed_post_date = final_clean_dates.max()


if final_earliest_date != pd.Timestamp("2021-01-01"):

    raise ValueError(

        "\nUnexpected earliest Reddit post date.\n"
        f"Expected: 2021-01-01\n"
        f"Found:    {final_earliest_date.date()}"

    )


# The raw Reddit extraction ends on 2025-12-30.
# The complete daily coverage calendar itself continues
# through 2025-12-31.

if (
    final_latest_observed_post_date
    !=
    pd.Timestamp("2025-12-30")
):

    raise ValueError(

        "\nUnexpected latest observed Reddit post date.\n"
        f"Expected: 2025-12-30\n"
        f"Found:    "
        f"{final_latest_observed_post_date.date()}"

    )


# ============================================================
# FINAL ASSET CHECK
# ============================================================

final_assets = set(

    check_clean[
        "asset"
    ]
    .dropna()
    .astype(str)
    .unique()

)


if final_assets != {"BTC", "ETH"}:

    raise ValueError(

        "\nUnexpected assets in the final cleaned "
        "Reddit dataset.\n"
        f"Found: {sorted(final_assets)}"

    )


# ============================================================
# FINAL SUBREDDIT CHECK
# ============================================================

final_subreddits = set(

    check_clean[
        "subreddit"
    ]
    .dropna()
    .astype(str)
    .unique()

)


expected_final_subreddits = set(

    EXPECTED_SUBREDDIT_COUNTS.keys()

)


if final_subreddits != expected_final_subreddits:

    raise ValueError(

        "\nUnexpected subreddit coverage in the "
        "final cleaned Reddit dataset.\n"
        f"Expected: "
        f"{sorted(expected_final_subreddits)}\n"
        f"Found:    "
        f"{sorted(final_subreddits)}"

    )


# ============================================================
# FINAL PRIMARY SAMPLE ID CHECK
# ============================================================

final_retained_ids = set(

    check_clean.loc[

        check_clean[
            "retain_primary"
        ].eq(True),

        "post_id"

    ]
    .astype(str)

)


final_primary_ids = set(

    check_primary[
        "post_id"
    ]
    .astype(str)

)


if final_retained_ids != final_primary_ids:

    raise ValueError(

        "\nFinal primary sample IDs do not exactly "
        "match posts marked retain_primary=True."

    )


# ============================================================
# FINAL STAGE 02 RESULTS
# ============================================================

final_retained_percent = (

    retained_primary_count
    /
    EXPECTED_ROWS
    *
    100

)


final_excluded_percent = (

    excluded_primary_count
    /
    EXPECTED_ROWS
    *
    100

)


print(
    f"\nOriginal Reddit posts: "
    f"{EXPECTED_ROWS:,}"
)


print(
    f"Retained for primary sentiment sample: "
    f"{retained_primary_count:,} "
    f"({final_retained_percent:.2f}%)"
)


print(
    f"Excluded from primary sentiment sample: "
    f"{excluded_primary_count:,} "
    f"({final_excluded_percent:.2f}%)"
)


print(
    f"\nObserved Reddit post period: "
    f"{final_earliest_date.date()} "
    f"to "
    f"{final_latest_observed_post_date.date()}"
)


print(
    "\nDaily coverage calendar: "
    "2021-01-01 to 2025-12-31"
)


print(
    "\nAssets: BTC, ETH"
)


print(
    "Primary BTC subreddits: "
    "r/Bitcoin, r/BitcoinMarkets"
)


print(
    "Primary ETH subreddit: "
    "r/ethereum"
)


# ============================================================
# DISPLAY FINAL OUTPUT FILES
# ============================================================

print(
    "\nStage 02 output files:"
)


for file_path in final_output_files:

    print(
        f"  - {file_path.name}"
    )


# ============================================================
# METHODOLOGICAL BOUNDARY
# ============================================================

print(
    "\nCleaning/filtering only."
)


print(
    "No sentiment scoring has been performed."
)


print(
    "No daily sentiment aggregation has been performed."
)


print(
    "No cryptocurrency market data has been merged."
)


print(
    "No return calculation has been performed."
)


print(
    "No predictive modelling or significance testing "
    "has been performed."
)


# ============================================================
# FINAL SUCCESS MESSAGE
# ============================================================

print(
    "\n"
    + "=" * 78
)


print(
    "STAGE 02 — REDDIT CLEANING AND FILTERING: COMPLETE"
)


print(
    "=" * 78
)


print(
    "\nAll Stage 02 validation checks passed."
)


print(
    "\nNext step: inspect the cleaning audit, exclusion "
    "examples, and daily coverage before proceeding "
    "to sentiment analysis."
)