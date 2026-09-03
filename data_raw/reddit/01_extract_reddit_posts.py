"""
01_extract_reddit_posts.py

Extract the primary Reddit POST dataset for the dissertation.

Study period:
    2021-01-01 to 2025-12-31 inclusive

Primary communities:
    BTC:
        r/Bitcoin
        r/BitcoinMarkets

    ETH:
        r/ethereum

Source:
    Reddit for Researchers (RFR)
    Google BigQuery project:
        rddt-eng-rfrexternal1-prod

    Dataset:
        for_researchers_external

Authentication:
    Uses the already-authenticated Google BigQuery CLI (`bq`).

IMPORTANT:
    - POSTS only.
    - No comment text.
    - No accounts.
    - No author identifiers.
    - No mock dataset.
    - No API key.
    - No cleaning at this stage.
    - No sentiment scoring at this stage.

The script verifies the FINAL CSV on disk and will fail unless it contains:
    2021: 54,502
    2022: 25,197
    2023: 17,692
    2024: 22,418
    2025: 21,751

Total:
    141,560 posts
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd


# =====================================================================
# 1. CONFIGURATION
# =====================================================================

PROJECT_ID = "rddt-eng-rfrexternal1-prod"

DATASET = (
    "rddt-eng-rfrexternal1-prod."
    "for_researchers_external"
)

START_DATE = "2021-01-01"
END_DATE = "2025-12-31"

EXPECTED_YEARS = [
    2021,
    2022,
    2023,
    2024,
    2025,
]

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

EXPECTED_SUBREDDIT_YEAR_COUNTS = {
    2021: {
        "Bitcoin": 44_174,
        "BitcoinMarkets": 837,
        "ethereum": 9_491,
    },
    2022: {
        "Bitcoin": 19_538,
        "BitcoinMarkets": 453,
        "ethereum": 5_206,
    },
    2023: {
        "Bitcoin": 14_957,
        "BitcoinMarkets": 425,
        "ethereum": 2_310,
    },
    2024: {
        "Bitcoin": 20_201,
        "BitcoinMarkets": 493,
        "ethereum": 1_724,
    },
    2025: {
        "Bitcoin": 19_621,
        "BitcoinMarkets": 452,
        "ethereum": 1_678,
    },
}

EXPECTED_TOTAL_ROWS = 141_560

EXPECTED_COLUMNS = [
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


# =====================================================================
# 2. PATHS
# =====================================================================

SCRIPT_PATH = Path(__file__).resolve()

SCRIPT_DIR = SCRIPT_PATH.parent

OUTPUT_FILE = (
    SCRIPT_DIR
    / "reddit_posts_raw_minimized.csv"
)


# =====================================================================
# 3. DISPLAY HELPERS
# =====================================================================

def section(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def passed(message: str) -> None:
    print(f"PASS: {message}")


def fail(message: str) -> None:
    raise RuntimeError(message)


# =====================================================================
# 4. FIND BIGQUERY CLI
# =====================================================================

def find_bq() -> str:

    section("STEP 1: FIND BIGQUERY CLI")

    candidates = [
        shutil.which("bq"),
        "/opt/homebrew/bin/bq",
        "/usr/local/bin/bq",
    ]

    for candidate in candidates:

        if candidate and Path(candidate).exists():

            print(f"BigQuery CLI: {candidate}")

            result = subprocess.run(
                [
                    candidate,
                    "version",
                ],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                continue

            version_text = (
                result.stdout.strip()
                or result.stderr.strip()
            )

            print(version_text)

            passed(
                "BigQuery CLI is available."
            )

            return candidate

    fail(
        "Could not find a working BigQuery CLI.\n\n"
        "Run this in Terminal first:\n"
        "    bq version"
    )


# =====================================================================
# 5. TEST REDDIT FOR RESEARCHERS ACCESS
# =====================================================================

def test_access(
    bq_path: str,
) -> None:

    section(
        "STEP 2: TEST REDDIT FOR RESEARCHERS ACCESS"
    )

    sql = f"""
    SELECT
        COUNT(*) AS n
    FROM
        `{DATASET}.subreddits`
    """

    result = subprocess.run(
        [
            bq_path,
            "query",
            "--use_legacy_sql=false",
            f"--project_id={PROJECT_ID}",
            "--format=json",
            sql,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(result.stderr)

        fail(
            "Could not access the approved "
            "Reddit for Researchers dataset."
        )

    try:

        rows = json.loads(
            result.stdout
        )

    except json.JSONDecodeError as exc:

        fail(
            "Could not parse the BigQuery "
            f"access-test response: {exc}"
        )

    if not rows:
        fail(
            "BigQuery access test returned no result."
        )

    print(f"Project: {PROJECT_ID}")
    print(f"Dataset: {DATASET}")

    passed(
        "Authenticated BigQuery access works."
    )


# =====================================================================
# 6. CHECK SOURCE DATA BY YEAR BEFORE EXTRACTION
# =====================================================================

def verify_bigquery_source(
    bq_path: str,
) -> None:

    section(
        "STEP 3: VERIFY BIGQUERY SOURCE DATA 2021-2025"
    )

    sql = f"""
    SELECT
        EXTRACT(YEAR FROM p.created_at) AS year,
        s.name AS subreddit,
        COUNT(*) AS post_count
    FROM
        `{DATASET}.posts` AS p
    INNER JOIN
        `{DATASET}.subreddits` AS s
    ON
        p.subreddit_id = s.id
    WHERE
        DATE(p.created_at)
            BETWEEN DATE '{START_DATE}'
            AND DATE '{END_DATE}'
        AND LOWER(s.name) IN (
            'bitcoin',
            'bitcoinmarkets',
            'ethereum'
        )
    GROUP BY
        year,
        subreddit
    ORDER BY
        year,
        subreddit
    """

    result = subprocess.run(
        [
            bq_path,
            "query",
            "--use_legacy_sql=false",
            f"--project_id={PROJECT_ID}",
            "--format=json",
            sql,
        ],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        print(result.stderr)

        fail(
            "Could not obtain source coverage "
            "from BigQuery."
        )

    try:

        rows = json.loads(
            result.stdout
        )

    except json.JSONDecodeError as exc:

        fail(
            "Could not parse source coverage "
            f"response: {exc}"
        )

    if not rows:
        fail(
            "BigQuery source coverage query "
            "returned zero rows."
        )

    source = pd.DataFrame(
        rows
    )

    source["year"] = pd.to_numeric(
        source["year"],
        errors="raise",
    ).astype(int)

    source["post_count"] = pd.to_numeric(
        source["post_count"],
        errors="raise",
    ).astype(int)

    table = (
        source
        .pivot_table(
            index="year",
            columns="subreddit",
            values="post_count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(
            index=EXPECTED_YEARS,
            columns=[
                "Bitcoin",
                "BitcoinMarkets",
                "ethereum",
            ],
            fill_value=0,
        )
    )

    table["TOTAL"] = (
        table.sum(axis=1)
    )

    print()
    print(table.to_string())

    # -------------------------------------------------------------
    # Verify every expected cell
    # -------------------------------------------------------------

    for year in EXPECTED_YEARS:

        for subreddit in [
            "Bitcoin",
            "BitcoinMarkets",
            "ethereum",
        ]:

            actual = int(
                table.loc[
                    year,
                    subreddit,
                ]
            )

            expected = (
                EXPECTED_SUBREDDIT_YEAR_COUNTS
                [year]
                [subreddit]
            )

            if actual != expected:

                fail(
                    "BigQuery source count does not "
                    "match the previously verified "
                    "source data.\n\n"
                    f"Year: {year}\n"
                    f"Subreddit: {subreddit}\n"
                    f"Expected: {expected:,}\n"
                    f"Actual: {actual:,}"
                )

    # -------------------------------------------------------------
    # Verify yearly totals
    # -------------------------------------------------------------

    for year in EXPECTED_YEARS:

        actual = int(
            table.loc[
                year,
                "TOTAL",
            ]
        )

        expected = (
            EXPECTED_YEAR_COUNTS[year]
        )

        if actual != expected:

            fail(
                f"{year}: expected "
                f"{expected:,} posts but "
                f"BigQuery returned {actual:,}."
            )

    passed(
        "BigQuery contains the complete verified "
        "2021-2025 source sample."
    )


# =====================================================================
# 7. EXTRACTION SQL
# =====================================================================

def build_sql() -> str:

    return f"""
    SELECT
        p.id AS post_id,
        p.created_at,
        DATE(p.created_at) AS post_date,
        s.name AS subreddit,
        p.title,
        p.body,
        p.score,
        p.upvote_ratio,
        p.num_comments,
        p.crosspost_parent_id,
        p.url,
        p.permalink
    FROM
        `{DATASET}.posts` AS p
    INNER JOIN
        `{DATASET}.subreddits` AS s
    ON
        p.subreddit_id = s.id
    WHERE
        DATE(p.created_at)
            BETWEEN DATE '{START_DATE}'
            AND DATE '{END_DATE}'
        AND LOWER(s.name) IN (
            'bitcoin',
            'bitcoinmarkets',
            'ethereum'
        )
    ORDER BY
        p.created_at ASC,
        p.id ASC
    """


# =====================================================================
# 8. RUN BIGQUERY EXTRACTION
# =====================================================================

def extract_posts(
    bq_path: str,
) -> pd.DataFrame:

    section(
        "STEP 4: EXTRACT ALL POSTS FROM 2021-2025"
    )

    sql = build_sql()

    print(
        f"Requested period: {START_DATE} to {END_DATE}"
    )

    print(
        "BTC communities: r/Bitcoin + "
        "r/BitcoinMarkets"
    )

    print(
        "ETH community:   r/ethereum"
    )

    # -------------------------------------------------------------
    # Temporary JSON file
    # -------------------------------------------------------------

    temp = tempfile.NamedTemporaryFile(
        suffix=".json",
        prefix="reddit_rfr_",
        delete=False,
    )

    temp_path = Path(
        temp.name
    )

    temp.close()

    print()
    print(
        f"Temporary result file:\n{temp_path}"
    )

    try:

        # ---------------------------------------------------------
        # IMPORTANT:
        #
        # Explicit --max_rows prevents the CLI from presenting
        # only a small default result set.
        # ---------------------------------------------------------

        command = [
            bq_path,
            "query",
            "--use_legacy_sql=false",
            f"--project_id={PROJECT_ID}",
            "--format=json",
            "--max_rows=200000",
            sql,
        ]

        print()
        print(
            "Running full BigQuery extraction..."
        )

        with temp_path.open(
            "w",
            encoding="utf-8",
        ) as output:

            process = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.PIPE,
                text=True,
            )

        if process.returncode != 0:

            print()
            print(process.stderr)

            fail(
                "BigQuery extraction failed."
            )

        if not temp_path.exists():

            fail(
                "Temporary BigQuery result file "
                "was not created."
            )

        size_mb = (
            temp_path.stat().st_size
            / 1024**2
        )

        print(
            f"Temporary JSON size: {size_mb:.2f} MB"
        )

        print()
        print(
            "Reading complete BigQuery JSON..."
        )

        with temp_path.open(
            "r",
            encoding="utf-8",
        ) as handle:

            rows = json.load(
                handle
            )

        if not isinstance(
            rows,
            list,
        ):
            fail(
                "Unexpected BigQuery JSON structure."
            )

        df = pd.DataFrame(
            rows
        )

        print()
        print(
            f"Rows returned by BigQuery: "
            f"{len(df):,}"
        )

        print(
            f"Columns returned: "
            f"{len(df.columns):,}"
        )

        # ---------------------------------------------------------
        # This is a critical protection against truncated results.
        # ---------------------------------------------------------

        if len(df) != EXPECTED_TOTAL_ROWS:

            fail(
                "BIGQUERY EXTRACTION IS INCOMPLETE.\n\n"
                f"Expected: {EXPECTED_TOTAL_ROWS:,}\n"
                f"Received: {len(df):,}\n\n"
                "The CSV has NOT been accepted."
            )

        passed(
            "BigQuery returned all 141,560 posts."
        )

        return df

    finally:

        if temp_path.exists():

            temp_path.unlink()

            print()
            print(
                "Temporary BigQuery JSON removed."
            )


# =====================================================================
# 9. VALIDATE COLUMNS
# =====================================================================

def validate_columns(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 5: VALIDATE COLUMN STRUCTURE"
    )

    missing = [
        column
        for column in EXPECTED_COLUMNS
        if column not in df.columns
    ]

    extra = [
        column
        for column in df.columns
        if column not in EXPECTED_COLUMNS
    ]

    if missing:

        fail(
            "Missing columns:\n"
            + "\n".join(missing)
        )

    if extra:

        fail(
            "Unexpected columns:\n"
            + "\n".join(extra)
        )

    print(
        "Columns:"
    )

    for column in EXPECTED_COLUMNS:

        print(
            f"  {column}"
        )

    # -------------------------------------------------------------
    # Data minimization / privacy
    # -------------------------------------------------------------

    forbidden = [
        "author_id",
        "author",
        "username",
        "account_id",
    ]

    for column in df.columns:

        if column.lower() in forbidden:

            fail(
                "Author/account information was "
                f"unexpectedly extracted: {column}"
            )

    passed(
        "Minimized column structure verified."
    )

    passed(
        "No author/account identifiers extracted."
    )


# =====================================================================
# 10. PARSE AND VALIDATE DATES
# =====================================================================

def validate_dates(
    df: pd.DataFrame,
) -> pd.DataFrame:

    section(
        "STEP 6: VALIDATE EXTRACTED DATES"
    )

    df = df.copy()

    df["created_at"] = pd.to_datetime(
        df["created_at"],
        format="mixed",
        errors="coerce",
        utc=True,
    )

    df["post_date"] = pd.to_datetime(
        df["post_date"],
        errors="coerce",
    )

    invalid_timestamp = int(
        df["created_at"].isna().sum()
    )

    invalid_date = int(
        df["post_date"].isna().sum()
    )

    print(
        f"Invalid created_at: {invalid_timestamp:,}"
    )

    print(
        f"Invalid post_date:  {invalid_date:,}"
    )

    if invalid_timestamp != 0:

        fail(
            "Invalid created_at timestamps found."
        )

    if invalid_date != 0:

        fail(
            "Invalid post_date values found."
        )

    df["post_date"] = (
        df["post_date"]
        .dt.normalize()
    )

    earliest = (
        df["post_date"].min()
    )

    latest = (
        df["post_date"].max()
    )

    print()
    print(
        f"Earliest observation: {earliest.date()}"
    )

    print(
        f"Latest observation:   {latest.date()}"
    )

    start = pd.Timestamp(
        START_DATE
    )

    end = pd.Timestamp(
        END_DATE
    )

    outside = (
        (df["post_date"] < start)
        |
        (df["post_date"] > end)
    )

    if outside.any():

        fail(
            "Observations outside the dissertation "
            "study period were found."
        )

    # -------------------------------------------------------------
    # Confirm DATE(created_at) consistency
    # -------------------------------------------------------------

    timestamp_date = (
        df["created_at"]
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    mismatches = int(
        (
            timestamp_date
            != df["post_date"]
        ).sum()
    )

    print(
        f"created_at/post_date mismatches: "
        f"{mismatches:,}"
    )

    if mismatches != 0:

        fail(
            "post_date does not match the UTC "
            "created_at calendar date."
        )

    passed(
        "Date fields are valid."
    )

    return df


# =====================================================================
# 11. VERIFY ALL FIVE YEARS IN MEMORY
# =====================================================================

def verify_years(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 7: VERIFY ALL FIVE YEARS IN MEMORY"
    )

    year_counts = (
        df["post_date"]
        .dt.year
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "ROWS BY YEAR"
    )

    print(
        year_counts.to_string()
    )

    print()

    for year in EXPECTED_YEARS:

        actual = int(
            year_counts.get(
                year,
                0,
            )
        )

        expected = (
            EXPECTED_YEAR_COUNTS[year]
        )

        print(
            f"{year}: "
            f"{actual:,} "
            f"(expected {expected:,})"
        )

        if actual != expected:

            fail(
                f"Year {year} is incomplete.\n"
                f"Expected {expected:,}; "
                f"found {actual:,}."
            )

    unexpected_years = (
        set(
            year_counts.index.astype(int)
        )
        - set(EXPECTED_YEARS)
    )

    if unexpected_years:

        fail(
            "Unexpected years found: "
            f"{sorted(unexpected_years)}"
        )

    passed(
        "2021-2025 are all present."
    )


# =====================================================================
# 12. VERIFY SUBREDDIT COUNTS
# =====================================================================

def verify_subreddits(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 8: VERIFY SUBREDDIT COUNTS"
    )

    counts = (
        df["subreddit"]
        .value_counts()
    )

    print(
        counts.to_string()
    )

    print()

    for subreddit, expected in (
        EXPECTED_SUBREDDIT_COUNTS.items()
    ):

        actual = int(
            counts.get(
                subreddit,
                0,
            )
        )

        print(
            f"r/{subreddit}: "
            f"{actual:,}"
        )

        if actual != expected:

            fail(
                f"r/{subreddit}: expected "
                f"{expected:,}, found {actual:,}."
            )

    passed(
        "Subreddit counts match verified source."
    )


# =====================================================================
# 13. VERIFY SUBREDDIT × YEAR
# =====================================================================

def verify_subreddit_year(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 9: VERIFY SUBREDDIT × YEAR"
    )

    table = (
        df.groupby(
            [
                df["post_date"].dt.year,
                "subreddit",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=EXPECTED_YEARS,
            columns=[
                "Bitcoin",
                "BitcoinMarkets",
                "ethereum",
            ],
            fill_value=0,
        )
    )

    table["TOTAL"] = (
        table.sum(axis=1)
    )

    print()
    print(
        table.to_string()
    )

    print()

    for year in EXPECTED_YEARS:

        for subreddit in [
            "Bitcoin",
            "BitcoinMarkets",
            "ethereum",
        ]:

            expected = (
                EXPECTED_SUBREDDIT_YEAR_COUNTS
                [year]
                [subreddit]
            )

            actual = int(
                table.loc[
                    year,
                    subreddit,
                ]
            )

            if actual != expected:

                fail(
                    "Subreddit/year count mismatch:\n"
                    f"{year} / r/{subreddit}\n"
                    f"Expected: {expected:,}\n"
                    f"Actual: {actual:,}"
                )

    passed(
        "Every subreddit/year count matches "
        "the verified source."
    )


# =====================================================================
# 14. VERIFY POST IDs
# =====================================================================

def verify_ids(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 10: VERIFY POST IDS"
    )

    missing = int(
        df["post_id"].isna().sum()
    )

    duplicate = int(
        df["post_id"].duplicated().sum()
    )

    print(
        f"Missing post IDs:   {missing:,}"
    )

    print(
        f"Duplicate post IDs: {duplicate:,}"
    )

    if missing != 0:

        fail(
            "Missing post IDs found."
        )

    if duplicate != 0:

        fail(
            "Duplicate post IDs found."
        )

    passed(
        "All post IDs are present and unique."
    )


# =====================================================================
# 15. RAW-DATA DIAGNOSTICS
# =====================================================================

def diagnostics(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 11: RAW-DATA DIAGNOSTICS"
    )

    title = (
        df["title"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    body = (
        df["body"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    crosspost = (
        df["crosspost_parent_id"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    print(
        f"Missing title: "
        f"{int(df['title'].isna().sum()):,}"
    )

    print(
        f"Missing body: "
        f"{int(df['body'].isna().sum()):,}"
    )

    print(
        f"Empty title: "
        f"{int(title.eq('').sum()):,}"
    )

    print(
        f"Empty body: "
        f"{int(body.eq('').sum()):,}"
    )

    print(
        f"[deleted] title: "
        f"{int(title.str.lower().eq('[deleted]').sum()):,}"
    )

    print(
        f"[removed] title: "
        f"{int(title.str.lower().eq('[removed]').sum()):,}"
    )

    print(
        f"[deleted] body: "
        f"{int(body.str.lower().eq('[deleted]').sum()):,}"
    )

    print(
        f"[removed] body: "
        f"{int(body.str.lower().eq('[removed]').sum()):,}"
    )

    print(
        f"Crossposts: "
        f"{int(crosspost.ne('').sum()):,}"
    )

    print(
        f"Missing score: "
        f"{int(df['score'].isna().sum()):,}"
    )

    print(
        f"Missing upvote_ratio: "
        f"{int(df['upvote_ratio'].isna().sum()):,}"
    )

    print(
        f"Missing num_comments: "
        f"{int(df['num_comments'].isna().sum()):,}"
    )

    print(
        f"Missing URL: "
        f"{int(df['url'].isna().sum()):,}"
    )

    print(
        f"Missing permalink: "
        f"{int(df['permalink'].isna().sum()):,}"
    )

    print()
    print(
        "No observations are removed in Stage 01."
    )


# =====================================================================
# 16. SAVE CSV
# =====================================================================

def save_csv(
    df: pd.DataFrame,
) -> None:

    section(
        "STEP 12: SAVE COMPLETE RAW CSV"
    )

    output = df.copy()

    # -------------------------------------------------------------
    # Sort
    # -------------------------------------------------------------

    output = (
        output
        .sort_values(
            [
                "created_at",
                "post_id",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    # -------------------------------------------------------------
    # Stable timestamp/date representation
    # -------------------------------------------------------------

    output["created_at"] = (
        output["created_at"]
        .dt.strftime(
            "%Y-%m-%d %H:%M:%S.%f+00:00"
        )
    )

    output["post_date"] = (
        output["post_date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    # -------------------------------------------------------------
    # Exact column order
    # -------------------------------------------------------------

    output = output[
        EXPECTED_COLUMNS
    ]

    # -------------------------------------------------------------
    # Remove previous output before writing.
    #
    # This prevents accidentally inspecting an older version if
    # writing fails.
    # -------------------------------------------------------------

    if OUTPUT_FILE.exists():

        print(
            "Removing previous CSV:"
        )

        print(
            OUTPUT_FILE
        )

        OUTPUT_FILE.unlink()

    print()
    print(
        "Writing complete CSV..."
    )

    output.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8",
        lineterminator="\n",
    )

    if not OUTPUT_FILE.exists():

        fail(
            "Output CSV was not created."
        )

    file_size_mb = (
        OUTPUT_FILE.stat().st_size
        / (1024 ** 2)
    )

    print()
    print(
        f"Output path:\n{OUTPUT_FILE}"
    )

    print()
    print(
        f"Rows written: {len(output):,}"
    )

    print(
        f"Columns:      {len(output.columns):,}"
    )

    print(
        f"File size:    {file_size_mb:.2f} MB"
    )

    passed(
        "CSV writing completed."
    )


# =====================================================================
# 17. CRITICAL: REOPEN PHYSICAL CSV
# =====================================================================

def verify_saved_csv() -> None:

    section(
        "STEP 13: REOPEN AND VERIFY THE PHYSICAL CSV"
    )

    if not OUTPUT_FILE.exists():

        fail(
            "CSV does not exist."
        )

    print(
        "Reading the CSV back from:"
    )

    print(
        OUTPUT_FILE
    )

    saved = pd.read_csv(
        OUTPUT_FILE,
        low_memory=False,
    )

    print()
    print(
        f"Physical CSV rows: "
        f"{len(saved):,}"
    )

    print(
        f"Physical CSV columns: "
        f"{len(saved.columns):,}"
    )

    # -------------------------------------------------------------
    # Total rows
    # -------------------------------------------------------------

    if len(saved) != EXPECTED_TOTAL_ROWS:

        fail(
            "THE PHYSICAL CSV IS INCOMPLETE.\n\n"
            f"Expected rows: {EXPECTED_TOTAL_ROWS:,}\n"
            f"Actual rows:   {len(saved):,}"
        )

    # -------------------------------------------------------------
    # Columns
    # -------------------------------------------------------------

    if list(saved.columns) != EXPECTED_COLUMNS:

        fail(
            "Physical CSV columns do not match "
            "the expected structure."
        )

    # -------------------------------------------------------------
    # Dates
    # -------------------------------------------------------------

    dates = pd.to_datetime(
        saved["post_date"],
        errors="coerce",
    )

    if dates.isna().any():

        fail(
            "Physical CSV contains invalid dates."
        )

    print()
    print(
        f"FIRST DATE IN PHYSICAL CSV: "
        f"{dates.min().date()}"
    )

    print(
        f"LAST DATE IN PHYSICAL CSV:  "
        f"{dates.max().date()}"
    )

    # -------------------------------------------------------------
    # Annual counts
    # -------------------------------------------------------------

    year_counts = (
        dates
        .dt.year
        .value_counts()
        .sort_index()
    )

    print()
    print(
        "PHYSICAL CSV — ROWS BY YEAR"
    )

    print()
    print(
        year_counts.to_string()
    )

    print()

    for year in EXPECTED_YEARS:

        expected = (
            EXPECTED_YEAR_COUNTS[year]
        )

        actual = int(
            year_counts.get(
                year,
                0,
            )
        )

        if actual != expected:

            fail(
                "PHYSICAL CSV YEAR CHECK FAILED.\n\n"
                f"Year: {year}\n"
                f"Expected: {expected:,}\n"
                f"Actual: {actual:,}"
            )

        print(
            f"PASS: {year} = {actual:,}"
        )

    # -------------------------------------------------------------
    # Subreddit/year verification
    # -------------------------------------------------------------

    saved = saved.copy()

    saved["year"] = (
        dates.dt.year
    )

    table = (
        saved
        .groupby(
            [
                "year",
                "subreddit",
            ]
        )
        .size()
        .unstack(
            fill_value=0
        )
        .reindex(
            index=EXPECTED_YEARS,
            columns=[
                "Bitcoin",
                "BitcoinMarkets",
                "ethereum",
            ],
            fill_value=0,
        )
    )

    table["TOTAL"] = (
        table.sum(axis=1)
    )

    print()
    print(
        "PHYSICAL CSV — SUBREDDIT × YEAR"
    )

    print()
    print(
        table.to_string()
    )

    for year in EXPECTED_YEARS:

        for subreddit in [
            "Bitcoin",
            "BitcoinMarkets",
            "ethereum",
        ]:

            actual = int(
                table.loc[
                    year,
                    subreddit,
                ]
            )

            expected = (
                EXPECTED_SUBREDDIT_YEAR_COUNTS
                [year]
                [subreddit]
            )

            if actual != expected:

                fail(
                    "Physical CSV subreddit/year "
                    "verification failed:\n"
                    f"{year} / {subreddit}\n"
                    f"Expected {expected:,}; "
                    f"found {actual:,}."
                )

    # -------------------------------------------------------------
    # IDs
    # -------------------------------------------------------------

    missing_ids = int(
        saved["post_id"].isna().sum()
    )

    duplicate_ids = int(
        saved["post_id"].duplicated().sum()
    )

    if missing_ids != 0:

        fail(
            "Physical CSV contains missing post IDs."
        )

    if duplicate_ids != 0:

        fail(
            "Physical CSV contains duplicate post IDs."
        )

    passed(
        "Physical CSV contains all 141,560 posts."
    )

    passed(
        "Physical CSV contains all five years."
    )

    passed(
        "Physical CSV subreddit/year counts "
        "match BigQuery."
    )


# =====================================================================
# 18. FINAL SUMMARY
# =====================================================================

def final_summary() -> None:

    section(
        "STAGE 01 COMPLETE"
    )

    print(
        "COMPLETE REDDIT RAW DATASET CREATED"
    )

    print()
    print(
        f"File:\n{OUTPUT_FILE}"
    )

    print()
    print(
        "Study period requested:"
    )

    print(
        "2021-01-01 through 2025-12-31"
    )

    print()
    print(
        "Rows physically verified in CSV:"
    )

    print(
        f"{EXPECTED_TOTAL_ROWS:,}"
    )

    print()
    print(
        "Annual counts:"
    )

    for year in EXPECTED_YEARS:

        print(
            f"  {year}: "
            f"{EXPECTED_YEAR_COUNTS[year]:,}"
        )

    print()
    print(
        "Subreddit totals:"
    )

    for subreddit, count in (
        EXPECTED_SUBREDDIT_COUNTS.items()
    ):

        print(
            f"  r/{subreddit}: {count:,}"
        )

    print()
    print(
        "IMPORTANT:"
    )

    print(
        "The script has reopened the CSV from disk "
        "and verified these counts."
    )

    print()
    print(
        "Therefore, if another program displays "
        "only part of 2021, that program is showing "
        "a preview/truncated representation rather "
        "than the complete CSV verified here."
    )

    print()
    print(
        "Do not copy/paste the CSV contents to "
        "create another file."
    )

    print(
        "Use reddit_posts_raw_minimized.csv itself."
    )

    print()
    print(
        "NEXT:"
    )

    print(
        "data_clean/reddit/"
        "02_clean_reddit_posts.py"
    )

    print()
    print("=" * 80)


# =====================================================================
# 19. MAIN
# =====================================================================

def main() -> None:

    print()
    print("=" * 80)
    print(
        "REDDIT FOR RESEARCHERS"
    )
    print(
        "STAGE 01 — COMPLETE 2021-2025 POST EXTRACTION"
    )
    print("=" * 80)

    print()
    print(
        f"Script:\n{SCRIPT_PATH}"
    )

    print()
    print(
        f"Output:\n{OUTPUT_FILE}"
    )

    try:

        # 1
        bq_path = find_bq()

        # 2
        test_access(
            bq_path
        )

        # 3
        verify_bigquery_source(
            bq_path
        )

        # 4
        df = extract_posts(
            bq_path
        )

        # 5
        validate_columns(
            df
        )

        # 6
        df = validate_dates(
            df
        )

        # 7
        verify_years(
            df
        )

        # 8
        verify_subreddits(
            df
        )

        # 9
        verify_subreddit_year(
            df
        )

        # 10
        verify_ids(
            df
        )

        # 11
        diagnostics(
            df
        )

        # 12
        save_csv(
            df
        )

        # 13
        verify_saved_csv()

        # Finish
        final_summary()

    except Exception as exc:

        print()
        print("=" * 80)
        print(
            "STAGE 01 FAILED"
        )
        print("=" * 80)

        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )

        print()
        print(
            "The Reddit raw dataset should NOT "
            "be used for Stage 02 until this "
            "error is resolved."
        )

        sys.exit(1)


# =====================================================================
# 20. RUN
# =====================================================================

if __name__ == "__main__":
    main()