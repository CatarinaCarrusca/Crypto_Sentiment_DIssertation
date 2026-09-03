# ============================================================
# 03_validate_reddit_cleaning_coverage.py
# Stage 03: Reddit cleaning and coverage validation
# ============================================================

from pathlib import Path

import pandas as pd


# ============================================================
# SECTION 1 — PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

INPUT_DIR = (
    PROJECT_ROOT
    / "data_clean"
    / "reddit"
)

OUTPUT_DIR = (
    INPUT_DIR
    / "stage03_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# STAGE 02 INPUT FILES
# ============================================================

CLEAN_FULL_FILE = (
    INPUT_DIR
    / "reddit_posts_cleaned_full.csv"
)

PRIMARY_FILE = (
    INPUT_DIR
    / "reddit_posts_primary_sentiment_sample.csv"
)

AUDIT_FILE = (
    INPUT_DIR
    / "reddit_cleaning_audit.csv"
)

YEAR_ASSET_FILE = (
    INPUT_DIR
    / "reddit_cleaning_by_year_asset.csv"
)

SUBREDDIT_FILE = (
    INPUT_DIR
    / "reddit_cleaning_by_subreddit.csv"
)

DAILY_FILE = (
    INPUT_DIR
    / "reddit_daily_coverage_after_cleaning.csv"
)

EXAMPLES_FILE = (
    INPUT_DIR
    / "reddit_exclusion_examples.csv"
)


# ============================================================
# STAGE 03 OUTPUT FILES
# ============================================================

FLAG_SUMMARY_FILE = (
    OUTPUT_DIR
    / "reddit_exclusion_flag_summary.csv"
)

OVERLAP_MATRIX_FILE = (
    OUTPUT_DIR
    / "reddit_exclusion_overlap_matrix.csv"
)

FLAG_COUNT_DISTRIBUTION_FILE = (
    OUTPUT_DIR
    / "reddit_exclusion_flag_count_distribution.csv"
)

EXCLUSION_COMBINATIONS_FILE = (
    OUTPUT_DIR
    / "reddit_exclusion_reason_combinations.csv"
)

YEAR_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_validation_by_year.csv"
)

ASSET_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_validation_by_asset.csv"
)

YEAR_ASSET_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_validation_by_year_asset.csv"
)

SUBREDDIT_VALIDATION_FILE = (
    OUTPUT_DIR
    / "reddit_validation_by_subreddit.csv"
)

ZERO_COVERAGE_FILE = (
    OUTPUT_DIR
    / "reddit_zero_coverage_dates.csv"
)

TEMPORAL_REPRESENTATION_FILE = (
    OUTPUT_DIR
    / "reddit_temporal_representation.csv"
)

SUBREDDIT_REPRESENTATION_FILE = (
    OUTPUT_DIR
    / "reddit_subreddit_representation.csv"
)

ETH_2023_FILE = (
    OUTPUT_DIR
    / "reddit_eth_2023_excluded_posts.csv"
)

LINK_UNUSABLE_QC_FILE = (
    OUTPUT_DIR
    / "reddit_link_only_unusable_qc.csv"
)

COVERAGE_SUMMARY_FILE = (
    OUTPUT_DIR
    / "reddit_daily_coverage_summary.csv"
)

FINAL_QC_FILE = (
    OUTPUT_DIR
    / "reddit_stage03_final_qc_summary.csv"
)

DECISION_FILE = (
    OUTPUT_DIR
    / "reddit_stage03_methodological_decisions.csv"
)


# ============================================================
# EXPECTED VALUES FROM COMPLETED STAGE 02
# ============================================================

EXPECTED_INPUT_ROWS = 141_560
EXPECTED_RETAINED_ROWS = 136_019
EXPECTED_EXCLUDED_ROWS = 5_541

EXPECTED_CALENDAR_START = pd.Timestamp("2021-01-01")
EXPECTED_CALENDAR_END = pd.Timestamp("2025-12-31")

EXPECTED_CALENDAR_DAYS = 1_826
EXPECTED_DATE_ASSET_ROWS = 3_652

EXPECTED_ASSETS = {
    "BTC",
    "ETH",
}

EXPECTED_SUBREDDITS = {
    "Bitcoin",
    "BitcoinMarkets",
    "ethereum",
}

EXPECTED_REVIEW_CATEGORIES = {
    "unusable_text",
    "crosspost",
    "promotional",
    "recurring_template",
    "exact_duplicate",
    "link_only",
    "repeated_text_5plus_diagnostic",
}


PRIMARY_FLAGS = [
    "flag_unusable_text",
    "flag_crosspost",
    "flag_promotional",
    "flag_template",
    "flag_exact_duplicate",
    "flag_link_only",
]


# ============================================================
# BASIC PRINTING FUNCTION
# ============================================================

def section(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ============================================================
# SECTION 2 — CHECK AND LOAD ALL STAGE 02 FILES
# ============================================================

section("LOADING STAGE 02 REDDIT OUTPUTS")


stage02_files = [
    CLEAN_FULL_FILE,
    PRIMARY_FILE,
    AUDIT_FILE,
    YEAR_ASSET_FILE,
    SUBREDDIT_FILE,
    DAILY_FILE,
    EXAMPLES_FILE,
]


for file_path in stage02_files:

    if not file_path.exists():

        raise FileNotFoundError(
            "\nRequired Stage 02 file not found:\n"
            f"{file_path}"
        )


full = pd.read_csv(
    CLEAN_FULL_FILE,
    low_memory=False
)

primary = pd.read_csv(
    PRIMARY_FILE,
    low_memory=False
)

audit = pd.read_csv(
    AUDIT_FILE
)

year_asset_stage02 = pd.read_csv(
    YEAR_ASSET_FILE
)

subreddit_stage02 = pd.read_csv(
    SUBREDDIT_FILE
)

daily = pd.read_csv(
    DAILY_FILE
)

examples = pd.read_csv(
    EXAMPLES_FILE,
    low_memory=False
)


print("\nAll seven Stage 02 files loaded successfully.")

print(
    f"Full cleaned rows: {len(full):,}"
)

print(
    f"Primary sample rows: {len(primary):,}"
)

print(
    f"Audit rows: {len(audit):,}"
)

print(
    f"Year/asset summary rows: {len(year_asset_stage02):,}"
)

print(
    f"Subreddit summary rows: {len(subreddit_stage02):,}"
)

print(
    f"Daily coverage rows: {len(daily):,}"
)

print(
    f"Manual-review rows: {len(examples):,}"
)


# ============================================================
# SECTION 3 — REQUIRED COLUMN VALIDATION
# ============================================================

section("VALIDATING REQUIRED STAGE 02 COLUMNS")


required_full_columns = {
    "post_id",
    "post_date",
    "year",
    "subreddit",
    "asset",
    "analysis_text",
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
}


missing_full_columns = (
    required_full_columns
    - set(full.columns)
)


if missing_full_columns:

    raise ValueError(
        "\nMissing required columns in full cleaned dataset:\n"
        f"{sorted(missing_full_columns)}"
    )


required_daily_columns = {
    "post_date",
    "asset",
    "raw_post_count",
    "retained_post_count",
    "excluded_post_count",
}


missing_daily_columns = (
    required_daily_columns
    - set(daily.columns)
)


if missing_daily_columns:

    raise ValueError(
        "\nMissing required columns in daily coverage file:\n"
        f"{sorted(missing_daily_columns)}"
    )


print("\nRequired Stage 02 columns: PASS")


# ============================================================
# SECTION 4 — PARSE DATES
# ============================================================

section("PARSING VALIDATION DATES")


full["post_date"] = pd.to_datetime(
    full["post_date"],
    errors="raise"
)

primary["post_date"] = pd.to_datetime(
    primary["post_date"],
    errors="raise"
)

daily["post_date"] = pd.to_datetime(
    daily["post_date"],
    errors="raise"
)


print("\nDate parsing: PASS")


# ============================================================
# SECTION 5 — BASIC DATASET ACCOUNTING
# ============================================================

section("VALIDATING STAGE 02 DATASET ACCOUNTING")


if len(full) != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nUnexpected full cleaned row count.\n"
        f"Expected: {EXPECTED_INPUT_ROWS:,}\n"
        f"Found:    {len(full):,}"
    )


if len(primary) != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nUnexpected primary sample row count.\n"
        f"Expected: {EXPECTED_RETAINED_ROWS:,}\n"
        f"Found:    {len(primary):,}"
    )


retained_count = int(
    full["retain_primary"].sum()
)

excluded_count = int(
    full["exclude_primary"].sum()
)


if retained_count != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nUnexpected retained count.\n"
        f"Expected: {EXPECTED_RETAINED_ROWS:,}\n"
        f"Found:    {retained_count:,}"
    )


if excluded_count != EXPECTED_EXCLUDED_ROWS:

    raise ValueError(
        "\nUnexpected excluded count.\n"
        f"Expected: {EXPECTED_EXCLUDED_ROWS:,}\n"
        f"Found:    {excluded_count:,}"
    )


if (
    retained_count
    + excluded_count
    != EXPECTED_INPUT_ROWS
):

    raise ValueError(
        "\nRetained + excluded does not equal input."
    )


invalid_decisions = int(
    (
        full["retain_primary"]
        == full["exclude_primary"]
    ).sum()
)


if invalid_decisions != 0:

    raise ValueError(
        "\nInvalid retain/exclude decisions found."
    )


print(
    f"\nInput posts: {len(full):,}"
)

print(
    f"Retained posts: {retained_count:,}"
)

print(
    f"Excluded posts: {excluded_count:,}"
)

print(
    f"Invalid retain/exclude decisions: {invalid_decisions:,}"
)

print(
    "\nDataset accounting: PASS"
)


# ============================================================
# SECTION 6 — POST-ID AND PRIMARY SAMPLE INTEGRITY
# ============================================================

section("VALIDATING POST IDS AND PRIMARY SAMPLE")


full_missing_ids = int(
    full["post_id"]
    .isna()
    .sum()
)

full_duplicate_ids = int(
    full["post_id"]
    .duplicated()
    .sum()
)

primary_missing_ids = int(
    primary["post_id"]
    .isna()
    .sum()
)

primary_duplicate_ids = int(
    primary["post_id"]
    .duplicated()
    .sum()
)


if full_missing_ids != 0:

    raise ValueError(
        "\nMissing post IDs in full cleaned dataset."
    )


if full_duplicate_ids != 0:

    raise ValueError(
        "\nDuplicate post IDs in full cleaned dataset."
    )


if primary_missing_ids != 0:

    raise ValueError(
        "\nMissing post IDs in primary sample."
    )


if primary_duplicate_ids != 0:

    raise ValueError(
        "\nDuplicate post IDs in primary sample."
    )


retained_ids = set(
    full.loc[
        full["retain_primary"],
        "post_id"
    ]
)

excluded_ids = set(
    full.loc[
        full["exclude_primary"],
        "post_id"
    ]
)

primary_ids = set(
    primary["post_id"]
)


if retained_ids != primary_ids:

    raise ValueError(
        "\nPrimary sample does not exactly match "
        "retain_primary=True posts."
    )


if excluded_ids & primary_ids:

    raise ValueError(
        "\nExcluded posts were found in the primary sample."
    )


print(
    f"\nFull missing post IDs: {full_missing_ids:,}"
)

print(
    f"Full duplicate post IDs: {full_duplicate_ids:,}"
)

print(
    f"Primary missing post IDs: {primary_missing_ids:,}"
)

print(
    f"Primary duplicate post IDs: {primary_duplicate_ids:,}"
)

print(
    "Primary sample exactly matches retained posts: PASS"
)


# ============================================================
# SECTION 7 — RECONCILE WITH STAGE 02 AUDIT
# ============================================================

section("RECONCILING WITH STAGE 02 CLEANING AUDIT")


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
    - actual_audit_measures
)


if missing_audit_measures:

    raise ValueError(
        "\nRequired Stage 02 audit measures missing:\n"
        f"{sorted(missing_audit_measures)}"
    )


audit_input = int(
    audit.loc[
        audit["measure"] == "Input posts",
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

audit_retained = int(
    audit.loc[
        audit["measure"]
        == "Retained for primary sample",
        "count"
    ].iloc[0]
)


if audit_input != len(full):

    raise ValueError(
        "\nAudit input count does not match full dataset."
    )


if audit_retained != retained_count:

    raise ValueError(
        "\nAudit retained count does not match full dataset."
    )


if audit_excluded != excluded_count:

    raise ValueError(
        "\nAudit excluded count does not match full dataset."
    )


if (
    audit_retained
    + audit_excluded
    != audit_input
):

    raise ValueError(
        "\nStage 02 audit does not reconcile."
    )


print(
    f"\nAudit input posts: {audit_input:,}"
)

print(
    f"Audit retained posts: {audit_retained:,}"
)

print(
    f"Audit excluded posts: {audit_excluded:,}"
)

print(
    "\nStage 02 audit reconciliation: PASS"
)


# ============================================================
# SECTION 8 — MANUAL-REVIEW EXAMPLE COVERAGE
# ============================================================

section("VALIDATING MANUAL-REVIEW EXAMPLES")


if "review_category" not in examples.columns:

    raise ValueError(
        "\nreview_category missing from examples file."
    )


actual_review_categories = set(
    examples[
        "review_category"
    ]
    .dropna()
    .unique()
)


if actual_review_categories != EXPECTED_REVIEW_CATEGORIES:

    raise ValueError(
        "\nUnexpected manual-review categories.\n"
        f"Expected: {sorted(EXPECTED_REVIEW_CATEGORIES)}\n"
        f"Found:    {sorted(actual_review_categories)}"
    )


example_counts = (
    examples[
        "review_category"
    ]
    .value_counts()
    .sort_index()
)


print(
    "\nExamples per review category:"
)

print(
    example_counts.to_string()
)


if (example_counts < 25).any():

    raise ValueError(
        "\nAt least one review category contains "
        "fewer than 25 examples."
    )


unknown_example_ids = (
    set(examples["post_id"])
    - set(full["post_id"])
)


if unknown_example_ids:

    raise ValueError(
        "\nManual-review file contains unknown post IDs."
    )


print(
    f"\nReview categories represented: "
    f"{len(actual_review_categories)}"
)

print(
    f"Manual-review rows: {len(examples):,}"
)

print(
    "Manual-review coverage and IDs: PASS"
)


# ============================================================
# SECTION 9 — PRIMARY EXCLUSION FLAG SUMMARY
# ============================================================

section("ANALYSING PRIMARY EXCLUSION FLAGS")


for flag in PRIMARY_FLAGS:

    if flag not in full.columns:

        raise ValueError(
            f"\nMissing primary exclusion flag: {flag}"
        )


flag_summary_rows = []


for flag in PRIMARY_FLAGS:

    count = int(
        full[flag].sum()
    )

    flag_summary_rows.append(
        {
            "flag": flag,
            "count": count,
            "percent_of_input": round(
                count
                / len(full)
                * 100,
                4
            ),
        }
    )


flag_summary = pd.DataFrame(
    flag_summary_rows
)


print(
    "\nPrimary exclusion flag summary:"
)

print(
    flag_summary.to_string(
        index=False
    )
)


# ============================================================
# SECTION 10 — NUMBER OF FLAGS PER POST
# ============================================================

section("ANALYSING MULTIPLE EXCLUSION FLAGS")


full["stage03_primary_flag_count"] = (
    full[PRIMARY_FLAGS]
    .sum(axis=1)
    .astype(int)
)


flag_count_distribution = (
    full[
        "stage03_primary_flag_count"
    ]
    .value_counts()
    .sort_index()
    .rename_axis(
        "number_of_primary_flags"
    )
    .reset_index(
        name="post_count"
    )
)


flag_count_distribution[
    "percent_of_input"
] = (
    flag_count_distribution[
        "post_count"
    ]
    / len(full)
    * 100
).round(4)


flag_count_distribution[
    "percent_of_excluded"
] = 0.0


excluded_mask = (
    flag_count_distribution[
        "number_of_primary_flags"
    ] > 0
)


flag_count_distribution.loc[
    excluded_mask,
    "percent_of_excluded"
] = (
    flag_count_distribution.loc[
        excluded_mask,
        "post_count"
    ]
    / excluded_count
    * 100
).round(4)


print(
    "\nNumber of primary exclusion flags per post:"
)

print(
    flag_count_distribution.to_string(
        index=False
    )
)


excluded_without_flag = int(
    (
        full["exclude_primary"]
        & full[
            "stage03_primary_flag_count"
        ].eq(0)
    ).sum()
)


retained_with_flag = int(
    (
        full["retain_primary"]
        & full[
            "stage03_primary_flag_count"
        ].gt(0)
    ).sum()
)


if excluded_without_flag != 0:

    raise ValueError(
        "\nExcluded posts without primary flags found."
    )


if retained_with_flag != 0:

    raise ValueError(
        "\nRetained posts with primary flags found."
    )


excluded_one_flag = int(
    (
        full["exclude_primary"]
        & full[
            "stage03_primary_flag_count"
        ].eq(1)
    ).sum()
)

excluded_two_flags = int(
    (
        full["exclude_primary"]
        & full[
            "stage03_primary_flag_count"
        ].eq(2)
    ).sum()
)

excluded_three_plus = int(
    (
        full["exclude_primary"]
        & full[
            "stage03_primary_flag_count"
        ].ge(3)
    ).sum()
)


print(
    "\nExcluded posts by number of overlapping rules:"
)

print(
    f"Exactly 1 flag: {excluded_one_flag:,}"
)

print(
    f"Exactly 2 flags: {excluded_two_flags:,}"
)

print(
    f"3 or more flags: {excluded_three_plus:,}"
)


if (
    excluded_one_flag
    + excluded_two_flags
    + excluded_three_plus
    != excluded_count
):

    raise ValueError(
        "\nMulti-flag exclusion accounting does not reconcile."
    )


print(
    "\nMulti-flag exclusion accounting: PASS"
)


# ============================================================
# SECTION 11 — PAIRWISE FLAG OVERLAP MATRIX
# ============================================================

section("CREATING PAIRWISE EXCLUSION OVERLAP MATRIX")


overlap_matrix = pd.DataFrame(
    index=PRIMARY_FLAGS,
    columns=PRIMARY_FLAGS,
    dtype=int
)


for flag_a in PRIMARY_FLAGS:

    for flag_b in PRIMARY_FLAGS:

        overlap_matrix.loc[
            flag_a,
            flag_b
        ] = int(
            (
                full[flag_a]
                & full[flag_b]
            ).sum()
        )


overlap_matrix = (
    overlap_matrix
    .astype(int)
)


print(
    "\nPairwise exclusion overlap matrix:"
)

print(
    overlap_matrix.to_string()
)


# ============================================================
# SECTION 12 — LINK-ONLY / UNUSABLE-TEXT VALIDATION
# ============================================================

section("VALIDATING LINK-ONLY AND UNUSABLE-TEXT OVERLAP")


link_only_count = int(
    full[
        "flag_link_only"
    ].sum()
)

unusable_count = int(
    full[
        "flag_unusable_text"
    ].sum()
)

both_count = int(
    (
        full["flag_link_only"]
        & full["flag_unusable_text"]
    ).sum()
)

link_only_not_unusable = int(
    (
        full["flag_link_only"]
        & ~full["flag_unusable_text"]
    ).sum()
)

unusable_not_link_only = int(
    (
        full["flag_unusable_text"]
        & ~full["flag_link_only"]
    ).sum()
)


link_unusable_qc = pd.DataFrame(
    [
        {
            "measure": "Link-only posts",
            "count": link_only_count,
        },
        {
            "measure": "Unusable-text posts",
            "count": unusable_count,
        },
        {
            "measure": "Both link-only and unusable",
            "count": both_count,
        },
        {
            "measure": "Link-only but not unusable",
            "count": link_only_not_unusable,
        },
        {
            "measure": "Unusable but not link-only",
            "count": unusable_not_link_only,
        },
    ]
)


print(
    "\nLink-only / unusable-text QC:"
)

print(
    link_unusable_qc.to_string(
        index=False
    )
)


# ============================================================
# DOCUMENT RELATIONSHIP
# ============================================================

if both_count == link_only_count:

    print(
        "\nAll link-only posts are also captured by "
        "the unusable-text rule."
    )

else:

    print(
        "\nLink-only and unusable-text flags "
        "partially overlap."
    )


print(
    "Link-only/unusable relationship documented: PASS"
)


# ============================================================
# SECTION 13 — EXCLUSION REASON COMBINATIONS
# ============================================================

section("ANALYSING EXCLUSION-REASON COMBINATIONS")


excluded_posts = full.loc[
    full["exclude_primary"]
].copy()


reason_combinations = (
    excluded_posts[
        "exclusion_reason"
    ]
    .value_counts()
    .rename_axis(
        "exclusion_reason"
    )
    .reset_index(
        name="post_count"
    )
)


reason_combinations[
    "percent_of_excluded"
] = (
    reason_combinations[
        "post_count"
    ]
    / excluded_count
    * 100
).round(4)


if int(
    reason_combinations[
        "post_count"
    ].sum()
) != excluded_count:

    raise ValueError(
        "\nExclusion-reason combinations do not "
        "sum to total excluded posts."
    )


print(
    "\nExclusion-reason combinations:"
)

print(
    reason_combinations.to_string(
        index=False
    )
)


print(
    "\nExclusion-reason accounting: PASS"
)


# ============================================================
# SECTION 14 — VALIDATION BY ASSET
# ============================================================

section("VALIDATING CLEANING BY ASSET")


asset_validation = (

    full.groupby(
        "asset",
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

    )

    .reset_index()

)


asset_validation[
    "retained_percent"
] = (
    asset_validation[
        "retained_posts"
    ]
    / asset_validation[
        "input_posts"
    ]
    * 100
).round(4)


asset_validation[
    "excluded_percent"
] = (
    asset_validation[
        "excluded_posts"
    ]
    / asset_validation[
        "input_posts"
    ]
    * 100
).round(4)


if set(
    asset_validation[
        "asset"
    ]
) != EXPECTED_ASSETS:

    raise ValueError(
        "\nUnexpected assets found."
    )


if int(
    asset_validation[
        "input_posts"
    ].sum()
) != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nAsset input totals do not reconcile."
    )


if int(
    asset_validation[
        "retained_posts"
    ].sum()
) != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nAsset retained totals do not reconcile."
    )


if int(
    asset_validation[
        "excluded_posts"
    ].sum()
) != EXPECTED_EXCLUDED_ROWS:

    raise ValueError(
        "\nAsset excluded totals do not reconcile."
    )


print(
    "\nCleaning validation by asset:"
)

print(
    asset_validation.to_string(
        index=False
    )
)


print(
    "\nAsset-level validation: PASS"
)


# ============================================================
# SECTION 15 — VALIDATION BY YEAR
# ============================================================

section("VALIDATING CLEANING BY YEAR")


year_validation = (

    full.groupby(
        "year",
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

    )

    .reset_index()

)


year_validation[
    "retained_percent"
] = (
    year_validation[
        "retained_posts"
    ]
    / year_validation[
        "input_posts"
    ]
    * 100
).round(4)


year_validation[
    "excluded_percent"
] = (
    year_validation[
        "excluded_posts"
    ]
    / year_validation[
        "input_posts"
    ]
    * 100
).round(4)


if set(
    year_validation[
        "year"
    ]
) != {
    2021,
    2022,
    2023,
    2024,
    2025,
}:

    raise ValueError(
        "\nUnexpected year composition."
    )


if int(
    year_validation[
        "input_posts"
    ].sum()
) != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nYear input totals do not reconcile."
    )


print(
    "\nCleaning validation by year:"
)

print(
    year_validation.to_string(
        index=False
    )
)


print(
    "\nYear-level validation: PASS"
)


# ============================================================
# SECTION 16 — VALIDATION BY YEAR AND ASSET
# ============================================================

section("VALIDATING CLEANING BY YEAR AND ASSET")


year_asset_validation = (

    full.groupby(
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

    )

    .reset_index()

)


year_asset_validation[
    "retained_percent"
] = (
    year_asset_validation[
        "retained_posts"
    ]
    / year_asset_validation[
        "input_posts"
    ]
    * 100
).round(4)


year_asset_validation[
    "excluded_percent"
] = (
    year_asset_validation[
        "excluded_posts"
    ]
    / year_asset_validation[
        "input_posts"
    ]
    * 100
).round(4)


year_asset_validation = (
    year_asset_validation
    .sort_values(
        [
            "year",
            "asset",
        ]
    )
    .reset_index(drop=True)
)


if len(
    year_asset_validation
) != 10:

    raise ValueError(
        "\nExpected exactly 10 year-asset groups."
    )


if int(
    year_asset_validation[
        "input_posts"
    ].sum()
) != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nYear-asset input totals do not reconcile."
    )


if int(
    year_asset_validation[
        "retained_posts"
    ].sum()
) != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nYear-asset retained totals do not reconcile."
    )


if int(
    year_asset_validation[
        "excluded_posts"
    ].sum()
) != EXPECTED_EXCLUDED_ROWS:

    raise ValueError(
        "\nYear-asset excluded totals do not reconcile."
    )


print(
    "\nCleaning validation by year and asset:"
)

print(
    year_asset_validation.to_string(
        index=False
    )
)


# ============================================================
# RECONCILE AGAINST STAGE 02 YEAR-ASSET TABLE
# ============================================================

year_asset_compare_columns = [
    "year",
    "asset",
    "input_posts",
    "retained_posts",
    "excluded_posts",
]


stage02_year_asset_compare = (
    year_asset_stage02[
        year_asset_compare_columns
    ]
    .sort_values(
        [
            "year",
            "asset",
        ]
    )
    .reset_index(drop=True)
)


stage03_year_asset_compare = (
    year_asset_validation[
        year_asset_compare_columns
    ]
    .sort_values(
        [
            "year",
            "asset",
        ]
    )
    .reset_index(drop=True)
)


if not stage02_year_asset_compare.equals(
    stage03_year_asset_compare
):

    raise ValueError(
        "\nStage 03 year-asset results do not "
        "match Stage 02."
    )


print(
    "\nStage 02/Stage 03 year-asset reconciliation: PASS"
)


# ============================================================
# SECTION 17 — VALIDATION BY SUBREDDIT
# ============================================================

section("VALIDATING CLEANING BY SUBREDDIT")


subreddit_validation = (

    full.groupby(
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

    )

    .reset_index()

)


subreddit_validation[
    "retained_percent"
] = (
    subreddit_validation[
        "retained_posts"
    ]
    / subreddit_validation[
        "input_posts"
    ]
    * 100
).round(4)


subreddit_validation[
    "excluded_percent"
] = (
    subreddit_validation[
        "excluded_posts"
    ]
    / subreddit_validation[
        "input_posts"
    ]
    * 100
).round(4)


if set(
    subreddit_validation[
        "subreddit"
    ]
) != EXPECTED_SUBREDDITS:

    raise ValueError(
        "\nUnexpected subreddit composition."
    )


if int(
    subreddit_validation[
        "input_posts"
    ].sum()
) != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nSubreddit input totals do not reconcile."
    )


if int(
    subreddit_validation[
        "retained_posts"
    ].sum()
) != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nSubreddit retained totals do not reconcile."
    )


if int(
    subreddit_validation[
        "excluded_posts"
    ].sum()
) != EXPECTED_EXCLUDED_ROWS:

    raise ValueError(
        "\nSubreddit excluded totals do not reconcile."
    )


print(
    "\nCleaning validation by subreddit:"
)

print(
    subreddit_validation.to_string(
        index=False
    )
)


# ============================================================
# RECONCILE AGAINST STAGE 02 SUBREDDIT TABLE
# ============================================================

subreddit_compare_columns = [
    "subreddit",
    "input_posts",
    "retained_posts",
    "excluded_posts",
]


stage02_subreddit_compare = (
    subreddit_stage02[
        subreddit_compare_columns
    ]
    .sort_values(
        "subreddit"
    )
    .reset_index(drop=True)
)


stage03_subreddit_compare = (
    subreddit_validation[
        subreddit_compare_columns
    ]
    .sort_values(
        "subreddit"
    )
    .reset_index(drop=True)
)


if not stage02_subreddit_compare.equals(
    stage03_subreddit_compare
):

    raise ValueError(
        "\nStage 03 subreddit results do not "
        "match Stage 02."
    )


print(
    "\nStage 02/Stage 03 subreddit reconciliation: PASS"
)


# ============================================================
# SECTION 18 — INVESTIGATE ETH 2023
# ============================================================

section("INVESTIGATING ETH 2023")


eth_2023 = full.loc[
    (full["year"] == 2023)
    & (full["asset"] == "ETH")
].copy()


eth_2023_excluded = eth_2023.loc[
    eth_2023[
        "exclude_primary"
    ]
].copy()


eth_2023_flag_counts = {
    flag: int(
        eth_2023_excluded[
            flag
        ].sum()
    )
    for flag in PRIMARY_FLAGS
}


print(
    f"\nETH 2023 total posts: "
    f"{len(eth_2023):,}"
)

print(
    f"ETH 2023 retained posts: "
    f"{int(eth_2023['retain_primary'].sum()):,}"
)

print(
    f"ETH 2023 excluded posts: "
    f"{len(eth_2023_excluded):,}"
)

print(
    "ETH 2023 exclusion rate: "
    f"{len(eth_2023_excluded) / len(eth_2023) * 100:.4f}%"
)


print(
    "\nETH 2023 exclusion flags:"
)


for flag, count in eth_2023_flag_counts.items():

    print(
        f"{flag}: {count:,}"
    )


# ============================================================
# CONFIRM KNOWN ETH 2023 RESULTS
# ============================================================

if len(eth_2023) != 2_310:

    raise ValueError(
        "\nUnexpected ETH 2023 input count."
    )


if int(
    eth_2023[
        "retain_primary"
    ].sum()
) != 2_052:

    raise ValueError(
        "\nUnexpected ETH 2023 retained count."
    )


if len(
    eth_2023_excluded
) != 258:

    raise ValueError(
        "\nUnexpected ETH 2023 excluded count."
    )


if eth_2023_flag_counts[
    "flag_crosspost"
] != 125:

    raise ValueError(
        "\nUnexpected ETH 2023 crosspost count."
    )


if eth_2023_flag_counts[
    "flag_exact_duplicate"
] != 133:

    raise ValueError(
        "\nUnexpected ETH 2023 exact-duplicate count."
    )


if (
    eth_2023_flag_counts[
        "flag_unusable_text"
    ] != 0
    or eth_2023_flag_counts[
        "flag_promotional"
    ] != 0
    or eth_2023_flag_counts[
        "flag_template"
    ] != 0
    or eth_2023_flag_counts[
        "flag_link_only"
    ] != 0
):

    raise ValueError(
        "\nUnexpected additional ETH 2023 "
        "exclusion-rule counts."
    )


print(
    "\nETH 2023 higher exclusion rate is accounted "
    "for by crossposts and exact duplicates."
)

print(
    "ETH 2023 investigation: PASS"
)


# ============================================================
# SECTION 19 — DAILY COVERAGE STRUCTURE
# ============================================================

section("VALIDATING DAILY BTC/ETH COVERAGE")


if len(daily) != EXPECTED_DATE_ASSET_ROWS:

    raise ValueError(
        "\nUnexpected number of daily date-asset rows.\n"
        f"Expected: {EXPECTED_DATE_ASSET_ROWS:,}\n"
        f"Found:    {len(daily):,}"
    )


calendar_start = (
    daily[
        "post_date"
    ].min()
)

calendar_end = (
    daily[
        "post_date"
    ].max()
)

calendar_days = int(
    daily[
        "post_date"
    ].nunique()
)


if calendar_start != EXPECTED_CALENDAR_START:

    raise ValueError(
        "\nUnexpected calendar start date."
    )


if calendar_end != EXPECTED_CALENDAR_END:

    raise ValueError(
        "\nUnexpected calendar end date."
    )


if calendar_days != EXPECTED_CALENDAR_DAYS:

    raise ValueError(
        "\nUnexpected number of calendar days."
    )


if set(
    daily["asset"]
    .unique()
) != EXPECTED_ASSETS:

    raise ValueError(
        "\nUnexpected assets in daily coverage."
    )


duplicate_date_asset = int(
    daily[
        [
            "post_date",
            "asset",
        ]
    ]
    .duplicated()
    .sum()
)


if duplicate_date_asset != 0:

    raise ValueError(
        "\nDuplicate date-asset rows found."
    )


print(
    f"\nCalendar start: {calendar_start.date()}"
)

print(
    f"Calendar end: {calendar_end.date()}"
)

print(
    f"Calendar days: {calendar_days:,}"
)

print(
    f"Date-asset rows: {len(daily):,}"
)

print(
    f"Duplicate date-asset rows: "
    f"{duplicate_date_asset:,}"
)


# ============================================================
# EXACT CALENDAR CONTINUITY
# ============================================================

expected_dates = pd.date_range(
    start=EXPECTED_CALENDAR_START,
    end=EXPECTED_CALENDAR_END,
    freq="D"
)


actual_dates = pd.DatetimeIndex(
    sorted(
        daily[
            "post_date"
        ].unique()
    )
)


missing_dates = (
    expected_dates
    .difference(
        actual_dates
    )
)


unexpected_dates = (
    actual_dates
    .difference(
        expected_dates
    )
)


if len(missing_dates) != 0:

    raise ValueError(
        "\nMissing dates in daily calendar:\n"
        f"{list(missing_dates)}"
    )


if len(unexpected_dates) != 0:

    raise ValueError(
        "\nUnexpected dates in daily calendar:\n"
        f"{list(unexpected_dates)}"
    )


print(
    "Complete daily calendar continuity: PASS"
)


# ============================================================
# CONFIRM EXACTLY BTC AND ETH FOR EVERY DATE
# ============================================================

assets_per_date = (
    daily.groupby(
        "post_date"
    )["asset"]
    .nunique()
)


dates_missing_asset = int(
    assets_per_date
    .ne(2)
    .sum()
)


if dates_missing_asset != 0:

    raise ValueError(
        "\nSome calendar dates do not contain "
        "both BTC and ETH."
    )


print(
    "BTC and ETH present for every calendar date: PASS"
)


# ============================================================
# SECTION 20 — DAILY TOTAL RECONCILIATION
# ============================================================

section("RECONCILING DAILY COVERAGE TOTALS")


daily_raw_total = int(
    daily[
        "raw_post_count"
    ].sum()
)

daily_retained_total = int(
    daily[
        "retained_post_count"
    ].sum()
)

daily_excluded_total = int(
    daily[
        "excluded_post_count"
    ].sum()
)


invalid_daily_accounting = int(
    (
        daily[
            "retained_post_count"
        ]
        + daily[
            "excluded_post_count"
        ]
        != daily[
            "raw_post_count"
        ]
    ).sum()
)


print(
    f"\nDaily raw-post total: "
    f"{daily_raw_total:,}"
)

print(
    f"Daily retained-post total: "
    f"{daily_retained_total:,}"
)

print(
    f"Daily excluded-post total: "
    f"{daily_excluded_total:,}"
)

print(
    f"Invalid daily accounting rows: "
    f"{invalid_daily_accounting:,}"
)


if daily_raw_total != EXPECTED_INPUT_ROWS:

    raise ValueError(
        "\nDaily raw-post total does not match "
        "the full dataset."
    )


if daily_retained_total != EXPECTED_RETAINED_ROWS:

    raise ValueError(
        "\nDaily retained-post total does not match "
        "the primary sample."
    )


if daily_excluded_total != EXPECTED_EXCLUDED_ROWS:

    raise ValueError(
        "\nDaily excluded-post total does not match "
        "the excluded sample."
    )


if invalid_daily_accounting != 0:

    raise ValueError(
        "\nInvalid daily accounting rows found."
    )


print(
    "\nDaily total reconciliation: PASS"
)


# ============================================================
# SECTION 21 — COVERAGE BY ASSET
# ============================================================

section("ASSESSING DAILY COVERAGE BY ASSET")


coverage_rows = []


for asset in [
    "BTC",
    "ETH",
]:

    asset_daily = daily.loc[
        daily["asset"] == asset
    ].copy()

    calendar_count = len(
        asset_daily
    )

    raw_days = int(
        asset_daily[
            "raw_post_count"
        ]
        .gt(0)
        .sum()
    )

    zero_raw_days = int(
        asset_daily[
            "raw_post_count"
        ]
        .eq(0)
        .sum()
    )

    retained_days = int(
        asset_daily[
            "retained_post_count"
        ]
        .gt(0)
        .sum()
    )

    zero_retained_days = int(
        asset_daily[
            "retained_post_count"
        ]
        .eq(0)
        .sum()
    )

    coverage_rows.append(
        {
            "asset": asset,
            "calendar_days": calendar_count,
            "days_with_raw_posts": raw_days,
            "days_without_raw_posts": zero_raw_days,
            "days_with_retained_posts": retained_days,
            "days_without_retained_posts": zero_retained_days,
            "raw_day_coverage_percent": round(
                raw_days
                / calendar_count
                * 100,
                4
            ),
            "retained_day_coverage_percent": round(
                retained_days
                / calendar_count
                * 100,
                4
            ),
        }
    )


coverage_summary = pd.DataFrame(
    coverage_rows
)


print(
    "\nDaily coverage by asset:"
)

print(
    coverage_summary.to_string(
        index=False
    )
)


# ============================================================
# SECTION 22 — ZERO-OBSERVATION DATES
# ============================================================

section("IDENTIFYING ZERO-OBSERVATION DATES")


zero_coverage = daily.loc[
    (
        daily[
            "raw_post_count"
        ].eq(0)
        |
        daily[
            "retained_post_count"
        ].eq(0)
    )
].copy()


zero_coverage[
    "zero_raw_posts"
] = (
    zero_coverage[
        "raw_post_count"
    ].eq(0)
)


zero_coverage[
    "zero_retained_posts"
] = (
    zero_coverage[
        "retained_post_count"
    ].eq(0)
)


print(
    f"\nDate-asset rows with zero raw "
    f"or retained posts: "
    f"{len(zero_coverage):,}"
)


for asset in [
    "BTC",
    "ETH",
]:

    asset_zero = zero_coverage.loc[
        zero_coverage[
            "asset"
        ] == asset
    ]

    print(
        f"\n{asset} zero-coverage dates:"
    )

    if len(asset_zero) == 0:

        print(
            "None"
        )

    else:

        print(
            asset_zero[
                [
                    "post_date",
                    "raw_post_count",
                    "retained_post_count",
                    "excluded_post_count",
                    "zero_raw_posts",
                    "zero_retained_posts",
                ]
            ]
            .to_string(
                index=False
            )
        )


# ============================================================
# SECTION 23 — TEMPORAL REPRESENTATION BY YEAR × ASSET
# ============================================================

section("ASSESSING TEMPORAL REPRESENTATION")


temporal_representation = (

    full.groupby(
        [
            "year",
            "asset",
        ],
        observed=True
    )

    .agg(

        raw_posts=(
            "post_id",
            "size"
        ),

        retained_posts=(
            "retain_primary",
            "sum"
        ),

    )

    .reset_index()

)


total_raw = int(
    temporal_representation[
        "raw_posts"
    ].sum()
)

total_retained = int(
    temporal_representation[
        "retained_posts"
    ].sum()
)


temporal_representation[
    "raw_share_percent"
] = (
    temporal_representation[
        "raw_posts"
    ]
    / total_raw
    * 100
)


temporal_representation[
    "retained_share_percent"
] = (
    temporal_representation[
        "retained_posts"
    ]
    / total_retained
    * 100
)


temporal_representation[
    "share_change_percentage_points"
] = (
    temporal_representation[
        "retained_share_percent"
    ]
    - temporal_representation[
        "raw_share_percent"
    ]
)


temporal_representation[
    "raw_share_percent"
] = (
    temporal_representation[
        "raw_share_percent"
    ]
    .round(4)
)


temporal_representation[
    "retained_share_percent"
] = (
    temporal_representation[
        "retained_share_percent"
    ]
    .round(4)
)


temporal_representation[
    "share_change_percentage_points"
] = (
    temporal_representation[
        "share_change_percentage_points"
    ]
    .round(4)
)


temporal_representation[
    "absolute_share_change_pp"
] = (
    temporal_representation[
        "share_change_percentage_points"
    ]
    .abs()
    .round(4)
)


temporal_representation = (
    temporal_representation
    .sort_values(
        [
            "year",
            "asset",
        ]
    )
    .reset_index(drop=True)
)


print(
    "\nRaw versus retained representation "
    "by year and asset:"
)

print(
    temporal_representation.to_string(
        index=False
    )
)


largest_temporal_shift = (
    temporal_representation
    .sort_values(
        "absolute_share_change_pp",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nLargest year-asset representation shift:"
)

print(
    f"{int(largest_temporal_shift['year'])} "
    f"{largest_temporal_shift['asset']}: "
    f"{largest_temporal_shift['share_change_percentage_points']:+.4f} "
    "percentage points"
)


# ============================================================
# SECTION 24 — SUBREDDIT REPRESENTATION
# ============================================================

section("ASSESSING SUBREDDIT REPRESENTATION")


subreddit_representation = (

    full.groupby(
        "subreddit",
        observed=True
    )

    .agg(

        raw_posts=(
            "post_id",
            "size"
        ),

        retained_posts=(
            "retain_primary",
            "sum"
        ),

    )

    .reset_index()

)


subreddit_representation[
    "raw_share_percent"
] = (
    subreddit_representation[
        "raw_posts"
    ]
    / EXPECTED_INPUT_ROWS
    * 100
)


subreddit_representation[
    "retained_share_percent"
] = (
    subreddit_representation[
        "retained_posts"
    ]
    / EXPECTED_RETAINED_ROWS
    * 100
)


subreddit_representation[
    "share_change_percentage_points"
] = (
    subreddit_representation[
        "retained_share_percent"
    ]
    - subreddit_representation[
        "raw_share_percent"
    ]
)


subreddit_representation[
    "raw_share_percent"
] = (
    subreddit_representation[
        "raw_share_percent"
    ]
    .round(4)
)


subreddit_representation[
    "retained_share_percent"
] = (
    subreddit_representation[
        "retained_share_percent"
    ]
    .round(4)
)


subreddit_representation[
    "share_change_percentage_points"
] = (
    subreddit_representation[
        "share_change_percentage_points"
    ]
    .round(4)
)


subreddit_representation[
    "absolute_share_change_pp"
] = (
    subreddit_representation[
        "share_change_percentage_points"
    ]
    .abs()
    .round(4)
)


print(
    "\nRaw versus retained subreddit representation:"
)

print(
    subreddit_representation.to_string(
        index=False
    )
)


largest_subreddit_shift = (
    subreddit_representation
    .sort_values(
        "absolute_share_change_pp",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nLargest subreddit representation shift:"
)

print(
    f"{largest_subreddit_shift['subreddit']}: "
    f"{largest_subreddit_shift['share_change_percentage_points']:+.4f} "
    "percentage points"
)


# ============================================================
# SECTION 25 — IDENTIFY HIGHEST EXCLUSION GROUPS
# ============================================================

section("IDENTIFYING HIGHEST EXCLUSION GROUPS")


highest_year_asset = (
    year_asset_validation
    .sort_values(
        "excluded_percent",
        ascending=False
    )
    .iloc[0]
)


highest_subreddit = (
    subreddit_validation
    .sort_values(
        "excluded_percent",
        ascending=False
    )
    .iloc[0]
)


highest_year = (
    year_validation
    .sort_values(
        "excluded_percent",
        ascending=False
    )
    .iloc[0]
)


highest_asset = (
    asset_validation
    .sort_values(
        "excluded_percent",
        ascending=False
    )
    .iloc[0]
)


print(
    "\nHighest year-asset exclusion rate:"
)

print(
    f"{int(highest_year_asset['year'])} "
    f"{highest_year_asset['asset']}: "
    f"{highest_year_asset['excluded_percent']:.4f}%"
)


print(
    "\nHighest subreddit exclusion rate:"
)

print(
    f"{highest_subreddit['subreddit']}: "
    f"{highest_subreddit['excluded_percent']:.4f}%"
)


print(
    "\nHighest annual exclusion rate:"
)

print(
    f"{int(highest_year['year'])}: "
    f"{highest_year['excluded_percent']:.4f}%"
)


print(
    "\nHighest asset-level exclusion rate:"
)

print(
    f"{highest_asset['asset']}: "
    f"{highest_asset['excluded_percent']:.4f}%"
)


# ============================================================
# SECTION 26 — FINAL QUANTITATIVE QC SUMMARY
# ============================================================

section("CREATING FINAL QUANTITATIVE QC SUMMARY")


btc_coverage = (
    coverage_summary.loc[
        coverage_summary[
            "asset"
        ] == "BTC"
    ]
    .iloc[0]
)


eth_coverage = (
    coverage_summary.loc[
        coverage_summary[
            "asset"
        ] == "ETH"
    ]
    .iloc[0]
)


final_qc_rows = [
    {
        "validation_check": "Original input posts",
        "result": EXPECTED_INPUT_ROWS,
        "status": "PASS",
    },
    {
        "validation_check": "Retained posts",
        "result": EXPECTED_RETAINED_ROWS,
        "status": "PASS",
    },
    {
        "validation_check": "Excluded posts",
        "result": EXPECTED_EXCLUDED_ROWS,
        "status": "PASS",
    },
    {
        "validation_check": "Stage 02 audit reconciliation",
        "result": "Exact",
        "status": "PASS",
    },
    {
        "validation_check": "Primary sample membership",
        "result": "Exact retained-ID match",
        "status": "PASS",
    },
    {
        "validation_check": "Manual-review categories",
        "result": len(
            actual_review_categories
        ),
        "status": "PASS",
    },
    {
        "validation_check": "Manual-review rows",
        "result": len(
            examples
        ),
        "status": "PASS",
    },
    {
        "validation_check": "Excluded posts with no primary flag",
        "result": excluded_without_flag,
        "status": "PASS",
    },
    {
        "validation_check": "Retained posts with primary flag",
        "result": retained_with_flag,
        "status": "PASS",
    },
    {
        "validation_check": "Calendar start",
        "result": str(
            calendar_start.date()
        ),
        "status": "PASS",
    },
    {
        "validation_check": "Calendar end",
        "result": str(
            calendar_end.date()
        ),
        "status": "PASS",
    },
    {
        "validation_check": "Calendar days",
        "result": calendar_days,
        "status": "PASS",
    },
    {
        "validation_check": "Date-asset rows",
        "result": len(
            daily
        ),
        "status": "PASS",
    },
    {
        "validation_check": "Daily raw total",
        "result": daily_raw_total,
        "status": "PASS",
    },
    {
        "validation_check": "Daily retained total",
        "result": daily_retained_total,
        "status": "PASS",
    },
    {
        "validation_check": "Daily excluded total",
        "result": daily_excluded_total,
        "status": "PASS",
    },
    {
        "validation_check": "BTC days without retained posts",
        "result": int(
            btc_coverage[
                "days_without_retained_posts"
            ]
        ),
        "status": "REVIEWED",
    },
    {
        "validation_check": "ETH days without retained posts",
        "result": int(
            eth_coverage[
                "days_without_retained_posts"
            ]
        ),
        "status": "REVIEWED",
    },
    {
        "validation_check": "Highest year-asset exclusion group",
        "result": (
            f"{int(highest_year_asset['year'])} "
            f"{highest_year_asset['asset']} "
            f"({highest_year_asset['excluded_percent']:.4f}%)"
        ),
        "status": "REVIEWED",
    },
    {
        "validation_check": "ETH 2023 exclusions",
        "result": (
            "258 = 125 crossposts + "
            "133 exact duplicates"
        ),
        "status": "REVIEWED",
    },
    {
        "validation_check": "Largest year-asset share shift",
        "result": (
            f"{int(largest_temporal_shift['year'])} "
            f"{largest_temporal_shift['asset']} "
            f"({largest_temporal_shift['share_change_percentage_points']:+.4f} pp)"
        ),
        "status": "REVIEWED",
    },
    {
        "validation_check": "Largest subreddit share shift",
        "result": (
            f"{largest_subreddit_shift['subreddit']} "
            f"({largest_subreddit_shift['share_change_percentage_points']:+.4f} pp)"
        ),
        "status": "REVIEWED",
    },
]


final_qc = pd.DataFrame(
    final_qc_rows
)


print(
    "\nFinal quantitative QC summary:"
)

print(
    final_qc.to_string(
        index=False
    )
)


# ============================================================
# SECTION 27 — METHODOLOGICAL DECISION TABLE
# ============================================================

section("CREATING METHODOLOGICAL DECISION TABLE")


decision_rows = [
    {
        "validation_area": "Dataset accounting",
        "evidence": (
            f"{EXPECTED_INPUT_ROWS:,} input; "
            f"{EXPECTED_RETAINED_ROWS:,} retained; "
            f"{EXPECTED_EXCLUDED_ROWS:,} excluded"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Primary sample integrity",
        "evidence": (
            "Primary post IDs exactly match "
            "retain_primary=True post IDs"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Exclusion-rule overlap",
        "evidence": (
            "Pairwise and multi-rule overlap "
            "quantified and documented"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Link-only / unusable overlap",
        "evidence": (
            f"{both_count:,} posts carry both flags; "
            f"{link_only_not_unusable:,} link-only posts "
            "are outside unusable-text"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "ETH 2023 higher exclusion rate",
        "evidence": (
            "258 exclusions: 125 crossposts and "
            "133 exact duplicates"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Daily calendar structure",
        "evidence": (
            f"{EXPECTED_CALENDAR_DAYS:,} complete dates "
            f"and {EXPECTED_DATE_ASSET_ROWS:,} "
            "BTC/ETH date-asset rows"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Daily accounting",
        "evidence": (
            "Daily raw, retained and excluded totals "
            "reconcile exactly to Stage 02"
        ),
        "decision": "Accept",
    },
    {
        "validation_area": "Temporal representation",
        "evidence": (
            "Raw versus retained year-asset shares "
            "quantified"
        ),
        "decision": "Review output",
    },
    {
        "validation_area": "Subreddit representation",
        "evidence": (
            "Raw versus retained subreddit shares "
            "quantified"
        ),
        "decision": "Review output",
    },
    {
        "validation_area": "Manual exclusion examples",
        "evidence": (
            f"{len(examples):,} examples across "
            f"{len(actual_review_categories)} categories"
        ),
        "decision": "Manual review required",
    },
]


methodological_decisions = pd.DataFrame(
    decision_rows
)


print(
    "\nMethodological decision table:"
)

print(
    methodological_decisions.to_string(
        index=False
    )
)


# ============================================================
# SECTION 28 — SAVE STAGE 03 OUTPUTS
# ============================================================

section("SAVING STAGE 03 VALIDATION OUTPUTS")


flag_summary.to_csv(
    FLAG_SUMMARY_FILE,
    index=False
)

overlap_matrix.to_csv(
    OVERLAP_MATRIX_FILE,
    index=True,
    index_label="flag"
)

flag_count_distribution.to_csv(
    FLAG_COUNT_DISTRIBUTION_FILE,
    index=False
)

reason_combinations.to_csv(
    EXCLUSION_COMBINATIONS_FILE,
    index=False
)

year_validation.to_csv(
    YEAR_VALIDATION_FILE,
    index=False
)

asset_validation.to_csv(
    ASSET_VALIDATION_FILE,
    index=False
)

year_asset_validation.to_csv(
    YEAR_ASSET_VALIDATION_FILE,
    index=False
)

subreddit_validation.to_csv(
    SUBREDDIT_VALIDATION_FILE,
    index=False
)

zero_coverage.to_csv(
    ZERO_COVERAGE_FILE,
    index=False
)

temporal_representation.to_csv(
    TEMPORAL_REPRESENTATION_FILE,
    index=False
)

subreddit_representation.to_csv(
    SUBREDDIT_REPRESENTATION_FILE,
    index=False
)

eth_2023_excluded.to_csv(
    ETH_2023_FILE,
    index=False
)

link_unusable_qc.to_csv(
    LINK_UNUSABLE_QC_FILE,
    index=False
)

coverage_summary.to_csv(
    COVERAGE_SUMMARY_FILE,
    index=False
)

final_qc.to_csv(
    FINAL_QC_FILE,
    index=False
)

methodological_decisions.to_csv(
    DECISION_FILE,
    index=False
)


print(
    "\nStage 03 output directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nStage 03 files written:"
)


stage03_output_files = [
    FLAG_SUMMARY_FILE,
    OVERLAP_MATRIX_FILE,
    FLAG_COUNT_DISTRIBUTION_FILE,
    EXCLUSION_COMBINATIONS_FILE,
    YEAR_VALIDATION_FILE,
    ASSET_VALIDATION_FILE,
    YEAR_ASSET_VALIDATION_FILE,
    SUBREDDIT_VALIDATION_FILE,
    ZERO_COVERAGE_FILE,
    TEMPORAL_REPRESENTATION_FILE,
    SUBREDDIT_REPRESENTATION_FILE,
    ETH_2023_FILE,
    LINK_UNUSABLE_QC_FILE,
    COVERAGE_SUMMARY_FILE,
    FINAL_QC_FILE,
    DECISION_FILE,
]


for file_path in stage03_output_files:

    print(
        f"  - {file_path.name}"
    )


# ============================================================
# SECTION 29 — RELOAD ALL STAGE 03 OUTPUTS
# ============================================================

section("RELOADING STAGE 03 OUTPUTS")


for file_path in stage03_output_files:

    if not file_path.exists():

        raise FileNotFoundError(
            "\nStage 03 output file missing:\n"
            f"{file_path}"
        )


reloaded_outputs = {}


for file_path in stage03_output_files:

    reloaded_outputs[
        file_path.name
    ] = pd.read_csv(
        file_path,
        low_memory=False
    )


print(
    f"\nAll {len(stage03_output_files)} "
    "Stage 03 CSV files reloaded successfully."
)


# ============================================================
# SECTION 30 — FINAL OUTPUT INTEGRITY CHECKS
# ============================================================

section("VALIDATING RELOADED STAGE 03 OUTPUTS")


if len(
    reloaded_outputs[
        YEAR_VALIDATION_FILE.name
    ]
) != 5:

    raise ValueError(
        "\nReloaded year validation should contain 5 rows."
    )


if len(
    reloaded_outputs[
        ASSET_VALIDATION_FILE.name
    ]
) != 2:

    raise ValueError(
        "\nReloaded asset validation should contain 2 rows."
    )


if len(
    reloaded_outputs[
        YEAR_ASSET_VALIDATION_FILE.name
    ]
) != 10:

    raise ValueError(
        "\nReloaded year-asset validation "
        "should contain 10 rows."
    )


if len(
    reloaded_outputs[
        SUBREDDIT_VALIDATION_FILE.name
    ]
) != 3:

    raise ValueError(
        "\nReloaded subreddit validation "
        "should contain 3 rows."
    )


if len(
    reloaded_outputs[
        COVERAGE_SUMMARY_FILE.name
    ]
) != 2:

    raise ValueError(
        "\nReloaded coverage summary "
        "should contain 2 rows."
    )


if len(
    reloaded_outputs[
        FINAL_QC_FILE.name
    ]
) != len(
    final_qc
):

    raise ValueError(
        "\nReloaded final QC row count mismatch."
    )


if len(
    reloaded_outputs[
        DECISION_FILE.name
    ]
) != len(
    methodological_decisions
):

    raise ValueError(
        "\nReloaded methodological decision "
        "row count mismatch."
    )


print(
    "\nReloaded Stage 03 output structure: PASS"
)


# ============================================================
# SECTION 31 — CONFIRM STAGE 02 FILES WERE NOT TARGETED
# ============================================================

section("CONFIRMING OUTPUT SEPARATION")


for output_file in stage03_output_files:

    if output_file.parent != OUTPUT_DIR:

        raise ValueError(
            "\nA Stage 03 output is outside the "
            "Stage 03 validation directory."
        )


print(
    "\nAll Stage 03 outputs are isolated in:"
)

print(
    OUTPUT_DIR
)

print(
    "\nStage 02 output paths are not used as "
    "Stage 03 output targets: PASS"
)


# ============================================================
# SECTION 32 — FINAL COMPLETION SUMMARY
# ============================================================

section("STAGE 03 — FINAL COMPLETION SUMMARY")


print(
    f"\nOriginal Reddit posts: "
    f"{EXPECTED_INPUT_ROWS:,}"
)

print(
    f"Retained for primary sentiment analysis: "
    f"{EXPECTED_RETAINED_ROWS:,} "
    f"({EXPECTED_RETAINED_ROWS / EXPECTED_INPUT_ROWS * 100:.2f}%)"
)

print(
    f"Excluded from primary sentiment analysis: "
    f"{EXPECTED_EXCLUDED_ROWS:,} "
    f"({EXPECTED_EXCLUDED_ROWS / EXPECTED_INPUT_ROWS * 100:.2f}%)"
)


print(
    f"\nCalendar period: "
    f"{EXPECTED_CALENDAR_START.date()} "
    f"to {EXPECTED_CALENDAR_END.date()}"
)

print(
    f"Calendar days: "
    f"{EXPECTED_CALENDAR_DAYS:,}"
)

print(
    f"BTC/ETH date-asset rows: "
    f"{EXPECTED_DATE_ASSET_ROWS:,}"
)


print(
    "\nQuantitative cleaning validation: COMPLETE"
)

print(
    "Exclusion overlap assessment: COMPLETE"
)

print(
    "Year/asset/subreddit assessment: COMPLETE"
)

print(
    "ETH 2023 investigation: COMPLETE"
)

print(
    "Daily coverage assessment: COMPLETE"
)

print(
    "Zero-observation date identification: COMPLETE"
)

print(
    "Temporal representation assessment: COMPLETE"
)

print(
    "Subreddit representation assessment: COMPLETE"
)


print(
    "\nIMPORTANT:"
)

print(
    "Stage 02 cleaning decisions have NOT been changed."
)

print(
    "Stage 02 output CSV files have NOT been modified "
    "by this script."
)

print(
    "No sentiment scoring has been performed."
)

print(
    "No market data has been merged."
)


print(
    "\nFinal remaining Stage 03 task:"
)

print(
    "Review the quantitative outputs and manual-review "
    "examples before formally accepting the cleaned sample."
)


print(
    "\nSTAGE 03 — CLEANING AND COVERAGE VALIDATION: "
    "SCRIPT COMPLETE"
)