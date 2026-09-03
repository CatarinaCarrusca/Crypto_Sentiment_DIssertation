"""
SECTION 08 — FINAL MODELLING DATASET VALIDATION
================================================

Dissertation:
Do Social Media Sentiment Signals Improve the Prediction of Cryptocurrency
Returns Beyond Traditional Market Indicators? Evidence from Bitcoin and
Ethereum

Purpose
-------
This script validates the final integrated modelling dataset produced in
Section 07 and establishes the exact samples that will be used in Section 09.

It performs:

1. Structural validation
2. Date/calendar validation
3. Missing-value diagnostics
4. Target reconstruction checks
5. Predictor reconstruction checks
6. Reddit integration checks against Section 06
7. Traditional-market timing/leakage checks
8. Reddit timing/leakage checks
9. Descriptive statistics
10. Predictor-distribution diagnostics
11. Correlation diagnostics
12. Multicollinearity / VIF diagnostics
13. Extreme-return audit
14. Exact model-specification definitions
15. Common-sample construction
16. Initial training / genuine OOS period construction
17. Weekend-preservation checks
18. Alternative Reddit lag validation
19. Hard QC summary
20. Final modelling-ready dataset output

IMPORTANT
---------
This script VALIDATES the modelling data.

It does NOT:
    - estimate H1-H5;
    - run forecasting models;
    - impute Reddit sentiment on no-post days;
    - winsorise or delete extreme returns;
    - forward-fill raw financial variables;
    - use same-day financial information;
    - remove weekends;
    - randomly split the sample.

Primary forecasting design
--------------------------
Initial training period:
    2021-01-01 to 2023-12-31

Genuine out-of-sample period:
    2024-01-01 to 2025-12-31

Section 09 will use expanding-window one-step-ahead forecasting.

Primary model sequence
----------------------
M0 Benchmark:
    controls only

M1 Activity:
    controls + Reddit activity

M2 Sentiment:
    controls + Reddit sentiment

M3 Both:
    controls + Reddit activity + Reddit sentiment

Cross-cryptocurrency lagged returns are retained for robustness analysis but
are NOT part of the primary benchmark specification.

Fair-comparison principle
-------------------------
Nested model comparisons must use identical observations.

Therefore:
    Benchmark vs Activity
        uses Activity_Comparison_Sample

    Benchmark vs Sentiment
        uses Sentiment_Comparison_Sample

    Benchmark vs Both
        uses Both_Comparison_Sample

The main four-model explanatory comparison may use:
    Common_Main_Model_Sample

which requires all M3 variables and therefore gives all four specifications
the same observations.

No missing Reddit sentiment is replaced by zero.

VIF methodology
---------------
Variance Inflation Factors are calculated using conventional auxiliary
regressions that INCLUDE an intercept/constant.

The constant is included in the auxiliary design matrix but its VIF is not
reported. Only substantive predictors are reported.

This avoids misleading uncentred VIF values for variables with large
non-zero means, such as logged cryptocurrency volume and logged Reddit
activity.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
except ImportError as exc:
    raise ImportError(
        "\nSection 08 requires statsmodels.\n"
        "Install it with:\n\n"
        "    pip install statsmodels\n"
    ) from exc


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

STAGE07_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "stage07_reddit_market_integration"
    / "combined_market_reddit_dataset.csv"
)

STAGE06_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "reddit"
    / "stage06_reddit_forecast"
    / "reddit_forecast_ready.csv"
)

MARKET_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "information_aligned_dataset.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "stage08_final_modelling_dataset"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. STUDY DESIGN CONSTANTS
# =============================================================================

STUDY_START = pd.Timestamp("2021-01-01")
STUDY_END = pd.Timestamp("2025-12-31")

INITIAL_TRAIN_START = pd.Timestamp("2021-01-01")
INITIAL_TRAIN_END = pd.Timestamp("2023-12-31")

OOS_START = pd.Timestamp("2024-01-01")
OOS_END = pd.Timestamp("2025-12-31")

EXPECTED_CALENDAR_DAYS = len(
    pd.date_range(STUDY_START, STUDY_END, freq="D")
)

EXPECTED_ASSETS = ["BTC", "ETH"]

EXPECTED_DATE_ASSET_ROWS = (
    EXPECTED_CALENDAR_DAYS * len(EXPECTED_ASSETS)
)

FLOAT_ATOL = 1e-12
FLOAT_RTOL = 1e-9

EXTREME_RETURN_THRESHOLD = 0.25


# =============================================================================
# 3. PRIMARY MODELLING VARIABLES
# =============================================================================

TARGET = "Target_Return"

BENCHMARK_PREDICTORS = [
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
]

ACTIVITY_PREDICTOR = "Lagged_Log_Reddit_Post_Count"

SENTIMENT_PREDICTOR = "Lagged_Reddit_Sentiment"

CROSS_CRYPTO_PREDICTOR = "Cross_Crypto_Lagged_Return"

ACTIVITY_MODEL_PREDICTORS = (
    BENCHMARK_PREDICTORS
    + [ACTIVITY_PREDICTOR]
)

SENTIMENT_MODEL_PREDICTORS = (
    BENCHMARK_PREDICTORS
    + [SENTIMENT_PREDICTOR]
)

BOTH_MODEL_PREDICTORS = (
    BENCHMARK_PREDICTORS
    + [
        ACTIVITY_PREDICTOR,
        SENTIMENT_PREDICTOR,
    ]
)

BENCHMARK_CROSS_PREDICTORS = (
    BENCHMARK_PREDICTORS
    + [CROSS_CRYPTO_PREDICTOR]
)

ACTIVITY_CROSS_PREDICTORS = (
    ACTIVITY_MODEL_PREDICTORS
    + [CROSS_CRYPTO_PREDICTOR]
)

SENTIMENT_CROSS_PREDICTORS = (
    SENTIMENT_MODEL_PREDICTORS
    + [CROSS_CRYPTO_PREDICTOR]
)

BOTH_CROSS_PREDICTORS = (
    BOTH_MODEL_PREDICTORS
    + [CROSS_CRYPTO_PREDICTOR]
)

ALL_PRIMARY_NUMERIC_VARIABLES = [
    TARGET,
    *BENCHMARK_PREDICTORS,
    ACTIVITY_PREDICTOR,
    SENTIMENT_PREDICTOR,
    CROSS_CRYPTO_PREDICTOR,
]

ALL_PRIMARY_NUMERIC_VARIABLES = list(
    dict.fromkeys(ALL_PRIMARY_NUMERIC_VARIABLES)
)


# =============================================================================
# 4. TRADITIONAL SOURCE-DATE VARIABLES
# =============================================================================

TRADITIONAL_ALIGNMENT = {
    "SP500": {
        "value": "Lagged_SP500_Return_Aligned",
        "source_date": "SP500_Source_Date",
    },
    "VIX": {
        "value": "Lagged_VIX_Change_Aligned",
        "source_date": "VIX_Source_Date",
    },
    "Gold": {
        "value": "Lagged_Gold_Return_Aligned",
        "source_date": "Gold_Source_Date",
    },
    "DXY": {
        "value": "Lagged_DXY_Return_Aligned",
        "source_date": "DXY_Source_Date",
    },
    "US10Y": {
        "value": "Lagged_US10Y_Change_Aligned",
        "source_date": "US10Y_Source_Date",
    },
}


# =============================================================================
# 5. ALTERNATIVE REDDIT LAGS
# =============================================================================

ALTERNATIVE_REDDIT_LAGS = {
    2: {
        "sentiment": "Reddit_Sentiment_Lag_2",
        "activity": "Log_Reddit_Post_Count_Lag_2",
        "source_date": "Reddit_Source_Date_Lag_2",
    },
    3: {
        "sentiment": "Reddit_Sentiment_Lag_3",
        "activity": "Log_Reddit_Post_Count_Lag_3",
        "source_date": "Reddit_Source_Date_Lag_3",
    },
    7: {
        "sentiment": "Reddit_Sentiment_Lag_7",
        "activity": "Log_Reddit_Post_Count_Lag_7",
        "source_date": "Reddit_Source_Date_Lag_7",
    },
}


# =============================================================================
# 6. HELPER FUNCTIONS
# =============================================================================

def banner(title: str) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            f"\nRequired input file not found:\n{path}\n"
        )


def require_columns(
    df: pd.DataFrame,
    columns: Iterable[str],
    dataframe_name: str,
) -> None:

    missing = [
        col
        for col in columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"\nMissing required columns in {dataframe_name}:\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )


def parse_date_column(
    df: pd.DataFrame,
    column: str,
) -> None:

    if column in df.columns:
        df[column] = pd.to_datetime(
            df[column],
            errors="coerce",
        )


def parse_all_date_like_columns(
    df: pd.DataFrame,
) -> None:

    date_columns = [
        col
        for col in df.columns
        if (
            col == "Date"
            or col.endswith("_Date")
            or "Source_Date" in col
        )
    ]

    for col in date_columns:
        parse_date_column(df, col)


def same_numeric(
    left: pd.Series,
    right: pd.Series,
    atol: float = FLOAT_ATOL,
    rtol: float = FLOAT_RTOL,
) -> pd.Series:

    left_num = pd.to_numeric(
        left,
        errors="coerce",
    )

    right_num = pd.to_numeric(
        right,
        errors="coerce",
    )

    both_missing = (
        left_num.isna()
        & right_num.isna()
    )

    both_present = (
        left_num.notna()
        & right_num.notna()
    )

    close = pd.Series(
        False,
        index=left.index,
        dtype=bool,
    )

    if both_present.any():
        close.loc[both_present] = np.isclose(
            left_num.loc[both_present],
            right_num.loc[both_present],
            atol=atol,
            rtol=rtol,
            equal_nan=False,
        )

    return both_missing | close


def same_dates(
    left: pd.Series,
    right: pd.Series,
) -> pd.Series:

    left_date = pd.to_datetime(
        left,
        errors="coerce",
    )

    right_date = pd.to_datetime(
        right,
        errors="coerce",
    )

    both_missing = (
        left_date.isna()
        & right_date.isna()
    )

    both_present_equal = (
        left_date.notna()
        & right_date.notna()
        & (left_date == right_date)
    )

    return (
        both_missing
        | both_present_equal
    )


def finite_complete_mask(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.Series:

    numeric = df[columns].apply(
        pd.to_numeric,
        errors="coerce",
    )

    return (
        numeric.notna().all(axis=1)
        & np.isfinite(numeric).all(axis=1)
    )


def add_qc(
    qc_rows: list[dict],
    check: str,
    expected,
    actual,
    passed: bool,
    category: str,
    notes: str = "",
) -> None:

    qc_rows.append(
        {
            "Check": check,
            "Expected": expected,
            "Actual": actual,
            "Pass": bool(passed),
            "Category": category,
            "Notes": notes,
        }
    )


def safe_pct(
    numerator: int,
    denominator: int,
) -> float:

    if denominator == 0:
        return np.nan

    return (
        100.0
        * numerator
        / denominator
    )


# =============================================================================
# 7. CORRECTED VIF FUNCTION
# =============================================================================

def vif_table(
    df: pd.DataFrame,
    asset: str,
    predictors: list[str],
    sample_name: str,
) -> pd.DataFrame:
    """
    Calculate conventional Variance Inflation Factors.

    IMPORTANT
    ---------
    VIF is based on an auxiliary regression of each predictor on all other
    predictors.

    These auxiliary regressions should contain an intercept. Therefore this
    implementation explicitly adds a constant using statsmodels.add_constant.

    The constant itself is NOT reported in the final VIF table.

    This is important for predictors with large non-zero means, including
    logged cryptocurrency volume and logged Reddit post counts. Calculating
    VIF without a constant can produce misleadingly large uncentred VIFs.
    """

    work = (
        df.loc[
            df["Asset"].eq(asset),
            predictors,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .replace(
            [np.inf, -np.inf],
            np.nan,
        )
        .dropna()
        .copy()
    )

    rows = []

    if work.empty:

        for predictor in predictors:

            rows.append(
                {
                    "Asset": asset,
                    "Sample": sample_name,
                    "Predictor": predictor,
                    "N": 0,
                    "VIF": np.nan,
                    "Zero_Variance": np.nan,
                }
            )

        return pd.DataFrame(rows)

    # -------------------------------------------------------------------------
    # Add an intercept for conventional centred VIF calculation.
    # -------------------------------------------------------------------------

    design = sm.add_constant(
        work,
        has_constant="add",
    )

    for predictor in predictors:

        zero_variance = bool(
            np.isclose(
                work[predictor].var(ddof=0),
                0.0,
                atol=FLOAT_ATOL,
            )
        )

        if zero_variance:

            vif_value = np.inf

        else:

            predictor_index = (
                design.columns.get_loc(
                    predictor
                )
            )

            try:

                vif_value = (
                    variance_inflation_factor(
                        design.to_numpy(
                            dtype=float
                        ),
                        predictor_index,
                    )
                )

            except Exception:

                vif_value = np.nan

        rows.append(
            {
                "Asset": asset,
                "Sample": sample_name,
                "Predictor": predictor,
                "N": len(work),
                "VIF": vif_value,
                "Zero_Variance": zero_variance,
            }
        )

    return pd.DataFrame(rows)


# =============================================================================
# 8. LOAD INPUT DATA
# =============================================================================

banner(
    "SECTION 08 — FINAL MODELLING DATASET VALIDATION"
)

print("\nInput files:")

print(
    f"  Section 07: {STAGE07_FILE}"
)

print(
    f"  Section 06: {STAGE06_FILE}"
)

print(
    f"  Market:     {MARKET_FILE}"
)

print(
    f"\nOutput directory:\n  {OUTPUT_DIR}"
)

for input_file in [
    STAGE07_FILE,
    STAGE06_FILE,
    MARKET_FILE,
]:
    require_file(input_file)

df = pd.read_csv(
    STAGE07_FILE
)

reddit06 = pd.read_csv(
    STAGE06_FILE
)

market = pd.read_csv(
    MARKET_FILE
)

parse_all_date_like_columns(df)
parse_all_date_like_columns(reddit06)
parse_all_date_like_columns(market)

df = (
    df
    .sort_values(
        ["Date", "Asset"]
    )
    .reset_index(drop=True)
)

reddit06 = (
    reddit06
    .sort_values(
        ["Date", "Asset"]
    )
    .reset_index(drop=True)
)

market = (
    market
    .sort_values("Date")
    .reset_index(drop=True)
)

print("\nRows loaded:")

print(
    f"  Section 07 integrated dataset: "
    f"{len(df):,}"
)

print(
    f"  Section 06 Reddit dataset:     "
    f"{len(reddit06):,}"
)

print(
    f"  Market dataset:                "
    f"{len(market):,}"
)


# =============================================================================
# 9. REQUIRED SCHEMA VALIDATION
# =============================================================================

banner(
    "1. SCHEMA AND STRUCTURAL VALIDATION"
)

required_stage07 = [
    "Date",
    "Asset",
    TARGET,
    *BENCHMARK_PREDICTORS,
    ACTIVITY_PREDICTOR,
    SENTIMENT_PREDICTOR,
    CROSS_CRYPTO_PREDICTOR,
]

require_columns(
    df,
    required_stage07,
    "Section 07 integrated dataset",
)

required_stage06 = [
    "Date",
    "Asset",
    "Lagged_Reddit_Sentiment",
    "Lagged_Log_Reddit_Post_Count",
]

require_columns(
    reddit06,
    required_stage06,
    "Section 06 Reddit dataset",
)

required_market = [
    "Date",
    "BTC_Return",
    "BTC_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "ETH_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return_Aligned",
    "SP500_Source_Date",
    "Lagged_VIX_Change_Aligned",
    "VIX_Source_Date",
    "Lagged_Gold_Return_Aligned",
    "Gold_Source_Date",
    "Lagged_DXY_Return_Aligned",
    "DXY_Source_Date",
    "Lagged_US10Y_Change_Aligned",
    "US10Y_Source_Date",
]

require_columns(
    market,
    required_market,
    "information-aligned market dataset",
)

qc_rows: list[dict] = []

add_qc(
    qc_rows,
    "Integrated_Rows",
    EXPECTED_DATE_ASSET_ROWS,
    len(df),
    len(df) == EXPECTED_DATE_ASSET_ROWS,
    "Structure",
)

calendar_days = (
    df["Date"].nunique()
)

add_qc(
    qc_rows,
    "Calendar_Days",
    EXPECTED_CALENDAR_DAYS,
    calendar_days,
    calendar_days == EXPECTED_CALENDAR_DAYS,
    "Structure",
)

duplicate_rows = int(
    df
    .duplicated(
        ["Date", "Asset"]
    )
    .sum()
)

add_qc(
    qc_rows,
    "Duplicate_Date_Asset_Rows",
    0,
    duplicate_rows,
    duplicate_rows == 0,
    "Structure",
)

asset_values = sorted(
    df["Asset"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

add_qc(
    qc_rows,
    "Assets",
    ",".join(EXPECTED_ASSETS),
    ",".join(asset_values),
    asset_values == EXPECTED_ASSETS,
    "Structure",
)

for asset in EXPECTED_ASSETS:

    actual = int(
        df["Asset"]
        .eq(asset)
        .sum()
    )

    add_qc(
        qc_rows,
        f"{asset}_Rows",
        EXPECTED_CALENDAR_DAYS,
        actual,
        actual == EXPECTED_CALENDAR_DAYS,
        "Structure",
    )

actual_min_date = (
    df["Date"].min()
)

actual_max_date = (
    df["Date"].max()
)

add_qc(
    qc_rows,
    "Study_Start",
    STUDY_START.date(),
    (
        actual_min_date.date()
        if pd.notna(actual_min_date)
        else None
    ),
    actual_min_date == STUDY_START,
    "Structure",
)

add_qc(
    qc_rows,
    "Study_End",
    STUDY_END.date(),
    (
        actual_max_date.date()
        if pd.notna(actual_max_date)
        else None
    ),
    actual_max_date == STUDY_END,
    "Structure",
)


# =============================================================================
# 10. COMPLETE DATE × ASSET GRID
# =============================================================================

expected_dates = pd.date_range(
    STUDY_START,
    STUDY_END,
    freq="D",
)

expected_grid = (
    pd.MultiIndex.from_product(
        [
            expected_dates,
            EXPECTED_ASSETS,
        ],
        names=[
            "Date",
            "Asset",
        ],
    )
)

actual_grid = (
    pd.MultiIndex.from_frame(
        df[
            [
                "Date",
                "Asset",
            ]
        ]
    )
)

missing_grid = (
    expected_grid
    .difference(actual_grid)
)

unexpected_grid = (
    actual_grid
    .difference(expected_grid)
)

add_qc(
    qc_rows,
    "Missing_Date_Asset_Combinations",
    0,
    len(missing_grid),
    len(missing_grid) == 0,
    "Calendar",
)

add_qc(
    qc_rows,
    "Unexpected_Date_Asset_Combinations",
    0,
    len(unexpected_grid),
    len(unexpected_grid) == 0,
    "Calendar",
)

calendar_audit_rows = []

for date_value, asset_value in missing_grid:

    calendar_audit_rows.append(
        {
            "Date": date_value,
            "Asset": asset_value,
            "Issue": (
                "MISSING_EXPECTED_DATE_ASSET_ROW"
            ),
        }
    )

for date_value, asset_value in unexpected_grid:

    calendar_audit_rows.append(
        {
            "Date": date_value,
            "Asset": asset_value,
            "Issue": (
                "UNEXPECTED_DATE_ASSET_ROW"
            ),
        }
    )

calendar_audit = pd.DataFrame(
    calendar_audit_rows,
    columns=[
        "Date",
        "Asset",
        "Issue",
    ],
)


# =============================================================================
# 11. NUMERIC / INFINITE VALIDATION
# =============================================================================

banner(
    "2. NUMERIC VALIDATION"
)

infinite_rows = []

for variable in ALL_PRIMARY_NUMERIC_VARIABLES:

    numeric = pd.to_numeric(
        df[variable],
        errors="coerce",
    )

    original_nonmissing = (
        df[variable].notna()
    )

    coerced_missing = (
        numeric.isna()
    )

    nonnumeric_count = int(
        (
            original_nonmissing
            & coerced_missing
        ).sum()
    )

    add_qc(
        qc_rows,
        f"{variable}_NonNumeric_Values",
        0,
        nonnumeric_count,
        nonnumeric_count == 0,
        "Numeric",
    )

    inf_mask = (
        numeric.notna()
        & ~np.isfinite(numeric)
    )

    for idx in df.index[inf_mask]:

        infinite_rows.append(
            {
                "Date": df.loc[
                    idx,
                    "Date",
                ],
                "Asset": df.loc[
                    idx,
                    "Asset",
                ],
                "Variable": variable,
                "Value": df.loc[
                    idx,
                    variable,
                ],
            }
        )

    add_qc(
        qc_rows,
        f"{variable}_Infinite_Values",
        0,
        int(inf_mask.sum()),
        int(inf_mask.sum()) == 0,
        "Numeric",
    )

infinite_audit = pd.DataFrame(
    infinite_rows,
    columns=[
        "Date",
        "Asset",
        "Variable",
        "Value",
    ],
)


# =============================================================================
# 12. TARGET AND CRYPTO PREDICTOR RECONSTRUCTION
# =============================================================================

banner(
    "3. TARGET AND CRYPTO PREDICTOR RECONSTRUCTION"
)

market_verify_columns = [
    "Date",
    "BTC_Return",
    "BTC_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "ETH_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_ETH_Volume",
]

market_verify = (
    market[
        market_verify_columns
    ]
    .copy()
)

verify = (
    df[
        [
            "Date",
            "Asset",
            TARGET,
            "Own_Lagged_Return",
            "Lagged_Log_Crypto_Volume",
            "Cross_Crypto_Lagged_Return",
        ]
    ]
    .merge(
        market_verify,
        on="Date",
        how="left",
        validate="many_to_one",
    )
)

verify[
    "Expected_Target_Return"
] = np.where(
    verify["Asset"].eq("BTC"),
    verify["BTC_Return"],
    verify["ETH_Return"],
)

verify[
    "Expected_Own_Lagged_Return"
] = np.where(
    verify["Asset"].eq("BTC"),
    verify["BTC_Lagged_Return"],
    verify["ETH_Lagged_Return"],
)

verify[
    "Expected_Lagged_Log_Crypto_Volume"
] = np.where(
    verify["Asset"].eq("BTC"),
    verify["Lagged_Log_BTC_Volume"],
    verify["Lagged_Log_ETH_Volume"],
)

verify[
    "Expected_Cross_Crypto_Lagged_Return"
] = np.where(
    verify["Asset"].eq("BTC"),
    verify["ETH_Lagged_Return"],
    verify["BTC_Lagged_Return"],
)

reconstruction_specs = [
    (
        TARGET,
        "Expected_Target_Return",
        "Target_Return",
    ),
    (
        "Own_Lagged_Return",
        "Expected_Own_Lagged_Return",
        "Own_Lagged_Return",
    ),
    (
        "Lagged_Log_Crypto_Volume",
        "Expected_Lagged_Log_Crypto_Volume",
        "Lagged_Log_Crypto_Volume",
    ),
    (
        "Cross_Crypto_Lagged_Return",
        "Expected_Cross_Crypto_Lagged_Return",
        "Cross_Crypto_Lagged_Return",
    ),
]

reconstruction_issue_frames = []

for (
    observed_col,
    expected_col,
    label,
) in reconstruction_specs:

    match = same_numeric(
        verify[observed_col],
        verify[expected_col],
    )

    mismatches = (
        verify.loc[
            ~match,
            [
                "Date",
                "Asset",
                observed_col,
                expected_col,
            ],
        ]
        .copy()
    )

    mismatches["Check"] = label

    mismatches = (
        mismatches.rename(
            columns={
                observed_col:
                    "Observed_Value",
                expected_col:
                    "Expected_Value",
            }
        )
    )

    reconstruction_issue_frames.append(
        mismatches[
            [
                "Date",
                "Asset",
                "Check",
                "Observed_Value",
                "Expected_Value",
            ]
        ]
    )

    add_qc(
        qc_rows,
        f"{label}_Reconstruction_Mismatches",
        0,
        int((~match).sum()),
        int((~match).sum()) == 0,
        "Reconstruction",
    )

reconstruction_issues = pd.concat(
    reconstruction_issue_frames,
    ignore_index=True,
)


# =============================================================================
# 13. INDEPENDENT OWN-RETURN CALENDAR LAG
# =============================================================================

calendar_lag = (
    df[
        [
            "Date",
            "Asset",
            TARGET,
            "Own_Lagged_Return",
        ]
    ]
    .copy()
)

calendar_lag = (
    calendar_lag
    .sort_values(
        [
            "Asset",
            "Date",
        ]
    )
    .reset_index(drop=True)
)

calendar_lag[
    "Independent_t1_Return"
] = (
    calendar_lag
    .groupby("Asset")[TARGET]
    .shift(1)
)

calendar_lag_match = same_numeric(
    calendar_lag[
        "Own_Lagged_Return"
    ],
    calendar_lag[
        "Independent_t1_Return"
    ],
)

calendar_lag_mismatch_count = int(
    (~calendar_lag_match).sum()
)

add_qc(
    qc_rows,
    "Own_Return_Independent_Calendar_t1_Mismatches",
    0,
    calendar_lag_mismatch_count,
    calendar_lag_mismatch_count == 0,
    "Lag_Validation",
)


# =============================================================================
# 14. REDDIT SECTION 06 → SECTION 07 RECONSTRUCTION
# =============================================================================

banner(
    "4. REDDIT INTEGRATION AND LAG VALIDATION"
)

reddit_compare_cols = [
    "Date",
    "Asset",
    "Lagged_Reddit_Sentiment",
    "Lagged_Log_Reddit_Post_Count",
]

optional_reddit_compare_cols = [
    "Lagged_Reddit_Source_Date",
    "Reddit_Sentiment_Lag_2",
    "Log_Reddit_Post_Count_Lag_2",
    "Reddit_Source_Date_Lag_2",
    "Reddit_Sentiment_Lag_3",
    "Log_Reddit_Post_Count_Lag_3",
    "Reddit_Source_Date_Lag_3",
    "Reddit_Sentiment_Lag_7",
    "Log_Reddit_Post_Count_Lag_7",
    "Reddit_Source_Date_Lag_7",
]

for col in optional_reddit_compare_cols:

    if (
        col in reddit06.columns
        and col in df.columns
    ):
        reddit_compare_cols.append(col)

reddit_reference = (
    reddit06[
        reddit_compare_cols
    ]
    .copy()
)

reddit_merge = (
    df[
        ["Date", "Asset"]
        + [
            col
            for col in reddit_compare_cols
            if col not in [
                "Date",
                "Asset",
            ]
        ]
    ]
    .merge(
        reddit_reference,
        on=[
            "Date",
            "Asset",
        ],
        how="left",
        suffixes=(
            "_s07",
            "_s06",
        ),
        validate="one_to_one",
    )
)

reddit_reconstruction_rows = []

for col in [
    c
    for c in reddit_compare_cols
    if c not in [
        "Date",
        "Asset",
    ]
]:

    left_col = (
        f"{col}_s07"
    )

    right_col = (
        f"{col}_s06"
    )

    if "Date" in col:

        match = same_dates(
            reddit_merge[left_col],
            reddit_merge[right_col],
        )

    else:

        match = same_numeric(
            reddit_merge[left_col],
            reddit_merge[right_col],
        )

    mismatch_count = int(
        (~match).sum()
    )

    add_qc(
        qc_rows,
        (
            f"Reddit_{col}_"
            "Section06_to_Section07_Mismatches"
        ),
        0,
        mismatch_count,
        mismatch_count == 0,
        "Reddit_Reconstruction",
    )

    if mismatch_count > 0:

        issue = (
            reddit_merge.loc[
                ~match,
                [
                    "Date",
                    "Asset",
                    left_col,
                    right_col,
                ],
            ]
            .copy()
        )

        issue["Variable"] = col

        issue = (
            issue.rename(
                columns={
                    left_col:
                        "Section07_Value",
                    right_col:
                        "Section06_Value",
                }
            )
        )

        reddit_reconstruction_rows.append(
            issue[
                [
                    "Date",
                    "Asset",
                    "Variable",
                    "Section07_Value",
                    "Section06_Value",
                ]
            ]
        )

if reddit_reconstruction_rows:

    reddit_reconstruction_issues = (
        pd.concat(
            reddit_reconstruction_rows,
            ignore_index=True,
        )
    )

else:

    reddit_reconstruction_issues = (
        pd.DataFrame(
            columns=[
                "Date",
                "Asset",
                "Variable",
                "Section07_Value",
                "Section06_Value",
            ]
        )
    )


# =============================================================================
# 15. REDDIT SOURCE-DATE TIMING
# =============================================================================

timing_rows = []

if "Lagged_Reddit_Source_Date" in df.columns:

    reddit_source = pd.to_datetime(
        df["Lagged_Reddit_Source_Date"],
        errors="coerce",
    )

    reddit_has_source = (
        reddit_source.notna()
    )

    reddit_gap = (
        df["Date"]
        - reddit_source
    ).dt.days

    same_or_future = (
        reddit_has_source
        & (
            reddit_source
            >= df["Date"]
        )
    )

    wrong_t1_gap = (
        reddit_has_source
        & reddit_gap.ne(1)
    )

    reddit_t1_pass = bool(
        same_or_future.sum() == 0
        and wrong_t1_gap.sum() == 0
    )

    timing_rows.append(
        {
            "Variable_Group":
                "Reddit_Primary_t1",

            "Rows_With_Source_Date":
                int(
                    reddit_has_source.sum()
                ),

            "Same_Or_Future_Source_Dates":
                int(
                    same_or_future.sum()
                ),

            "Wrong_Calendar_Gap":
                int(
                    wrong_t1_gap.sum()
                ),

            "Expected_Gap_Days":
                1,

            "Pass":
                reddit_t1_pass,
        }
    )

    add_qc(
        qc_rows,
        "Reddit_Primary_t1_No_Lookahead",
        True,
        reddit_t1_pass,
        reddit_t1_pass,
        "Leakage",
    )

else:

    print(
        "\nNOTE: Lagged_Reddit_Source_Date is not present "
        "in the Section 07 file."
    )


# =============================================================================
# 16. ALTERNATIVE REDDIT LAG TIMING
# =============================================================================

for (
    lag,
    lag_info,
) in ALTERNATIVE_REDDIT_LAGS.items():

    source_col = (
        lag_info["source_date"]
    )

    if source_col not in df.columns:
        continue

    source_date = pd.to_datetime(
        df[source_col],
        errors="coerce",
    )

    has_source = (
        source_date.notna()
    )

    gap = (
        df["Date"]
        - source_date
    ).dt.days

    same_or_future = (
        has_source
        & (
            source_date
            >= df["Date"]
        )
    )

    wrong_gap = (
        has_source
        & gap.ne(lag)
    )

    passed = bool(
        same_or_future.sum() == 0
        and wrong_gap.sum() == 0
    )

    timing_rows.append(
        {
            "Variable_Group":
                f"Reddit_t{lag}",

            "Rows_With_Source_Date":
                int(
                    has_source.sum()
                ),

            "Same_Or_Future_Source_Dates":
                int(
                    same_or_future.sum()
                ),

            "Wrong_Calendar_Gap":
                int(
                    wrong_gap.sum()
                ),

            "Expected_Gap_Days":
                lag,

            "Pass":
                passed,
        }
    )

    add_qc(
        qc_rows,
        f"Reddit_t{lag}_No_Lookahead",
        True,
        passed,
        passed,
        "Leakage",
    )


# =============================================================================
# 17. TRADITIONAL MARKET RECONSTRUCTION + NO-LOOKAHEAD
# =============================================================================

banner(
    "5. TRADITIONAL-MARKET ALIGNMENT AND LEAKAGE VALIDATION"
)

market_alignment_cols = [
    "Date"
]

for item in TRADITIONAL_ALIGNMENT.values():

    market_alignment_cols.extend(
        [
            item["value"],
            item["source_date"],
        ]
    )

market_alignment_cols = list(
    dict.fromkeys(
        market_alignment_cols
    )
)

market_alignment_reference = (
    market[
        market_alignment_cols
    ]
    .copy()
)

market_alignment_merge = (
    df.merge(
        market_alignment_reference,
        on="Date",
        how="left",
        suffixes=(
            "_s07",
            "_market",
        ),
        validate="many_to_one",
    )
)

traditional_reconstruction_issues = []

for (
    label,
    item,
) in TRADITIONAL_ALIGNMENT.items():

    value_col = (
        item["value"]
    )

    source_col = (
        item["source_date"]
    )

    value_s07 = (
        f"{value_col}_s07"
        if f"{value_col}_s07"
        in market_alignment_merge.columns
        else value_col
    )

    value_market = (
        f"{value_col}_market"
    )

    source_s07 = (
        f"{source_col}_s07"
        if f"{source_col}_s07"
        in market_alignment_merge.columns
        else source_col
    )

    source_market = (
        f"{source_col}_market"
    )

    value_match = same_numeric(
        market_alignment_merge[
            value_s07
        ],
        market_alignment_merge[
            value_market
        ],
    )

    source_match = same_dates(
        market_alignment_merge[
            source_s07
        ],
        market_alignment_merge[
            source_market
        ],
    )

    value_mismatch_count = int(
        (~value_match).sum()
    )

    source_mismatch_count = int(
        (~source_match).sum()
    )

    add_qc(
        qc_rows,
        f"{label}_Aligned_Value_Mismatches",
        0,
        value_mismatch_count,
        value_mismatch_count == 0,
        "Traditional_Reconstruction",
    )

    add_qc(
        qc_rows,
        f"{label}_Source_Date_Mismatches",
        0,
        source_mismatch_count,
        source_mismatch_count == 0,
        "Traditional_Reconstruction",
    )

    source_dates = pd.to_datetime(
        market_alignment_merge[
            source_s07
        ],
        errors="coerce",
    )

    has_source = (
        source_dates.notna()
    )

    same_or_future = (
        has_source
        & (
            source_dates
            >= market_alignment_merge[
                "Date"
            ]
        )
    )

    strict_pre_target_pass = bool(
        same_or_future.sum() == 0
    )

    timing_rows.append(
        {
            "Variable_Group":
                f"{label}_Strict_PreTarget",

            "Rows_With_Source_Date":
                int(
                    has_source.sum()
                ),

            "Same_Or_Future_Source_Dates":
                int(
                    same_or_future.sum()
                ),

            "Wrong_Calendar_Gap":
                np.nan,

            "Expected_Gap_Days":
                ">=1, varies with market closure",

            "Pass":
                strict_pre_target_pass,
        }
    )

    add_qc(
        qc_rows,
        (
            f"{label}_"
            "Strict_PreTarget_No_Lookahead"
        ),
        True,
        strict_pre_target_pass,
        strict_pre_target_pass,
        "Leakage",
    )

    issue_mask = (
        ~value_match
        | ~source_match
        | same_or_future
    )

    if issue_mask.any():

        temp = (
            market_alignment_merge.loc[
                issue_mask,
                [
                    "Date",
                    "Asset",
                    value_s07,
                    value_market,
                    source_s07,
                    source_market,
                ],
            ]
            .copy()
        )

        temp[
            "Variable_Group"
        ] = label

        temp = (
            temp.rename(
                columns={
                    value_s07:
                        "Section07_Value",

                    value_market:
                        "Market_Value",

                    source_s07:
                        "Section07_Source_Date",

                    source_market:
                        "Market_Source_Date",
                }
            )
        )

        traditional_reconstruction_issues.append(
            temp[
                [
                    "Date",
                    "Asset",
                    "Variable_Group",
                    "Section07_Value",
                    "Market_Value",
                    "Section07_Source_Date",
                    "Market_Source_Date",
                ]
            ]
        )

if traditional_reconstruction_issues:

    traditional_reconstruction_issues_df = (
        pd.concat(
            traditional_reconstruction_issues,
            ignore_index=True,
        )
    )

else:

    traditional_reconstruction_issues_df = (
        pd.DataFrame(
            columns=[
                "Date",
                "Asset",
                "Variable_Group",
                "Section07_Value",
                "Market_Value",
                "Section07_Source_Date",
                "Market_Source_Date",
            ]
        )
    )

timing_validation = pd.DataFrame(
    timing_rows
)


# =============================================================================
# 18. RANGE / TRANSFORMATION SANITY CHECKS
# =============================================================================

banner(
    "6. PREDICTOR RANGE CHECKS"
)

sentiment_numeric = pd.to_numeric(
    df[SENTIMENT_PREDICTOR],
    errors="coerce",
)

sentiment_out_of_range = (
    sentiment_numeric.notna()
    & (
        sentiment_numeric.lt(
            -1.0 - FLOAT_ATOL
        )
        | sentiment_numeric.gt(
            1.0 + FLOAT_ATOL
        )
    )
)

add_qc(
    qc_rows,
    "Lagged_Reddit_Sentiment_Within_Minus1_Plus1",
    0,
    int(
        sentiment_out_of_range.sum()
    ),
    (
        sentiment_out_of_range.sum()
        == 0
    ),
    "Range",
)

activity_numeric = pd.to_numeric(
    df[ACTIVITY_PREDICTOR],
    errors="coerce",
)

activity_negative = (
    activity_numeric.notna()
    & activity_numeric.lt(
        -FLOAT_ATOL
    )
)

add_qc(
    qc_rows,
    "Lagged_Log_Reddit_Post_Count_Negative_Values",
    0,
    int(
        activity_negative.sum()
    ),
    (
        activity_negative.sum()
        == 0
    ),
    "Range",
)

volume_numeric = pd.to_numeric(
    df[
        "Lagged_Log_Crypto_Volume"
    ],
    errors="coerce",
)

volume_negative = (
    volume_numeric.notna()
    & volume_numeric.lt(
        -FLOAT_ATOL
    )
)

add_qc(
    qc_rows,
    "Lagged_Log_Crypto_Volume_Negative_Values",
    0,
    int(
        volume_negative.sum()
    ),
    (
        volume_negative.sum()
        == 0
    ),
    "Range",
)


# =============================================================================
# 19. MISSINGNESS AUDIT
# =============================================================================

banner(
    "7. MISSING-VALUE AUDIT"
)

missingness_rows = []
missing_date_rows = []

for asset in EXPECTED_ASSETS:

    asset_df = (
        df.loc[
            df["Asset"].eq(asset)
        ]
        .copy()
    )

    for variable in ALL_PRIMARY_NUMERIC_VARIABLES:

        missing_mask = (
            asset_df[
                variable
            ].isna()
        )

        available = int(
            (~missing_mask).sum()
        )

        missing = int(
            missing_mask.sum()
        )

        valid_dates = (
            asset_df.loc[
                ~missing_mask,
                "Date",
            ]
        )

        missingness_rows.append(
            {
                "Asset": asset,
                "Variable": variable,
                "Total_Rows": len(
                    asset_df
                ),
                "Available": available,
                "Missing": missing,
                "Missing_Percent": safe_pct(
                    missing,
                    len(asset_df),
                ),
                "First_Available_Date": (
                    valid_dates.min()
                    if not valid_dates.empty
                    else pd.NaT
                ),
                "Last_Available_Date": (
                    valid_dates.max()
                    if not valid_dates.empty
                    else pd.NaT
                ),
            }
        )

        missing_subset = (
            asset_df.loc[
                missing_mask,
                [
                    "Date",
                    "Asset",
                ],
            ]
            .copy()
        )

        if not missing_subset.empty:

            missing_subset[
                "Variable"
            ] = variable

            if (
                variable
                == SENTIMENT_PREDICTOR
                and
                "Reddit_Observation_Status_Lag_1"
                in asset_df.columns
            ):

                status_map = (
                    asset_df.set_index(
                        [
                            "Date",
                            "Asset",
                        ]
                    )[
                        "Reddit_Observation_Status_Lag_1"
                    ]
                )

                missing_subset[
                    "Context_Status"
                ] = [
                    status_map.get(
                        (
                            row.Date,
                            row.Asset,
                        ),
                        np.nan,
                    )
                    for row
                    in missing_subset.itertuples()
                ]

            else:

                missing_subset[
                    "Context_Status"
                ] = np.nan

            missing_date_rows.append(
                missing_subset
            )

missingness = pd.DataFrame(
    missingness_rows
)

if missing_date_rows:

    missing_dates = pd.concat(
        missing_date_rows,
        ignore_index=True,
    )

else:

    missing_dates = pd.DataFrame(
        columns=[
            "Date",
            "Asset",
            "Variable",
            "Context_Status",
        ]
    )


# =============================================================================
# 20. MODEL ELIGIBILITY FLAGS
# =============================================================================

banner(
    "8. MODEL SAMPLE CONSTRUCTION"
)

df[
    "Eligible_Benchmark"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *BENCHMARK_PREDICTORS,
    ],
)

df[
    "Eligible_Activity"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *ACTIVITY_MODEL_PREDICTORS,
    ],
)

df[
    "Eligible_Sentiment"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *SENTIMENT_MODEL_PREDICTORS,
    ],
)

df[
    "Eligible_Both"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *BOTH_MODEL_PREDICTORS,
    ],
)

df[
    "Eligible_Benchmark_Cross"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *BENCHMARK_CROSS_PREDICTORS,
    ],
)

df[
    "Eligible_Activity_Cross"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *ACTIVITY_CROSS_PREDICTORS,
    ],
)

df[
    "Eligible_Sentiment_Cross"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *SENTIMENT_CROSS_PREDICTORS,
    ],
)

df[
    "Eligible_Both_Cross"
] = finite_complete_mask(
    df,
    [
        TARGET,
        *BOTH_CROSS_PREDICTORS,
    ],
)


# =============================================================================
# 21. FAIR COMPARISON SAMPLES
# =============================================================================

df[
    "Activity_Comparison_Sample"
] = df[
    "Eligible_Activity"
]

df[
    "Sentiment_Comparison_Sample"
] = df[
    "Eligible_Sentiment"
]

df[
    "Both_Comparison_Sample"
] = df[
    "Eligible_Both"
]

df[
    "Common_Main_Model_Sample"
] = df[
    "Eligible_Both"
]

df[
    "Common_Cross_Robustness_Sample"
] = df[
    "Eligible_Both_Cross"
]


# =============================================================================
# 22. TRAIN / OOS FLAGS
# =============================================================================

df[
    "Evaluation_Period"
] = np.select(
    [
        df["Date"].between(
            INITIAL_TRAIN_START,
            INITIAL_TRAIN_END,
            inclusive="both",
        ),

        df["Date"].between(
            OOS_START,
            OOS_END,
            inclusive="both",
        ),
    ],
    [
        "INITIAL_TRAIN",
        "OUT_OF_SAMPLE",
    ],
    default=(
        "OUTSIDE_DEFINED_PERIOD"
    ),
)

df[
    "Is_Initial_Train_Period"
] = (
    df[
        "Evaluation_Period"
    ]
    .eq(
        "INITIAL_TRAIN"
    )
)

df[
    "Is_OOS_Period"
] = (
    df[
        "Evaluation_Period"
    ]
    .eq(
        "OUT_OF_SAMPLE"
    )
)

df[
    "Day_Of_Week"
] = (
    df["Date"]
    .dt.day_name()
)

df[
    "Is_Weekend"
] = (
    df["Date"]
    .dt.dayofweek
    >= 5
)

df[
    "Year"
] = (
    df["Date"]
    .dt.year
)


# =============================================================================
# 23. MODEL SAMPLE SUMMARY
# =============================================================================

sample_flags = [
    "Eligible_Benchmark",
    "Eligible_Activity",
    "Eligible_Sentiment",
    "Eligible_Both",
    "Activity_Comparison_Sample",
    "Sentiment_Comparison_Sample",
    "Both_Comparison_Sample",
    "Common_Main_Model_Sample",
    "Eligible_Benchmark_Cross",
    "Eligible_Activity_Cross",
    "Eligible_Sentiment_Cross",
    "Eligible_Both_Cross",
    "Common_Cross_Robustness_Sample",
]

sample_summary_rows = []

for asset in EXPECTED_ASSETS:

    asset_df = (
        df.loc[
            df["Asset"].eq(asset)
        ]
    )

    for flag in sample_flags:

        for period in [
            "FULL_SAMPLE",
            "INITIAL_TRAIN",
            "OUT_OF_SAMPLE",
        ]:

            if period == "FULL_SAMPLE":

                period_mask = pd.Series(
                    True,
                    index=asset_df.index,
                )

            else:

                period_mask = (
                    asset_df[
                        "Evaluation_Period"
                    ]
                    .eq(period)
                )

            mask = (
                asset_df[flag]
                & period_mask
            )

            dates = (
                asset_df.loc[
                    mask,
                    "Date",
                ]
            )

            weekend_count = int(
                asset_df.loc[
                    mask,
                    "Is_Weekend",
                ]
                .sum()
            )

            sample_summary_rows.append(
                {
                    "Asset": asset,
                    "Sample": flag,
                    "Period": period,
                    "N": int(
                        mask.sum()
                    ),
                    "First_Date": (
                        dates.min()
                        if not dates.empty
                        else pd.NaT
                    ),
                    "Last_Date": (
                        dates.max()
                        if not dates.empty
                        else pd.NaT
                    ),
                    "Weekend_Observations":
                        weekend_count,
                    "Weekday_Observations":
                        (
                            int(
                                mask.sum()
                            )
                            - weekend_count
                        ),
                }
            )

sample_summary = pd.DataFrame(
    sample_summary_rows
)


# =============================================================================
# 24. COMMON-SAMPLE VALIDATION
# =============================================================================

common_sample_rows = []

comparison_map = {
    "Benchmark_vs_Activity": {
        "flag":
            "Activity_Comparison_Sample",

        "required": [
            TARGET,
            *ACTIVITY_MODEL_PREDICTORS,
        ],
    },

    "Benchmark_vs_Sentiment": {
        "flag":
            "Sentiment_Comparison_Sample",

        "required": [
            TARGET,
            *SENTIMENT_MODEL_PREDICTORS,
        ],
    },

    "Benchmark_vs_Both": {
        "flag":
            "Both_Comparison_Sample",

        "required": [
            TARGET,
            *BOTH_MODEL_PREDICTORS,
        ],
    },

    "M0_M1_M2_M3_Common": {
        "flag":
            "Common_Main_Model_Sample",

        "required": [
            TARGET,
            *BOTH_MODEL_PREDICTORS,
        ],
    },
}

for asset in EXPECTED_ASSETS:

    asset_mask = (
        df["Asset"].eq(asset)
    )

    for (
        comparison_name,
        info,
    ) in comparison_map.items():

        sample_mask = (
            asset_mask
            & df[
                info["flag"]
            ]
        )

        required_complete = (
            finite_complete_mask(
                df,
                info["required"],
            )
        )

        invalid_inside_sample = int(
            (
                sample_mask
                & ~required_complete
            )
            .sum()
        )

        passed = (
            invalid_inside_sample
            == 0
        )

        common_sample_rows.append(
            {
                "Asset":
                    asset,

                "Comparison":
                    comparison_name,

                "Sample_Flag":
                    info["flag"],

                "N":
                    int(
                        sample_mask.sum()
                    ),

                "Rows_With_Missing_Required_Data":
                    invalid_inside_sample,

                "Pass":
                    passed,
            }
        )

        add_qc(
            qc_rows,
            (
                f"{asset}_"
                f"{comparison_name}_"
                "Complete_Data_Within_Sample"
            ),
            0,
            invalid_inside_sample,
            passed,
            "Sample_Construction",
        )

common_sample_validation = (
    pd.DataFrame(
        common_sample_rows
    )
)


# =============================================================================
# 25. TRAIN / OOS DESIGN VALIDATION
# =============================================================================

period_outside_count = int(
    df[
        "Evaluation_Period"
    ]
    .eq(
        "OUTSIDE_DEFINED_PERIOD"
    )
    .sum()
)

add_qc(
    qc_rows,
    "Rows_Outside_Train_Or_OOS_Definition",
    0,
    period_outside_count,
    period_outside_count == 0,
    "Evaluation_Design",
)

train_calendar_days = len(
    pd.date_range(
        INITIAL_TRAIN_START,
        INITIAL_TRAIN_END,
        freq="D",
    )
)

oos_calendar_days = len(
    pd.date_range(
        OOS_START,
        OOS_END,
        freq="D",
    )
)

for asset in EXPECTED_ASSETS:

    asset_train_rows = int(
        (
            df["Asset"].eq(asset)
            & df[
                "Is_Initial_Train_Period"
            ]
        )
        .sum()
    )

    asset_oos_rows = int(
        (
            df["Asset"].eq(asset)
            & df[
                "Is_OOS_Period"
            ]
        )
        .sum()
    )

    add_qc(
        qc_rows,
        f"{asset}_Training_Calendar_Rows",
        train_calendar_days,
        asset_train_rows,
        (
            asset_train_rows
            == train_calendar_days
        ),
        "Evaluation_Design",
    )

    add_qc(
        qc_rows,
        f"{asset}_OOS_Calendar_Rows",
        oos_calendar_days,
        asset_oos_rows,
        (
            asset_oos_rows
            == oos_calendar_days
        ),
        "Evaluation_Design",
    )

    sentiment_train_n = int(
        (
            df["Asset"].eq(asset)
            & df[
                "Is_Initial_Train_Period"
            ]
            & df[
                "Sentiment_Comparison_Sample"
            ]
        )
        .sum()
    )

    sentiment_oos_n = int(
        (
            df["Asset"].eq(asset)
            & df[
                "Is_OOS_Period"
            ]
            & df[
                "Sentiment_Comparison_Sample"
            ]
        )
        .sum()
    )

    add_qc(
        qc_rows,
        (
            f"{asset}_"
            "Sentiment_Comparison_Training_N_Positive"
        ),
        True,
        sentiment_train_n > 0,
        sentiment_train_n > 0,
        "Evaluation_Design",
        notes=(
            f"N={sentiment_train_n}"
        ),
    )

    add_qc(
        qc_rows,
        (
            f"{asset}_"
            "Sentiment_Comparison_OOS_N_Positive"
        ),
        True,
        sentiment_oos_n > 0,
        sentiment_oos_n > 0,
        "Evaluation_Design",
        notes=(
            f"N={sentiment_oos_n}"
        ),
    )


# =============================================================================
# 26. WEEKEND PRESERVATION
# =============================================================================

weekend_summary_rows = []

for asset in EXPECTED_ASSETS:

    for flag in [
        "Eligible_Benchmark",
        "Sentiment_Comparison_Sample",
        "Common_Main_Model_Sample",
    ]:

        mask = (
            df["Asset"].eq(asset)
            & df[
                "Is_OOS_Period"
            ]
            & df[flag]
        )

        n_total = int(
            mask.sum()
        )

        n_weekend = int(
            (
                mask
                & df[
                    "Is_Weekend"
                ]
            )
            .sum()
        )

        weekend_summary_rows.append(
            {
                "Asset":
                    asset,

                "Sample":
                    flag,

                "OOS_Total":
                    n_total,

                "OOS_Weekend":
                    n_weekend,

                "OOS_Weekday":
                    (
                        n_total
                        - n_weekend
                    ),

                "Weekend_Preserved":
                    n_weekend > 0,
            }
        )

        add_qc(
            qc_rows,
            (
                f"{asset}_"
                f"{flag}_"
                "OOS_Weekend_Preserved"
            ),
            True,
            n_weekend > 0,
            n_weekend > 0,
            "Evaluation_Design",
            notes=(
                "OOS weekend observations="
                f"{n_weekend}"
            ),
        )

weekend_summary = pd.DataFrame(
    weekend_summary_rows
)


# =============================================================================
# 27. DESCRIPTIVE STATISTICS
# =============================================================================

banner(
    "9. DESCRIPTIVE STATISTICS"
)

descriptive_rows = []

descriptive_samples = {
    "AVAILABLE_VARIABLE_OBSERVATIONS":
        None,

    "COMMON_MAIN_MODEL_SAMPLE":
        "Common_Main_Model_Sample",
}

for asset in EXPECTED_ASSETS:

    asset_df = (
        df.loc[
            df["Asset"].eq(asset)
        ]
        .copy()
    )

    for (
        sample_name,
        sample_flag,
    ) in descriptive_samples.items():

        if sample_flag is None:

            sample_df = (
                asset_df
            )

        else:

            sample_df = (
                asset_df.loc[
                    asset_df[
                        sample_flag
                    ]
                ]
            )

        for variable in (
            ALL_PRIMARY_NUMERIC_VARIABLES
        ):

            values = pd.to_numeric(
                sample_df[
                    variable
                ],
                errors="coerce",
            )

            values = (
                values
                .replace(
                    [
                        np.inf,
                        -np.inf,
                    ],
                    np.nan,
                )
                .dropna()
            )

            if values.empty:

                descriptive_rows.append(
                    {
                        "Asset":
                            asset,

                        "Sample":
                            sample_name,

                        "Variable":
                            variable,

                        "N":
                            0,

                        "Mean":
                            np.nan,

                        "Std":
                            np.nan,

                        "Min":
                            np.nan,

                        "P01":
                            np.nan,

                        "P05":
                            np.nan,

                        "P25":
                            np.nan,

                        "Median":
                            np.nan,

                        "P75":
                            np.nan,

                        "P95":
                            np.nan,

                        "P99":
                            np.nan,

                        "Max":
                            np.nan,

                        "Skewness":
                            np.nan,

                        "Excess_Kurtosis":
                            np.nan,

                        "Zero_Count":
                            np.nan,
                    }
                )

                continue

            descriptive_rows.append(
                {
                    "Asset":
                        asset,

                    "Sample":
                        sample_name,

                    "Variable":
                        variable,

                    "N":
                        len(values),

                    "Mean":
                        values.mean(),

                    "Std":
                        values.std(
                            ddof=1
                        ),

                    "Min":
                        values.min(),

                    "P01":
                        values.quantile(
                            0.01
                        ),

                    "P05":
                        values.quantile(
                            0.05
                        ),

                    "P25":
                        values.quantile(
                            0.25
                        ),

                    "Median":
                        values.median(),

                    "P75":
                        values.quantile(
                            0.75
                        ),

                    "P95":
                        values.quantile(
                            0.95
                        ),

                    "P99":
                        values.quantile(
                            0.99
                        ),

                    "Max":
                        values.max(),

                    "Skewness":
                        values.skew(),

                    "Excess_Kurtosis":
                        values.kurt(),

                    "Zero_Count":
                        int(
                            np.isclose(
                                values,
                                0.0,
                                atol=FLOAT_ATOL,
                            )
                            .sum()
                        ),
                }
            )

descriptive_statistics = (
    pd.DataFrame(
        descriptive_rows
    )
)


# =============================================================================
# 28. PREDICTOR DISTRIBUTIONS
# =============================================================================

distribution_rows = []

for asset in EXPECTED_ASSETS:

    asset_df = (
        df.loc[
            df["Asset"].eq(asset)
            & df[
                "Common_Main_Model_Sample"
            ]
        ]
    )

    for predictor in BOTH_MODEL_PREDICTORS:

        x = (
            pd.to_numeric(
                asset_df[
                    predictor
                ],
                errors="coerce",
            )
            .dropna()
        )

        if x.empty:
            continue

        zero_count = int(
            np.isclose(
                x,
                0.0,
                atol=FLOAT_ATOL,
            )
            .sum()
        )

        distribution_rows.append(
            {
                "Asset":
                    asset,

                "Predictor":
                    predictor,

                "N":
                    len(x),

                "Unique_Values":
                    x.nunique(),

                "Mean":
                    x.mean(),

                "Std":
                    x.std(
                        ddof=1
                    ),

                "Skewness":
                    x.skew(),

                "Excess_Kurtosis":
                    x.kurt(),

                "Minimum":
                    x.min(),

                "Maximum":
                    x.max(),

                "Zero_Count":
                    zero_count,

                "Zero_Percent":
                    safe_pct(
                        zero_count,
                        len(x),
                    ),
            }
        )

predictor_distributions = (
    pd.DataFrame(
        distribution_rows
    )
)


# =============================================================================
# 29. CORRELATION MATRICES
# =============================================================================

banner(
    "10. CORRELATION AND MULTICOLLINEARITY DIAGNOSTICS"
)

correlation_variables = [
    TARGET,
    *BOTH_MODEL_PREDICTORS,
]

correlation_variables = list(
    dict.fromkeys(
        correlation_variables
    )
)

correlation_long_rows = []

for asset in EXPECTED_ASSETS:

    corr_data = (
        df.loc[
            df["Asset"].eq(asset)
            & df[
                "Common_Main_Model_Sample"
            ],
            correlation_variables,
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    corr_matrix = (
        corr_data.corr(
            method="pearson"
        )
    )

    corr_matrix.to_csv(
        OUTPUT_DIR
        / (
            f"stage08_correlation_matrix_"
            f"{asset}.csv"
        )
    )

    for row_var in corr_matrix.index:

        for col_var in corr_matrix.columns:

            correlation_long_rows.append(
                {
                    "Asset":
                        asset,

                    "Variable_1":
                        row_var,

                    "Variable_2":
                        col_var,

                    "Correlation":
                        corr_matrix.loc[
                            row_var,
                            col_var,
                        ],
                }
            )

correlation_long = (
    pd.DataFrame(
        correlation_long_rows
    )
)


# =============================================================================
# 30. HIGH-CORRELATION AUDIT
# =============================================================================

high_correlation_rows = []

for asset in EXPECTED_ASSETS:

    asset_corr = (
        correlation_long.loc[
            correlation_long[
                "Asset"
            ].eq(asset)
        ]
    )

    predictor_corr = (
        asset_corr.loc[
            asset_corr[
                "Variable_1"
            ].isin(
                BOTH_MODEL_PREDICTORS
            )
            & asset_corr[
                "Variable_2"
            ].isin(
                BOTH_MODEL_PREDICTORS
            )
            & (
                asset_corr[
                    "Variable_1"
                ]
                <
                asset_corr[
                    "Variable_2"
                ]
            )
        ]
        .copy()
    )

    predictor_corr = (
        predictor_corr.loc[
            predictor_corr[
                "Correlation"
            ]
            .abs()
            .ge(0.80)
        ]
    )

    if not predictor_corr.empty:

        high_correlation_rows.append(
            predictor_corr
        )

if high_correlation_rows:

    high_correlations = (
        pd.concat(
            high_correlation_rows,
            ignore_index=True,
        )
    )

else:

    high_correlations = (
        pd.DataFrame(
            columns=[
                "Asset",
                "Variable_1",
                "Variable_2",
                "Correlation",
            ]
        )
    )


# =============================================================================
# 31. CORRECTED VARIANCE INFLATION FACTORS
# =============================================================================

vif_frames = []

for asset in EXPECTED_ASSETS:

    common_df = (
        df.loc[
            df[
                "Common_Main_Model_Sample"
            ]
        ]
        .copy()
    )

    vif_frames.append(
        vif_table(
            common_df,
            asset,
            BENCHMARK_PREDICTORS,
            (
                "Benchmark_on_"
                "Common_Main_Sample"
            ),
        )
    )

    vif_frames.append(
        vif_table(
            common_df,
            asset,
            BOTH_MODEL_PREDICTORS,
            (
                "Full_Both_on_"
                "Common_Main_Sample"
            ),
        )
    )

vif_diagnostics = pd.concat(
    vif_frames,
    ignore_index=True,
)

vif_diagnostics[
    "VIF_Above_5"
] = (
    vif_diagnostics[
        "VIF"
    ]
    > 5
)

vif_diagnostics[
    "VIF_Above_10"
] = (
    vif_diagnostics[
        "VIF"
    ]
    > 10
)


# =============================================================================
# 32. EXTREME RETURN AUDIT
# =============================================================================

banner(
    "11. EXTREME-RETURN AUDIT"
)

target_numeric = pd.to_numeric(
    df[TARGET],
    errors="coerce",
)

df[
    "_Abs_Target_Return"
] = (
    target_numeric.abs()
)

threshold_extremes = (
    df.loc[
        target_numeric.notna()
        & (
            df[
                "_Abs_Target_Return"
            ]
            >= EXTREME_RETURN_THRESHOLD
        ),
        [
            "Date",
            "Asset",
            TARGET,
            "_Abs_Target_Return",
        ],
    ]
    .copy()
)

threshold_extremes[
    "Audit_Type"
] = (
    "ABS_RETURN_GE_"
    f"{EXTREME_RETURN_THRESHOLD}"
)

top_extremes = (
    df.loc[
        target_numeric.notna(),
        [
            "Date",
            "Asset",
            TARGET,
            "_Abs_Target_Return",
        ],
    ]
    .sort_values(
        [
            "Asset",
            "_Abs_Target_Return",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .groupby(
        "Asset",
        as_index=False,
    )
    .head(10)
    .copy()
)

top_extremes[
    "Audit_Type"
] = (
    "TOP_10_ABSOLUTE_RETURNS_PER_ASSET"
)

extreme_return_audit = (
    pd.concat(
        [
            threshold_extremes,
            top_extremes,
        ],
        ignore_index=True,
    )
    .drop_duplicates(
        subset=[
            "Date",
            "Asset",
            TARGET,
            "Audit_Type",
        ]
    )
)

extreme_return_audit = (
    extreme_return_audit
    .rename(
        columns={
            "_Abs_Target_Return":
                "Absolute_Return"
        }
    )
    .sort_values(
        [
            "Asset",
            "Absolute_Return",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(drop=True)
)

print(
    "\nRows with absolute target return >= "
    f"{EXTREME_RETURN_THRESHOLD:.0%}: "
    f"{len(threshold_extremes):,}"
)

if not threshold_extremes.empty:

    print(
        "\nThese observations are NOT removed or changed. "
        "They require source verification before Section 09."
    )

    print(
        threshold_extremes[
            [
                "Date",
                "Asset",
                TARGET,
                "_Abs_Target_Return",
            ]
        ]
        .to_string(
            index=False
        )
    )


# =============================================================================
# 33. YEAR / REGIME COVERAGE
# =============================================================================

year_coverage_rows = []

for asset in EXPECTED_ASSETS:

    for year in range(
        2021,
        2026,
    ):

        subset = (
            df.loc[
                df["Asset"].eq(asset)
                & df["Year"].eq(year)
            ]
        )

        year_coverage_rows.append(
            {
                "Asset":
                    asset,

                "Year":
                    year,

                "Calendar_Rows":
                    len(subset),

                "Target_Available":
                    int(
                        subset[
                            TARGET
                        ]
                        .notna()
                        .sum()
                    ),

                "Benchmark_Eligible":
                    int(
                        subset[
                            "Eligible_Benchmark"
                        ]
                        .sum()
                    ),

                "Activity_Eligible":
                    int(
                        subset[
                            "Eligible_Activity"
                        ]
                        .sum()
                    ),

                "Sentiment_Eligible":
                    int(
                        subset[
                            "Eligible_Sentiment"
                        ]
                        .sum()
                    ),

                "Both_Eligible":
                    int(
                        subset[
                            "Eligible_Both"
                        ]
                        .sum()
                    ),
            }
        )

year_coverage = (
    pd.DataFrame(
        year_coverage_rows
    )
)


# =============================================================================
# 34. MODEL SPECIFICATION DICTIONARY
# =============================================================================

model_spec_rows = []

model_specs = {
    "M0_Benchmark": {
        "role":
            "Primary benchmark",

        "predictors":
            BENCHMARK_PREDICTORS,

        "sample":
            "Eligible_Benchmark",
    },

    "M1_Activity": {
        "role":
            "Benchmark + Reddit activity",

        "predictors":
            ACTIVITY_MODEL_PREDICTORS,

        "sample":
            "Eligible_Activity",
    },

    "M2_Sentiment": {
        "role":
            "Benchmark + Reddit sentiment",

        "predictors":
            SENTIMENT_MODEL_PREDICTORS,

        "sample":
            "Eligible_Sentiment",
    },

    "M3_Both": {
        "role":
            (
                "Benchmark + activity "
                "+ sentiment"
            ),

        "predictors":
            BOTH_MODEL_PREDICTORS,

        "sample":
            "Eligible_Both",
    },

    "R0_Benchmark_Cross": {
        "role":
            (
                "Cross-crypto robustness "
                "benchmark"
            ),

        "predictors":
            BENCHMARK_CROSS_PREDICTORS,

        "sample":
            "Eligible_Benchmark_Cross",
    },

    "R1_Activity_Cross": {
        "role":
            (
                "Cross-crypto robustness "
                "+ activity"
            ),

        "predictors":
            ACTIVITY_CROSS_PREDICTORS,

        "sample":
            "Eligible_Activity_Cross",
    },

    "R2_Sentiment_Cross": {
        "role":
            (
                "Cross-crypto robustness "
                "+ sentiment"
            ),

        "predictors":
            SENTIMENT_CROSS_PREDICTORS,

        "sample":
            "Eligible_Sentiment_Cross",
    },

    "R3_Both_Cross": {
        "role":
            (
                "Cross-crypto robustness "
                "+ activity + sentiment"
            ),

        "predictors":
            BOTH_CROSS_PREDICTORS,

        "sample":
            "Eligible_Both_Cross",
    },
}

for (
    model_name,
    spec,
) in model_specs.items():

    for (
        order,
        predictor,
    ) in enumerate(
        spec["predictors"],
        start=1,
    ):

        model_spec_rows.append(
            {
                "Model":
                    model_name,

                "Role":
                    spec["role"],

                "Dependent_Variable":
                    TARGET,

                "Predictor_Order":
                    order,

                "Predictor":
                    predictor,

                "Native_Eligibility_Flag":
                    spec["sample"],
            }
        )

model_specifications = (
    pd.DataFrame(
        model_spec_rows
    )
)


# =============================================================================
# 35. FORECAST COMPARISON DESIGN
# =============================================================================

forecast_design = pd.DataFrame(
    [
        {
            "Comparison":
                "Activity",

            "Benchmark_Model":
                "M0_Benchmark",

            "Extended_Model":
                "M1_Activity",

            "Common_Sample_Flag":
                "Activity_Comparison_Sample",

            "Primary_Hypothesis":
                (
                    "Supplementary activity "
                    "comparison"
                ),
        },

        {
            "Comparison":
                "Sentiment",

            "Benchmark_Model":
                "M0_Benchmark",

            "Extended_Model":
                "M2_Sentiment",

            "Common_Sample_Flag":
                "Sentiment_Comparison_Sample",

            "Primary_Hypothesis":
                "H3 BTC / H4 ETH",
        },

        {
            "Comparison":
                "Both",

            "Benchmark_Model":
                "M0_Benchmark",

            "Extended_Model":
                "M3_Both",

            "Common_Sample_Flag":
                "Both_Comparison_Sample",

            "Primary_Hypothesis":
                (
                    "Incremental sentiment "
                    "+ activity comparison"
                ),
        },
    ]
)


# =============================================================================
# 36. FINAL MODELLING DATASET
# =============================================================================

audit_columns = [
    "Date",
    "Asset",
    "Evaluation_Period",
    "Is_Initial_Train_Period",
    "Is_OOS_Period",
    "Year",
    "Day_Of_Week",
    "Is_Weekend",
    TARGET,
    *BENCHMARK_PREDICTORS,
    ACTIVITY_PREDICTOR,
    SENTIMENT_PREDICTOR,
    CROSS_CRYPTO_PREDICTOR,
]

source_and_reddit_audit_candidates = [
    "Lagged_Reddit_Source_Date",
    "Reddit_Observation_Status_Lag_1",
    "Reddit_Activity_Available_Lag_1",
    "Reddit_Sentiment_Available_Lag_1",
    "SP500_Source_Date",
    "VIX_Source_Date",
    "Gold_Source_Date",
    "DXY_Source_Date",
    "US10Y_Source_Date",
    "SP500_Information_Age_Days",
    "VIX_Information_Age_Days",
    "Gold_Information_Age_Days",
    "DXY_Information_Age_Days",
    "US10Y_Information_Age_Days",
]

for lag_info in (
    ALTERNATIVE_REDDIT_LAGS.values()
):

    source_and_reddit_audit_candidates.extend(
        [
            lag_info[
                "sentiment"
            ],
            lag_info[
                "activity"
            ],
            lag_info[
                "source_date"
            ],
        ]
    )

audit_columns.extend(
    [
        col
        for col
        in source_and_reddit_audit_candidates
        if col in df.columns
    ]
)

audit_columns.extend(
    sample_flags
)

audit_columns = list(
    dict.fromkeys(
        audit_columns
    )
)

final_modelling_dataset = (
    df[
        audit_columns
    ]
    .copy()
)


# =============================================================================
# 37. MODEL SAMPLE MEMBERSHIP
# =============================================================================

membership_columns = [
    "Date",
    "Asset",
    "Evaluation_Period",
    "Is_Initial_Train_Period",
    "Is_OOS_Period",
    "Is_Weekend",
    *sample_flags,
]

model_sample_membership = (
    df[
        membership_columns
    ]
    .copy()
)


# =============================================================================
# 38. FINAL HARD QC
# =============================================================================

banner(
    "12. FINAL HARD QC"
)

all_declared_predictors = list(
    dict.fromkeys(
        BENCHMARK_PREDICTORS
        + ACTIVITY_MODEL_PREDICTORS
        + SENTIMENT_MODEL_PREDICTORS
        + BOTH_MODEL_PREDICTORS
        + BOTH_CROSS_PREDICTORS
    )
)

target_leakage_by_name = (
    TARGET
    in all_declared_predictors
)

add_qc(
    qc_rows,
    "Target_Not_In_Predictor_List",
    False,
    target_leakage_by_name,
    not target_leakage_by_name,
    "Leakage",
)

traditional_leakage_failures = 0

for item in (
    TRADITIONAL_ALIGNMENT.values()
):

    source_col = (
        item["source_date"]
    )

    if source_col in df.columns:

        source = pd.to_datetime(
            df[source_col],
            errors="coerce",
        )

        traditional_leakage_failures += int(
            (
                source.notna()
                & (
                    source
                    >= df["Date"]
                )
            )
            .sum()
        )

add_qc(
    qc_rows,
    "Traditional_Source_Date_Leakage_Rows",
    0,
    traditional_leakage_failures,
    traditional_leakage_failures == 0,
    "Leakage",
)

hierarchy_activity_fail = int(
    (
        df[
            "Eligible_Activity"
        ]
        & ~df[
            "Eligible_Benchmark"
        ]
    )
    .sum()
)

hierarchy_sentiment_fail = int(
    (
        df[
            "Eligible_Sentiment"
        ]
        & ~df[
            "Eligible_Benchmark"
        ]
    )
    .sum()
)

hierarchy_both_fail = int(
    (
        df[
            "Eligible_Both"
        ]
        & ~df[
            "Eligible_Activity"
        ]
    )
    .sum()
    +
    (
        df[
            "Eligible_Both"
        ]
        & ~df[
            "Eligible_Sentiment"
        ]
    )
    .sum()
)

add_qc(
    qc_rows,
    "Activity_Sample_Subset_Of_Benchmark",
    0,
    hierarchy_activity_fail,
    hierarchy_activity_fail == 0,
    "Sample_Construction",
)

add_qc(
    qc_rows,
    "Sentiment_Sample_Subset_Of_Benchmark",
    0,
    hierarchy_sentiment_fail,
    hierarchy_sentiment_fail == 0,
    "Sample_Construction",
)

add_qc(
    qc_rows,
    "Both_Sample_Subset_Of_Activity_And_Sentiment",
    0,
    hierarchy_both_fail,
    hierarchy_both_fail == 0,
    "Sample_Construction",
)

qc = pd.DataFrame(
    qc_rows
)

all_hard_qc_pass = bool(
    qc[
        "Pass"
    ]
    .all()
)


# =============================================================================
# 39. METHODOLOGY NOTE
# =============================================================================

methodology_note = f"""
SECTION 08 — FINAL MODELLING DATASET VALIDATION
===============================================

Purpose
-------
Section 08 validates the integrated Reddit-market dataset before regression
estimation and genuine out-of-sample forecasting.

Study period
------------
{STUDY_START.date()} to {STUDY_END.date()}

Assets
------
BTC
ETH

Target calendar
---------------
The modelling structure preserves the complete cryptocurrency calendar:

    {EXPECTED_CALENDAR_DAYS:,} dates
    2 assets
    {EXPECTED_DATE_ASSET_ROWS:,} Date × Asset observations

Weekends are retained.

Dependent variable
------------------
Target_Return

Target_Return is independently reconstructed against the original market-side
return variables for BTC and ETH.

Primary benchmark predictors
----------------------------
{chr(10).join("    " + x for x in BENCHMARK_PREDICTORS)}

Reddit activity
---------------
{ACTIVITY_PREDICTOR}

Reddit sentiment
----------------
{SENTIMENT_PREDICTOR}

Cross-cryptocurrency robustness
-------------------------------
{CROSS_CRYPTO_PREDICTOR}

Primary models
--------------
M0: controls only
M1: controls + Reddit activity
M2: controls + Reddit sentiment
M3: controls + Reddit activity + Reddit sentiment

Cross-cryptocurrency lagged return is reserved for robustness.

Common-sample rule
------------------
Benchmark versus activity:
    Activity_Comparison_Sample

Benchmark versus sentiment:
    Sentiment_Comparison_Sample

Benchmark versus both:
    Both_Comparison_Sample

M0/M1/M2/M3 common explanatory sample:
    Common_Main_Model_Sample

Nested model comparisons must use identical observations.

Missing Reddit sentiment
------------------------
Missing Reddit sentiment is NOT imputed as zero.

Training / forecasting design
-----------------------------
Initial estimation:
    {INITIAL_TRAIN_START.date()} to {INITIAL_TRAIN_END.date()}

Genuine out-of-sample evaluation:
    {OOS_START.date()} to {OOS_END.date()}

Section 09 will use expanding-window one-step-ahead forecasts.

No random train/test split is used.

Information timing
------------------
Own cryptocurrency return:
    previous calendar day

Cryptocurrency volume:
    previous calendar day

Primary Reddit variables:
    previous calendar day

Traditional controls:
    latest completed transformed observation strictly before target date.

VIF methodology
---------------
Variance Inflation Factors are calculated using conventional auxiliary
regressions that include an intercept.

The constant is added to the VIF design matrix but is not itself reported.

This produces centred/conventional VIF diagnostics and avoids misleading
uncentred VIF values caused by variables with large non-zero means.

Correlation and VIF diagnostics are not automatic variable-exclusion rules.

Extreme returns
---------------
Observations with absolute Target_Return >=
{EXTREME_RETURN_THRESHOLD:.0%}
are flagged for manual source verification.

They are not deleted, winsorised or modified.

Inference and forecasting
-------------------------
Section 08 does not estimate H1-H5.

Section 09 should distinguish explanatory evidence from genuine predictive
evidence.

Explanatory regressions should use appropriate HAC/Newey-West inference.

H3/H4 forecasting comparisons should use identical comparison-specific
samples and genuinely held-out 2024-2025 one-step-ahead forecasts.

Primary forecast metrics should include:
    RMSE
    MAE
    out-of-sample R-squared

Nested forecast comparisons involving sentiment should use an appropriate
formal forecast comparison such as Clark-West.

BTC-versus-ETH sentiment coefficient differences should be tested formally
rather than inferred by comparing separate p-values.

SECTION 08 END
"""


# =============================================================================
# 40. SAVE OUTPUTS
# =============================================================================

banner(
    "13. SAVING SECTION 08 OUTPUTS"
)

output_files = {
    "final_modelling_dataset.csv":
        final_modelling_dataset,

    "stage08_model_sample_membership.csv":
        model_sample_membership,

    "stage08_model_sample_summary.csv":
        sample_summary,

    "stage08_missingness_audit.csv":
        missingness,

    "stage08_missing_dates.csv":
        missing_dates,

    "stage08_descriptive_statistics.csv":
        descriptive_statistics,

    "stage08_predictor_distributions.csv":
        predictor_distributions,

    "stage08_correlations_long.csv":
        correlation_long,

    "stage08_high_correlations.csv":
        high_correlations,

    "stage08_vif_diagnostics.csv":
        vif_diagnostics,

    "stage08_extreme_return_audit.csv":
        extreme_return_audit,

    "stage08_calendar_audit.csv":
        calendar_audit,

    "stage08_reconstruction_issues.csv":
        reconstruction_issues,

    "stage08_reddit_reconstruction_issues.csv":
        reddit_reconstruction_issues,

    "stage08_traditional_reconstruction_issues.csv":
        traditional_reconstruction_issues_df,

    "stage08_information_timing_validation.csv":
        timing_validation,

    "stage08_common_sample_validation.csv":
        common_sample_validation,

    "stage08_weekend_oos_summary.csv":
        weekend_summary,

    "stage08_year_coverage.csv":
        year_coverage,

    "stage08_model_specifications.csv":
        model_specifications,

    "stage08_forecast_comparison_design.csv":
        forecast_design,

    "stage08_qc.csv":
        qc,

    "stage08_infinite_value_audit.csv":
        infinite_audit,
}

for (
    filename,
    data,
) in output_files.items():

    output_path = (
        OUTPUT_DIR
        / filename
    )

    data.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved: "
        f"{filename:<48} "
        f"rows={len(data):,}"
    )

methodology_path = (
    OUTPUT_DIR
    / "stage08_methodology_note.txt"
)

methodology_path.write_text(
    methodology_note.strip()
    + "\n",
    encoding="utf-8",
)

print(
    f"Saved: "
    f"{'stage08_methodology_note.txt':<48}"
)


# =============================================================================
# 41. OUTPUT RELOAD VALIDATION
# =============================================================================

banner(
    "14. OUTPUT RELOAD VALIDATION"
)

final_output_path = (
    OUTPUT_DIR
    / "final_modelling_dataset.csv"
)

reloaded = pd.read_csv(
    final_output_path,
    parse_dates=[
        "Date"
    ],
)

reload_row_match = (
    len(reloaded)
    == len(
        final_modelling_dataset
    )
)

reload_duplicate_count = int(
    reloaded
    .duplicated(
        [
            "Date",
            "Asset",
        ]
    )
    .sum()
)

print(
    "Final modelling dataset rows: "
    f"{len(reloaded):,}"
)

print(
    "Duplicate Date × Asset rows: "
    f"{reload_duplicate_count:,}"
)

print(
    "Reload row-count match: "
    f"{reload_row_match}"
)

reload_pass = bool(
    reload_row_match
    and reload_duplicate_count == 0
)

reload_qc_row = pd.DataFrame(
    [
        {
            "Check":
                "Final_Output_Reload",

            "Expected":
                True,

            "Actual":
                reload_pass,

            "Pass":
                reload_pass,

            "Category":
                "Persistence",

            "Notes":
                (
                    f"Rows={len(reloaded)}; "
                    "duplicates="
                    f"{reload_duplicate_count}"
                ),
        }
    ]
)

qc = pd.concat(
    [
        qc,
        reload_qc_row,
    ],
    ignore_index=True,
)

qc.to_csv(
    OUTPUT_DIR
    / "stage08_qc.csv",
    index=False,
)

all_hard_qc_pass = bool(
    qc[
        "Pass"
    ]
    .all()
)


# =============================================================================
# 42. CONSOLE SUMMARY
# =============================================================================

banner(
    "15. SECTION 08 SUMMARY"
)

print(
    "\nSTRUCTURE"
)

print(
    "---------"
)

print(
    "Calendar dates:              "
    f"{df['Date'].nunique():,}"
)

print(
    "Date × Asset rows:           "
    f"{len(df):,}"
)

print(
    "BTC rows:                    "
    f"{df['Asset'].eq('BTC').sum():,}"
)

print(
    "ETH rows:                    "
    f"{df['Asset'].eq('ETH').sum():,}"
)

print(
    "Duplicate Date × Asset rows: "
    f"{duplicate_rows:,}"
)


# -----------------------------------------------------------------------------
# Primary samples
# -----------------------------------------------------------------------------

print(
    "\nPRIMARY MODEL SAMPLE COUNTS"
)

print(
    "---------------------------"
)

display_samples = (
    sample_summary.loc[
        sample_summary[
            "Sample"
        ].isin(
            [
                "Eligible_Benchmark",
                "Eligible_Activity",
                "Eligible_Sentiment",
                "Eligible_Both",
                "Sentiment_Comparison_Sample",
                "Common_Main_Model_Sample",
            ]
        )
        &
        sample_summary[
            "Period"
        ].isin(
            [
                "FULL_SAMPLE",
                "INITIAL_TRAIN",
                "OUT_OF_SAMPLE",
            ]
        )
    ]
)

print(
    display_samples[
        [
            "Asset",
            "Sample",
            "Period",
            "N",
            "Weekend_Observations",
        ]
    ]
    .to_string(
        index=False
    )
)


# -----------------------------------------------------------------------------
# Missingness
# -----------------------------------------------------------------------------

print(
    "\nMISSINGNESS"
)

print(
    "-----------"
)

print(
    missingness[
        [
            "Asset",
            "Variable",
            "Available",
            "Missing",
            "Missing_Percent",
        ]
    ]
    .to_string(
        index=False,
        float_format=(
            lambda x:
                f"{x:.3f}"
        ),
    )
)


# -----------------------------------------------------------------------------
# Corrected VIF diagnostics
# -----------------------------------------------------------------------------

print(
    "\nVIF DIAGNOSTICS "
    "(CONVENTIONAL VIF WITH INTERCEPT)"
)

print(
    "-------------------------------------------"
)

print(
    vif_diagnostics[
        [
            "Asset",
            "Sample",
            "Predictor",
            "N",
            "VIF",
            "VIF_Above_5",
            "VIF_Above_10",
        ]
    ]
    .to_string(
        index=False,
        float_format=(
            lambda x:
                f"{x:.3f}"
        ),
    )
)


# -----------------------------------------------------------------------------
# High correlations
# -----------------------------------------------------------------------------

print(
    "\nHIGH ABSOLUTE PREDICTOR "
    "CORRELATIONS >= 0.80"
)

print(
    "-------------------------------------------"
)

if high_correlations.empty:

    print(
        "None."
    )

else:

    print(
        high_correlations
        .to_string(
            index=False,
            float_format=(
                lambda x:
                    f"{x:.4f}"
            ),
        )
    )


# -----------------------------------------------------------------------------
# Extreme returns
# -----------------------------------------------------------------------------

print(
    "\nEXTREME RETURN AUDIT"
)

print(
    "--------------------"
)

print(
    "Absolute-return threshold: "
    f"{EXTREME_RETURN_THRESHOLD:.0%}"
)

print(
    "Threshold observations: "
    f"{len(threshold_extremes):,}"
)


# -----------------------------------------------------------------------------
# Hard QC
# -----------------------------------------------------------------------------

print(
    "\nHARD QC"
)

print(
    "-------"
)

print(
    qc[
        [
            "Check",
            "Expected",
            "Actual",
            "Pass",
        ]
    ]
    .to_string(
        index=False
    )
)

failed_qc = (
    qc.loc[
        ~qc["Pass"]
    ]
)


# =============================================================================
# 43. FINAL STATUS
# =============================================================================

print(
    "\nFINAL STATUS"
)

print(
    "------------"
)

if failed_qc.empty:

    print(
        "SECTION 08 PASS"
    )

    print(
        "All hard QC checks passed."
    )

    print(
        "The final modelling dataset is structurally "
        "ready for Section 09, subject to manual review "
        "of diagnostic/extreme-return audit outputs."
    )

    print(
        "\nIMPORTANT: A QC PASS does not automatically mean "
        "that flagged extreme returns are economically valid. "
        "Verify suspicious source observations before modelling."
    )

else:

    print(
        "SECTION 08 FAIL"
    )

    print(
        f"{len(failed_qc):,} "
        "hard QC check(s) failed:"
    )

    print(
        failed_qc[
            [
                "Check",
                "Expected",
                "Actual",
                "Category",
                "Notes",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nDo NOT proceed to Section 09 until "
        "the failed hard QC checks have been investigated."
    )


# =============================================================================
# 44. CLEAN TEMPORARY COLUMN
# =============================================================================

if (
    "_Abs_Target_Return"
    in df.columns
):

    df.drop(
        columns=[
            "_Abs_Target_Return"
        ],
        inplace=True,
    )


# =============================================================================
# 45. EXIT STATUS
# =============================================================================

if all_hard_qc_pass:
    sys.exit(0)

else:
    sys.exit(1)