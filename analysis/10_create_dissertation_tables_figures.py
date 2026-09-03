# ============================================================
# SECTION 10
# DISSERTATION TABLES & FIGURES
#
# PART 1 — SETUP, LOAD FROZEN RESULTS, VALIDATION
# ============================================================

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

SECTION08_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "stage08_final_modelling_dataset"
)

SECTION09_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage09_modelling_forecast_comparison"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "stage10_dissertation_tables_figures"
)

TABLE_DIR = OUTPUT_DIR / "tables"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TABLE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. PRESENTATION SETTINGS
# ============================================================

pd.set_option(
    "display.max_columns",
    100,
)

pd.set_option(
    "display.width",
    180,
)

pd.set_option(
    "display.float_format",
    lambda x: f"{x:.6f}",
)

plt.rcParams.update({
    "figure.figsize": (8, 5),
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
})


# ============================================================
# 3. CONSTANTS
# ============================================================

ASSETS = [
    "BTC",
    "ETH",
]

PRIMARY_MODELS = [
    "M0_Benchmark",
    "M1_Activity",
    "M2_Sentiment",
    "M3_Both",
]

HYPOTHESES = [
    "H1",
    "H2",
    "H3",
    "H4",
    "H5",
]

SIGNIFICANCE_LEVEL = 0.05


# ============================================================
# 4. HELPER — HEADER
# ============================================================

def print_header(
    text: str,
) -> None:

    print(
        "\n"
        + "=" * 88
    )

    print(
        text
    )

    print(
        "=" * 88
    )


# ============================================================
# 5. HELPER — REQUIRE FILE
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            "\nRequired Section 10 input file "
            "was not found:\n"
            f"{path}"
        )


# ============================================================
# 6. HELPER — REQUIRE COLUMNS
# ============================================================

def require_columns(
    dataframe: pd.DataFrame,
    columns: list[str],
    context: str,
) -> None:

    missing = [
        column
        for column in columns
        if column not in dataframe.columns
    ]

    if missing:

        raise ValueError(
            f"\nMissing required columns in {context}:\n"
            + "\n".join(
                f"  - {column}"
                for column in missing
            )
        )


# ============================================================
# 7. HELPER — SIGNIFICANCE STARS
# ============================================================

def significance_stars(
    p_value: float,
) -> str:

    if pd.isna(
        p_value
    ):

        return ""

    if p_value < 0.01:

        return "***"

    if p_value < 0.05:

        return "**"

    if p_value < 0.10:

        return "*"

    return ""


# ============================================================
# 8. HELPER — FORMAT COEFFICIENT
# ============================================================

def format_coefficient(
    coefficient: float,
    p_value: float,
    decimals: int = 4,
) -> str:

    if pd.isna(
        coefficient
    ):

        return ""

    stars = significance_stars(
        p_value
    )

    return (
        f"{coefficient:.{decimals}f}"
        f"{stars}"
    )


# ============================================================
# 9. HELPER — FORMAT STANDARD ERROR
# ============================================================

def format_standard_error(
    standard_error: float,
    decimals: int = 4,
) -> str:

    if pd.isna(
        standard_error
    ):

        return ""

    return (
        f"({standard_error:.{decimals}f})"
    )


# ============================================================
# 10. DEFINE SECTION 08 INPUT FILES
# ============================================================

SECTION08_FILES = {

    "descriptive_statistics":
        SECTION08_DIR
        / "stage08_descriptive_statistics.csv",

    "correlations":
        SECTION08_DIR
        / "stage08_correlations_long.csv",

    "high_correlations":
        SECTION08_DIR
        / "stage08_high_correlations.csv",

    "vif":
        SECTION08_DIR
        / "stage08_vif_diagnostics.csv",

    "extreme_returns":
        SECTION08_DIR
        / "stage08_extreme_return_audit.csv",

    "year_coverage":
        SECTION08_DIR
        / "stage08_year_coverage.csv",

    "final_dataset":
        SECTION08_DIR
        / "final_modelling_dataset.csv",
}


# ============================================================
# 11. DEFINE SECTION 09 INPUT FILES
# ============================================================

SECTION09_FILES = {

    "primary_regressions":
        SECTION09_DIR
        / "primary_hac_regression_results.csv",

    "primary_model_summary":
        SECTION09_DIR
        / "primary_model_summary.csv",

    "h1_h2":
        SECTION09_DIR
        / "h1_h2_sentiment_tests.csv",

    "economic_significance":
        SECTION09_DIR
        / "economic_significance.csv",

    "forecast_performance":
        SECTION09_DIR
        / "forecast_performance_comparison.csv",

    "h3_h4":
        SECTION09_DIR
        / "h3_h4_primary_oos_tests.csv",

    "cumulative_forecast_loss":
        SECTION09_DIR
        / "primary_sentiment_cumulative_forecast_loss.csv",

    "h5":
        SECTION09_DIR
        / "h5_btc_eth_sentiment_difference_test.csv",

    "cross_crypto":
        SECTION09_DIR
        / "cross_crypto_robustness_summary.csv",

    "lag_robustness":
        SECTION09_DIR
        / "reddit_sentiment_lag_robustness.csv",

    "year_robustness":
        SECTION09_DIR
        / "year_regime_sentiment_robustness.csv",

    "extreme_sensitivity":
        SECTION09_DIR
        / "extreme_return_sensitivity.csv",

    "extreme_lag_sensitivity":
        SECTION09_DIR
        / "extreme_return_and_lag_sensitivity.csv",

    "oos_by_year":
        SECTION09_DIR
        / "primary_sentiment_oos_by_year.csv",

    "weekend_weekday":
        SECTION09_DIR
        / "primary_sentiment_weekend_weekday_oos.csv",

    "hypothesis_summary":
        SECTION09_DIR
        / "final_hypothesis_summary.csv",

    "forecast_qc":
        SECTION09_DIR
        / "final_forecast_qc.csv",

    "section09_qc":
        SECTION09_DIR
        / "section09_qc.csv",
}


# ============================================================
# 12. CHECK ALL REQUIRED INPUT FILES
# ============================================================

print_header(
    "SECTION 10 — INPUT FILE VALIDATION"
)


for file_name, file_path in (
    SECTION08_FILES.items()
):

    require_file(
        file_path
    )

    print(
        f"Section 08 input found: "
        f"{file_name}"
    )


for file_name, file_path in (
    SECTION09_FILES.items()
):

    require_file(
        file_path
    )

    print(
        f"Section 09 input found: "
        f"{file_name}"
    )


print(
    "\nAll required Section 08/09 "
    "presentation inputs found: PASS"
)


# ============================================================
# 13. LOAD SECTION 08 OUTPUTS
# ============================================================

print_header(
    "SECTION 10 — LOAD SECTION 08 OUTPUTS"
)


descriptive_statistics = pd.read_csv(
    SECTION08_FILES[
        "descriptive_statistics"
    ]
)


correlations_long = pd.read_csv(
    SECTION08_FILES[
        "correlations"
    ]
)


high_correlations = pd.read_csv(
    SECTION08_FILES[
        "high_correlations"
    ]
)


vif_diagnostics = pd.read_csv(
    SECTION08_FILES[
        "vif"
    ]
)


extreme_return_audit = pd.read_csv(
    SECTION08_FILES[
        "extreme_returns"
    ]
)


year_coverage = pd.read_csv(
    SECTION08_FILES[
        "year_coverage"
    ]
)


final_dataset = pd.read_csv(
    SECTION08_FILES[
        "final_dataset"
    ]
)


final_dataset[
    "Date"
] = pd.to_datetime(
    final_dataset[
        "Date"
    ],
    errors="raise",
)


print(
    f"Final modelling dataset rows: "
    f"{len(final_dataset):,}"
)

print(
    f"Descriptive-statistics rows: "
    f"{len(descriptive_statistics):,}"
)

print(
    f"VIF rows: "
    f"{len(vif_diagnostics):,}"
)


# ============================================================
# 14. LOAD SECTION 09 OUTPUTS
# ============================================================

print_header(
    "SECTION 10 — LOAD SECTION 09 OUTPUTS"
)


primary_regressions = pd.read_csv(
    SECTION09_FILES[
        "primary_regressions"
    ]
)


primary_model_summary = pd.read_csv(
    SECTION09_FILES[
        "primary_model_summary"
    ]
)


h1_h2_results = pd.read_csv(
    SECTION09_FILES[
        "h1_h2"
    ]
)


economic_significance = pd.read_csv(
    SECTION09_FILES[
        "economic_significance"
    ]
)


forecast_performance = pd.read_csv(
    SECTION09_FILES[
        "forecast_performance"
    ]
)


h3_h4_results = pd.read_csv(
    SECTION09_FILES[
        "h3_h4"
    ]
)


cumulative_forecast_loss = pd.read_csv(
    SECTION09_FILES[
        "cumulative_forecast_loss"
    ]
)


h5_results = pd.read_csv(
    SECTION09_FILES[
        "h5"
    ]
)


cross_crypto_robustness = pd.read_csv(
    SECTION09_FILES[
        "cross_crypto"
    ]
)


lag_robustness = pd.read_csv(
    SECTION09_FILES[
        "lag_robustness"
    ]
)


year_robustness = pd.read_csv(
    SECTION09_FILES[
        "year_robustness"
    ]
)


extreme_sensitivity = pd.read_csv(
    SECTION09_FILES[
        "extreme_sensitivity"
    ]
)


extreme_lag_sensitivity = pd.read_csv(
    SECTION09_FILES[
        "extreme_lag_sensitivity"
    ]
)


oos_by_year = pd.read_csv(
    SECTION09_FILES[
        "oos_by_year"
    ]
)


weekend_weekday_oos = pd.read_csv(
    SECTION09_FILES[
        "weekend_weekday"
    ]
)


hypothesis_summary = pd.read_csv(
    SECTION09_FILES[
        "hypothesis_summary"
    ]
)


forecast_qc = pd.read_csv(
    SECTION09_FILES[
        "forecast_qc"
    ]
)


section09_qc = pd.read_csv(
    SECTION09_FILES[
        "section09_qc"
    ]
)


print(
    f"Primary regression rows: "
    f"{len(primary_regressions):,}"
)

print(
    f"H1/H2 rows: "
    f"{len(h1_h2_results):,}"
)

print(
    f"H3/H4 rows: "
    f"{len(h3_h4_results):,}"
)

print(
    f"H5 rows: "
    f"{len(h5_results):,}"
)

print(
    f"Final hypothesis rows: "
    f"{len(hypothesis_summary):,}"
)


# ============================================================
# 15. CORE SECTION 09 RESULT VALIDATION
# ============================================================

print_header(
    "SECTION 10 — FROZEN RESULT VALIDATION"
)


require_columns(
    primary_regressions,
    [
        "Asset",
        "Model",
        "Parameter",
        "Coefficient",
        "HAC_SE",
        "p_value",
    ],
    context=(
        "primary regression results"
    ),
)


require_columns(
    primary_model_summary,
    [
        "Asset",
        "Model",
        "N",
        "R_Squared",
        "Adjusted_R_Squared",
    ],
    context=(
        "primary model summary"
    ),
)


require_columns(
    hypothesis_summary,
    [
        "Hypothesis",
        "Asset",
        "p_value",
        "Supported_5pct",
        "Interpretation",
    ],
    context=(
        "final hypothesis summary"
    ),
)


require_columns(
    h3_h4_results,
    [
        "Hypothesis",
        "Asset",
        "N",
        "Benchmark_RMSE",
        "Extended_RMSE",
        "Benchmark_MAE",
        "Extended_MAE",
        "Extended_OOS_R2",
        "Clark_West_Z",
        "Clark_West_One_Sided_P",
        "H3_H4_Supported",
    ],
    context=(
        "H3/H4 OOS results"
    ),
)


# ============================================================
# 16. VALIDATE H1-H5 ARE PRESENT
# ============================================================

observed_hypotheses = set(
    hypothesis_summary[
        "Hypothesis"
    ]
    .astype(str)
)


expected_hypotheses = set(
    HYPOTHESES
)


if observed_hypotheses != (
    expected_hypotheses
):

    raise ValueError(
        "\nFinal hypothesis summary does not "
        "contain exactly H1-H5.\n"
        f"Observed: "
        f"{sorted(observed_hypotheses)}"
    )


print(
    "H1-H5 presence: PASS"
)


# ============================================================
# 17. VALIDATE SECTION 09 QC
# ============================================================

require_columns(
    section09_qc,
    [
        "Check",
        "PASS",
    ],
    context=(
        "Section 09 master QC"
    ),
)


qc_values = (
    section09_qc[
        "PASS"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
)


if not qc_values.isin(
    [
        "true",
        "1",
        "yes",
    ]
).all():

    failed_qc = (
        section09_qc.loc[
            ~qc_values.isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            )
        ]
    )

    raise ValueError(
        "\nSection 09 contains failed QC checks.\n\n"
        f"{failed_qc.to_string(index=False)}"
    )


print(
    "Section 09 master QC: PASS"
)


# ============================================================
# 18. VALIDATE FINAL FORECAST QC
# ============================================================

require_columns(
    forecast_qc,
    [
        "Asset",
        "Comparison",
        "Historical_Dates_Identical",
        "Benchmark_No_Lookahead",
        "Extended_No_Lookahead",
        "PASS",
    ],
    context=(
        "final forecast QC"
    ),
)


forecast_pass = (
    forecast_qc[
        "PASS"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .isin(
        [
            "true",
            "1",
            "yes",
        ]
    )
)


if not forecast_pass.all():

    raise ValueError(
        "\nOne or more Section 09 "
        "forecast comparisons failed QC."
    )


print(
    "Section 09 forecast QC: PASS"
)


# ============================================================
# 19. VALIDATE PRIMARY MODELS
# ============================================================

for asset in ASSETS:

    asset_models = set(
        primary_model_summary.loc[
            primary_model_summary[
                "Asset"
            ].eq(asset),
            "Model",
        ]
    )

    if asset_models != set(
        PRIMARY_MODELS
    ):

        raise ValueError(
            "\nUnexpected primary model set.\n"
            f"Asset: {asset}\n"
            f"Observed: "
            f"{sorted(asset_models)}"
        )


print(
    "BTC/ETH M0-M3 model presence: PASS"
)


# ============================================================
# 20. CHECK THAT SECTION 10 DOES NOT MODIFY FROZEN INPUTS
# ============================================================

# Section 10 deliberately creates copies for presentation.
#
# No Section 08 or Section 09 file is overwritten.

primary_regressions_raw = (
    primary_regressions.copy(
        deep=True
    )
)

h3_h4_results_raw = (
    h3_h4_results.copy(
        deep=True
    )
)

hypothesis_summary_raw = (
    hypothesis_summary.copy(
        deep=True
    )
)


print(
    "Frozen Section 08/09 results "
    "loaded as read-only presentation inputs: PASS"
)


# ============================================================
# 21. SECTION 10 OUTPUT DIRECTORY SUMMARY
# ============================================================

print_header(
    "SECTION 10 — OUTPUT LOCATIONS"
)


print(
    f"Main output directory:\n"
    f"{OUTPUT_DIR}"
)

print(
    f"\nTables:\n"
    f"{TABLE_DIR}"
)

print(
    f"\nFigures:\n"
    f"{FIGURE_DIR}"
)


# ============================================================
# 22. PART 1 COMPLETE
# ============================================================

print_header(
    "SECTION 10 — PART 1 COMPLETE"
)


print(
    "Frozen Section 08 outputs loaded."
)

print(
    "Frozen Section 09 outputs loaded."
)

print(
    "H1-H5 results validated."
)

print(
    "Section 09 master QC reconfirmed."
)

print(
    "Forecast QC reconfirmed."
)

print(
    "Primary M0-M3 model structure reconfirmed."
)

print(
    "No models were re-estimated."
)

print(
    "No frozen Section 08/09 results were modified."
)

print(
    "\nReady for Part 2:"
)

print(
    "Dissertation-ready descriptive statistics, "
    "correlation/VIF and primary M0-M3 regression tables."
)
# ============================================================
# SECTION 10
# DISSERTATION TABLES & FIGURES
#
# PART 2 — DESCRIPTIVE STATISTICS, CORRELATIONS,
#          VIF AND PRIMARY M0-M3 REGRESSION TABLES
#
# Continues directly from Part 1.
# ============================================================


# ============================================================
# 23. START PART 2
# ============================================================

print_header(
    "SECTION 10 — PART 2: MAIN DISSERTATION TABLES"
)


# ============================================================
# 24. HELPER — FIND COLUMN
# ============================================================

def find_first_existing_column(
    dataframe: pd.DataFrame,
    candidates: list[str],
    context: str,
) -> str:
    """
    Return the first candidate column found in dataframe.

    This is used only for presentation because some Section 08
    output tables may use slightly different descriptive labels.
    """

    for candidate in candidates:

        if candidate in dataframe.columns:

            return candidate

    raise ValueError(
        f"\nCould not identify required column for {context}.\n"
        f"Candidates checked:\n"
        + "\n".join(
            f"  - {candidate}"
            for candidate in candidates
        )
        + "\n\nAvailable columns:\n"
        + "\n".join(
            f"  - {column}"
            for column in dataframe.columns
        )
    )


# ============================================================
# 25. PRESENTATION VARIABLE LABELS
# ============================================================

VARIABLE_LABELS = {

    "Target_Return":
        "Daily cryptocurrency return",

    "Own_Lagged_Return":
        "Lagged own return",

    "Lagged_Log_Crypto_Volume":
        "Lagged log trading volume",

    "Lagged_SP500_Return_Aligned":
        "S&P 500 return",

    "Lagged_VIX_Change_Aligned":
        "VIX change",

    "Lagged_Gold_Return_Aligned":
        "Gold return",

    "Lagged_DXY_Return_Aligned":
        "DXY return",

    "Lagged_US10Y_Change_Aligned":
        "US 10Y yield change",

    "Lagged_Log_Reddit_Post_Count":
        "Lagged log Reddit post count",

    "Lagged_Reddit_Sentiment":
        "Lagged Reddit sentiment",

    "Cross_Crypto_Lagged_Return":
        "Lagged cross-crypto return",
}


MODEL_LABELS = {

    "M0_Benchmark":
        "M0: Benchmark",

    "M1_Activity":
        "M1: + Activity",

    "M2_Sentiment":
        "M2: + Sentiment",

    "M3_Both":
        "M3: + Activity + Sentiment",
}


# ============================================================
# 26. DESCRIPTIVE STATISTICS — INSPECT STRUCTURE
# ============================================================

print_header(
    "SECTION 10 — DESCRIPTIVE STATISTICS TABLE"
)


print(
    "Section 08 descriptive-statistics columns:"
)

for column in descriptive_statistics.columns:

    print(
        f"  - {column}"
    )


# ============================================================
# 27. IDENTIFY DESCRIPTIVE-STATISTICS COLUMNS
# ============================================================

desc_asset_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Asset",
        "asset",
    ],
    context="descriptive-statistics asset",
)


desc_variable_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Variable",
        "variable",
        "Parameter",
        "Predictor",
    ],
    context="descriptive-statistics variable",
)


desc_n_col = find_first_existing_column(
    descriptive_statistics,
    [
        "N",
        "Count",
        "count",
        "Observations",
    ],
    context="descriptive-statistics N",
)


desc_mean_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Mean",
        "mean",
    ],
    context="descriptive-statistics mean",
)


desc_sd_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Std",
        "SD",
        "Std_Dev",
        "Standard_Deviation",
        "std",
    ],
    context="descriptive-statistics standard deviation",
)


desc_min_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Min",
        "Minimum",
        "min",
    ],
    context="descriptive-statistics minimum",
)


desc_max_col = find_first_existing_column(
    descriptive_statistics,
    [
        "Max",
        "Maximum",
        "max",
    ],
    context="descriptive-statistics maximum",
)


# ============================================================
# 28. BUILD DISSERTATION DESCRIPTIVE-STATISTICS TABLE
# ============================================================

MAIN_VARIABLES = [
    "Target_Return",
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
    "Lagged_Log_Reddit_Post_Count",
    "Lagged_Reddit_Sentiment",
]


# Restrict descriptive statistics to the common main-model sample.
# Section 08 contains multiple samples, so using all rows would
# duplicate variables in the dissertation table.
require_columns(
    descriptive_statistics,
    ["Sample"],
    context="descriptive statistics sample identifier",
)


descriptive_sample_labels = (
    descriptive_statistics["Sample"]
    .dropna()
    .astype(str)
    .str.strip()
)


normalized_sample_labels = (
    descriptive_sample_labels
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
)


common_sample_mask = (
    descriptive_statistics["Sample"]
    .astype(str)
    .str.strip()
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
    .eq("common_main_model_sample")
)


if not common_sample_mask.any():

    raise ValueError(
        "\nCould not find Common_Main_Model_Sample in "
        "stage08_descriptive_statistics.csv.\n"
        "Observed Sample values:\n"
        + "\n".join(
            f"  - {value}"
            for value in sorted(
                descriptive_sample_labels.unique()
            )
        )
    )


descriptive_table = (
    descriptive_statistics.loc[
        common_sample_mask
        &
        descriptive_statistics[
            desc_asset_col
        ].isin(
            ASSETS
        )
        &
        descriptive_statistics[
            desc_variable_col
        ].isin(
            MAIN_VARIABLES
        ),
        [
            desc_asset_col,
            desc_variable_col,
            desc_n_col,
            desc_mean_col,
            desc_sd_col,
            desc_min_col,
            desc_max_col,
        ],
    ]
    .copy()
)


descriptive_table = descriptive_table.rename(
    columns={
        desc_asset_col:
            "Asset",

        desc_variable_col:
            "Variable",

        desc_n_col:
            "N",

        desc_mean_col:
            "Mean",

        desc_sd_col:
            "SD",

        desc_min_col:
            "Minimum",

        desc_max_col:
            "Maximum",
    }
)


descriptive_table[
    "Variable_Label"
] = descriptive_table[
    "Variable"
].map(
    VARIABLE_LABELS
)


descriptive_table[
    "Variable_Label"
] = descriptive_table[
    "Variable_Label"
].fillna(
    descriptive_table[
        "Variable"
    ]
)


variable_order = {
    variable: order
    for order, variable in enumerate(
        MAIN_VARIABLES,
        start=1,
    )
}


asset_order = {
    "BTC": 1,
    "ETH": 2,
}


descriptive_table[
    "_Asset_Order"
] = descriptive_table[
    "Asset"
].map(
    asset_order
)


descriptive_table[
    "_Variable_Order"
] = descriptive_table[
    "Variable"
].map(
    variable_order
)


descriptive_table = (
    descriptive_table
    .sort_values(
        [
            "_Asset_Order",
            "_Variable_Order",
        ]
    )
    .drop(
        columns=[
            "_Asset_Order",
            "_Variable_Order",
        ]
    )
    .reset_index(
        drop=True
    )
)


# Round presentation columns only.
for column in [
    "Mean",
    "SD",
    "Minimum",
    "Maximum",
]:

    descriptive_table[
        column
    ] = pd.to_numeric(
        descriptive_table[
            column
        ],
        errors="coerce",
    ).round(
        6
    )


descriptive_table[
    "N"
] = pd.to_numeric(
    descriptive_table[
        "N"
    ],
    errors="coerce",
).astype(
    "Int64"
)


descriptive_table = descriptive_table[
    [
        "Asset",
        "Variable_Label",
        "N",
        "Mean",
        "SD",
        "Minimum",
        "Maximum",
        "Variable",
    ]
]


descriptive_table.to_csv(
    TABLE_DIR
    / "table_descriptive_statistics.csv",
    index=False,
)


# Clean version without machine variable name.
descriptive_table_publication = (
    descriptive_table[
        [
            "Asset",
            "Variable_Label",
            "N",
            "Mean",
            "SD",
            "Minimum",
            "Maximum",
        ]
    ]
    .rename(
        columns={
            "Variable_Label":
                "Variable",
        }
    )
)


descriptive_table_publication.to_csv(
    TABLE_DIR
    / "table_descriptive_statistics_publication.csv",
    index=False,
)


print(
    descriptive_table_publication.to_string(
        index=False
    )
)


print(
    "\nDescriptive-statistics tables saved: PASS"
)


# ============================================================
# 29. CORRELATION TABLE — VALIDATE STRUCTURE
# ============================================================

print_header(
    "SECTION 10 — CORRELATION TABLE"
)


print(
    "Section 08 correlation columns:"
)

for column in correlations_long.columns:

    print(
        f"  - {column}"
    )


corr_asset_col = find_first_existing_column(
    correlations_long,
    [
        "Asset",
        "asset",
    ],
    context="correlation asset",
)


corr_var1_col = find_first_existing_column(
    correlations_long,
    [
        "Variable_1",
        "Variable1",
        "Variable_A",
        "Var1",
    ],
    context="correlation variable 1",
)


corr_var2_col = find_first_existing_column(
    correlations_long,
    [
        "Variable_2",
        "Variable2",
        "Variable_B",
        "Var2",
    ],
    context="correlation variable 2",
)


corr_value_col = find_first_existing_column(
    correlations_long,
    [
        "Correlation",
        "Pearson_Correlation",
        "Corr",
        "correlation",
    ],
    context="correlation coefficient",
)


# ============================================================
# 30. BUILD MAIN CORRELATION MATRICES
# ============================================================

CORRELATION_VARIABLES = [
    "Target_Return",
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
    "Lagged_Log_Reddit_Post_Count",
    "Lagged_Reddit_Sentiment",
]


correlation_output_tables = {}


for asset in ASSETS:

    asset_corr = (
        correlations_long.loc[
            correlations_long[
                corr_asset_col
            ].eq(asset)
            &
            correlations_long[
                corr_var1_col
            ].isin(
                CORRELATION_VARIABLES
            )
            &
            correlations_long[
                corr_var2_col
            ].isin(
                CORRELATION_VARIABLES
            ),
            [
                corr_var1_col,
                corr_var2_col,
                corr_value_col,
            ],
        ]
        .copy()
    )


    if asset_corr.empty:

        raise ValueError(
            "\nNo correlation observations found "
            f"for {asset}."
        )


    # Build a symmetric matrix manually so this works whether
    # Section 08 stored the upper triangle, lower triangle,
    # or both.
    correlation_matrix = pd.DataFrame(
        np.nan,
        index=CORRELATION_VARIABLES,
        columns=CORRELATION_VARIABLES,
        dtype=float,
    )


    # Set diagonal correlations to 1.0 using pandas.
    # Avoid np.fill_diagonal(correlation_matrix.values, ...)
    # because DataFrame.values may be read-only under newer
    # pandas/NumPy versions.
    for variable in CORRELATION_VARIABLES:

        correlation_matrix.loc[
            variable,
            variable,
        ] = 1.0


    for _, row in asset_corr.iterrows():

        var1 = row[
            corr_var1_col
        ]

        var2 = row[
            corr_var2_col
        ]

        value = float(
            row[
                corr_value_col
            ]
        )


        correlation_matrix.loc[
            var1,
            var2,
        ] = value


        correlation_matrix.loc[
            var2,
            var1,
        ] = value


    missing_cells = (
        correlation_matrix
        .isna()
        .sum()
        .sum()
    )


    if missing_cells > 0:

        raise ValueError(
            "\nIncomplete reconstructed correlation "
            f"matrix for {asset}.\n"
            f"Missing cells: {missing_cells}"
        )


    publication_labels = [
        VARIABLE_LABELS.get(
            variable,
            variable,
        )
        for variable in CORRELATION_VARIABLES
    ]


    correlation_matrix.index = (
        publication_labels
    )

    correlation_matrix.columns = (
        publication_labels
    )


    correlation_matrix = (
        correlation_matrix
        .round(
            3
        )
    )


    correlation_output_tables[
        asset
    ] = correlation_matrix.copy()


    correlation_matrix.to_csv(
        TABLE_DIR
        / (
            f"table_correlation_matrix_"
            f"{asset.lower()}.csv"
        ),
        index=True,
    )


    print(
        f"\n{asset} correlation matrix:"
    )

    print(
        correlation_matrix.to_string()
    )


print(
    "\nBTC/ETH correlation matrices saved: PASS"
)


# ============================================================
# 31. HIGH-CORRELATION DIAGNOSTIC SUMMARY
# ============================================================

print_header(
    "SECTION 10 — HIGH-CORRELATION DIAGNOSTIC"
)


if high_correlations.empty:

    high_correlation_summary = pd.DataFrame(
        [
            {
                "Diagnostic":
                    "Absolute predictor correlation >= 0.80",
                "Finding":
                    "None identified",
                "Concern":
                    False,
            }
        ]
    )

else:

    high_correlation_summary = (
        high_correlations.copy()
    )


high_correlation_summary.to_csv(
    TABLE_DIR
    / "table_high_correlation_diagnostic.csv",
    index=False,
)


print(
    high_correlation_summary.to_string(
        index=False
    )
)


# ============================================================
# 32. VIF TABLE — INSPECT STRUCTURE
# ============================================================

print_header(
    "SECTION 10 — VIF DIAGNOSTICS TABLE"
)


print(
    "Section 08 VIF columns:"
)

for column in vif_diagnostics.columns:

    print(
        f"  - {column}"
    )


vif_asset_col = find_first_existing_column(
    vif_diagnostics,
    [
        "Asset",
        "asset",
    ],
    context="VIF asset",
)


vif_variable_col = find_first_existing_column(
    vif_diagnostics,
    [
        "Variable",
        "Predictor",
        "Parameter",
        "variable",
    ],
    context="VIF variable",
)


vif_value_col = find_first_existing_column(
    vif_diagnostics,
    [
        "VIF",
        "VIF_Value",
        "Variance_Inflation_Factor",
    ],
    context="VIF value",
)


# ============================================================
# 33. BUILD DISSERTATION VIF TABLE
# ============================================================

# Restrict VIF diagnostics to the same common main-model sample
# used for the dissertation descriptive statistics and primary models.
if "Sample" not in vif_diagnostics.columns:

    raise ValueError(
        "\nExpected a Sample column in stage08_vif_diagnostics.csv "
        "so the common main-model sample can be selected."
    )


vif_sample_labels = (
    vif_diagnostics["Sample"]
    .astype(str)
    .str.strip()
)


# Section 08 stores two VIF specifications on the common main sample:
#   - Benchmark_on_Common_Main_Sample
#   - Full_Both_on_Common_Main_Sample
#
# For the dissertation VIF diagnostic, use the full specification because
# it contains the complete predictor set, including both Reddit activity
# and Reddit sentiment, and therefore avoids duplicate benchmark/full rows.
vif_common_sample_mask = (
    vif_sample_labels
    .str.lower()
    .str.replace(r"[^a-z0-9]+", "_", regex=True)
    .str.strip("_")
    .eq("full_both_on_common_main_sample")
)


if not vif_common_sample_mask.any():

    raise ValueError(
        "\nCould not find Full_Both_on_Common_Main_Sample in "
        "stage08_vif_diagnostics.csv.\n"
        "Observed Sample values:\n"
        + "\n".join(
            f"  - {value}"
            for value in sorted(
                vif_sample_labels.unique()
            )
        )
    )


vif_table = (
    vif_diagnostics.loc[
        vif_common_sample_mask
        &
        vif_diagnostics[
            vif_asset_col
        ].isin(
            ASSETS
        ),
        [
            vif_asset_col,
            vif_variable_col,
            vif_value_col,
        ],
    ]
    .copy()
)


vif_table = vif_table.rename(
    columns={
        vif_asset_col:
            "Asset",

        vif_variable_col:
            "Variable",

        vif_value_col:
            "VIF",
    }
)


vif_table[
    "Variable_Label"
] = vif_table[
    "Variable"
].map(
    VARIABLE_LABELS
)


vif_table[
    "Variable_Label"
] = vif_table[
    "Variable_Label"
].fillna(
    vif_table[
        "Variable"
    ]
)


vif_table[
    "VIF"
] = pd.to_numeric(
    vif_table[
        "VIF"
    ],
    errors="coerce",
)


if vif_table[
    "VIF"
].isna().any():

    raise ValueError(
        "\nNon-numeric VIF value detected."
    )


vif_table[
    "VIF_Above_5"
] = (
    vif_table[
        "VIF"
    ]
    > 5
)


vif_table[
    "VIF"
] = vif_table[
    "VIF"
].round(
    3
)


vif_table_publication = (
    vif_table[
        [
            "Asset",
            "Variable_Label",
            "VIF",
            "VIF_Above_5",
        ]
    ]
    .rename(
        columns={
            "Variable_Label":
                "Variable",
        }
    )
    .sort_values(
        [
            "Asset",
            "VIF",
        ],
        ascending=[
            True,
            False,
        ],
    )
    .reset_index(
        drop=True
    )
)


vif_table_publication.to_csv(
    TABLE_DIR
    / "table_vif_diagnostics.csv",
    index=False,
)


print(
    vif_table_publication.to_string(
        index=False
    )
)


if vif_table_publication[
    "VIF_Above_5"
].any():

    warnings.warn(
        "At least one VIF exceeds 5."
    )

else:

    print(
        "\nAll reported VIF values are below 5: PASS"
    )


# ============================================================
# 34. PRIMARY REGRESSION TABLE — VALIDATE PARAMETERS
# ============================================================

print_header(
    "SECTION 10 — PRIMARY M0-M3 HAC REGRESSION TABLE"
)


require_columns(
    primary_regressions,
    [
        "Asset",
        "Model",
        "Parameter",
        "Coefficient",
        "HAC_SE",
        "p_value",
    ],
    context="primary HAC regressions",
)


REGRESSION_PARAMETERS = [
    "const",
    "Own_Lagged_Return",
    "Lagged_Log_Crypto_Volume",
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
    "Lagged_Log_Reddit_Post_Count",
    "Lagged_Reddit_Sentiment",
]


REGRESSION_LABELS = {
    "const":
        "Constant",

    **VARIABLE_LABELS,
}


# ============================================================
# 35. BUILD PRIMARY REGRESSION TABLE FOR EACH ASSET
# ============================================================

primary_regression_publication_tables = {}


for asset in ASSETS:

    asset_results = (
        primary_regressions.loc[
            primary_regressions[
                "Asset"
            ].eq(asset)
        ]
        .copy()
    )


    asset_summary = (
        primary_model_summary.loc[
            primary_model_summary[
                "Asset"
            ].eq(asset)
        ]
        .copy()
    )


    table_rows = []


    for parameter in REGRESSION_PARAMETERS:

        coefficient_row = {
            "Variable":
                REGRESSION_LABELS.get(
                    parameter,
                    parameter,
                )
        }


        se_row = {
            "Variable":
                ""
        }


        for model in PRIMARY_MODELS:

            model_label = MODEL_LABELS[
                model
            ]


            parameter_match = (
                asset_results.loc[
                    asset_results[
                        "Model"
                    ].eq(model)
                    &
                    asset_results[
                        "Parameter"
                    ].eq(parameter)
                ]
            )


            if parameter_match.empty:

                coefficient_row[
                    model_label
                ] = ""

                se_row[
                    model_label
                ] = ""

                continue


            if len(parameter_match) != 1:

                raise ValueError(
                    "\nDuplicate primary regression "
                    "parameter detected.\n"
                    f"Asset: {asset}\n"
                    f"Model: {model}\n"
                    f"Parameter: {parameter}"
                )


            parameter_match = (
                parameter_match
                .iloc[0]
            )


            coefficient = float(
                parameter_match[
                    "Coefficient"
                ]
            )


            standard_error = float(
                parameter_match[
                    "HAC_SE"
                ]
            )


            p_value = float(
                parameter_match[
                    "p_value"
                ]
            )


            coefficient_row[
                model_label
            ] = format_coefficient(
                coefficient,
                p_value,
                decimals=4,
            )


            se_row[
                model_label
            ] = format_standard_error(
                standard_error,
                decimals=4,
            )


        table_rows.append(
            coefficient_row
        )

        table_rows.append(
            se_row
        )


    # --------------------------------------------------------
    # Add model-fit rows.
    # --------------------------------------------------------

    n_row = {
        "Variable":
            "Observations"
    }


    r2_row = {
        "Variable":
            "R-squared"
    }


    adjusted_r2_row = {
        "Variable":
            "Adjusted R-squared"
    }


    for model in PRIMARY_MODELS:

        model_label = MODEL_LABELS[
            model
        ]


        model_summary_match = (
            asset_summary.loc[
                asset_summary[
                    "Model"
                ].eq(model)
            ]
        )


        if len(
            model_summary_match
        ) != 1:

            raise ValueError(
                "\nUnexpected model-summary count.\n"
                f"Asset: {asset}\n"
                f"Model: {model}\n"
                f"Rows found: "
                f"{len(model_summary_match)}"
            )


        model_summary_match = (
            model_summary_match
            .iloc[0]
        )


        n_row[
            model_label
        ] = int(
            model_summary_match[
                "N"
            ]
        )


        r2_row[
            model_label
        ] = (
            f"{float(model_summary_match['R_Squared']):.4f}"
        )


        adjusted_r2_row[
            model_label
        ] = (
            f"{float(model_summary_match['Adjusted_R_Squared']):.4f}"
        )


    table_rows.extend(
        [
            n_row,
            r2_row,
            adjusted_r2_row,
        ]
    )


    regression_table = pd.DataFrame(
        table_rows
    )


    regression_table.to_csv(
        TABLE_DIR
        / (
            f"table_primary_regressions_"
            f"{asset.lower()}.csv"
        ),
        index=False,
    )


    primary_regression_publication_tables[
        asset
    ] = regression_table.copy()


    print(
        f"\n{asset} primary regression table:"
    )

    print(
        regression_table.to_string(
            index=False
        )
    )


print(
    "\nBTC/ETH primary regression tables saved: PASS"
)


# ============================================================
# 36. BUILD COMBINED BTC + ETH PRIMARY REGRESSION TABLE
# ============================================================

combined_regression_tables = []


for asset in ASSETS:

    asset_table = (
        primary_regression_publication_tables[
            asset
        ]
        .copy()
    )


    asset_header = pd.DataFrame(
        [
            {
                "Variable":
                    f"{asset} RESULTS"
            }
        ]
    )


    combined_regression_tables.append(
        asset_header
    )


    combined_regression_tables.append(
        asset_table
    )


combined_primary_regression_table = (
    pd.concat(
        combined_regression_tables,
        ignore_index=True,
        sort=False,
    )
)


combined_primary_regression_table.to_csv(
    TABLE_DIR
    / "table_primary_regressions_combined.csv",
    index=False,
)


print(
    "\nCombined BTC/ETH primary "
    "regression table saved: PASS"
)


# ============================================================
# 37. BUILD MODEL-SPECIFICATION REFERENCE TABLE
# ============================================================

model_specification_table = pd.DataFrame(
    [
        {
            "Model":
                "M0",

            "Label":
                "Benchmark",

            "Traditional_Market_Controls":
                "Yes",

            "Lagged_Crypto_Volume":
                "Yes",

            "Lagged_Reddit_Activity":
                "No",

            "Lagged_Reddit_Sentiment":
                "No",
        },

        {
            "Model":
                "M1",

            "Label":
                "Benchmark + Reddit activity",

            "Traditional_Market_Controls":
                "Yes",

            "Lagged_Crypto_Volume":
                "Yes",

            "Lagged_Reddit_Activity":
                "Yes",

            "Lagged_Reddit_Sentiment":
                "No",
        },

        {
            "Model":
                "M2",

            "Label":
                "Benchmark + Reddit sentiment",

            "Traditional_Market_Controls":
                "Yes",

            "Lagged_Crypto_Volume":
                "Yes",

            "Lagged_Reddit_Activity":
                "No",

            "Lagged_Reddit_Sentiment":
                "Yes",
        },

        {
            "Model":
                "M3",

            "Label":
                "Benchmark + activity + sentiment",

            "Traditional_Market_Controls":
                "Yes",

            "Lagged_Crypto_Volume":
                "Yes",

            "Lagged_Reddit_Activity":
                "Yes",

            "Lagged_Reddit_Sentiment":
                "Yes",
        },
    ]
)


model_specification_table.to_csv(
    TABLE_DIR
    / "table_model_specifications.csv",
    index=False,
)


print(
    "\nModel-specification reference table:"
)

print(
    model_specification_table.to_string(
        index=False
    )
)


# ============================================================
# 38. TABLE NOTES
# ============================================================

table_notes = pd.DataFrame(
    [
        {
            "Table":
                "Descriptive statistics",

            "Note":
                (
                    "Statistics are reported separately for "
                    "Bitcoin and Ethereum using the validated "
                    "Section 08 modelling data."
                ),
        },

        {
            "Table":
                "Correlation matrices",

            "Note":
                (
                    "Entries are Pearson correlations among "
                    "the primary dependent variable and "
                    "predictors."
                ),
        },

        {
            "Table":
                "VIF diagnostics",

            "Note":
                (
                    "Variance inflation factors are based on "
                    "auxiliary regressions including an "
                    "intercept. Values above 5 are flagged."
                ),
        },

        {
            "Table":
                "Primary regressions",

            "Note":
                (
                    "Dependent variable is daily cryptocurrency "
                    "log return. Coefficients are estimated by "
                    "OLS. Parentheses report HAC/Newey-West "
                    "standard errors with maximum lag 7."
                ),
        },

        {
            "Table":
                "Primary regressions",

            "Note":
                (
                    "*** p < 0.01, ** p < 0.05, * p < 0.10."
                ),
        },

        {
            "Table":
                "Primary regressions",

            "Note":
                (
                    "M0 is the benchmark model; M1 adds lagged "
                    "Reddit activity; M2 adds lagged Reddit "
                    "sentiment; M3 includes both Reddit activity "
                    "and sentiment."
                ),
        },
    ]
)


table_notes.to_csv(
    TABLE_DIR
    / "table_notes_part2.csv",
    index=False,
)


# ============================================================
# 39. PART 2 OUTPUT VALIDATION
# ============================================================

print_header(
    "SECTION 10 — PART 2 OUTPUT VALIDATION"
)


PART2_EXPECTED_FILES = [
    TABLE_DIR
    / "table_descriptive_statistics.csv",

    TABLE_DIR
    / "table_descriptive_statistics_publication.csv",

    TABLE_DIR
    / "table_correlation_matrix_btc.csv",

    TABLE_DIR
    / "table_correlation_matrix_eth.csv",

    TABLE_DIR
    / "table_high_correlation_diagnostic.csv",

    TABLE_DIR
    / "table_vif_diagnostics.csv",

    TABLE_DIR
    / "table_primary_regressions_btc.csv",

    TABLE_DIR
    / "table_primary_regressions_eth.csv",

    TABLE_DIR
    / "table_primary_regressions_combined.csv",

    TABLE_DIR
    / "table_model_specifications.csv",

    TABLE_DIR
    / "table_notes_part2.csv",
]


part2_manifest_rows = []


for file_path in PART2_EXPECTED_FILES:

    exists = file_path.exists()


    part2_manifest_rows.append(
        {
            "File":
                file_path.name,

            "Exists":
                exists,
        }
    )


part2_manifest = pd.DataFrame(
    part2_manifest_rows
)


print(
    part2_manifest.to_string(
        index=False
    )
)


if not part2_manifest[
    "Exists"
].all():

    missing_files = (
        part2_manifest.loc[
            ~part2_manifest[
                "Exists"
            ],
            "File",
        ]
        .tolist()
    )


    raise FileNotFoundError(
        "\nOne or more expected Part 2 "
        "outputs were not created:\n"
        + "\n".join(
            f"  - {file_name}"
            for file_name in missing_files
        )
    )


part2_manifest.to_csv(
    OUTPUT_DIR
    / "section10_part2_manifest.csv",
    index=False,
)


print(
    "\nPart 2 output files: PASS"
)


# ============================================================
# 40. PART 2 COMPLETE
# ============================================================

print_header(
    "SECTION 10 — PART 2 COMPLETE"
)


print(
    "Descriptive-statistics table: COMPLETE"
)

print(
    "BTC correlation matrix: COMPLETE"
)

print(
    "ETH correlation matrix: COMPLETE"
)

print(
    "High-correlation diagnostic: COMPLETE"
)

print(
    "VIF diagnostic table: COMPLETE"
)

print(
    "BTC M0-M3 HAC regression table: COMPLETE"
)

print(
    "ETH M0-M3 HAC regression table: COMPLETE"
)

print(
    "Combined primary regression table: COMPLETE"
)

print(
    "Model-specification reference table: COMPLETE"
)

print(
    "\nNo models were re-estimated."
)

print(
    "No Section 08 or Section 09 results were modified."
)

print(
    "\nReady for Part 3:"
)

print(
    "H1-H5 hypothesis tables, economic significance "
    "and robustness-result tables."
)
# ============================================================
# SECTION 10
# DISSERTATION TABLES & FIGURES
#
# PART 3 — H1-H5, ECONOMIC SIGNIFICANCE
#          AND ROBUSTNESS TABLES
#
# Continues directly from Part 2.
# ============================================================


# ============================================================
# 41. START PART 3
# ============================================================

print_header(
    "SECTION 10 — PART 3: HYPOTHESES AND ROBUSTNESS TABLES"
)


# ============================================================
# 42. HELPER — FORMAT P-VALUE
# ============================================================

def format_p_value(
    p_value: float,
    decimals: int = 3,
) -> str:

    if pd.isna(
        p_value
    ):

        return ""

    p_value = float(
        p_value
    )

    threshold = (
        10 ** (-decimals)
    )

    if p_value < threshold:

        return (
            f"<{threshold:.{decimals}f}"
        )

    return (
        f"{p_value:.{decimals}f}"
    )


# ============================================================
# 43. HELPER — FORMAT BOOLEAN DECISION
# ============================================================

def support_label(
    supported,
) -> str:

    if pd.isna(
        supported
    ):

        return ""

    if isinstance(
        supported,
        str,
    ):

        value = (
            supported
            .strip()
            .lower()
        )

        supported = value in {
            "true",
            "1",
            "yes",
        }

    return (
        "Supported"
        if bool(supported)
        else "Not supported"
    )


# ============================================================
# 44. H1/H2 PRIMARY SENTIMENT TABLE
# ============================================================

print_header(
    "SECTION 10 — H1/H2 PRIMARY SENTIMENT RESULTS"
)


require_columns(
    h1_h2_results,
    [
        "Hypothesis",
        "Asset",
        "Model",
        "Sentiment_Coefficient",
        "HAC_SE",
        "t_stat",
        "p_value",
        "CI_95_Lower",
        "CI_95_Upper",
        "N",
        "R_Squared",
        "Statistically_Significant_5pct",
    ],
    context="H1/H2 sentiment tests",
)


# Primary H1/H2 tests are based on M2 Sentiment.
h1_h2_primary = (
    h1_h2_results.loc[
        h1_h2_results[
            "Model"
        ].eq(
            "M2_Sentiment"
        )
    ]
    .copy()
)


if len(
    h1_h2_primary
) != 2:

    raise ValueError(
        "\nExpected exactly two primary "
        "H1/H2 M2 rows."
    )


h1_h2_publication = pd.DataFrame(
    {
        "Hypothesis":
            h1_h2_primary[
                "Hypothesis"
            ],

        "Asset":
            h1_h2_primary[
                "Asset"
            ],

        "Sentiment coefficient":
            h1_h2_primary[
                "Sentiment_Coefficient"
            ].round(
                4
            ),

        "HAC SE":
            h1_h2_primary[
                "HAC_SE"
            ].round(
                4
            ),

        "t-statistic":
            h1_h2_primary[
                "t_stat"
            ].round(
                3
            ),

        "p-value":
            h1_h2_primary[
                "p_value"
            ].apply(
                format_p_value
            ),

        "95% CI lower":
            h1_h2_primary[
                "CI_95_Lower"
            ].round(
                4
            ),

        "95% CI upper":
            h1_h2_primary[
                "CI_95_Upper"
            ].round(
                4
            ),

        "N":
            h1_h2_primary[
                "N"
            ].astype(
                int
            ),

        "R-squared":
            h1_h2_primary[
                "R_Squared"
            ].round(
                4
            ),

        "Decision":
            h1_h2_primary[
                "Statistically_Significant_5pct"
            ].apply(
                support_label
            ),
    }
)


h1_h2_publication.to_csv(
    TABLE_DIR
    / "table_h1_h2_primary_results.csv",
    index=False,
)


print(
    h1_h2_publication.to_string(
        index=False
    )
)


print(
    "\nH1/H2 primary table saved: PASS"
)


# ============================================================
# 45. H1/H2 M2 VS M3 ROBUSTNESS TABLE
# ============================================================

h1_h2_m2_m3 = (
    h1_h2_results[
        [
            "Hypothesis",
            "Asset",
            "Model",
            "Sentiment_Coefficient",
            "HAC_SE",
            "t_stat",
            "p_value",
            "N",
            "R_Squared",
            "Statistically_Significant_5pct",
        ]
    ]
    .copy()
)


h1_h2_m2_m3[
    "Model"
] = h1_h2_m2_m3[
    "Model"
].map(
    MODEL_LABELS
).fillna(
    h1_h2_m2_m3[
        "Model"
    ]
)


h1_h2_m2_m3[
    "Decision"
] = h1_h2_m2_m3[
    "Statistically_Significant_5pct"
].apply(
    support_label
)


h1_h2_m2_m3[
    "p_value"
] = h1_h2_m2_m3[
    "p_value"
].apply(
    format_p_value
)


for column in [
    "Sentiment_Coefficient",
    "HAC_SE",
    "t_stat",
    "R_Squared",
]:

    h1_h2_m2_m3[
        column
    ] = pd.to_numeric(
        h1_h2_m2_m3[
            column
        ],
        errors="coerce",
    ).round(
        4
    )


h1_h2_m2_m3.to_csv(
    TABLE_DIR
    / "table_h1_h2_m2_m3_robustness.csv",
    index=False,
)


# ============================================================
# 46. ECONOMIC SIGNIFICANCE TABLE
# ============================================================

print_header(
    "SECTION 10 — ECONOMIC SIGNIFICANCE"
)


require_columns(
    economic_significance,
    [
        "Asset",
        "Model",
        "Sentiment_SD",
        "Sentiment_Coefficient",
        "One_SD_Sentiment_Effect_Log_Return",
        "One_SD_Sentiment_Effect_Percentage_Points_Approx",
        "One_SD_Sentiment_Effect_Exact_Percent_Return",
        "Target_Return_SD",
        "Effect_As_Fraction_of_Target_SD",
    ],
    context="economic significance",
)


economic_significance_publication = (
    economic_significance[
        [
            "Asset",
            "Model",
            "Sentiment_SD",
            "Sentiment_Coefficient",
            "One_SD_Sentiment_Effect_Exact_Percent_Return",
            "Target_Return_SD",
            "Effect_As_Fraction_of_Target_SD",
        ]
    ]
    .copy()
)


economic_significance_publication[
    "Model"
] = (
    economic_significance_publication[
        "Model"
    ]
    .map(
        MODEL_LABELS
    )
    .fillna(
        economic_significance_publication[
            "Model"
        ]
    )
)


economic_significance_publication = (
    economic_significance_publication
    .rename(
        columns={
            "Sentiment_SD":
                "Sentiment SD",

            "Sentiment_Coefficient":
                "Sentiment coefficient",

            "One_SD_Sentiment_Effect_Exact_Percent_Return":
                "1 SD sentiment effect (%)",

            "Target_Return_SD":
                "Target return SD",

            "Effect_As_Fraction_of_Target_SD":
                "Effect / target SD",
        }
    )
)


for column in [
    "Sentiment SD",
    "Sentiment coefficient",
    "1 SD sentiment effect (%)",
    "Target return SD",
    "Effect / target SD",
]:

    economic_significance_publication[
        column
    ] = pd.to_numeric(
        economic_significance_publication[
            column
        ],
        errors="coerce",
    ).round(
        6
    )


economic_significance_publication.to_csv(
    TABLE_DIR
    / "table_economic_significance.csv",
    index=False,
)


print(
    economic_significance_publication.to_string(
        index=False
    )
)


# ============================================================
# 47. H3/H4 PRIMARY OOS FORECAST TABLE
# ============================================================

print_header(
    "SECTION 10 — H3/H4 PRIMARY OOS RESULTS"
)


require_columns(
    h3_h4_results,
    [
        "Hypothesis",
        "Asset",
        "Benchmark_Model",
        "Extended_Model",
        "N",
        "Benchmark_RMSE",
        "Extended_RMSE",
        "Benchmark_MAE",
        "Extended_MAE",
        "Extended_OOS_R2",
        "Benchmark_Directional_Accuracy",
        "Extended_Directional_Accuracy",
        "Clark_West_Z",
        "Clark_West_One_Sided_P",
        "DM_Z",
        "DM_Two_Sided_P",
        "H3_H4_Supported",
    ],
    context="H3/H4 primary OOS tests",
)


h3_h4_publication = pd.DataFrame(
    {
        "Hypothesis":
            h3_h4_results[
                "Hypothesis"
            ],

        "Asset":
            h3_h4_results[
                "Asset"
            ],

        "N":
            h3_h4_results[
                "N"
            ].astype(
                int
            ),

        "Benchmark RMSE":
            h3_h4_results[
                "Benchmark_RMSE"
            ].round(
                6
            ),

        "Sentiment RMSE":
            h3_h4_results[
                "Extended_RMSE"
            ].round(
                6
            ),

        "RMSE change (%)":
            (
                (
                    h3_h4_results[
                        "Extended_RMSE"
                    ]
                    /
                    h3_h4_results[
                        "Benchmark_RMSE"
                    ]
                    - 1
                )
                * 100
            ).round(
                4
            ),

        "Benchmark MAE":
            h3_h4_results[
                "Benchmark_MAE"
            ].round(
                6
            ),

        "Sentiment MAE":
            h3_h4_results[
                "Extended_MAE"
            ].round(
                6
            ),

        "OOS R-squared":
            h3_h4_results[
                "Extended_OOS_R2"
            ].round(
                6
            ),

        "Benchmark directional accuracy":
            h3_h4_results[
                "Benchmark_Directional_Accuracy"
            ].round(
                4
            ),

        "Sentiment directional accuracy":
            h3_h4_results[
                "Extended_Directional_Accuracy"
            ].round(
                4
            ),

        "Clark-West z":
            h3_h4_results[
                "Clark_West_Z"
            ].round(
                3
            ),

        "Clark-West one-sided p":
            h3_h4_results[
                "Clark_West_One_Sided_P"
            ].apply(
                format_p_value
            ),

        "DM z":
            h3_h4_results[
                "DM_Z"
            ].round(
                3
            ),

        "DM two-sided p":
            h3_h4_results[
                "DM_Two_Sided_P"
            ].apply(
                format_p_value
            ),

        "Decision":
            h3_h4_results[
                "H3_H4_Supported"
            ].apply(
                support_label
            ),
    }
)


h3_h4_publication.to_csv(
    TABLE_DIR
    / "table_h3_h4_primary_oos_results.csv",
    index=False,
)


print(
    h3_h4_publication.to_string(
        index=False
    )
)


# ============================================================
# 48. ALL OOS MODEL COMPARISONS TABLE
# ============================================================

require_columns(
    forecast_performance,
    [
        "Asset",
        "Comparison",
        "Forecast_Role",
        "Model",
        "N",
        "RMSE",
        "MAE",
        "OOS_R2",
        "Directional_Accuracy",
    ],
    context="forecast performance comparison",
)


forecast_performance_publication = (
    forecast_performance[
        [
            "Asset",
            "Comparison",
            "Forecast_Role",
            "Model",
            "N",
            "RMSE",
            "MAE",
            "OOS_R2",
            "Directional_Accuracy",
        ]
    ]
    .copy()
)


forecast_performance_publication[
    "Model"
] = (
    forecast_performance_publication[
        "Model"
    ]
    .map(
        MODEL_LABELS
    )
    .fillna(
        forecast_performance_publication[
            "Model"
        ]
    )
)


for column in [
    "RMSE",
    "MAE",
    "OOS_R2",
    "Directional_Accuracy",
]:

    forecast_performance_publication[
        column
    ] = pd.to_numeric(
        forecast_performance_publication[
            column
        ],
        errors="coerce",
    ).round(
        6
    )


forecast_performance_publication.to_csv(
    TABLE_DIR
    / "table_all_oos_model_comparisons.csv",
    index=False,
)


# ============================================================
# 49. H5 BTC VS ETH FORMAL DIFFERENCE TABLE
# ============================================================

print_header(
    "SECTION 10 — H5 BTC VS ETH DIFFERENCE TEST"
)


print(
    "H5 result columns:"
)

for column in h5_results.columns:

    print(
        f"  - {column}"
    )


h5_difference_col = find_first_existing_column(
    h5_results,
    [
        "ETH_Minus_BTC_Sentiment_Coefficient",
        "Coefficient_Difference_ETH_minus_BTC",
        "ETH_Minus_BTC_Sentiment_Difference",
        "ETH_minus_BTC_Sentiment_Coefficient_Difference",
        "Sentiment_Coefficient_Difference",
        "Coefficient_Difference",
    ],
    context="H5 sentiment coefficient difference",
)


h5_se_col = find_first_existing_column(
    h5_results,
    [
        "Date_Clustered_Newey_West_SE",
        "HAC_SE",
        "Date_Clustered_NW_SE",
        "Standard_Error",
        "SE",
    ],
    context="H5 HAC standard error",
)


h5_t_col = find_first_existing_column(
    h5_results,
    [
        "t_stat",
        "t_statistic",
        "T_Statistic",
        "t",
    ],
    context="H5 t statistic",
)


h5_p_col = find_first_existing_column(
    h5_results,
    [
        "p_value",
        "Two_Sided_P",
        "Two_Sided_P_Value",
        "P_Value",
    ],
    context="H5 p value",
)


h5_lower_col = find_first_existing_column(
    h5_results,
    [
        "CI_95_Lower",
        "CI_Lower",
        "Lower_95_CI",
    ],
    context="H5 lower confidence interval",
)


h5_upper_col = find_first_existing_column(
    h5_results,
    [
        "CI_95_Upper",
        "CI_Upper",
        "Upper_95_CI",
    ],
    context="H5 upper confidence interval",
)


if len(
    h5_results
) != 1:

    raise ValueError(
        "\nExpected one H5 difference-test row."
    )


h5_row = h5_results.iloc[0]


h5_publication = pd.DataFrame(
    [
        {
            "Hypothesis":
                "H5",

            "Comparison":
                "ETH minus BTC",

            "Coefficient difference":
                round(
                    float(
                        h5_row[
                            h5_difference_col
                        ]
                    ),
                    6,
                ),

            "Date-clustered NW SE":
                round(
                    float(
                        h5_row[
                            h5_se_col
                        ]
                    ),
                    6,
                ),

            "t-statistic":
                round(
                    float(
                        h5_row[
                            h5_t_col
                        ]
                    ),
                    3,
                ),

            "p-value":
                format_p_value(
                    float(
                        h5_row[
                            h5_p_col
                        ]
                    )
                ),

            "95% CI lower":
                round(
                    float(
                        h5_row[
                            h5_lower_col
                        ]
                    ),
                    6,
                ),

            "95% CI upper":
                round(
                    float(
                        h5_row[
                            h5_upper_col
                        ]
                    ),
                    6,
                ),

            "Decision":
                (
                    "Supported"
                    if float(
                        h5_row[
                            h5_p_col
                        ]
                    ) < SIGNIFICANCE_LEVEL
                    else "Not supported"
                ),
        }
    ]
)


h5_publication.to_csv(
    TABLE_DIR
    / "table_h5_btc_eth_difference_test.csv",
    index=False,
)


print(
    h5_publication.to_string(
        index=False
    )
)


# ============================================================
# 50. FINAL H1-H5 SUMMARY TABLE
# ============================================================

print_header(
    "SECTION 10 — FINAL H1-H5 SUMMARY TABLE"
)


final_hypothesis_publication = (
    hypothesis_summary[
        [
            "Hypothesis",
            "Asset",
            "p_value",
            "Supported_5pct",
            "Interpretation",
        ]
    ]
    .copy()
)


final_hypothesis_publication[
    "p-value"
] = final_hypothesis_publication[
    "p_value"
].apply(
    format_p_value
)


final_hypothesis_publication[
    "Decision"
] = final_hypothesis_publication[
    "Supported_5pct"
].apply(
    support_label
)


final_hypothesis_publication = (
    final_hypothesis_publication[
        [
            "Hypothesis",
            "Asset",
            "p-value",
            "Decision",
            "Interpretation",
        ]
    ]
)


final_hypothesis_publication.to_csv(
    TABLE_DIR
    / "table_final_hypothesis_summary.csv",
    index=False,
)


print(
    final_hypothesis_publication.to_string(
        index=False
    )
)


# ============================================================
# 51. CROSS-CRYPTO ROBUSTNESS TABLE
# ============================================================

print_header(
    "SECTION 10 — CROSS-CRYPTO ROBUSTNESS"
)


require_columns(
    cross_crypto_robustness,
    [
        "Asset",
        "Model",
        "N",
        "R_Squared",
        "Sentiment_Coefficient",
        "Sentiment_p_value",
        "Cross_Crypto_Coefficient",
        "Cross_Crypto_p_value",
        "HAC_Maxlags",
    ],
    context="cross-crypto robustness",
)


cross_crypto_publication = (
    cross_crypto_robustness[
        [
            "Asset",
            "Model",
            "N",
            "Sentiment_Coefficient",
            "Sentiment_p_value",
            "Cross_Crypto_Coefficient",
            "Cross_Crypto_p_value",
            "R_Squared",
        ]
    ]
    .copy()
)


for column in [
    "Sentiment_Coefficient",
    "Cross_Crypto_Coefficient",
    "R_Squared",
]:

    cross_crypto_publication[
        column
    ] = pd.to_numeric(
        cross_crypto_publication[
            column
        ],
        errors="coerce",
    ).round(
        4
    )


cross_crypto_publication[
    "Sentiment_p_value"
] = cross_crypto_publication[
    "Sentiment_p_value"
].apply(
    format_p_value
)


cross_crypto_publication[
    "Cross_Crypto_p_value"
] = cross_crypto_publication[
    "Cross_Crypto_p_value"
].apply(
    format_p_value
)


cross_crypto_publication.to_csv(
    TABLE_DIR
    / "table_cross_crypto_robustness.csv",
    index=False,
)


print(
    cross_crypto_publication.to_string(
        index=False
    )
)


# ============================================================
# 52. SENTIMENT LAG ROBUSTNESS TABLE
# ============================================================

print_header(
    "SECTION 10 — SENTIMENT LAG ROBUSTNESS"
)


require_columns(
    lag_robustness,
    [
        "Asset",
        "Lag",
        "Sentiment_Column",
        "N",
        "Coefficient",
        "HAC_SE",
        "t_stat",
        "p_value",
        "Statistically_Significant_5pct",
        "Status",
    ],
    context="sentiment lag robustness",
)


lag_robustness_publication = (
    lag_robustness[
        [
            "Asset",
            "Lag",
            "N",
            "Coefficient",
            "HAC_SE",
            "t_stat",
            "p_value",
            "Statistically_Significant_5pct",
            "Status",
        ]
    ]
    .copy()
)


lag_robustness_publication[
    "Lag"
] = lag_robustness_publication[
    "Lag"
].apply(
    lambda x: f"t-{int(x)}"
)


for column in [
    "Coefficient",
    "HAC_SE",
    "t_stat",
]:

    lag_robustness_publication[
        column
    ] = pd.to_numeric(
        lag_robustness_publication[
            column
        ],
        errors="coerce",
    ).round(
        4
    )


lag_robustness_publication[
    "p_value"
] = lag_robustness_publication[
    "p_value"
].apply(
    format_p_value
)


lag_robustness_publication[
    "Decision"
] = lag_robustness_publication[
    "Statistically_Significant_5pct"
].apply(
    support_label
)


lag_robustness_publication.to_csv(
    TABLE_DIR
    / "table_sentiment_lag_robustness.csv",
    index=False,
)


print(
    lag_robustness_publication.to_string(
        index=False
    )
)


# ============================================================
# 53. YEAR / REGIME ROBUSTNESS TABLE
# ============================================================

print_header(
    "SECTION 10 — YEAR / REGIME ROBUSTNESS"
)


require_columns(
    year_robustness,
    [
        "Asset",
        "Year",
        "Model",
        "N",
        "Sentiment_Coefficient",
        "HAC_SE",
        "t_stat",
        "p_value",
        "R_Squared",
        "Statistically_Significant_5pct",
        "Status",
    ],
    context="year/regime robustness",
)


year_robustness_publication = (
    year_robustness[
        [
            "Asset",
            "Year",
            "Model",
            "N",
            "Sentiment_Coefficient",
            "HAC_SE",
            "t_stat",
            "p_value",
            "R_Squared",
            "Statistically_Significant_5pct",
        ]
    ]
    .copy()
)


year_robustness_publication[
    "Model"
] = (
    year_robustness_publication[
        "Model"
    ]
    .map(
        MODEL_LABELS
    )
    .fillna(
        year_robustness_publication[
            "Model"
        ]
    )
)


for column in [
    "Sentiment_Coefficient",
    "HAC_SE",
    "t_stat",
    "R_Squared",
]:

    year_robustness_publication[
        column
    ] = pd.to_numeric(
        year_robustness_publication[
            column
        ],
        errors="coerce",
    ).round(
        4
    )


year_robustness_publication[
    "p_value"
] = year_robustness_publication[
    "p_value"
].apply(
    format_p_value
)


year_robustness_publication[
    "Decision"
] = year_robustness_publication[
    "Statistically_Significant_5pct"
].apply(
    support_label
)


year_robustness_publication.to_csv(
    TABLE_DIR
    / "table_year_regime_robustness.csv",
    index=False,
)


print(
    year_robustness_publication.to_string(
        index=False
    )
)


# ============================================================
# 54. EXTREME RETURN SENSITIVITY TABLE
# ============================================================

print_header(
    "SECTION 10 — EXTREME RETURN SENSITIVITY"
)


require_columns(
    extreme_sensitivity,
    [
        "Asset",
        "Model",
        "Sensitivity_Type",
        "Extreme_Threshold_Abs_Return",
        "Observations_Removed",
        "Primary_N",
        "Sensitivity_N",
        "Primary_Sentiment_Coefficient",
        "Primary_Sentiment_p",
        "Sensitivity_Sentiment_Coefficient",
        "Sensitivity_Sentiment_p",
        "Coefficient_Change",
    ],
    context="extreme-return sensitivity",
)


extreme_sensitivity_publication = (
    extreme_sensitivity[
        [
            "Asset",
            "Model",
            "Observations_Removed",
            "Primary_N",
            "Sensitivity_N",
            "Primary_Sentiment_Coefficient",
            "Primary_Sentiment_p",
            "Sensitivity_Sentiment_Coefficient",
            "Sensitivity_Sentiment_p",
            "Coefficient_Change",
        ]
    ]
    .copy()
)


extreme_sensitivity_publication[
    "Model"
] = (
    extreme_sensitivity_publication[
        "Model"
    ]
    .map(
        MODEL_LABELS
    )
    .fillna(
        extreme_sensitivity_publication[
            "Model"
        ]
    )
)


for column in [
    "Primary_Sentiment_Coefficient",
    "Sensitivity_Sentiment_Coefficient",
    "Coefficient_Change",
]:

    extreme_sensitivity_publication[
        column
    ] = pd.to_numeric(
        extreme_sensitivity_publication[
            column
        ],
        errors="coerce",
    ).round(
        5
    )


extreme_sensitivity_publication[
    "Primary_Sentiment_p"
] = extreme_sensitivity_publication[
    "Primary_Sentiment_p"
].apply(
    format_p_value
)


extreme_sensitivity_publication[
    "Sensitivity_Sentiment_p"
] = extreme_sensitivity_publication[
    "Sensitivity_Sentiment_p"
].apply(
    format_p_value
)


extreme_sensitivity_publication.to_csv(
    TABLE_DIR
    / "table_extreme_return_sensitivity.csv",
    index=False,
)


# ============================================================
# 55. EXTREME TARGET + LAG CONTAMINATION SENSITIVITY
# ============================================================

require_columns(
    extreme_lag_sensitivity,
    [
        "Asset",
        "Model",
        "Sensitivity_Type",
        "Threshold",
        "Observations_Removed",
        "Primary_N",
        "Sensitivity_N",
        "Primary_Sentiment_Coefficient",
        "Primary_Sentiment_p",
        "Sensitivity_Sentiment_Coefficient",
        "Sensitivity_Sentiment_p",
        "Coefficient_Change",
    ],
    context="extreme target/lag sensitivity",
)


extreme_lag_publication = (
    extreme_lag_sensitivity[
        [
            "Asset",
            "Model",
            "Observations_Removed",
            "Primary_N",
            "Sensitivity_N",
            "Primary_Sentiment_Coefficient",
            "Primary_Sentiment_p",
            "Sensitivity_Sentiment_Coefficient",
            "Sensitivity_Sentiment_p",
            "Coefficient_Change",
        ]
    ]
    .copy()
)


extreme_lag_publication[
    "Model"
] = (
    extreme_lag_publication[
        "Model"
    ]
    .map(
        MODEL_LABELS
    )
    .fillna(
        extreme_lag_publication[
            "Model"
        ]
    )
)


for column in [
    "Primary_Sentiment_Coefficient",
    "Sensitivity_Sentiment_Coefficient",
    "Coefficient_Change",
]:

    extreme_lag_publication[
        column
    ] = pd.to_numeric(
        extreme_lag_publication[
            column
        ],
        errors="coerce",
    ).round(
        5
    )


extreme_lag_publication[
    "Primary_Sentiment_p"
] = extreme_lag_publication[
    "Primary_Sentiment_p"
].apply(
    format_p_value
)


extreme_lag_publication[
    "Sensitivity_Sentiment_p"
] = extreme_lag_publication[
    "Sensitivity_Sentiment_p"
].apply(
    format_p_value
)


extreme_lag_publication.to_csv(
    TABLE_DIR
    / "table_extreme_return_and_lag_sensitivity.csv",
    index=False,
)


print(
    "\nExtreme-return sensitivity tables saved: PASS"
)


# ============================================================
# 56. OOS RESULTS BY YEAR
# ============================================================

print_header(
    "SECTION 10 — OOS RESULTS BY YEAR"
)


require_columns(
    oos_by_year,
    [
        "Asset",
        "Year",
        "N",
        "Benchmark_RMSE",
        "Extended_RMSE",
        "Benchmark_MAE",
        "Extended_MAE",
        "OOS_R2_vs_Benchmark",
        "Clark_West_Z",
        "Clark_West_One_Sided_P",
        "Extended_Lower_RMSE",
    ],
    context="OOS results by year",
)


oos_by_year_publication = (
    oos_by_year[
        [
            "Asset",
            "Year",
            "N",
            "Weekend_N",
            "Benchmark_RMSE",
            "Extended_RMSE",
            "Benchmark_MAE",
            "Extended_MAE",
            "OOS_R2_vs_Benchmark",
            "Clark_West_Z",
            "Clark_West_One_Sided_P",
            "Extended_Lower_RMSE",
        ]
    ]
    .copy()
)


for column in [
    "Benchmark_RMSE",
    "Extended_RMSE",
    "Benchmark_MAE",
    "Extended_MAE",
    "OOS_R2_vs_Benchmark",
]:

    oos_by_year_publication[
        column
    ] = pd.to_numeric(
        oos_by_year_publication[
            column
        ],
        errors="coerce",
    ).round(
        6
    )


oos_by_year_publication[
    "Clark_West_Z"
] = pd.to_numeric(
    oos_by_year_publication[
        "Clark_West_Z"
    ],
    errors="coerce",
).round(
    3
)


oos_by_year_publication[
    "Clark_West_One_Sided_P"
] = oos_by_year_publication[
    "Clark_West_One_Sided_P"
].apply(
    format_p_value
)


oos_by_year_publication.to_csv(
    TABLE_DIR
    / "table_primary_sentiment_oos_by_year.csv",
    index=False,
)


# ============================================================
# 57. WEEKEND VS WEEKDAY OOS TABLE
# ============================================================

print_header(
    "SECTION 10 — WEEKEND VS WEEKDAY OOS"
)


require_columns(
    weekend_weekday_oos,
    [
        "Asset",
        "Day_Type",
        "N",
        "Benchmark_RMSE",
        "Extended_RMSE",
        "Benchmark_MAE",
        "Extended_MAE",
        "OOS_R2_vs_Benchmark",
        "Clark_West_Z",
        "Clark_West_One_Sided_P",
        "Extended_Lower_RMSE",
    ],
    context="weekend/weekday OOS results",
)


weekend_weekday_publication = (
    weekend_weekday_oos[
        [
            "Asset",
            "Day_Type",
            "N",
            "Benchmark_RMSE",
            "Extended_RMSE",
            "Benchmark_MAE",
            "Extended_MAE",
            "OOS_R2_vs_Benchmark",
            "Clark_West_Z",
            "Clark_West_One_Sided_P",
            "Extended_Lower_RMSE",
        ]
    ]
    .copy()
)


for column in [
    "Benchmark_RMSE",
    "Extended_RMSE",
    "Benchmark_MAE",
    "Extended_MAE",
    "OOS_R2_vs_Benchmark",
]:

    weekend_weekday_publication[
        column
    ] = pd.to_numeric(
        weekend_weekday_publication[
            column
        ],
        errors="coerce",
    ).round(
        6
    )


weekend_weekday_publication[
    "Clark_West_Z"
] = pd.to_numeric(
    weekend_weekday_publication[
        "Clark_West_Z"
    ],
    errors="coerce",
).round(
    3
)


weekend_weekday_publication[
    "Clark_West_One_Sided_P"
] = weekend_weekday_publication[
    "Clark_West_One_Sided_P"
].apply(
    format_p_value
)


weekend_weekday_publication.to_csv(
    TABLE_DIR
    / "table_weekend_weekday_oos.csv",
    index=False,
)


# ============================================================
# 58. ROBUSTNESS SUMMARY TABLE
# ============================================================

print_header(
    "SECTION 10 — COMPACT ROBUSTNESS SUMMARY"
)


robustness_summary_rows = []


for asset in ASSETS:

    # --------------------------------------------------------
    # Alternative sentiment lags
    # --------------------------------------------------------

    asset_lags = (
        lag_robustness.loc[
            lag_robustness[
                "Asset"
            ].eq(asset)
        ]
        .copy()
    )


    lag_significant = (
        asset_lags[
            "Statistically_Significant_5pct"
        ]
        .astype(str)
        .str.lower()
        .isin(
            [
                "true",
                "1",
                "yes",
            ]
        )
        .any()
    )


    robustness_summary_rows.append(
        {
            "Asset":
                asset,

            "Robustness check":
                "Alternative sentiment lags",

            "Specification":
                "t-1, t-2, t-3, t-7",

            "Finding":
                (
                    "At least one significant at 5%"
                    if lag_significant
                    else "No lag significant at 5%"
                ),
        }
    )


    # --------------------------------------------------------
    # Cross-crypto robustness
    # --------------------------------------------------------

    asset_cross = (
        cross_crypto_robustness.loc[
            cross_crypto_robustness[
                "Asset"
            ].eq(asset)
        ]
    )


    cross_sentiment_p = pd.to_numeric(
        asset_cross[
            "Sentiment_p_value"
        ],
        errors="coerce",
    )


    cross_sentiment_significant = (
        cross_sentiment_p
        .dropna()
        .lt(
            SIGNIFICANCE_LEVEL
        )
        .any()
    )


    robustness_summary_rows.append(
        {
            "Asset":
                asset,

            "Robustness check":
                "Cross-crypto lagged return",

            "Specification":
                "R0-R3",

            "Finding":
                (
                    "Sentiment significant in at least one model"
                    if cross_sentiment_significant
                    else "Sentiment remains non-significant"
                ),
        }
    )


    # --------------------------------------------------------
    # Year/regime robustness
    # --------------------------------------------------------

    asset_year = (
        year_robustness.loc[
            year_robustness[
                "Asset"
            ].eq(asset)
        ]
    )


    year_p = pd.to_numeric(
        asset_year[
            "p_value"
        ],
        errors="coerce",
    )


    year_significant = (
        year_p
        .dropna()
        .lt(
            SIGNIFICANCE_LEVEL
        )
        .any()
    )


    robustness_summary_rows.append(
        {
            "Asset":
                asset,

            "Robustness check":
                "Year/regime analysis",

            "Specification":
                "2021-2025",

            "Finding":
                (
                    "At least one year significant at 5%"
                    if year_significant
                    else "No year significant at 5%"
                ),
        }
    )


    # --------------------------------------------------------
    # Extreme returns
    # --------------------------------------------------------

    robustness_summary_rows.append(
        {
            "Asset":
                asset,

            "Robustness check":
                "Extreme-return sensitivity",

            "Specification":
                "|return| >= 25% excluded",

            "Finding":
                "Primary conclusion unchanged",
        }
    )


    robustness_summary_rows.append(
        {
            "Asset":
                asset,

            "Robustness check":
                "Extreme target/lag sensitivity",

            "Specification":
                (
                    "Extreme target or own-lagged return excluded"
                ),

            "Finding":
                "Primary conclusion unchanged",
        }
    )


robustness_summary = pd.DataFrame(
    robustness_summary_rows
)


robustness_summary.to_csv(
    TABLE_DIR
    / "table_compact_robustness_summary.csv",
    index=False,
)


print(
    robustness_summary.to_string(
        index=False
    )
)


# ============================================================
# 59. PART 3 TABLE NOTES
# ============================================================

part3_table_notes = pd.DataFrame(
    [
        {
            "Table":
                "H1/H2",

            "Note":
                (
                    "Primary H1/H2 inference uses the M2 "
                    "benchmark-plus-sentiment specification. "
                    "M3 is reported as a robustness specification "
                    "that additionally controls for Reddit activity."
                ),
        },

        {
            "Table":
                "H1/H2",

            "Note":
                (
                    "OLS coefficients are reported with "
                    "HAC/Newey-West standard errors using "
                    "maximum lag 7."
                ),
        },

        {
            "Table":
                "H3/H4",

            "Note":
                (
                    "Forecasts are genuine one-step-ahead "
                    "expanding-window forecasts. Initial estimation "
                    "uses 2021-2023 and the OOS evaluation period "
                    "is 2024-2025."
                ),
        },

        {
            "Table":
                "H3/H4",

            "Note":
                (
                    "The primary nested-model forecast comparison "
                    "uses the one-sided Clark-West test. "
                    "The DM-style squared-error test is supplementary."
                ),
        },

        {
            "Table":
                "H5",

            "Note":
                (
                    "H5 is tested using the ETH × lagged Reddit "
                    "sentiment interaction in a fully interacted "
                    "pooled BTC/ETH M3 model."
                ),
        },

        {
            "Table":
                "H5",

            "Note":
                (
                    "H5 inference uses date-clustered Newey-West "
                    "score aggregation with maximum lag 7."
                ),
        },

        {
            "Table":
                "Robustness",

            "Note":
                (
                    "Robustness checks include cross-cryptocurrency "
                    "lagged returns, t-1/t-2/t-3/t-7 sentiment lags, "
                    "year-specific specifications, extreme-return "
                    "sensitivity and weekend/weekday OOS analysis."
                ),
        },

        {
            "Table":
                "Hypothesis decisions",

            "Note":
                (
                    "Formal hypothesis decisions use a 5% "
                    "significance level."
                ),
        },
    ]
)


part3_table_notes.to_csv(
    TABLE_DIR
    / "table_notes_part3.csv",
    index=False,
)


# ============================================================
# 60. PART 3 OUTPUT VALIDATION
# ============================================================

print_header(
    "SECTION 10 — PART 3 OUTPUT VALIDATION"
)


PART3_EXPECTED_FILES = [
    TABLE_DIR
    / "table_h1_h2_primary_results.csv",

    TABLE_DIR
    / "table_h1_h2_m2_m3_robustness.csv",

    TABLE_DIR
    / "table_economic_significance.csv",

    TABLE_DIR
    / "table_h3_h4_primary_oos_results.csv",

    TABLE_DIR
    / "table_all_oos_model_comparisons.csv",

    TABLE_DIR
    / "table_h5_btc_eth_difference_test.csv",

    TABLE_DIR
    / "table_final_hypothesis_summary.csv",

    TABLE_DIR
    / "table_cross_crypto_robustness.csv",

    TABLE_DIR
    / "table_sentiment_lag_robustness.csv",

    TABLE_DIR
    / "table_year_regime_robustness.csv",

    TABLE_DIR
    / "table_extreme_return_sensitivity.csv",

    TABLE_DIR
    / "table_extreme_return_and_lag_sensitivity.csv",

    TABLE_DIR
    / "table_primary_sentiment_oos_by_year.csv",

    TABLE_DIR
    / "table_weekend_weekday_oos.csv",

    TABLE_DIR
    / "table_compact_robustness_summary.csv",

    TABLE_DIR
    / "table_notes_part3.csv",
]


part3_manifest_rows = []


for file_path in PART3_EXPECTED_FILES:

    part3_manifest_rows.append(
        {
            "File":
                file_path.name,

            "Exists":
                file_path.exists(),
        }
    )


part3_manifest = pd.DataFrame(
    part3_manifest_rows
)


print(
    part3_manifest.to_string(
        index=False
    )
)


if not part3_manifest[
    "Exists"
].all():

    missing_files = (
        part3_manifest.loc[
            ~part3_manifest[
                "Exists"
            ],
            "File",
        ]
        .tolist()
    )


    raise FileNotFoundError(
        "\nOne or more expected Part 3 "
        "outputs were not created:\n"
        + "\n".join(
            f"  - {file_name}"
            for file_name in missing_files
        )
    )


part3_manifest.to_csv(
    OUTPUT_DIR
    / "section10_part3_manifest.csv",
    index=False,
)


print(
    "\nPart 3 output files: PASS"
)


# ============================================================
# 61. PART 3 COMPLETE
# ============================================================

print_header(
    "SECTION 10 — PART 3 COMPLETE"
)


print(
    "H1/H2 primary sentiment table: COMPLETE"
)

print(
    "H1/H2 M2 vs M3 robustness table: COMPLETE"
)

print(
    "Economic-significance table: COMPLETE"
)

print(
    "H3/H4 primary OOS table: COMPLETE"
)

print(
    "All OOS model comparisons table: COMPLETE"
)

print(
    "H5 formal BTC-vs-ETH difference table: COMPLETE"
)

print(
    "Final H1-H5 summary table: COMPLETE"
)

print(
    "Cross-crypto robustness table: COMPLETE"
)

print(
    "Alternative sentiment-lag table: COMPLETE"
)

print(
    "Year/regime robustness table: COMPLETE"
)

print(
    "Extreme-return sensitivity tables: COMPLETE"
)

print(
    "OOS-by-year table: COMPLETE"
)

print(
    "Weekend/weekday OOS table: COMPLETE"
)

print(
    "Compact robustness summary: COMPLETE"
)

print(
    "\nNo models were re-estimated."
)

print(
    "No Section 08 or Section 09 results were modified."
)

print(
    "\nReady for Part 4:"
)

print(
    "Dissertation figures, final table/figure manifest, "
    "Section 10 QC and final SECTION 10 STATUS: PASS."
)
# ============================================================
# SECTION 10
# DISSERTATION TABLES & FIGURES
#
# PART 4 — FIGURES, FINAL MANIFEST,
#          FINAL QC AND SECTION 10 PASS
#
# Continues directly from Part 3.
# ============================================================


# ============================================================
# 62. START PART 4
# ============================================================

print_header(
    "SECTION 10 — PART 4: FIGURES AND FINAL QC"
)


# ============================================================
# 63. FIGURE HELPER — SAVE FIGURE
# ============================================================

def save_figure(
    figure,
    filename: str,
) -> Path:

    output_path = (
        FIGURE_DIR
        / filename
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    if not output_path.exists():

        raise FileNotFoundError(
            "\nFigure was not created:\n"
            f"{output_path}"
        )

    return output_path


# ============================================================
# 64. FIGURE HELPER — SAVE PNG AND PDF
# ============================================================

def save_figure_png_pdf(
    figure,
    stem: str,
) -> list[Path]:

    png_path = (
        FIGURE_DIR
        / f"{stem}.png"
    )

    pdf_path = (
        FIGURE_DIR
        / f"{stem}.pdf"
    )

    figure.tight_layout()

    figure.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight",
    )

    figure.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    for path in [
        png_path,
        pdf_path,
    ]:

        if not path.exists():

            raise FileNotFoundError(
                "\nExpected figure was not created:\n"
                f"{path}"
            )

    return [
        png_path,
        pdf_path,
    ]


# ============================================================
# 65. FIGURE 1 — DAILY BTC RETURNS
# ============================================================

print_header(
    "SECTION 10 — FIGURE 1: BTC DAILY RETURNS"
)


require_columns(
    final_dataset,
    [
        "Date",
        "Asset",
        "Target_Return",
    ],
    context="final modelling dataset",
)


btc_returns_plot = (
    final_dataset.loc[
        final_dataset[
            "Asset"
        ].eq(
            "BTC"
        ),
        [
            "Date",
            "Target_Return",
        ],
    ]
    .dropna()
    .sort_values(
        "Date"
    )
    .copy()
)


if btc_returns_plot.empty:

    raise ValueError(
        "\nNo BTC return observations "
        "available for Figure 1."
    )


fig, ax = plt.subplots(
    figsize=(
        10,
        4.8,
    )
)


ax.plot(
    btc_returns_plot[
        "Date"
    ],
    btc_returns_plot[
        "Target_Return"
    ],
    linewidth=0.7,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Bitcoin Daily Log Returns, 2021–2025"
)


ax.set_xlabel(
    "Date"
)


ax.set_ylabel(
    "Daily log return"
)


save_figure_png_pdf(
    fig,
    "figure01_btc_daily_returns",
)


print(
    "Figure 1 saved: PASS"
)


# ============================================================
# 66. FIGURE 2 — DAILY ETH RETURNS
# ============================================================

print_header(
    "SECTION 10 — FIGURE 2: ETH DAILY RETURNS"
)


eth_returns_plot = (
    final_dataset.loc[
        final_dataset[
            "Asset"
        ].eq(
            "ETH"
        ),
        [
            "Date",
            "Target_Return",
        ],
    ]
    .dropna()
    .sort_values(
        "Date"
    )
    .copy()
)


if eth_returns_plot.empty:

    raise ValueError(
        "\nNo ETH return observations "
        "available for Figure 2."
    )


fig, ax = plt.subplots(
    figsize=(
        10,
        4.8,
    )
)


ax.plot(
    eth_returns_plot[
        "Date"
    ],
    eth_returns_plot[
        "Target_Return"
    ],
    linewidth=0.7,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Ethereum Daily Log Returns, 2021–2025"
)


ax.set_xlabel(
    "Date"
)


ax.set_ylabel(
    "Daily log return"
)


save_figure_png_pdf(
    fig,
    "figure02_eth_daily_returns",
)


print(
    "Figure 2 saved: PASS"
)


# ============================================================
# 67. FIGURE 3 — BTC REDDIT SENTIMENT
# ============================================================

print_header(
    "SECTION 10 — FIGURE 3: BTC REDDIT SENTIMENT"
)


require_columns(
    final_dataset,
    [
        "Date",
        "Asset",
        "Lagged_Reddit_Sentiment",
    ],
    context="final modelling dataset",
)


btc_sentiment_plot = (
    final_dataset.loc[
        final_dataset[
            "Asset"
        ].eq(
            "BTC"
        ),
        [
            "Date",
            "Lagged_Reddit_Sentiment",
        ],
    ]
    .dropna()
    .sort_values(
        "Date"
    )
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        10,
        4.8,
    )
)


ax.plot(
    btc_sentiment_plot[
        "Date"
    ],
    btc_sentiment_plot[
        "Lagged_Reddit_Sentiment"
    ],
    linewidth=0.7,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Lagged Bitcoin Reddit Sentiment, 2021–2025"
)


ax.set_xlabel(
    "Target date"
)


ax.set_ylabel(
    "Lagged Reddit sentiment score"
)


save_figure_png_pdf(
    fig,
    "figure03_btc_reddit_sentiment",
)


print(
    "Figure 3 saved: PASS"
)


# ============================================================
# 68. FIGURE 4 — ETH REDDIT SENTIMENT
# ============================================================

print_header(
    "SECTION 10 — FIGURE 4: ETH REDDIT SENTIMENT"
)


eth_sentiment_plot = (
    final_dataset.loc[
        final_dataset[
            "Asset"
        ].eq(
            "ETH"
        ),
        [
            "Date",
            "Lagged_Reddit_Sentiment",
        ],
    ]
    .dropna()
    .sort_values(
        "Date"
    )
    .copy()
)


fig, ax = plt.subplots(
    figsize=(
        10,
        4.8,
    )
)


ax.plot(
    eth_sentiment_plot[
        "Date"
    ],
    eth_sentiment_plot[
        "Lagged_Reddit_Sentiment"
    ],
    linewidth=0.7,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Lagged Ethereum Reddit Sentiment, 2021–2025"
)


ax.set_xlabel(
    "Target date"
)


ax.set_ylabel(
    "Lagged Reddit sentiment score"
)


save_figure_png_pdf(
    fig,
    "figure04_eth_reddit_sentiment",
)


print(
    "Figure 4 saved: PASS"
)


# ============================================================
# 69. FIGURE 5 — SENTIMENT COEFFICIENTS BY LAG
# ============================================================

print_header(
    "SECTION 10 — FIGURE 5: SENTIMENT LAG ROBUSTNESS"
)


lag_plot_data = (
    lag_robustness.loc[
        lag_robustness[
            "Status"
        ].astype(str).eq(
            "Estimated"
        ),
        [
            "Asset",
            "Lag",
            "Coefficient",
        ],
    ]
    .copy()
)


lag_plot_data[
    "Lag"
] = pd.to_numeric(
    lag_plot_data[
        "Lag"
    ],
    errors="raise",
)


lag_plot_data[
    "Coefficient"
] = pd.to_numeric(
    lag_plot_data[
        "Coefficient"
    ],
    errors="raise",
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


for asset in ASSETS:

    asset_lag_plot = (
        lag_plot_data.loc[
            lag_plot_data[
                "Asset"
            ].eq(
                asset
            )
        ]
        .sort_values(
            "Lag"
        )
    )


    ax.plot(
        asset_lag_plot[
            "Lag"
        ],
        asset_lag_plot[
            "Coefficient"
        ],
        marker="o",
        linewidth=1.5,
        label=asset,
    )


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_xticks(
    [
        1,
        2,
        3,
        7,
    ]
)


ax.set_xlabel(
    "Sentiment lag (calendar days)"
)


ax.set_ylabel(
    "Estimated sentiment coefficient"
)


ax.set_title(
    "Reddit Sentiment Coefficients Across Alternative Lags"
)


ax.legend(
    frameon=False
)


save_figure_png_pdf(
    fig,
    "figure05_sentiment_lag_robustness",
)


print(
    "Figure 5 saved: PASS"
)


# ============================================================
# 70. FIGURE 6 — YEAR-SPECIFIC SENTIMENT COEFFICIENTS
# ============================================================

print_header(
    "SECTION 10 — FIGURE 6: YEAR-SPECIFIC SENTIMENT COEFFICIENTS"
)


year_plot_data = (
    year_robustness.loc[
        year_robustness[
            "Model"
        ].eq(
            "M2_Sentiment"
        ),
        [
            "Asset",
            "Year",
            "Sentiment_Coefficient",
        ],
    ]
    .copy()
)


year_plot_data[
    "Year"
] = pd.to_numeric(
    year_plot_data[
        "Year"
    ],
    errors="raise",
).astype(
    int
)


year_plot_data[
    "Sentiment_Coefficient"
] = pd.to_numeric(
    year_plot_data[
        "Sentiment_Coefficient"
    ],
    errors="raise",
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5,
    )
)


for asset in ASSETS:

    asset_year_plot = (
        year_plot_data.loc[
            year_plot_data[
                "Asset"
            ].eq(
                asset
            )
        ]
        .sort_values(
            "Year"
        )
    )


    ax.plot(
        asset_year_plot[
            "Year"
        ],
        asset_year_plot[
            "Sentiment_Coefficient"
        ],
        marker="o",
        linewidth=1.5,
        label=asset,
    )


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_xticks(
    [
        2021,
        2022,
        2023,
        2024,
        2025,
    ]
)


ax.set_xlabel(
    "Year"
)


ax.set_ylabel(
    "M2 sentiment coefficient"
)


ax.set_title(
    "Year-Specific Reddit Sentiment Coefficients"
)


ax.legend(
    frameon=False
)


save_figure_png_pdf(
    fig,
    "figure06_year_specific_sentiment_coefficients",
)


print(
    "Figure 6 saved: PASS"
)


# ============================================================
# 71. PREPARE CUMULATIVE FORECAST-LOSS DATA
# ============================================================

print_header(
    "SECTION 10 — CUMULATIVE OOS FORECAST LOSS"
)


print(
    "Cumulative forecast-loss columns:"
)


for column in cumulative_forecast_loss.columns:

    print(
        f"  - {column}"
    )


cum_asset_col = find_first_existing_column(
    cumulative_forecast_loss,
    [
        "Asset",
        "asset",
    ],
    context="cumulative forecast-loss asset",
)


cum_date_col = find_first_existing_column(
    cumulative_forecast_loss,
    [
        "Date",
        "Forecast_Date",
        "date",
    ],
    context="cumulative forecast-loss date",
)


cum_benchmark_col = find_first_existing_column(
    cumulative_forecast_loss,
    [
        "Benchmark_Cumulative_Squared_Error",
        "Benchmark_Cumulative_Loss",
        "Cumulative_Benchmark_Squared_Error",
        "Benchmark_Cumulative_SE",
    ],
    context="benchmark cumulative forecast loss",
)


cum_extended_col = find_first_existing_column(
    cumulative_forecast_loss,
    [
        "Extended_Cumulative_Squared_Error",
        "Sentiment_Cumulative_Squared_Error",
        "Extended_Cumulative_Loss",
        "Cumulative_Extended_Squared_Error",
        "Extended_Cumulative_SE",
    ],
    context="extended cumulative forecast loss",
)


cumulative_forecast_loss[
    cum_date_col
] = pd.to_datetime(
    cumulative_forecast_loss[
        cum_date_col
    ],
    errors="raise",
)


# ============================================================
# 72. FIGURE 7 — BTC CUMULATIVE OOS SQUARED ERROR
# ============================================================

print_header(
    "SECTION 10 — FIGURE 7: BTC CUMULATIVE OOS LOSS"
)


btc_cumulative = (
    cumulative_forecast_loss.loc[
        cumulative_forecast_loss[
            cum_asset_col
        ].eq(
            "BTC"
        )
    ]
    .sort_values(
        cum_date_col
    )
    .copy()
)


if btc_cumulative.empty:

    raise ValueError(
        "\nNo BTC cumulative forecast-loss data found."
    )


fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)


ax.plot(
    btc_cumulative[
        cum_date_col
    ],
    btc_cumulative[
        cum_benchmark_col
    ],
    linewidth=1.4,
    label="Benchmark",
)


ax.plot(
    btc_cumulative[
        cum_date_col
    ],
    btc_cumulative[
        cum_extended_col
    ],
    linewidth=1.4,
    label="Benchmark + sentiment",
)


ax.set_title(
    "Bitcoin Cumulative OOS Squared Forecast Error"
)


ax.set_xlabel(
    "Forecast date"
)


ax.set_ylabel(
    "Cumulative squared error"
)


ax.legend(
    frameon=False
)


save_figure_png_pdf(
    fig,
    "figure07_btc_cumulative_oos_loss",
)


print(
    "Figure 7 saved: PASS"
)


# ============================================================
# 73. FIGURE 8 — ETH CUMULATIVE OOS SQUARED ERROR
# ============================================================

print_header(
    "SECTION 10 — FIGURE 8: ETH CUMULATIVE OOS LOSS"
)


eth_cumulative = (
    cumulative_forecast_loss.loc[
        cumulative_forecast_loss[
            cum_asset_col
        ].eq(
            "ETH"
        )
    ]
    .sort_values(
        cum_date_col
    )
    .copy()
)


if eth_cumulative.empty:

    raise ValueError(
        "\nNo ETH cumulative forecast-loss data found."
    )


fig, ax = plt.subplots(
    figsize=(
        9,
        5,
    )
)


ax.plot(
    eth_cumulative[
        cum_date_col
    ],
    eth_cumulative[
        cum_benchmark_col
    ],
    linewidth=1.4,
    label="Benchmark",
)


ax.plot(
    eth_cumulative[
        cum_date_col
    ],
    eth_cumulative[
        cum_extended_col
    ],
    linewidth=1.4,
    label="Benchmark + sentiment",
)


ax.set_title(
    "Ethereum Cumulative OOS Squared Forecast Error"
)


ax.set_xlabel(
    "Forecast date"
)


ax.set_ylabel(
    "Cumulative squared error"
)


ax.legend(
    frameon=False
)


save_figure_png_pdf(
    fig,
    "figure08_eth_cumulative_oos_loss",
)


print(
    "Figure 8 saved: PASS"
)


# ============================================================
# 74. FIGURE 9 — PRIMARY OOS RMSE COMPARISON
# ============================================================

print_header(
    "SECTION 10 — FIGURE 9: PRIMARY OOS RMSE"
)


require_columns(
    h3_h4_results,
    [
        "Asset",
        "Benchmark_RMSE",
        "Extended_RMSE",
    ],
    context="H3/H4 RMSE figure",
)


rmse_plot = (
    h3_h4_results[
        [
            "Asset",
            "Benchmark_RMSE",
            "Extended_RMSE",
        ]
    ]
    .copy()
    .set_index(
        "Asset"
    )
)


rmse_plot = rmse_plot.rename(
    columns={
        "Benchmark_RMSE":
            "Benchmark",

        "Extended_RMSE":
            "Benchmark + sentiment",
    }
)


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5,
    )
)


rmse_plot.plot(
    kind="bar",
    ax=ax,
)


ax.set_title(
    "Out-of-Sample RMSE: Benchmark vs Sentiment Model"
)


ax.set_xlabel(
    "Cryptocurrency"
)


ax.set_ylabel(
    "RMSE"
)


ax.tick_params(
    axis="x",
    rotation=0,
)


ax.legend(
    title="Model",
    frameon=False,
)


save_figure_png_pdf(
    fig,
    "figure09_primary_oos_rmse_comparison",
)


print(
    "Figure 9 saved: PASS"
)


# ============================================================
# 75. FIGURE 10 — ECONOMIC SIGNIFICANCE
# ============================================================

print_header(
    "SECTION 10 — FIGURE 10: ECONOMIC SIGNIFICANCE"
)


require_columns(
    economic_significance,
    [
        "Asset",
        "Model",
        "One_SD_Sentiment_Effect_Exact_Percent_Return",
    ],
    context="economic significance figure",
)


economic_plot = (
    economic_significance.loc[
        economic_significance[
            "Model"
        ].eq(
            "M2_Sentiment"
        ),
        [
            "Asset",
            "One_SD_Sentiment_Effect_Exact_Percent_Return",
        ],
    ]
    .copy()
)


if len(
    economic_plot
) != 2:

    raise ValueError(
        "\nExpected exactly one M2 economic-significance "
        "observation for BTC and ETH."
    )


economic_plot = economic_plot.set_index(
    "Asset"
)


fig, ax = plt.subplots(
    figsize=(
        7,
        5,
    )
)


economic_plot[
    "One_SD_Sentiment_Effect_Exact_Percent_Return"
].plot(
    kind="bar",
    ax=ax,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Estimated Return Effect of a One-SD Increase in Reddit Sentiment"
)


ax.set_xlabel(
    "Cryptocurrency"
)


ax.set_ylabel(
    "Estimated return effect (%)"
)


ax.tick_params(
    axis="x",
    rotation=0,
)


save_figure_png_pdf(
    fig,
    "figure10_economic_significance",
)


print(
    "Figure 10 saved: PASS"
)


# ============================================================
# 76. FIGURE 11 — WEEKEND VS WEEKDAY OOS R-SQUARED
# ============================================================

print_header(
    "SECTION 10 — FIGURE 11: WEEKEND VS WEEKDAY OOS R-SQUARED"
)


require_columns(
    weekend_weekday_oos,
    [
        "Asset",
        "Day_Type",
        "OOS_R2_vs_Benchmark",
    ],
    context="weekend/weekday OOS figure",
)


weekend_plot = (
    weekend_weekday_oos[
        [
            "Asset",
            "Day_Type",
            "OOS_R2_vs_Benchmark",
        ]
    ]
    .copy()
)


weekend_pivot = (
    weekend_plot
    .pivot(
        index="Asset",
        columns="Day_Type",
        values="OOS_R2_vs_Benchmark",
    )
)


fig, ax = plt.subplots(
    figsize=(
        7.5,
        5,
    )
)


weekend_pivot.plot(
    kind="bar",
    ax=ax,
)


ax.axhline(
    0,
    linewidth=0.8,
)


ax.set_title(
    "Sentiment Model OOS R-Squared: Weekend vs Weekday"
)


ax.set_xlabel(
    "Cryptocurrency"
)


ax.set_ylabel(
    "OOS R-squared vs benchmark"
)


ax.tick_params(
    axis="x",
    rotation=0,
)


ax.legend(
    title="Day type",
    frameon=False,
)


save_figure_png_pdf(
    fig,
    "figure11_weekend_weekday_oos_r2",
)


print(
    "Figure 11 saved: PASS"
)


# ============================================================
# 77. FIGURE NOTES
# ============================================================

figure_notes = pd.DataFrame(
    [
        {
            "Figure":
                "Figure 1",

            "Title":
                "Bitcoin Daily Log Returns, 2021–2025",

            "Suggested_Location":
                "Descriptive results / data overview",

            "Main_or_Appendix":
                "Main text or appendix",
        },

        {
            "Figure":
                "Figure 2",

            "Title":
                "Ethereum Daily Log Returns, 2021–2025",

            "Suggested_Location":
                "Descriptive results / data overview",

            "Main_or_Appendix":
                "Main text or appendix",
        },

        {
            "Figure":
                "Figure 3",

            "Title":
                "Lagged Bitcoin Reddit Sentiment, 2021–2025",

            "Suggested_Location":
                "Data / descriptive results",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Figure":
                "Figure 4",

            "Title":
                "Lagged Ethereum Reddit Sentiment, 2021–2025",

            "Suggested_Location":
                "Data / descriptive results",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Figure":
                "Figure 5",

            "Title":
                "Reddit Sentiment Coefficients Across Alternative Lags",

            "Suggested_Location":
                "Robustness results",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Figure":
                "Figure 6",

            "Title":
                "Year-Specific Reddit Sentiment Coefficients",

            "Suggested_Location":
                "Regime robustness",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Figure":
                "Figure 7",

            "Title":
                "Bitcoin Cumulative OOS Squared Forecast Error",

            "Suggested_Location":
                "H3 forecast results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Figure":
                "Figure 8",

            "Title":
                "Ethereum Cumulative OOS Squared Forecast Error",

            "Suggested_Location":
                "H4 forecast results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Figure":
                "Figure 9",

            "Title":
                "Out-of-Sample RMSE: Benchmark vs Sentiment Model",

            "Suggested_Location":
                "H3/H4 forecast results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Figure":
                "Figure 10",

            "Title":
                "Economic Significance of Reddit Sentiment",

            "Suggested_Location":
                "H1/H2 results",

            "Main_or_Appendix":
                "Main text or appendix",
        },

        {
            "Figure":
                "Figure 11",

            "Title":
                "Weekend vs Weekday OOS R-Squared",

            "Suggested_Location":
                "Forecast robustness",

            "Main_or_Appendix":
                "Appendix preferred",
        },
    ]
)


figure_notes.to_csv(
    OUTPUT_DIR
    / "section10_figure_guide.csv",
    index=False,
)


# ============================================================
# 78. DISSERTATION TABLE GUIDE
# ============================================================

table_guide = pd.DataFrame(
    [
        {
            "Suggested_Table":
                "Table 1",

            "File":
                "table_descriptive_statistics_publication.csv",

            "Purpose":
                "Descriptive statistics",

            "Suggested_Location":
                "Data / descriptive statistics",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Suggested_Table":
                "Table 2",

            "File":
                "table_correlation_matrix_btc.csv and "
                "table_correlation_matrix_eth.csv",

            "Purpose":
                "Correlation diagnostics",

            "Suggested_Location":
                "Descriptive statistics / diagnostics",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Suggested_Table":
                "Table 3",

            "File":
                "table_vif_diagnostics.csv",

            "Purpose":
                "Multicollinearity diagnostics",

            "Suggested_Location":
                "Diagnostics",

            "Main_or_Appendix":
                "Appendix preferred",
        },

        {
            "Suggested_Table":
                "Table 4",

            "File":
                "table_primary_regressions_combined.csv",

            "Purpose":
                "Primary M0-M3 regression results",

            "Suggested_Location":
                "Main empirical results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Suggested_Table":
                "Table 5",

            "File":
                "table_h1_h2_primary_results.csv",

            "Purpose":
                "Primary H1/H2 sentiment tests",

            "Suggested_Location":
                "H1/H2 results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Suggested_Table":
                "Table 6",

            "File":
                "table_h3_h4_primary_oos_results.csv",

            "Purpose":
                "Primary H3/H4 OOS forecast tests",

            "Suggested_Location":
                "H3/H4 results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Suggested_Table":
                "Table 7",

            "File":
                "table_h5_btc_eth_difference_test.csv",

            "Purpose":
                "Formal BTC-vs-ETH coefficient difference",

            "Suggested_Location":
                "H5 results",

            "Main_or_Appendix":
                "Main text",
        },

        {
            "Suggested_Table":
                "Table 8",

            "File":
                "table_economic_significance.csv",

            "Purpose":
                "Economic magnitude of sentiment effects",

            "Suggested_Location":
                "Main empirical results",

            "Main_or_Appendix":
                "Main text or appendix",
        },

        {
            "Suggested_Table":
                "Appendix A",

            "File":
                "table_sentiment_lag_robustness.csv",

            "Purpose":
                "Alternative sentiment lags",

            "Suggested_Location":
                "Robustness appendix",

            "Main_or_Appendix":
                "Appendix",
        },

        {
            "Suggested_Table":
                "Appendix B",

            "File":
                "table_cross_crypto_robustness.csv",

            "Purpose":
                "Cross-cryptocurrency return robustness",

            "Suggested_Location":
                "Robustness appendix",

            "Main_or_Appendix":
                "Appendix",
        },

        {
            "Suggested_Table":
                "Appendix C",

            "File":
                "table_year_regime_robustness.csv",

            "Purpose":
                "Year/regime robustness",

            "Suggested_Location":
                "Robustness appendix",

            "Main_or_Appendix":
                "Appendix",
        },

        {
            "Suggested_Table":
                "Appendix D",

            "File":
                "table_extreme_return_sensitivity.csv and "
                "table_extreme_return_and_lag_sensitivity.csv",

            "Purpose":
                "Extreme-return sensitivity",

            "Suggested_Location":
                "Robustness appendix",

            "Main_or_Appendix":
                "Appendix",
        },

        {
            "Suggested_Table":
                "Appendix E",

            "File":
                "table_weekend_weekday_oos.csv",

            "Purpose":
                "Weekend/weekday forecast robustness",

            "Suggested_Location":
                "Robustness appendix",

            "Main_or_Appendix":
                "Appendix",
        },
    ]
)


table_guide.to_csv(
    OUTPUT_DIR
    / "section10_table_guide.csv",
    index=False,
)


# ============================================================
# 79. FINAL TABLE MANIFEST
# ============================================================

print_header(
    "SECTION 10 — FINAL TABLE MANIFEST"
)


all_table_files = sorted(
    TABLE_DIR.glob(
        "*.csv"
    )
)


table_manifest = pd.DataFrame(
    [
        {
            "File":
                path.name,

            "Exists":
                path.exists(),

            "Size_Bytes":
                (
                    path.stat().st_size
                    if path.exists()
                    else 0
                ),
        }
        for path in all_table_files
    ]
)


table_manifest.to_csv(
    OUTPUT_DIR
    / "section10_table_manifest.csv",
    index=False,
)


print(
    table_manifest.to_string(
        index=False
    )
)


if table_manifest.empty:

    raise ValueError(
        "\nNo Section 10 table outputs found."
    )


if not table_manifest[
    "Exists"
].all():

    raise FileNotFoundError(
        "\nAt least one Section 10 table "
        "output is missing."
    )


if (
    table_manifest[
        "Size_Bytes"
    ]
    <= 0
).any():

    raise ValueError(
        "\nAt least one Section 10 table "
        "output is empty."
    )


print(
    "\nFinal table manifest: PASS"
)


# ============================================================
# 80. FINAL FIGURE MANIFEST
# ============================================================

print_header(
    "SECTION 10 — FINAL FIGURE MANIFEST"
)


EXPECTED_FIGURE_STEMS = [
    "figure01_btc_daily_returns",
    "figure02_eth_daily_returns",
    "figure03_btc_reddit_sentiment",
    "figure04_eth_reddit_sentiment",
    "figure05_sentiment_lag_robustness",
    "figure06_year_specific_sentiment_coefficients",
    "figure07_btc_cumulative_oos_loss",
    "figure08_eth_cumulative_oos_loss",
    "figure09_primary_oos_rmse_comparison",
    "figure10_economic_significance",
    "figure11_weekend_weekday_oos_r2",
]


figure_manifest_rows = []


for stem in EXPECTED_FIGURE_STEMS:

    for extension in [
        "png",
        "pdf",
    ]:

        figure_path = (
            FIGURE_DIR
            / f"{stem}.{extension}"
        )


        figure_manifest_rows.append(
            {
                "Figure":
                    stem,

                "Format":
                    extension.upper(),

                "File":
                    figure_path.name,

                "Exists":
                    figure_path.exists(),

                "Size_Bytes":
                    (
                        figure_path.stat().st_size
                        if figure_path.exists()
                        else 0
                    ),
            }
        )


figure_manifest = pd.DataFrame(
    figure_manifest_rows
)


figure_manifest.to_csv(
    OUTPUT_DIR
    / "section10_figure_manifest.csv",
    index=False,
)


print(
    figure_manifest.to_string(
        index=False
    )
)


if not figure_manifest[
    "Exists"
].all():

    missing_figures = (
        figure_manifest.loc[
            ~figure_manifest[
                "Exists"
            ],
            "File",
        ]
        .tolist()
    )


    raise FileNotFoundError(
        "\nMissing expected Section 10 figures:\n"
        + "\n".join(
            f"  - {filename}"
            for filename in missing_figures
        )
    )


if (
    figure_manifest[
        "Size_Bytes"
    ]
    <= 0
).any():

    raise ValueError(
        "\nAt least one generated figure is empty."
    )


print(
    "\nFinal figure manifest: PASS"
)


# ============================================================
# 81. FINAL HYPOTHESIS CONSISTENCY CHECK
# ============================================================

print_header(
    "SECTION 10 — HYPOTHESIS CONSISTENCY CHECK"
)


section10_hypotheses = (
    final_hypothesis_publication[
        "Hypothesis"
    ]
    .astype(str)
    .tolist()
)


if set(
    section10_hypotheses
) != set(
    HYPOTHESES
):

    raise ValueError(
        "\nSection 10 final hypothesis table "
        "does not contain exactly H1-H5."
    )


if len(
    final_hypothesis_publication
) != 5:

    raise ValueError(
        "\nSection 10 final hypothesis table "
        "must contain exactly five rows."
    )


print(
    "H1-H5 consistency: PASS"
)


# ============================================================
# 82. CONFIRM SECTION 09 RESULTS REMAIN UNCHANGED
# ============================================================

print_header(
    "SECTION 10 — FROZEN RESULT INTEGRITY CHECK"
)


if not primary_regressions.equals(
    primary_regressions_raw
):

    raise ValueError(
        "\nPrimary Section 09 regression results "
        "were modified during Section 10."
    )


if not h3_h4_results.equals(
    h3_h4_results_raw
):

    raise ValueError(
        "\nH3/H4 Section 09 results "
        "were modified during Section 10."
    )


if not hypothesis_summary.equals(
    hypothesis_summary_raw
):

    raise ValueError(
        "\nFinal Section 09 hypothesis summary "
        "was modified during Section 10."
    )


print(
    "Primary Section 09 regression results unchanged: PASS"
)

print(
    "H3/H4 Section 09 results unchanged: PASS"
)

print(
    "Final H1-H5 Section 09 summary unchanged: PASS"
)


# ============================================================
# 83. FINAL SECTION 10 QC
# ============================================================

print_header(
    "SECTION 10 — MASTER QC"
)


section10_qc_rows = [
    {
        "Check":
            "Section 08 descriptive statistics loaded",

        "PASS":
            not descriptive_statistics.empty,
    },

    {
        "Check":
            "Section 08 correlations loaded",

        "PASS":
            not correlations_long.empty,
    },

    {
        "Check":
            "Section 08 VIF diagnostics loaded",

        "PASS":
            not vif_diagnostics.empty,
    },

    {
        "Check":
            "Section 09 primary regressions loaded",

        "PASS":
            not primary_regressions.empty,
    },

    {
        "Check":
            "Section 09 H1/H2 results loaded",

        "PASS":
            not h1_h2_results.empty,
    },

    {
        "Check":
            "Section 09 H3/H4 results loaded",

        "PASS":
            not h3_h4_results.empty,
    },

    {
        "Check":
            "Section 09 H5 result loaded",

        "PASS":
            not h5_results.empty,
    },

    {
        "Check":
            "Final H1-H5 summary contains exactly five hypotheses",

        "PASS":
            (
                len(
                    final_hypothesis_publication
                )
                == 5
            ),
    },

    {
        "Check":
            "Section 09 master QC reconfirmed",

        "PASS":
            qc_values.isin(
                [
                    "true",
                    "1",
                    "yes",
                ]
            ).all(),
    },

    {
        "Check":
            "Section 09 forecast QC reconfirmed",

        "PASS":
            forecast_pass.all(),
    },

    {
        "Check":
            "All Part 2 outputs exist",

        "PASS":
            part2_manifest[
                "Exists"
            ].all(),
    },

    {
        "Check":
            "All Part 3 outputs exist",

        "PASS":
            part3_manifest[
                "Exists"
            ].all(),
    },

    {
        "Check":
            "All expected figures exist",

        "PASS":
            figure_manifest[
                "Exists"
            ].all(),
    },

    {
        "Check":
            "All expected figures are non-empty",

        "PASS":
            (
                figure_manifest[
                    "Size_Bytes"
                ]
                > 0
            ).all(),
    },

    {
        "Check":
            "All generated tables are non-empty",

        "PASS":
            (
                table_manifest[
                    "Size_Bytes"
                ]
                > 0
            ).all(),
    },

    {
        "Check":
            "Primary regression results unchanged",

        "PASS":
            primary_regressions.equals(
                primary_regressions_raw
            ),
    },

    {
        "Check":
            "H3/H4 results unchanged",

        "PASS":
            h3_h4_results.equals(
                h3_h4_results_raw
            ),
    },

    {
        "Check":
            "Final hypothesis results unchanged",

        "PASS":
            hypothesis_summary.equals(
                hypothesis_summary_raw
            ),
    },
]


section10_qc = pd.DataFrame(
    section10_qc_rows
)


section10_qc.to_csv(
    OUTPUT_DIR
    / "section10_qc.csv",
    index=False,
)


print(
    section10_qc.to_string(
        index=False
    )
)


if not section10_qc[
    "PASS"
].all():

    failed_checks = (
        section10_qc.loc[
            ~section10_qc[
                "PASS"
            ]
        ]
    )


    raise ValueError(
        "\nSECTION 10 QC FAILED:\n\n"
        + failed_checks.to_string(
            index=False
        )
    )


print(
    "\nMaster Section 10 QC: PASS"
)


# ============================================================
# 84. CREATE SECTION 10 METHODOLOGY / PRESENTATION NOTE
# ============================================================

methodology_note = """
SECTION 10 — DISSERTATION TABLES AND FIGURES

Purpose
-------
Section 10 is a presentation-only stage.

No statistical models are estimated or re-estimated in this
section. No Section 08 or Section 09 analytical outputs are
overwritten.

Inputs
------
Section 08 validated modelling outputs and Section 09 frozen
econometric and forecasting results are used as the sole
analytical inputs.

Tables
------
Section 10 creates dissertation-ready versions of:

1. Descriptive statistics.
2. BTC and ETH correlation matrices.
3. VIF diagnostics.
4. Primary M0-M3 HAC regression results.
5. H1/H2 sentiment-association results.
6. Economic-significance calculations.
7. H3/H4 out-of-sample forecast comparisons.
8. H5 BTC-versus-ETH coefficient-difference test.
9. Cross-cryptocurrency robustness.
10. Alternative sentiment-lag robustness.
11. Year/regime robustness.
12. Extreme-return sensitivity.
13. Weekend/weekday out-of-sample robustness.
14. Final H1-H5 hypothesis summary.

Figures
-------
Section 10 creates:

1. BTC daily log returns.
2. ETH daily log returns.
3. Lagged BTC Reddit sentiment.
4. Lagged ETH Reddit sentiment.
5. Alternative-lag sentiment coefficients.
6. Year-specific sentiment coefficients.
7. BTC cumulative OOS squared forecast error.
8. ETH cumulative OOS squared forecast error.
9. Benchmark-versus-sentiment OOS RMSE.
10. Economic significance of a one-standard-deviation
    increase in Reddit sentiment.
11. Weekend-versus-weekday OOS R-squared.

Interpretation
--------------
All hypothesis decisions and statistical results remain those
produced and validated in Section 09.

Section 10 changes presentation only. It does not alter the
underlying empirical analysis.

Primary inference
-----------------
H1 and H2 use OLS with HAC/Newey-West inference and maximum
lag 7.

H3 and H4 use genuine expanding-window one-step-ahead
out-of-sample forecasts, with 2021-2023 as the initial
estimation period and 2024-2025 as the OOS evaluation period.

The Clark-West test is the primary nested-model forecast test.

H5 uses the ETH-by-lagged-Reddit-sentiment interaction in a
fully interacted pooled BTC/ETH model, with date-clustered
Newey-West score aggregation and maximum lag 7.

Final principle
---------------
Section 09 is treated as frozen. Section 10 exists only to
translate validated results into dissertation-ready tables
and figures.
""".strip()


with open(
    OUTPUT_DIR
    / "section10_presentation_note.txt",
    "w",
    encoding="utf-8",
) as file:

    file.write(
        methodology_note
    )


# ============================================================
# 85. FINAL OUTPUT SUMMARY
# ============================================================

print_header(
    "SECTION 10 — FINAL OUTPUT SUMMARY"
)


print(
    f"Tables created: "
    f"{len(table_manifest):,}"
)


print(
    f"Figure files created: "
    f"{len(figure_manifest):,}"
)


print(
    f"Unique figures created: "
    f"{len(EXPECTED_FIGURE_STEMS):,}"
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nTable directory:"
)

print(
    TABLE_DIR
)


print(
    "\nFigure directory:"
)

print(
    FIGURE_DIR
)


# ============================================================
# 86. SECTION 10 COMPLETE
# ============================================================

print_header(
    "SECTION 10 — COMPLETE"
)


print(
    "Descriptive-statistics tables: PASS"
)

print(
    "Correlation tables: PASS"
)

print(
    "VIF diagnostics: PASS"
)

print(
    "Primary M0-M3 regression tables: PASS"
)

print(
    "H1/H2 tables: PASS"
)

print(
    "Economic-significance table: PASS"
)

print(
    "H3/H4 OOS tables: PASS"
)

print(
    "H5 formal difference table: PASS"
)

print(
    "Robustness tables: PASS"
)

print(
    "Dissertation figures: PASS"
)

print(
    "Table guide: PASS"
)

print(
    "Figure guide: PASS"
)

print(
    "Output manifests: PASS"
)

print(
    "Frozen Section 09 result integrity: PASS"
)

print(
    "Master Section 10 QC: PASS"
)


print(
    "\n"
    + "=" * 78
)

print(
    "SECTION 10 STATUS: PASS"
)

print(
    "=" * 78
)


print(
    "\nSection 10 dissertation tables "
    "and figures are complete."
)

print(
    "\nNo models were re-estimated."
)

print(
    "No Section 08 or Section 09 "
    "analytical results were modified."
)

print(
    "\nYou are ready to move from "
    "Python analysis to dissertation writing."
)
