# ======================================================================
# 11_descriptive_statistics_multicollinearity.py
#
# Crypto Sentiment Dissertation
#
# Purpose:
#   1. Produce descriptive statistics
#   2. Examine missingness
#   3. Produce Pearson correlation matrices
#   4. Identify high pairwise correlations
#   5. Calculate Variance Inflation Factors (VIF)
#   6. Save dissertation-ready diagnostic tables
#
# IMPORTANT:
#   This script is DIAGNOSTIC.
#
#   It does NOT:
#       - forward-fill traditional-market variables
#       - replace missing values with zero
#       - permanently drop dates from the master dataset
#       - estimate forecasting models
#
#   Complete cases are used only where mathematically required
#   for correlation/VIF calculations.
# ======================================================================

from pathlib import Path

import numpy as np
import pandas as pd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant


# ======================================================================
# SETTINGS
# ======================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

INPUT_FILE = (
    PROJECT_ROOT
    / "data_processed"
    / "forecast_structure.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data_processed"
    / "diagnostics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


print("=" * 70)
print("DESCRIPTIVE STATISTICS AND MULTICOLLINEARITY DIAGNOSTICS")
print("=" * 70)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found:\n{INPUT_FILE}"
    )


# ======================================================================
# IMPORT DATA
# ======================================================================

print("\n" + "=" * 70)
print("IMPORTING FORECAST STRUCTURE DATA")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nImported shape:")
print(df.shape)

print("\nColumns found:")
for column in df.columns:
    print(column)


# ======================================================================
# DATE VALIDATION
# ======================================================================

print("\n" + "=" * 70)
print("VALIDATING DATE VARIABLE")
print("=" * 70)

if "Date" not in df.columns:
    raise KeyError("Date column not found.")

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates:")
print(invalid_dates)

if invalid_dates > 0:
    raise ValueError(
        "Invalid dates detected. Fix these before continuing."
    )

df = (
    df.sort_values("Date")
    .reset_index(drop=True)
)

duplicate_dates = df["Date"].duplicated().sum()

print("\nDuplicate dates:")
print(duplicate_dates)

print("\nSample period:")
print(df["Date"].min(), "to", df["Date"].max())

print("\nNumber of observations:")
print(len(df))


# ======================================================================
# DEFINE VARIABLES
# ======================================================================

print("\n" + "=" * 70)
print("DEFINING VARIABLES")
print("=" * 70)

# --------------------------------------------------
# Dependent variables
# --------------------------------------------------

dependent_variables = [
    "BTC_Return",
    "ETH_Return",
]


# --------------------------------------------------
# Crypto activity variables
# --------------------------------------------------

activity_variables = [
    "Log_BTC_Volume",
    "Lagged_Log_BTC_Volume",
    "Log_ETH_Volume",
    "Lagged_Log_ETH_Volume",
]


# --------------------------------------------------
# Crypto return variables
# --------------------------------------------------

crypto_variables = [
    "BTC_Return",
    "BTC_Lagged_Return",
    "ETH_Return",
    "ETH_Lagged_Return",
]


# --------------------------------------------------
# Traditional-market transformed variables
# --------------------------------------------------

market_variables = [
    "SP500_Return",
    "Lagged_SP500_Return",
    "VIX_Change",
    "Lagged_VIX_Change",
    "Gold_Return",
    "Lagged_Gold_Return",
    "DXY_Return",
    "Lagged_DXY_Return",
    "US10Y_Change",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------
# Main descriptive-statistics variables
# --------------------------------------------------

descriptive_variables = [
    "BTC_Return",
    "ETH_Return",

    "BTC_Lagged_Return",
    "ETH_Lagged_Return",

    "Lagged_Log_BTC_Volume",
    "Lagged_Log_ETH_Volume",

    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------
# BTC benchmark predictors
# --------------------------------------------------

btc_benchmark_predictors = [
    "BTC_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------
# BTC robustness predictors
# Add lagged ETH return
# --------------------------------------------------

btc_robustness_predictors = [
    "BTC_Lagged_Return",
    "ETH_Lagged_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------
# ETH benchmark predictors
# --------------------------------------------------

eth_benchmark_predictors = [
    "ETH_Lagged_Return",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# --------------------------------------------------
# ETH robustness predictors
# Add lagged BTC return
# --------------------------------------------------

eth_robustness_predictors = [
    "ETH_Lagged_Return",
    "BTC_Lagged_Return",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


# ======================================================================
# CHECK REQUIRED VARIABLES
# ======================================================================

required_variables = sorted(
    set(
        descriptive_variables
        + btc_robustness_predictors
        + eth_robustness_predictors
    )
)

missing_columns = [
    column
    for column in required_variables
    if column not in df.columns
]

if missing_columns:
    print("\nERROR: Missing required columns:")
    for column in missing_columns:
        print(column)

    raise KeyError(
        "Required variables are missing from the dataset."
    )

print("\nAll required diagnostic variables are available.")


# ======================================================================
# FORCE VARIABLES TO NUMERIC
# ======================================================================

for column in required_variables:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ======================================================================
# MISSING-VALUE DIAGNOSTICS
# ======================================================================

print("\n" + "=" * 70)
print("MISSING-VALUE DIAGNOSTICS")
print("=" * 70)

missing_table = pd.DataFrame({
    "Variable": required_variables,
    "N_Total": len(df),
    "N_Missing": [
        df[column].isna().sum()
        for column in required_variables
    ]
})

missing_table["Percent_Missing"] = (
    100
    * missing_table["N_Missing"]
    / missing_table["N_Total"]
)

missing_table["N_Available"] = (
    missing_table["N_Total"]
    - missing_table["N_Missing"]
)

missing_table = missing_table[
    [
        "Variable",
        "N_Total",
        "N_Available",
        "N_Missing",
        "Percent_Missing",
    ]
]

print("\nMissing-value table:")
print(
    missing_table.to_string(
        index=False
    )
)

missing_output = (
    OUTPUT_DIR
    / "missing_value_diagnostics.csv"
)

missing_table.to_csv(
    missing_output,
    index=False
)


# ======================================================================
# DESCRIPTIVE STATISTICS FUNCTION
# ======================================================================

def calculate_descriptive_statistics(data, variables):

    rows = []

    for variable in variables:

        series = data[variable].dropna()

        if len(series) == 0:
            continue

        row = {
            "Variable": variable,
            "N": series.count(),
            "Mean": series.mean(),
            "Std_Dev": series.std(),
            "Min": series.min(),
            "P25": series.quantile(0.25),
            "Median": series.median(),
            "P75": series.quantile(0.75),
            "Max": series.max(),
            "Skewness": series.skew(),
            "Kurtosis": series.kurt(),
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================
# FULL-SAMPLE DESCRIPTIVE STATISTICS
# ======================================================================

print("\n" + "=" * 70)
print("FULL-SAMPLE DESCRIPTIVE STATISTICS")
print("=" * 70)

descriptive_stats = calculate_descriptive_statistics(
    df,
    descriptive_variables
)

print("\nDescriptive statistics:")
print(
    descriptive_stats.to_string(
        index=False,
        float_format=lambda x: f"{x:.6f}"
    )
)

descriptive_output = (
    OUTPUT_DIR
    / "descriptive_statistics_full_sample.csv"
)

descriptive_stats.to_csv(
    descriptive_output,
    index=False
)


# ======================================================================
# OPTIONAL: INITIAL ESTIMATION VS OUT-OF-SAMPLE DESCRIPTIVES
# ======================================================================

if "Forecast_Sample" in df.columns:

    print("\n" + "=" * 70)
    print("DESCRIPTIVE STATISTICS BY FORECAST SAMPLE")
    print("=" * 70)

    initial_df = df[
        df["Forecast_Sample"] == "Initial_Estimation"
    ].copy()

    oos_df = df[
        df["Forecast_Sample"] == "Out_of_Sample"
    ].copy()

    initial_stats = calculate_descriptive_statistics(
        initial_df,
        descriptive_variables
    )

    initial_stats.insert(
        0,
        "Sample",
        "Initial_Estimation"
    )

    oos_stats = calculate_descriptive_statistics(
        oos_df,
        descriptive_variables
    )

    oos_stats.insert(
        0,
        "Sample",
        "Out_of_Sample"
    )

    sample_stats = pd.concat(
        [
            initial_stats,
            oos_stats
        ],
        ignore_index=True
    )

    print("\nInitial estimation observations:")
    print(len(initial_df))

    print("\nOut-of-sample observations:")
    print(len(oos_df))

    sample_stats.to_csv(
        OUTPUT_DIR
        / "descriptive_statistics_by_forecast_sample.csv",
        index=False
    )


# ======================================================================
# CORRELATION FUNCTION
# ======================================================================

def calculate_correlation_matrix(
    data,
    variables,
    name
):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    # Pearson correlations using available pairs.
    correlation_matrix = (
        data[variables]
        .corr(method="pearson")
    )

    print("\nPearson correlation matrix:")
    print(
        correlation_matrix.round(4).to_string()
    )

    return correlation_matrix


# ======================================================================
# BTC BENCHMARK CORRELATION MATRIX
# ======================================================================

btc_corr = calculate_correlation_matrix(
    df,
    btc_benchmark_predictors,
    "BTC BENCHMARK PREDICTOR CORRELATION MATRIX"
)

btc_corr.to_csv(
    OUTPUT_DIR
    / "btc_benchmark_correlation_matrix.csv"
)


# ======================================================================
# ETH BENCHMARK CORRELATION MATRIX
# ======================================================================

eth_corr = calculate_correlation_matrix(
    df,
    eth_benchmark_predictors,
    "ETH BENCHMARK PREDICTOR CORRELATION MATRIX"
)

eth_corr.to_csv(
    OUTPUT_DIR
    / "eth_benchmark_correlation_matrix.csv"
)


# ======================================================================
# CROSS-CRYPTO ROBUSTNESS CORRELATIONS
# ======================================================================

btc_robust_corr = calculate_correlation_matrix(
    df,
    btc_robustness_predictors,
    "BTC ROBUSTNESS PREDICTOR CORRELATION MATRIX"
)

btc_robust_corr.to_csv(
    OUTPUT_DIR
    / "btc_robustness_correlation_matrix.csv"
)


eth_robust_corr = calculate_correlation_matrix(
    df,
    eth_robustness_predictors,
    "ETH ROBUSTNESS PREDICTOR CORRELATION MATRIX"
)

eth_robust_corr.to_csv(
    OUTPUT_DIR
    / "eth_robustness_correlation_matrix.csv"
)


# ======================================================================
# IDENTIFY HIGH PAIRWISE CORRELATIONS
# ======================================================================

def identify_high_correlations(
    correlation_matrix,
    threshold=0.70
):

    high_correlations = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):

        for j in range(i + 1, len(columns)):

            variable_1 = columns[i]
            variable_2 = columns[j]

            correlation = correlation_matrix.iloc[i, j]

            if pd.isna(correlation):
                continue

            if abs(correlation) >= threshold:

                high_correlations.append({
                    "Variable_1": variable_1,
                    "Variable_2": variable_2,
                    "Correlation": correlation,
                    "Absolute_Correlation": abs(correlation),
                })

    result = pd.DataFrame(high_correlations)

    if not result.empty:

        result = result.sort_values(
            "Absolute_Correlation",
            ascending=False
        )

    return result


print("\n" + "=" * 70)
print("HIGH PAIRWISE CORRELATIONS")
print("=" * 70)

btc_high_corr = identify_high_correlations(
    btc_robust_corr,
    threshold=0.70
)

eth_high_corr = identify_high_correlations(
    eth_robust_corr,
    threshold=0.70
)


print("\nBTC model correlations with |r| >= 0.70:")

if btc_high_corr.empty:
    print("None detected.")
else:
    print(
        btc_high_corr.to_string(
            index=False
        )
    )


print("\nETH model correlations with |r| >= 0.70:")

if eth_high_corr.empty:
    print("None detected.")
else:
    print(
        eth_high_corr.to_string(
            index=False
        )
    )


btc_high_corr.to_csv(
    OUTPUT_DIR
    / "btc_high_correlations.csv",
    index=False
)

eth_high_corr.to_csv(
    OUTPUT_DIR
    / "eth_high_correlations.csv",
    index=False
)


# ======================================================================
# VIF FUNCTION
# ======================================================================

def calculate_vif(
    data,
    predictors,
    model_name
):

    print("\n" + "=" * 70)
    print(f"VIF: {model_name}")
    print("=" * 70)

    # VIF requires the SAME observations for every predictor.
    # Therefore complete cases are used locally for this calculation.
    #
    # This does NOT alter the original dataframe.

    vif_data = (
        data[predictors]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )

    print("\nPredictors:")
    for variable in predictors:
        print(variable)

    print("\nComplete observations used for VIF:")
    print(len(vif_data))

    print("\nObservations excluded locally because of missing predictors:")
    print(len(data) - len(vif_data))

    if len(vif_data) == 0:
        raise ValueError(
            f"No complete observations available for {model_name}."
        )

    # Remove zero-variance variables if any.
    zero_variance = [
        column
        for column in predictors
        if vif_data[column].nunique() <= 1
    ]

    if zero_variance:

        print("\nWARNING: Zero-variance variables detected:")

        for column in zero_variance:
            print(column)

        vif_data = vif_data.drop(
            columns=zero_variance
        )

    X = add_constant(
        vif_data,
        has_constant="add"
    )

    vif_results = []

    # Do not report VIF for the intercept.
    for i, column in enumerate(X.columns):

        if column == "const":
            continue

        vif_value = variance_inflation_factor(
            X.values,
            i
        )

        vif_results.append({
            "Variable": column,
            "VIF": vif_value,
        })

    vif_table = pd.DataFrame(vif_results)

    vif_table = vif_table.sort_values(
        "VIF",
        ascending=False
    ).reset_index(drop=True)

    # Simple diagnostic categories.
    def classify_vif(value):

        if value < 5:
            return "Low / acceptable"

        elif value < 10:
            return "Potential concern"

        else:
            return "High multicollinearity"

    vif_table["Diagnostic"] = (
        vif_table["VIF"]
        .apply(classify_vif)
    )

    print("\nVIF results:")
    print(
        vif_table.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    return vif_table, len(vif_data)


# ======================================================================
# BTC BENCHMARK VIF
# ======================================================================

btc_benchmark_vif, btc_benchmark_n = calculate_vif(
    df,
    btc_benchmark_predictors,
    "BTC BENCHMARK MODEL"
)

btc_benchmark_vif.to_csv(
    OUTPUT_DIR
    / "btc_benchmark_vif.csv",
    index=False
)


# ======================================================================
# ETH BENCHMARK VIF
# ======================================================================

eth_benchmark_vif, eth_benchmark_n = calculate_vif(
    df,
    eth_benchmark_predictors,
    "ETH BENCHMARK MODEL"
)

eth_benchmark_vif.to_csv(
    OUTPUT_DIR
    / "eth_benchmark_vif.csv",
    index=False
)


# ======================================================================
# BTC CROSS-CRYPTO ROBUSTNESS VIF
# ======================================================================

btc_robust_vif, btc_robust_n = calculate_vif(
    df,
    btc_robustness_predictors,
    "BTC + LAGGED ETH RETURN ROBUSTNESS MODEL"
)

btc_robust_vif.to_csv(
    OUTPUT_DIR
    / "btc_cross_crypto_robustness_vif.csv",
    index=False
)


# ======================================================================
# ETH CROSS-CRYPTO ROBUSTNESS VIF
# ======================================================================

eth_robust_vif, eth_robust_n = calculate_vif(
    df,
    eth_robustness_predictors,
    "ETH + LAGGED BTC RETURN ROBUSTNESS MODEL"
)

eth_robust_vif.to_csv(
    OUTPUT_DIR
    / "eth_cross_crypto_robustness_vif.csv",
    index=False
)


# ======================================================================
# BTC-ETH RETURN CORRELATIONS
# ======================================================================

print("\n" + "=" * 70)
print("BTC-ETH RETURN CORRELATIONS")
print("=" * 70)

crypto_corr_variables = [
    "BTC_Return",
    "ETH_Return",
    "BTC_Lagged_Return",
    "ETH_Lagged_Return",
]

crypto_corr = (
    df[crypto_corr_variables]
    .corr(method="pearson")
)

print("\nCrypto return correlation matrix:")
print(
    crypto_corr.round(4).to_string()
)

crypto_corr.to_csv(
    OUTPUT_DIR
    / "btc_eth_return_correlations.csv"
)


# ======================================================================
# SUMMARY OF DIAGNOSTIC SAMPLE SIZES
# ======================================================================

print("\n" + "=" * 70)
print("DIAGNOSTIC SAMPLE SIZE SUMMARY")
print("=" * 70)

sample_size_summary = pd.DataFrame({
    "Model": [
        "BTC benchmark",
        "ETH benchmark",
        "BTC cross-crypto robustness",
        "ETH cross-crypto robustness",
    ],
    "Complete_Case_N": [
        btc_benchmark_n,
        eth_benchmark_n,
        btc_robust_n,
        eth_robust_n,
    ]
})

print(
    sample_size_summary.to_string(
        index=False
    )
)

sample_size_summary.to_csv(
    OUTPUT_DIR
    / "diagnostic_sample_sizes.csv",
    index=False
)


# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 70)
print("FINAL DIAGNOSTIC SUMMARY")
print("=" * 70)

print("\nFull sample observations:")
print(len(df))

print("\nDate range:")
print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)

print("\nDiagnostic files saved in:")
print(OUTPUT_DIR)

print("\nFiles created:")

for file in sorted(OUTPUT_DIR.glob("*.csv")):
    print(file.name)


print("\n" + "=" * 70)
print("IMPORTANT INTERPRETATION NOTES")
print("=" * 70)

print("""
1. Descriptive statistics are calculated using all available observations
   for each variable.

2. Pearson correlations are diagnostic and do not establish causality.

3. VIF calculations require a common complete-case sample. The complete-case
   restriction is applied ONLY inside the VIF calculation and does not alter
   the master/forecast dataset.

4. As a practical diagnostic guide:
       VIF < 5       = generally low/acceptable
       VIF 5 to <10  = investigate potential multicollinearity
       VIF >= 10     = potentially serious multicollinearity

   These are diagnostic conventions rather than mechanical rules for
   deleting variables.

5. A pairwise correlation threshold of |r| >= 0.70 is flagged for inspection.
   Again, this is a diagnostic flag rather than an automatic deletion rule.

6. Traditional-market missingness has NOT been solved by this script.
   Do not interpret the complete-case sample as the final forecasting sample.

7. The final correlation/VIF tables should be recalculated after:
       - the traditional-market information convention is finalized; and
       - Reddit sentiment/activity variables are added.

8. When Reddit data become available, diagnostics should include:
       Lagged sentiment
       Lagged log(1 + Reddit activity)
       and the existing controls.

9. The purpose of these diagnostics is to investigate whether predictors
   contain strongly overlapping information, particularly variables such as
   S&P 500 returns, VIX changes, DXY returns and Treasury-yield changes.
""")

print("\n" + "=" * 70)
print("DESCRIPTIVE STATISTICS AND MULTICOLLINEARITY DIAGNOSTICS COMPLETE")
print("=" * 70)