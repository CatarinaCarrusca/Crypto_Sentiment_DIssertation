# =============================================================================
# 01_baseline_explanatory_models.py
#
# BASELINE EXPLANATORY REGRESSION MODELS
#
# Purpose:
#   Estimate baseline in-sample explanatory models for daily Bitcoin (BTC)
#   and Ethereum (ETH) returns BEFORE adding Reddit sentiment/activity.
#
# IMPORTANT:
#
#   These regressions are EXPLANATORY / ASSOCIATIONAL.
#
#   They are NOT out-of-sample forecasting models.
#
#   Therefore:
#
#       - R-squared measures explanatory fit.
#       - Statistical significance measures evidence of association.
#       - Neither should be described as evidence of predictive performance.
#
#   Genuine predictive performance will be evaluated separately using
#   expanding-window out-of-sample forecasts.
#
# Main models:
#
#   BTC_Return(t)
#       <- BTC_Return(t-1)
#       <- BTC log volume(t-1)
#       <- aligned lagged S&P 500 return
#       <- aligned lagged VIX change
#       <- aligned lagged Gold return
#       <- aligned lagged DXY return
#       <- aligned lagged US10Y change
#
#   ETH_Return(t)
#       <- ETH_Return(t-1)
#       <- ETH log volume(t-1)
#       <- same traditional-market controls
#
# Robustness models:
#
#   BTC baseline + ETH_Return(t-1)
#
#   ETH baseline + BTC_Return(t-1)
#
# Inference:
#
#   OLS coefficients with HAC / Newey-West standard errors.
#
# =============================================================================


from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm


# =============================================================================
# 1. PROJECT PATHS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

PROCESSED_DIR = PROJECT_ROOT / "data_processed"

RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Use the FINAL information-aligned forecasting dataset created previously.

INPUT_FILE = (
    PROCESSED_DIR
    / "final_forecast_dataset.csv"
)


# Output files

BTC_RESULTS_FILE = (
    RESULTS_DIR
    / "baseline_btc_explanatory_results.csv"
)

ETH_RESULTS_FILE = (
    RESULTS_DIR
    / "baseline_eth_explanatory_results.csv"
)

BTC_ROBUST_RESULTS_FILE = (
    RESULTS_DIR
    / "baseline_btc_crosscrypto_robustness_results.csv"
)

ETH_ROBUST_RESULTS_FILE = (
    RESULTS_DIR
    / "baseline_eth_crosscrypto_robustness_results.csv"
)

MODEL_SUMMARY_FILE = (
    RESULTS_DIR
    / "baseline_explanatory_model_summary.csv"
)

ECONOMIC_EFFECTS_FILE = (
    RESULTS_DIR
    / "baseline_explanatory_economic_effects.csv"
)


# =============================================================================
# 2. SAMPLE SETTINGS
# =============================================================================

SAMPLE_START = pd.Timestamp(
    "2021-01-01"
)

SAMPLE_END = pd.Timestamp(
    "2025-12-31"
)


# =============================================================================
# 3. HAC / NEWEY-WEST SETTINGS
# =============================================================================
#
# Main specification:
#
#   maxlags = 7
#
# This allows the covariance estimator to be robust to short-run serial
# dependence over approximately one week of daily crypto observations.
#
# This affects inference (standard errors / p-values), NOT the OLS
# coefficient estimates themselves.
#
# We can later check alternative HAC lag lengths as a robustness exercise.
# =============================================================================

HAC_MAX_LAGS = 7


# =============================================================================
# 4. VARIABLE DEFINITIONS
# =============================================================================

BTC_DEPENDENT = "BTC_Return"

ETH_DEPENDENT = "ETH_Return"


# -----------------------------------------------------------------------------
# Traditional-market controls
# -----------------------------------------------------------------------------
#
# These are the information-aligned variables constructed previously.
#
# They represent the most recent transformed traditional-market observation
# available strictly before each crypto date.
# -----------------------------------------------------------------------------

MARKET_CONTROLS = [

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned",
]


# -----------------------------------------------------------------------------
# BTC baseline predictors
# -----------------------------------------------------------------------------

BTC_BASELINE_PREDICTORS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume",

] + MARKET_CONTROLS


# -----------------------------------------------------------------------------
# ETH baseline predictors
# -----------------------------------------------------------------------------

ETH_BASELINE_PREDICTORS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume",

] + MARKET_CONTROLS


# -----------------------------------------------------------------------------
# Cross-crypto robustness specifications
# -----------------------------------------------------------------------------

BTC_ROBUST_PREDICTORS = (

    BTC_BASELINE_PREDICTORS

    + [

        "ETH_Lagged_Return"

    ]
)


ETH_ROBUST_PREDICTORS = (

    ETH_BASELINE_PREDICTORS

    + [

        "BTC_Lagged_Return"

    ]
)


# =============================================================================
# 5. HELPER FUNCTIONS
# =============================================================================

def print_section(title):

    print("\n" + "=" * 78)

    print(title)

    print("=" * 78)


# -----------------------------------------------------------------------------


def significance_stars(p_value):

    if pd.isna(p_value):

        return ""

    if p_value < 0.01:

        return "***"

    elif p_value < 0.05:

        return "**"

    elif p_value < 0.10:

        return "*"

    else:

        return ""


# -----------------------------------------------------------------------------


def estimate_hac_ols(
    data,
    dependent,
    predictors,
    model_name,
    hac_lags=7
):

    """
    Estimate an OLS regression using HAC/Newey-West standard errors.

    Parameters
    ----------
    data : pandas.DataFrame
        Input dataset.

    dependent : str
        Dependent variable.

    predictors : list[str]
        Explanatory variables.

    model_name : str
        Name used in printed and saved output.

    hac_lags : int
        Maximum lag used in HAC covariance estimator.

    Returns
    -------
    result :
        statsmodels regression result with HAC covariance.

    regression_data :
        Complete-case sample used in the model.

    result_table :
        DataFrame containing coefficient results.
    """


    print_section(
        model_name
    )


    variables = [

        dependent

    ] + predictors


    # -------------------------------------------------------------------------
    # Complete-case model sample
    # -------------------------------------------------------------------------

    regression_data = (

        data[
            ["Date"] + variables
        ]

        .dropna()

        .copy()
    )


    print(
        "\nDependent variable:"
    )

    print(
        dependent
    )


    print(
        "\nPredictors:"
    )

    for variable in predictors:

        print(
            variable
        )


    print(
        "\nObservations used:"
    )

    print(
        len(regression_data)
    )


    print(
        "\nModel sample:"
    )

    print(
        regression_data["Date"].min(),
        "to",
        regression_data["Date"].max()
    )


    if len(regression_data) == 0:

        raise ValueError(
            f"No complete observations available for {model_name}."
        )


    # -------------------------------------------------------------------------
    # Dependent variable
    # -------------------------------------------------------------------------

    y = regression_data[
        dependent
    ].astype(float)


    # -------------------------------------------------------------------------
    # Explanatory variables
    # -------------------------------------------------------------------------

    X = regression_data[
        predictors
    ].astype(float)


    # Add regression intercept.

    X = sm.add_constant(
        X,
        has_constant="add"
    )


    # -------------------------------------------------------------------------
    # Estimate OLS with HAC / Newey-West covariance
    # -------------------------------------------------------------------------

    model = sm.OLS(
        y,
        X
    )


    result = model.fit(
        cov_type="HAC",
        cov_kwds={
            "maxlags": hac_lags
        }
    )


    # -------------------------------------------------------------------------
    # Display complete statsmodels output
    # -------------------------------------------------------------------------

    print(
        "\nRegression results:"
    )

    print(
        result.summary()
    )


    # -------------------------------------------------------------------------
    # Build clean result table
    # -------------------------------------------------------------------------

    confidence_intervals = (
        result.conf_int(
            alpha=0.05
        )
    )


    result_table = pd.DataFrame({

        "Variable":
            result.params.index,

        "Coefficient":
            result.params.values,

        "HAC_Std_Error":
            result.bse.values,

        "t_Statistic":
            result.tvalues.values,

        "p_Value":
            result.pvalues.values,

        "CI_95_Lower":
            confidence_intervals.iloc[:, 0].values,

        "CI_95_Upper":
            confidence_intervals.iloc[:, 1].values,
    })


    result_table[
        "Significance"
    ] = (

        result_table[
            "p_Value"
        ]

        .apply(
            significance_stars
        )
    )


    result_table.insert(
        0,
        "Model",
        model_name
    )


    print(
        "\nClean coefficient table:"
    )

    print(
        result_table
        .to_string(index=False)
    )


    print(
        "\nR-squared:"
    )

    print(
        result.rsquared
    )


    print(
        "\nAdjusted R-squared:"
    )

    print(
        result.rsquared_adj
    )


    print(
        "\nHAC / Newey-West maximum lags:"
    )

    print(
        hac_lags
    )


    return (
        result,
        regression_data,
        result_table
    )


# =============================================================================
# 6. START
# =============================================================================

print_section(
    "BASELINE EXPLANATORY MODELS WITHOUT REDDIT"
)


print(
    """
IMPORTANT INTERPRETATION:

These models measure IN-SAMPLE explanatory associations.

Do NOT interpret:

    statistical significance
    R-squared
    adjusted R-squared

as evidence that the model forecasts cryptocurrency returns successfully.

Predictive performance will be evaluated separately using genuine
out-of-sample expanding-window forecasts.
"""
)


# =============================================================================
# 7. CHECK INPUT FILE
# =============================================================================

print_section(
    "CHECKING INPUT FILE"
)


print(
    "\nInput file:"
)

print(
    INPUT_FILE
)


print(
    "\nDoes input file exist?"
)

print(
    INPUT_FILE.exists()
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(

        "\nFinal forecast dataset was not found:\n"
        f"{INPUT_FILE}\n\n"
        "Run the updated forecast-structure script first."
    )


# =============================================================================
# 8. IMPORT DATA
# =============================================================================

print_section(
    "IMPORTING DATA"
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    "\nImported shape:"
)

print(
    df.shape
)


print(
    "\nColumns:"
)

for column in df.columns:

    print(
        column
    )


# =============================================================================
# 9. VALIDATE DATE
# =============================================================================

print_section(
    "VALIDATING DATE"
)


if "Date" not in df.columns:

    raise KeyError(
        "Date column was not found."
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


invalid_dates = (
    df["Date"]
    .isna()
    .sum()
)


print(
    "\nInvalid dates:"
)

print(
    invalid_dates
)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates detected."
    )


# =============================================================================
# 10. SORT CHRONOLOGICALLY
# =============================================================================

df = (

    df
    .sort_values("Date")
    .reset_index(drop=True)

)


# =============================================================================
# 11. RESTRICT TO STUDY SAMPLE
# =============================================================================

print_section(
    "RESTRICTING TO STUDY SAMPLE"
)


df = df.loc[

    (df["Date"] >= SAMPLE_START)

    &

    (df["Date"] <= SAMPLE_END)

].copy()


df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print(
    "\nObservations:"
)

print(
    len(df)
)


print(
    "\nDate range:"
)

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =============================================================================
# 12. VALIDATE REQUIRED VARIABLES
# =============================================================================

print_section(
    "VALIDATING REQUIRED VARIABLES"
)


required_variables = [

    BTC_DEPENDENT,

    ETH_DEPENDENT,

] + BTC_ROBUST_PREDICTORS + ETH_ROBUST_PREDICTORS


required_variables = list(
    dict.fromkeys(
        required_variables
    )
)


missing_variables = [

    variable

    for variable in required_variables

    if variable not in df.columns
]


if missing_variables:

    print(
        "\nMissing variables:"
    )

    for variable in missing_variables:

        print(
            variable
        )

    raise KeyError(
        "Required regression variables are missing."
    )


print(
    "\nAll required variables are present."
)


# =============================================================================
# 13. CHECK MISSING VALUES
# =============================================================================

print_section(
    "MISSING VALUES IN REGRESSION VARIABLES"
)


print(

    df[
        required_variables
    ]

    .isna()

    .sum()

)


# =============================================================================
# 14. BTC BASELINE EXPLANATORY MODEL
# =============================================================================

(
    btc_result,
    btc_sample,
    btc_table

) = estimate_hac_ols(

    data=df,

    dependent=BTC_DEPENDENT,

    predictors=BTC_BASELINE_PREDICTORS,

    model_name="BTC Baseline",

    hac_lags=HAC_MAX_LAGS
)


# =============================================================================
# 15. ETH BASELINE EXPLANATORY MODEL
# =============================================================================

(
    eth_result,
    eth_sample,
    eth_table

) = estimate_hac_ols(

    data=df,

    dependent=ETH_DEPENDENT,

    predictors=ETH_BASELINE_PREDICTORS,

    model_name="ETH Baseline",

    hac_lags=HAC_MAX_LAGS
)


# =============================================================================
# 16. BTC CROSS-CRYPTO ROBUSTNESS MODEL
# =============================================================================

(
    btc_robust_result,
    btc_robust_sample,
    btc_robust_table

) = estimate_hac_ols(

    data=df,

    dependent=BTC_DEPENDENT,

    predictors=BTC_ROBUST_PREDICTORS,

    model_name="BTC + Lagged ETH Robustness",

    hac_lags=HAC_MAX_LAGS
)


# =============================================================================
# 17. ETH CROSS-CRYPTO ROBUSTNESS MODEL
# =============================================================================

(
    eth_robust_result,
    eth_robust_sample,
    eth_robust_table

) = estimate_hac_ols(

    data=df,

    dependent=ETH_DEPENDENT,

    predictors=ETH_ROBUST_PREDICTORS,

    model_name="ETH + Lagged BTC Robustness",

    hac_lags=HAC_MAX_LAGS
)


# =============================================================================
# 18. MODEL-LEVEL SUMMARY
# =============================================================================

print_section(
    "MODEL COMPARISON SUMMARY"
)


model_summary = pd.DataFrame({

    "Model": [

        "BTC Baseline",

        "BTC + Lagged ETH Robustness",

        "ETH Baseline",

        "ETH + Lagged BTC Robustness",
    ],

    "Dependent_Variable": [

        "BTC_Return",

        "BTC_Return",

        "ETH_Return",

        "ETH_Return",
    ],

    "N": [

        int(btc_result.nobs),

        int(btc_robust_result.nobs),

        int(eth_result.nobs),

        int(eth_robust_result.nobs),
    ],

    "R_Squared": [

        btc_result.rsquared,

        btc_robust_result.rsquared,

        eth_result.rsquared,

        eth_robust_result.rsquared,
    ],

    "Adjusted_R_Squared": [

        btc_result.rsquared_adj,

        btc_robust_result.rsquared_adj,

        eth_result.rsquared_adj,

        eth_robust_result.rsquared_adj,
    ],

    "HAC_Max_Lags": [

        HAC_MAX_LAGS,

        HAC_MAX_LAGS,

        HAC_MAX_LAGS,

        HAC_MAX_LAGS,
    ]
})


print(
    model_summary
    .to_string(index=False)
)


# =============================================================================
# 19. CROSS-CRYPTO COEFFICIENT CHECK
# =============================================================================

print_section(
    "CROSS-CRYPTO ROBUSTNESS RESULTS"
)


btc_cross_coefficient = (
    btc_robust_result.params[
        "ETH_Lagged_Return"
    ]
)


btc_cross_pvalue = (
    btc_robust_result.pvalues[
        "ETH_Lagged_Return"
    ]
)


eth_cross_coefficient = (
    eth_robust_result.params[
        "BTC_Lagged_Return"
    ]
)


eth_cross_pvalue = (
    eth_robust_result.pvalues[
        "BTC_Lagged_Return"
    ]
)


print(
    "\nBTC model: lagged ETH return coefficient:"
)

print(
    btc_cross_coefficient
)


print(
    "\nBTC model: lagged ETH return p-value:"
)

print(
    btc_cross_pvalue
)


print(
    "\nETH model: lagged BTC return coefficient:"
)

print(
    eth_cross_coefficient
)


print(
    "\nETH model: lagged BTC return p-value:"
)

print(
    eth_cross_pvalue
)


# =============================================================================
# 20. ECONOMIC SIGNIFICANCE
# =============================================================================
#
# Statistical significance alone is not sufficient.
#
# For each non-constant predictor we calculate:
#
#     beta × standard deviation of predictor
#
# This gives the change in the fitted daily crypto return associated with
# a one-standard-deviation change in the predictor.
#
# We report this in:
#
#     return units
#     percentage points
#     basis points
#
# =============================================================================

print_section(
    "ECONOMIC SIGNIFICANCE"
)


economic_effect_records = []


models_for_effects = [

    (
        "BTC Baseline",
        btc_result,
        btc_sample,
        BTC_BASELINE_PREDICTORS
    ),

    (
        "BTC + Lagged ETH Robustness",
        btc_robust_result,
        btc_robust_sample,
        BTC_ROBUST_PREDICTORS
    ),

    (
        "ETH Baseline",
        eth_result,
        eth_sample,
        ETH_BASELINE_PREDICTORS
    ),

    (
        "ETH + Lagged BTC Robustness",
        eth_robust_result,
        eth_robust_sample,
        ETH_ROBUST_PREDICTORS
    ),
]


for (
    model_name,
    result,
    sample,
    predictors

) in models_for_effects:


    for variable in predictors:

        beta = (
            result.params[
                variable
            ]
        )


        predictor_std = (
            sample[
                variable
            ]
            .std()
        )


        one_sd_effect = (
            beta
            * predictor_std
        )


        percentage_point_effect = (
            one_sd_effect
            * 100
        )


        basis_point_effect = (
            one_sd_effect
            * 10000
        )


        economic_effect_records.append({

            "Model":
                model_name,

            "Variable":
                variable,

            "Coefficient":
                beta,

            "Predictor_SD":
                predictor_std,

            "One_SD_Return_Effect":
                one_sd_effect,

            "One_SD_Effect_Percentage_Points":
                percentage_point_effect,

            "One_SD_Effect_Basis_Points":
                basis_point_effect,

            "p_Value":
                result.pvalues[
                    variable
                ],

            "Significance":
                significance_stars(
                    result.pvalues[
                        variable
                    ]
                )
        })


economic_effects = pd.DataFrame(
    economic_effect_records
)


print(
    economic_effects
    .to_string(index=False)
)


# =============================================================================
# 21. SAVE COEFFICIENT RESULTS
# =============================================================================

print_section(
    "SAVING RESULTS"
)


btc_table.to_csv(
    BTC_RESULTS_FILE,
    index=False
)


eth_table.to_csv(
    ETH_RESULTS_FILE,
    index=False
)


btc_robust_table.to_csv(
    BTC_ROBUST_RESULTS_FILE,
    index=False
)


eth_robust_table.to_csv(
    ETH_ROBUST_RESULTS_FILE,
    index=False
)


model_summary.to_csv(
    MODEL_SUMMARY_FILE,
    index=False
)


economic_effects.to_csv(
    ECONOMIC_EFFECTS_FILE,
    index=False
)


print(
    "\nBTC baseline results:"
)

print(
    BTC_RESULTS_FILE
)


print(
    "\nETH baseline results:"
)

print(
    ETH_RESULTS_FILE
)


print(
    "\nBTC cross-crypto robustness results:"
)

print(
    BTC_ROBUST_RESULTS_FILE
)


print(
    "\nETH cross-crypto robustness results:"
)

print(
    ETH_ROBUST_RESULTS_FILE
)


print(
    "\nModel summary:"
)

print(
    MODEL_SUMMARY_FILE
)


print(
    "\nEconomic significance:"
)

print(
    ECONOMIC_EFFECTS_FILE
)


# =============================================================================
# 22. INTERPRETATION REMINDER
# =============================================================================

print_section(
    "INTERPRETATION REMINDER"
)


print(
    """
These are BASELINE EXPLANATORY results.

You may discuss:

    - coefficient signs;
    - coefficient magnitudes;
    - HAC/Newey-West statistical significance;
    - confidence intervals;
    - R-squared;
    - adjusted R-squared;
    - economic magnitude of coefficients;
    - whether cross-crypto lagged returns affect the results.

You should NOT use these regressions to claim:

    - forecasting ability;
    - predictive accuracy;
    - improved prediction;
    - out-of-sample predictive power.

Those claims require the separate expanding-window out-of-sample
forecasting exercise.

When Reddit data become available, the explanatory specifications can
later be extended sequentially:

    Model 1: Controls only

    Model 2: Controls + Reddit activity

    Model 3: Controls + Reddit sentiment

    Model 4: Controls + Reddit activity + sentiment

This will separate the contribution of Reddit attention/activity from
the contribution of Reddit sentiment.
"""
)


# =============================================================================
# 23. NEXT STEP
# =============================================================================

print_section(
    "NEXT STEP"
)


print(
    """
After validating these baseline explanatory regressions, the next step
is to estimate the BASELINE OUT-OF-SAMPLE FORECASTING MODELS.

Those models will use:

    - 2021-2023 as the initial estimation window;
    - 2024-2025 as the out-of-sample evaluation period;
    - one-day-ahead forecasts;
    - expanding estimation windows;
    - exactly the same information-aligned predictor set.

The baseline forecasts will eventually provide the benchmark against
which the Reddit sentiment/activity forecasting models will be compared.
"""
)


print_section(
    "BASELINE EXPLANATORY ANALYSIS COMPLETE"
)