# =====================================================================
# 07_residual_time_series_diagnostics.py
#
# FORMAL TIME-SERIES DIAGNOSTICS FOR REGRESSION RESIDUALS
#
# Dissertation:
# Do Social Media Sentiment Signals Improve the Prediction of
# Cryptocurrency Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum
#
# =====================================================================
#
# PURPOSE
# ---------------------------------------------------------------------
# Conduct formal residual diagnostics for the PRIMARY t-1 baseline
# explanatory regressions.
#
# This script deliberately DOES NOT repeat:
#
#   - descriptive statistics
#   - correlation matrices
#   - VIF analysis
#   - alternative lag-length robustness
#   - trading-volume verification
#
# Those analyses are handled elsewhere.
#
# ---------------------------------------------------------------------
# PRIMARY MODELS
# ---------------------------------------------------------------------
#
# BTC:
#
#   BTC_Return_t =
#       alpha
#       + beta1 BTC_Return_(t-1)
#       + beta2 Log_BTC_Volume_(t-1)
#       + beta3 aligned SP500 information
#       + beta4 aligned VIX information
#       + beta5 aligned Gold information
#       + beta6 aligned DXY information
#       + beta7 aligned US10Y information
#       + error_t
#
#
# ETH:
#
#   ETH_Return_t =
#       alpha
#       + beta1 ETH_Return_(t-1)
#       + beta2 Log_ETH_Volume_(t-1)
#       + beta3 aligned SP500 information
#       + beta4 aligned VIX information
#       + beta5 aligned Gold information
#       + beta6 aligned DXY information
#       + beta7 aligned US10Y information
#       + error_t
#
# ---------------------------------------------------------------------
# DIAGNOSTICS
# ---------------------------------------------------------------------
#
# Heteroskedasticity:
#
#   1. Breusch-Pagan LM test
#   2. White test
#
# Serial correlation:
#
#   3. Breusch-Godfrey LM tests:
#          lag 1
#          lag 3
#          lag 7
#
#   4. Ljung-Box Q tests:
#          lag 1
#          lag 3
#          lag 7
#          lag 14
#          lag 30
#
# Supporting residual diagnostics:
#
#   5. Residual autocorrelation coefficients
#   6. Durbin-Watson statistic
#   7. Jarque-Bera residual-normality test
#
# Robust inference comparison:
#
#   8. Conventional OLS standard errors
#      versus
#      HAC / Newey-West standard errors
#
# ---------------------------------------------------------------------
# IMPORTANT INTERPRETATION
# ---------------------------------------------------------------------
#
# Evidence of heteroskedasticity and/or serial correlation does NOT
# automatically mean that the OLS coefficient estimates are invalid.
#
# Instead, these properties make conventional homoskedastic,
# independently distributed OLS standard errors unreliable.
#
# The dissertation therefore reports HAC/Newey-West inference for the
# explanatory regressions.
#
# HAC maxlags = 7 is retained for consistency with the established
# dissertation specification.
#
# =====================================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from statsmodels.stats.diagnostic import (
    acorr_breusch_godfrey,
    acorr_ljungbox,
    het_breuschpagan,
    het_white
)

from statsmodels.stats.stattools import (
    durbin_watson,
    jarque_bera
)

from statsmodels.tsa.stattools import acf


warnings.filterwarnings("ignore")


# =====================================================================
# 1. PROJECT PATHS
# =====================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

DATA_PROCESSED = (
    PROJECT_ROOT
    / "data_processed"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "analysis_outputs"
    / "residual_time_series_diagnostics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


INPUT_FILE = (
    DATA_PROCESSED
    / "final_forecast_dataset.csv"
)


# =====================================================================
# 2. SETTINGS
# =====================================================================

HAC_MAXLAGS = 7

SIGNIFICANCE_LEVEL = 0.05

BG_LAGS = [
    1,
    3,
    7
]

LJUNG_BOX_LAGS = [
    1,
    3,
    7,
    14,
    30
]

ACF_MAX_LAG = 30


# =====================================================================
# 3. MODEL SPECIFICATIONS
# =====================================================================

BTC_DEPENDENT = "BTC_Return"

BTC_PREDICTORS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume",

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


ETH_DEPENDENT = "ETH_Return"

ETH_PREDICTORS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume",

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


# =====================================================================
# 4. HELPER FUNCTIONS
# =====================================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def significance_label(p_value):

    if pd.isna(p_value):

        return "NA"

    if p_value < 0.01:

        return "***"

    if p_value < 0.05:

        return "**"

    if p_value < 0.10:

        return "*"

    return ""


def reject_at_5_percent(p_value):

    if pd.isna(p_value):

        return False

    return bool(
        p_value < SIGNIFICANCE_LEVEL
    )


def evidence_label(
    p_value,
    alternative_description
):

    if pd.isna(p_value):

        return "Not available"

    if p_value < SIGNIFICANCE_LEVEL:

        return (
            f"Evidence of {alternative_description}"
        )

    return (
        f"No evidence of {alternative_description} "
        f"at the 5% level"
    )


def safe_float(value):

    try:

        return float(value)

    except Exception:

        return np.nan


# =====================================================================
# 5. START
# =====================================================================

section(
    "FORMAL TIME-SERIES DIAGNOSTICS FOR REGRESSION RESIDUALS"
)


print(
    "\nInput dataset:"
)

print(
    INPUT_FILE
)


print(
    "\nHAC/Newey-West maximum lag:"
)

print(
    HAC_MAXLAGS
)


print(
    "\nPrimary significance level for diagnostic interpretation:"
)

print(
    SIGNIFICANCE_LEVEL
)


print(
    "\nThis script does NOT repeat descriptive statistics, "
    "correlations, or VIF analysis."
)


# =====================================================================
# 6. CHECK INPUT FILE
# =====================================================================

section(
    "CHECKING INPUT FILE"
)


print(
    "\nExists:"
)

print(
    INPUT_FILE.exists()
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nRequired file not found:\n{INPUT_FILE}"
    )


# =====================================================================
# 7. LOAD DATA
# =====================================================================

section(
    "LOADING FINAL DATASET"
)


df = pd.read_csv(
    INPUT_FILE
)


print(
    "\nDataset shape:"
)

print(
    df.shape
)


print(
    "\nDataset columns:"
)

for column in df.columns:

    print(
        " -",
        column
    )


# =====================================================================
# 8. DATE VALIDATION
# =====================================================================

section(
    "DATE VALIDATION"
)


if "Date" not in df.columns:

    raise KeyError(
        "Date column is missing from final_forecast_dataset.csv."
    )


df[
    "Date"
] = pd.to_datetime(

    df[
        "Date"
    ],

    errors="coerce"

)


invalid_dates = int(
    df[
        "Date"
    ]
    .isna()
    .sum()
)


duplicate_dates = int(
    df[
        "Date"
    ]
    .duplicated()
    .sum()
)


print(
    "\nInvalid dates:"
)

print(
    invalid_dates
)


print(
    "\nDuplicate dates:"
)

print(
    duplicate_dates
)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates found."
    )


if duplicate_dates > 0:

    raise ValueError(
        "Duplicate dates found."
    )


df = (

    df

    .sort_values(
        "Date"
    )

    .reset_index(
        drop=True
    )

)


print(
    "\nDate range:"
)

print(
    df[
        "Date"
    ].min(),
    "to",
    df[
        "Date"
    ].max()
)


# =====================================================================
# 9. CALENDAR CHECK
# =====================================================================

section(
    "CALENDAR CHECK"
)


date_difference = (
    df[
        "Date"
    ]
    .diff()
)


calendar_gap_mask = (

    date_difference.notna()

    &

    (
        date_difference
        != pd.Timedelta(
            days=1
        )
    )

)


calendar_gaps = int(
    calendar_gap_mask.sum()
)


weekend_observations = int(

    (
        df[
            "Date"
        ]
        .dt.dayofweek >= 5
    )
    .sum()

)


print(
    "\nObservations:"
)

print(
    len(
        df
    )
)


print(
    "\nNon-consecutive calendar gaps:"
)

print(
    calendar_gaps
)


print(
    "\nWeekend observations:"
)

print(
    weekend_observations
)


if calendar_gaps > 0:

    raise ValueError(
        "The final dataset is not a continuous daily calendar."
    )


print(
    "\nPASS: Continuous daily cryptocurrency calendar confirmed."
)


# =====================================================================
# 10. SAFETY CHECK:
#     REQUIRE INFORMATION-ALIGNED MARKET CONTROLS
# =====================================================================

section(
    "INFORMATION-ALIGNMENT SAFETY CHECK"
)


ALIGNED_CONTROLS = [

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


OLD_NONALIGNED_CONTROLS = [

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"

]


for variable in ALIGNED_CONTROLS:

    if variable not in df.columns:

        raise KeyError(
            f"Required aligned control missing: {variable}"
        )


print(
    "\nAligned controls required by this script:"
)

for variable in ALIGNED_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nOld non-aligned controls will NOT be used:"
)

for variable in OLD_NONALIGNED_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nPASS: Residual diagnostics will use "
    "information-aligned market controls."
)


# =====================================================================
# 11. REQUIRED MODEL COLUMNS
# =====================================================================

section(
    "MODEL-COLUMN VALIDATION"
)


REQUIRED_COLUMNS = (

    [
        BTC_DEPENDENT,
        ETH_DEPENDENT
    ]

    +

    BTC_PREDICTORS

    +

    ETH_PREDICTORS

)


REQUIRED_COLUMNS = list(
    dict.fromkeys(
        REQUIRED_COLUMNS
    )
)


missing_columns = [

    variable

    for variable in REQUIRED_COLUMNS

    if variable not in df.columns

]


if missing_columns:

    raise KeyError(
        "\nMissing required variables:\n"
        +
        "\n".join(
            missing_columns
        )
    )


print(
    "\nPASS: All required model variables are available."
)


# =====================================================================
# 12. CONVERT MODEL VARIABLES TO NUMERIC
# =====================================================================

section(
    "NUMERIC CONVERSION"
)


for variable in REQUIRED_COLUMNS:

    df[
        variable
    ] = (

        pd.to_numeric(
            df[
                variable
            ],
            errors="coerce"
        )

        .replace(
            [
                np.inf,
                -np.inf
            ],
            np.nan
        )

    )


print(
    "\nNumeric conversion complete."
)


# =====================================================================
# 13. PREPARE MODEL SAMPLE
# =====================================================================

section(
    "PREPARING PRIMARY MODEL SAMPLES"
)


def prepare_model_sample(
    dataframe,
    asset,
    dependent,
    predictors
):

    required = (
        [
            "Date",
            dependent
        ]
        +
        predictors
    )


    sample = (

        dataframe[
            required
        ]

        .dropna()

        .copy()

    )


    sample = (

        sample

        .sort_values(
            "Date"
        )

        .reset_index(
            drop=True
        )

    )


    sample[
        "Weekend"
    ] = (

        sample[
            "Date"
        ]
        .dt.dayofweek >= 5

    )


    sample_date_difference = (
        sample[
            "Date"
        ]
        .diff()
    )


    # Missing observations at the beginning of the full dataset can
    # result from lag construction / initial traditional-market
    # information availability.
    #
    # Once the estimation sample begins, we want to know whether it
    # remains a continuous calendar.
    sample_internal_gaps = int(

        (
            sample_date_difference.notna()

            &

            (
                sample_date_difference
                != pd.Timedelta(
                    days=1
                )
            )

        ).sum()

    )


    print(
        f"\n{asset} estimation observations:"
    )

    print(
        len(
            sample
        )
    )


    print(
        f"{asset} estimation period:"
    )

    print(
        sample[
            "Date"
        ].min(),
        "to",
        sample[
            "Date"
        ].max()
    )


    print(
        f"{asset} weekend observations:"
    )

    print(
        int(
            sample[
                "Weekend"
            ]
            .sum()
        )
    )


    print(
        f"{asset} internal estimation-sample date gaps:"
    )

    print(
        sample_internal_gaps
    )


    if sample_internal_gaps > 0:

        print(
            f"\nWARNING: {asset} estimation sample contains "
            f"internal date gaps."
        )


    return sample


btc_sample = prepare_model_sample(

    df,

    "BTC",

    BTC_DEPENDENT,

    BTC_PREDICTORS

)


eth_sample = prepare_model_sample(

    df,

    "ETH",

    ETH_DEPENDENT,

    ETH_PREDICTORS

)


# =====================================================================
# 14. ESTIMATE PRIMARY OLS MODELS
# =====================================================================

section(
    "ESTIMATING PRIMARY t-1 BASELINE MODELS"
)


def estimate_model(
    sample,
    dependent,
    predictors
):

    y = sample[
        dependent
    ]


    X = sm.add_constant(
        sample[
            predictors
        ],
        has_constant="add"
    )


    ordinary_model = sm.OLS(
        y,
        X
    ).fit()


    hac_model = ordinary_model.get_robustcov_results(

        cov_type="HAC",

        maxlags=HAC_MAXLAGS

    )


    return (
        ordinary_model,
        hac_model,
        X
    )


(
    btc_ols,
    btc_hac,
    btc_X
) = estimate_model(

    btc_sample,

    BTC_DEPENDENT,

    BTC_PREDICTORS

)


(
    eth_ols,
    eth_hac,
    eth_X
) = estimate_model(

    eth_sample,

    ETH_DEPENDENT,

    ETH_PREDICTORS

)


print(
    "\nBTC primary model:"
)

print(
    "Dependent:",
    BTC_DEPENDENT
)

print(
    "N:",
    int(
        btc_ols.nobs
    )
)

print(
    "R-squared:",
    btc_ols.rsquared
)


print(
    "\nETH primary model:"
)

print(
    "Dependent:",
    ETH_DEPENDENT
)

print(
    "N:",
    int(
        eth_ols.nobs
    )
)

print(
    "R-squared:",
    eth_ols.rsquared
)


# =====================================================================
# 15. SAVE RESIDUAL SERIES
# =====================================================================

section(
    "CONSTRUCTING RESIDUAL SERIES"
)


btc_residuals = pd.DataFrame(

    {

        "Date":
            btc_sample[
                "Date"
            ].values,

        "Asset":
            "BTC",

        "Observed_Return":
            btc_sample[
                BTC_DEPENDENT
            ].values,

        "Fitted_Return":
            btc_ols.fittedvalues.values,

        "Residual":
            btc_ols.resid.values

    }

)


eth_residuals = pd.DataFrame(

    {

        "Date":
            eth_sample[
                "Date"
            ].values,

        "Asset":
            "ETH",

        "Observed_Return":
            eth_sample[
                ETH_DEPENDENT
            ].values,

        "Fitted_Return":
            eth_ols.fittedvalues.values,

        "Residual":
            eth_ols.resid.values

    }

)


all_residuals = pd.concat(

    [
        btc_residuals,
        eth_residuals
    ],

    ignore_index=True

)


print(
    "\nBTC residual observations:"
)

print(
    len(
        btc_residuals
    )
)


print(
    "\nETH residual observations:"
)

print(
    len(
        eth_residuals
    )
)


# =====================================================================
# 16. BREUSCH-PAGAN TEST
# =====================================================================
#
# H0:
#     Homoskedasticity.
#
# H1:
#     Conditional error variance is related to the regressors.
#
# =====================================================================

section(
    "BREUSCH-PAGAN HETEROSKEDASTICITY TEST"
)


def run_breusch_pagan(
    asset,
    model,
    X
):

    lm_statistic, lm_pvalue, f_statistic, f_pvalue = (
        het_breuschpagan(
            model.resid,
            X
        )
    )


    print(
        f"\n{asset}:"
    )

    print(
        "LM statistic:",
        lm_statistic
    )

    print(
        "LM p-value:",
        lm_pvalue,
        significance_label(
            lm_pvalue
        )
    )

    print(
        "F statistic:",
        f_statistic
    )

    print(
        "F p-value:",
        f_pvalue,
        significance_label(
            f_pvalue
        )
    )

    print(
        "Interpretation:",
        evidence_label(
            lm_pvalue,
            "heteroskedasticity"
        )
    )


    return {

        "Asset":
            asset,

        "Test":
            "Breusch-Pagan",

        "Lag":
            np.nan,

        "Statistic":
            lm_statistic,

        "P_Value":
            lm_pvalue,

        "Secondary_Statistic":
            f_statistic,

        "Secondary_P_Value":
            f_pvalue,

        "Reject_H0_5pct":
            reject_at_5_percent(
                lm_pvalue
            ),

        "Null_Hypothesis":
            "Homoskedasticity",

        "Interpretation":
            evidence_label(
                lm_pvalue,
                "heteroskedasticity"
            )

    }


btc_bp = run_breusch_pagan(
    "BTC",
    btc_ols,
    btc_X
)


eth_bp = run_breusch_pagan(
    "ETH",
    eth_ols,
    eth_X
)


# =====================================================================
# 17. WHITE HETEROSKEDASTICITY TEST
# =====================================================================
#
# H0:
#     Homoskedasticity.
#
# White is more general than Breusch-Pagan because it can capture
# nonlinear forms of variance dependence.
#
# =====================================================================

section(
    "WHITE HETEROSKEDASTICITY TEST"
)


def run_white_test(
    asset,
    model,
    X
):

    lm_statistic, lm_pvalue, f_statistic, f_pvalue = (
        het_white(
            model.resid,
            X
        )
    )


    print(
        f"\n{asset}:"
    )

    print(
        "LM statistic:",
        lm_statistic
    )

    print(
        "LM p-value:",
        lm_pvalue,
        significance_label(
            lm_pvalue
        )
    )

    print(
        "F statistic:",
        f_statistic
    )

    print(
        "F p-value:",
        f_pvalue,
        significance_label(
            f_pvalue
        )
    )

    print(
        "Interpretation:",
        evidence_label(
            lm_pvalue,
            "heteroskedasticity"
        )
    )


    return {

        "Asset":
            asset,

        "Test":
            "White",

        "Lag":
            np.nan,

        "Statistic":
            lm_statistic,

        "P_Value":
            lm_pvalue,

        "Secondary_Statistic":
            f_statistic,

        "Secondary_P_Value":
            f_pvalue,

        "Reject_H0_5pct":
            reject_at_5_percent(
                lm_pvalue
            ),

        "Null_Hypothesis":
            "Homoskedasticity",

        "Interpretation":
            evidence_label(
                lm_pvalue,
                "heteroskedasticity"
            )

    }


btc_white = run_white_test(
    "BTC",
    btc_ols,
    btc_X
)


eth_white = run_white_test(
    "ETH",
    eth_ols,
    eth_X
)


# =====================================================================
# 18. DURBIN-WATSON
# =====================================================================
#
# DW approximately:
#
#   2   -> little first-order residual autocorrelation
#   < 2 -> positive autocorrelation tendency
#   > 2 -> negative autocorrelation tendency
#
# We treat this as descriptive/supporting evidence rather than the
# principal formal serial-correlation test because the model includes
# a lagged dependent variable.
#
# =====================================================================

section(
    "DURBIN-WATSON SUPPORTING DIAGNOSTIC"
)


btc_dw = safe_float(
    durbin_watson(
        btc_ols.resid
    )
)


eth_dw = safe_float(
    durbin_watson(
        eth_ols.resid
    )
)


print(
    "\nBTC Durbin-Watson:"
)

print(
    btc_dw
)


print(
    "\nETH Durbin-Watson:"
)

print(
    eth_dw
)


print(
    "\nNote: Durbin-Watson is reported as a supporting descriptive "
    "diagnostic. Formal serial-correlation inference below relies "
    "primarily on Breusch-Godfrey and Ljung-Box tests."
)


# =====================================================================
# 19. BREUSCH-GODFREY SERIAL-CORRELATION TESTS
# =====================================================================
#
# H0:
#     No residual serial correlation through the specified lag.
#
# H1:
#     Residual serial correlation is present.
#
# =====================================================================

section(
    "BREUSCH-GODFREY SERIAL-CORRELATION TESTS"
)


def run_breusch_godfrey(
    asset,
    model
):

    results = []


    for lag in BG_LAGS:

        (
            lm_statistic,
            lm_pvalue,
            f_statistic,
            f_pvalue
        ) = acorr_breusch_godfrey(

            model,

            nlags=lag

        )


        print(
            f"\n{asset} - BG lag {lag}:"
        )

        print(
            "LM statistic:",
            lm_statistic
        )

        print(
            "LM p-value:",
            lm_pvalue,
            significance_label(
                lm_pvalue
            )
        )

        print(
            "F statistic:",
            f_statistic
        )

        print(
            "F p-value:",
            f_pvalue,
            significance_label(
                f_pvalue
            )
        )

        print(
            "Interpretation:",
            evidence_label(
                lm_pvalue,
                f"residual serial correlation through lag {lag}"
            )
        )


        results.append(

            {

                "Asset":
                    asset,

                "Test":
                    "Breusch-Godfrey",

                "Lag":
                    lag,

                "Statistic":
                    lm_statistic,

                "P_Value":
                    lm_pvalue,

                "Secondary_Statistic":
                    f_statistic,

                "Secondary_P_Value":
                    f_pvalue,

                "Reject_H0_5pct":
                    reject_at_5_percent(
                        lm_pvalue
                    ),

                "Null_Hypothesis":
                    (
                        "No residual serial correlation "
                        f"through lag {lag}"
                    ),

                "Interpretation":
                    evidence_label(
                        lm_pvalue,
                        (
                            "residual serial correlation "
                            f"through lag {lag}"
                        )
                    )

            }

        )


    return results


btc_bg = run_breusch_godfrey(
    "BTC",
    btc_ols
)


eth_bg = run_breusch_godfrey(
    "ETH",
    eth_ols
)


# =====================================================================
# 20. LJUNG-BOX RESIDUAL SERIAL-CORRELATION TESTS
# =====================================================================
#
# H0:
#     Residual autocorrelations jointly equal zero through lag m.
#
# =====================================================================

section(
    "LJUNG-BOX RESIDUAL AUTOCORRELATION TESTS"
)


def run_ljung_box(
    asset,
    model
):

    lb = acorr_ljungbox(

        model.resid,

        lags=LJUNG_BOX_LAGS,

        return_df=True

    )


    results = []


    print(
        f"\n{asset}:"
    )


    for lag in LJUNG_BOX_LAGS:

        statistic = safe_float(
            lb.loc[
                lag,
                "lb_stat"
            ]
        )


        p_value = safe_float(
            lb.loc[
                lag,
                "lb_pvalue"
            ]
        )


        print(
            f"\nLag {lag}:"
        )

        print(
            "Q statistic:",
            statistic
        )

        print(
            "p-value:",
            p_value,
            significance_label(
                p_value
            )
        )

        print(
            "Interpretation:",
            evidence_label(
                p_value,
                (
                    "joint residual autocorrelation "
                    f"through lag {lag}"
                )
            )
        )


        results.append(

            {

                "Asset":
                    asset,

                "Test":
                    "Ljung-Box",

                "Lag":
                    lag,

                "Statistic":
                    statistic,

                "P_Value":
                    p_value,

                "Secondary_Statistic":
                    np.nan,

                "Secondary_P_Value":
                    np.nan,

                "Reject_H0_5pct":
                    reject_at_5_percent(
                        p_value
                    ),

                "Null_Hypothesis":
                    (
                        "Residual autocorrelations jointly "
                        f"equal zero through lag {lag}"
                    ),

                "Interpretation":
                    evidence_label(
                        p_value,
                        (
                            "joint residual autocorrelation "
                            f"through lag {lag}"
                        )
                    )

            }

        )


    return results


btc_lb = run_ljung_box(
    "BTC",
    btc_ols
)


eth_lb = run_ljung_box(
    "ETH",
    eth_ols
)


# =====================================================================
# 21. RESIDUAL AUTOCORRELATION COEFFICIENTS
# =====================================================================

section(
    "RESIDUAL AUTOCORRELATION COEFFICIENTS"
)


def residual_acf_table(
    asset,
    residuals
):

    acf_values = acf(

        residuals,

        nlags=ACF_MAX_LAG,

        fft=True,

        missing="drop"

    )


    table = pd.DataFrame(

        {

            "Asset":
                asset,

            "Lag":
                np.arange(
                    len(
                        acf_values
                    )
                ),

            "Residual_ACF":
                acf_values

        }

    )


    # Approximate 95% reference bound:
    #
    # +/- 1.96 / sqrt(N)
    #
    # This is included only as a supporting descriptive reference.

    reference_bound = (
        1.96
        /
        np.sqrt(
            len(
                residuals
            )
        )
    )


    table[
        "Approx_95pct_Bound"
    ] = reference_bound


    table[
        "Outside_Approx_95pct_Bound"
    ] = (

        table[
            "Residual_ACF"
        ].abs()

        >

        reference_bound

    )


    print(
        f"\n{asset} approximate 95% ACF reference bound:"
    )

    print(
        reference_bound
    )


    print(
        f"\n{asset} selected residual ACF values:"
    )


    selected_lags = [
        1,
        2,
        3,
        7,
        14,
        30
    ]


    print(

        table.loc[

            table[
                "Lag"
            ].isin(
                selected_lags
            )

        ].to_string(
            index=False
        )

    )


    return table


btc_acf = residual_acf_table(

    "BTC",

    btc_ols.resid

)


eth_acf = residual_acf_table(

    "ETH",

    eth_ols.resid

)


all_acf = pd.concat(

    [
        btc_acf,
        eth_acf
    ],

    ignore_index=True

)


# =====================================================================
# 22. JARQUE-BERA SUPPLEMENTARY DIAGNOSTIC
# =====================================================================
#
# H0:
#     Residuals are normally distributed.
#
# Normality is NOT required for HAC consistency in a large-sample
# setting, so this test is supplementary rather than the central
# justification for HAC.
#
# =====================================================================

section(
    "JARQUE-BERA SUPPLEMENTARY NORMALITY DIAGNOSTIC"
)


def run_jarque_bera(
    asset,
    residuals
):

    (
        jb_statistic,
        jb_pvalue,
        skewness,
        kurtosis
    ) = jarque_bera(
        residuals
    )


    print(
        f"\n{asset}:"
    )

    print(
        "Jarque-Bera statistic:",
        jb_statistic
    )

    print(
        "p-value:",
        jb_pvalue,
        significance_label(
            jb_pvalue
        )
    )

    print(
        "Residual skewness:",
        skewness
    )

    print(
        "Residual kurtosis:",
        kurtosis
    )


    if jb_pvalue < SIGNIFICANCE_LEVEL:

        interpretation = (
            "Residual normality rejected at the 5% level"
        )

    else:

        interpretation = (
            "Residual normality not rejected at the 5% level"
        )


    print(
        "Interpretation:",
        interpretation
    )


    return {

        "Asset":
            asset,

        "Test":
            "Jarque-Bera",

        "Lag":
            np.nan,

        "Statistic":
            jb_statistic,

        "P_Value":
            jb_pvalue,

        "Secondary_Statistic":
            skewness,

        "Secondary_P_Value":
            kurtosis,

        "Reject_H0_5pct":
            reject_at_5_percent(
                jb_pvalue
            ),

        "Null_Hypothesis":
            "Residual normality",

        "Interpretation":
            interpretation

    }


btc_jb = run_jarque_bera(

    "BTC",

    btc_ols.resid

)


eth_jb = run_jarque_bera(

    "ETH",

    eth_ols.resid

)


# =====================================================================
# 23. OLS VS HAC / NEWEY-WEST INFERENCE
# =====================================================================
#
# Coefficient estimates are identical.
#
# Only the covariance estimator, standard errors, test statistics,
# confidence intervals, and p-values change.
#
# =====================================================================

section(
    "CONVENTIONAL OLS VS HAC / NEWEY-WEST INFERENCE"
)


def inference_comparison(
    asset,
    ordinary_model,
    hac_model,
    X
):

    variable_names = list(
        X.columns
    )


    ordinary_params = np.asarray(
        ordinary_model.params
    )


    ordinary_se = np.asarray(
        ordinary_model.bse
    )


    ordinary_pvalues = np.asarray(
        ordinary_model.pvalues
    )


    hac_params = np.asarray(
        hac_model.params
    )


    hac_se = np.asarray(
        hac_model.bse
    )


    hac_pvalues = np.asarray(
        hac_model.pvalues
    )


    rows = []


    for i, variable in enumerate(
        variable_names
    ):

        coefficient_difference = (

            hac_params[i]
            -
            ordinary_params[i]

        )


        if ordinary_se[i] != 0:

            se_ratio = (
                hac_se[i]
                /
                ordinary_se[i]
            )

        else:

            se_ratio = np.nan


        rows.append(

            {

                "Asset":
                    asset,

                "Variable":
                    variable,

                "Coefficient":
                    ordinary_params[i],

                "HAC_Coefficient":
                    hac_params[i],

                "Coefficient_Difference":
                    coefficient_difference,

                "OLS_SE":
                    ordinary_se[i],

                "HAC_SE":
                    hac_se[i],

                "HAC_to_OLS_SE_Ratio":
                    se_ratio,

                "OLS_P_Value":
                    ordinary_pvalues[i],

                "HAC_P_Value":
                    hac_pvalues[i],

                "OLS_Significance":
                    significance_label(
                        ordinary_pvalues[i]
                    ),

                "HAC_Significance":
                    significance_label(
                        hac_pvalues[i]
                    )

            }

        )


    comparison = pd.DataFrame(
        rows
    )


    max_coefficient_difference = (
        comparison[
            "Coefficient_Difference"
        ]
        .abs()
        .max()
    )


    print(
        f"\n{asset}:"
    )

    print(
        "Maximum coefficient difference "
        "between OLS and HAC objects:"
    )

    print(
        max_coefficient_difference
    )


    print(
        "\nInference comparison:"
    )

    print(

        comparison[
            [
                "Variable",
                "Coefficient",
                "OLS_SE",
                "HAC_SE",
                "HAC_to_OLS_SE_Ratio",
                "OLS_P_Value",
                "HAC_P_Value"
            ]
        ]
        .to_string(
            index=False
        )

    )


    return comparison


btc_inference = inference_comparison(

    "BTC",

    btc_ols,

    btc_hac,

    btc_X

)


eth_inference = inference_comparison(

    "ETH",

    eth_ols,

    eth_hac,

    eth_X

)


all_inference = pd.concat(

    [
        btc_inference,
        eth_inference
    ],

    ignore_index=True

)


# =====================================================================
# 24. COMBINE FORMAL TEST RESULTS
# =====================================================================

section(
    "COMBINING FORMAL DIAGNOSTIC RESULTS"
)


formal_test_rows = [

    btc_bp,
    eth_bp,

    btc_white,
    eth_white,

    *btc_bg,
    *eth_bg,

    *btc_lb,
    *eth_lb,

    btc_jb,
    eth_jb

]


formal_tests = pd.DataFrame(
    formal_test_rows
)


print(
    "\n",
    formal_tests.to_string(
        index=False
    )
)


# =====================================================================
# 25. MODEL-LEVEL DIAGNOSTIC SUMMARY
# =====================================================================

section(
    "MODEL-LEVEL DIAGNOSTIC SUMMARY"
)


def create_model_summary(
    asset,
    model,
    sample,
    bp_result,
    white_result,
    bg_results,
    lb_results,
    jb_result,
    dw_statistic
):

    heteroskedasticity_detected = (

        bp_result[
            "Reject_H0_5pct"
        ]

        or

        white_result[
            "Reject_H0_5pct"
        ]

    )


    bg_serial_correlation_detected = any(

        result[
            "Reject_H0_5pct"
        ]

        for result in bg_results

    )


    lb_serial_correlation_detected = any(

        result[
            "Reject_H0_5pct"
        ]

        for result in lb_results

    )


    serial_correlation_detected = (

        bg_serial_correlation_detected

        or

        lb_serial_correlation_detected

    )


    hac_empirically_supported = (

        heteroskedasticity_detected

        or

        serial_correlation_detected

    )


    return {

        "Asset":
            asset,

        "N":
            int(
                model.nobs
            ),

        "Start_Date":
            sample[
                "Date"
            ].min(),

        "End_Date":
            sample[
                "Date"
            ].max(),

        "Weekend_Observations":
            int(
                sample[
                    "Weekend"
                ].sum()
            ),

        "R_Squared":
            model.rsquared,

        "Adjusted_R_Squared":
            model.rsquared_adj,

        "Durbin_Watson":
            dw_statistic,

        "Breusch_Pagan_P":
            bp_result[
                "P_Value"
            ],

        "White_P":
            white_result[
                "P_Value"
            ],

        "Heteroskedasticity_Detected_5pct":
            heteroskedasticity_detected,

        "BG_Serial_Correlation_Detected_5pct":
            bg_serial_correlation_detected,

        "Ljung_Box_Serial_Correlation_Detected_5pct":
            lb_serial_correlation_detected,

        "Any_Serial_Correlation_Detected_5pct":
            serial_correlation_detected,

        "Jarque_Bera_P":
            jb_result[
                "P_Value"
            ],

        "Residual_Normality_Rejected_5pct":
            jb_result[
                "Reject_H0_5pct"
            ],

        "HAC_Empirically_Supported":
            hac_empirically_supported,

        "HAC_Maxlags":
            HAC_MAXLAGS

    }


btc_summary = create_model_summary(

    "BTC",

    btc_ols,

    btc_sample,

    btc_bp,

    btc_white,

    btc_bg,

    btc_lb,

    btc_jb,

    btc_dw

)


eth_summary = create_model_summary(

    "ETH",

    eth_ols,

    eth_sample,

    eth_bp,

    eth_white,

    eth_bg,

    eth_lb,

    eth_jb,

    eth_dw

)


model_summary = pd.DataFrame(

    [
        btc_summary,
        eth_summary
    ]

)


print(
    "\n",
    model_summary.to_string(
        index=False
    )
)


# =====================================================================
# 26. AUTOMATED INTERPRETATION
# =====================================================================

section(
    "AUTOMATED INTERPRETATION"
)


def print_asset_interpretation(
    summary
):

    asset = summary[
        "Asset"
    ]


    print(
        f"\n{asset}:"
    )


    if summary[
        "Heteroskedasticity_Detected_5pct"
    ]:

        print(
            "- At least one formal heteroskedasticity test "
            "rejects homoskedasticity at the 5% level."
        )

    else:

        print(
            "- Neither Breusch-Pagan nor White provides "
            "5%-level evidence of heteroskedasticity."
        )


    if summary[
        "Any_Serial_Correlation_Detected_5pct"
    ]:

        print(
            "- At least one formal serial-correlation test "
            "detects residual dependence at the 5% level."
        )

    else:

        print(
            "- The selected Breusch-Godfrey and Ljung-Box tests "
            "do not detect residual serial correlation "
            "at the 5% level."
        )


    if summary[
        "Residual_Normality_Rejected_5pct"
    ]:

        print(
            "- Jarque-Bera rejects residual normality."
        )

    else:

        print(
            "- Jarque-Bera does not reject residual normality "
            "at the 5% level."
        )


    if summary[
        "HAC_Empirically_Supported"
    ]:

        print(
            "- The residual diagnostics provide direct empirical "
            "support for using HAC/Newey-West inference rather "
            "than conventional OLS standard errors."
        )

    else:

        print(
            "- The selected diagnostics do not provide strong "
            "5%-level evidence of heteroskedasticity or serial "
            "correlation in this specification. HAC inference "
            "can still be retained as a conservative time-series "
            "robust covariance estimator, but it should not be "
            "claimed that these tests demonstrate a violation "
            "that they did not detect."
        )


print_asset_interpretation(
    btc_summary
)


print_asset_interpretation(
    eth_summary
)


# =====================================================================
# 27. SAVE OUTPUTS
# =====================================================================

section(
    "SAVING DIAGNOSTIC OUTPUTS"
)


formal_tests_file = (
    OUTPUT_DIR
    / "formal_residual_diagnostic_tests.csv"
)


model_summary_file = (
    OUTPUT_DIR
    / "residual_diagnostic_model_summary.csv"
)


residuals_file = (
    OUTPUT_DIR
    / "regression_residuals.csv"
)


acf_file = (
    OUTPUT_DIR
    / "residual_acf.csv"
)


inference_file = (
    OUTPUT_DIR
    / "ols_vs_hac_inference.csv"
)


formal_tests.to_csv(
    formal_tests_file,
    index=False
)


model_summary.to_csv(
    model_summary_file,
    index=False
)


all_residuals.to_csv(
    residuals_file,
    index=False
)


all_acf.to_csv(
    acf_file,
    index=False
)


all_inference.to_csv(
    inference_file,
    index=False
)


for filepath in [

    formal_tests_file,
    model_summary_file,
    residuals_file,
    acf_file,
    inference_file

]:

    print(
        "\nSaved:"
    )

    print(
        filepath
    )


# =====================================================================
# 28. SAVE METHODOLOGY NOTE
# =====================================================================

section(
    "SAVING METHODOLOGY NOTE"
)


methodology_file = (
    OUTPUT_DIR
    / "residual_diagnostics_methodology_note.txt"
)


methodology_note = f"""
FORMAL RESIDUAL DIAGNOSTICS

Purpose
-------
Formal residual diagnostics were conducted for the primary one-day
lag explanatory regressions for Bitcoin and Ethereum.

The diagnostics were performed using the same information-aligned
control variables employed in the main explanatory specifications.
This preserves the seven-day cryptocurrency calendar while preventing
the use of unavailable contemporaneous traditional-market information.

Heteroskedasticity
------------------
Residual heteroskedasticity was examined using the Breusch-Pagan and
White tests.

For both tests, the null hypothesis is homoskedasticity. A p-value
below 0.05 is therefore interpreted as evidence against constant
residual variance.

Serial correlation
------------------
Residual serial dependence was examined using Breusch-Godfrey LM
tests at lags 1, 3, and 7 and Ljung-Box Q tests at lags 1, 3, 7, 14,
and 30.

For these tests, the null hypothesis is the absence of residual serial
correlation through the specified lag horizon.

The Breusch-Godfrey test is treated as an important formal diagnostic
because the return equations include a lagged dependent variable.

Durbin-Watson
-------------
The Durbin-Watson statistic is reported as a supporting descriptive
diagnostic. Because the regression specifications include lagged
dependent variables, it is not treated as the principal formal test
of residual serial correlation.

Residual normality
------------------
The Jarque-Bera test is reported as a supplementary diagnostic.
Residual normality is not the central justification for HAC inference
in this large-sample time-series setting.

HAC/Newey-West inference
------------------------
The dissertation uses heteroskedasticity and autocorrelation
consistent (HAC/Newey-West) covariance estimates with a maximum lag
of {HAC_MAXLAGS}.

HAC estimation does not alter the OLS point estimates. Instead, it
adjusts the estimated covariance matrix and therefore the reported
standard errors, test statistics, confidence intervals, and p-values.

Evidence of heteroskedasticity and/or residual serial dependence
provides empirical support for using HAC inference rather than
conventional OLS standard errors.

Even when individual residual tests fail to reject their null
hypotheses, HAC inference can be retained as a conservative
time-series robust covariance estimator. However, the empirical
results should be reported accurately and the dissertation should
not claim that a diagnostic test detected heteroskedasticity or
autocorrelation when it did not.

Scope
-----
These diagnostics concern the in-sample explanatory regressions.
They should not be interpreted as evidence of out-of-sample
predictive performance.
""".strip()


with open(
    methodology_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        methodology_note
    )


print(
    "\nSaved:"
)

print(
    methodology_file
)


# =====================================================================
# 29. SAVE MODEL SPECIFICATION NOTE
# =====================================================================

section(
    "SAVING MODEL SPECIFICATION NOTE"
)


specification_file = (
    OUTPUT_DIR
    / "residual_diagnostics_model_specifications.txt"
)


specification_note = f"""
RESIDUAL DIAGNOSTIC MODEL SPECIFICATIONS

BTC dependent variable
----------------------
{BTC_DEPENDENT}

BTC predictors
--------------
{chr(10).join(BTC_PREDICTORS)}

ETH dependent variable
----------------------
{ETH_DEPENDENT}

ETH predictors
--------------
{chr(10).join(ETH_PREDICTORS)}

Covariance estimator used for dissertation inference
-----------------------------------------------------
HAC / Newey-West

Maximum HAC lag
---------------
{HAC_MAXLAGS}

Diagnostic significance level
-----------------------------
{SIGNIFICANCE_LEVEL}

Breusch-Godfrey lag horizons
----------------------------
{BG_LAGS}

Ljung-Box lag horizons
----------------------
{LJUNG_BOX_LAGS}

Residual ACF maximum lag
------------------------
{ACF_MAX_LAG}

Important
---------
The regression residuals are generated from the primary t-1
in-sample explanatory models. These diagnostics do not measure
out-of-sample forecasting performance.
""".strip()


with open(
    specification_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        specification_note
    )


print(
    "\nSaved:"
)

print(
    specification_file
)


# =====================================================================
# 30. FINAL VALIDATION
# =====================================================================

section(
    "FINAL VALIDATION"
)


validation_checks = {

    "BTC model estimated":
        (
            btc_ols.nobs > 0
        ),

    "ETH model estimated":
        (
            eth_ols.nobs > 0
        ),

    "BTC residual count equals model N":
        (
            len(
                btc_residuals
            )
            ==
            int(
                btc_ols.nobs
            )
        ),

    "ETH residual count equals model N":
        (
            len(
                eth_residuals
            )
            ==
            int(
                eth_ols.nobs
            )
        ),

    "BTC Breusch-Pagan completed":
        (
            not pd.isna(
                btc_bp[
                    "P_Value"
                ]
            )
        ),

    "ETH Breusch-Pagan completed":
        (
            not pd.isna(
                eth_bp[
                    "P_Value"
                ]
            )
        ),

    "BTC White test completed":
        (
            not pd.isna(
                btc_white[
                    "P_Value"
                ]
            )
        ),

    "ETH White test completed":
        (
            not pd.isna(
                eth_white[
                    "P_Value"
                ]
            )
        ),

    "BTC all BG tests completed":
        all(
            not pd.isna(
                result[
                    "P_Value"
                ]
            )
            for result in btc_bg
        ),

    "ETH all BG tests completed":
        all(
            not pd.isna(
                result[
                    "P_Value"
                ]
            )
            for result in eth_bg
        ),

    "BTC all Ljung-Box tests completed":
        all(
            not pd.isna(
                result[
                    "P_Value"
                ]
            )
            for result in btc_lb
        ),

    "ETH all Ljung-Box tests completed":
        all(
            not pd.isna(
                result[
                    "P_Value"
                ]
            )
            for result in eth_lb
        ),

    "BTC HAC coefficient equality":
        (
            btc_inference[
                "Coefficient_Difference"
            ]
            .abs()
            .max()
            <
            1e-12
        ),

    "ETH HAC coefficient equality":
        (
            eth_inference[
                "Coefficient_Difference"
            ]
            .abs()
            .max()
            <
            1e-12
        ),

    "Full dataset calendar continuous":
        (
            calendar_gaps == 0
        )

}


for check_name, condition in validation_checks.items():

    print(
        f"\n{check_name}: "
        f"{'PASS' if condition else 'FAIL'}"
    )


overall_validation = all(
    validation_checks.values()
)


print(
    "\n" + "-" * 80
)


print(
    "\nOVERALL RESIDUAL-DIAGNOSTIC VALIDATION:"
)

print(
    "PASS"
    if overall_validation
    else "FAIL"
)


if not overall_validation:

    raise ValueError(
        "\nOne or more residual-diagnostic validation "
        "checks failed."
    )


# =====================================================================
# 31. FINAL REMINDERS
# =====================================================================

section(
    "FINAL REMINDERS"
)


print(
    "\n1. These diagnostics apply to the primary t-1 "
    "IN-SAMPLE EXPLANATORY regressions."
)


print(
    "\n2. They do NOT establish out-of-sample predictive power."
)


print(
    "\n3. Breusch-Pagan and White diagnose heteroskedasticity."
)


print(
    "\n4. Breusch-Godfrey and Ljung-Box diagnose "
    "residual serial dependence."
)


print(
    "\n5. Durbin-Watson is supporting evidence only because "
    "the models contain lagged dependent variables."
)


print(
    "\n6. Jarque-Bera is supplementary; residual normality "
    "is not the central justification for HAC inference."
)


print(
    "\n7. HAC/Newey-West changes inference, not OLS "
    "coefficient point estimates."
)


print(
    "\n8. Report the actual diagnostic results. Do not claim "
    "heteroskedasticity or serial correlation unless the "
    "corresponding tests support that conclusion."
)


section(
    "RESIDUAL TIME-SERIES DIAGNOSTICS COMPLETE"
)