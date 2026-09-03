# =====================================================================
# 03_cross_crypto_robustness_models.py
#
# FINAL CROSS-CRYPTO ROBUSTNESS EXPLANATORY MODELS
#
# Dissertation:
# Do Social Media Sentiment Signals Improve the Prediction of
# Cryptocurrency Returns Beyond Traditional Market Indicators?
# Evidence from Bitcoin and Ethereum
#
# PURPOSE
# ---------------------------------------------------------------------
# Test whether the explanatory benchmark specifications are robust
# to controlling for lagged returns of the other cryptocurrency.
#
# BTC robustness:
#   BTC baseline controls + ETH_Lagged_Return
#
# ETH robustness:
#   ETH baseline controls + BTC_Lagged_Return
#
# IMPORTANT TIMING DESIGN
# ---------------------------------------------------------------------
# Cryptocurrency trades seven days per week, whereas traditional
# financial markets do not.
#
# Therefore, traditional-market predictors use the information-aligned
# variables constructed in final_forecast_dataset.csv:
#
#   Lagged_SP500_Return_Aligned
#   Lagged_VIX_Change_Aligned
#   Lagged_Gold_Return_Aligned
#   Lagged_DXY_Return_Aligned
#   Lagged_US10Y_Change_Aligned
#
# These represent the most recently available traditional-market
# information strictly prior to the cryptocurrency return date.
#
# ESTIMATION
# ---------------------------------------------------------------------
# OLS with HAC / Newey-West standard errors using 7 lags.
#
# These are IN-SAMPLE EXPLANATORY robustness models.
# They are NOT out-of-sample forecasting models.
# =====================================================================


from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm


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
    / "cross_crypto_explanatory"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =====================================================================
# 2. INPUT FILE
# =====================================================================

INPUT_FILE = (
    DATA_PROCESSED
    / "final_forecast_dataset.csv"
)


# =====================================================================
# 3. HAC SETTING
# =====================================================================

HAC_MAXLAGS = 7


# =====================================================================
# 4. HELPER
# =====================================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =====================================================================
# 5. START
# =====================================================================

section(
    "FINAL CROSS-CRYPTO ROBUSTNESS EXPLANATORY MODELS"
)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}"
    )


# =====================================================================
# 6. IMPORT DATA
# =====================================================================

section(
    "IMPORTING FINAL INFORMATION-ALIGNED DATASET"
)

df = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(df.shape)


print("\nColumns found:")

for column in df.columns:
    print(" -", column)


# =====================================================================
# 7. DATE VALIDATION
# =====================================================================

section(
    "VALIDATING DATES"
)


if "Date" not in df.columns:

    raise KeyError(
        "Date column not found."
    )


df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)


invalid_dates = int(
    df["Date"].isna().sum()
)

duplicate_dates = int(
    df["Date"].duplicated().sum()
)


print("\nInvalid dates:")
print(invalid_dates)

print("\nDuplicate dates:")
print(duplicate_dates)


if invalid_dates > 0:

    raise ValueError(
        "Invalid dates were found."
    )


if duplicate_dates > 0:

    raise ValueError(
        "Duplicate dates were found."
    )


df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


print("\nDataset date range:")

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =====================================================================
# 8. CALENDAR VARIABLES
# =====================================================================

df["Day_Name"] = (
    df["Date"]
    .dt.day_name()
)


df["Is_Weekend"] = (
    df["Date"]
    .dt.dayofweek >= 5
)


DAY_ORDER = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday"
]


# =====================================================================
# 9. CALENDAR DIAGNOSTICS
# =====================================================================

section(
    "CALENDAR / WEEKEND DIAGNOSTICS"
)


print("\nTotal observations:")
print(len(df))


print("\nWeekday observations:")

print(
    int(
        (~df["Is_Weekend"]).sum()
    )
)


print("\nWeekend observations:")

print(
    int(
        df["Is_Weekend"].sum()
    )
)


print("\nObservations by day of week:")

print(
    df["Day_Name"]
    .value_counts()
    .reindex(DAY_ORDER)
)


# =====================================================================
# 10. VARIABLE DEFINITIONS
# =====================================================================

BTC_DEPENDENT = "BTC_Return"
ETH_DEPENDENT = "ETH_Return"


# ---------------------------------------------------------------------
# INFORMATION-ALIGNED TRADITIONAL-MARKET CONTROLS
# ---------------------------------------------------------------------
#
# THESE ARE THE CORRECT VARIABLES.
#
# Do not substitute the old non-aligned columns.
# ---------------------------------------------------------------------

ALIGNED_MARKET_CONTROLS = [

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


# ---------------------------------------------------------------------
# BTC BASELINE
# ---------------------------------------------------------------------

BTC_BASELINE_CONTROLS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume",

] + ALIGNED_MARKET_CONTROLS


# ---------------------------------------------------------------------
# BTC CROSS-CRYPTO ROBUSTNESS
# ---------------------------------------------------------------------

BTC_ROBUSTNESS_CONTROLS = [

    "BTC_Lagged_Return",

    "ETH_Lagged_Return",

    "Lagged_Log_BTC_Volume",

] + ALIGNED_MARKET_CONTROLS


# ---------------------------------------------------------------------
# ETH BASELINE
# ---------------------------------------------------------------------

ETH_BASELINE_CONTROLS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume",

] + ALIGNED_MARKET_CONTROLS


# ---------------------------------------------------------------------
# ETH CROSS-CRYPTO ROBUSTNESS
# ---------------------------------------------------------------------

ETH_ROBUSTNESS_CONTROLS = [

    "ETH_Lagged_Return",

    "BTC_Lagged_Return",

    "Lagged_Log_ETH_Volume",

] + ALIGNED_MARKET_CONTROLS


# =====================================================================
# 11. SAFETY CHECK:
#     PREVENT ACCIDENTAL USE OF OLD NON-ALIGNED VARIABLES
# =====================================================================

section(
    "VERIFYING INFORMATION-ALIGNED SPECIFICATION"
)


OLD_NON_ALIGNED_CONTROLS = [

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"

]


all_model_predictors = list(
    dict.fromkeys(

        BTC_ROBUSTNESS_CONTROLS
        +
        ETH_ROBUSTNESS_CONTROLS

    )
)


incorrect_variables_used = [

    variable

    for variable
    in OLD_NON_ALIGNED_CONTROLS

    if variable
    in all_model_predictors

]


if incorrect_variables_used:

    raise ValueError(

        "\nERROR: Non-aligned traditional-market variables "
        "were included in the model:\n"

        +

        "\n".join(
            incorrect_variables_used
        )

    )


print(
    "\nPASS: No old non-aligned traditional-market "
    "variables are used."
)


print(
    "\nAligned traditional-market controls:"
)

for variable in ALIGNED_MARKET_CONTROLS:

    print(
        " -",
        variable
    )


# =====================================================================
# 12. CHECK REQUIRED VARIABLES
# =====================================================================

section(
    "CHECKING REQUIRED VARIABLES"
)


required_variables = list(
    dict.fromkeys(

        [
            BTC_DEPENDENT,
            ETH_DEPENDENT
        ]

        +

        BTC_ROBUSTNESS_CONTROLS

        +

        ETH_ROBUSTNESS_CONTROLS

    )
)


missing_variables = []


for variable in required_variables:

    if variable in df.columns:

        print(
            f"{variable}: FOUND"
        )

    else:

        print(
            f"{variable}: MISSING"
        )

        missing_variables.append(
            variable
        )


if missing_variables:

    raise KeyError(

        "\nRequired variables missing:\n"

        +

        "\n".join(
            missing_variables
        )

    )


print(
    "\nAll required variables are available."
)


# =====================================================================
# 13. NUMERIC CONVERSION
# =====================================================================

section(
    "VALIDATING NUMERIC VARIABLES"
)


for variable in required_variables:

    df[variable] = pd.to_numeric(
        df[variable],
        errors="coerce"
    )

    df[variable] = (
        df[variable]
        .replace(
            [np.inf, -np.inf],
            np.nan
        )
    )


    print(
        f"{variable}: "
        f"missing = "
        f"{df[variable].isna().sum()}"
    )


# =====================================================================
# 14. ALIGNED-CONTROL MISSINGNESS
# =====================================================================

section(
    "ALIGNED MARKET-CONTROL MISSINGNESS"
)


missing_rows = []


for variable in required_variables:

    total_missing = int(
        df[variable]
        .isna()
        .sum()
    )


    weekday_missing = int(

        df.loc[
            ~df["Is_Weekend"],
            variable
        ]

        .isna()
        .sum()

    )


    weekend_missing = int(

        df.loc[
            df["Is_Weekend"],
            variable
        ]

        .isna()
        .sum()

    )


    missing_rows.append(

        {

            "Variable":
                variable,

            "Total_Observations":
                len(df),

            "Missing_Total":
                total_missing,

            "Missing_Percent":
                (
                    total_missing
                    /
                    len(df)
                    *
                    100
                ),

            "Missing_Weekday":
                weekday_missing,

            "Missing_Weekend":
                weekend_missing

        }

    )


missing_diagnostics = pd.DataFrame(
    missing_rows
)


print(
    "\n",
    missing_diagnostics
    .to_string(index=False)
)


# =====================================================================
# 15. SPECIFICATION DISPLAY
# =====================================================================

section(
    "FINAL MODEL SPECIFICATIONS"
)


print("\nBTC BASELINE")

print(
    "Dependent variable:",
    BTC_DEPENDENT
)

for variable in BTC_BASELINE_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nBTC CROSS-CRYPTO ROBUSTNESS"
)

print(
    "Dependent variable:",
    BTC_DEPENDENT
)

for variable in BTC_ROBUSTNESS_CONTROLS:

    print(
        " -",
        variable
    )


print("\nETH BASELINE")

print(
    "Dependent variable:",
    ETH_DEPENDENT
)

for variable in ETH_BASELINE_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nETH CROSS-CRYPTO ROBUSTNESS"
)

print(
    "Dependent variable:",
    ETH_DEPENDENT
)

for variable in ETH_ROBUSTNESS_CONTROLS:

    print(
        " -",
        variable
    )


# =====================================================================
# 16. CREATE COMMON BTC SAMPLE
# =====================================================================
#
# Complete cases are determined using the FULL BTC robustness
# specification.
#
# Therefore BTC baseline and BTC robustness are estimated on
# identical observations.
# =====================================================================

section(
    "CREATING COMMON BTC REGRESSION SAMPLE"
)


btc_sample_columns = list(
    dict.fromkeys(

        [
            "Date",
            "Day_Name",
            "Is_Weekend",
            BTC_DEPENDENT
        ]

        +

        BTC_ROBUSTNESS_CONTROLS

    )
)


btc_sample = (

    df[
        btc_sample_columns
    ]

    .dropna(
        subset=[

            BTC_DEPENDENT

        ]

        +

        BTC_ROBUSTNESS_CONTROLS
    )

    .copy()

)


if btc_sample.empty:

    raise ValueError(
        "No usable BTC regression observations."
    )


print("\nBTC observations:")
print(len(btc_sample))


print("\nBTC regression date range:")

print(
    btc_sample["Date"].min(),
    "to",
    btc_sample["Date"].max()
)


print("\nBTC weekday observations:")

print(
    int(
        (~btc_sample["Is_Weekend"])
        .sum()
    )
)


print("\nBTC weekend observations:")

print(
    int(
        btc_sample["Is_Weekend"]
        .sum()
    )
)


print("\nBTC observations by day:")

print(
    btc_sample["Day_Name"]
    .value_counts()
    .reindex(DAY_ORDER)
)


# =====================================================================
# 17. CREATE COMMON ETH SAMPLE
# =====================================================================

section(
    "CREATING COMMON ETH REGRESSION SAMPLE"
)


eth_sample_columns = list(
    dict.fromkeys(

        [
            "Date",
            "Day_Name",
            "Is_Weekend",
            ETH_DEPENDENT
        ]

        +

        ETH_ROBUSTNESS_CONTROLS

    )
)


eth_sample = (

    df[
        eth_sample_columns
    ]

    .dropna(
        subset=[

            ETH_DEPENDENT

        ]

        +

        ETH_ROBUSTNESS_CONTROLS
    )

    .copy()

)


if eth_sample.empty:

    raise ValueError(
        "No usable ETH regression observations."
    )


print("\nETH observations:")
print(len(eth_sample))


print("\nETH regression date range:")

print(
    eth_sample["Date"].min(),
    "to",
    eth_sample["Date"].max()
)


print("\nETH weekday observations:")

print(
    int(
        (~eth_sample["Is_Weekend"])
        .sum()
    )
)


print("\nETH weekend observations:")

print(
    int(
        eth_sample["Is_Weekend"]
        .sum()
    )
)


print("\nETH observations by day:")

print(
    eth_sample["Day_Name"]
    .value_counts()
    .reindex(DAY_ORDER)
)


# =====================================================================
# 18. IMPORTANT WEEKEND VALIDATION
# =====================================================================

section(
    "WEEKEND RETENTION VALIDATION"
)


btc_weekends = int(
    btc_sample["Is_Weekend"]
    .sum()
)


eth_weekends = int(
    eth_sample["Is_Weekend"]
    .sum()
)


print(
    "\nBTC weekend observations retained:"
)

print(
    btc_weekends
)


print(
    "\nETH weekend observations retained:"
)

print(
    eth_weekends
)


if btc_weekends == 0:

    print(
        "\nWARNING:"
        "\nBTC still contains no weekend regression observations."
        "\nInspect the aligned variables before treating the "
        "results as final."
    )


if eth_weekends == 0:

    print(
        "\nWARNING:"
        "\nETH still contains no weekend regression observations."
        "\nInspect the aligned variables before treating the "
        "results as final."
    )


if (
    btc_weekends > 0
    and
    eth_weekends > 0
):

    print(
        "\nPASS:"
        "\nWeekend cryptocurrency observations are retained "
        "after information alignment."
    )


# =====================================================================
# 19. SAMPLE COMPARABILITY
# =====================================================================

section(
    "SAMPLE COMPARABILITY CHECK"
)


print("\nBTC baseline observations:")
print(len(btc_sample))


print("\nBTC robustness observations:")
print(len(btc_sample))


print(
    "\nBTC baseline and robustness "
    "use identical observations:"
)

print(True)


print("\nETH baseline observations:")
print(len(eth_sample))


print("\nETH robustness observations:")
print(len(eth_sample))


print(
    "\nETH baseline and robustness "
    "use identical observations:"
)

print(True)


# =====================================================================
# 20. HAC MODEL FUNCTION
# =====================================================================

def estimate_hac_model(
    data,
    dependent,
    predictors,
    model_name
):

    """
    Estimate OLS coefficients with HAC/Newey-West
    standard errors.

    HAC affects statistical inference, not the OLS
    coefficient estimates themselves.
    """

    y = (
        data[dependent]
        .astype(float)
    )


    X = (
        data[predictors]
        .astype(float)
    )


    X = sm.add_constant(
        X,
        has_constant="add"
    )


    model = (

        sm.OLS(
            y,
            X
        )

        .fit(

            cov_type="HAC",

            cov_kwds={

                "maxlags":
                    HAC_MAXLAGS,

                "use_correction":
                    True

            }

        )

    )


    print(
        "\n" + "-" * 80
    )

    print(model_name)

    print(
        "-" * 80
    )

    print(
        model.summary()
    )


    return model


# =====================================================================
# 21. BTC BASELINE
# =====================================================================

section(
    "BTC BASELINE EXPLANATORY MODEL"
)


btc_baseline = estimate_hac_model(

    data=
        btc_sample,

    dependent=
        BTC_DEPENDENT,

    predictors=
        BTC_BASELINE_CONTROLS,

    model_name=
        "BTC BASELINE EXPLANATORY MODEL "
        "(INFORMATION-ALIGNED; HAC / NEWEY-WEST)"

)


# =====================================================================
# 22. BTC CROSS-CRYPTO ROBUSTNESS
# =====================================================================

section(
    "BTC CROSS-CRYPTO ROBUSTNESS MODEL"
)


btc_cross = estimate_hac_model(

    data=
        btc_sample,

    dependent=
        BTC_DEPENDENT,

    predictors=
        BTC_ROBUSTNESS_CONTROLS,

    model_name=
        "BTC CROSS-CRYPTO ROBUSTNESS MODEL "
        "(ADDS ETH_Lagged_Return)"

)


# =====================================================================
# 23. ETH BASELINE
# =====================================================================

section(
    "ETH BASELINE EXPLANATORY MODEL"
)


eth_baseline = estimate_hac_model(

    data=
        eth_sample,

    dependent=
        ETH_DEPENDENT,

    predictors=
        ETH_BASELINE_CONTROLS,

    model_name=
        "ETH BASELINE EXPLANATORY MODEL "
        "(INFORMATION-ALIGNED; HAC / NEWEY-WEST)"

)


# =====================================================================
# 24. ETH CROSS-CRYPTO ROBUSTNESS
# =====================================================================

section(
    "ETH CROSS-CRYPTO ROBUSTNESS MODEL"
)


eth_cross = estimate_hac_model(

    data=
        eth_sample,

    dependent=
        ETH_DEPENDENT,

    predictors=
        ETH_ROBUSTNESS_CONTROLS,

    model_name=
        "ETH CROSS-CRYPTO ROBUSTNESS MODEL "
        "(ADDS BTC_Lagged_Return)"

)


# =====================================================================
# 25. COEFFICIENT TABLE FUNCTION
# =====================================================================

def coefficient_table(
    model,
    model_name,
    asset
):

    confidence = (
        model
        .conf_int(alpha=0.05)
    )


    table = pd.DataFrame(

        {

            "Variable":
                model.params.index,

            "Coefficient":
                model.params.values,

            "HAC_Std_Error":
                model.bse.values,

            "z_statistic":
                model.tvalues.values,

            "p_value":
                model.pvalues.values,

            "CI_95_Lower":
                confidence.iloc[:, 0].values,

            "CI_95_Upper":
                confidence.iloc[:, 1].values

        }

    )


    table.insert(
        0,
        "Model",
        model_name
    )


    table.insert(
        0,
        "Asset",
        asset
    )


    return table


# =====================================================================
# 26. CREATE COEFFICIENT TABLES
# =====================================================================

section(
    "CREATING COEFFICIENT TABLES"
)


btc_baseline_coefficients = coefficient_table(
    btc_baseline,
    "Baseline",
    "BTC"
)


btc_cross_coefficients = coefficient_table(
    btc_cross,
    "Cross_Crypto",
    "BTC"
)


eth_baseline_coefficients = coefficient_table(
    eth_baseline,
    "Baseline",
    "ETH"
)


eth_cross_coefficients = coefficient_table(
    eth_cross,
    "Cross_Crypto",
    "ETH"
)


all_coefficients = pd.concat(

    [

        btc_baseline_coefficients,

        btc_cross_coefficients,

        eth_baseline_coefficients,

        eth_cross_coefficients

    ],

    ignore_index=True

)


# =====================================================================
# 27. MODEL COMPARISON
# =====================================================================

section(
    "BASELINE VS CROSS-CRYPTO MODEL COMPARISON"
)


model_comparison = pd.DataFrame(

    {

        "Asset": [
            "BTC",
            "BTC",
            "ETH",
            "ETH"
        ],

        "Model": [
            "Baseline",
            "Cross_Crypto",
            "Baseline",
            "Cross_Crypto"
        ],

        "N": [
            int(btc_baseline.nobs),
            int(btc_cross.nobs),
            int(eth_baseline.nobs),
            int(eth_cross.nobs)
        ],

        "R_squared": [
            btc_baseline.rsquared,
            btc_cross.rsquared,
            eth_baseline.rsquared,
            eth_cross.rsquared
        ],

        "Adjusted_R_squared": [
            btc_baseline.rsquared_adj,
            btc_cross.rsquared_adj,
            eth_baseline.rsquared_adj,
            eth_cross.rsquared_adj
        ],

        "AIC": [
            btc_baseline.aic,
            btc_cross.aic,
            eth_baseline.aic,
            eth_cross.aic
        ],

        "BIC": [
            btc_baseline.bic,
            btc_cross.bic,
            eth_baseline.bic,
            eth_cross.bic
        ]

    }

)


print(
    "\n",
    model_comparison
    .to_string(index=False)
)


# =====================================================================
# 28. KEY CROSS-CRYPTO RESULTS
# =====================================================================

section(
    "KEY CROSS-CRYPTO RESULTS"
)


BTC_CROSS_VARIABLE = (
    "ETH_Lagged_Return"
)


ETH_CROSS_VARIABLE = (
    "BTC_Lagged_Return"
)


btc_ci = (
    btc_cross
    .conf_int()
    .loc[
        BTC_CROSS_VARIABLE
    ]
)


eth_ci = (
    eth_cross
    .conf_int()
    .loc[
        ETH_CROSS_VARIABLE
    ]
)


key_results = pd.DataFrame(

    {

        "Asset": [
            "BTC",
            "ETH"
        ],

        "Dependent_Variable": [
            "BTC_Return",
            "ETH_Return"
        ],

        "Cross_Crypto_Variable": [
            BTC_CROSS_VARIABLE,
            ETH_CROSS_VARIABLE
        ],

        "Coefficient": [

            btc_cross.params[
                BTC_CROSS_VARIABLE
            ],

            eth_cross.params[
                ETH_CROSS_VARIABLE
            ]

        ],

        "HAC_Std_Error": [

            btc_cross.bse[
                BTC_CROSS_VARIABLE
            ],

            eth_cross.bse[
                ETH_CROSS_VARIABLE
            ]

        ],

        "z_statistic": [

            btc_cross.tvalues[
                BTC_CROSS_VARIABLE
            ],

            eth_cross.tvalues[
                ETH_CROSS_VARIABLE
            ]

        ],

        "p_value": [

            btc_cross.pvalues[
                BTC_CROSS_VARIABLE
            ],

            eth_cross.pvalues[
                ETH_CROSS_VARIABLE
            ]

        ],

        "CI_95_Lower": [
            btc_ci.iloc[0],
            eth_ci.iloc[0]
        ],

        "CI_95_Upper": [
            btc_ci.iloc[1],
            eth_ci.iloc[1]
        ]

    }

)


print(
    "\n",
    key_results
    .to_string(index=False)
)


# =====================================================================
# 29. INCREMENTAL EXPLANATORY POWER
# =====================================================================

section(
    "INCREMENTAL EXPLANATORY POWER"
)


incremental_results = pd.DataFrame(

    {

        "Asset": [
            "BTC",
            "ETH"
        ],

        "Baseline_R2": [
            btc_baseline.rsquared,
            eth_baseline.rsquared
        ],

        "Cross_Crypto_R2": [
            btc_cross.rsquared,
            eth_cross.rsquared
        ],

        "Change_in_R2": [

            (
                btc_cross.rsquared
                -
                btc_baseline.rsquared
            ),

            (
                eth_cross.rsquared
                -
                eth_baseline.rsquared
            )

        ],

        "Baseline_Adjusted_R2": [
            btc_baseline.rsquared_adj,
            eth_baseline.rsquared_adj
        ],

        "Cross_Crypto_Adjusted_R2": [
            btc_cross.rsquared_adj,
            eth_cross.rsquared_adj
        ],

        "Change_in_Adjusted_R2": [

            (
                btc_cross.rsquared_adj
                -
                btc_baseline.rsquared_adj
            ),

            (
                eth_cross.rsquared_adj
                -
                eth_baseline.rsquared_adj
            )

        ]

    }

)


print(
    "\n",
    incremental_results
    .to_string(index=False)
)


# =====================================================================
# 30. SIGNIFICANCE LABEL
# =====================================================================

def significance_label(p_value):

    if p_value < 0.01:

        return "Significant at 1%"

    elif p_value < 0.05:

        return "Significant at 5%"

    elif p_value < 0.10:

        return "Significant at 10%"

    else:

        return (
            "Not statistically significant "
            "at conventional levels"
        )


# =====================================================================
# 31. SIGNIFICANCE CHECK
# =====================================================================

section(
    "CROSS-CRYPTO SIGNIFICANCE CHECK"
)


btc_beta = float(

    btc_cross.params[
        BTC_CROSS_VARIABLE
    ]

)


btc_p = float(

    btc_cross.pvalues[
        BTC_CROSS_VARIABLE
    ]

)


eth_beta = float(

    eth_cross.params[
        ETH_CROSS_VARIABLE
    ]

)


eth_p = float(

    eth_cross.pvalues[
        ETH_CROSS_VARIABLE
    ]

)


print("\nBTC robustness model")

print(
    "ETH_Lagged_Return coefficient:",
    btc_beta
)

print(
    "HAC p-value:",
    btc_p
)

print(
    significance_label(
        btc_p
    )
)


print("\nETH robustness model")

print(
    "BTC_Lagged_Return coefficient:",
    eth_beta
)

print(
    "HAC p-value:",
    eth_p
)

print(
    significance_label(
        eth_p
    )
)


# =====================================================================
# 32. SAMPLE DIAGNOSTICS
# =====================================================================

section(
    "SAMPLE DIAGNOSTICS"
)


sample_diagnostics = pd.DataFrame(

    {

        "Asset": [
            "BTC",
            "ETH"
        ],

        "N_Regression_Observations": [
            len(btc_sample),
            len(eth_sample)
        ],

        "Start_Date": [
            btc_sample["Date"].min(),
            eth_sample["Date"].min()
        ],

        "End_Date": [
            btc_sample["Date"].max(),
            eth_sample["Date"].max()
        ],

        "Weekday_Observations": [

            int(
                (~btc_sample["Is_Weekend"])
                .sum()
            ),

            int(
                (~eth_sample["Is_Weekend"])
                .sum()
            )

        ],

        "Weekend_Observations": [

            int(
                btc_sample["Is_Weekend"]
                .sum()
            ),

            int(
                eth_sample["Is_Weekend"]
                .sum()
            )

        ],

        "HAC_Max_Lags": [
            HAC_MAXLAGS,
            HAC_MAXLAGS
        ]

    }

)


print(
    "\n",
    sample_diagnostics
    .to_string(index=False)
)


# =====================================================================
# 33. SAVE RESULTS
# =====================================================================

section(
    "SAVING CSV RESULTS"
)


coefficients_file = (
    OUTPUT_DIR
    / "cross_crypto_robustness_coefficients.csv"
)


comparison_file = (
    OUTPUT_DIR
    / "cross_crypto_model_comparison.csv"
)


key_results_file = (
    OUTPUT_DIR
    / "cross_crypto_key_results.csv"
)


incremental_file = (
    OUTPUT_DIR
    / "cross_crypto_incremental_explanatory_power.csv"
)


missing_file = (
    OUTPUT_DIR
    / "cross_crypto_missing_value_diagnostics.csv"
)


sample_file = (
    OUTPUT_DIR
    / "cross_crypto_sample_diagnostics.csv"
)


all_coefficients.to_csv(
    coefficients_file,
    index=False
)


model_comparison.to_csv(
    comparison_file,
    index=False
)


key_results.to_csv(
    key_results_file,
    index=False
)


incremental_results.to_csv(
    incremental_file,
    index=False
)


missing_diagnostics.to_csv(
    missing_file,
    index=False
)


sample_diagnostics.to_csv(
    sample_file,
    index=False
)


for output_file in [

    coefficients_file,
    comparison_file,
    key_results_file,
    incremental_file,
    missing_file,
    sample_file

]:

    print("\nSaved:")
    print(output_file)


# =====================================================================
# 34. SAVE FULL REGRESSION SUMMARIES
# =====================================================================

section(
    "SAVING FULL REGRESSION SUMMARIES"
)


summary_files = {

    "btc_baseline_same_sample.txt":
        btc_baseline,

    "btc_cross_crypto_robustness.txt":
        btc_cross,

    "eth_baseline_same_sample.txt":
        eth_baseline,

    "eth_cross_crypto_robustness.txt":
        eth_cross

}


for filename, model in summary_files.items():

    output_path = (
        OUTPUT_DIR
        / filename
    )


    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            model
            .summary()
            .as_text()
        )


    print("\nSaved:")
    print(output_path)


# =====================================================================
# 35. FINAL INTERPRETATION
# =====================================================================

section(
    "FINAL INTERPRETATION CHECK"
)


print("\nBTC:")


if btc_p < 0.05:

    print(
        "Lagged ETH returns are statistically significant "
        "in the BTC explanatory robustness model "
        "at the 5% level."
    )

else:

    print(
        "Lagged ETH returns are NOT statistically significant "
        "in the BTC explanatory robustness model "
        "at the 5% level."
    )


print("\nETH:")


if eth_p < 0.05:

    print(
        "Lagged BTC returns are statistically significant "
        "in the ETH explanatory robustness model "
        "at the 5% level."
    )

else:

    print(
        "Lagged BTC returns are NOT statistically significant "
        "in the ETH explanatory robustness model "
        "at the 5% level."
    )


# =====================================================================
# 36. FINAL VALIDATION
# =====================================================================

section(
    "FINAL VALIDATION"
)


print("\nInput dataset:")
print(INPUT_FILE.name)


print(
    "\nTraditional-market specification:"
)

print(
    "INFORMATION-ALIGNED VARIABLES"
)


print(
    "\nHAC / Newey-West maximum lags:"
)

print(
    HAC_MAXLAGS
)


print(
    "\nBTC baseline observations:"
)

print(
    int(
        btc_baseline.nobs
    )
)


print(
    "\nBTC robustness observations:"
)

print(
    int(
        btc_cross.nobs
    )
)


print(
    "\nETH baseline observations:"
)

print(
    int(
        eth_baseline.nobs
    )
)


print(
    "\nETH robustness observations:"
)

print(
    int(
        eth_cross.nobs
    )
)


print(
    "\nBTC weekend observations:"
)

print(
    btc_weekends
)


print(
    "\nETH weekend observations:"
)

print(
    eth_weekends
)


print("\nIMPORTANT:")

print(
    "These regressions are IN-SAMPLE explanatory "
    "robustness models."
)

print(
    "They do NOT establish out-of-sample "
    "forecasting performance."
)

print(
    "Forecasting performance is evaluated separately "
    "using expanding-window one-step-ahead forecasts."
)


section(
    "FINAL CROSS-CRYPTO ROBUSTNESS EXPLANATORY ANALYSIS COMPLETE"
)