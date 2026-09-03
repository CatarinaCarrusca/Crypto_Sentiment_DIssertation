# =====================================================================
# 10_economic_significance_framework.py
#
# ECONOMIC-SIGNIFICANCE FRAMEWORK
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
# Translate regression coefficients into economically interpretable
# magnitudes rather than relying only on statistical significance.
#
# For each continuous predictor X:
#
#       1-SD Effect = beta_X * SD(X)
#
# Interpretation:
#
# "A one-standard-deviation increase in X is associated with an
# estimated beta_X * SD(X) change in expected daily cryptocurrency
# log return, holding the other regressors constant."
#
# The script reports the effect as:
#
#   1. daily log-return units;
#   2. approximate percentage points;
#   3. approximate basis points;
#   4. exact simple-return percentage-point equivalent:
#
#          100 * [exp(effect) - 1]
#
# ---------------------------------------------------------------------
# IMPORTANT METHODOLOGICAL POINTS
# ---------------------------------------------------------------------
#
# 1. This is an EXPLANATORY economic-significance exercise.
#
# 2. It does NOT establish out-of-sample predictive performance.
#
# 3. Coefficient inference uses HAC/Newey-West standard errors with
#    maximum lag 7, consistent with the dissertation's residual
#    diagnostics.
#
# 4. The economic magnitude itself is based on the estimated OLS
#    coefficient and the sample SD of the predictor.
#
# 5. HAC changes standard errors / confidence intervals / p-values,
#    NOT the OLS coefficient.
#
# 6. Predictor SDs are calculated on the EXACT estimation sample
#    used for each regression.
#
# 7. BTC and ETH are estimated separately.
#
# 8. Only the corrected information-aligned traditional-market
#    controls are used.
#
# 9. Once Reddit data arrive, sentiment can be added to the model
#    specification and the same framework can be used directly.
#
# =====================================================================


from pathlib import Path
import warnings
import math

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

ANALYSIS_OUTPUTS = (
    PROJECT_ROOT
    / "analysis_outputs"
)

OUTPUT_DIR = (
    ANALYSIS_OUTPUTS
    / "economic_significance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_DATASET_FILE = (
    DATA_PROCESSED
    / "final_forecast_dataset.csv"
)


# =====================================================================
# 2. SETTINGS
# =====================================================================

HAC_MAXLAGS = 7

CONFIDENCE_LEVEL = 0.95

ALPHA = (
    1.0
    -
    CONFIDENCE_LEVEL
)


# =====================================================================
# 3. PRIMARY BASELINE MODEL SPECIFICATIONS
# =====================================================================
#
# These match the corrected information-aligned baseline explanatory
# models.
#
# =====================================================================

MODEL_SPECS = {

    "BTC": {

        "dependent":
            "BTC_Return",

        "predictors": [

            "BTC_Lagged_Return",

            "Lagged_Log_BTC_Volume",

            "Lagged_SP500_Return_Aligned",

            "Lagged_VIX_Change_Aligned",

            "Lagged_Gold_Return_Aligned",

            "Lagged_DXY_Return_Aligned",

            "Lagged_US10Y_Change_Aligned"

        ]

    },

    "ETH": {

        "dependent":
            "ETH_Return",

        "predictors": [

            "ETH_Lagged_Return",

            "Lagged_Log_ETH_Volume",

            "Lagged_SP500_Return_Aligned",

            "Lagged_VIX_Change_Aligned",

            "Lagged_Gold_Return_Aligned",

            "Lagged_DXY_Return_Aligned",

            "Lagged_US10Y_Change_Aligned"

        ]

    }

}


# =====================================================================
# 4. FUTURE REDDIT VARIABLES
# =====================================================================
#
# These are NOT required yet.
#
# When Reddit data arrive, you can add the actual variable names here.
#
# Example eventual variables might be:
#
#   BTC:
#       Lagged_BTC_Reddit_Sentiment
#       Lagged_Log_BTC_Reddit_Post_Count
#
#   ETH:
#       Lagged_ETH_Reddit_Sentiment
#       Lagged_Log_ETH_Reddit_Post_Count
#
# IMPORTANT:
#
# Do not create fake variables now.
#
# The script simply reports whether any configured future sentiment
# variables are available.
#
# =====================================================================

FUTURE_REDDIT_VARIABLES = {

    "BTC": [

        "Lagged_BTC_Reddit_Sentiment",

        "Lagged_Log_BTC_Reddit_Post_Count"

    ],

    "ETH": [

        "Lagged_ETH_Reddit_Sentiment",

        "Lagged_Log_ETH_Reddit_Post_Count"

    ]

}


# =====================================================================
# 5. OLD NON-ALIGNED CONTROLS
# =====================================================================
#
# These columns may still exist in final_forecast_dataset.csv.
#
# Their presence in the dataset is not itself an error.
#
# They must simply NOT be used as predictors.
#
# =====================================================================

OLD_NONALIGNED_CONTROLS = [

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"

]


# =====================================================================
# 6. FRIENDLY VARIABLE LABELS
# =====================================================================

VARIABLE_LABELS = {

    "BTC_Lagged_Return":
        "Bitcoin lagged return",

    "ETH_Lagged_Return":
        "Ethereum lagged return",

    "Lagged_Log_BTC_Volume":
        "Bitcoin lagged log trading volume",

    "Lagged_Log_ETH_Volume":
        "Ethereum lagged log trading volume",

    "Lagged_SP500_Return_Aligned":
        "Lagged S&P 500 return",

    "Lagged_VIX_Change_Aligned":
        "Lagged VIX change",

    "Lagged_Gold_Return_Aligned":
        "Lagged gold futures return",

    "Lagged_DXY_Return_Aligned":
        "Lagged US Dollar Index return",

    "Lagged_US10Y_Change_Aligned":
        "Lagged US 10-year Treasury yield change",

    "Lagged_BTC_Reddit_Sentiment":
        "Lagged Bitcoin Reddit sentiment",

    "Lagged_ETH_Reddit_Sentiment":
        "Lagged Ethereum Reddit sentiment",

    "Lagged_Log_BTC_Reddit_Post_Count":
        "Lagged Bitcoin Reddit activity",

    "Lagged_Log_ETH_Reddit_Post_Count":
        "Lagged Ethereum Reddit activity"

}


# =====================================================================
# 7. HELPER FUNCTIONS
# =====================================================================


def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def safe_float(value):

    try:

        return float(value)

    except Exception:

        return np.nan


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


def variable_label(variable):

    return VARIABLE_LABELS.get(
        variable,
        variable
    )


# =====================================================================
# 8. RETURN-UNIT CONVERSION FUNCTIONS
# =====================================================================


def log_return_to_approx_percentage_points(
    log_return_effect
):

    """
    Approximate percentage-point interpretation:

        log return * 100

    For small daily returns this is approximately the change in
    simple-return percentage points.
    """

    return safe_float(
        100.0
        *
        log_return_effect
    )


def log_return_to_basis_points(
    log_return_effect
):

    """
    Approximate basis-point interpretation:

        log return * 10,000
    """

    return safe_float(
        10000.0
        *
        log_return_effect
    )


def log_return_to_exact_simple_return_percent(
    log_return_effect
):

    """
    Exact conversion of a log-return effect into the corresponding
    simple-return percentage change:

        100 * [exp(effect) - 1]
    """

    try:

        return safe_float(
            100.0
            *
            (
                math.exp(
                    log_return_effect
                )
                -
                1.0
            )
        )

    except Exception:

        return np.nan


# =====================================================================
# 9. LOAD FINAL DATASET
# =====================================================================

section(
    "ECONOMIC-SIGNIFICANCE FRAMEWORK"
)


print(
    "\nFinal dataset:"
)

print(
    FINAL_DATASET_FILE
)


print(
    "\nHAC/Newey-West maximum lag:"
)

print(
    HAC_MAXLAGS
)


print(
    "\nConfidence level:"
)

print(
    CONFIDENCE_LEVEL
)


if not FINAL_DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nRequired dataset not found:\n"
        f"{FINAL_DATASET_FILE}"
    )


df = pd.read_csv(
    FINAL_DATASET_FILE
)


print(
    "\nDataset shape:"
)

print(
    df.shape
)


# =====================================================================
# 10. DATE VALIDATION
# =====================================================================

section(
    "DATE VALIDATION"
)


if "Date" not in df.columns:

    raise KeyError(
        "Date column is missing from final dataset."
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
# 11. DAILY CALENDAR VALIDATION
# =====================================================================

section(
    "DAILY CALENDAR VALIDATION"
)


date_difference = (
    df[
        "Date"
    ]
    .diff()
)


calendar_gaps = int(
    (
        date_difference.notna()
        &
        (
            date_difference
            !=
            pd.Timedelta(
                days=1
            )
        )
    )
    .sum()
)


weekend_observations = int(
    (
        df[
            "Date"
        ]
        .dt.dayofweek
        >=
        5
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
    "\nCalendar gaps:"
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
        "Final dataset is not a continuous daily calendar."
    )


print(
    "\nPASS: Continuous daily cryptocurrency calendar confirmed."
)


# =====================================================================
# 12. MODEL SPECIFICATION VALIDATION
# =====================================================================

section(
    "MODEL SPECIFICATION VALIDATION"
)


required_variables = [
    "Date"
]


for asset, spec in MODEL_SPECS.items():

    required_variables.append(
        spec[
            "dependent"
        ]
    )

    required_variables.extend(
        spec[
            "predictors"
        ]
    )


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
        "\nMissing required variables:"
    )

    for variable in missing_variables:

        print(
            " -",
            variable
        )

    raise KeyError(
        "One or more required baseline variables are missing."
    )


print(
    "\nPASS: All baseline model variables are available."
)


# =====================================================================
# 13. INFORMATION-ALIGNMENT SAFETY CHECK
# =====================================================================

section(
    "INFORMATION-ALIGNMENT SAFETY CHECK"
)


used_predictors = []


for asset in MODEL_SPECS:

    used_predictors.extend(
        MODEL_SPECS[
            asset
        ][
            "predictors"
        ]
    )


old_controls_used = [

    variable

    for variable in OLD_NONALIGNED_CONTROLS

    if variable in used_predictors

]


print(
    "\nOld non-aligned controls used in specifications:"
)

print(
    old_controls_used
)


if old_controls_used:

    raise ValueError(
        "Old non-aligned traditional-market controls "
        "must not be used."
    )


aligned_controls = [

    "Lagged_SP500_Return_Aligned",

    "Lagged_VIX_Change_Aligned",

    "Lagged_Gold_Return_Aligned",

    "Lagged_DXY_Return_Aligned",

    "Lagged_US10Y_Change_Aligned"

]


print(
    "\nInformation-aligned controls:"
)


for variable in aligned_controls:

    print(
        " -",
        variable
    )


print(
    "\nPASS: Only corrected information-aligned "
    "traditional-market controls are used."
)


# =====================================================================
# 14. NUMERIC CONVERSION
# =====================================================================

section(
    "NUMERIC CONVERSION"
)


numeric_variables = [

    variable

    for variable in required_variables

    if variable != "Date"

]


for variable in numeric_variables:

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
    "\nPASS: Required variables converted to numeric."
)


# =====================================================================
# 15. REDDIT VARIABLE AVAILABILITY CHECK
# =====================================================================

section(
    "FUTURE REDDIT VARIABLE AVAILABILITY"
)


reddit_availability_rows = []


for asset in MODEL_SPECS:

    for variable in FUTURE_REDDIT_VARIABLES[
        asset
    ]:

        available = (
            variable
            in
            df.columns
        )


        reddit_availability_rows.append(
            {

                "Asset":
                    asset,

                "Variable":
                    variable,

                "Available":
                    available

            }
        )


        print(
            f"\n{asset} | {variable}: "
            f"{'AVAILABLE' if available else 'NOT YET AVAILABLE'}"
        )


reddit_availability_df = pd.DataFrame(
    reddit_availability_rows
)


# =====================================================================
# 16. ECONOMIC-SIGNIFICANCE ESTIMATION FUNCTION
# =====================================================================


def estimate_economic_significance(
    dataframe,
    asset,
    dependent,
    predictors,
    model_name="Baseline"
):

    section(
        f"{asset}: {model_name.upper()} ECONOMIC SIGNIFICANCE"
    )


    required = (
        [
            "Date",
            dependent
        ]
        +
        predictors
    )


    model_data = (
        dataframe[
            required
        ]
        .dropna(
            subset=[
                dependent
            ]
            +
            predictors
        )
        .copy()
        .sort_values(
            "Date"
        )
        .reset_index(
            drop=True
        )
    )


    if model_data.empty:

        raise ValueError(
            f"{asset}: estimation sample is empty."
        )


    n = len(
        model_data
    )


    start_date = (
        model_data[
            "Date"
        ].min()
    )


    end_date = (
        model_data[
            "Date"
        ].max()
    )


    weekend_count = int(
        (
            model_data[
                "Date"
            ]
            .dt.dayofweek
            >=
            5
        )
        .sum()
    )


    print(
        "\nEstimation observations:"
    )

    print(
        n
    )


    print(
        "\nEstimation period:"
    )

    print(
        start_date,
        "to",
        end_date
    )


    print(
        "\nWeekend observations:"
    )

    print(
        weekend_count
    )


    # ---------------------------------------------------------
    # Estimate OLS model
    # ---------------------------------------------------------

    y = (
        model_data[
            dependent
        ]
        .astype(float)
    )


    X = (
        model_data[
            predictors
        ]
        .astype(float)
    )


    X = sm.add_constant(
        X,
        has_constant="add"
    )


    ordinary_model = sm.OLS(
        y,
        X
    ).fit()


    # ---------------------------------------------------------
    # HAC / Newey-West inference
    # ---------------------------------------------------------

    hac_model = ordinary_model.get_robustcov_results(

        cov_type="HAC",

        maxlags=
            HAC_MAXLAGS,

        use_correction=True

    )


    # ---------------------------------------------------------
    # Statsmodels robust results may return arrays rather than
    # labelled pandas Series, so explicitly reconstruct labels.
    # ---------------------------------------------------------

    parameter_names = list(
        X.columns
    )


    coefficients = pd.Series(
        np.asarray(
            hac_model.params
        ),
        index=parameter_names,
        dtype=float
    )


    standard_errors = pd.Series(
        np.asarray(
            hac_model.bse
        ),
        index=parameter_names,
        dtype=float
    )


    p_values = pd.Series(
        np.asarray(
            hac_model.pvalues
        ),
        index=parameter_names,
        dtype=float
    )


    confidence_intervals_array = np.asarray(
        hac_model.conf_int(
            alpha=ALPHA
        )
    )


    confidence_intervals = pd.DataFrame(

        confidence_intervals_array,

        index=parameter_names,

        columns=[
            "Lower",
            "Upper"
        ]

    )


    # ---------------------------------------------------------
    # Dependent-variable volatility
    #
    # Useful for expressing the economic effect relative to the
    # unconditional volatility of daily crypto returns.
    # ---------------------------------------------------------

    dependent_sd = safe_float(
        y.std(
            ddof=1
        )
    )


    dependent_mean = safe_float(
        y.mean()
    )


    print(
        "\nDependent-variable mean daily log return:"
    )

    print(
        dependent_mean
    )


    print(
        "\nDependent-variable standard deviation:"
    )

    print(
        dependent_sd
    )


    print(
        "\nOLS R-squared:"
    )

    print(
        safe_float(
            ordinary_model.rsquared
        )
    )


    print(
        "\nAdjusted R-squared:"
    )

    print(
        safe_float(
            ordinary_model.rsquared_adj
        )
    )


    # ---------------------------------------------------------
    # Economic significance calculations
    # ---------------------------------------------------------

    result_rows = []


    for predictor in predictors:

        coefficient = safe_float(
            coefficients[
                predictor
            ]
        )


        hac_se = safe_float(
            standard_errors[
                predictor
            ]
        )


        p_value = safe_float(
            p_values[
                predictor
            ]
        )


        coefficient_ci_lower = safe_float(
            confidence_intervals.loc[
                predictor,
                "Lower"
            ]
        )


        coefficient_ci_upper = safe_float(
            confidence_intervals.loc[
                predictor,
                "Upper"
            ]
        )


        predictor_mean = safe_float(
            model_data[
                predictor
            ]
            .mean()
        )


        predictor_sd = safe_float(
            model_data[
                predictor
            ]
            .std(
                ddof=1
            )
        )


        predictor_min = safe_float(
            model_data[
                predictor
            ]
            .min()
        )


        predictor_max = safe_float(
            model_data[
                predictor
            ]
            .max()
        )


        # -----------------------------------------------------
        # Main economic-significance calculation
        # -----------------------------------------------------

        one_sd_effect = (
            coefficient
            *
            predictor_sd
        )


        # -----------------------------------------------------
        # CI for the one-SD economic effect
        #
        # Since SD(X) is treated as the descriptive scaling
        # factor:
        #
        # lower = beta_CI_lower * SD(X)
        # upper = beta_CI_upper * SD(X)
        # -----------------------------------------------------

        effect_ci_value_1 = (
            coefficient_ci_lower
            *
            predictor_sd
        )


        effect_ci_value_2 = (
            coefficient_ci_upper
            *
            predictor_sd
        )


        effect_ci_lower = min(
            effect_ci_value_1,
            effect_ci_value_2
        )


        effect_ci_upper = max(
            effect_ci_value_1,
            effect_ci_value_2
        )


        # -----------------------------------------------------
        # Approximate percentage-point effect
        # -----------------------------------------------------

        approximate_percentage_points = (
            log_return_to_approx_percentage_points(
                one_sd_effect
            )
        )


        # -----------------------------------------------------
        # Approximate basis-point effect
        # -----------------------------------------------------

        approximate_basis_points = (
            log_return_to_basis_points(
                one_sd_effect
            )
        )


        # -----------------------------------------------------
        # Exact simple-return equivalent
        # -----------------------------------------------------

        exact_simple_return_percent = (
            log_return_to_exact_simple_return_percent(
                one_sd_effect
            )
        )


        exact_ci_lower_percent = (
            log_return_to_exact_simple_return_percent(
                effect_ci_lower
            )
        )


        exact_ci_upper_percent = (
            log_return_to_exact_simple_return_percent(
                effect_ci_upper
            )
        )


        # -----------------------------------------------------
        # Effect relative to one SD of daily crypto return
        # -----------------------------------------------------

        if (
            dependent_sd is not None
            and
            not pd.isna(
                dependent_sd
            )
            and
            dependent_sd != 0
        ):

            effect_relative_to_return_sd = (
                one_sd_effect
                /
                dependent_sd
            )

        else:

            effect_relative_to_return_sd = np.nan


        # -----------------------------------------------------
        # Absolute magnitude
        # -----------------------------------------------------

        absolute_basis_point_effect = abs(
            approximate_basis_points
        )


        result_rows.append(
            {

                "Asset":
                    asset,

                "Model":
                    model_name,

                "Dependent_Variable":
                    dependent,

                "Predictor":
                    predictor,

                "Predictor_Label":
                    variable_label(
                        predictor
                    ),

                "N":
                    int(
                        n
                    ),

                "Sample_Start":
                    start_date,

                "Sample_End":
                    end_date,

                "Weekend_Observations":
                    weekend_count,

                "Coefficient":
                    coefficient,

                "HAC_SE":
                    hac_se,

                "HAC_P_Value":
                    p_value,

                "Significance":
                    significance_label(
                        p_value
                    ),

                "Coefficient_CI_Lower":
                    coefficient_ci_lower,

                "Coefficient_CI_Upper":
                    coefficient_ci_upper,

                "Predictor_Mean":
                    predictor_mean,

                "Predictor_SD":
                    predictor_sd,

                "Predictor_Min":
                    predictor_min,

                "Predictor_Max":
                    predictor_max,

                "One_SD_Effect_Log_Return":
                    safe_float(
                        one_sd_effect
                    ),

                "One_SD_Effect_CI_Lower_Log_Return":
                    safe_float(
                        effect_ci_lower
                    ),

                "One_SD_Effect_CI_Upper_Log_Return":
                    safe_float(
                        effect_ci_upper
                    ),

                "One_SD_Effect_Approx_Percentage_Points":
                    approximate_percentage_points,

                "One_SD_Effect_Approx_Basis_Points":
                    approximate_basis_points,

                "Absolute_One_SD_Effect_Basis_Points":
                    safe_float(
                        absolute_basis_point_effect
                    ),

                "One_SD_Effect_Exact_Simple_Return_Percent":
                    exact_simple_return_percent,

                "One_SD_Effect_Exact_CI_Lower_Percent":
                    exact_ci_lower_percent,

                "One_SD_Effect_Exact_CI_Upper_Percent":
                    exact_ci_upper_percent,

                "Dependent_Return_Mean":
                    dependent_mean,

                "Dependent_Return_SD":
                    dependent_sd,

                "Effect_as_Fraction_of_Daily_Return_SD":
                    safe_float(
                        effect_relative_to_return_sd
                    ),

                "R_Squared":
                    safe_float(
                        ordinary_model.rsquared
                    ),

                "Adjusted_R_Squared":
                    safe_float(
                        ordinary_model.rsquared_adj
                    ),

                "HAC_Maxlags":
                    int(
                        HAC_MAXLAGS
                    )

            }
        )


    results = pd.DataFrame(
        result_rows
    )


    # ---------------------------------------------------------
    # Print compact economic-significance table
    # ---------------------------------------------------------

    display_columns = [

        "Predictor_Label",

        "Coefficient",

        "Predictor_SD",

        "One_SD_Effect_Log_Return",

        "One_SD_Effect_Approx_Percentage_Points",

        "One_SD_Effect_Approx_Basis_Points",

        "HAC_P_Value",

        "Significance"

    ]


    print(
        "\nEconomic-significance results:"
    )


    print(
        "\n",
        results[
            display_columns
        ]
        .to_string(
            index=False
        )
    )


    return (
        results,
        model_data,
        ordinary_model,
        hac_model
    )


# =====================================================================
# 17. RUN PRIMARY BTC AND ETH BASELINE MODELS
# =====================================================================

all_results = []

model_objects = {}

model_samples = {}


for asset in [

    "BTC",

    "ETH"

]:

    specification = (
        MODEL_SPECS[
            asset
        ]
    )


    (
        results,
        model_sample,
        ordinary_model,
        hac_model

    ) = estimate_economic_significance(

        dataframe=
            df,

        asset=
            asset,

        dependent=
            specification[
                "dependent"
            ],

        predictors=
            specification[
                "predictors"
            ],

        model_name=
            "Primary_Baseline"

    )


    all_results.append(
        results
    )


    model_samples[
        asset
    ] = model_sample


    model_objects[
        asset
    ] = {

        "OLS":
            ordinary_model,

        "HAC":
            hac_model

    }


economic_significance_df = pd.concat(
    all_results,
    ignore_index=True
)


# =====================================================================
# 18. PRIMARY ECONOMIC-SIGNIFICANCE TABLE
# =====================================================================

section(
    "PRIMARY ECONOMIC-SIGNIFICANCE RESULTS"
)


primary_table = (
    economic_significance_df[
        [

            "Asset",

            "Predictor_Label",

            "Coefficient",

            "Predictor_SD",

            "One_SD_Effect_Log_Return",

            "One_SD_Effect_Approx_Percentage_Points",

            "One_SD_Effect_Approx_Basis_Points",

            "One_SD_Effect_Exact_Simple_Return_Percent",

            "Effect_as_Fraction_of_Daily_Return_SD",

            "HAC_P_Value",

            "Significance"

        ]
    ]
    .copy()
)


print(
    "\n",
    primary_table.to_string(
        index=False
    )
)


# =====================================================================
# 19. RANK PREDICTORS BY ABSOLUTE ECONOMIC MAGNITUDE
# =====================================================================

section(
    "RANKING BY ABSOLUTE ONE-SD ECONOMIC EFFECT"
)


ranked_results = (
    economic_significance_df
    .copy()
)


ranked_results[
    "Absolute_Economic_Effect"
] = np.abs(
    ranked_results[
        "One_SD_Effect_Log_Return"
    ]
)


ranked_results[
    "Economic_Effect_Rank"
] = (
    ranked_results
    .groupby(
        "Asset"
    )[
        "Absolute_Economic_Effect"
    ]
    .rank(
        method="dense",
        ascending=False
    )
    .astype(int)
)


ranked_results = (
    ranked_results
    .sort_values(
        [
            "Asset",
            "Economic_Effect_Rank"
        ]
    )
    .reset_index(
        drop=True
    )
)


print(
    "\n",
    ranked_results[
        [

            "Asset",

            "Economic_Effect_Rank",

            "Predictor_Label",

            "One_SD_Effect_Approx_Basis_Points",

            "Effect_as_Fraction_of_Daily_Return_SD",

            "HAC_P_Value"

        ]
    ]
    .to_string(
        index=False
    )
)


# =====================================================================
# 20. ECONOMIC VS STATISTICAL SIGNIFICANCE CLASSIFICATION
# =====================================================================
#
# We do NOT impose an arbitrary threshold for what counts as
# "economically significant".
#
# Instead, the script transparently reports:
#
#   - magnitude in basis points;
#   - magnitude relative to daily return volatility;
#   - statistical significance.
#
# This avoids falsely declaring that, for example, 10 bp is always
# economically large or small.
#
# =====================================================================

section(
    "ECONOMIC VS STATISTICAL SIGNIFICANCE"
)


economic_statistical_table = (
    ranked_results[
        [

            "Asset",

            "Predictor_Label",

            "One_SD_Effect_Approx_Basis_Points",

            "Absolute_One_SD_Effect_Basis_Points",

            "Effect_as_Fraction_of_Daily_Return_SD",

            "HAC_P_Value",

            "Significance"

        ]
    ]
    .copy()
)


economic_statistical_table[
    "Statistically_Significant_5pct"
] = (
    economic_statistical_table[
        "HAC_P_Value"
    ]
    <
    0.05
)


economic_statistical_table[
    "Statistically_Significant_10pct"
] = (
    economic_statistical_table[
        "HAC_P_Value"
    ]
    <
    0.10
)


print(
    "\n",
    economic_statistical_table.to_string(
        index=False
    )
)


# =====================================================================
# 21. AUTOMATIC INTERPRETATION SENTENCES
# =====================================================================

section(
    "AUTOMATIC ECONOMIC-SIGNIFICANCE INTERPRETATION"
)


interpretation_rows = []


for _, row in economic_significance_df.iterrows():

    asset = row[
        "Asset"
    ]


    predictor = row[
        "Predictor_Label"
    ]


    sd = row[
        "Predictor_SD"
    ]


    coefficient = row[
        "Coefficient"
    ]


    effect = row[
        "One_SD_Effect_Log_Return"
    ]


    percentage_effect = row[
        "One_SD_Effect_Approx_Percentage_Points"
    ]


    basis_point_effect = row[
        "One_SD_Effect_Approx_Basis_Points"
    ]


    p_value = row[
        "HAC_P_Value"
    ]


    if effect > 0:

        direction_word = "increase"

    elif effect < 0:

        direction_word = "decrease"

    else:

        direction_word = "change"


    sentence = (

        f"For {asset}, a one-standard-deviation increase in "
        f"{predictor} (SD = {sd:.6f}) is associated with an "
        f"estimated {direction_word} of "
        f"{abs(effect):.8f} in expected daily log return, "
        f"equivalent to approximately "
        f"{abs(percentage_effect):.4f} percentage points "
        f"({abs(basis_point_effect):.2f} basis points), "
        f"holding the other regressors constant. "
        f"The estimated coefficient is {coefficient:.6f} "
        f"with HAC p-value {p_value:.4f}."

    )


    interpretation_rows.append(
        {

            "Asset":
                asset,

            "Predictor":
                row[
                    "Predictor"
                ],

            "Interpretation":
                sentence

        }
    )


    print(
        "\n" + sentence
    )


interpretation_df = pd.DataFrame(
    interpretation_rows
)


# =====================================================================
# 22. CONFIDENCE-INTERVAL TABLE
# =====================================================================

section(
    "ONE-SD ECONOMIC-EFFECT CONFIDENCE INTERVALS"
)


confidence_interval_table = (
    economic_significance_df[
        [

            "Asset",

            "Predictor_Label",

            "One_SD_Effect_Log_Return",

            "One_SD_Effect_CI_Lower_Log_Return",

            "One_SD_Effect_CI_Upper_Log_Return",

            "One_SD_Effect_Approx_Basis_Points",

            "One_SD_Effect_Exact_CI_Lower_Percent",

            "One_SD_Effect_Exact_CI_Upper_Percent",

            "HAC_P_Value"

        ]
    ]
    .copy()
)


print(
    "\n",
    confidence_interval_table.to_string(
        index=False
    )
)


# =====================================================================
# 23. MODEL SAMPLE DIAGNOSTICS
# =====================================================================

section(
    "MODEL SAMPLE DIAGNOSTICS"
)


sample_diagnostic_rows = []


for asset in [

    "BTC",

    "ETH"

]:

    sample = (
        model_samples[
            asset
        ]
    )


    sample_diagnostic_rows.append(
        {

            "Asset":
                asset,

            "N":
                int(
                    len(
                        sample
                    )
                ),

            "Start_Date":
                sample[
                    "Date"
                ].min(),

            "End_Date":
                sample[
                    "Date"
                ].max(),

            "Weekdays":
                int(
                    (
                        sample[
                            "Date"
                        ]
                        .dt.dayofweek
                        <
                        5
                    )
                    .sum()
                ),

            "Weekends":
                int(
                    (
                        sample[
                            "Date"
                        ]
                        .dt.dayofweek
                        >=
                        5
                    )
                    .sum()
                )

        }
    )


sample_diagnostics_df = pd.DataFrame(
    sample_diagnostic_rows
)


print(
    "\n",
    sample_diagnostics_df.to_string(
        index=False
    )
)


# =====================================================================
# 24. VERIFY ECONOMIC-EFFECT CALCULATIONS
# =====================================================================

section(
    "ECONOMIC-EFFECT CALCULATION VALIDATION"
)


calculation_validation_rows = []


for _, row in economic_significance_df.iterrows():

    reconstructed_effect = (
        row[
            "Coefficient"
        ]
        *
        row[
            "Predictor_SD"
        ]
    )


    stored_effect = (
        row[
            "One_SD_Effect_Log_Return"
        ]
    )


    effect_match = bool(
        np.isclose(

            reconstructed_effect,

            stored_effect,

            rtol=1e-12,

            atol=1e-15

        )
    )


    reconstructed_basis_points = (
        stored_effect
        *
        10000.0
    )


    basis_points_match = bool(
        np.isclose(

            reconstructed_basis_points,

            row[
                "One_SD_Effect_Approx_Basis_Points"
            ],

            rtol=1e-12,

            atol=1e-12

        )
    )


    calculation_validation_rows.append(
        {

            "Asset":
                row[
                    "Asset"
                ],

            "Predictor":
                row[
                    "Predictor"
                ],

            "Coefficient_x_SD_Match":
                effect_match,

            "Basis_Point_Conversion_Match":
                basis_points_match

        }
    )


calculation_validation_df = pd.DataFrame(
    calculation_validation_rows
)


print(
    "\n",
    calculation_validation_df.to_string(
        index=False
    )
)


# =====================================================================
# 25. VERIFY HAC DOES NOT CHANGE COEFFICIENTS
# =====================================================================

section(
    "OLS VS HAC COEFFICIENT VALIDATION"
)


coefficient_validation_rows = []


for asset in [

    "BTC",

    "ETH"

]:

    ordinary_model = (
        model_objects[
            asset
        ][
            "OLS"
        ]
    )


    hac_model = (
        model_objects[
            asset
        ][
            "HAC"
        ]
    )


    parameter_names = list(
        ordinary_model.params.index
    )


    ols_coefficients = np.asarray(
        ordinary_model.params
    )


    hac_coefficients = np.asarray(
        hac_model.params
    )


    maximum_difference = safe_float(
        np.max(
            np.abs(
                ols_coefficients
                -
                hac_coefficients
            )
        )
    )


    coefficient_match = bool(
        np.allclose(

            ols_coefficients,

            hac_coefficients,

            rtol=1e-12,

            atol=1e-15

        )
    )


    coefficient_validation_rows.append(
        {

            "Asset":
                asset,

            "Parameters":
                len(
                    parameter_names
                ),

            "Maximum_Absolute_Coefficient_Difference":
                maximum_difference,

            "OLS_HAC_Coefficients_Identical":
                coefficient_match

        }
    )


coefficient_validation_df = pd.DataFrame(
    coefficient_validation_rows
)


print(
    "\n",
    coefficient_validation_df.to_string(
        index=False
    )
)


# =====================================================================
# 26. SAVE OUTPUTS
# =====================================================================

section(
    "SAVING ECONOMIC-SIGNIFICANCE OUTPUTS"
)


output_files = {

    "economic_significance_full_results.csv":
        economic_significance_df,

    "economic_significance_primary_table.csv":
        primary_table,

    "economic_significance_ranked.csv":
        ranked_results,

    "economic_vs_statistical_significance.csv":
        economic_statistical_table,

    "economic_significance_interpretations.csv":
        interpretation_df,

    "economic_effect_confidence_intervals.csv":
        confidence_interval_table,

    "economic_significance_sample_diagnostics.csv":
        sample_diagnostics_df,

    "economic_effect_calculation_validation.csv":
        calculation_validation_df,

    "ols_hac_coefficient_validation.csv":
        coefficient_validation_df,

    "reddit_variable_availability.csv":
        reddit_availability_df

}


for filename, dataframe in output_files.items():

    filepath = (
        OUTPUT_DIR
        /
        filename
    )


    dataframe.to_csv(
        filepath,
        index=False
    )


    print(
        "\nSaved:"
    )

    print(
        filepath
    )


# =====================================================================
# 27. SAVE MODEL SPECIFICATION
# =====================================================================

section(
    "SAVING MODEL SPECIFICATION"
)


model_specification_rows = []


for asset, specification in MODEL_SPECS.items():

    for predictor_number, predictor in enumerate(
        specification[
            "predictors"
        ],
        start=1
    ):

        model_specification_rows.append(
            {

                "Asset":
                    asset,

                "Dependent_Variable":
                    specification[
                        "dependent"
                    ],

                "Predictor_Number":
                    predictor_number,

                "Predictor":
                    predictor,

                "Predictor_Label":
                    variable_label(
                        predictor
                    ),

                "Information_Aligned":
                    (
                        "_Aligned"
                        in
                        predictor
                    )
                    if predictor in aligned_controls
                    else "Not applicable"

            }
        )


model_specification_df = pd.DataFrame(
    model_specification_rows
)


model_specification_file = (
    OUTPUT_DIR
    /
    "economic_significance_model_specification.csv"
)


model_specification_df.to_csv(
    model_specification_file,
    index=False
)


print(
    "\nSaved:"
)

print(
    model_specification_file
)


# =====================================================================
# 28. SAVE METHODOLOGY NOTE
# =====================================================================

section(
    "SAVING ECONOMIC-SIGNIFICANCE METHODOLOGY NOTE"
)


methodology_file = (
    OUTPUT_DIR
    /
    "economic_significance_methodology_note.txt"
)


methodology_note = f"""
ECONOMIC-SIGNIFICANCE FRAMEWORK

PURPOSE
-------
The economic-significance analysis supplements conventional
statistical inference by translating regression coefficients into
economically interpretable changes in expected daily cryptocurrency
returns.

PRIMARY CALCULATION
-------------------
For each continuous predictor X, the economic effect of a
one-standard-deviation increase is calculated as:

One-SD Effect = beta_X * SD(X)

where beta_X is the estimated regression coefficient and SD(X) is the
sample standard deviation of predictor X.

The predictor standard deviation is calculated using exactly the same
complete-case estimation sample as the corresponding regression.

DEPENDENT VARIABLE
------------------
The dependent variables are daily cryptocurrency log returns:

BTC_Return
ETH_Return

INTERPRETATION
--------------
The primary interpretation is:

"A one-standard-deviation increase in predictor X is associated with
an estimated Y change in expected daily cryptocurrency log return,
holding the other regressors constant."

For readability, Y is additionally expressed as:

1. approximate percentage points:

   100 * Y

2. approximate basis points:

   10,000 * Y

3. exact simple-return percentage equivalent:

   100 * [exp(Y) - 1]

For small daily return effects, the approximate percentage-point and
exact simple-return interpretations will be very similar.

CONFIDENCE INTERVALS
--------------------
HAC/Newey-West confidence intervals for the regression coefficient
are translated into one-standard-deviation economic-effect intervals
by multiplying both coefficient confidence limits by SD(X).

Because the transformation is linear in the coefficient, the lower
and upper economic-effect bounds are ordered after transformation.

INFERENCE
---------
OLS coefficients are retained, while statistical inference uses
HAC/Newey-West standard errors with maximum lag:

{HAC_MAXLAGS}

This is consistent with the dissertation's formal residual
diagnostics, which provided strong evidence of heteroskedasticity and
some evidence of residual serial dependence.

HAC inference changes standard errors, confidence intervals and
p-values. It does not change the underlying OLS coefficient estimate.

INFORMATION ALIGNMENT
---------------------
The primary models use only the corrected information-aligned
traditional-market controls:

Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

The old non-aligned traditional-market variables are not used.

ECONOMIC VERSUS STATISTICAL SIGNIFICANCE
----------------------------------------
Statistical and economic significance are conceptually distinct.

A coefficient may have a small p-value but a very small economic
magnitude.

Conversely, an estimated effect may be economically non-trivial while
being estimated imprecisely and therefore statistically
insignificant.

No arbitrary threshold is imposed for what constitutes an
"economically significant" number of basis points.

Instead, the analysis reports:

1. the one-SD effect in log-return units;
2. the approximate effect in percentage points;
3. the approximate effect in basis points;
4. the exact simple-return equivalent;
5. the effect relative to the standard deviation of daily crypto
   returns;
6. the HAC confidence interval; and
7. the HAC p-value.

This allows economic magnitude and statistical precision to be
discussed separately.

CAUSAL INTERPRETATION
---------------------
The economic-effect calculation does not establish causality.

The appropriate language is:

"is associated with"

rather than:

"causes".

FORECASTING INTERPRETATION
--------------------------
This is an explanatory economic-significance analysis.

It does not establish out-of-sample predictive performance.

Forecasting performance is evaluated separately using the dissertation
forecast-evaluation framework.

FUTURE REDDIT SENTIMENT
-----------------------
Once Reddit variables are available, the same calculation should be
applied to the lagged sentiment coefficient:

One-SD Sentiment Effect
    =
beta_sentiment * SD(lagged sentiment)

The resulting quantity can be interpreted as the estimated change in
expected next-day cryptocurrency return associated with a
one-standard-deviation increase in lagged Reddit sentiment, holding
the other regressors constant.

Reddit activity should be analysed separately from sentiment.

If Reddit post activity is measured as log(1 + post count), its
economic magnitude can likewise be expressed using a one-standard-
deviation change in the transformed activity variable.
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
# 29. SAVE DISSERTATION REPORTING TEMPLATE
# =====================================================================

section(
    "SAVING DISSERTATION REPORTING TEMPLATE"
)


reporting_template_file = (
    OUTPUT_DIR
    /
    "economic_significance_reporting_template.txt"
)


reporting_template = """
ECONOMIC-SIGNIFICANCE REPORTING TEMPLATE

The economic significance of the estimated relationships was assessed
in addition to conventional statistical significance. For each
continuous predictor, the estimated regression coefficient was
multiplied by the predictor's sample standard deviation. This
expresses the coefficient as the expected change in daily
cryptocurrency return associated with a one-standard-deviation change
in the predictor, holding the remaining regressors constant.

A suitable variable-specific sentence is:

"For [ASSET], a one-standard-deviation increase in [PREDICTOR] was
associated with an estimated [INCREASE/DECREASE] of [Y] in expected
daily log return, equivalent to approximately [Z] basis points,
holding the remaining regressors constant."

Statistical precision should then be discussed separately:

"The corresponding HAC/Newey-West p-value was [P], indicating that
the estimated economic magnitude was [precisely/imprecisely]
estimated."

IMPORTANT:

Do not automatically describe a coefficient as economically important
only because it is statistically significant.

Do not automatically describe an economically sizeable point estimate
as statistically reliable if its confidence interval is wide.

Once sentiment is added, use the same language:

"A one-standard-deviation increase in lagged Reddit sentiment was
associated with an estimated [Y] basis-point change in expected
next-day [Bitcoin/Ethereum] return, holding traditional market
indicators and the other controls constant."
""".strip()


with open(
    reporting_template_file,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        reporting_template
    )


print(
    "\nSaved:"
)

print(
    reporting_template_file
)


# =====================================================================
# 30. FINAL VALIDATION
# =====================================================================

section(
    "FINAL ECONOMIC-SIGNIFICANCE VALIDATION"
)


all_effect_calculations_match = bool(
    calculation_validation_df[
        "Coefficient_x_SD_Match"
    ]
    .all()
)


all_basis_point_conversions_match = bool(
    calculation_validation_df[
        "Basis_Point_Conversion_Match"
    ]
    .all()
)


all_ols_hac_coefficients_match = bool(
    coefficient_validation_df[
        "OLS_HAC_Coefficients_Identical"
    ]
    .all()
)


btc_sample_nonempty = bool(
    len(
        model_samples[
            "BTC"
        ]
    )
    >
    0
)


eth_sample_nonempty = bool(
    len(
        model_samples[
            "ETH"
        ]
    )
    >
    0
)


btc_weekends_retained = bool(
    (
        model_samples[
            "BTC"
        ][
            "Date"
        ]
        .dt.dayofweek
        >=
        5
    )
    .any()
)


eth_weekends_retained = bool(
    (
        model_samples[
            "ETH"
        ][
            "Date"
        ]
        .dt.dayofweek
        >=
        5
    )
    .any()
)


predictor_sd_positive = bool(
    (
        economic_significance_df[
            "Predictor_SD"
        ]
        >
        0
    )
    .all()
)


effects_complete = bool(
    economic_significance_df[
        [

            "Coefficient",

            "Predictor_SD",

            "One_SD_Effect_Log_Return",

            "One_SD_Effect_Approx_Basis_Points",

            "HAC_P_Value"

        ]
    ]
    .notna()
    .all()
    .all()
)


validation_checks = {

    "Final dataset exists":
        FINAL_DATASET_FILE.exists(),

    "No invalid dates":
        (
            invalid_dates
            ==
            0
        ),

    "No duplicate dates":
        (
            duplicate_dates
            ==
            0
        ),

    "Continuous daily calendar":
        (
            calendar_gaps
            ==
            0
        ),

    "All baseline variables available":
        (
            len(
                missing_variables
            )
            ==
            0
        ),

    "No old non-aligned controls used":
        (
            len(
                old_controls_used
            )
            ==
            0
        ),

    "BTC estimation sample non-empty":
        btc_sample_nonempty,

    "ETH estimation sample non-empty":
        eth_sample_nonempty,

    "BTC weekend observations retained":
        btc_weekends_retained,

    "ETH weekend observations retained":
        eth_weekends_retained,

    "All predictor SDs positive":
        predictor_sd_positive,

    "Economic-effect calculations complete":
        effects_complete,

    "Coefficient x SD calculations validated":
        all_effect_calculations_match,

    "Basis-point conversions validated":
        all_basis_point_conversions_match,

    "OLS and HAC coefficients identical":
        all_ols_hac_coefficients_match

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
    "\nOVERALL ECONOMIC-SIGNIFICANCE VALIDATION:"
)


print(
    "PASS"
    if overall_validation
    else "FAIL"
)


if not overall_validation:

    failed_checks = [

        name

        for name, condition
        in validation_checks.items()

        if not condition

    ]


    print(
        "\nFailed checks:"
    )


    for check in failed_checks:

        print(
            " -",
            check
        )


    raise ValueError(
        "\nOne or more economic-significance validation "
        "checks failed."
    )


# =====================================================================
# 31. FINAL INTERPRETATION REMINDERS
# =====================================================================

section(
    "INTERPRETATION REMINDERS"
)


print(
    "\n1. Economic significance and statistical significance "
    "are different concepts."
)


print(
    "\n2. The main economic effect is coefficient x predictor SD."
)


print(
    "\n3. Predictor SD is calculated on the exact regression "
    "estimation sample."
)


print(
    "\n4. Because the dependent variable is a daily log return, "
    "multiplying the effect by 100 gives an approximate "
    "percentage-point interpretation."
)


print(
    "\n5. Multiplying the log-return effect by 10,000 gives an "
    "approximate basis-point interpretation."
)


print(
    "\n6. The script also reports the exact simple-return "
    "equivalent using exp(effect) - 1."
)


print(
    "\n7. HAC/Newey-West inference changes standard errors and "
    "p-values, not the OLS coefficients."
)


print(
    "\n8. Do not call an effect economically important solely "
    "because p < 0.05."
)


print(
    "\n9. Do not call an imprecisely estimated effect reliable "
    "solely because its point estimate is large."
)


print(
    "\n10. Use 'associated with', not causal language such as "
    "'causes'."
)


print(
    "\n11. This is an explanatory economic-significance "
    "analysis, not an OOS forecasting test."
)


print(
    "\n12. Once Reddit arrives, apply exactly the same "
    "coefficient x SD calculation to lagged sentiment."
)


print(
    "\n13. Reddit activity and Reddit sentiment should remain "
    "conceptually separate."
)


# =====================================================================
# 32. COMPLETE
# =====================================================================

section(
    "ECONOMIC-SIGNIFICANCE FRAMEWORK COMPLETE"
)


print(
    "\nAssets analysed:"
)

print(
    "BTC, ETH"
)


print(
    "\nModels:"
)

print(
    "Primary information-aligned baseline explanatory models"
)


print(
    "\nHAC maximum lag:"
)

print(
    HAC_MAXLAGS
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nReddit sentiment:"
)

if (
    reddit_availability_df[
        "Available"
    ]
    .any()
):

    print(
        "At least one configured Reddit variable is available."
    )

else:

    print(
        "Not yet available - baseline economic-significance "
        "framework is ready."
    )


print(
    "\nOverall validation:"
)

print(
    "PASS"
)