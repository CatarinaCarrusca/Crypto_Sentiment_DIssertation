# =====================================================================
# 08_market_regime_structural_stability.py
#
# MARKET-REGIME / STRUCTURAL-STABILITY ROBUSTNESS
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
# Examine whether the coefficients in the primary baseline explanatory
# regressions are stable across the 2021-2025 sample.
#
# ROBUSTNESS APPROACHES
# ---------------------------------------------------------------------
#
# 1. Predetermined calendar-year subsamples:
#
#       2021
#       2022
#       2023
#       2024
#       2025
#
# 2. 365-day rolling coefficient estimation.
#
# Calendar years are transparent, predetermined sample partitions.
# They are NOT statistically estimated structural-break dates.
#
# Rolling coefficients provide supporting descriptive evidence about
# whether estimated relationships vary through time.
#
# IMPORTANT
# ---------------------------------------------------------------------
# This is an IN-SAMPLE EXPLANATORY robustness analysis.
#
# It is NOT:
#   - an out-of-sample forecasting exercise;
#   - a formal unknown-break-date test;
#   - evidence of predictive performance;
#   - a replacement for the primary full-sample model.
#
# Once Reddit sentiment is available, the same framework can be
# extended to the lagged sentiment coefficient.
#
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
    / "market_regime_structural_stability"
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

CONFIDENCE_LEVEL = 0.95

ROLLING_WINDOW = 365

# Estimate a rolling model every seven days.
# This keeps the output manageable while retaining a detailed
# representation of temporal coefficient variation.

ROLLING_STEP = 7

MIN_ROLLING_OBSERVATIONS = 330


# =====================================================================
# 3. PREDETERMINED SUBSAMPLES
# =====================================================================

SUBSAMPLES = {

    "2021": (
        "2021-01-01",
        "2021-12-31"
    ),

    "2022": (
        "2022-01-01",
        "2022-12-31"
    ),

    "2023": (
        "2023-01-01",
        "2023-12-31"
    ),

    "2024": (
        "2024-01-01",
        "2024-12-31"
    ),

    "2025": (
        "2025-01-01",
        "2025-12-31"
    )

}


# =====================================================================
# 4. MODEL SPECIFICATIONS
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
# 5. KEY COEFFICIENTS
# =====================================================================

BTC_KEY_COEFFICIENTS = [

    "BTC_Lagged_Return",

    "Lagged_Log_BTC_Volume"

]


ETH_KEY_COEFFICIENTS = [

    "ETH_Lagged_Return",

    "Lagged_Log_ETH_Volume"

]


# =====================================================================
# 6. HELPER FUNCTIONS
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


def sign_label(coefficient):

    if pd.isna(coefficient):
        return "NA"

    if coefficient > 0:
        return "Positive"

    if coefficient < 0:
        return "Negative"

    return "Zero"


def safe_float(value):

    try:
        return float(value)

    except Exception:
        return np.nan


# =====================================================================
# 7. START
# =====================================================================

section(
    "MARKET-REGIME / STRUCTURAL-STABILITY ROBUSTNESS"
)

print(
    "\nInput dataset:"
)

print(
    INPUT_FILE
)

print(
    "\nPredetermined annual subsamples:"
)

for name, dates in SUBSAMPLES.items():

    print(
        f" - {name}: {dates[0]} to {dates[1]}"
    )

print(
    "\nRolling window:"
)

print(
    ROLLING_WINDOW,
    "daily observations"
)

print(
    "\nRolling estimation step:"
)

print(
    ROLLING_STEP,
    "days"
)

print(
    "\nMinimum rolling observations:"
)

print(
    MIN_ROLLING_OBSERVATIONS
)

print(
    "\nHAC/Newey-West maximum lag:"
)

print(
    HAC_MAXLAGS
)


# =====================================================================
# 8. CHECK INPUT FILE
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
# 9. LOAD DATA
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


# =====================================================================
# 10. DATE VALIDATION
# =====================================================================

section(
    "DATE VALIDATION"
)

if "Date" not in df.columns:

    raise KeyError(
        "Date column is missing."
    )

df["Date"] = pd.to_datetime(
    df["Date"],
    errors="coerce"
)

invalid_dates = int(
    df["Date"]
    .isna()
    .sum()
)

duplicate_dates = int(
    df["Date"]
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
    .sort_values("Date")
    .reset_index(drop=True)
)

print(
    "\nDate range:"
)

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =====================================================================
# 11. DAILY CALENDAR VALIDATION
# =====================================================================

section(
    "DAILY CALENDAR VALIDATION"
)

date_differences = (
    df["Date"]
    .diff()
)

calendar_gap_mask = (

    date_differences.notna()

    &

    (
        date_differences
        != pd.Timedelta(days=1)
    )
)

calendar_gaps = int(
    calendar_gap_mask.sum()
)

weekend_observations = int(
    (
        df["Date"]
        .dt.dayofweek
        >= 5
    )
    .sum()
)

print(
    "\nObservations:"
)

print(
    len(df)
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
        "Dataset is not a continuous daily calendar."
    )

print(
    "\nPASS: Continuous daily cryptocurrency calendar confirmed."
)


# =====================================================================
# 12. INFORMATION-ALIGNMENT SAFETY CHECK
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
            f"Missing aligned control: {variable}"
        )


print(
    "\nAligned controls used:"
)

for variable in ALIGNED_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nOld non-aligned controls explicitly excluded:"
)

for variable in OLD_NONALIGNED_CONTROLS:

    print(
        " -",
        variable
    )


print(
    "\nPASS: Only information-aligned traditional-market "
    "controls will be used."
)


# =====================================================================
# 13. REQUIRED VARIABLE VALIDATION
# =====================================================================

section(
    "MODEL VARIABLE VALIDATION"
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
# 14. NUMERIC CONVERSION
# =====================================================================

section(
    "NUMERIC CONVERSION"
)

for variable in REQUIRED_COLUMNS:

    df[variable] = (
        pd.to_numeric(
            df[variable],
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
# 15. PREPARE ESTIMATION SAMPLE
# =====================================================================

section(
    "PREPARING FULL ESTIMATION SAMPLES"
)


def prepare_estimation_sample(
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
        dataframe[required]
        .dropna()
        .copy()
        .sort_values("Date")
        .reset_index(drop=True)
    )

    sample["Weekend"] = (
        sample["Date"]
        .dt.dayofweek
        >= 5
    )

    internal_gaps = int(
        (
            sample["Date"]
            .diff()
            .dropna()
            != pd.Timedelta(days=1)
        )
        .sum()
    )

    print(
        f"\n{asset} observations:"
    )

    print(
        len(sample)
    )

    print(
        f"{asset} period:"
    )

    print(
        sample["Date"].min(),
        "to",
        sample["Date"].max()
    )

    print(
        f"{asset} weekend observations:"
    )

    print(
        int(
            sample["Weekend"].sum()
        )
    )

    print(
        f"{asset} internal date gaps:"
    )

    print(
        internal_gaps
    )

    if internal_gaps > 0:

        raise ValueError(
            f"{asset} estimation sample contains "
            "internal calendar gaps."
        )

    return sample


btc_sample = prepare_estimation_sample(
    df,
    "BTC",
    BTC_DEPENDENT,
    BTC_PREDICTORS
)

eth_sample = prepare_estimation_sample(
    df,
    "ETH",
    ETH_DEPENDENT,
    ETH_PREDICTORS
)


# =====================================================================
# 16. HAC MODEL ESTIMATION
# =====================================================================

def estimate_hac_model(
    sample,
    dependent,
    predictors
):

    y = sample[
        dependent
    ].astype(float)

    X = sample[
        predictors
    ].astype(float)

    X = sm.add_constant(
        X,
        has_constant="add"
    )

    ols_model = sm.OLS(
        y,
        X
    ).fit()

    hac_model = ols_model.get_robustcov_results(
        cov_type="HAC",
        maxlags=HAC_MAXLAGS
    )

    return (
        ols_model,
        hac_model,
        X
    )


# =====================================================================
# 17. COEFFICIENT EXTRACTION
# =====================================================================

def extract_coefficients(
    asset,
    period,
    start_date,
    end_date,
    ols_model,
    hac_model,
    X
):

    variable_names = list(
        X.columns
    )

    coefficients = np.asarray(
        hac_model.params
    )

    standard_errors = np.asarray(
        hac_model.bse
    )

    t_values = np.asarray(
        hac_model.tvalues
    )

    p_values = np.asarray(
        hac_model.pvalues
    )

    confidence_intervals = np.asarray(
        hac_model.conf_int(
            alpha=1 - CONFIDENCE_LEVEL
        )
    )

    rows = []

    for i, variable in enumerate(
        variable_names
    ):

        coefficient = safe_float(
            coefficients[i]
        )

        p_value = safe_float(
            p_values[i]
        )

        rows.append(
            {

                "Asset":
                    asset,

                "Period":
                    period,

                "Start_Date":
                    start_date,

                "End_Date":
                    end_date,

                "N":
                    int(
                        ols_model.nobs
                    ),

                "Variable":
                    variable,

                "Coefficient":
                    coefficient,

                "HAC_SE":
                    safe_float(
                        standard_errors[i]
                    ),

                "T_Statistic":
                    safe_float(
                        t_values[i]
                    ),

                "P_Value":
                    p_value,

                "CI_95_Lower":
                    safe_float(
                        confidence_intervals[i, 0]
                    ),

                "CI_95_Upper":
                    safe_float(
                        confidence_intervals[i, 1]
                    ),

                "Sign":
                    sign_label(
                        coefficient
                    ),

                "Significance":
                    significance_label(
                        p_value
                    ),

                "Significant_5pct":
                    bool(
                        p_value
                        <
                        SIGNIFICANCE_LEVEL
                    )
                    if not pd.isna(p_value)
                    else False,

                "R_Squared":
                    safe_float(
                        ols_model.rsquared
                    ),

                "Adjusted_R_Squared":
                    safe_float(
                        ols_model.rsquared_adj
                    ),

                "HAC_Maxlags":
                    HAC_MAXLAGS

            }
        )

    return rows


# =====================================================================
# 18. FULL-SAMPLE REFERENCE MODELS
# =====================================================================

section(
    "FULL-SAMPLE REFERENCE MODELS"
)


(
    btc_full_ols,
    btc_full_hac,
    btc_full_X
) = estimate_hac_model(
    btc_sample,
    BTC_DEPENDENT,
    BTC_PREDICTORS
)


(
    eth_full_ols,
    eth_full_hac,
    eth_full_X
) = estimate_hac_model(
    eth_sample,
    ETH_DEPENDENT,
    ETH_PREDICTORS
)


print(
    "\nBTC full sample:"
)

print(
    "N:",
    int(
        btc_full_ols.nobs
    )
)

print(
    "R-squared:",
    btc_full_ols.rsquared
)

print(
    "Adjusted R-squared:",
    btc_full_ols.rsquared_adj
)


print(
    "\nETH full sample:"
)

print(
    "N:",
    int(
        eth_full_ols.nobs
    )
)

print(
    "R-squared:",
    eth_full_ols.rsquared
)

print(
    "Adjusted R-squared:",
    eth_full_ols.rsquared_adj
)


full_sample_coefficient_rows = []


full_sample_coefficient_rows.extend(
    extract_coefficients(
        "BTC",
        "Full_2021_2025",
        btc_sample["Date"].min(),
        btc_sample["Date"].max(),
        btc_full_ols,
        btc_full_hac,
        btc_full_X
    )
)


full_sample_coefficient_rows.extend(
    extract_coefficients(
        "ETH",
        "Full_2021_2025",
        eth_sample["Date"].min(),
        eth_sample["Date"].max(),
        eth_full_ols,
        eth_full_hac,
        eth_full_X
    )
)


full_sample_coefficients = pd.DataFrame(
    full_sample_coefficient_rows
)


# =====================================================================
# 19. FULL-SAMPLE KEY COEFFICIENTS
# =====================================================================

section(
    "FULL-SAMPLE KEY COEFFICIENTS"
)


def print_full_key_coefficients(
    asset,
    key_variables
):

    temp = full_sample_coefficients.loc[
        (
            full_sample_coefficients["Asset"]
            == asset
        )
        &
        (
            full_sample_coefficients["Variable"]
            .isin(key_variables)
        )
    ].copy()

    print(
        f"\n{asset}:"
    )

    print(
        temp[
            [
                "Variable",
                "Coefficient",
                "HAC_SE",
                "P_Value",
                "CI_95_Lower",
                "CI_95_Upper",
                "Sign",
                "Significance"
            ]
        ]
        .to_string(
            index=False
        )
    )


print_full_key_coefficients(
    "BTC",
    BTC_KEY_COEFFICIENTS
)

print_full_key_coefficients(
    "ETH",
    ETH_KEY_COEFFICIENTS
)


# =====================================================================
# 20. PREDETERMINED SUBSAMPLE ESTIMATION
# =====================================================================

section(
    "PREDETERMINED CALENDAR-YEAR SUBSAMPLE ESTIMATION"
)


subsample_coefficient_rows = []

subsample_model_rows = []


def run_subsamples(
    asset,
    sample,
    dependent,
    predictors
):

    for period, (
        start_string,
        end_string
    ) in SUBSAMPLES.items():

        start_date = pd.Timestamp(
            start_string
        )

        end_date = pd.Timestamp(
            end_string
        )

        period_sample = (
            sample.loc[
                (
                    sample["Date"]
                    >= start_date
                )
                &
                (
                    sample["Date"]
                    <= end_date
                )
            ]
            .copy()
            .reset_index(drop=True)
        )

        print(
            f"\n{asset} - {period}"
        )

        print(
            "Observations:",
            len(period_sample)
        )

        if len(period_sample) <= (
            len(predictors)
            + 10
        ):

            print(
                "SKIPPED: insufficient observations."
            )

            continue

        (
            ols_model,
            hac_model,
            X
        ) = estimate_hac_model(
            period_sample,
            dependent,
            predictors
        )

        actual_start = (
            period_sample["Date"].min()
        )

        actual_end = (
            period_sample["Date"].max()
        )

        weekend_count = int(
            (
                period_sample["Date"]
                .dt.dayofweek
                >= 5
            )
            .sum()
        )

        print(
            "Actual period:",
            actual_start,
            "to",
            actual_end
        )

        print(
            "Weekend observations:",
            weekend_count
        )

        print(
            "R-squared:",
            ols_model.rsquared
        )

        print(
            "Adjusted R-squared:",
            ols_model.rsquared_adj
        )

        coefficient_rows = extract_coefficients(
            asset,
            period,
            actual_start,
            actual_end,
            ols_model,
            hac_model,
            X
        )

        subsample_coefficient_rows.extend(
            coefficient_rows
        )

        subsample_model_rows.append(
            {

                "Asset":
                    asset,

                "Period":
                    period,

                "Start_Date":
                    actual_start,

                "End_Date":
                    actual_end,

                "N":
                    int(
                        ols_model.nobs
                    ),

                "Weekend_Observations":
                    weekend_count,

                "R_Squared":
                    safe_float(
                        ols_model.rsquared
                    ),

                "Adjusted_R_Squared":
                    safe_float(
                        ols_model.rsquared_adj
                    ),

                "HAC_Maxlags":
                    HAC_MAXLAGS

            }
        )


run_subsamples(
    "BTC",
    btc_sample,
    BTC_DEPENDENT,
    BTC_PREDICTORS
)


run_subsamples(
    "ETH",
    eth_sample,
    ETH_DEPENDENT,
    ETH_PREDICTORS
)


subsample_coefficients = pd.DataFrame(
    subsample_coefficient_rows
)


subsample_models = pd.DataFrame(
    subsample_model_rows
)


# =====================================================================
# 21. PRINT KEY SUBSAMPLE RESULTS
# =====================================================================

section(
    "KEY COEFFICIENTS ACROSS PREDETERMINED SUBSAMPLES"
)


def print_key_subsample_results(
    asset,
    key_variables
):

    asset_results = (
        subsample_coefficients.loc[
            (
                subsample_coefficients["Asset"]
                == asset
            )
            &
            (
                subsample_coefficients["Variable"]
                .isin(key_variables)
            )
        ]
        .copy()
    )

    columns_to_show = [

        "Asset",
        "Period",
        "Variable",
        "N",
        "Coefficient",
        "HAC_SE",
        "P_Value",
        "CI_95_Lower",
        "CI_95_Upper",
        "Sign",
        "Significance",
        "R_Squared"

    ]

    print(
        f"\n{asset}:"
    )

    print(
        asset_results[
            columns_to_show
        ]
        .to_string(
            index=False
        )
    )


print_key_subsample_results(
    "BTC",
    BTC_KEY_COEFFICIENTS
)


print_key_subsample_results(
    "ETH",
    ETH_KEY_COEFFICIENTS
)


# =====================================================================
# 22. SUBSAMPLE VERSUS FULL-SAMPLE COMPARISON
# =====================================================================
#
# IMPORTANT PANDAS COMPATIBILITY FIX
# ---------------------------------------------------------------------
#
# Pandas in newer Python environments does not allow np.nan to be
# assigned directly into an ordinary bool column.
#
# Therefore Same_Sign_As_Full is explicitly created using Pandas'
# nullable BooleanDtype, which supports:
#
#       True
#       False
#       pd.NA
#
# =====================================================================

section(
    "SUBSAMPLE VERSUS FULL-SAMPLE COEFFICIENT COMPARISON"
)


full_reference = (
    full_sample_coefficients[
        [
            "Asset",
            "Variable",
            "Coefficient"
        ]
    ]
    .rename(
        columns={
            "Coefficient":
                "Full_Sample_Coefficient"
        }
    )
)


subsample_comparison = (
    subsample_coefficients
    .merge(
        full_reference,
        on=[
            "Asset",
            "Variable"
        ],
        how="left"
    )
)


subsample_comparison[
    "Difference_From_Full"
] = (
    subsample_comparison["Coefficient"]
    -
    subsample_comparison[
        "Full_Sample_Coefficient"
    ]
)


subsample_comparison[
    "Absolute_Difference_From_Full"
] = (
    subsample_comparison[
        "Difference_From_Full"
    ]
    .abs()
)


# ---------------------------------------------------------
# Correct nullable Boolean implementation
# ---------------------------------------------------------

same_sign_values = (

    np.sign(
        subsample_comparison["Coefficient"]
    )

    ==

    np.sign(
        subsample_comparison[
            "Full_Sample_Coefficient"
        ]
    )

)


subsample_comparison[
    "Same_Sign_As_Full"
] = pd.Series(
    same_sign_values,
    index=subsample_comparison.index,
    dtype="boolean"
)


zero_or_missing_mask = (

    subsample_comparison["Coefficient"]
    .isna()

    |

    subsample_comparison[
        "Full_Sample_Coefficient"
    ]
    .isna()

    |

    (
        subsample_comparison["Coefficient"]
        == 0
    )

    |

    (
        subsample_comparison[
            "Full_Sample_Coefficient"
        ]
        == 0
    )

)


subsample_comparison.loc[
    zero_or_missing_mask,
    "Same_Sign_As_Full"
] = pd.NA


print(
    "\nPASS: Subsample/full-sample comparison constructed."
)


# =====================================================================
# 23. SUBSAMPLE STABILITY SUMMARY
# =====================================================================

section(
    "SUBSAMPLE STABILITY SUMMARY"
)


subsample_stability_rows = []


for asset in [
    "BTC",
    "ETH"
]:

    if asset == "BTC":

        variables = BTC_PREDICTORS

    else:

        variables = ETH_PREDICTORS


    for variable in variables:

        temp = (
            subsample_comparison.loc[
                (
                    subsample_comparison["Asset"]
                    == asset
                )
                &
                (
                    subsample_comparison["Variable"]
                    == variable
                )
            ]
            .copy()
        )

        if temp.empty:

            continue


        coefficients = (
            temp["Coefficient"]
            .dropna()
        )


        p_values = (
            temp["P_Value"]
            .dropna()
        )


        if coefficients.empty:

            continue


        positive_count = int(
            (
                coefficients > 0
            )
            .sum()
        )


        negative_count = int(
            (
                coefficients < 0
            )
            .sum()
        )


        significant_5_count = int(
            (
                p_values < 0.05
            )
            .sum()
        )


        significant_10_count = int(
            (
                p_values < 0.10
            )
            .sum()
        )


        sign_changes = bool(
            positive_count > 0
            and
            negative_count > 0
        )


        full_coefficient = safe_float(
            temp[
                "Full_Sample_Coefficient"
            ]
            .iloc[0]
        )


        same_sign_nonmissing = (
            temp[
                "Same_Sign_As_Full"
            ]
            .dropna()
        )


        if len(
            same_sign_nonmissing
        ) > 0:

            share_same_sign = safe_float(
                same_sign_nonmissing
                .astype(float)
                .mean()
            )

        else:

            share_same_sign = np.nan


        subsample_stability_rows.append(
            {

                "Asset":
                    asset,

                "Variable":
                    variable,

                "Full_Sample_Coefficient":
                    full_coefficient,

                "Number_of_Subsamples":
                    int(
                        len(coefficients)
                    ),

                "Minimum_Subsample_Coefficient":
                    safe_float(
                        coefficients.min()
                    ),

                "Maximum_Subsample_Coefficient":
                    safe_float(
                        coefficients.max()
                    ),

                "Mean_Subsample_Coefficient":
                    safe_float(
                        coefficients.mean()
                    ),

                "Std_Subsample_Coefficient":
                    safe_float(
                        coefficients.std(
                            ddof=1
                        )
                    ),

                "Positive_Subsamples":
                    positive_count,

                "Negative_Subsamples":
                    negative_count,

                "Sign_Changes_Across_Subsamples":
                    sign_changes,

                "Significant_at_5pct_Subsamples":
                    significant_5_count,

                "Significant_at_10pct_Subsamples":
                    significant_10_count,

                "Share_Same_Sign_As_Full":
                    share_same_sign,

                "Maximum_Absolute_Difference_From_Full":
                    safe_float(
                        temp[
                            "Absolute_Difference_From_Full"
                        ]
                        .max()
                    )

            }
        )


subsample_stability_summary = pd.DataFrame(
    subsample_stability_rows
)


print(
    "\n",
    subsample_stability_summary.to_string(
        index=False
    )
)


# =====================================================================
# 24. ROLLING 365-DAY MODELS
# =====================================================================

section(
    "365-DAY ROLLING COEFFICIENT ESTIMATION"
)


def run_rolling_models(
    asset,
    sample,
    dependent,
    predictors
):

    rolling_rows = []

    model_rows = []


    sample = (
        sample
        .sort_values("Date")
        .reset_index(drop=True)
    )


    number_of_observations = len(
        sample
    )


    print(
        f"\n{asset} total usable observations:"
    )

    print(
        number_of_observations
    )


    if number_of_observations < ROLLING_WINDOW:

        raise ValueError(
            f"{asset}: insufficient observations "
            "for rolling analysis."
        )


    rolling_end_positions = range(
        ROLLING_WINDOW - 1,
        number_of_observations,
        ROLLING_STEP
    )


    for end_position in rolling_end_positions:

        start_position = (
            end_position
            -
            ROLLING_WINDOW
            +
            1
        )


        window = (
            sample.iloc[
                start_position:
                end_position + 1
            ]
            .copy()
            .reset_index(drop=True)
        )


        window_start = (
            window["Date"].min()
        )


        window_end = (
            window["Date"].max()
        )


        usable_window = (
            window[
                [
                    "Date",
                    dependent
                ]
                +
                predictors
            ]
            .dropna()
            .copy()
        )


        n_usable = len(
            usable_window
        )


        if n_usable < MIN_ROLLING_OBSERVATIONS:

            continue


        (
            ols_model,
            hac_model,
            X
        ) = estimate_hac_model(
            usable_window,
            dependent,
            predictors
        )


        variable_names = list(
            X.columns
        )


        coefficients = np.asarray(
            hac_model.params
        )


        standard_errors = np.asarray(
            hac_model.bse
        )


        t_values = np.asarray(
            hac_model.tvalues
        )


        p_values = np.asarray(
            hac_model.pvalues
        )


        confidence_intervals = np.asarray(
            hac_model.conf_int(
                alpha=1 - CONFIDENCE_LEVEL
            )
        )


        for i, variable in enumerate(
            variable_names
        ):

            coefficient = safe_float(
                coefficients[i]
            )


            p_value = safe_float(
                p_values[i]
            )


            rolling_rows.append(
                {

                    "Asset":
                        asset,

                    "Window_Start":
                        window_start,

                    "Window_End":
                        window_end,

                    "N":
                        int(
                            ols_model.nobs
                        ),

                    "Variable":
                        variable,

                    "Coefficient":
                        coefficient,

                    "HAC_SE":
                        safe_float(
                            standard_errors[i]
                        ),

                    "T_Statistic":
                        safe_float(
                            t_values[i]
                        ),

                    "P_Value":
                        p_value,

                    "CI_95_Lower":
                        safe_float(
                            confidence_intervals[
                                i,
                                0
                            ]
                        ),

                    "CI_95_Upper":
                        safe_float(
                            confidence_intervals[
                                i,
                                1
                            ]
                        ),

                    "Sign":
                        sign_label(
                            coefficient
                        ),

                    "Significance":
                        significance_label(
                            p_value
                        ),

                    "Significant_5pct":
                        bool(
                            p_value < 0.05
                        )
                        if not pd.isna(
                            p_value
                        )
                        else False,

                    "R_Squared":
                        safe_float(
                            ols_model.rsquared
                        ),

                    "Adjusted_R_Squared":
                        safe_float(
                            ols_model.rsquared_adj
                        ),

                    "HAC_Maxlags":
                        HAC_MAXLAGS

                }
            )


        model_rows.append(
            {

                "Asset":
                    asset,

                "Window_Start":
                    window_start,

                "Window_End":
                    window_end,

                "N":
                    int(
                        ols_model.nobs
                    ),

                "R_Squared":
                    safe_float(
                        ols_model.rsquared
                    ),

                "Adjusted_R_Squared":
                    safe_float(
                        ols_model.rsquared_adj
                    ),

                "HAC_Maxlags":
                    HAC_MAXLAGS

            }
        )


    rolling_coefficients = pd.DataFrame(
        rolling_rows
    )


    rolling_models = pd.DataFrame(
        model_rows
    )


    print(
        f"\n{asset} rolling models estimated:"
    )

    print(
        len(
            rolling_models
        )
    )


    if not rolling_models.empty:

        rolling_models = (
            rolling_models
            .sort_values(
                "Window_End"
            )
            .reset_index(
                drop=True
            )
        )


        first_row = (
            rolling_models.iloc[0]
        )


        last_row = (
            rolling_models.iloc[-1]
        )


        print(
            f"\n{asset} first rolling window:"
        )

        print(
            first_row[
                "Window_Start"
            ],
            "to",
            first_row[
                "Window_End"
            ]
        )


        print(
            f"\n{asset} final rolling window:"
        )

        print(
            last_row[
                "Window_Start"
            ],
            "to",
            last_row[
                "Window_End"
            ]
        )


        print(
            f"\n{asset} minimum rolling N:"
        )

        print(
            int(
                rolling_models["N"]
                .min()
            )
        )


        print(
            f"\n{asset} maximum rolling N:"
        )

        print(
            int(
                rolling_models["N"]
                .max()
            )
        )


    return (
        rolling_coefficients,
        rolling_models
    )


(
    btc_rolling_coefficients,
    btc_rolling_models
) = run_rolling_models(
    "BTC",
    btc_sample,
    BTC_DEPENDENT,
    BTC_PREDICTORS
)


(
    eth_rolling_coefficients,
    eth_rolling_models
) = run_rolling_models(
    "ETH",
    eth_sample,
    ETH_DEPENDENT,
    ETH_PREDICTORS
)


rolling_coefficients = pd.concat(
    [
        btc_rolling_coefficients,
        eth_rolling_coefficients
    ],
    ignore_index=True
)


rolling_models = pd.concat(
    [
        btc_rolling_models,
        eth_rolling_models
    ],
    ignore_index=True
)


# =====================================================================
# 25. ROLLING STABILITY SUMMARY
# =====================================================================

section(
    "ROLLING-COEFFICIENT STABILITY SUMMARY"
)


rolling_stability_rows = []


for asset in [
    "BTC",
    "ETH"
]:

    if asset == "BTC":

        variables = BTC_PREDICTORS

    else:

        variables = ETH_PREDICTORS


    for variable in variables:

        temp = (
            rolling_coefficients.loc[
                (
                    rolling_coefficients["Asset"]
                    == asset
                )
                &
                (
                    rolling_coefficients["Variable"]
                    == variable
                )
            ]
            .copy()
        )


        if temp.empty:

            continue


        coefficients = (
            temp["Coefficient"]
            .dropna()
        )


        if coefficients.empty:

            continue


        positive_count = int(
            (
                coefficients > 0
            )
            .sum()
        )


        negative_count = int(
            (
                coefficients < 0
            )
            .sum()
        )


        significant_5_count = int(
            (
                temp["P_Value"]
                < 0.05
            )
            .sum()
        )


        significant_10_count = int(
            (
                temp["P_Value"]
                < 0.10
            )
            .sum()
        )


        number_estimates = int(
            len(
                coefficients
            )
        )


        rolling_stability_rows.append(
            {

                "Asset":
                    asset,

                "Variable":
                    variable,

                "Rolling_Estimates":
                    number_estimates,

                "Mean_Coefficient":
                    safe_float(
                        coefficients.mean()
                    ),

                "Median_Coefficient":
                    safe_float(
                        coefficients.median()
                    ),

                "Minimum_Coefficient":
                    safe_float(
                        coefficients.min()
                    ),

                "Maximum_Coefficient":
                    safe_float(
                        coefficients.max()
                    ),

                "Std_Coefficient":
                    safe_float(
                        coefficients.std(
                            ddof=1
                        )
                    ),

                "Positive_Windows":
                    positive_count,

                "Negative_Windows":
                    negative_count,

                "Sign_Changes_Across_Windows":
                    bool(
                        positive_count > 0
                        and
                        negative_count > 0
                    ),

                "Significant_5pct_Windows":
                    significant_5_count,

                "Significant_10pct_Windows":
                    significant_10_count,

                "Share_Significant_5pct":
                    safe_float(
                        significant_5_count
                        /
                        number_estimates
                    ),

                "Share_Significant_10pct":
                    safe_float(
                        significant_10_count
                        /
                        number_estimates
                    )

            }
        )


rolling_stability_summary = pd.DataFrame(
    rolling_stability_rows
)


print(
    "\n",
    rolling_stability_summary.to_string(
        index=False
    )
)


# =====================================================================
# 26. KEY ROLLING COEFFICIENT SUMMARY
# =====================================================================

section(
    "KEY ROLLING COEFFICIENT SUMMARY"
)


def print_key_rolling_summary(
    asset,
    variables
):

    temp = (
        rolling_stability_summary.loc[
            (
                rolling_stability_summary["Asset"]
                == asset
            )
            &
            (
                rolling_stability_summary["Variable"]
                .isin(
                    variables
                )
            )
        ]
        .copy()
    )


    print(
        f"\n{asset}:"
    )


    print(
        temp.to_string(
            index=False
        )
    )


print_key_rolling_summary(
    "BTC",
    BTC_KEY_COEFFICIENTS
)


print_key_rolling_summary(
    "ETH",
    ETH_KEY_COEFFICIENTS
)


# =====================================================================
# 27. KEY COEFFICIENT COMBINED STABILITY SUMMARY
# =====================================================================

section(
    "KEY COEFFICIENT COMBINED STABILITY SUMMARY"
)


key_stability_rows = []


for asset, variables in [

    (
        "BTC",
        BTC_KEY_COEFFICIENTS
    ),

    (
        "ETH",
        ETH_KEY_COEFFICIENTS
    )

]:

    for variable in variables:

        subsample_temp = (
            subsample_stability_summary.loc[
                (
                    subsample_stability_summary["Asset"]
                    == asset
                )
                &
                (
                    subsample_stability_summary["Variable"]
                    == variable
                )
            ]
        )


        rolling_temp = (
            rolling_stability_summary.loc[
                (
                    rolling_stability_summary["Asset"]
                    == asset
                )
                &
                (
                    rolling_stability_summary["Variable"]
                    == variable
                )
            ]
        )


        if (
            subsample_temp.empty
            or
            rolling_temp.empty
        ):

            continue


        key_stability_rows.append(
            {

                "Asset":
                    asset,

                "Variable":
                    variable,

                "Full_Sample_Coefficient":
                    safe_float(
                        subsample_temp[
                            "Full_Sample_Coefficient"
                        ]
                        .iloc[0]
                    ),

                "Annual_Sign_Changes":
                    bool(
                        subsample_temp[
                            "Sign_Changes_Across_Subsamples"
                        ]
                        .iloc[0]
                    ),

                "Annual_5pct_Significant_Periods":
                    int(
                        subsample_temp[
                            "Significant_at_5pct_Subsamples"
                        ]
                        .iloc[0]
                    ),

                "Annual_10pct_Significant_Periods":
                    int(
                        subsample_temp[
                            "Significant_at_10pct_Subsamples"
                        ]
                        .iloc[0]
                    ),

                "Share_Annual_Same_Sign_As_Full":
                    safe_float(
                        subsample_temp[
                            "Share_Same_Sign_As_Full"
                        ]
                        .iloc[0]
                    ),

                "Rolling_Sign_Changes":
                    bool(
                        rolling_temp[
                            "Sign_Changes_Across_Windows"
                        ]
                        .iloc[0]
                    ),

                "Rolling_5pct_Significant_Share":
                    safe_float(
                        rolling_temp[
                            "Share_Significant_5pct"
                        ]
                        .iloc[0]
                    ),

                "Rolling_10pct_Significant_Share":
                    safe_float(
                        rolling_temp[
                            "Share_Significant_10pct"
                        ]
                        .iloc[0]
                    )

            }
        )


key_stability_summary = pd.DataFrame(
    key_stability_rows
)


print(
    "\n",
    key_stability_summary.to_string(
        index=False
    )
)


# =====================================================================
# 28. SUBSAMPLE COVERAGE VALIDATION
# =====================================================================

section(
    "VALIDATING PREDETERMINED SUBSAMPLE COVERAGE"
)


expected_periods = set(
    SUBSAMPLES.keys()
)


btc_periods = set(
    subsample_models.loc[
        subsample_models["Asset"]
        == "BTC",
        "Period"
    ]
)


eth_periods = set(
    subsample_models.loc[
        subsample_models["Asset"]
        == "ETH",
        "Period"
    ]
)


btc_period_coverage_pass = (
    btc_periods
    ==
    expected_periods
)


eth_period_coverage_pass = (
    eth_periods
    ==
    expected_periods
)


print(
    "\nBTC periods estimated:"
)

print(
    sorted(
        btc_periods
    )
)

print(
    "Validation:",
    "PASS"
    if btc_period_coverage_pass
    else "FAIL"
)


print(
    "\nETH periods estimated:"
)

print(
    sorted(
        eth_periods
    )
)

print(
    "Validation:",
    "PASS"
    if eth_period_coverage_pass
    else "FAIL"
)


# =====================================================================
# 29. ROLLING OUTPUT VALIDATION
# =====================================================================

section(
    "VALIDATING ROLLING OUTPUT"
)


btc_rolling_pass = (
    not
    btc_rolling_models.empty
)


eth_rolling_pass = (
    not
    eth_rolling_models.empty
)


if btc_rolling_pass:

    btc_rolling_n_pass = bool(
        (
            btc_rolling_models["N"]
            >=
            MIN_ROLLING_OBSERVATIONS
        )
        .all()
    )

else:

    btc_rolling_n_pass = False


if eth_rolling_pass:

    eth_rolling_n_pass = bool(
        (
            eth_rolling_models["N"]
            >=
            MIN_ROLLING_OBSERVATIONS
        )
        .all()
    )

else:

    eth_rolling_n_pass = False


print(
    "\nBTC rolling models generated:"
)

print(
    "PASS"
    if btc_rolling_pass
    else "FAIL"
)


print(
    "\nETH rolling models generated:"
)

print(
    "PASS"
    if eth_rolling_pass
    else "FAIL"
)


print(
    "\nBTC minimum rolling-N requirement:"
)

print(
    "PASS"
    if btc_rolling_n_pass
    else "FAIL"
)


print(
    "\nETH minimum rolling-N requirement:"
)

print(
    "PASS"
    if eth_rolling_n_pass
    else "FAIL"
)


# =====================================================================
# 30. SAVE OUTPUTS
# =====================================================================

section(
    "SAVING STRUCTURAL-STABILITY OUTPUTS"
)


output_files = {

    "full_sample_coefficients.csv":
        full_sample_coefficients,

    "subsample_model_summary.csv":
        subsample_models,

    "subsample_coefficients.csv":
        subsample_coefficients,

    "subsample_vs_full_comparison.csv":
        subsample_comparison,

    "subsample_stability_summary.csv":
        subsample_stability_summary,

    "rolling_365d_model_summary.csv":
        rolling_models,

    "rolling_365d_coefficients.csv":
        rolling_coefficients,

    "rolling_365d_stability_summary.csv":
        rolling_stability_summary,

    "key_coefficient_stability_summary.csv":
        key_stability_summary

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
# 31. SAVE METHODOLOGY NOTE
# =====================================================================

section(
    "SAVING METHODOLOGY NOTE"
)


methodology_file = (
    OUTPUT_DIR
    /
    "structural_stability_methodology_note.txt"
)


methodology_note = f"""
MARKET-REGIME / STRUCTURAL-STABILITY ROBUSTNESS

PURPOSE
-------
The 2021-2025 cryptocurrency sample spans materially different market
environments. Structural-stability robustness therefore examines
whether relationships estimated in the primary explanatory regressions
remain stable through time.

PREDETERMINED SUBSAMPLES
------------------------
The subsample robustness exercise uses calendar-year periods:

2021
2022
2023
2024
2025

Calendar years were selected as predetermined and transparent sample
partitions.

The annual boundaries are not statistically estimated structural
breaks.

The analysis therefore avoids selecting break dates retrospectively
on the basis of coefficient estimates or statistical significance.

ROLLING COEFFICIENTS
--------------------
The baseline model is additionally estimated using a trailing
{ROLLING_WINDOW}-day rolling window.

Rolling regressions are evaluated every {ROLLING_STEP} days.

A rolling window must contain at least
{MIN_ROLLING_OBSERVATIONS} usable observations.

Rolling windows overlap substantially. Therefore, individual rolling
estimates should not be interpreted as independent statistical tests.

INFORMATION ALIGNMENT
---------------------
Traditional-market predictors use:

Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

The old non-aligned traditional-market variables are not used.

TRADING VOLUME
--------------
Bitcoin and Ethereum volume enter through:

Lagged_Log_BTC_Volume
Lagged_Log_ETH_Volume

The underlying volume transformation has been separately verified as:

log(1 + Volume)

INFERENCE
---------
Models use OLS coefficient estimates with HAC/Newey-West covariance
estimates.

HAC maximum lag:

{HAC_MAXLAGS}

INTERPRETATION
--------------
Coefficient stability should be assessed using:

1. coefficient signs;
2. coefficient magnitudes;
3. HAC standard errors;
4. confidence intervals;
5. statistical significance;
6. annual versus full-sample estimates; and
7. rolling coefficient behaviour.

A change in statistical significance alone does not establish a
structural break.

SCOPE
-----
This is an in-sample explanatory robustness analysis.

It does not establish out-of-sample predictive performance.

FUTURE SENTIMENT EXTENSION
--------------------------
Once Reddit sentiment becomes available, the same predetermined
subsample and rolling-coefficient framework can be applied to the
lagged sentiment coefficient.

This will allow the dissertation to examine whether any estimated
relationship between lagged Reddit sentiment and subsequent Bitcoin
or Ethereum returns is stable across different market periods.
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
# 32. SAVE MODEL SPECIFICATION NOTE
# =====================================================================

section(
    "SAVING MODEL SPECIFICATION NOTE"
)


specification_file = (
    OUTPUT_DIR
    /
    "structural_stability_model_specifications.txt"
)


specification_note = f"""
STRUCTURAL-STABILITY MODEL SPECIFICATIONS

BTC DEPENDENT VARIABLE
----------------------
{BTC_DEPENDENT}

BTC PREDICTORS
--------------
{chr(10).join(BTC_PREDICTORS)}

ETH DEPENDENT VARIABLE
----------------------
{ETH_DEPENDENT}

ETH PREDICTORS
--------------
{chr(10).join(ETH_PREDICTORS)}

PREDETERMINED SUBSAMPLES
------------------------
{chr(10).join(SUBSAMPLES.keys())}

ROLLING WINDOW
--------------
{ROLLING_WINDOW} daily observations

ROLLING STEP
------------
{ROLLING_STEP} days

MINIMUM ROLLING OBSERVATIONS
----------------------------
{MIN_ROLLING_OBSERVATIONS}

INFERENCE
---------
OLS coefficients with HAC/Newey-West covariance.

HAC MAXIMUM LAG
---------------
{HAC_MAXLAGS}

CONFIDENCE LEVEL
----------------
{CONFIDENCE_LEVEL}

PRIMARY SIGNIFICANCE LEVEL
--------------------------
{SIGNIFICANCE_LEVEL}

IMPORTANT
---------
These are in-sample explanatory robustness models.

They are not out-of-sample forecasts.

Calendar-year boundaries are predetermined robustness partitions,
not statistically estimated structural breaks.

Rolling windows overlap and should not be interpreted as independent
hypothesis tests.
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
# 33. FULL-SAMPLE VALIDATION
# =====================================================================

section(
    "VALIDATING FULL-SAMPLE REFERENCE MODELS"
)


btc_full_n_pass = (
    int(
        btc_full_ols.nobs
    )
    ==
    len(
        btc_sample
    )
)


eth_full_n_pass = (
    int(
        eth_full_ols.nobs
    )
    ==
    len(
        eth_sample
    )
)


print(
    "\nBTC full-sample N:"
)

print(
    int(
        btc_full_ols.nobs
    )
)

print(
    "Validation:",
    "PASS"
    if btc_full_n_pass
    else "FAIL"
)


print(
    "\nETH full-sample N:"
)

print(
    int(
        eth_full_ols.nobs
    )
)

print(
    "Validation:",
    "PASS"
    if eth_full_n_pass
    else "FAIL"
)


# =====================================================================
# 34. FINAL VALIDATION
# =====================================================================

section(
    "FINAL VALIDATION"
)


validation_checks = {

    "Input dataset exists":
        INPUT_FILE.exists(),

    "No invalid dates":
        (
            invalid_dates == 0
        ),

    "No duplicate dates":
        (
            duplicate_dates == 0
        ),

    "Continuous daily calendar":
        (
            calendar_gaps == 0
        ),

    "BTC full model estimated":
        (
            btc_full_ols.nobs > 0
        ),

    "ETH full model estimated":
        (
            eth_full_ols.nobs > 0
        ),

    "BTC full-sample N validated":
        btc_full_n_pass,

    "ETH full-sample N validated":
        eth_full_n_pass,

    "BTC all predetermined subsamples estimated":
        btc_period_coverage_pass,

    "ETH all predetermined subsamples estimated":
        eth_period_coverage_pass,

    "BTC rolling models generated":
        btc_rolling_pass,

    "ETH rolling models generated":
        eth_rolling_pass,

    "BTC rolling sample sizes valid":
        btc_rolling_n_pass,

    "ETH rolling sample sizes valid":
        eth_rolling_n_pass,

    "Full-sample coefficient output non-empty":
        (
            not
            full_sample_coefficients.empty
        ),

    "Subsample model output non-empty":
        (
            not
            subsample_models.empty
        ),

    "Subsample coefficient output non-empty":
        (
            not
            subsample_coefficients.empty
        ),

    "Subsample comparison output non-empty":
        (
            not
            subsample_comparison.empty
        ),

    "Subsample stability summary non-empty":
        (
            not
            subsample_stability_summary.empty
        ),

    "Rolling model output non-empty":
        (
            not
            rolling_models.empty
        ),

    "Rolling coefficient output non-empty":
        (
            not
            rolling_coefficients.empty
        ),

    "Rolling stability summary non-empty":
        (
            not
            rolling_stability_summary.empty
        ),

    "Key stability summary non-empty":
        (
            not
            key_stability_summary.empty
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
    "\nOVERALL MARKET-REGIME / "
    "STRUCTURAL-STABILITY VALIDATION:"
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
        "\nFailed validation checks:"
    )


    for check in failed_checks:

        print(
            " -",
            check
        )


    raise ValueError(
        "\nOne or more structural-stability "
        "validation checks failed."
    )


# =====================================================================
# 35. INTERPRETATION REMINDERS
# =====================================================================

section(
    "INTERPRETATION REMINDERS"
)


print(
    "\n1. Calendar-year subsamples are predetermined robustness "
    "periods, not statistically estimated break dates."
)


print(
    "\n2. Do not call a coefficient structurally unstable merely "
    "because it is significant in one year and insignificant "
    "in another."
)


print(
    "\n3. Examine coefficient signs, magnitudes, confidence "
    "intervals and temporal patterns together."
)


print(
    "\n4. Rolling coefficients are supporting stability evidence, "
    "not separate independent hypothesis tests."
)


print(
    "\n5. The 365-day rolling windows overlap substantially."
)


print(
    "\n6. HAC/Newey-West inference with maxlags=7 is retained "
    "throughout."
)


print(
    "\n7. Only information-aligned traditional-market controls "
    "are used."
)


print(
    "\n8. The primary model remains the full-sample t-1 "
    "explanatory specification."
)


print(
    "\n9. This robustness analysis is IN-SAMPLE. It does not "
    "establish out-of-sample predictive performance."
)


print(
    "\n10. Once Reddit sentiment is available, repeat this "
    "framework for the lagged sentiment coefficient."
)


# =====================================================================
# 36. COMPLETE
# =====================================================================

section(
    "MARKET-REGIME / STRUCTURAL-STABILITY ROBUSTNESS COMPLETE"
)


print(
    "\nAll structural-stability robustness procedures "
    "completed successfully."
)


print(
    "\nOutput directory:"
)

print(
    OUTPUT_DIR
)


print(
    "\nOverall validation:"
)

print(
    "PASS"
)