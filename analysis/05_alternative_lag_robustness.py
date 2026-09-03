# =====================================================================
# 05_alternative_lag_robustness.py
#
# FINAL ALTERNATIVE LAG-LENGTH ROBUSTNESS ANALYSIS
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
# Test whether the explanatory results are sensitive to the choice
# of a one-day lag.
#
# Calendar-day information horizons:
#
#       t-1   PRIMARY SPECIFICATION
#       t-2   ROBUSTNESS
#       t-3   ROBUSTNESS
#       t-7   ROBUSTNESS
#
# IMPORTANT DESIGN FEATURES
# ---------------------------------------------------------------------
# 1. t-1 remains the PRIMARY specification.
#
# 2. t-2, t-3 and t-7 are robustness / sensitivity checks.
#
# 3. Lag horizons are estimated in SEPARATE regressions.
#    We do NOT include all four lag lengths simultaneously.
#
# 4. Cryptocurrency variables use calendar-day lags because crypto
#    trades seven days per week.
#
# 5. Traditional-market predictors are constructed from the already
#    INFORMATION-ALIGNED t-1 variables.
#
# 6. For traditional-market controls:
#
#       t-1 = aligned t-1 information set
#       t-2 = aligned series shifted by 1 calendar day
#       t-3 = aligned series shifted by 2 calendar days
#       t-7 = aligned series shifted by 6 calendar days
#
#    These should therefore be described as alternative
#    CALENDAR-DAY INFORMATION HORIZONS.
#
# 7. ALL lag specifications for a given asset are estimated on the
#    SAME COMMON SAMPLE.
#
#    The common sample is determined by the most demanding complete-
#    case requirements across t-1, t-2, t-3 and t-7, including the
#    cross-crypto specification.
#
#    This means differences across lag specifications cannot be
#    attributed to different regression samples.
#
# 8. OLS coefficients are estimated with HAC / Newey-West
#    standard errors using maxlags = 7.
#
# 9. These are IN-SAMPLE EXPLANATORY robustness regressions.
#    They are NOT out-of-sample forecasting models.
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
    / "alternative_lag_robustness"
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

LAG_LENGTHS = [
    1,
    2,
    3,
    7
]

PRIMARY_LAG = 1

HAC_MAXLAGS = 7


# =====================================================================
# 3. HELPER
# =====================================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =====================================================================
# 4. LOAD DATA
# =====================================================================

section(
    "FINAL ALTERNATIVE LAG-LENGTH ROBUSTNESS ANALYSIS"
)

print("\nInput file:")
print(INPUT_FILE)

print("\nDoes input file exist?")
print(INPUT_FILE.exists())


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find:\n{INPUT_FILE}"
    )


df = pd.read_csv(
    INPUT_FILE
)


print("\nDataset shape:")
print(df.shape)


# =====================================================================
# 5. DATE VALIDATION
# =====================================================================

section(
    "VALIDATING DATES"
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


print("\nInvalid dates:")
print(invalid_dates)

print("\nDuplicate dates:")
print(duplicate_dates)


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


print("\nDate range:")

print(
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =====================================================================
# 6. VERIFY CONSECUTIVE DAILY CRYPTO CALENDAR
# =====================================================================

section(
    "VERIFYING DAILY CRYPTO CALENDAR"
)


date_difference = (
    df["Date"]
    .diff()
)


non_daily_gap_mask = (
    date_difference.notna()
    &
    (
        date_difference
        != pd.Timedelta(days=1)
    )
)


number_of_gaps = int(
    non_daily_gap_mask.sum()
)


print("\nTotal observations:")
print(len(df))

print("\nNon-consecutive calendar gaps:")
print(number_of_gaps)


if number_of_gaps > 0:

    gap_rows = df.loc[
        non_daily_gap_mask,
        ["Date"]
    ].copy()

    gap_rows["Previous_Date"] = (
        df["Date"]
        .shift(1)
        .loc[
            non_daily_gap_mask
        ]
        .values
    )

    gap_rows["Gap_Days"] = (
        gap_rows["Date"]
        -
        gap_rows["Previous_Date"]
    ).dt.days

    print("\nCalendar gaps:")
    print(
        gap_rows
        .to_string(index=False)
    )

    raise ValueError(
        "\nDataset is not a consecutive daily calendar. "
        "Calendar-day lag construction should not proceed."
    )


print(
    "\nPASS: Dataset follows a consecutive daily calendar."
)


# =====================================================================
# 7. CALENDAR VARIABLES
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


print("\nWeekend observations:")

print(
    int(
        df["Is_Weekend"]
        .sum()
    )
)


print("\nObservations by day:")

print(
    df["Day_Name"]
    .value_counts()
    .reindex(DAY_ORDER)
)


# =====================================================================
# 8. SOURCE VARIABLES
# =====================================================================

section(
    "CHECKING SOURCE VARIABLES"
)


REQUIRED_SOURCE_VARIABLES = [

    # Crypto returns
    "BTC_Return",
    "ETH_Return",

    # Logged crypto volume
    "Log_BTC_Volume",
    "Log_ETH_Volume",

    # Existing primary crypto lags used for validation
    "BTC_Lagged_Return",
    "ETH_Lagged_Return",

    "Lagged_Log_BTC_Volume",
    "Lagged_Log_ETH_Volume",

    # Correct information-aligned traditional-market predictors
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned"

]


missing_variables = []


for variable in REQUIRED_SOURCE_VARIABLES:

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
        "\nMissing required variables:\n"
        +
        "\n".join(
            missing_variables
        )
    )


print(
    "\nPASS: All required source variables are available."
)


# =====================================================================
# 9. NUMERIC CONVERSION
# =====================================================================

section(
    "VALIDATING NUMERIC SOURCE VARIABLES"
)


for variable in REQUIRED_SOURCE_VARIABLES:

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
# 10. INFORMATION-ALIGNED MARKET VARIABLES
# =====================================================================

ALIGNED_MARKET_VARIABLES = {

    "SP500":
        "Lagged_SP500_Return_Aligned",

    "VIX":
        "Lagged_VIX_Change_Aligned",

    "Gold":
        "Lagged_Gold_Return_Aligned",

    "DXY":
        "Lagged_DXY_Return_Aligned",

    "US10Y":
        "Lagged_US10Y_Change_Aligned"

}


# =====================================================================
# 11. SAFETY CHECK AGAINST OLD NON-ALIGNED VARIABLES
# =====================================================================

section(
    "INFORMATION-ALIGNMENT SAFETY CHECK"
)


OLD_NON_ALIGNED_VARIABLES = [

    "Lagged_SP500_Return",

    "Lagged_VIX_Change",

    "Lagged_Gold_Return",

    "Lagged_DXY_Return",

    "Lagged_US10Y_Change"

]


print(
    "\nThe following OLD columns may exist in the dataset, "
    "but they will NOT be used:"
)


for variable in OLD_NON_ALIGNED_VARIABLES:

    print(
        " -",
        variable
    )


print(
    "\nThe analysis uses only:"
)


for variable in ALIGNED_MARKET_VARIABLES.values():

    print(
        " -",
        variable
    )


# =====================================================================
# 12. CONSTRUCT CRYPTO RETURN LAGS
# =====================================================================

section(
    "CONSTRUCTING CRYPTO RETURN LAGS"
)


for lag in LAG_LENGTHS:

    btc_variable = (
        f"BTC_Return_Lag_{lag}"
    )

    eth_variable = (
        f"ETH_Return_Lag_{lag}"
    )


    df[btc_variable] = (
        df["BTC_Return"]
        .shift(lag)
    )


    df[eth_variable] = (
        df["ETH_Return"]
        .shift(lag)
    )


    print(
        f"Created {btc_variable}"
    )

    print(
        f"Created {eth_variable}"
    )


# =====================================================================
# 13. VALIDATE PRIMARY RETURN LAGS
# =====================================================================

section(
    "VALIDATING PRIMARY t-1 RETURN LAGS"
)


btc_return_validation = pd.concat(

    [
        df["BTC_Lagged_Return"],
        df["BTC_Return_Lag_1"]
    ],

    axis=1

).dropna()


btc_return_difference = (

    btc_return_validation.iloc[:, 0]
    -
    btc_return_validation.iloc[:, 1]

).abs().max()


eth_return_validation = pd.concat(

    [
        df["ETH_Lagged_Return"],
        df["ETH_Return_Lag_1"]
    ],

    axis=1

).dropna()


eth_return_difference = (

    eth_return_validation.iloc[:, 0]
    -
    eth_return_validation.iloc[:, 1]

).abs().max()


print(
    "\nMaximum BTC t-1 difference:"
)

print(
    btc_return_difference
)


print(
    "\nMaximum ETH t-1 difference:"
)

print(
    eth_return_difference
)


if not np.isclose(
    btc_return_difference,
    0.0,
    atol=1e-12
):

    raise ValueError(
        "Constructed BTC t-1 return lag does not match "
        "BTC_Lagged_Return."
    )


if not np.isclose(
    eth_return_difference,
    0.0,
    atol=1e-12
):

    raise ValueError(
        "Constructed ETH t-1 return lag does not match "
        "ETH_Lagged_Return."
    )


print(
    "\nPASS: Constructed t-1 return lags exactly reproduce "
    "the existing primary return lags."
)


# =====================================================================
# 14. CONSTRUCT LOG-VOLUME LAGS
# =====================================================================

section(
    "CONSTRUCTING LOG-VOLUME LAGS"
)


for lag in LAG_LENGTHS:

    btc_variable = (
        f"Log_BTC_Volume_Lag_{lag}"
    )

    eth_variable = (
        f"Log_ETH_Volume_Lag_{lag}"
    )


    df[btc_variable] = (
        df["Log_BTC_Volume"]
        .shift(lag)
    )


    df[eth_variable] = (
        df["Log_ETH_Volume"]
        .shift(lag)
    )


    print(
        f"Created {btc_variable}"
    )

    print(
        f"Created {eth_variable}"
    )


# =====================================================================
# 15. VALIDATE PRIMARY VOLUME LAGS
# =====================================================================

section(
    "VALIDATING PRIMARY t-1 VOLUME LAGS"
)


btc_volume_validation = pd.concat(

    [
        df["Lagged_Log_BTC_Volume"],
        df["Log_BTC_Volume_Lag_1"]
    ],

    axis=1

).dropna()


btc_volume_difference = (

    btc_volume_validation.iloc[:, 0]
    -
    btc_volume_validation.iloc[:, 1]

).abs().max()


eth_volume_validation = pd.concat(

    [
        df["Lagged_Log_ETH_Volume"],
        df["Log_ETH_Volume_Lag_1"]
    ],

    axis=1

).dropna()


eth_volume_difference = (

    eth_volume_validation.iloc[:, 0]
    -
    eth_volume_validation.iloc[:, 1]

).abs().max()


print(
    "\nMaximum BTC volume t-1 difference:"
)

print(
    btc_volume_difference
)


print(
    "\nMaximum ETH volume t-1 difference:"
)

print(
    eth_volume_difference
)


if not np.isclose(
    btc_volume_difference,
    0.0,
    atol=1e-12
):

    raise ValueError(
        "Constructed BTC t-1 volume lag does not match "
        "Lagged_Log_BTC_Volume."
    )


if not np.isclose(
    eth_volume_difference,
    0.0,
    atol=1e-12
):

    raise ValueError(
        "Constructed ETH t-1 volume lag does not match "
        "Lagged_Log_ETH_Volume."
    )


print(
    "\nPASS: Constructed t-1 volume lags exactly reproduce "
    "the existing primary volume lags."
)


# =====================================================================
# 16. CONSTRUCT ALTERNATIVE ALIGNED MARKET INFORMATION HORIZONS
# =====================================================================
#
# IMPORTANT INTERPRETATION
# ---------------------------------------------------------------------
# The existing *_Aligned variable represents the information available
# strictly before the crypto return date under the primary t-1 design.
#
# Therefore:
#
# t-1 -> aligned series, no additional shift
# t-2 -> aligned series shifted 1 additional calendar day
# t-3 -> aligned series shifted 2 additional calendar days
# t-7 -> aligned series shifted 6 additional calendar days
#
# These are CALENDAR-DAY INFORMATION HORIZONS.
#
# They should not be described as "second previous trading day",
# "third previous trading day", etc.
# =====================================================================

section(
    "CONSTRUCTING ALTERNATIVE ALIGNED MARKET INFORMATION HORIZONS"
)


for short_name, source_variable in ALIGNED_MARKET_VARIABLES.items():

    for lag in LAG_LENGTHS:

        new_variable = (
            f"{short_name}_Aligned_Lag_{lag}"
        )


        additional_shift = (
            lag - 1
        )


        df[new_variable] = (
            df[source_variable]
            .shift(
                additional_shift
            )
        )


        print(
            f"Created {new_variable} "
            f"| source = {source_variable} "
            f"| additional calendar shift = "
            f"{additional_shift}"
        )


# =====================================================================
# 17. VALIDATE PRIMARY ALIGNED MARKET VARIABLES
# =====================================================================

section(
    "VALIDATING PRIMARY t-1 ALIGNED MARKET VARIABLES"
)


for short_name, source_variable in ALIGNED_MARKET_VARIABLES.items():

    generated_variable = (
        f"{short_name}_Aligned_Lag_1"
    )


    validation = pd.concat(

        [
            df[source_variable],
            df[generated_variable]
        ],

        axis=1

    ).dropna()


    max_difference = (

        validation.iloc[:, 0]
        -
        validation.iloc[:, 1]

    ).abs().max()


    print(
        f"{short_name}: "
        f"maximum t-1 difference = "
        f"{max_difference}"
    )


    if not np.isclose(
        max_difference,
        0.0,
        atol=1e-12
    ):

        raise ValueError(
            f"{short_name} constructed t-1 aligned variable "
            f"does not reproduce {source_variable}."
        )


print(
    "\nPASS: All constructed t-1 aligned market variables "
    "exactly reproduce the existing primary aligned variables."
)


# =====================================================================
# 18. MODEL-PREDICTOR FUNCTION
# =====================================================================

def get_predictors(
    asset,
    lag,
    cross_crypto=False
):

    market_predictors = [

        f"SP500_Aligned_Lag_{lag}",

        f"VIX_Aligned_Lag_{lag}",

        f"Gold_Aligned_Lag_{lag}",

        f"DXY_Aligned_Lag_{lag}",

        f"US10Y_Aligned_Lag_{lag}"

    ]


    if asset == "BTC":

        predictors = [

            f"BTC_Return_Lag_{lag}",

            f"Log_BTC_Volume_Lag_{lag}"

        ]


        if cross_crypto:

            predictors.insert(
                1,
                f"ETH_Return_Lag_{lag}"
            )


    elif asset == "ETH":

        predictors = [

            f"ETH_Return_Lag_{lag}",

            f"Log_ETH_Volume_Lag_{lag}"

        ]


        if cross_crypto:

            predictors.insert(
                1,
                f"BTC_Return_Lag_{lag}"
            )


    else:

        raise ValueError(
            "Asset must be BTC or ETH."
        )


    return (
        predictors
        +
        market_predictors
    )


# =====================================================================
# 19. DISPLAY MODEL SPECIFICATIONS
# =====================================================================

section(
    "MODEL SPECIFICATIONS BY CALENDAR-DAY INFORMATION HORIZON"
)


for lag in LAG_LENGTHS:

    print(
        "\n" + "-" * 80
    )

    print(
        f"t-{lag} CALENDAR-DAY INFORMATION HORIZON"
    )


    if lag == PRIMARY_LAG:

        print(
            "PRIMARY SPECIFICATION"
        )

    else:

        print(
            "ROBUSTNESS SPECIFICATION"
        )


    for asset in [
        "BTC",
        "ETH"
    ]:

        print(
            f"\n{asset} baseline predictors:"
        )


        for variable in get_predictors(
            asset,
            lag,
            cross_crypto=False
        ):

            print(
                " -",
                variable
            )


        print(
            f"\n{asset} cross-crypto predictors:"
        )


        for variable in get_predictors(
            asset,
            lag,
            cross_crypto=True
        ):

            print(
                " -",
                variable
            )


# =====================================================================
# 20. BUILD COMMON SAMPLE REQUIREMENTS
# =====================================================================
#
# IMPROVEMENT:
#
# Rather than allowing:
#
# t-1 N = 1821
# t-2 N = 1820
# t-3 N = 1819
# t-7 N = 1815
#
# we determine one common complete-case sample for each asset using
# ALL variables needed across ALL lag horizons.
#
# Baseline and cross-crypto models also use the same observations.
# =====================================================================

section(
    "BUILDING COMMON SAMPLE ACROSS ALL LAG HORIZONS"
)


btc_all_predictors = []

eth_all_predictors = []


for lag in LAG_LENGTHS:

    btc_all_predictors.extend(

        get_predictors(
            "BTC",
            lag,
            cross_crypto=True
        )

    )


    eth_all_predictors.extend(

        get_predictors(
            "ETH",
            lag,
            cross_crypto=True
        )

    )


btc_all_predictors = list(
    dict.fromkeys(
        btc_all_predictors
    )
)


eth_all_predictors = list(
    dict.fromkeys(
        eth_all_predictors
    )
)


print(
    "\nNumber of unique BTC common-sample predictors:"
)

print(
    len(btc_all_predictors)
)


print(
    "\nNumber of unique ETH common-sample predictors:"
)

print(
    len(eth_all_predictors)
)


# =====================================================================
# 21. CREATE COMMON BTC SAMPLE
# =====================================================================

btc_required_columns = list(
    dict.fromkeys(

        [
            "Date",
            "Day_Name",
            "Is_Weekend",
            "BTC_Return"
        ]

        +

        btc_all_predictors

    )
)


btc_common_sample = (

    df[
        btc_required_columns
    ]

    .dropna(
        subset=[

            "BTC_Return"

        ]

        +

        btc_all_predictors
    )

    .copy()

)


if btc_common_sample.empty:

    raise ValueError(
        "No observations remain in the BTC common sample."
    )


# =====================================================================
# 22. CREATE COMMON ETH SAMPLE
# =====================================================================

eth_required_columns = list(
    dict.fromkeys(

        [
            "Date",
            "Day_Name",
            "Is_Weekend",
            "ETH_Return"
        ]

        +

        eth_all_predictors

    )
)


eth_common_sample = (

    df[
        eth_required_columns
    ]

    .dropna(
        subset=[

            "ETH_Return"

        ]

        +

        eth_all_predictors
    )

    .copy()

)


if eth_common_sample.empty:

    raise ValueError(
        "No observations remain in the ETH common sample."
    )


# =====================================================================
# 23. COMMON-SAMPLE DIAGNOSTICS
# =====================================================================

section(
    "COMMON-SAMPLE DIAGNOSTICS"
)


print("\nBTC common-sample observations:")

print(
    len(
        btc_common_sample
    )
)


print("\nBTC common-sample date range:")

print(
    btc_common_sample["Date"].min(),
    "to",
    btc_common_sample["Date"].max()
)


print("\nBTC common-sample weekend observations:")

print(
    int(
        btc_common_sample[
            "Is_Weekend"
        ].sum()
    )
)


print("\nBTC observations by day:")

print(
    btc_common_sample[
        "Day_Name"
    ]
    .value_counts()
    .reindex(DAY_ORDER)
)


print("\nETH common-sample observations:")

print(
    len(
        eth_common_sample
    )
)


print("\nETH common-sample date range:")

print(
    eth_common_sample["Date"].min(),
    "to",
    eth_common_sample["Date"].max()
)


print("\nETH common-sample weekend observations:")

print(
    int(
        eth_common_sample[
            "Is_Weekend"
        ].sum()
    )
)


print("\nETH observations by day:")

print(
    eth_common_sample[
        "Day_Name"
    ]
    .value_counts()
    .reindex(DAY_ORDER)
)


# =====================================================================
# 24. CHECK WHETHER BTC AND ETH COMMON DATES ARE IDENTICAL
# =====================================================================

section(
    "BTC / ETH COMMON-DATE COMPARABILITY"
)


btc_dates = set(
    btc_common_sample["Date"]
)

eth_dates = set(
    eth_common_sample["Date"]
)


same_asset_dates = (
    btc_dates
    ==
    eth_dates
)


print(
    "\nBTC and ETH common samples use identical dates:"
)

print(
    same_asset_dates
)


print(
    "\nBTC-only dates:"
)

print(
    len(
        btc_dates
        -
        eth_dates
    )
)


print(
    "\nETH-only dates:"
)

print(
    len(
        eth_dates
        -
        btc_dates
    )
)


# =====================================================================
# 25. HAC MODEL FUNCTION
# =====================================================================

def estimate_hac_model(
    data,
    dependent,
    predictors
):

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


    return model


# =====================================================================
# 26. RESULT CONTAINERS
# =====================================================================

model_results = []

coefficient_results = []

cross_crypto_results = []

own_lag_results = []

sample_results = []


# =====================================================================
# 27. ESTIMATE ALL MODELS ON COMMON SAMPLES
# =====================================================================

section(
    "ESTIMATING COMMON-SAMPLE ALTERNATIVE-LAG MODELS"
)


for lag in LAG_LENGTHS:

    print(
        "\n" + "=" * 80
    )

    print(
        f"ESTIMATING t-{lag} CALENDAR-DAY INFORMATION HORIZON"
    )

    print(
        "=" * 80
    )


    # -----------------------------------------------------------------
    # BTC predictors
    # -----------------------------------------------------------------

    btc_baseline_predictors = get_predictors(
        "BTC",
        lag,
        cross_crypto=False
    )


    btc_cross_predictors = get_predictors(
        "BTC",
        lag,
        cross_crypto=True
    )


    # -----------------------------------------------------------------
    # ETH predictors
    # -----------------------------------------------------------------

    eth_baseline_predictors = get_predictors(
        "ETH",
        lag,
        cross_crypto=False
    )


    eth_cross_predictors = get_predictors(
        "ETH",
        lag,
        cross_crypto=True
    )


    # -----------------------------------------------------------------
    # Estimate BTC models
    # -----------------------------------------------------------------

    btc_baseline_model = estimate_hac_model(

        btc_common_sample,

        "BTC_Return",

        btc_baseline_predictors

    )


    btc_cross_model = estimate_hac_model(

        btc_common_sample,

        "BTC_Return",

        btc_cross_predictors

    )


    # -----------------------------------------------------------------
    # Estimate ETH models
    # -----------------------------------------------------------------

    eth_baseline_model = estimate_hac_model(

        eth_common_sample,

        "ETH_Return",

        eth_baseline_predictors

    )


    eth_cross_model = estimate_hac_model(

        eth_common_sample,

        "ETH_Return",

        eth_cross_predictors

    )


    # -----------------------------------------------------------------
    # Validation: same N for every model
    # -----------------------------------------------------------------

    print(
        f"\nt-{lag} BTC baseline N:"
    )

    print(
        int(
            btc_baseline_model.nobs
        )
    )


    print(
        f"t-{lag} BTC cross-crypto N:"
    )

    print(
        int(
            btc_cross_model.nobs
        )
    )


    print(
        f"t-{lag} ETH baseline N:"
    )

    print(
        int(
            eth_baseline_model.nobs
        )
    )


    print(
        f"t-{lag} ETH cross-crypto N:"
    )

    print(
        int(
            eth_cross_model.nobs
        )
    )


    # =================================================================
    # MODEL-LEVEL RESULTS
    # =================================================================

    models_to_store = [

        (
            "BTC",
            "Baseline",
            btc_baseline_model
        ),

        (
            "BTC",
            "Cross_Crypto",
            btc_cross_model
        ),

        (
            "ETH",
            "Baseline",
            eth_baseline_model
        ),

        (
            "ETH",
            "Cross_Crypto",
            eth_cross_model
        )

    ]


    for asset, model_type, model in models_to_store:

        model_results.append(

            {

                "Asset":
                    asset,

                "Lag_Days":
                    lag,

                "Information_Horizon":
                    f"t-{lag}",

                "Specification":
                    (
                        "Primary"
                        if lag == PRIMARY_LAG
                        else "Robustness"
                    ),

                "Model":
                    model_type,

                "N":
                    int(
                        model.nobs
                    ),

                "R_squared":
                    model.rsquared,

                "Adjusted_R_squared":
                    model.rsquared_adj,

                "AIC":
                    model.aic,

                "BIC":
                    model.bic

            }

        )


        confidence = (
            model
            .conf_int(
                alpha=0.05
            )
        )


        for variable in model.params.index:

            coefficient_results.append(

                {

                    "Asset":
                        asset,

                    "Lag_Days":
                        lag,

                    "Information_Horizon":
                        f"t-{lag}",

                    "Specification":
                        (
                            "Primary"
                            if lag == PRIMARY_LAG
                            else "Robustness"
                        ),

                    "Model":
                        model_type,

                    "Variable":
                        variable,

                    "Coefficient":
                        model.params[
                            variable
                        ],

                    "HAC_Std_Error":
                        model.bse[
                            variable
                        ],

                    "z_statistic":
                        model.tvalues[
                            variable
                        ],

                    "p_value":
                        model.pvalues[
                            variable
                        ],

                    "CI_95_Lower":
                        confidence.loc[
                            variable,
                            0
                        ],

                    "CI_95_Upper":
                        confidence.loc[
                            variable,
                            1
                        ]

                }

            )


    # =================================================================
    # CROSS-CRYPTO RESULTS
    # =================================================================

    btc_cross_variable = (
        f"ETH_Return_Lag_{lag}"
    )


    eth_cross_variable = (
        f"BTC_Return_Lag_{lag}"
    )


    btc_cross_ci = (
        btc_cross_model
        .conf_int()
        .loc[
            btc_cross_variable
        ]
    )


    eth_cross_ci = (
        eth_cross_model
        .conf_int()
        .loc[
            eth_cross_variable
        ]
    )


    cross_crypto_results.append(

        {

            "Asset":
                "BTC",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "Specification":
                (
                    "Primary"
                    if lag == PRIMARY_LAG
                    else "Robustness"
                ),

            "Cross_Crypto_Variable":
                btc_cross_variable,

            "Coefficient":
                btc_cross_model.params[
                    btc_cross_variable
                ],

            "HAC_Std_Error":
                btc_cross_model.bse[
                    btc_cross_variable
                ],

            "z_statistic":
                btc_cross_model.tvalues[
                    btc_cross_variable
                ],

            "p_value":
                btc_cross_model.pvalues[
                    btc_cross_variable
                ],

            "CI_95_Lower":
                btc_cross_ci.iloc[0],

            "CI_95_Upper":
                btc_cross_ci.iloc[1],

            "Baseline_R2":
                btc_baseline_model.rsquared,

            "Cross_Crypto_R2":
                btc_cross_model.rsquared,

            "Change_in_R2":
                (
                    btc_cross_model.rsquared
                    -
                    btc_baseline_model.rsquared
                ),

            "Baseline_Adjusted_R2":
                btc_baseline_model.rsquared_adj,

            "Cross_Crypto_Adjusted_R2":
                btc_cross_model.rsquared_adj,

            "Change_in_Adjusted_R2":
                (
                    btc_cross_model.rsquared_adj
                    -
                    btc_baseline_model.rsquared_adj
                ),

            "N":
                int(
                    btc_cross_model.nobs
                )

        }

    )


    cross_crypto_results.append(

        {

            "Asset":
                "ETH",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "Specification":
                (
                    "Primary"
                    if lag == PRIMARY_LAG
                    else "Robustness"
                ),

            "Cross_Crypto_Variable":
                eth_cross_variable,

            "Coefficient":
                eth_cross_model.params[
                    eth_cross_variable
                ],

            "HAC_Std_Error":
                eth_cross_model.bse[
                    eth_cross_variable
                ],

            "z_statistic":
                eth_cross_model.tvalues[
                    eth_cross_variable
                ],

            "p_value":
                eth_cross_model.pvalues[
                    eth_cross_variable
                ],

            "CI_95_Lower":
                eth_cross_ci.iloc[0],

            "CI_95_Upper":
                eth_cross_ci.iloc[1],

            "Baseline_R2":
                eth_baseline_model.rsquared,

            "Cross_Crypto_R2":
                eth_cross_model.rsquared,

            "Change_in_R2":
                (
                    eth_cross_model.rsquared
                    -
                    eth_baseline_model.rsquared
                ),

            "Baseline_Adjusted_R2":
                eth_baseline_model.rsquared_adj,

            "Cross_Crypto_Adjusted_R2":
                eth_cross_model.rsquared_adj,

            "Change_in_Adjusted_R2":
                (
                    eth_cross_model.rsquared_adj
                    -
                    eth_baseline_model.rsquared_adj
                ),

            "N":
                int(
                    eth_cross_model.nobs
                )

        }

    )


    # =================================================================
    # OWN-RETURN-LAG RESULTS
    # =================================================================

    btc_own_variable = (
        f"BTC_Return_Lag_{lag}"
    )


    eth_own_variable = (
        f"ETH_Return_Lag_{lag}"
    )


    btc_own_ci = (
        btc_baseline_model
        .conf_int()
        .loc[
            btc_own_variable
        ]
    )


    eth_own_ci = (
        eth_baseline_model
        .conf_int()
        .loc[
            eth_own_variable
        ]
    )


    own_lag_results.append(

        {

            "Asset":
                "BTC",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "Specification":
                (
                    "Primary"
                    if lag == PRIMARY_LAG
                    else "Robustness"
                ),

            "Variable":
                btc_own_variable,

            "Coefficient":
                btc_baseline_model.params[
                    btc_own_variable
                ],

            "HAC_Std_Error":
                btc_baseline_model.bse[
                    btc_own_variable
                ],

            "z_statistic":
                btc_baseline_model.tvalues[
                    btc_own_variable
                ],

            "p_value":
                btc_baseline_model.pvalues[
                    btc_own_variable
                ],

            "CI_95_Lower":
                btc_own_ci.iloc[0],

            "CI_95_Upper":
                btc_own_ci.iloc[1],

            "N":
                int(
                    btc_baseline_model.nobs
                )

        }

    )


    own_lag_results.append(

        {

            "Asset":
                "ETH",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "Specification":
                (
                    "Primary"
                    if lag == PRIMARY_LAG
                    else "Robustness"
                ),

            "Variable":
                eth_own_variable,

            "Coefficient":
                eth_baseline_model.params[
                    eth_own_variable
                ],

            "HAC_Std_Error":
                eth_baseline_model.bse[
                    eth_own_variable
                ],

            "z_statistic":
                eth_baseline_model.tvalues[
                    eth_own_variable
                ],

            "p_value":
                eth_baseline_model.pvalues[
                    eth_own_variable
                ],

            "CI_95_Lower":
                eth_own_ci.iloc[0],

            "CI_95_Upper":
                eth_own_ci.iloc[1],

            "N":
                int(
                    eth_baseline_model.nobs
                )

        }

    )


    # =================================================================
    # SAMPLE RESULTS
    # =================================================================

    sample_results.append(

        {

            "Asset":
                "BTC",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "N":
                len(
                    btc_common_sample
                ),

            "Start_Date":
                btc_common_sample[
                    "Date"
                ].min(),

            "End_Date":
                btc_common_sample[
                    "Date"
                ].max(),

            "Weekend_Observations":
                int(
                    btc_common_sample[
                        "Is_Weekend"
                    ].sum()
                ),

            "Common_Sample":
                True

        }

    )


    sample_results.append(

        {

            "Asset":
                "ETH",

            "Lag_Days":
                lag,

            "Information_Horizon":
                f"t-{lag}",

            "N":
                len(
                    eth_common_sample
                ),

            "Start_Date":
                eth_common_sample[
                    "Date"
                ].min(),

            "End_Date":
                eth_common_sample[
                    "Date"
                ].max(),

            "Weekend_Observations":
                int(
                    eth_common_sample[
                        "Is_Weekend"
                    ].sum()
                ),

            "Common_Sample":
                True

        }

    )


# =====================================================================
# 28. CREATE RESULT DATAFRAMES
# =====================================================================

model_results_df = pd.DataFrame(
    model_results
)


coefficient_results_df = pd.DataFrame(
    coefficient_results
)


cross_crypto_results_df = pd.DataFrame(
    cross_crypto_results
)


own_lag_results_df = pd.DataFrame(
    own_lag_results
)


sample_results_df = pd.DataFrame(
    sample_results
)


# =====================================================================
# 29. CHECK THAT N IS IDENTICAL ACROSS LAGS
# =====================================================================

section(
    "COMMON-SAMPLE VALIDATION ACROSS LAGS"
)


for asset in [
    "BTC",
    "ETH"
]:

    asset_results = (
        model_results_df[
            model_results_df[
                "Asset"
            ]
            ==
            asset
        ]
    )


    unique_n = (
        asset_results[
            "N"
        ]
        .unique()
    )


    print(
        f"\n{asset} unique regression sample sizes:"
    )

    print(
        unique_n
    )


    if len(unique_n) != 1:

        raise ValueError(
            f"{asset} models are not using "
            f"one common sample."
        )


print(
    "\nPASS: Every lag specification uses the same "
    "asset-specific regression sample."
)


# =====================================================================
# 30. DISPLAY CROSS-CRYPTO RESULTS
# =====================================================================

section(
    "COMMON-SAMPLE CROSS-CRYPTO RESULTS"
)


cross_display_columns = [

    "Asset",

    "Lag_Days",

    "Specification",

    "Cross_Crypto_Variable",

    "N",

    "Coefficient",

    "HAC_Std_Error",

    "p_value",

    "Change_in_R2",

    "Change_in_Adjusted_R2"

]


print(
    "\n",
    cross_crypto_results_df[
        cross_display_columns
    ]
    .to_string(
        index=False
    )
)


# =====================================================================
# 31. DISPLAY OWN-RETURN-LAG RESULTS
# =====================================================================

section(
    "COMMON-SAMPLE OWN-RETURN-LAG RESULTS"
)


own_display_columns = [

    "Asset",

    "Lag_Days",

    "Specification",

    "N",

    "Coefficient",

    "HAC_Std_Error",

    "p_value"

]


print(
    "\n",
    own_lag_results_df[
        own_display_columns
    ]
    .to_string(
        index=False
    )
)


# =====================================================================
# 32. SIGNIFICANCE LABEL
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
# 33. CREATE INTERPRETATION TABLE
# =====================================================================

cross_crypto_results_df[
    "Significance"
] = (

    cross_crypto_results_df[
        "p_value"
    ]

    .apply(
        significance_label
    )

)


own_lag_results_df[
    "Significance"
] = (

    own_lag_results_df[
        "p_value"
    ]

    .apply(
        significance_label
    )

)


# =====================================================================
# 34. SAVE CONSTRUCTED LAG DATASET
# =====================================================================

section(
    "SAVING CONSTRUCTED ALTERNATIVE-LAG DATASET"
)


LAG_DATA_FILE = (
    DATA_PROCESSED
    / "alternative_lag_dataset.csv"
)


df.to_csv(
    LAG_DATA_FILE,
    index=False
)


print("\nSaved:")
print(LAG_DATA_FILE)


# =====================================================================
# 35. MODEL-SPECIFICATION TABLE
# =====================================================================

specification_rows = []


for lag in LAG_LENGTHS:

    for asset in [
        "BTC",
        "ETH"
    ]:

        for cross_crypto in [
            False,
            True
        ]:

            predictors = get_predictors(
                asset,
                lag,
                cross_crypto
            )


            specification_rows.append(

                {

                    "Asset":
                        asset,

                    "Lag_Days":
                        lag,

                    "Information_Horizon":
                        f"t-{lag}",

                    "Specification":
                        (
                            "Primary"
                            if lag == PRIMARY_LAG
                            else "Robustness"
                        ),

                    "Model":
                        (
                            "Cross_Crypto"
                            if cross_crypto
                            else "Baseline"
                        ),

                    "Dependent_Variable":
                        (
                            "BTC_Return"
                            if asset == "BTC"
                            else "ETH_Return"
                        ),

                    "Sample_Design":
                        "Common sample across t-1, t-2, t-3, t-7",

                    "Predictors":
                        " | ".join(
                            predictors
                        )

                }

            )


specification_df = pd.DataFrame(
    specification_rows
)


# =====================================================================
# 36. COMMON SAMPLE SUMMARY
# =====================================================================

common_sample_summary = pd.DataFrame(

    {

        "Asset": [
            "BTC",
            "ETH"
        ],

        "N": [
            len(
                btc_common_sample
            ),
            len(
                eth_common_sample
            )
        ],

        "Start_Date": [
            btc_common_sample[
                "Date"
            ].min(),
            eth_common_sample[
                "Date"
            ].min()
        ],

        "End_Date": [
            btc_common_sample[
                "Date"
            ].max(),
            eth_common_sample[
                "Date"
            ].max()
        ],

        "Weekend_Observations": [
            int(
                btc_common_sample[
                    "Is_Weekend"
                ].sum()
            ),
            int(
                eth_common_sample[
                    "Is_Weekend"
                ].sum()
            )
        ],

        "Lag_Horizons": [
            "1, 2, 3, 7 calendar days",
            "1, 2, 3, 7 calendar days"
        ],

        "HAC_Maxlags": [
            HAC_MAXLAGS,
            HAC_MAXLAGS
        ]

    }

)


# =====================================================================
# 37. SAVE RESULTS
# =====================================================================

section(
    "SAVING FINAL RESULTS"
)


files_to_save = {

    "alternative_lag_common_sample_model_summary.csv":
        model_results_df,

    "alternative_lag_common_sample_all_coefficients.csv":
        coefficient_results_df,

    "alternative_lag_common_sample_cross_crypto_results.csv":
        cross_crypto_results_df,

    "alternative_lag_common_sample_own_return_results.csv":
        own_lag_results_df,

    "alternative_lag_common_sample_diagnostics.csv":
        sample_results_df,

    "alternative_lag_common_sample_summary.csv":
        common_sample_summary,

    "alternative_lag_model_specifications.csv":
        specification_df

}


for filename, dataframe in files_to_save.items():

    output_path = (
        OUTPUT_DIR
        / filename
    )


    dataframe.to_csv(
        output_path,
        index=False
    )


    print("\nSaved:")
    print(output_path)


# =====================================================================
# 38. FINAL SUMMARY
# =====================================================================

section(
    "FINAL ROBUSTNESS SUMMARY"
)


print(
    "\nPRIMARY SPECIFICATION:"
)

print(
    "t-1 calendar-day information horizon"
)


print(
    "\nROBUSTNESS SPECIFICATIONS:"
)

print(
    "t-2, t-3 and t-7 calendar-day information horizons"
)


print(
    "\nLag specifications estimated separately:"
)

print(
    True
)


print(
    "\nCommon sample used across all lag horizons:"
)

print(
    True
)


print(
    "\nBTC common-sample N:"
)

print(
    len(
        btc_common_sample
    )
)


print(
    "\nETH common-sample N:"
)

print(
    len(
        eth_common_sample
    )
)


print(
    "\nBTC weekend observations:"
)

print(
    int(
        btc_common_sample[
            "Is_Weekend"
        ].sum()
    )
)


print(
    "\nETH weekend observations:"
)

print(
    int(
        eth_common_sample[
            "Is_Weekend"
        ].sum()
    )
)


print(
    "\nHAC / Newey-West maximum lags:"
)

print(
    HAC_MAXLAGS
)


print(
    "\nTRADITIONAL-MARKET TIMING:"
)

print(
    "Traditional-market predictors are constructed from "
    "the information-aligned series."
)


print(
    "\nINTERPRETATION:"
)

print(
    "The alternative specifications represent "
    "calendar-day information horizons."
)

print(
    "They should not be described as the second, third, "
    "or seventh previous traditional-market trading day."
)


print(
    "\nIMPORTANT:"
)

print(
    "The t-1 specification remains the PRIMARY specification."
)

print(
    "t-2, t-3 and t-7 are robustness checks."
)

print(
    "Results should not be selected according to which lag "
    "produces the smallest p-value."
)

print(
    "These are IN-SAMPLE explanatory robustness regressions."
)

print(
    "They do NOT establish out-of-sample forecasting performance."
)


section(
    "FINAL ALTERNATIVE LAG-LENGTH ROBUSTNESS ANALYSIS COMPLETE"
)