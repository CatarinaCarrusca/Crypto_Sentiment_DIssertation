# =============================================================================
# 09_construct_master_dataset.py
#
# MASTER ALIGNED DATASET CONSTRUCTION
#
# Purpose:
#   Combine all processed non-Reddit variables into one master dataset
#   using the full BTC/ETH calendar as the base.
#
# IMPORTANT:
#   - Crypto trades 7 days per week.
#   - Traditional financial markets do not.
#   - Therefore this script uses LEFT MERGES onto the crypto calendar.
#   - Missing traditional-market observations are PRESERVED.
#   - No forward filling is performed.
#   - No missing values are replaced with zero.
#   - No observations are dropped.
#
# The forecasting/regression sample will be constructed separately.
# =============================================================================


from pathlib import Path
import pandas as pd
import numpy as np


# =============================================================================
# SETTINGS
# =============================================================================

PROJECT_ROOT = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

PROCESSED_DIR = PROJECT_ROOT / "data_processed"

OUTPUT_FILE = PROCESSED_DIR / "master_aligned_dataset.csv"


# =============================================================================
# INPUT FILES
# =============================================================================

CROSS_CRYPTO_FILE = PROCESSED_DIR / "cross_crypto_returns.csv"

BTC_VOLUME_FILE = PROCESSED_DIR / "btc_volume_processed.csv"

ETH_VOLUME_FILE = PROCESSED_DIR / "eth_volume_processed.csv"

SP500_FILE = PROCESSED_DIR / "sp500_returns.csv"

VIX_FILE = PROCESSED_DIR / "vix_change.csv"

GOLD_FILE = PROCESSED_DIR / "gold_returns.csv"

DXY_FILE = PROCESSED_DIR / "dxy_returns.csv"

US10Y_FILE = PROCESSED_DIR / "us10y_change_processed.csv"


# =============================================================================
# HELPER FUNCTION
# =============================================================================

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# =============================================================================
# START
# =============================================================================

print_section("MASTER ALIGNED DATASET CONSTRUCTION")


# =============================================================================
# CHECK INPUT FILES
# =============================================================================

print_section("CHECKING INPUT FILES")

files_to_check = {
    "Cross Crypto Returns": CROSS_CRYPTO_FILE,
    "BTC Volume": BTC_VOLUME_FILE,
    "ETH Volume": ETH_VOLUME_FILE,
    "S&P 500": SP500_FILE,
    "VIX": VIX_FILE,
    "Gold": GOLD_FILE,
    "DXY": DXY_FILE,
    "US 10-Year Treasury": US10Y_FILE,
}


for name, path in files_to_check.items():

    print(f"\n{name}")
    print(path)
    print("Exists:", path.exists())

    if not path.exists():

        raise FileNotFoundError(
            f"\n{name} file was not found.\n"
            f"Expected location:\n{path}\n\n"
            f"Check the filename used in this script."
        )


# =============================================================================
# IMPORT DATA
# =============================================================================

print_section("IMPORTING PROCESSED DATASETS")


cross_crypto = pd.read_csv(CROSS_CRYPTO_FILE)

btc_volume = pd.read_csv(BTC_VOLUME_FILE)

eth_volume = pd.read_csv(ETH_VOLUME_FILE)

sp500 = pd.read_csv(SP500_FILE)

vix = pd.read_csv(VIX_FILE)

gold = pd.read_csv(GOLD_FILE)

dxy = pd.read_csv(DXY_FILE)

us10y = pd.read_csv(US10Y_FILE)


datasets = {
    "Cross Crypto Returns": cross_crypto,
    "BTC Volume": btc_volume,
    "ETH Volume": eth_volume,
    "S&P 500": sp500,
    "VIX": vix,
    "Gold": gold,
    "DXY": dxy,
    "US 10-Year Treasury": us10y,
}


# =============================================================================
# DISPLAY IMPORT INFORMATION
# =============================================================================

for name, df in datasets.items():

    print(f"\n{name}")
    print("-" * 50)

    print("Shape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())


# =============================================================================
# VALIDATE DATE COLUMNS
# =============================================================================

print_section("VALIDATING DATE COLUMNS")


for name, df in datasets.items():

    if "Date" not in df.columns:

        raise ValueError(
            f"{name} does not contain a 'Date' column."
        )

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    invalid_dates = df["Date"].isna().sum()

    print(f"\n{name}")
    print("Invalid dates:", invalid_dates)

    if invalid_dates > 0:

        raise ValueError(
            f"{invalid_dates} invalid dates found in {name}."
        )

    # Sort chronologically
    df.sort_values(
        by="Date",
        inplace=True
    )

    df.reset_index(
        drop=True,
        inplace=True
    )


# =============================================================================
# CHECK DUPLICATE DATES
# =============================================================================

print_section("CHECKING DUPLICATE DATES")


for name, df in datasets.items():

    duplicate_dates = df["Date"].duplicated().sum()

    print(f"\n{name}")
    print("Duplicate dates:", duplicate_dates)

    if duplicate_dates > 0:

        print("\nDuplicate observations:")

        print(
            df[
                df["Date"].duplicated(
                    keep=False
                )
            ].sort_values("Date")
        )

        raise ValueError(
            f"Duplicate dates found in {name}."
        )


# =============================================================================
# CHECK DATE RANGES
# =============================================================================

print_section("CHECKING SOURCE DATE RANGES")


for name, df in datasets.items():

    print(f"\n{name}")

    print("Number of observations:")
    print(len(df))

    print("First date:")
    print(df["Date"].min())

    print("Last date:")
    print(df["Date"].max())


# =============================================================================
# REQUIRED VARIABLES
# =============================================================================

print_section("CHECKING REQUIRED VARIABLES")


required_columns = {

    "Cross Crypto Returns": [
        "Date",
        "BTC_Return",
        "BTC_Lagged_Return",
        "ETH_Return",
        "ETH_Lagged_Return",
    ],

    "BTC Volume": [
        "Date",
        "Log_BTC_Volume",
        "Lagged_Log_BTC_Volume",
    ],

    "ETH Volume": [
        "Date",
        "Log_ETH_Volume",
        "Lagged_Log_ETH_Volume",
    ],

    "S&P 500": [
        "Date",
        "SP500_Return",
        "Lagged_SP500_Return",
    ],

    "VIX": [
        "Date",
        "VIX_Change",
        "Lagged_VIX_Change",
    ],

    "Gold": [
        "Date",
        "Gold_Return",
        "Lagged_Gold_Return",
    ],

    "DXY": [
        "Date",
        "DXY_Return",
        "Lagged_DXY_Return",
    ],

    "US 10-Year Treasury": [
        "Date",
        "US10Y_Change",
        "Lagged_US10Y_Change",
    ],
}


for name, columns in required_columns.items():

    df = datasets[name]

    missing_columns = [
        column
        for column in columns
        if column not in df.columns
    ]

    print(f"\n{name}")

    if len(missing_columns) == 0:

        print("All required variables are present.")

    else:

        print("Missing required variables:")
        print(missing_columns)

        print("\nActual columns found:")
        print(df.columns.tolist())

        raise ValueError(
            f"Required columns are missing from {name}."
        )


# =============================================================================
# KEEP ONLY VARIABLES REQUIRED FOR MASTER DATASET
# =============================================================================

print_section("SELECTING VARIABLES FOR MASTER DATASET")


cross_crypto = cross_crypto[
    [
        "Date",
        "BTC_Return",
        "BTC_Lagged_Return",
        "ETH_Return",
        "ETH_Lagged_Return",
    ]
].copy()


btc_volume = btc_volume[
    [
        "Date",
        "Log_BTC_Volume",
        "Lagged_Log_BTC_Volume",
    ]
].copy()


eth_volume = eth_volume[
    [
        "Date",
        "Log_ETH_Volume",
        "Lagged_Log_ETH_Volume",
    ]
].copy()


sp500 = sp500[
    [
        "Date",
        "SP500_Return",
        "Lagged_SP500_Return",
    ]
].copy()


vix = vix[
    [
        "Date",
        "VIX_Change",
        "Lagged_VIX_Change",
    ]
].copy()


gold = gold[
    [
        "Date",
        "Gold_Return",
        "Lagged_Gold_Return",
    ]
].copy()


dxy = dxy[
    [
        "Date",
        "DXY_Return",
        "Lagged_DXY_Return",
    ]
].copy()


us10y = us10y[
    [
        "Date",
        "US10Y_Change",
        "Lagged_US10Y_Change",
    ]
].copy()


# =============================================================================
# CREATE MASTER BASE
# =============================================================================

print_section("CREATING MASTER CRYPTO CALENDAR")


master = cross_crypto.copy()


print("\nStarting master shape:")
print(master.shape)

print("\nStarting date range:")
print(
    master["Date"].min(),
    "to",
    master["Date"].max()
)


# =============================================================================
# MERGE FUNCTION
# =============================================================================

def merge_dataset(master_df, new_df, dataset_name):

    before_rows = len(master_df)

    merged = master_df.merge(
        new_df,
        on="Date",
        how="left",
        validate="one_to_one"
    )

    after_rows = len(merged)

    print(f"\nMerged: {dataset_name}")

    print("Shape after merge:")
    print(merged.shape)

    print("Rows before merge:")
    print(before_rows)

    print("Rows after merge:")
    print(after_rows)

    if before_rows != after_rows:

        raise ValueError(
            f"Row count changed after merging {dataset_name}."
        )

    return merged


# =============================================================================
# MERGE ALL DATASETS
# =============================================================================

print_section("MERGING PROCESSED DATASETS")


master = merge_dataset(
    master,
    btc_volume,
    "BTC Volume"
)


master = merge_dataset(
    master,
    eth_volume,
    "ETH Volume"
)


master = merge_dataset(
    master,
    sp500,
    "S&P 500"
)


master = merge_dataset(
    master,
    vix,
    "VIX"
)


master = merge_dataset(
    master,
    gold,
    "Gold"
)


master = merge_dataset(
    master,
    dxy,
    "DXY"
)


master = merge_dataset(
    master,
    us10y,
    "US 10-Year Treasury"
)


# =============================================================================
# SORT FINAL MASTER DATASET
# =============================================================================

master = (
    master
    .sort_values("Date")
    .reset_index(drop=True)
)


# =============================================================================
# MASTER DATASET CHECK
# =============================================================================

print_section("MASTER DATASET BASIC CHECK")


print("\nMaster shape:")
print(master.shape)


print("\nNumber of observations:")
print(len(master))


print("\nFirst date:")
print(master["Date"].min())


print("\nLast date:")
print(master["Date"].max())


print("\nDuplicate dates:")
print(master["Date"].duplicated().sum())


print("\nFinal variables:")

for column in master.columns:
    print(column)


# =============================================================================
# CHECK CALENDAR CONTINUITY
# =============================================================================

print_section("CHECKING CRYPTO CALENDAR CONTINUITY")


date_difference = master["Date"].diff()

non_consecutive = (
    date_difference.dropna()
    != pd.Timedelta(days=1)
).sum()


print("\nNon-consecutive calendar observations:")
print(non_consecutive)


if non_consecutive == 0:

    print(
        "\nMaster dataset retains the complete "
        "consecutive crypto calendar."
    )

else:

    print(
        "\nWARNING: Gaps exist in the master crypto calendar."
    )


# =============================================================================
# VERIFY BTC RETURN LAG
# =============================================================================

print_section("VERIFYING BTC RETURN LAG")


expected_btc_lag = master["BTC_Return"].shift(1)


btc_lag_mismatch = (

    master["BTC_Lagged_Return"].notna()

    &

    expected_btc_lag.notna()

    &

    (
        np.abs(
            master["BTC_Lagged_Return"]
            - expected_btc_lag
        ) > 1e-12
    )
)


print("\nBTC lag mismatches:")
print(btc_lag_mismatch.sum())


if btc_lag_mismatch.sum() == 0:

    print(
        "BTC_Lagged_Return remains correctly aligned."
    )


# =============================================================================
# VERIFY ETH RETURN LAG
# =============================================================================

print_section("VERIFYING ETH RETURN LAG")


expected_eth_lag = master["ETH_Return"].shift(1)


eth_lag_mismatch = (

    master["ETH_Lagged_Return"].notna()

    &

    expected_eth_lag.notna()

    &

    (
        np.abs(
            master["ETH_Lagged_Return"]
            - expected_eth_lag
        ) > 1e-12
    )
)


print("\nETH lag mismatches:")
print(eth_lag_mismatch.sum())


if eth_lag_mismatch.sum() == 0:

    print(
        "ETH_Lagged_Return remains correctly aligned."
    )


# =============================================================================
# VERIFY BTC VOLUME LAG
# =============================================================================

print_section("VERIFYING BTC VOLUME LAG")


expected_btc_volume_lag = (
    master["Log_BTC_Volume"].shift(1)
)


btc_volume_lag_mismatch = (

    master["Lagged_Log_BTC_Volume"].notna()

    &

    expected_btc_volume_lag.notna()

    &

    (
        np.abs(
            master["Lagged_Log_BTC_Volume"]
            - expected_btc_volume_lag
        ) > 1e-12
    )
)


print("\nBTC volume lag mismatches:")
print(btc_volume_lag_mismatch.sum())


if btc_volume_lag_mismatch.sum() == 0:

    print(
        "Lagged_Log_BTC_Volume remains correctly aligned."
    )


# =============================================================================
# VERIFY ETH VOLUME LAG
# =============================================================================

print_section("VERIFYING ETH VOLUME LAG")


expected_eth_volume_lag = (
    master["Log_ETH_Volume"].shift(1)
)


eth_volume_lag_mismatch = (

    master["Lagged_Log_ETH_Volume"].notna()

    &

    expected_eth_volume_lag.notna()

    &

    (
        np.abs(
            master["Lagged_Log_ETH_Volume"]
            - expected_eth_volume_lag
        ) > 1e-12
    )
)


print("\nETH volume lag mismatches:")
print(eth_volume_lag_mismatch.sum())


if eth_volume_lag_mismatch.sum() == 0:

    print(
        "Lagged_Log_ETH_Volume remains correctly aligned."
    )


# =============================================================================
# MISSING VALUE REPORT
# =============================================================================

print_section("MISSING VALUE REPORT")


missing_report = pd.DataFrame(
    {
        "Missing_N":
            master.isna().sum(),

        "Missing_Percent":
            master.isna().mean() * 100,
    }
)


missing_report["Missing_Percent"] = (
    missing_report["Missing_Percent"]
    .round(2)
)


print(missing_report.to_string())


# =============================================================================
# CHECK FIRST OBSERVATIONS
# =============================================================================

print_section("CHECKING FIRST 10 OBSERVATIONS")


print(
    master.head(10).to_string(
        index=False
    )
)


# =============================================================================
# CHECK LAST OBSERVATIONS
# =============================================================================

print_section("CHECKING LAST 10 OBSERVATIONS")


print(
    master.tail(10).to_string(
        index=False
    )
)


# =============================================================================
# WEEKEND / MARKET CALENDAR CHECK
# =============================================================================

print_section("CHECKING WEEKEND ALIGNMENT")


weekend_check = master[
    (
        master["Date"]
        >= pd.Timestamp("2024-01-01")
    )
    &
    (
        master["Date"]
        <= pd.Timestamp("2024-01-15")
    )
].copy()


weekend_columns = [
    "Date",
    "BTC_Return",
    "ETH_Return",
    "Lagged_Log_BTC_Volume",
    "Lagged_Log_ETH_Volume",
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


print(
    weekend_check[
        weekend_columns
    ].to_string(
        index=False
    )
)


# =============================================================================
# COUNT WEEKEND OBSERVATIONS
# =============================================================================

print_section("WEEKEND OBSERVATION CHECK")


master["Day_of_Week"] = master["Date"].dt.day_name()

master["Is_Weekend"] = (
    master["Date"].dt.dayofweek >= 5
)


print("\nNumber of weekend observations:")
print(master["Is_Weekend"].sum())


print("\nNumber of weekday observations:")
print((~master["Is_Weekend"]).sum())


# =============================================================================
# TRADITIONAL MARKET AVAILABILITY ON WEEKENDS
# =============================================================================

print_section("TRADITIONAL MARKET AVAILABILITY ON WEEKENDS")


traditional_lagged_variables = [
    "Lagged_SP500_Return",
    "Lagged_VIX_Change",
    "Lagged_Gold_Return",
    "Lagged_DXY_Return",
    "Lagged_US10Y_Change",
]


weekend_rows = master[
    master["Is_Weekend"]
]


for variable in traditional_lagged_variables:

    available = weekend_rows[variable].notna().sum()

    missing = weekend_rows[variable].isna().sum()

    print(f"\n{variable}")

    print("Weekend values available:")
    print(available)

    print("Weekend values missing:")
    print(missing)


# =============================================================================
# REMOVE TEMPORARY DIAGNOSTIC VARIABLES
# =============================================================================

master.drop(
    columns=[
        "Day_of_Week",
        "Is_Weekend",
    ],
    inplace=True
)


# =============================================================================
# FINAL COLUMN ORDER
# =============================================================================

final_columns = [

    "Date",

    # ---------------------------------------------------------
    # BTC
    # ---------------------------------------------------------

    "BTC_Return",
    "BTC_Lagged_Return",

    "Log_BTC_Volume",
    "Lagged_Log_BTC_Volume",

    # ---------------------------------------------------------
    # ETH
    # ---------------------------------------------------------

    "ETH_Return",
    "ETH_Lagged_Return",

    "Log_ETH_Volume",
    "Lagged_Log_ETH_Volume",

    # ---------------------------------------------------------
    # S&P 500
    # ---------------------------------------------------------

    "SP500_Return",
    "Lagged_SP500_Return",

    # ---------------------------------------------------------
    # VIX
    # ---------------------------------------------------------

    "VIX_Change",
    "Lagged_VIX_Change",

    # ---------------------------------------------------------
    # GOLD
    # ---------------------------------------------------------

    "Gold_Return",
    "Lagged_Gold_Return",

    # ---------------------------------------------------------
    # DXY
    # ---------------------------------------------------------

    "DXY_Return",
    "Lagged_DXY_Return",

    # ---------------------------------------------------------
    # US 10-YEAR TREASURY
    # ---------------------------------------------------------

    "US10Y_Change",
    "Lagged_US10Y_Change",
]


master = master[
    final_columns
].copy()


# =============================================================================
# SAVE MASTER DATASET
# =============================================================================

print_section("SAVING MASTER ALIGNED DATASET")


master.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\nFile saved successfully:")
print(OUTPUT_FILE)


# =============================================================================
# FINAL CHECK
# =============================================================================

print_section("FINAL CHECK")


print("\nNumber of observations:")
print(len(master))


print("\nDate range:")

print(
    master["Date"].min(),
    "to",
    master["Date"].max()
)


print("\nNumber of variables:")
print(len(master.columns))


print("\nFinal variables:")

for variable in master.columns:
    print(variable)


print("\nMissing values:")

print(
    master.isna().sum()
)


# =============================================================================
# IMPORTANT METHODOLOGICAL MESSAGE
# =============================================================================

print_section("IMPORTANT: NEXT STEP")


print(
    """
The master aligned dataset has been constructed using the complete
cryptocurrency calendar.

No traditional-market missing values have been filled.

No forward filling has been performed.

No missing values have been replaced with zero.

No observations have been dropped.

This is intentional.

BTC and ETH trade seven days per week, whereas the S&P 500, VIX,
Gold, DXY and US 10-year Treasury series follow traditional market
calendars.

The next step is therefore NOT to immediately run the regressions.

The next step is to construct the estimation / forecasting dataset
and explicitly define what traditional-market information was
available before each cryptocurrency return being predicted.

This is particularly important because the dissertation evaluates
predictive performance and must avoid look-ahead bias.
"""
)


print_section(
    "MASTER ALIGNED DATASET CONSTRUCTION COMPLETE"
)