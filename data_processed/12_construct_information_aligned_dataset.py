from pathlib import Path
import pandas as pd
import numpy as np

# ======================================================================
# INFORMATION-ALIGNED FORECASTING DATASET
# ======================================================================

print("=" * 70)
print("INFORMATION-ALIGNED FORECASTING DATASET CONSTRUCTION")
print("=" * 70)

# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MASTER_FILE = BASE_DIR / "master_aligned_dataset.csv"

SP500_FILE = BASE_DIR / "sp500_returns.csv"
VIX_FILE = BASE_DIR / "vix_change.csv"
GOLD_FILE = BASE_DIR / "gold_returns.csv"
DXY_FILE = BASE_DIR / "dxy_returns.csv"
US10Y_FILE = BASE_DIR / "us10y_change_processed.csv"

OUTPUT_FILE = BASE_DIR / "information_aligned_dataset.csv"


# ----------------------------------------------------------------------
# HELPER FUNCTION
# ----------------------------------------------------------------------

def print_section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ----------------------------------------------------------------------
# CHECK FILES
# ----------------------------------------------------------------------

print_section("CHECKING INPUT FILES")

files = {
    "Master dataset": MASTER_FILE,
    "S&P 500": SP500_FILE,
    "VIX": VIX_FILE,
    "Gold": GOLD_FILE,
    "DXY": DXY_FILE,
    "US10Y": US10Y_FILE,
}

for name, path in files.items():
    print(f"\n{name}:")
    print(path)
    print("Exists:", path.exists())

    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# ----------------------------------------------------------------------
# LOAD MASTER DATASET
# ----------------------------------------------------------------------

print_section("IMPORTING MASTER DATASET")

master = pd.read_csv(MASTER_FILE)

print("Shape:")
print(master.shape)

print("\nColumns:")
print(master.columns.tolist())

if "Date" not in master.columns:
    raise KeyError("Master dataset does not contain a Date column.")

master["Date"] = pd.to_datetime(master["Date"], errors="coerce")

print("\nInvalid master dates:")
print(master["Date"].isna().sum())

if master["Date"].isna().any():
    raise ValueError("Invalid dates found in master dataset.")

master = master.sort_values("Date").reset_index(drop=True)

print("\nDate range:")
print(master["Date"].min(), "to", master["Date"].max())

print("\nNumber of observations:")
print(len(master))


# ----------------------------------------------------------------------
# LOAD TRADITIONAL-MARKET DATA
# ----------------------------------------------------------------------

print_section("IMPORTING TRADITIONAL-MARKET DATA")

sp500 = pd.read_csv(SP500_FILE)
vix = pd.read_csv(VIX_FILE)
gold = pd.read_csv(GOLD_FILE)
dxy = pd.read_csv(DXY_FILE)
us10y = pd.read_csv(US10Y_FILE)

datasets = {
    "SP500": sp500,
    "VIX": vix,
    "Gold": gold,
    "DXY": dxy,
    "US10Y": us10y,
}

for name, df in datasets.items():

    print(f"\n{name} shape:")
    print(df.shape)

    print(f"{name} columns:")
    print(df.columns.tolist())

    if "Date" not in df.columns:
        raise KeyError(f"{name} dataset does not contain Date.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    if df["Date"].isna().any():
        raise ValueError(f"Invalid dates found in {name} dataset.")

    if df["Date"].duplicated().any():
        raise ValueError(f"Duplicate dates found in {name} dataset.")

    df.sort_values("Date", inplace=True)
    df.reset_index(drop=True, inplace=True)


# ----------------------------------------------------------------------
# AUTOMATICALLY IDENTIFY REQUIRED TRANSFORMED VARIABLES
# ----------------------------------------------------------------------

print_section("IDENTIFYING TRANSFORMED MARKET VARIABLES")

required_variables = {
    "SP500": "SP500_Return",
    "VIX": "VIX_Change",
    "Gold": "Gold_Return",
    "DXY": "DXY_Return",
    "US10Y": "US10Y_Change",
}

for name, variable in required_variables.items():

    df = datasets[name]

    print(f"\n{name}: looking for {variable}")

    if variable not in df.columns:
        print(f"ERROR: {variable} not found.")
        print("Available columns:")
        print(df.columns.tolist())

        raise KeyError(
            f"Required variable '{variable}' not found in {name} dataset."
        )

    print("Found successfully.")


# ----------------------------------------------------------------------
# CREATE CRYPTO CALENDAR
# ----------------------------------------------------------------------

print_section("CREATING CRYPTO FORECAST CALENDAR")

calendar = master[["Date"]].copy()

print("Crypto calendar observations:")
print(len(calendar))

print("\nFirst date:")
print(calendar["Date"].min())

print("\nLast date:")
print(calendar["Date"].max())

calendar_diffs = calendar["Date"].diff().dt.days

print("\nNon-consecutive crypto calendar observations:")
print((calendar_diffs.dropna() != 1).sum())

if (calendar_diffs.dropna() != 1).sum() == 0:
    print("Crypto calendar contains consecutive calendar days.")


# ----------------------------------------------------------------------
# STRICT PREVIOUS-OBSERVATION ALIGNMENT FUNCTION
# ----------------------------------------------------------------------

def align_previous_market_information(
        crypto_calendar,
        market_df,
        variable,
        output_variable,
        source_date_variable
):

    """
    For every crypto date t, obtain the most recent traditional-market
    observation with market date STRICTLY EARLIER than t.

    This prevents contemporaneous traditional-market information from
    entering the forecasting model.

    Example:

    Saturday BTC forecast -> Friday market information
    Sunday BTC forecast   -> Friday market information
    Monday BTC forecast   -> Friday market information
    Tuesday BTC forecast  -> Monday market information
    """

    market = market_df[["Date", variable]].copy()

    # Keep source date so we can audit exactly which traditional-market
    # observation was used.
    market[source_date_variable] = market["Date"]

    market = market.sort_values("Date").reset_index(drop=True)

    result = pd.merge_asof(
        crypto_calendar.sort_values("Date"),
        market,
        on="Date",
        direction="backward",
        allow_exact_matches=False
    )

    result = result.rename(columns={variable: output_variable})

    return result[
        ["Date", output_variable, source_date_variable]
    ]


# ----------------------------------------------------------------------
# ALIGN S&P 500
# ----------------------------------------------------------------------

print_section("ALIGNING S&P 500 INFORMATION")

sp500_aligned = align_previous_market_information(
    calendar,
    sp500,
    "SP500_Return",
    "Lagged_SP500_Return_Aligned",
    "SP500_Source_Date"
)

print(sp500_aligned.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# ALIGN VIX
# ----------------------------------------------------------------------

print_section("ALIGNING VIX INFORMATION")

vix_aligned = align_previous_market_information(
    calendar,
    vix,
    "VIX_Change",
    "Lagged_VIX_Change_Aligned",
    "VIX_Source_Date"
)

print(vix_aligned.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# ALIGN GOLD
# ----------------------------------------------------------------------

print_section("ALIGNING GOLD INFORMATION")

gold_aligned = align_previous_market_information(
    calendar,
    gold,
    "Gold_Return",
    "Lagged_Gold_Return_Aligned",
    "Gold_Source_Date"
)

print(gold_aligned.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# ALIGN DXY
# ----------------------------------------------------------------------

print_section("ALIGNING DXY INFORMATION")

dxy_aligned = align_previous_market_information(
    calendar,
    dxy,
    "DXY_Return",
    "Lagged_DXY_Return_Aligned",
    "DXY_Source_Date"
)

print(dxy_aligned.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# ALIGN US10Y
# ----------------------------------------------------------------------

print_section("ALIGNING US10Y INFORMATION")

us10y_aligned = align_previous_market_information(
    calendar,
    us10y,
    "US10Y_Change",
    "Lagged_US10Y_Change_Aligned",
    "US10Y_Source_Date"
)

print(us10y_aligned.head(10).to_string(index=False))


# ----------------------------------------------------------------------
# MERGE ALIGNED VARIABLES
# ----------------------------------------------------------------------

print_section("MERGING INFORMATION-ALIGNED VARIABLES")

aligned = calendar.copy()

for aligned_df in [
    sp500_aligned,
    vix_aligned,
    gold_aligned,
    dxy_aligned,
    us10y_aligned
]:

    aligned = aligned.merge(
        aligned_df,
        on="Date",
        how="left",
        validate="one_to_one"
    )

print("Aligned dataset shape:")
print(aligned.shape)


# ----------------------------------------------------------------------
# VERIFY NO FUTURE OR SAME-DAY INFORMATION
# ----------------------------------------------------------------------

print_section("CHECKING FOR LOOK-AHEAD BIAS")

source_columns = [
    "SP500_Source_Date",
    "VIX_Source_Date",
    "Gold_Source_Date",
    "DXY_Source_Date",
    "US10Y_Source_Date",
]

total_violations = 0

for col in source_columns:

    valid = aligned[col].notna()

    violations = (
        aligned.loc[valid, col] >= aligned.loc[valid, "Date"]
    ).sum()

    total_violations += violations

    print(f"{col}: {violations} timing violations")

print("\nTotal timing violations:")
print(total_violations)

if total_violations != 0:
    raise ValueError(
        "LOOK-AHEAD BIAS DETECTED: source date is not strictly before "
        "crypto forecast date."
    )

print("\nPASS: All traditional-market information comes from dates")
print("strictly before the corresponding crypto date.")


# ----------------------------------------------------------------------
# CALCULATE INFORMATION AGE
# ----------------------------------------------------------------------

print_section("CALCULATING INFORMATION AGE")

markets = {
    "SP500": "SP500_Source_Date",
    "VIX": "VIX_Source_Date",
    "Gold": "Gold_Source_Date",
    "DXY": "DXY_Source_Date",
    "US10Y": "US10Y_Source_Date",
}

for name, source_col in markets.items():

    age_col = f"{name}_Information_Age_Days"

    aligned[age_col] = (
        aligned["Date"] - aligned[source_col]
    ).dt.days

    print(f"\n{name} information age:")
    print(aligned[age_col].value_counts().sort_index().head(10))


# ----------------------------------------------------------------------
# WEEKEND EXAMPLE
# ----------------------------------------------------------------------

print_section("CHECKING WEEKEND ALIGNMENT EXAMPLE")

weekend_sample = aligned[
    aligned["Date"].dt.dayofweek.isin([5, 6])
].head(10)

display_columns = [
    "Date",
    "Lagged_SP500_Return_Aligned",
    "SP500_Source_Date",
    "SP500_Information_Age_Days",
    "Lagged_VIX_Change_Aligned",
    "VIX_Source_Date",
]

print(weekend_sample[display_columns].to_string(index=False))


# ----------------------------------------------------------------------
# MERGE BACK INTO MASTER DATASET
# ----------------------------------------------------------------------

print_section("ADDING ALIGNED VARIABLES TO MASTER DATASET")

# Remove Date-only duplication by merging all new columns onto master.
final = master.merge(
    aligned,
    on="Date",
    how="left",
    validate="one_to_one"
)

print("Final shape:")
print(final.shape)


# ----------------------------------------------------------------------
# CHECK MISSING VALUES IN NEW FORECASTING VARIABLES
# ----------------------------------------------------------------------

print_section("CHECKING MISSING VALUES")

forecast_variables = [
    "Lagged_SP500_Return_Aligned",
    "Lagged_VIX_Change_Aligned",
    "Lagged_Gold_Return_Aligned",
    "Lagged_DXY_Return_Aligned",
    "Lagged_US10Y_Change_Aligned",
]

print(final[forecast_variables].isna().sum())


# ----------------------------------------------------------------------
# FINAL VALIDATION
# ----------------------------------------------------------------------

print_section("FINAL VALIDATION")

print("Number of observations:")
print(len(final))

print("\nDate range:")
print(final["Date"].min(), "to", final["Date"].max())

print("\nDuplicate dates:")
print(final["Date"].duplicated().sum())

print("\nAligned forecasting variables:")

for variable in forecast_variables:
    print(variable)


# ----------------------------------------------------------------------
# SAVE
# ----------------------------------------------------------------------

print_section("SAVING INFORMATION-ALIGNED DATASET")

final.to_csv(
    OUTPUT_FILE,
    index=False
)

print("File saved successfully:")
print(OUTPUT_FILE)


# ----------------------------------------------------------------------
# METHODOLOGICAL INTERPRETATION
# ----------------------------------------------------------------------

print_section("REGRESSION / FORECASTING USE")

print("""
For the predictive BTC and ETH models, use:

Lagged_SP500_Return_Aligned
Lagged_VIX_Change_Aligned
Lagged_Gold_Return_Aligned
Lagged_DXY_Return_Aligned
Lagged_US10Y_Change_Aligned

These variables represent the most recent transformed traditional-market
observation available strictly before each crypto forecast date.

This handles the mismatch between the 24/7 cryptocurrency calendar and
traditional financial-market trading calendars while preventing same-day
traditional-market information from entering the predictive specification.

The Source_Date variables are retained for auditability and allow the
information set used for every crypto observation to be verified.
""")

print("=" * 70)
print("INFORMATION ALIGNMENT COMPLETE")
print("=" * 70)