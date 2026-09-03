import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# BTC PRICE DATA VALIDATION
# ============================================================
#
# Input:
#     data_clean/btc_price_clean.csv
#
# Purpose:
#     Validate the cleaned BTC Mid Price series BEFORE
#     calculating BTC daily log returns.
#
# IMPORTANT:
#     This script DOES NOT change, delete, replace, interpolate,
#     winsorise, or otherwise modify the BTC data.
#
#     It only identifies observations that should be inspected.
#
# ============================================================


# ============================================================
# 1. FILE PATHS
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

input_file = (
    project_folder
    / "data_clean"
    / "btc_price_clean.csv"
)

output_file = (
    project_folder
    / "data_clean"
    / "btc_validation_flags.csv"
)


print("=" * 70)
print("BTC PRICE DATA VALIDATION")
print("=" * 70)

print("\nInput file:")
print(input_file)

print("\nDoes file exist?")
print(input_file.exists())


if not input_file.exists():
    raise FileNotFoundError(
        f"\nClean BTC file not found:\n{input_file}"
    )


# ============================================================
# 2. LOAD CLEAN BTC DATA
# ============================================================

btc = pd.read_csv(input_file)

print("\nBTC data loaded successfully.")

print("\nShape:")
print(btc.shape)

print("\nColumns:")
print(btc.columns.tolist())


# ============================================================
# 3. CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Date",
    "BTC_Price"
]

missing_columns = [
    column
    for column in required_columns
    if column not in btc.columns
]

if missing_columns:

    raise ValueError(
        "\nRequired columns are missing:\n"
        + str(missing_columns)
    )


# ============================================================
# 4. CONVERT DATA TYPES
# ============================================================

btc["Date"] = pd.to_datetime(
    btc["Date"],
    errors="coerce"
)

btc["BTC_Price"] = pd.to_numeric(
    btc["BTC_Price"],
    errors="coerce"
)


# ============================================================
# 5. SORT BY DATE
# ============================================================

btc = btc.sort_values(
    "Date"
).reset_index(drop=True)


# ============================================================
# 6. BASIC DATA QUALITY CHECKS
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC DATA QUALITY")
print("=" * 70)


# Missing dates

missing_dates = btc["Date"].isna().sum()

print(
    f"\nMissing dates: "
    f"{missing_dates}"
)


# Missing prices

missing_prices = btc["BTC_Price"].isna().sum()

print(
    f"Missing BTC prices: "
    f"{missing_prices}"
)


# Duplicate dates

duplicate_dates = btc["Date"].duplicated().sum()

print(
    f"Duplicate dates: "
    f"{duplicate_dates}"
)


# Invalid prices

invalid_prices = (
    btc["BTC_Price"] <= 0
).sum()

print(
    f"Zero/negative BTC prices: "
    f"{invalid_prices}"
)


# ============================================================
# 7. CHECK DATE RANGE
# ============================================================

print("\n" + "=" * 70)
print("2. DATE RANGE")
print("=" * 70)

print("\nFirst date:")
print(btc["Date"].min())

print("\nLast date:")
print(btc["Date"].max())

print("\nNumber of observations:")
print(len(btc))


# ============================================================
# 8. CHECK WHETHER EVERY CALENDAR DATE EXISTS
# ============================================================

expected_dates = pd.date_range(
    start=btc["Date"].min(),
    end=btc["Date"].max(),
    freq="D"
)

missing_calendar_dates = expected_dates.difference(
    btc["Date"]
)

print(
    "\nMissing calendar dates:"
)

print(
    len(missing_calendar_dates)
)

if len(missing_calendar_dates) > 0:

    print(
        "\nDates missing from BTC series:"
    )

    for date in missing_calendar_dates:

        print(
            date.strftime("%Y-%m-%d")
        )


# ============================================================
# 9. CALCULATE DIAGNOSTIC PRICE CHANGES
# ============================================================

# These variables are ONLY for validation.
#
# They are not yet the final dissertation BTC_Return variable.

btc["Previous_Price"] = (
    btc["BTC_Price"].shift(1)
)

btc["Price_Change"] = (
    btc["BTC_Price"]
    -
    btc["Previous_Price"]
)

btc["Pct_Change"] = (
    btc["BTC_Price"].pct_change(
        fill_method=None
    )
)

btc["Pct_Change_Percent"] = (
    btc["Pct_Change"] * 100
)

btc["Abs_Pct_Change"] = (
    btc["Pct_Change"].abs()
)


# ============================================================
# 10. IDENTIFY REPEATED PRICES
# ============================================================

print("\n" + "=" * 70)
print("3. CONSECUTIVE REPEATED PRICES")
print("=" * 70)

btc["Repeated_Previous_Price"] = (
    btc["BTC_Price"]
    ==
    btc["Previous_Price"]
)

repeated_prices = btc[
    btc["Repeated_Previous_Price"]
].copy()

print(
    "\nNumber of observations identical "
    "to the previous day's price:"
)

print(
    len(repeated_prices)
)


if len(repeated_prices) > 0:

    print(
        "\nRepeated consecutive prices:"
    )

    print(
        repeated_prices[
            [
                "Date",
                "Previous_Price",
                "BTC_Price"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 11. IDENTIFY RUNS OF REPEATED/STale PRICES
# ============================================================

# A single identical price is not necessarily an error.
# However, several consecutive identical daily prices deserve
# inspection.

btc["Same_As_Previous"] = (
    btc["BTC_Price"]
    ==
    btc["BTC_Price"].shift(1)
)

btc["Same_As_Next"] = (
    btc["BTC_Price"]
    ==
    btc["BTC_Price"].shift(-1)
)

btc["Potential_Stale_Price"] = (
    btc["Same_As_Previous"]
    |
    btc["Same_As_Next"]
)


stale_prices = btc[
    btc["Potential_Stale_Price"]
].copy()


print("\n" + "=" * 70)
print("4. POTENTIAL STALE PRICE OBSERVATIONS")
print("=" * 70)

print(
    "\nNumber of observations belonging "
    "to repeated-price sequences:"
)

print(
    len(stale_prices)
)


if len(stale_prices) > 0:

    print(
        "\nPotential stale-price observations:"
    )

    print(
        stale_prices[
            [
                "Date",
                "BTC_Price"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 12. IDENTIFY LARGE DAILY PRICE MOVEMENTS
# ============================================================

# We use several thresholds.
#
# These are NOT automatically classified as errors.
# They are observations requiring inspection.
#
# Crypto can genuinely move substantially in one day.

btc["Move_10pct"] = (
    btc["Abs_Pct_Change"] >= 0.10
)

btc["Move_15pct"] = (
    btc["Abs_Pct_Change"] >= 0.15
)

btc["Move_20pct"] = (
    btc["Abs_Pct_Change"] >= 0.20
)


print("\n" + "=" * 70)
print("5. LARGE DAILY PRICE MOVEMENTS")
print("=" * 70)


print(
    "\nDays with absolute price movement >= 10%:"
)

print(
    btc["Move_10pct"].sum()
)


print(
    "\nDays with absolute price movement >= 15%:"
)

print(
    btc["Move_15pct"].sum()
)


print(
    "\nDays with absolute price movement >= 20%:"
)

print(
    btc["Move_20pct"].sum()
)


# ============================================================
# 13. DISPLAY ALL >=10% MOVEMENTS
# ============================================================

large_moves = btc[
    btc["Move_10pct"]
].copy()


if len(large_moves) > 0:

    print("\n" + "=" * 70)
    print("OBSERVATIONS WITH >= 10% DAILY MOVEMENT")
    print("=" * 70)

    print(
        large_moves[
            [
                "Date",
                "Previous_Price",
                "BTC_Price",
                "Pct_Change_Percent"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 14. CHECK FOR REVERSAL PATTERNS
# ============================================================

# One useful way to detect a possible bad observation:
#
# Day t has an unusually large movement
# AND
# Day t+1 has an unusually large movement in the opposite
# direction.
#
# Example:
#
# 29,000
# 20,000   <- possible erroneous observation
# 29,000
#
# This creates a huge negative move followed immediately
# by a huge positive move.

btc["Next_Pct_Change"] = (
    btc["Pct_Change"].shift(-1)
)

btc["Large_Current_Move"] = (
    btc["Abs_Pct_Change"] >= 0.15
)

btc["Large_Next_Move"] = (
    btc["Next_Pct_Change"].abs() >= 0.15
)

btc["Opposite_Direction"] = (
    np.sign(btc["Pct_Change"])
    !=
    np.sign(btc["Next_Pct_Change"])
)

btc["Suspicious_Reversal"] = (
    btc["Large_Current_Move"]
    &
    btc["Large_Next_Move"]
    &
    btc["Opposite_Direction"]
)


reversals = btc[
    btc["Suspicious_Reversal"]
].copy()


print("\n" + "=" * 70)
print("6. POSSIBLE DATA-ERROR REVERSALS")
print("=" * 70)

print(
    "\nNumber of suspicious large "
    "one-day reversal patterns:"
)

print(
    len(reversals)
)


if len(reversals) > 0:

    print(
        "\nDates requiring manual inspection:"
    )

    print(
        reversals[
            [
                "Date",
                "Previous_Price",
                "BTC_Price",
                "Pct_Change_Percent",
                "Next_Pct_Change"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 15. ROLLING OUTLIER CHECK
# ============================================================

# Compare each BTC price with the local 7-day median.
#
# This can help detect a single observation that is dramatically
# different from surrounding days.

btc["Rolling_7D_Median"] = (
    btc["BTC_Price"]
    .rolling(
        window=7,
        center=True,
        min_periods=3
    )
    .median()
)


btc["Deviation_From_7D_Median"] = (
    (
        btc["BTC_Price"]
        -
        btc["Rolling_7D_Median"]
    )
    /
    btc["Rolling_7D_Median"]
)


btc["Abs_Deviation_From_7D_Median"] = (
    btc["Deviation_From_7D_Median"].abs()
)


# Flag observations more than 20% away from local median.

btc["Local_Price_Outlier"] = (
    btc["Abs_Deviation_From_7D_Median"]
    >= 0.20
)


local_outliers = btc[
    btc["Local_Price_Outlier"]
].copy()


print("\n" + "=" * 70)
print("7. LOCAL PRICE OUTLIER CHECK")
print("=" * 70)

print(
    "\nObservations >=20% away from "
    "their centred 7-day median:"
)

print(
    len(local_outliers)
)


if len(local_outliers) > 0:

    print(
        "\nLocal price outliers:"
    )

    print(
        local_outliers[
            [
                "Date",
                "BTC_Price",
                "Rolling_7D_Median",
                "Deviation_From_7D_Median"
            ]
        ].to_string(
            index=False
        )
    )


# ============================================================
# 16. SPECIFIC CHECK AROUND 30 JULY 2023
# ============================================================

# We noticed this date during preliminary inspection.
# Display surrounding observations so we can inspect it.

inspection_start = pd.Timestamp(
    "2023-07-26"
)

inspection_end = pd.Timestamp(
    "2023-08-03"
)

inspection_window = btc[
    (btc["Date"] >= inspection_start)
    &
    (btc["Date"] <= inspection_end)
].copy()


print("\n" + "=" * 70)
print("8. MANUAL INSPECTION: JULY/AUGUST 2023")
print("=" * 70)

print(
    inspection_window[
        [
            "Date",
            "BTC_Price",
            "Pct_Change_Percent"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# 17. SUMMARY STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("9. BTC PRICE SUMMARY STATISTICS")
print("=" * 70)

print(
    btc["BTC_Price"].describe()
)


# ============================================================
# 18. DAILY CHANGE SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("10. DAILY PERCENTAGE CHANGE SUMMARY")
print("=" * 70)

print(
    btc["Pct_Change_Percent"].describe()
)


# ============================================================
# 19. CREATE MASTER FLAG
# ============================================================

btc["Validation_Flag"] = (
    btc["Repeated_Previous_Price"]
    |
    btc["Move_15pct"]
    |
    btc["Suspicious_Reversal"]
    |
    btc["Local_Price_Outlier"]
)


flagged = btc[
    btc["Validation_Flag"]
].copy()


# ============================================================
# 20. SAVE FLAGGED OBSERVATIONS
# ============================================================

columns_to_save = [
    "Date",
    "Previous_Price",
    "BTC_Price",
    "Pct_Change_Percent",
    "Repeated_Previous_Price",
    "Move_10pct",
    "Move_15pct",
    "Move_20pct",
    "Suspicious_Reversal",
    "Rolling_7D_Median",
    "Deviation_From_7D_Median",
    "Local_Price_Outlier",
    "Validation_Flag"
]


flagged[
    columns_to_save
].to_csv(
    output_file,
    index=False,
    date_format="%Y-%m-%d"
)


# ============================================================
# 21. FINAL VALIDATION SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION SUMMARY")
print("=" * 70)

print(
    f"\nTotal BTC observations: "
    f"{len(btc):,}"
)

print(
    f"Missing calendar dates: "
    f"{len(missing_calendar_dates)}"
)

print(
    f"Duplicate dates: "
    f"{duplicate_dates}"
)

print(
    f"Missing prices: "
    f"{missing_prices}"
)

print(
    f"Zero/negative prices: "
    f"{invalid_prices}"
)

print(
    f"Consecutive repeated prices: "
    f"{len(repeated_prices)}"
)

print(
    f"Daily moves >=10%: "
    f"{btc['Move_10pct'].sum()}"
)

print(
    f"Daily moves >=15%: "
    f"{btc['Move_15pct'].sum()}"
)

print(
    f"Daily moves >=20%: "
    f"{btc['Move_20pct'].sum()}"
)

print(
    f"Suspicious reversal patterns: "
    f"{len(reversals)}"
)

print(
    f"Local 7-day price outliers: "
    f"{len(local_outliers)}"
)

print(
    f"\nTotal observations flagged "
    f"for inspection: {len(flagged)}"
)


print("\nFlagged observations saved to:")

print(output_file)


# ============================================================
# 22. DISPLAY FINAL FLAGGED OBSERVATIONS
# ============================================================

if len(flagged) > 0:

    print("\n" + "=" * 70)
    print("ALL OBSERVATIONS REQUIRING INSPECTION")
    print("=" * 70)

    print(
        flagged[
            columns_to_save
        ].to_string(
            index=False
        )
    )

else:

    print(
        "\nNo observations were flagged."
    )


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nIMPORTANT:"
)

print(
    "No BTC observations have been changed or deleted."
)

print(
    "Review the flagged observations before calculating "
    "the final BTC daily log-return variable."
)