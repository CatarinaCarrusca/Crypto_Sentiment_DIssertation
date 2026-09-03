import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# ETH CLEAN PRICE VALIDATION
# ============================================================

print("=" * 70)
print("ETH CLEAN PRICE VALIDATION")
print("=" * 70)

# ------------------------------------------------------------
# 1. FILE PATH
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

input_file = PROJECT_ROOT / "data_clean" / "eth_price_clean.csv"

print("\nLooking for cleaned ETH file:")
print(input_file)

if not input_file.exists():
    raise FileNotFoundError(
        f"\nClean ETH file not found:\n{input_file}"
    )

print("\nClean ETH file found successfully.")


# ------------------------------------------------------------
# 2. LOAD CLEAN ETH DATA
# ------------------------------------------------------------

eth = pd.read_csv(
    input_file,
    parse_dates=["Date"]
)

print("\nData loaded successfully.")

print("\nShape:")
print(eth.shape)

print("\nColumns:")
print(eth.columns.tolist())


# ------------------------------------------------------------
# 3. BASIC STRUCTURE CHECKS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("1. BASIC STRUCTURE CHECKS")
print("=" * 70)

print("\nFirst date:")
print(eth["Date"].min())

print("\nLast date:")
print(eth["Date"].max())

print("\nNumber of observations:")
print(len(eth))

print("\nMissing values:")
print(eth.isna().sum())

print("\nDuplicate dates:")
print(eth["Date"].duplicated().sum())

print("\nZero or negative prices:")
print((eth["ETH_Price"] <= 0).sum())


# ------------------------------------------------------------
# 4. CHECK FOR MISSING CALENDAR DATES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("2. MISSING CALENDAR DATE CHECK")
print("=" * 70)

expected_dates = pd.date_range(
    start=eth["Date"].min(),
    end=eth["Date"].max(),
    freq="D"
)

missing_dates = expected_dates.difference(eth["Date"])

print(f"\nExpected calendar days: {len(expected_dates)}")
print(f"Actual observations:    {len(eth)}")
print(f"Missing calendar days:  {len(missing_dates)}")

if len(missing_dates) > 0:
    print("\nMissing dates:")
    for date in missing_dates:
        print(date.date())


# ------------------------------------------------------------
# 5. TEMPORARY LOG RETURNS FOR VALIDATION ONLY
# ------------------------------------------------------------

# These returns are NOT being saved as the final dissertation
# return variable yet. They are used only to identify suspicious
# price movements.

eth["Validation_Log_Return"] = np.log(
    eth["ETH_Price"] / eth["ETH_Price"].shift(1)
)

eth["Validation_Return_Pct"] = (
    np.exp(eth["Validation_Log_Return"]) - 1
) * 100


# ------------------------------------------------------------
# 6. LARGEST ABSOLUTE DAILY MOVEMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("3. 20 LARGEST ABSOLUTE DAILY PRICE MOVEMENTS")
print("=" * 70)

largest_moves = (
    eth.dropna(subset=["Validation_Log_Return"])
    .assign(
        Absolute_Log_Return=lambda x:
        x["Validation_Log_Return"].abs()
    )
    .nlargest(20, "Absolute_Log_Return")
)

print(
    largest_moves[
        [
            "Date",
            "ETH_Price",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 7. FLAG VERY LARGE DAILY MOVEMENTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("4. DAILY MOVEMENTS GREATER THAN 20%")
print("=" * 70)

large_moves = eth[
    eth["Validation_Return_Pct"].abs() > 20
].copy()

print(
    f"\nNumber of absolute daily movements > 20%: "
    f"{len(large_moves)}"
)

if len(large_moves) > 0:

    print(
        large_moves[
            [
                "Date",
                "ETH_Price",
                "Validation_Log_Return",
                "Validation_Return_Pct"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 8. CHECK EXACTLY REPEATED CONSECUTIVE PRICES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("5. CONSECUTIVE IDENTICAL PRICE CHECK")
print("=" * 70)

eth["Previous_Price"] = eth["ETH_Price"].shift(1)

repeated_prices = eth[
    eth["ETH_Price"] == eth["Previous_Price"]
].copy()

print(
    f"\nNumber of consecutive identical prices: "
    f"{len(repeated_prices)}"
)

if len(repeated_prices) > 0:

    print(
        repeated_prices[
            [
                "Date",
                "Previous_Price",
                "ETH_Price"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 9. CHECK RUNS OF IDENTICAL PRICES
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("6. REPEATED PRICE RUNS")
print("=" * 70)

eth["Price_Changed"] = (
    eth["ETH_Price"] != eth["ETH_Price"].shift(1)
)

eth["Price_Run"] = eth["Price_Changed"].cumsum()

price_runs = (
    eth.groupby("Price_Run")
    .agg(
        Start_Date=("Date", "min"),
        End_Date=("Date", "max"),
        ETH_Price=("ETH_Price", "first"),
        Run_Length=("Date", "size")
    )
)

long_runs = price_runs[
    price_runs["Run_Length"] >= 2
].copy()

print(
    f"\nNumber of repeated-price runs "
    f"(2+ consecutive days): {len(long_runs)}"
)

if len(long_runs) > 0:

    print(
        long_runs[
            [
                "Start_Date",
                "End_Date",
                "ETH_Price",
                "Run_Length"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 10. IDENTIFY POSSIBLE ONE-DAY REVERSALS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("7. POSSIBLE ONE-DAY REVERSALS")
print("=" * 70)

# A possible reversal occurs when:
# 1. today's absolute return is > 15%, AND
# 2. tomorrow's movement is in the opposite direction, AND
# 3. tomorrow's absolute movement is > 10%.

eth["Next_Log_Return"] = eth[
    "Validation_Log_Return"
].shift(-1)

reversal_flag = (
    (eth["Validation_Log_Return"].abs() > 0.15)
    &
    (
        np.sign(eth["Validation_Log_Return"])
        !=
        np.sign(eth["Next_Log_Return"])
    )
    &
    (eth["Next_Log_Return"].abs() > 0.10)
)

reversals = eth[
    reversal_flag
].copy()

print(
    f"\nNumber of possible large one-day reversals: "
    f"{len(reversals)}"
)

if len(reversals) > 0:

    print(
        reversals[
            [
                "Date",
                "ETH_Price",
                "Validation_Log_Return",
                "Next_Log_Return"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 11. LOCAL MEDIAN OUTLIER CHECK
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("8. LOCAL PRICE OUTLIER CHECK")
print("=" * 70)

# Compare each price with the rolling 7-day median.
# This is only a diagnostic flag; it does NOT mean that
# observations are automatically incorrect.

eth["Rolling_7D_Median"] = (
    eth["ETH_Price"]
    .rolling(
        window=7,
        center=True,
        min_periods=3
    )
    .median()
)

eth["Deviation_From_Local_Median_Pct"] = (
    (
        eth["ETH_Price"]
        /
        eth["Rolling_7D_Median"]
    ) - 1
) * 100

local_outliers = eth[
    eth["Deviation_From_Local_Median_Pct"].abs() > 20
].copy()

print(
    f"\nObservations >20% from 7-day local median: "
    f"{len(local_outliers)}"
)

if len(local_outliers) > 0:

    print(
        local_outliers[
            [
                "Date",
                "ETH_Price",
                "Rolling_7D_Median",
                "Deviation_From_Local_Median_Pct"
            ]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# 12. DISTRIBUTION OF TEMPORARY RETURNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("9. TEMPORARY RETURN DISTRIBUTION")
print("=" * 70)

print(
    eth["Validation_Log_Return"].describe(
        percentiles=[
            0.01,
            0.025,
            0.05,
            0.50,
            0.95,
            0.975,
            0.99
        ]
    )
)


# ------------------------------------------------------------
# 13. LARGEST POSITIVE AND NEGATIVE RETURNS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("10. EXTREME POSITIVE RETURNS")
print("=" * 70)

print(
    eth.nlargest(
        10,
        "Validation_Log_Return"
    )[
        [
            "Date",
            "ETH_Price",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)

print("\n" + "=" * 70)
print("11. EXTREME NEGATIVE RETURNS")
print("=" * 70)

print(
    eth.nsmallest(
        10,
        "Validation_Log_Return"
    )[
        [
            "Date",
            "ETH_Price",
            "Validation_Log_Return",
            "Validation_Return_Pct"
        ]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# 14. FINAL VALIDATION SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("VALIDATION SUMMARY")
print("=" * 70)

print(f"\nObservations: {len(eth)}")
print(f"Missing dates: {len(missing_dates)}")
print(f"Duplicate dates: {eth['Date'].duplicated().sum()}")
print(f"Missing prices: {eth['ETH_Price'].isna().sum()}")
print(f"Non-positive prices: {(eth['ETH_Price'] <= 0).sum()}")

print(
    f"Daily movements >20%: "
    f"{len(large_moves)}"
)

print(
    f"Consecutive identical prices: "
    f"{len(repeated_prices)}"
)

print(
    f"Possible large one-day reversals: "
    f"{len(reversals)}"
)

print(
    f"Local median outliers >20%: "
    f"{len(local_outliers)}"
)

print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)

print(
    "\nNo observations have been deleted or modified "
    "by this validation script."
)

print(
    "Review the flagged observations before calculating "
    "the final ETH daily log return."
)