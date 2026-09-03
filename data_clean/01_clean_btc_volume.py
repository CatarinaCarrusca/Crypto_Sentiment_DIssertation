from pathlib import Path
import pandas as pd
import numpy as np

# ================================================================
# SETTINGS
# ================================================================

print("=" * 70)
print("BTC TRADING VOLUME DATA CLEANING")
print("=" * 70)

# Project folders
project_folder = Path(
    "/Users/catarinacarrusca/PycharmProjects/Crypto_Sentiment_Dissertation"
)

raw_folder = project_folder / "data_raw"
clean_folder = project_folder / "data_clean"

# Create clean folder if it does not already exist
clean_folder.mkdir(parents=True, exist_ok=True)

# Input and output files
excel_file = raw_folder / "Dissertation Data.xlsx"
output_file = clean_folder / "btc_volume_clean.csv"

# Exact Excel sheet name
sheet_name = "BTC Trading Volume(yfinance)"

print("\nLooking for Excel file:")
print(excel_file)

print("\nDoes file exist?")
print(excel_file.exists())

if not excel_file.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{excel_file}"
    )

print("\nExcel file found successfully.")


# ================================================================
# 1. IMPORT BTC TRADING VOLUME SHEET
# ================================================================

print("\nImporting BTC Trading Volume sheet...")

try:
    raw = pd.read_excel(
        excel_file,
        sheet_name=sheet_name,
        header=None
    )
except ValueError:
    print("\nERROR: BTC Trading Volume sheet was not found.")
    print(f"Sheet requested: {sheet_name}")

    xls = pd.ExcelFile(excel_file)

    print("\nAvailable sheets:")
    for sheet in xls.sheet_names:
        print(f"- {sheet}")

    raise

print("BTC Trading Volume sheet imported successfully.")

print("\nRaw shape:")
print(raw.shape)

print("\nFirst 10 raw rows:")
print(raw.head(10).to_string())

print("\nLast 10 raw rows:")
print(raw.tail(10).to_string())


# ================================================================
# 2. IDENTIFY BTC TRADING VOLUME DATA
# ================================================================

print("\n" + "=" * 70)
print("IDENTIFYING BTC TRADING VOLUME DATA")
print("=" * 70)

# The Excel sheet stores the date and volume together in one column,
# for example:
#
# 2021-01-01,40730301359
#
# Find the column containing these observations.

data_column = None

for col in raw.columns:
    sample = raw[col].dropna().astype(str)

    if sample.str.contains(
        r"\d{4}-\d{2}-\d{2},",
        regex=True
    ).any():
        data_column = col
        break

if data_column is None:
    raise ValueError(
        "Could not identify the BTC trading volume data column."
    )

print("\nBTC trading volume data found in raw column:")
print(data_column)

btc_raw = raw[[data_column]].copy()

btc_raw.columns = ["Raw_Data"]

# Keep rows that look like:
# YYYY-MM-DD,volume
btc_raw["Raw_Data"] = (
    btc_raw["Raw_Data"]
    .astype(str)
    .str.strip()
)

btc_raw = btc_raw[
    btc_raw["Raw_Data"].str.match(
        r"^\d{4}-\d{2}-\d{2},"
    )
].copy()

print("\nRows containing possible BTC volume observations:")
print(len(btc_raw))


# ================================================================
# 3. SPLIT DATE AND BTC VOLUME
# ================================================================

split_data = btc_raw["Raw_Data"].str.split(
    ",",
    n=1,
    expand=True
)

btc = pd.DataFrame()

btc["Date"] = split_data[0].str.strip()
btc["BTC_Volume"] = split_data[1].str.strip()


# ================================================================
# 4. CLEAN DATE VARIABLE
# ================================================================

print("\n" + "=" * 70)
print("CLEANING DATE VARIABLE")
print("=" * 70)

btc["Date"] = pd.to_datetime(
    btc["Date"],
    errors="coerce"
)

invalid_dates = btc["Date"].isna().sum()

print("\nInvalid dates found:")
print(invalid_dates)

# Remove invalid dates
btc = btc.dropna(subset=["Date"]).copy()

# Keep only dissertation period
before_period_filter = len(btc)

btc = btc[
    (btc["Date"] >= "2021-01-01") &
    (btc["Date"] <= "2025-12-31")
].copy()

outside_period = before_period_filter - len(btc)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ================================================================
# 5. CLEAN BTC TRADING VOLUME
# ================================================================

print("\n" + "=" * 70)
print("CLEANING BTC TRADING VOLUME")
print("=" * 70)

btc["BTC_Volume"] = pd.to_numeric(
    btc["BTC_Volume"],
    errors="coerce"
)

missing_volume = btc["BTC_Volume"].isna().sum()

print("\nNumber of missing BTC Volume observations:")
print(missing_volume)

if missing_volume > 0:
    print("\nDates with missing BTC Volume:")
    print(
        btc.loc[
            btc["BTC_Volume"].isna(),
            ["Date", "BTC_Volume"]
        ].to_string(index=False)
    )

# Remove genuinely missing observations
before_missing = len(btc)

btc = btc.dropna(
    subset=["BTC_Volume"]
).copy()

removed_missing = before_missing - len(btc)

print("\nMissing BTC Volume observations removed:")
print(removed_missing)

print("\nMissing BTC Volume observations remaining:")
print(btc["BTC_Volume"].isna().sum())


# ================================================================
# 6. ADD VERIFIED 2025-12-31 BTC VOLUME IF MISSING
# ================================================================

print("\n" + "=" * 70)
print("CHECKING VERIFIED FINAL BTC OBSERVATION")
print("=" * 70)

# This observation was manually verified using Yahoo Finance
# Historical Data for BTC-USD:
#
# Date:   2025-12-31
# Volume: 33,830,210,616
#
# It is added ONLY if the date is absent from the imported raw data.

verified_date = pd.Timestamp("2025-12-31")
verified_volume = 33830210616

if verified_date not in btc["Date"].values:

    print("\n2025-12-31 is missing from the imported BTC dataset.")
    print("Adding the manually verified Yahoo Finance observation.")

    verified_row = pd.DataFrame({
        "Date": [verified_date],
        "BTC_Volume": [verified_volume]
    })

    btc = pd.concat(
        [btc, verified_row],
        ignore_index=True
    )

    print("\nAdded:")
    print(
        verified_row.to_string(
            index=False
        )
    )

else:

    print("\n2025-12-31 already exists in the imported BTC dataset.")

    existing_value = btc.loc[
        btc["Date"] == verified_date,
        "BTC_Volume"
    ].iloc[0]

    print("\nExisting BTC Volume:")
    print(existing_value)

    # Check that the existing value agrees with the verified value
    if existing_value != verified_volume:

        print("\nWARNING:")
        print(
            "The existing 2025-12-31 BTC Volume differs "
            "from the manually verified Yahoo Finance value."
        )

        print("\nExisting value:")
        print(existing_value)

        print("\nVerified Yahoo Finance value:")
        print(verified_volume)

        raise ValueError(
            "2025-12-31 BTC Volume does not match the "
            "verified Yahoo Finance observation."
        )

    else:
        print(
            "\nExisting value matches the verified "
            "Yahoo Finance observation."
        )


# ================================================================
# 7. SORT DATA
# ================================================================

btc = btc.sort_values(
    "Date"
).reset_index(drop=True)


# ================================================================
# 8. DUPLICATE DATE CHECK
# ================================================================

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

duplicate_dates = btc["Date"].duplicated().sum()

print("\nDuplicate dates found:")
print(duplicate_dates)

if duplicate_dates > 0:

    print("\nDuplicate observations:")
    print(
        btc[
            btc["Date"].duplicated(
                keep=False
            )
        ].to_string(index=False)
    )

    raise ValueError(
        "Duplicate BTC Volume dates found."
    )


# ================================================================
# 9. ZERO OR NEGATIVE VOLUME CHECK
# ================================================================

print("\n" + "=" * 70)
print("ZERO OR NEGATIVE VOLUME CHECK")
print("=" * 70)

non_positive = (
    btc["BTC_Volume"] <= 0
).sum()

print("\nZero or negative BTC Volume observations found:")
print(non_positive)

if non_positive > 0:

    print(
        btc.loc[
            btc["BTC_Volume"] <= 0,
            ["Date", "BTC_Volume"]
        ].to_string(index=False)
    )

    raise ValueError(
        "Zero or negative BTC Volume observations found."
    )


# ================================================================
# 10. CALENDAR DATE CHECK
# ================================================================

print("\n" + "=" * 70)
print("CALENDAR DATE CHECK")
print("=" * 70)

expected_dates = pd.date_range(
    start="2021-01-01",
    end="2025-12-31",
    freq="D"
)

actual_dates = pd.DatetimeIndex(
    btc["Date"]
)

missing_dates = expected_dates.difference(
    actual_dates
)

print("\nExpected calendar days:")
print(len(expected_dates))

print("\nActual BTC Volume observations:")
print(len(btc))

print("\nMissing calendar dates:")
print(len(missing_dates))

if len(missing_dates) > 0:

    print("\nMissing dates:")

    for date in missing_dates:
        print(date.strftime("%Y-%m-%d"))


# ================================================================
# 11. REQUIRE COMPLETE 2021-2025 SERIES
# ================================================================

print("\n" + "=" * 70)
print("COMPLETE SERIES CHECK")
print("=" * 70)

if len(btc) != 1826:

    raise ValueError(
        f"Expected 1826 BTC Volume observations, "
        f"but found {len(btc)}."
    )

if len(missing_dates) != 0:

    raise ValueError(
        "BTC Volume dataset still contains missing calendar dates."
    )

print("\nPASS:")
print("BTC Volume contains all 1826 calendar days.")

print("\nDate range:")
print(
    btc["Date"].min().strftime("%Y-%m-%d"),
    "to",
    btc["Date"].max().strftime("%Y-%m-%d")
)


# ================================================================
# 12. VERIFY 2025-12-31
# ================================================================

print("\n" + "=" * 70)
print("VERIFYING 2025-12-31")
print("=" * 70)

final_observation = btc[
    btc["Date"] == verified_date
]

if final_observation.empty:

    raise ValueError(
        "2025-12-31 is still missing."
    )

print("\nFinal verified BTC observation:")
print(
    final_observation.to_string(
        index=False
    )
)

if (
    final_observation["BTC_Volume"].iloc[0]
    != verified_volume
):

    raise ValueError(
        "Final BTC Volume for 2025-12-31 is incorrect."
    )

print("\nPASS:")
print(
    "2025-12-31 BTC Volume = 33,830,210,616"
)


# ================================================================
# 13. BTC VOLUME DESCRIPTIVE CHECKS
# ================================================================

print("\n" + "=" * 70)
print("BTC VOLUME DESCRIPTIVE CHECKS")
print("=" * 70)

print("\nMinimum BTC Volume:")
print(btc["BTC_Volume"].min())

print("\nMaximum BTC Volume:")
print(btc["BTC_Volume"].max())

print("\nMedian BTC Volume:")
print(btc["BTC_Volume"].median())

print("\nMean BTC Volume:")
print(btc["BTC_Volume"].mean())


# ================================================================
# 14. HIGHEST BTC VOLUME OBSERVATIONS
# ================================================================

print("\n" + "=" * 70)
print("20 HIGHEST BTC TRADING VOLUME OBSERVATIONS")
print("=" * 70)

highest_volume = (
    btc.nlargest(
        20,
        "BTC_Volume"
    )
)

print(
    highest_volume[
        ["Date", "BTC_Volume"]
    ].to_string(index=False)
)


# ================================================================
# 15. LOWEST BTC VOLUME OBSERVATIONS
# ================================================================

print("\n" + "=" * 70)
print("20 LOWEST BTC TRADING VOLUME OBSERVATIONS")
print("=" * 70)

lowest_volume = (
    btc.nsmallest(
        20,
        "BTC_Volume"
    )
)

print(
    lowest_volume[
        ["Date", "BTC_Volume"]
    ].to_string(index=False)
)


# ================================================================
# 16. DAILY LOG VOLUME CHANGES - VALIDATION ONLY
# ================================================================

print("\n" + "=" * 70)
print(
    "20 LARGEST ABSOLUTE DAILY BTC VOLUME CHANGES "
    "- VALIDATION ONLY"
)
print("=" * 70)

btc["Validation_Log_Volume_Change"] = np.log(
    btc["BTC_Volume"]
).diff()

largest_changes = (
    btc.dropna(
        subset=["Validation_Log_Volume_Change"]
    )
    .assign(
        Absolute_Change=lambda x:
        x["Validation_Log_Volume_Change"].abs()
    )
    .nlargest(
        20,
        "Absolute_Change"
    )
)

print(
    largest_changes[
        [
            "Date",
            "BTC_Volume",
            "Validation_Log_Volume_Change"
        ]
    ].to_string(index=False)
)


# ================================================================
# 17. VERY LARGE BTC VOLUME CHANGES
# ================================================================

print("\n" + "=" * 70)
print("VERY LARGE BTC VOLUME CHANGES - VALIDATION ONLY")
print("=" * 70)

large_changes = btc[
    btc["Validation_Log_Volume_Change"].abs() > 1
].copy()

print("\nNumber of very large volume changes:")
print(len(large_changes))

if len(large_changes) > 0:

    print(
        large_changes[
            [
                "Date",
                "BTC_Volume",
                "Validation_Log_Volume_Change"
            ]
        ].to_string(index=False)
    )


# ================================================================
# 18. CONSECUTIVE IDENTICAL VOLUME CHECK
# ================================================================

print("\n" + "=" * 70)
print("CONSECUTIVE IDENTICAL BTC VOLUME CHECK")
print("=" * 70)

btc["Previous_Volume"] = btc["BTC_Volume"].shift(1)

identical = btc[
    btc["BTC_Volume"]
    == btc["Previous_Volume"]
].copy()

print(
    "\nNumber of consecutive identical "
    "BTC Volume observations:"
)
print(len(identical))

if len(identical) > 0:

    print(
        identical[
            [
                "Date",
                "Previous_Volume",
                "BTC_Volume"
            ]
        ].to_string(index=False)
    )


# ================================================================
# 19. REMOVE TEMPORARY VALIDATION VARIABLES
# ================================================================

btc = btc[
    [
        "Date",
        "BTC_Volume"
    ]
].copy()


# ================================================================
# 20. FINAL CLEANING CHECKS
# ================================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(btc.isna().sum())

print("\nDuplicate dates:")
print(
    btc["Date"].duplicated().sum()
)

print("\nZero or negative BTC Volume observations:")
print(
    (btc["BTC_Volume"] <= 0).sum()
)


# ================================================================
# 21. CLEANING SUMMARY
# ================================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean BTC Volume observations:")
print(len(btc))

print("\nFirst date:")
print(
    btc["Date"].min().strftime("%Y-%m-%d")
)

print("\nLast date:")
print(
    btc["Date"].max().strftime("%Y-%m-%d")
)

print("\nMissing BTC Volume values:")
print(
    btc["BTC_Volume"].isna().sum()
)

print("\nDuplicate dates:")
print(
    btc["Date"].duplicated().sum()
)

print("\nMinimum BTC Volume:")
print(
    btc["BTC_Volume"].min()
)

print("\nMaximum BTC Volume:")
print(
    btc["BTC_Volume"].max()
)


# ================================================================
# 22. FIRST AND LAST OBSERVATIONS
# ================================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN BTC VOLUME OBSERVATIONS")
print("=" * 70)

print(
    btc.head(10).to_string(
        index=False
    )
)

print("\n" + "=" * 70)
print("LAST 10 CLEAN BTC VOLUME OBSERVATIONS")
print("=" * 70)

print(
    btc.tail(10).to_string(
        index=False
    )
)


# ================================================================
# 23. SAVE CLEAN DATA
# ================================================================

btc.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean BTC Trading Volume data saved to:")
print(output_file)

print("\nFinal columns:")
print(btc.columns.tolist())

print("\nFinal shape:")
print(btc.shape)

print("\nBTC Trading Volume cleaning completed successfully.")

print(
    "\n2025-12-31 was manually verified using "
    "Yahoo Finance historical data."
)

print(
    "Verified BTC Volume for 2025-12-31: "
    "33,830,210,616"
)

print(
    "\nIMPORTANT: Large changes in trading volume were "
    "flagged for validation only and were NOT automatically deleted."
)

print(
    "The final BTC volume transformation for the regression "
    "has NOT yet been created."
)