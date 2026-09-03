from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# SETTINGS
# ============================================================

PROJECT_DIR = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation"
)

EXCEL_FILE = (
    PROJECT_DIR
    / "data_raw"
    / "Dissertation Data.xlsx"
)

OUTPUT_FILE = (
    PROJECT_DIR
    / "data_clean"
    / "eth_volume_clean.csv"
)

SHEET_NAME = "ETH Trading Volume(yfinance)"


# ============================================================
# VERIFIED SOURCE CORRECTION
# ============================================================
# The observation below was independently retrieved directly
# from yfinance for ETH-USD.
#
# It is missing from the saved intermediate Excel workbook,
# although it is present in the original yfinance extraction.
#
# This is NOT interpolation or imputation.
# It restores a verified source observation.

VERIFIED_DATE = pd.Timestamp("2025-12-31")
VERIFIED_ETH_VOLUME = 16451891101


# ============================================================
# START
# ============================================================

print("=" * 70)
print("ETH TRADING VOLUME DATA CLEANING")
print("=" * 70)

print("\nLooking for Excel file:")
print(EXCEL_FILE)

print("\nDoes file exist?")
print(EXCEL_FILE.exists())

if not EXCEL_FILE.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{EXCEL_FILE}"
    )

print("\nExcel file found successfully.")


# ============================================================
# CHECK SHEET NAME
# ============================================================

print("\nChecking Excel sheet names...")

xls = pd.ExcelFile(EXCEL_FILE)

if SHEET_NAME not in xls.sheet_names:

    print("\nERROR: Required sheet not found.")
    print("\nRequested sheet:")
    print(repr(SHEET_NAME))

    print("\nAvailable sheets:")

    for sheet in xls.sheet_names:
        print("-", repr(sheet))

    raise ValueError(
        f"Worksheet {SHEET_NAME!r} not found."
    )

print("\nETH Trading Volume sheet found successfully.")


# ============================================================
# IMPORT RAW EXCEL DATA
# ============================================================

print("\nImporting ETH Trading Volume sheet...")

raw = pd.read_excel(
    EXCEL_FILE,
    sheet_name=SHEET_NAME,
    header=None
)

print("ETH Trading Volume sheet imported successfully.")

print("\nRaw shape:")
print(raw.shape)

print("\nFirst 10 raw rows:")
print(
    raw.head(10).to_string(index=False)
)

print("\nLast 10 raw rows:")
print(
    raw.tail(10).to_string(index=False)
)


# ============================================================
# IDENTIFY DATA COLUMN
# ============================================================

print("\n" + "=" * 70)
print("IDENTIFYING ETH TRADING VOLUME DATA")
print("=" * 70)

# The Excel sheet stores observations in the form:
#
# 2021-01-01,13652004358
#
# in a single Excel column.
#
# Search each column for rows matching that structure.

data_column = None
best_count = 0

for column in raw.columns:

    values = raw[column].astype(str).str.strip()

    possible_rows = values.str.match(
        r"^\d{4}-\d{2}-\d{2}\s*,"
    )

    count = possible_rows.sum()

    if count > best_count:
        best_count = count
        data_column = column


if data_column is None or best_count == 0:

    raise ValueError(
        "Could not identify the ETH trading volume "
        "data column in the Excel sheet."
    )


print("\nETH trading volume data found in raw column:")
print(data_column)

print("\nRows containing possible ETH volume observations:")
print(best_count)


# ============================================================
# EXTRACT DATE AND VOLUME
# ============================================================

raw_text = (
    raw[data_column]
    .astype(str)
    .str.strip()
)

possible_data = raw_text[
    raw_text.str.match(
        r"^\d{4}-\d{2}-\d{2}\s*,"
    )
].copy()

split_data = possible_data.str.split(
    ",",
    n=1,
    expand=True
)

df = pd.DataFrame({
    "Date": split_data[0],
    "ETH_Volume": split_data[1]
})


# ============================================================
# CLEAN DATE VARIABLE
# ============================================================

print("\n" + "=" * 70)
print("CLEANING DATE VARIABLE")
print("=" * 70)

df["Date"] = pd.to_datetime(
    df["Date"].astype(str).str.strip(),
    errors="coerce"
)

invalid_dates = df["Date"].isna().sum()

print("\nInvalid dates found:")
print(invalid_dates)

if invalid_dates > 0:

    print("\nInvalid date rows:")

    print(
        df[
            df["Date"].isna()
        ].to_string(index=False)
    )

df = df.dropna(
    subset=["Date"]
).copy()


# ============================================================
# RESTRICT SAMPLE PERIOD
# ============================================================

before_period_filter = len(df)

df = df[
    (df["Date"] >= pd.Timestamp("2021-01-01"))
    &
    (df["Date"] <= pd.Timestamp("2025-12-31"))
].copy()

outside_period = (
    before_period_filter - len(df)
)

print("\nRows outside 2021-2025 removed:")
print(outside_period)


# ============================================================
# CLEAN ETH TRADING VOLUME
# ============================================================

print("\n" + "=" * 70)
print("CLEANING ETH TRADING VOLUME")
print("=" * 70)

df["ETH_Volume"] = (
    df["ETH_Volume"]
    .astype(str)
    .str.strip()
    .str.replace(",", "", regex=False)
)

df["ETH_Volume"] = pd.to_numeric(
    df["ETH_Volume"],
    errors="coerce"
)

missing_volume = df["ETH_Volume"].isna().sum()

print("\nNumber of missing ETH Volume observations:")
print(missing_volume)

if missing_volume > 0:

    print("\nRows with missing ETH Volume:")

    print(
        df[
            df["ETH_Volume"].isna()
        ].to_string(index=False)
    )


before_missing_removal = len(df)

df = df.dropna(
    subset=["ETH_Volume"]
).copy()

removed_missing = (
    before_missing_removal - len(df)
)

print("\nMissing ETH Volume observations removed:")
print(removed_missing)

print("\nMissing ETH Volume observations remaining:")
print(
    df["ETH_Volume"].isna().sum()
)


# ============================================================
# SORT BEFORE CORRECTION
# ============================================================

df = (
    df
    .sort_values("Date")
    .reset_index(drop=True)
)


# ============================================================
# RESTORE VERIFIED 2025-12-31 OBSERVATION
# ============================================================

print("\n" + "=" * 70)
print("VERIFIED YFINANCE SOURCE CORRECTION")
print("=" * 70)

existing_final_date = df[
    df["Date"] == VERIFIED_DATE
]

if existing_final_date.empty:

    print("\n2025-12-31 is missing from the imported Excel data.")

    print(
        "\nRestoring the observation verified directly "
        "from the original yfinance extraction."
    )

    verified_row = pd.DataFrame({
        "Date": [VERIFIED_DATE],
        "ETH_Volume": [VERIFIED_ETH_VOLUME]
    })

    df = pd.concat(
        [df, verified_row],
        ignore_index=True
    )

    df = (
        df
        .sort_values("Date")
        .reset_index(drop=True)
    )

    print("\nVerified observation added:")

    print(
        df[
            df["Date"] == VERIFIED_DATE
        ].to_string(index=False)
    )

else:

    print("\n2025-12-31 is already present in the imported data.")

    existing_value = existing_final_date[
        "ETH_Volume"
    ].iloc[0]

    print("\nExisting observation:")

    print(
        existing_final_date.to_string(index=False)
    )

    # Important:
    # Do not silently overwrite an existing observation.
    # If Excel already contains the date but the value differs
    # from the independently verified yfinance value, stop and
    # require manual investigation.

    if existing_value != VERIFIED_ETH_VOLUME:

        print("\nWARNING:")
        print(
            "The existing 2025-12-31 value differs "
            "from the verified yfinance observation."
        )

        print("\nExcel value:")
        print(existing_value)

        print("\nVerified yfinance value:")
        print(VERIFIED_ETH_VOLUME)

        raise ValueError(
            "Conflicting ETH Volume values found for "
            "2025-12-31. Review before continuing."
        )

    else:

        print(
            "\nExisting value matches the verified "
            "yfinance observation."
        )

        print("No correction was required.")


# ============================================================
# DUPLICATE DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("DUPLICATE DATE CHECK")
print("=" * 70)

duplicate_count = (
    df["Date"]
    .duplicated()
    .sum()
)

print("\nDuplicate dates found:")
print(duplicate_count)

if duplicate_count > 0:

    print("\nDuplicate observations:")

    duplicates = df[
        df["Date"].duplicated(
            keep=False
        )
    ].sort_values("Date")

    print(
        duplicates.to_string(index=False)
    )

    raise ValueError(
        "Duplicate ETH Volume dates detected. "
        "Review the data before continuing."
    )


# ============================================================
# ZERO OR NEGATIVE VOLUME CHECK
# ============================================================

print("\n" + "=" * 70)
print("ZERO OR NEGATIVE VOLUME CHECK")
print("=" * 70)

non_positive = df[
    df["ETH_Volume"] <= 0
].copy()

print(
    "\nZero or negative ETH Volume observations found:"
)
print(len(non_positive))

if len(non_positive) > 0:

    print("\nProblem observations:")

    print(
        non_positive.to_string(index=False)
    )

    raise ValueError(
        "Zero or negative ETH trading volume detected."
    )


# ============================================================
# CALENDAR DATE CHECK
# ============================================================

print("\n" + "=" * 70)
print("CALENDAR DATE CHECK")
print("=" * 70)

expected_dates = pd.date_range(
    start="2021-01-01",
    end="2025-12-31",
    freq="D"
)

actual_dates = pd.DatetimeIndex(
    df["Date"]
)

missing_dates = (
    expected_dates
    .difference(actual_dates)
)

print("\nExpected calendar days:")
print(len(expected_dates))

print("\nActual ETH Volume observations:")
print(len(df))

print("\nMissing calendar dates:")
print(len(missing_dates))

if len(missing_dates) > 0:

    print("\nMissing dates:")

    for date in missing_dates:
        print(
            date.strftime("%Y-%m-%d")
        )

else:

    print("\nNo calendar dates are missing.")


# ============================================================
# REQUIRE COMPLETE DAILY SERIES
# ============================================================

if len(missing_dates) > 0:

    raise ValueError(
        "ETH Volume dataset is still incomplete. "
        "Missing calendar dates remain."
    )

if len(df) != 1826:

    raise ValueError(
        f"Expected 1826 daily observations "
        f"for 2021-2025, but found {len(df)}."
    )


# ============================================================
# CHECK FINAL DATE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING FINAL DATE: 2025-12-31")
print("=" * 70)

final_observation = df[
    df["Date"] == VERIFIED_DATE
]

if final_observation.empty:

    raise ValueError(
        "2025-12-31 is still missing after "
        "the verified correction."
    )

print("\n2025-12-31 found successfully.")

print("\nFinal observation:")

print(
    final_observation.to_string(index=False)
)


# ============================================================
# ETH VOLUME DESCRIPTIVE CHECKS
# ============================================================

print("\n" + "=" * 70)
print("ETH VOLUME DESCRIPTIVE CHECKS")
print("=" * 70)

print("\nMinimum ETH Volume:")
print(
    df["ETH_Volume"].min()
)

print("\nMaximum ETH Volume:")
print(
    df["ETH_Volume"].max()
)

print("\nMedian ETH Volume:")
print(
    df["ETH_Volume"].median()
)

print("\nMean ETH Volume:")
print(
    df["ETH_Volume"].mean()
)


# ============================================================
# 20 HIGHEST ETH VOLUME OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("20 HIGHEST ETH TRADING VOLUME OBSERVATIONS")
print("=" * 70)

highest_volume = (
    df
    .nlargest(
        20,
        "ETH_Volume"
    )
)

print(
    highest_volume.to_string(index=False)
)


# ============================================================
# 20 LOWEST ETH VOLUME OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("20 LOWEST ETH TRADING VOLUME OBSERVATIONS")
print("=" * 70)

lowest_volume = (
    df
    .nsmallest(
        20,
        "ETH_Volume"
    )
)

print(
    lowest_volume.to_string(index=False)
)


# ============================================================
# TEMPORARY LOG VOLUME CHANGE
# VALIDATION ONLY
# ============================================================

df["Validation_Log_Volume_Change"] = (
    np.log(df["ETH_Volume"])
    -
    np.log(
        df["ETH_Volume"].shift(1)
    )
)


# ============================================================
# 20 LARGEST ABSOLUTE DAILY VOLUME CHANGES
# ============================================================

print("\n" + "=" * 70)
print(
    "20 LARGEST ABSOLUTE DAILY ETH VOLUME CHANGES "
    "- VALIDATION ONLY"
)
print("=" * 70)

validation_df = df.dropna(
    subset=[
        "Validation_Log_Volume_Change"
    ]
).copy()

validation_df["Absolute_Log_Volume_Change"] = (
    validation_df[
        "Validation_Log_Volume_Change"
    ].abs()
)

largest_changes = (
    validation_df
    .nlargest(
        20,
        "Absolute_Log_Volume_Change"
    )
)

print(
    largest_changes[
        [
            "Date",
            "ETH_Volume",
            "Validation_Log_Volume_Change"
        ]
    ].to_string(index=False)
)


# ============================================================
# VERY LARGE ETH VOLUME CHANGES
# ============================================================

print("\n" + "=" * 70)
print(
    "VERY LARGE ETH VOLUME CHANGES "
    "- VALIDATION ONLY"
)
print("=" * 70)

large_changes = df[
    df[
        "Validation_Log_Volume_Change"
    ].abs() > 1
].copy()

print("\nNumber of very large volume changes:")
print(len(large_changes))

if len(large_changes) > 0:

    print(
        large_changes[
            [
                "Date",
                "ETH_Volume",
                "Validation_Log_Volume_Change"
            ]
        ].to_string(index=False)
    )


# ============================================================
# CONSECUTIVE IDENTICAL VOLUME CHECK
# ============================================================

print("\n" + "=" * 70)
print("CONSECUTIVE IDENTICAL ETH VOLUME CHECK")
print("=" * 70)

df["Previous_ETH_Volume"] = (
    df["ETH_Volume"].shift(1)
)

identical = df[
    df["ETH_Volume"]
    ==
    df["Previous_ETH_Volume"]
].copy()

print(
    "\nNumber of consecutive identical "
    "ETH Volume observations:"
)

print(len(identical))

if len(identical) > 0:

    print(
        identical[
            [
                "Date",
                "Previous_ETH_Volume",
                "ETH_Volume"
            ]
        ].to_string(index=False)
    )


# ============================================================
# REMOVE TEMPORARY VALIDATION COLUMNS
# ============================================================

df = df[
    [
        "Date",
        "ETH_Volume"
    ]
].copy()


# ============================================================
# FINAL CLEANING CHECKS
# ============================================================

print("\n" + "=" * 70)
print("FINAL CLEANING CHECKS")
print("=" * 70)

print("\nMissing values:")
print(
    df.isna().sum()
)

print("\nDuplicate dates:")
print(
    df["Date"].duplicated().sum()
)

print(
    "\nZero or negative ETH Volume observations:"
)
print(
    (df["ETH_Volume"] <= 0).sum()
)


# ============================================================
# CLEANING SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("CLEANING SUMMARY")
print("=" * 70)

print("\nNumber of clean ETH Volume observations:")
print(len(df))

print("\nFirst date:")
print(
    df["Date"]
    .min()
    .strftime("%Y-%m-%d")
)

print("\nLast date:")
print(
    df["Date"]
    .max()
    .strftime("%Y-%m-%d")
)

print("\nMissing ETH Volume values:")
print(
    df["ETH_Volume"].isna().sum()
)

print("\nDuplicate dates:")
print(
    df["Date"].duplicated().sum()
)

print("\nMinimum ETH Volume:")
print(
    df["ETH_Volume"].min()
)

print("\nMaximum ETH Volume:")
print(
    df["ETH_Volume"].max()
)


# ============================================================
# FIRST 10 CLEAN OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("FIRST 10 CLEAN ETH VOLUME OBSERVATIONS")
print("=" * 70)

print(
    df.head(10).to_string(index=False)
)


# ============================================================
# LAST 10 CLEAN OBSERVATIONS
# ============================================================

print("\n" + "=" * 70)
print("LAST 10 CLEAN ETH VOLUME OBSERVATIONS")
print("=" * 70)

print(
    df.tail(10).to_string(index=False)
)


# ============================================================
# SAVE CLEAN DATA
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False,
    date_format="%Y-%m-%d"
)


# ============================================================
# SUCCESS
# ============================================================

print("\n" + "=" * 70)
print("SUCCESS")
print("=" * 70)

print("\nClean ETH Trading Volume data saved to:")
print(OUTPUT_FILE)

print("\nFinal columns:")
print(df.columns.tolist())

print("\nFinal shape:")
print(df.shape)

print(
    "\nETH Trading Volume cleaning "
    "completed successfully."
)

print(
    "\nIMPORTANT: The 2025-12-31 observation was "
    "restored only if it was absent from the "
    "intermediate Excel workbook."
)

print(
    "The restored value was obtained directly "
    "from the original yfinance extraction; "
    "it was NOT interpolated or estimated."
)

print(
    "Large changes in trading volume were "
    "flagged for validation only and were "
    "NOT automatically deleted."
)

print(
    "The final ETH volume transformation for "
    "the regression has NOT yet been created."
)