import pandas as pd
from pathlib import Path

# ============================================================
# CHECK FLAGGED ETH DATES AGAINST ORIGINAL REFINITIV DATA
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

excel_file = PROJECT_ROOT / "data_raw" / "Dissertation Data.xlsx"

sheet_name = "ETH Daily returns(REFINITIV)"

print("=" * 70)
print("CHECK FLAGGED ETH DATES - ORIGINAL REFINITIV DATA")
print("=" * 70)

# ------------------------------------------------------------
# 1. LOAD ORIGINAL ETH SHEET
# ------------------------------------------------------------

raw = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    header=None
)

# ------------------------------------------------------------
# 2. FIND HISTORICAL DATA HEADER
# ------------------------------------------------------------

header_row = None

for i in range(len(raw)):

    row_values = (
        raw.iloc[i]
        .astype(str)
        .str.strip()
        .tolist()
    )

    if "Exchange Date" in row_values and "Mid Price" in row_values:
        header_row = i
        break

if header_row is None:
    raise ValueError(
        "Could not find Exchange Date / Mid Price header."
    )

print(f"\nHistorical header found at row: {header_row}")

# ------------------------------------------------------------
# 3. CREATE HISTORICAL TABLE
# ------------------------------------------------------------

columns = raw.iloc[header_row].tolist()

eth = raw.iloc[header_row + 1:].copy()

eth.columns = columns

# Keep only the original Refinitiv OHLC/Mid variables

eth = eth[
    [
        "Exchange Date",
        "High",
        "Low",
        "Open",
        "Mid Price"
    ]
].copy()

# ------------------------------------------------------------
# 4. CLEAN TYPES
# ------------------------------------------------------------

eth["Exchange Date"] = pd.to_datetime(
    eth["Exchange Date"],
    errors="coerce",
    dayfirst=True
)

for column in [
    "High",
    "Low",
    "Open",
    "Mid Price"
]:

    eth[column] = pd.to_numeric(
        eth[column],
        errors="coerce"
    )

eth = eth.dropna(
    subset=["Exchange Date"]
).copy()

# ------------------------------------------------------------
# 5. DATES WE WANT TO INVESTIGATE
# ------------------------------------------------------------

flagged_dates = pd.to_datetime(
    [
        "2021-01-04",
        "2021-01-11",
        "2021-05-19",
        "2021-05-23",
        "2021-05-24",
        "2022-06-18",
        "2022-06-19",
        "2022-11-09",
        "2022-11-10",
        "2025-05-08"
    ]
)

# ------------------------------------------------------------
# 6. EXTRACT FLAGGED DATES
# ------------------------------------------------------------

flagged = eth[
    eth["Exchange Date"].isin(flagged_dates)
].copy()

flagged = flagged.sort_values(
    "Exchange Date"
)

# ------------------------------------------------------------
# 7. CHECK WHETHER MID PRICE IS WITHIN HIGH/LOW RANGE
# ------------------------------------------------------------

flagged["Mid_Within_High_Low"] = (
    (flagged["Mid Price"] >= flagged["Low"])
    &
    (flagged["Mid Price"] <= flagged["High"])
)

# ------------------------------------------------------------
# 8. DISPLAY RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("FLAGGED ETH OBSERVATIONS FROM ORIGINAL REFINITIV FILE")
print("=" * 70)

print(
    flagged.to_string(index=False)
)

# ------------------------------------------------------------
# 9. DISPLAY POSSIBLE PROBLEMS
# ------------------------------------------------------------

problems = flagged[
    flagged["Mid_Within_High_Low"] == False
]

print("\n" + "=" * 70)
print("MID PRICES OUTSIDE DAILY HIGH-LOW RANGE")
print("=" * 70)

if len(problems) == 0:

    print(
        "\nNone."
    )

    print(
        "All checked Mid Prices fall within their "
        "corresponding daily High-Low ranges."
    )

else:

    print(
        f"\nWARNING: {len(problems)} observation(s) found."
    )

    print(
        problems.to_string(index=False)
    )

# ------------------------------------------------------------
# 10. FINAL SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    f"\nFlagged dates requested: {len(flagged_dates)}"
)

print(
    f"Flagged dates found: {len(flagged)}"
)

print(
    f"Mid Prices within High-Low range: "
    f"{flagged['Mid_Within_High_Low'].sum()}"
)

print(
    f"Mid Prices outside High-Low range: "
    f"{(~flagged['Mid_Within_High_Low']).sum()}"
)

print("\nCheck complete.")