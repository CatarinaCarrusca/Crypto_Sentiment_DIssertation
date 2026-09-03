import pandas as pd
from pathlib import Path


# ============================================================
# FILE LOCATIONS
# ============================================================

project_folder = Path(__file__).parent.parent

excel_file = (
    project_folder
    / "data_raw"
    / "Dissertation Data.xlsx"
)


# ============================================================
# CHECK FILE
# ============================================================

if not excel_file.exists():

    raise FileNotFoundError(
        f"Excel workbook not found:\n{excel_file}"
    )


# ============================================================
# READ BTC SHEET WITHOUT ASSUMING A HEADER
# ============================================================

btc_raw = pd.read_excel(
    excel_file,
    sheet_name="BTC Daily retruns(REFINITIV) ",
    header=None,
    engine="openpyxl"
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n=======================================")
print("BTC RAW DATA")
print("=======================================")

print("\nShape:")
print(btc_raw.shape)


# ============================================================
# SHOW FIRST 40 ROWS COMPLETELY
# ============================================================

pd.set_option(
    "display.max_columns",
    None
)

pd.set_option(
    "display.width",
    200
)

pd.set_option(
    "display.max_colwidth",
    50
)

print("\nFirst 40 rows:")
print(
    btc_raw.head(40).to_string(
        index=True,
        header=False
    )
)


# ============================================================
# SEARCH FOR POSSIBLE HEADER ROW
# ============================================================

print("\n=======================================")
print("ROWS CONTAINING DATE/PRICE HEADINGS")
print("=======================================")

for index, row in btc_raw.iterrows():

    row_text = " ".join(
        row.dropna()
        .astype(str)
        .tolist()
    ).lower()

    if (
        "exchange date" in row_text
        or "date" in row_text
        or "close" in row_text
        or "mid price" in row_text
    ):

        print(f"\nRow {index}:")
        print(row.dropna().tolist())