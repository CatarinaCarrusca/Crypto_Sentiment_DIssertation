import pandas as pd
from pathlib import Path


# ============================================================
# 1. FIND THE DATA_RAW FOLDER
# ============================================================

data_raw_folder = Path(__file__).parent


# ============================================================
# 2. EXCEL FILE
# ============================================================

excel_file = data_raw_folder / "Dissertation Data.xlsx"


# ============================================================
# 3. CHECK THAT PYTHON CAN FIND IT
# ============================================================

print("Looking for Excel file:")
print(excel_file)

print("\nDoes the file exist?")
print(excel_file.exists())

if not excel_file.exists():
    raise FileNotFoundError(
        f"Excel file not found:\n{excel_file}"
    )


# ============================================================
# 4. IMPORT ALL EXCEL SHEETS
# ============================================================

print("\nImporting Excel workbook...")

all_sheets = pd.read_excel(
    excel_file,
    sheet_name=None,
    engine="openpyxl"
)


# ============================================================
# 5. CONFIRM SUCCESS
# ============================================================

print("\n=======================================")
print("SUCCESS - WORKBOOK IMPORTED")
print("=======================================")


# ============================================================
# 6. SHOW ALL SHEET NAMES
# ============================================================

print("\nSheets found:")

for sheet_name in all_sheets.keys():
    print(f"- {sheet_name}")


# ============================================================
# 7. SHOW SIZE OF EACH SHEET
# ============================================================

print("\n=======================================")
print("SHEET SIZES")
print("=======================================")

for sheet_name, df in all_sheets.items():

    print(
        f"{sheet_name}: "
        f"{len(df)} rows x "
        f"{len(df.columns)} columns"
    )


# ============================================================
# 8. SHOW FIRST 5 ROWS OF EACH SHEET
# ============================================================

print("\n=======================================")
print("FIRST 5 ROWS OF EACH SHEET")
print("=======================================")

for sheet_name, df in all_sheets.items():

    print(f"\n--- {sheet_name} ---")

    print(df.head())