from pathlib import Path
import pandas as pd
import os
from datetime import datetime

excel_file = Path(
    "/Users/catarinacarrusca/PycharmProjects/"
    "Crypto_Sentiment_Dissertation/data_raw/"
    "Dissertation Data.xlsx"
)

print("=" * 70)
print("CHECKING EXACT EXCEL FILE")
print("=" * 70)

print("\nExact file being checked:")
print(excel_file.resolve())

print("\nFile exists:")
print(excel_file.exists())

print("\nFile size:")
print(os.path.getsize(excel_file), "bytes")

modified = os.path.getmtime(excel_file)

print("\nLast modified:")
print(datetime.fromtimestamp(modified))

print("\nSheets actually inside this file:")

xls = pd.ExcelFile(excel_file)

for sheet in xls.sheet_names:
    print(repr(sheet))

sheet_name = "ETH Trading Volume(yfinance)"

print("\n" + "=" * 70)
print("READING ETH SHEET DIRECTLY")
print("=" * 70)

df = pd.read_excel(
    excel_file,
    sheet_name=sheet_name,
    header=None
)

print("\nShape:")
print(df.shape)

print("\nLAST 20 ROWS ACTUALLY STORED IN EXCEL:")
print(df.tail(20).to_string(index=False))

print("\n" + "=" * 70)
print("SEARCHING ENTIRE SHEET FOR 2025-12-31")
print("=" * 70)

found = False

for col in df.columns:

    matches = (
        df[col]
        .astype(str)
        .str.contains(
            "2025-12-31",
            na=False,
            regex=False
        )
    )

    if matches.any():

        found = True

        print(f"\nFOUND in column {col}:")
        print(df.loc[matches].to_string(index=False))

if not found:
    print("\n2025-12-31 DOES NOT EXIST IN THIS EXCEL SHEET.")

print("\n" + "=" * 70)
print("CHECK COMPLETE")
print("=" * 70)