from pathlib import Path
import pandas as pd

# Project folders
project_folder = Path(
    "/"
)

raw_folder = project_folder / "data_raw"
clean_folder = project_folder / "data_clean"

# Create data_clean if it does not already exist
clean_folder.mkdir(parents=True, exist_ok=True)


def clean_volume_file(input_filename, output_filename, volume_name):
    input_path = raw_folder / input_filename
    output_path = clean_folder / output_filename

    # Read the Yahoo Finance CSV
    # header=[0, 1] is needed because your file has two header rows:
    # Price and Ticker
    data = pd.read_csv(
        input_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True
    )

    # Select only the Volume column
    volume_data = data["Volume"].copy()

    # Convert the index back into a normal Date column
    volume_data = volume_data.reset_index()

    # Give the columns clear names
    volume_data.columns = ["Date", volume_name]

    # Sort by date
    volume_data = volume_data.sort_values("Date")

    # Remove duplicate dates, if any
    volume_data = volume_data.drop_duplicates(
        subset="Date",
        keep="first"
    )

    # Save the cleaned CSV
    volume_data.to_csv(output_path, index=False)

    print(f"Created: {output_path}")
    print(volume_data.head())
    print(volume_data.tail())
    print(f"Rows: {len(volume_data)}")
    print()


# Clean Bitcoin
clean_volume_file(
    input_filename="ethereum_volume_2021_2025.csv",
    output_filename="ETH_Volume.csv",
    volume_name="ETH_Volume"
)
