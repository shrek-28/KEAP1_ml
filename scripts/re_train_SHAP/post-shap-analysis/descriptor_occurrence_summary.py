import argparse
import pandas as pd
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Create a union-based summary table from descriptor occurrence CSVs."
)

parser.add_argument(
    "-i", "--input",
    required=True,
    help="Folder containing descriptor occurrence CSV files"
)

parser.add_argument(
    "-o", "--output",
    required=True,
    help="Output summary CSV path"
)

args = parser.parse_args()

input_folder = Path(args.input)

tables = []

for csv_file in sorted(input_folder.glob("*.csv")):

    df = pd.read_csv(csv_file)

    # Use filename (without extension) as the column name
    column_name = csv_file.stem
    column_name = column_name.replace("_descriptor_counts", "")  # Remove prefix if present

    # Keep only Descriptor and Occurrence
    df = df[["Descriptor", "Occurrence"]].copy()

    # Rename Occurrence to the filename
    df = df.rename(columns={"Occurrence": column_name})

    # Set Descriptor as index
    df = df.set_index("Descriptor")

    tables.append(df)

# UNION of all descriptors across all files
summary = pd.concat(tables, axis=1, join="outer").fillna(0)

# Convert occurrence values to integers
summary = summary.astype(int)

# Put Descriptor back as a column
summary = summary.reset_index()

# Save
summary.to_csv(args.output, index=False)

print(f"Processed {len(tables)} files.")
print(f"Total unique descriptors: {len(summary)}")
print(f"Saved summary to: {args.output}")