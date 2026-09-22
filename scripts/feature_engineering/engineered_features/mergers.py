import pandas as pd
import argparse 
from pathlib import Path

parser = argparse.ArgumentParser(
    description="Generating merged datasets"
)

parser.add_argument("-i", "--input", nargs="+", required=True, help="Input CSV file (descriptors + scores)")
parser.add_argument("-o", "--output", required=True, help="path to interactions CSV file")

args = parser.parse_args()

output_file = Path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)

# Load first file
merged = pd.read_csv(args.input[0])

# Merge remaining files
for file in args.input[1:]:
    df = pd.read_csv(file)

    duplicate_cols = [
        col for col in df.columns
        if col in merged.columns and col != "identifier"
    ]

    # Remove duplicate columns from incoming file
    df = df.drop(columns=duplicate_cols)

    merged = merged.merge(
        df,
        on="identifier",
        how="inner"
    )

# Save
merged.to_csv(output_file, index=False)

print(f"Saved: {output_file}")
print(f"Shape: {merged.shape}")