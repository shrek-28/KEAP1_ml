import pandas as pd
from pathlib import Path

# Folder containing the CSV files
folder = Path("/Users/shreyasree/Documents/GitHub/KEAP1_ml/data/Identifier_SMILES_Data")

# Get all CSV files
csv_files = sorted(folder.glob("*.csv"))

# Read and combine
combined_df = pd.concat(
    (pd.read_csv(file) for file in csv_files),
    ignore_index=True
)

# Save the combined file
output_path = "data/combined_smiles_data.csv"
combined_df.to_csv(output_path, index=False)

print(f"Combined {len(csv_files)} files into {output_path}")
print(f"Total rows: {len(combined_df)}")