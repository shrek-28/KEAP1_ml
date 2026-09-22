import pandas as pd
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Split RMSE results by dataset")
parser.add_argument("-i", "--input", required=True, help="Input RMSE CSV file")
parser.add_argument("-o", "--output", required=True, help="Output folder")
args = parser.parse_args()

output_folder = Path(args.output)
output_folder.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.input)

for dataset_name, group in df.groupby("dataset"):
    clean_name = Path(dataset_name).stem
    group.drop(columns=["dataset"]).to_csv(output_folder / f"{clean_name}.csv", index=False)
    print(f"Saved: {clean_name}.csv")