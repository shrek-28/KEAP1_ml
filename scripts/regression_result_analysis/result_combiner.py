#!/usr/bin/env python3

import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Combine regression model results")
parser.add_argument("-i", "--input", required=True, help="Input folder containing regression result CSVs")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
args = parser.parse_args()

cols_to_keep = [
    "dataset",
    "MAE_mean", "MAE_ci",
    "MSE_mean", "MSE_ci",
    "RMSE_mean", "RMSE_ci",
    "R2_mean", "R2_ci",
    "MAPE_mean", "MAPE_ci"
]

all_dfs = []

for file in os.listdir(args.input):
    if not file.endswith(".csv"):
        continue

    model_name = file.replace("_results.csv", "").replace(".csv", "")
    df = pd.read_csv(os.path.join(args.input, file))
    df = df[[c for c in cols_to_keep if c in df.columns]]
    df["model"] = model_name
    all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
combined.to_csv(args.output, index=False)

print(f"Saved: {args.output}")
print(f"Shape: {combined.shape}")