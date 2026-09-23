#!/usr/bin/env python3

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Extract top-scoring molecules from prediction results.")
parser.add_argument("--input", required=True, help="Input predictions CSV.")
parser.add_argument("--output_dir", required=True, help="Directory for top-scoring CSV files.")
parser.add_argument("--summary", required=True, help="Output summary CSV path.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Read prediction results
pred_df = pd.read_csv(args.input)

if "predicted_docking_score" not in pred_df.columns:
    raise ValueError("Input dataset must contain 'predicted_docking_score' column")

percentages = [0.1, 0.5, 1, 2, 5, 10]
results = []

# Extract top-scoring molecules at each cutoff
for pct in percentages:
    n = max(1, int(len(pred_df) * pct / 100))
    top_df = pred_df.nsmallest(n, "predicted_docking_score")
    top_df.to_csv(f"{args.output_dir}/top_{pct}_percent.csv", index=False)

    results.append({
        "top_percent": pct,
        "num_molecules": len(top_df),
        "cutoff_score": top_df["predicted_docking_score"].max()
    })

# Save summary
summary_df = pd.DataFrame(results)
summary_df.to_csv(args.summary, index=False)

print(f"Saved top scorers to: {args.output_dir}")
print(f"Saved summary to: {args.summary}")
print(summary_df)