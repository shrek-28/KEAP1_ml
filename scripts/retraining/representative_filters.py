#!/usr/bin/env python3

import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser(description="Filter docking results using predicted-score cutoffs.")
parser.add_argument("--input", required=True, help="Input docking score CSV.")
parser.add_argument("--cutoffs", required=True, help="CSV containing top_percent and cutoff_score.")
parser.add_argument("--output_dir", required=True, help="Directory for filtered representative CSVs.")
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

# Read docking scores and cutoff summary
df = pd.read_csv(args.input)
cutoffs = pd.read_csv(args.cutoffs)

if "Score" not in df.columns:
    raise ValueError("Input dataset must contain 'Score' column")

if not {"top_percent", "cutoff_score"}.issubset(cutoffs.columns):
    raise ValueError("Cutoff file must contain 'top_percent' and 'cutoff_score' columns")

# Generate representative sets using each cutoff
for _, row in cutoffs.iterrows():
    pct, cutoff_score = row["top_percent"], row["cutoff_score"]
    top_df = df[df["Score"] < cutoff_score]

    print(f"Top {pct}%: {len(top_df)} molecules (cutoff = {cutoff_score})")
    top_df.to_csv(f"{args.output_dir}/top_{pct}_percent.csv", index=False)

print(f"Saved filtered datasets to: {args.output_dir}")