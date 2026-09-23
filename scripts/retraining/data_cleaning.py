#!/usr/bin/env python3

import argparse
from itertools import permutations
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description="Generate all pairwise descriptor ratios for retraining data.")
parser.add_argument("--ratio_ip", required=True, help="Input ratio CSV used to define column order.")
parser.add_argument("--docking_ip", required=True, help="Docking score CSV; existing ligands are excluded.")
parser.add_argument("--desc", required=True, help="Input complete descriptor matrix CSV.")
parser.add_argument("--output", required=True, help="Output CSV path.")
args = parser.parse_args()

# Read ratio reference dataset and retain its column order
ratio_reference = pd.read_csv(args.ratio_ip)
col_list = ratio_reference.columns.tolist()
col_list.remove("Score")

# Read docking data and descriptor matrix
df1 = pd.read_csv(args.docking_ip)
df2 = pd.read_csv(args.desc)
df2 = df2.drop(["Unnamed: 0", "ConformerCount", "Aromatic_ring_count"], axis=1, errors="ignore")

# Remove ligands already present in the docking dataset
df = df2[~df2["identifier"].isin(df1["Ligand"])].copy()
descriptor_cols = [col for col in df.columns if col != "identifier"]

# Generate all ordered descriptor ratios
ratio_df = pd.DataFrame({"identifier": df["identifier"]})
for col1, col2 in permutations(descriptor_cols, 2):
    numerator = pd.to_numeric(df[col1], errors="coerce")
    denominator = pd.to_numeric(df[col2], errors="coerce")
    ratio_df[f"{col1}_div_{col2}"] = np.where(denominator != 0, numerator / denominator, 0)

# Match reference column order and save
ratio_df = ratio_df[col_list]
ratio_df.to_csv(args.output, index=False)

print(f"Output shape: {ratio_df.shape}")
print(f"Saved to: {args.output}")