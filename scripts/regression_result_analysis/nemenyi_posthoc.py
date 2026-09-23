#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

parser = argparse.ArgumentParser(description="Friedman and Nemenyi post-hoc analysis of regression results")
parser.add_argument("-i", "--input", required=True, help="Combined regression results CSV")
parser.add_argument("-o", "--output", required=True, help="Output folder")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

df = pd.read_csv(args.input)

pivot = df.pivot(index="model", columns="dataset", values="RMSE_mean")

print("\nRMSE matrix:\n")
print(pivot)

data = [pivot[col].values for col in pivot.columns]
stat, p = friedmanchisquare(*data)

print("\n--- Friedman Test ---")
print("Statistic:", stat)
print("p-value:", p)

ranks = pivot.rank(axis=1, method="average", ascending=True)
avg_ranks = ranks.mean().sort_values()

print("\n--- Average Ranks (lower is better) ---")
print(avg_ranks)

nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)
nemenyi.index = pivot.columns
nemenyi.columns = pivot.columns

print("\n--- Nemenyi p-value matrix ---")
print(nemenyi)

print("\n--- Significant differences (p < 0.05) ---")
sig_pairs = []

for i in nemenyi.index:
    for j in nemenyi.columns:
        if i != j and nemenyi.loc[i, j] < 0.05:
            sig_pairs.append((i, j, nemenyi.loc[i, j]))

if len(sig_pairs) == 0:
    print("No significant pairwise differences.")
else:
    for a, b, pval in sig_pairs:
        print(f"{a} vs {b} -> p = {pval:.4g}")

best = avg_ranks.index[0]

print("\n--- Final conclusion ---")
print("Best dataset (Friedman rank):", best)

if "ratios_only.csv" in avg_ranks.index and "all_4_combined.csv" in avg_ranks.index:
    print("\nDirect comparison:")
    print("ratios_only rank:", avg_ranks["ratios_only.csv"])
    print("all_4_combined rank:", avg_ranks["all_4_combined.csv"])

pivot.to_csv(os.path.join(args.output, "rmse_matrix.csv"))
avg_ranks.to_csv(os.path.join(args.output, "average_ranks.csv"), header=["average_rank"])
nemenyi.to_csv(os.path.join(args.output, "nemenyi_pvalues.csv"))

sig_df = pd.DataFrame(sig_pairs, columns=["dataset_1", "dataset_2", "p_value"])
sig_df.to_csv(os.path.join(args.output, "significant_pairs.csv"), index=False)

print(f"\nAll result CSV files saved to: {args.output}")