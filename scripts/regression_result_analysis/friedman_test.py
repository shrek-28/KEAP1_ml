#!/usr/bin/env python3

import os
import argparse
import pandas as pd
from scipy.stats import friedmanchisquare

parser = argparse.ArgumentParser(description="Friedman test and rank analysis of regression RMSE results")
parser.add_argument("-i", "--input", required=True, help="Combined regression results CSV")
parser.add_argument("-o", "--output", required=True, help="Output folder")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

df = pd.read_csv(args.input)

pivot = df.pivot(index="model", columns="dataset", values="RMSE_mean")

print("\nRMSE matrix (models × datasets):\n")
print(pivot)
pivot.to_csv(os.path.join(args.output, "rmse_matrix.csv"))

data = [pivot[col].values for col in pivot.columns]
stat, p = friedmanchisquare(*data)

print("\n--- Friedman Test ---")
print("Statistic:", stat)
print("p-value:", p)

friedman_df = pd.DataFrame([{
    "statistic": stat,
    "p_value": p,
    "significant_p<0.05": p < 0.05
}])
friedman_df.to_csv(os.path.join(args.output, "friedman_test_result.csv"), index=False)

ranks = pivot.rank(axis=1, method="average", ascending=True)

print("\nRank matrix (models × datasets):\n")
print(ranks)
ranks.to_csv(os.path.join(args.output, "rank_matrix.csv"))

avg_ranks = ranks.mean().sort_values()
avg_ranks_df = avg_ranks.reset_index()
avg_ranks_df.columns = ["dataset", "avg_rank"]

print("\n--- Average Ranks ---")
print(avg_ranks_df)
avg_ranks_df.to_csv(os.path.join(args.output, "average_ranks.csv"), index=False)

dataset_mean_rmse = df.groupby("dataset")["RMSE_mean"].mean().reset_index()
dataset_mean_rmse.to_csv(os.path.join(args.output, "dataset_mean_rmse.csv"), index=False)

print(f"\nSaved all outputs to: {args.output}")