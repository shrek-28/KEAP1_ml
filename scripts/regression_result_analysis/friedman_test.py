import pandas as pd
from scipy.stats import friedmanchisquare

df = pd.read_csv("data/regression_result_analysis/combined_results.csv")

pivot = df.pivot(index="model", columns="dataset", values="RMSE_mean")

print("\nRMSE matrix (models × datasets):\n")
print(pivot)

pivot.to_csv("data/stat_tests/rmse_matrix.csv")

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

friedman_df.to_csv("data/stat_tests/friedman_test_result.csv", index=False)

# -----------------------------
# 4. Rank computation (lower RMSE = better rank)
# -----------------------------
ranks = pivot.rank(axis=1, method="average", ascending=True)

print("\nRank matrix (models × datasets):\n")
print(ranks)

ranks.to_csv("data/stat_tests/rank_matrix.csv")

# Average rank per dataset
avg_ranks = ranks.mean().sort_values()

avg_ranks_df = avg_ranks.reset_index()
avg_ranks_df.columns = ["dataset", "avg_rank"]

print("\n--- Average Ranks ---")
print(avg_ranks_df)

avg_ranks_df.to_csv("data/stat_tests/average_ranks.csv", index=False)

# -----------------------------
# 5. Optional: dataset-level mean RMSE (useful plot)
# -----------------------------
dataset_mean_rmse = df.groupby("dataset")["RMSE_mean"].mean().reset_index()
dataset_mean_rmse.to_csv("data/stat_tests/dataset_mean_rmse.csv", index=False)

print("\nSaved all outputs to /data/stat_tests folder")