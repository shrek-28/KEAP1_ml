import pandas as pd
from scipy.stats import friedmanchisquare

# load data
df = pd.read_csv("data/regression_result_analysis/combined_results.csv")

# pivot: rows = models, columns = datasets, values = RMSE
pivot = df.pivot(index="model", columns="dataset", values="RMSE_mean")

print("\nRMSE matrix (models × datasets):\n")
print(pivot)

# prepare data for Friedman test (each dataset = column)
data = [pivot[col].values for col in pivot.columns]

# Friedman test
stat, p = friedmanchisquare(*data)

print("\n--- Friedman Test ---")
print("Statistic:", stat)
print("p-value:", p)

# average ranks (lower = better dataset)
ranks = pivot.rank(axis=1, method="average", ascending=True)
avg_ranks = ranks.mean().sort_values()

print("\n--- Average Ranks (lower is better) ---")
print(avg_ranks)

# final interpretation
print("\n--- Conclusion ---")
if p < 0.05:
    best = avg_ranks.index[0]
    print("Significant difference detected (p < 0.05).")
    print("Best dataset based on rank:", best)
else:
    print("No statistically significant difference detected (p ≥ 0.05).")
    print("Prefer simplest or most stable dataset (check variance / RMSE mean).")