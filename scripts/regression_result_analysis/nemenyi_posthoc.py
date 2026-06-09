import pandas as pd
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp

# load data
df = pd.read_csv("data/regression_result_analysis/combined_results.csv")

# pivot: rows=models, columns=datasets
pivot = df.pivot(index="model", columns="dataset", values="RMSE_mean")

print("\nRMSE matrix:\n")
print(pivot)

# -------------------------
# Friedman test
# -------------------------
data = [pivot[col].values for col in pivot.columns]
stat, p = friedmanchisquare(*data)

print("\n--- Friedman Test ---")
print("Statistic:", stat)
print("p-value:", p)

# -------------------------
# Average ranks
# -------------------------
ranks = pivot.rank(axis=1, method="average", ascending=True)
avg_ranks = ranks.mean().sort_values()

print("\n--- Average Ranks (lower is better) ---")
print(avg_ranks)

# -------------------------
# Nemenyi post-hoc test
# -------------------------
nemenyi = sp.posthoc_nemenyi_friedman(pivot.values)

nemenyi.index = pivot.columns
nemenyi.columns = pivot.columns

print("\n--- Nemenyi p-value matrix ---")
print(nemenyi)

# -------------------------
# Interpretation helper
# -------------------------
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

# -------------------------
# Final ranking conclusion
# -------------------------
best = avg_ranks.index[0]

print("\n--- Final conclusion ---")
print("Best dataset (Friedman rank):", best)

# check specifically your two contenders if present
if "ratios_only.csv" in avg_ranks.index and "all_4_combined.csv" in avg_ranks.index:
    print("\nDirect comparison:")
    print("ratios_only rank:", avg_ranks["ratios_only.csv"])
    print("all_4_combined rank:", avg_ranks["all_4_combined.csv"])