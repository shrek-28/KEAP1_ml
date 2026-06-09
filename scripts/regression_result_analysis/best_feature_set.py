import pandas as pd

df = pd.read_csv("data/regression_result_analysis/combined_results.csv")

# keep only what we need
df_rmse = df[["model", "dataset", "RMSE_mean"]]

# find best dataset per model (lowest RMSE)
best_per_model = df_rmse.loc[
    df_rmse.groupby("model")["RMSE_mean"].idxmin()
].reset_index(drop=True)

# sort for readability
best_per_model = best_per_model.sort_values("RMSE_mean")

print(best_per_model['dataset'].value_counts())

avg_rmse = (
    best_per_model.groupby("dataset")["RMSE_mean"]
    .mean()
    .reset_index()
    .sort_values("RMSE_mean")
)

print(avg_rmse)