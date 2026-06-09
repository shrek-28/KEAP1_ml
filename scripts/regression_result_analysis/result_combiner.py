import os
import pandas as pd

folder_path = "data/regression_results"

cols_to_keep = [
    "dataset",
    "MAE_mean", "MAE_ci",
    "MSE_mean", "MSE_ci",
    "RMSE_mean", "RMSE_ci",
    "R2_mean", "R2_ci",
    "MAPE_mean", "MAPE_ci"
]

all_dfs = []

for file in os.listdir(folder_path):
    if file.endswith(".csv"):
        model_name = file.replace("_results.csv", "").replace(".csv", "")
        
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # keep only required columns (ignore missing safely)
        df = df[[c for c in cols_to_keep if c in df.columns]]
        
        df["model"] = model_name
        
        all_dfs.append(df)

combined = pd.concat(all_dfs, ignore_index=True)
combined.to_csv("data/regression_result_analysis/combined_results.csv", index=False)