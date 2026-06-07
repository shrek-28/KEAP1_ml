import pandas as pd
import numpy as np

# --------- HARD-CODED INPUTS ---------
input_file = "data/combined_scores/combined_valid_scores.csv"
column_name = "Score"

# --------- Load data ---------
df = pd.read_csv(input_file)

if column_name not in df.columns:
    raise ValueError(f"Column {column_name} not found")

# Ensure numeric + remove non-finite values
x = pd.to_numeric(df[column_name], errors="coerce")
mask_finite = np.isfinite(x)
df = df[mask_finite]
x = x[mask_finite]

# --------- IQR OUTLIER REMOVAL ---------
Q1 = x.quantile(0.25)
Q3 = x.quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

df_clean = df[(x >= lower) & (x <= upper)]

# --------- Save cleaned dataset ---------
df_clean.to_csv("data/combined_scores/docking_score_data_no_outliers.csv", index=False)

# --------- Quick sanity output ---------
print("Original size:", len(mask_finite))
print("Cleaned size:", len(df_clean))
print("Removed:", len(mask_finite) - len(df_clean), "rows")
print("IQR bounds:", lower, "to", upper)