from itertools import permutations
import pandas as pd
import numpy as np

ratio_df = pd.read_csv("data/final_features_data/ratios_only.csv")
col_list = ratio_df.columns.tolist()
col_list.remove("Score")

print(ratio_df.columns)

df1 = pd.read_csv("data/combined_scores/docking_score_data_no_outliers.csv")
df2 = pd.read_csv("data/final_complete_descriptor_matrix.csv")
df2 = df2.drop(['Unnamed: 0', 'ConformerCount', 'Aromatic_ring_count'], axis=1, errors='ignore')

df = df2[~df2["identifier"].isin(df1["Ligand"])]

descriptor_cols = [
    col for col in df.columns
    if col not in ["identifier"]
]

# Output dataframe
ratio_df = pd.DataFrame()
ratio_df["identifier"] = df["identifier"]

for col1, col2 in permutations(descriptor_cols, 2):

    numerator = pd.to_numeric(df[col1], errors="coerce")
    denominator = pd.to_numeric(df[col2], errors="coerce")

    ratio_df[f"{col1}_div_{col2}"] = np.where(
        denominator != 0,
        numerator / denominator,
        0
    )

ratio_df = ratio_df[col_list]
print(ratio_df.shape)

ratio_df.to_csv("data/retraining_all_data.csv", index=False)

print(ratio_df.columns)
print()