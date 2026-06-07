import pandas as pd
import numpy as np
from itertools import permutations

# Load data
df = pd.read_csv("data/combined_scores/with_descriptors.csv")

# Descriptor columns
descriptor_cols = [
    col for col in df.columns
    if col not in ["identifier", "Score"]
]

# Output dataframe
ratio_df = pd.DataFrame()
ratio_df["identifier"] = df["identifier"]
ratio_df["Score"] = df["Score"]

for col1, col2 in permutations(descriptor_cols, 2):

    numerator = pd.to_numeric(df[col1], errors="coerce")
    denominator = pd.to_numeric(df[col2], errors="coerce")

    ratio_df[f"{col1}_div_{col2}"] = np.where(
        denominator != 0,
        numerator / denominator,
        0
    )

ratio_df.to_csv("data/engineered_features/descriptor_ratios_both_directions.csv", index=False)

print(f"Generated {len(ratio_df.columns)-1} ratio features")
print(f"Output shape: {ratio_df.shape}")