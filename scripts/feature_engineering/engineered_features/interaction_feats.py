import pandas as pd
from itertools import combinations

# Load data
df = pd.read_csv("data/combined_scores/with_descriptors.csv")

descriptor_cols = [
    col for col in df.columns
    if col not in ["identifier", "Score"]
]

interaction_features = {
    "identifier": df["identifier"],
    "Score": df["Score"]
}

for col1, col2 in combinations(descriptor_cols, 2):
    interaction_features[f"{col1}_x_{col2}"] = (
        pd.to_numeric(df[col1], errors="coerce")
        * pd.to_numeric(df[col2], errors="coerce")
    )

interaction_df = pd.DataFrame(interaction_features)

interaction_df.to_csv(
    "data/engineered_features/descriptor_interactions.csv",
    index=False
)

print(
    f"Generated {len(interaction_df.columns)-1} interaction features"
)