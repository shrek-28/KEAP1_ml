import pandas as pd
from itertools import combinations
import argparse 

parser = argparse.ArgumentParser(
    description="Generating a dataset with interactions of features"
)

parser.add_argument("-i", "--input", required=True, help="Input CSV file (descriptors + scores)")
parser.add_argument("-o", "--output", required=True, help="path to interactions CSV file")

args = parser.parse_args()

# Load data
# df = pd.read_csv("data/combined_scores/with_descriptors.csv")
df = pd.read_csv(args.input)

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

# interaction_df.to_csv(
#     "data/engineered_features/descriptor_interactions.csv",
#     index=False
# )

interaction_df.to_csv(args.output, index=False)

print(
    f"Generated {len(interaction_df.columns)-1} interaction features"
)