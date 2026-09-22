import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(
    description="Generating a dataset with transformations (log, square, cube, square root, cube root)"
)

parser.add_argument("-i", "--input", required=True, help="Input CSV file (descriptors + scores)")
parser.add_argument("-o", "--output", required=True, help="path to interactions CSV file")

args = parser.parse_args()

# Load data
# df = pd.read_csv("data/combined_scores/with_descriptors.csv")
df = pd.read_csv(args.input)

# Keep identifier
transformed = pd.DataFrame()
transformed["identifier"] = df["identifier"]
transformed["Score"] = df["Score"]

# Select descriptor columns
# Exclude identifier and Score
descriptor_cols = [
    col for col in df.columns
    if col not in ["identifier", "Score"]
]

for col in descriptor_cols:

    x = pd.to_numeric(df[col], errors="coerce")

    transformed[f"{col}_sq"] = x**2
    transformed[f"{col}_sqrt"] = np.where(x >= 0, np.sqrt(x), np.nan)

    transformed[f"{col}_cube"] = x**3
    transformed[f"{col}_cuberoot"] = np.cbrt(x)

    # Log only for positive values
    transformed[f"{col}_log"] = np.where(x > 0, np.log(x), np.nan)

# transformed.to_csv("data/engineered_features/descriptor_transformations.csv", index=False)
transformed.to_csv(args.output, index=False)
print(transformed.shape)