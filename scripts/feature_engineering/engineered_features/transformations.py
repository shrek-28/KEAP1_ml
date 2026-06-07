import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/combined_scores/with_descriptors.csv")

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

transformed.to_csv("data/engineered_features/descriptor_transformations.csv", index=False)

print(transformed.shape)