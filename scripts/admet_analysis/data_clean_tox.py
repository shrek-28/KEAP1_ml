import pandas as pd

# Load the files
tox_df = pd.read_csv("data/admet/toxicity_results.csv")
id_df = pd.read_csv("data/admet/filtered_smiles_data.csv")

# Ensure they have the same number of rows
assert len(tox_df) == len(id_df), "The two files have different numbers of rows."

# Copy the identifier by row order
tox_df["identifier"] = id_df["identifier"].values

# Optional: move identifier to the first column
cols = ["identifier"] + [c for c in tox_df.columns if c != "identifier"]
tox_df = tox_df[cols]

# Save
tox_df.to_csv("data/admet/toxicity_with_identifier.csv", index=False)