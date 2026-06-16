import pandas as pd 

df1 = pd.read_csv("data/combined_smiles_data.csv")
df2 = pd.read_csv("data/merged_intersection.csv")

merged = pd.merge(df1, df2, left_on="identifier", right_on="Ligand", how="inner")

merged.drop(['Ligand', 'actual_docking_score', 'predicted_docking_score'], axis=1, inplace=True)
merged = merged.rename(columns={'canonical_smiles': 'SMILES'})

merged.to_csv("data/admet/filtered_smiles_data.csv", index=False)

df = merged[["SMILES"]]
df.to_csv("data/admet/filtered_smiles_only.csv", index=False)