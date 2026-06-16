import pandas as pd 

df = pd.read_csv("data/admet/toxicity_with_identifier.csv")

interpretation_cols = [col for col in df.columns if col.startswith("Interpretation_")]

interpretation_cols = interpretation_cols + ["identifier", "SMILES"]
df = df[interpretation_cols]

df.columns = df.columns.str.replace("Interpretation_", "", regex=False)

df.to_csv("data/admet/interpretation_data.csv", index=False)