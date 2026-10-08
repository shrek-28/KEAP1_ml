import pandas as pd
from rdkit import Chem
from rdkit.Chem.EState.EState import EStateIndices

def calculate_estate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return None, None, None

    estate_values = list(EStateIndices(mol))

    if len(estate_values) == 0:
        return None, None, None

    avg_estate = sum(estate_values) / len(estate_values)
    min_estate = min(estate_values)
    max_estate = max(estate_values)

    return avg_estate, min_estate, max_estate


# Example: CSV containing a column named 'SMILES'
df = pd.read_csv("input.csv")

results = df["SMILES"].apply(calculate_estate_descriptors)

df["Avg_EState"] = results.apply(lambda x: x[0])
df["Min_EState"] = results.apply(lambda x: x[1])
df["Max_EState"] = results.apply(lambda x: x[2])

df.to_csv("estate_descriptors.csv", index=False)

print(df.head())