import pandas as pd 

df = pd.read_csv("data/admet/interpretation_data.csv")

cols = [
    'Genomic_AMES_Mutagenesis',
    'Genomic_Carcinogenesis',
    'Organic_hERG_I_Inhibitor',
    'Organic_hERG_II_Inhibitor',
    'Organic_Liver_Injury_I',
    'Organic_Liver_Injury_II',
]

mask = df[cols].apply(
    lambda col: col.astype(str).str.contains("High toxicity", case=False, na=False) 
).any(axis=1)

toxic_ligands = df[mask]

print(len(toxic_ligands))