import pandas as pd 

df1 = pd.read_csv("data/docking_files/docking_scores.csv")
df2 = pd.read_csv("data/new_data_pred/top_scorers/top_0.1_percent.csv")

df1['Ligand'] = df1['Ligand'].str.replace('minimized_', '', regex=False)

merged = pd.merge(df1, df2, left_on="Ligand", right_on="identifier", how="inner")

merged.drop(['identifier'], axis=1, inplace=True)

merged = merged.rename(columns={'DockingScore_kcal_per_mol': 'actual_docking_score'})

merged.to_csv("data/merged_intersection.csv", index=False)