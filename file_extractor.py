import pandas as pd

df1 = pd.read_csv("data/new_data_pred/top_scorers/top_0.1_percent.csv")
df2 = pd.read_csv("data/docking_files/docking_scores.csv")

# Clean ligand names
df2["Ligand"] = df2["Ligand"].str.replace("minimized_", "", regex=False)

# Identifiers in df1 that are NOT present in df2
missing_identifiers = df1.loc[
    ~df1["identifier"].isin(df2["Ligand"]),
    "identifier"
]

print(missing_identifiers)

for i in missing_identifiers:
    print("minimized_"+i+"_docked.pdbqt")

from pathlib import Path

files_to_delete = [
    "minimized_CNP0093120.1_docked.pdbqt",
    "minimized_CNP0166169.7_docked.pdbqt",
    "minimized_CNP0201578.1_docked.pdbqt",
    "minimized_CNP0188341.1_docked.pdbqt",
    "minimized_CNP0110452.1_docked.pdbqt",
    "minimized_CNP0198595.3_docked.pdbqt",
    "minimized_CNP0561828.2_docked.pdbqt",
    "minimized_CNP0194358.1_docked.pdbqt",
]

directory = Path("data/docking_files/output")  # Replace with the actual folder

for filename in files_to_delete:
    file_path = directory / filename
    if file_path.exists():
        file_path.unlink()
        print(f"Deleted: {filename}")
    else:
        print(f"Not found: {filename}")