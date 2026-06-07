## combines all data to produce csv of files that passed and csv of files that failed

import pandas as pd
from pathlib import Path

folder = Path("data/docking_scores_representatives")

valid_dfs = []
failed_dfs = []

for file in folder.glob("*.csv"):
    try:
        df = pd.read_csv(file)

        if df.empty:
            continue

        # Clean ligand names
        df["Ligand"] = (
            df["Ligand"]
            .str.replace("minimized_", "", regex=False)
            .str.replace(".pdbqt", "", regex=False)
        )

        # Identify failed entries
        failed_mask = (
            df["Score"].astype(str).str.upper().eq("FAILED") 
            | df["Score"].astype(str).str.upper().eq("SKIPPED") 
            | pd.to_numeric(df["Score"], errors="coerce").fillna(-999).eq(0)
        )

        failed_dfs.append(df[failed_mask])
        valid_dfs.append(df[~failed_mask])

    except pd.errors.EmptyDataError:
        print(f"Skipping empty file: {file.name}")
    except Exception as e:
        print(f"Error in {file.name}: {e}")

# Save valid scores
if valid_dfs:
    valid_df = pd.concat(valid_dfs, ignore_index=True)
    valid_df.to_csv("data/combined_scores/combined_valid_scores.csv", index=False)

# Save failed scores
if failed_dfs:
    failed_df = pd.concat(failed_dfs, ignore_index=True)
    failed_df.to_csv("data/combined_scores/combined_failed_scores.csv", index=False)

print("Done.")