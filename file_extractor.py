import pandas as pd
import shutil
from pathlib import Path

# paths
csv_file = "identifiers.csv"
source_folder = Path("all_cnp_files")
destination_folder = Path("selected_cnp_files")

destination_folder.mkdir(exist_ok=True)

# load identifiers
df = pd.read_csv(csv_file)

# get CNP IDs from identifier column
cnp_ids = set(df["identifier"].astype(str))

# copy matching files
for file in source_folder.iterdir():
    if file.is_file():
        file_stem = file.stem  # filename without extension

        if file_stem in cnp_ids:
            shutil.copy2(file, destination_folder / file.name)

print(f"Copied {len(list(destination_folder.iterdir()))} files.")