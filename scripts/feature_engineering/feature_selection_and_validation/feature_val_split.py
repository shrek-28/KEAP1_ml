import pandas as pd
from pathlib import Path

input_file = "data/all_datasets_rmse.csv"
output_folder = "data/rmse_split_results"

Path(output_folder).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(input_file)

for dataset_name, group in df.groupby("dataset"):

    clean_name = Path(dataset_name).stem

    group = group.drop(columns=["dataset"])

    group.to_csv(
        Path(output_folder) / f"{clean_name}.csv",
        index=False
    )

    print(f"Saved: {clean_name}.csv")