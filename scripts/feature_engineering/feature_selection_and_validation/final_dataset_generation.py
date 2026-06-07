import os
import pandas as pd

def extract_selected_columns(mapping_csv_path, input_folder_path, output_folder_path):
    """
    mapping_csv_path: CSV containing dataset,k,features
    input_folder_path: folder containing all feature CSV files
    output_folder_path: where reduced CSVs will be saved
    """

    os.makedirs(output_folder_path, exist_ok=True)

    df = pd.read_csv(mapping_csv_path)

    for _, row in df.iterrows():
        dataset = row["dataset"]
        features = [f.strip() for f in str(row["features"]).split(",")]

        input_file = os.path.join(input_folder_path, f"{dataset}.csv")

        if not os.path.exists(input_file):
            print(f"[SKIP] Missing file: {input_file}")
            continue

        data = pd.read_csv(input_file)

        # Always include these if present
        mandatory_cols = [
            col for col in ["identifier", "Score"]
            if col in data.columns
        ]

        # Feature columns that actually exist
        selected_features = [
            col for col in features
            if col in data.columns
        ]

        # Preserve order and avoid duplicates
        final_cols = mandatory_cols + [
            col for col in selected_features
            if col not in mandatory_cols
        ]

        if len(final_cols) == 0:
            print(f"[SKIP] No matching columns for {dataset}")
            continue

        reduced = data[final_cols]

        output_file = os.path.join(
            output_folder_path,
            f"{dataset}.csv"
        )

        reduced.to_csv(output_file, index=False)

        print(
            f"[OK] {dataset}: "
            f"{len(selected_features)} feature columns + "
            f"{len(mandatory_cols)} mandatory columns saved"
        )

extract_selected_columns(
    mapping_csv_path="data/final_feature_list.csv",
    input_folder_path="data/spearman_reduced_features",
    output_folder_path="data/final_features_data"
)