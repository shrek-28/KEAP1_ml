import pandas as pd

def merge_csvs(
    input_files,
    output_file,
    key_column="identifier",
    how="inner"
):

    merged = pd.read_csv(input_files[0])

    for file in input_files[1:]:
        df = pd.read_csv(file)

        duplicate_cols = [
            col for col in df.columns
            if col in merged.columns and col != key_column
        ]

        # Remove duplicates from incoming file
        df = df.drop(columns=duplicate_cols)

        merged = merged.merge(
            df,
            on=key_column,
            how=how
        )

    merged.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")
    print(f"Shape: {merged.shape}")

merge_csvs(input_files=["data/engineered_features/with_descriptors.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_transformations.csv"], output_file="data/engineered_features/merged/raw_descriptors_and_transformations.csv")
merge_csvs(input_files=["data/engineered_features/with_descriptors.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_interactions.csv"], output_file="data/engineered_features/merged/raw_descriptors_and_interactions.csv")
merge_csvs(input_files=["data/engineered_features/with_descriptors.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_ratios_both_directions.csv"], output_file="data/engineered_features/merged/raw_descriptors_and_ratios.csv")

merge_csvs(input_files=["data/engineered_features/descriptor_transformations.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_ratios_both_directions.csv"], output_file="data/engineered_features/merged/transformations_and_ratios.csv")
merge_csvs(input_files=["data/engineered_features/descriptor_transformations.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_interactions.csv"], output_file="data/engineered_features/merged/transformations_and_interactions.csv")

merge_csvs(input_files=["data/engineered_features/descriptor_interactions.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_ratios_both_directions.csv"], output_file="data/engineered_features/merged/interactions_and_ratios.csv")
merge_csvs(input_files=["data/engineered_features/descriptor_interactions.csv", "/Users/shreyasree/Documents/GitHub/KEAP1drugdiscovery/data/engineered_features/descriptor_ratios_both_directions.csv", "data/engineered_features/descriptor_transformations.csv", "data/engineered_features/with_descriptors.csv"], output_file="data/engineered_features/merged/all_4_combined.csv")