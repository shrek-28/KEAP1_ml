import pandas as pd

descriptors = pd.read_csv("data/final_complete_descriptor_matrix.csv")
scores = pd.read_csv("data/combined_scores/docking_score_data_no_outliers.csv")

combined = descriptors.merge(
    scores,
    left_on="identifier",
    right_on="Ligand",
    how="inner"
)

combined.drop(columns=["Ligand", "Unnamed: 0", "AromaticRingCount", "ConformerCount"], inplace=True)
combined.to_csv("data/combined_scores/with_descriptors.csv", index=False)
combined.to_csv("data/engineered_features/with_descriptors.csv", index=False)