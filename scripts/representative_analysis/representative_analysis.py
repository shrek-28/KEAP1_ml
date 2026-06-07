# generates files with docking scores for each representative type, using the combined valid scores and the cluster representatives information
## generates files for each representative type

import pandas as pd

# Load files
reps = pd.read_csv("data/cluster_representatives_7_per_cluster.csv")
scores = pd.read_csv("data/combined_scores/combined_valid_scores.csv")

# Merge representative information with docking scores
merged = reps.merge(
    scores,
    left_on="identifier",
    right_on="Ligand",
    how="left"
)

# Create one file per representative type
for rep_type in merged["rep_type"].unique():

    subset = (
        merged[merged["rep_type"] == rep_type]
        [["cluster", "identifier", "Score"]]
        .sort_values("cluster")
    )

    subset.to_csv(f"data/representative_docking_scores/{rep_type}_scores.csv", index=False)

    print(
        f"Saved {rep_type}_scores.csv "
        f"({len(subset)} entries)"
    )