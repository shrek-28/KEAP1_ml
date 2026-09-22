import pandas as pd
import argparse 

parser = argparse.ArgumentParser(
    description="combine the descriptors and docking scores CSV files using the COCONUT ID as key column"
)

parser.add_argument("-d", "--descriptors", required=True, help="Path to Descriptor matrix CSV file")
parser.add_argument("-s", "--scores", required=True, help="Path to Docking-score data file (no outliers)")
parser.add_argument("-o1", "--output_1", required=True, help="path to output CSV file")
parser.add_argument("-o2", "--output_2", required=True, help="path to output CSV file")

args = parser.parse_args()

# descriptors = pd.read_csv("data/final_complete_descriptor_matrix.csv")
# scores = pd.read_csv("data/combined_scores/docking_score_data_no_outliers.csv")
descriptors = pd.read_csv(args.descriptors)
scores = pd.read_csv(args.scores)

combined = descriptors.merge(
    scores,
    left_on="identifier",
    right_on="Ligand",
    how="inner"
)

combined.drop(columns=["Ligand", "Unnamed: 0", "AromaticRingCount", "ConformerCount"], inplace=True)
# combined.to_csv("data/combined_scores/with_descriptors.csv", index=False)
# combined.to_csv("data/engineered_features/with_descriptors.csv", index=False)
combined.to_csv(args.output_1, index=False)
combined.to_csv(args.output_2, index=False)