import argparse
import os
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(
    description="Select features based on their Spearman correlation with docking score."
)

parser.add_argument("-i", "--input", required=True, help="Input CSV file")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
parser.add_argument("-l", "--log", default="data/spearman_feature_selection_log.csv", help="Summary log CSV file")
parser.add_argument("-t", "--threshold", type=float, default=0.2, help="absolute spearman correlation thresold (default = 0.2)")

args = parser.parse_args()

# Load data
df = pd.read_csv(args.input)

# Numeric columns only
numeric_df = df.select_dtypes(include=[np.number])

# Ensure score exists
if "Score" not in numeric_df.columns:
    raise ValueError("Score column not found or is not numeric.")

score_series = numeric_df["Score"]

selected_features = []

for col in numeric_df.columns:

    if col == "Score":
        continue

    corr = numeric_df[col].corr(score_series, method="spearman")

    if not np.isnan(corr) and abs(corr) > args.threshold:
        selected_features.append(col)


# Build reduced dataframe
cols_to_keep = []

if "identifier" in df.columns:
    cols_to_keep.append("identifier")

cols_to_keep.extend(selected_features)
cols_to_keep.append("Score")

reduced_df = df[cols_to_keep]

# Create output directory
output_dir = os.path.dirname(args.output)

if output_dir:
    os.makedirs(output_dir, exist_ok=True)

# Save reduced dataset
reduced_df.to_csv(args.output, index=False)

# ---------------------------
# LOGGING
# ---------------------------

log_dir = os.path.dirname(args.log)

if log_dir:
    os.makedirs(log_dir, exist_ok=True)

log_entry = pd.DataFrame([{
    "input_file": args.input,
    "num_selected_features": len(selected_features),
    "selected_features": ";".join(selected_features)
}])

if os.path.exists(args.log):
    log_entry.to_csv(args.log, mode="a", header=False, index=False)
else:
    log_entry.to_csv(args.log, index=False)

print(f"Selected {len(selected_features)} features from {args.input}")
print(f"Output shape: {reduced_df.shape}")
print(f"Saved: {args.output}")