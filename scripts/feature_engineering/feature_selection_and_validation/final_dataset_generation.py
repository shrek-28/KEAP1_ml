import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Extract selected feature columns from datasets")
parser.add_argument("-m", "--mapping", required=True, help="Mapping CSV containing dataset,k,features")
parser.add_argument("-i", "--input", required=True, help="Input folder containing feature CSVs")
parser.add_argument("-o", "--output", required=True, help="Output folder")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)
df = pd.read_csv(args.mapping)

for _, row in df.iterrows():
    dataset = row["dataset"]
    features = [f.strip() for f in str(row["features"]).split(",")]
    input_file = os.path.join(args.input, f"{dataset}.csv")

    if not os.path.exists(input_file):
        print(f"[SKIP] Missing file: {input_file}")
        continue

    data = pd.read_csv(input_file)
    mandatory_cols = [col for col in ["identifier", "Score"] if col in data.columns]
    selected_features = [col for col in features if col in data.columns]
    final_cols = mandatory_cols + [col for col in selected_features if col not in mandatory_cols]

    if len(final_cols) == 0:
        print(f"[SKIP] No matching columns for {dataset}")
        continue

    output_file = os.path.join(args.output, f"{dataset}.csv")
    data[final_cols].to_csv(output_file, index=False)
    print(f"[OK] {dataset}: {len(selected_features)} feature columns + {len(mandatory_cols)} mandatory columns saved")