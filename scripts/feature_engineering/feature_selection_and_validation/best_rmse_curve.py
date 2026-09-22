import pandas as pd
import numpy as np
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Detect optimal feature counts using the Kneedle method")
parser.add_argument("-i", "--input", required=True, help="Input folder containing RMSE CSVs")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
args = parser.parse_args()

def kneedle_k(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    gain = 1 - y_norm
    diff = gain - x_norm
    return np.argmax(diff)

folder = Path(args.input)
all_files = list(folder.glob("*.csv"))
results = []

for file in all_files:
    print(f"\nProcessing: {file.name}")
    df = pd.read_csv(file).sort_values("n_features").reset_index(drop=True)
    x = df["n_features"].values
    y = df["rmse"].values
    kneedle_idx = kneedle_k(x, y)
    results.append({"dataset": file.stem, "k": int(x[kneedle_idx]), "rmse": float(y[kneedle_idx])})
    print(f"  Kneedle -> k={x[kneedle_idx]} (RMSE={y[kneedle_idx]:.4f})")

results_df = pd.DataFrame(results)
output_file = Path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_file, index=False)

print(f"\nSaved results to: {output_file}")
print("\nFinal Results")
print(results_df)