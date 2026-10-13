import argparse
import pandas as pd

parser = argparse.ArgumentParser(description="Calculate residuals for binding affinity predictions.")
parser.add_argument("--input", required=True, help="Input CSV file containing actual and predicted binding affinities.")
parser.add_argument("--output", required=True, help="Output summary CSV path.")
args = parser.parse_args()

df = pd.read_csv(args.input)

df["residual"] = round((df["Predicted"] - df["Actual"]), 2)

print(avg_residual := df["residual"].mean())
print(f"Average Residual: {avg_residual}")

df.to_csv(args.output, index=False)