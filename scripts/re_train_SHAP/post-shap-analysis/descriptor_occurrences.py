import argparse
import pandas as pd

parser = argparse.ArgumentParser(
    description="Count descriptor occurrences among the top 20 SHAP features."
)

parser.add_argument("-i", "--input", required=True, help="Input CSV path")
parser.add_argument("-o", "--output", required=True, help="Output CSV path")

args = parser.parse_args()

# Read input CSV
df = pd.read_csv(args.input)

# Take top 20 features by MeanAbsSHAP
top20 = df.sort_values("MeanAbsSHAP", ascending=False).head(20)

# Split ratio features into both descriptors and count occurrences
counts_df = (
    top20["Feature"]
    .str.split("_div_")
    .explode()
    .value_counts()
    .rename_axis("Descriptor")
    .reset_index(name="Occurrence")
)

# Save output
counts_df.to_csv(args.output, index=False)

print(f"Top 20 features processed.")
print(f"Descriptor counts saved to: {args.output}")