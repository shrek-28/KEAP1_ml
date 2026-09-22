import pandas as pd
import argparse

parser = argparse.ArgumentParser(description="Generate top-k feature lists")
parser.add_argument("-f", "--features", required=True, help="Feature table CSV")
parser.add_argument("-k", "--best-k", required=True, help="Best-k table CSV")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
args = parser.parse_args()

def clean_name(name):
    return name.replace(".csv", "").strip()

feature_df = pd.read_csv(args.features)
k_df = pd.read_csv(args.best_k)

feature_df["dataset"] = feature_df["dataset"].apply(clean_name)
k_df["dataset"] = k_df["dataset"].apply(clean_name)

feature_map = dict(zip(feature_df["dataset"], feature_df["features"]))
k_map = dict(zip(k_df["dataset"], k_df["k"]))

results = []

for dataset, k in k_map.items():
    if dataset not in feature_map:
        print(f"Skipping missing dataset: {dataset}")
        continue
    full_features = feature_map[dataset].split(",")
    top_k_features = full_features[:int(k)]
    results.append({"dataset": dataset, "k": int(k), "features": ",".join(top_k_features)})
    print(f"{dataset} | k={k}")

out_df = pd.DataFrame(results)
out_df.to_csv(args.output, index=False)
print(f"\nSaved -> {args.output}")
print(out_df)