import pandas as pd


def clean_name(name):
    """
    Remove .csv extension and standardize dataset names
    """
    return name.replace(".csv", "").strip()


def generate_topk_feature_lists(
    feature_table_path,
    best_k_table_path,
    output_csv
):

    feature_df = pd.read_csv(feature_table_path)
    k_df = pd.read_csv(best_k_table_path)

    # -----------------------------
    # Normalize dataset names
    # -----------------------------
    feature_df["dataset"] = feature_df["dataset"].apply(clean_name)
    k_df["dataset"] = k_df["dataset"].apply(clean_name)

    # -----------------------------
    # Build mappings
    # -----------------------------
    feature_map = dict(zip(feature_df["dataset"], feature_df["features"]))
    k_map = dict(zip(k_df["dataset"], k_df["k"]))

    results = []

    for dataset, k in k_map.items():

        if dataset not in feature_map:
            print(f"Skipping missing dataset: {dataset}")
            continue

        full_features = feature_map[dataset].split(",")

        top_k_features = full_features[:int(k)]

        results.append({
            "dataset": dataset,
            "k": int(k),
            "features": ",".join(top_k_features)
        })

        print(f"{dataset} | k={k}")

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)

    print(f"\nSaved -> {output_csv}")

    return out_df

generate_topk_feature_lists(
    feature_table_path="data/all_datasets_rmse.csv",
    best_k_table_path="data/knee_detection_results.csv",
    output_csv="data/final_feature_list.csv"
)