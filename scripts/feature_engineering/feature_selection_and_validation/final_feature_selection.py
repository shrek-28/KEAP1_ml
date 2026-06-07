import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
import os

# calculate RMSE for regression evaluation
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# Simple logging function to append entries to a CSV log file
def log_event(log_file, entry):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    df = pd.DataFrame([entry])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)


def multi_dataset_feature_rmse_analysis(
    data_files,
    output_file,
    log_file="data/rf_progress_log.csv",
    feature_sizes=(5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500),
    n_splits=5,
    corr_threshold=0.9,
    random_state=42
):

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    all_results = []
    total_files = len(data_files)

    for i, file in enumerate(data_files, 1):

        dataset_start = time.time()

        log_event(log_file, {
            "level": "dataset_start",
            "dataset": file,
            "timestamp": time.time()
        })

        print(f"\n[{i}/{total_files}] Processing: {file}")

        df = pd.read_csv(file)

        X = df.drop(columns=["Score", "identifier"], errors="ignore")
        y = df["Score"]

        # ---------------- FEATURE RANKING ----------------
        corr_matrix = X.corr(method="spearman").abs()
        score_corr = X.join(y).corr(method="spearman")["Score"].drop("Score")

        ranked_features = score_corr.abs().sort_values(ascending=False).index.tolist()

        filtered_features = []
        for f in ranked_features:
            if all(corr_matrix.loc[f, g] <= corr_threshold for g in filtered_features):
                filtered_features.append(f)

        # ---------------- FEATURE LOOP ----------------
        for k in feature_sizes:

            if len(filtered_features) < k:
                continue

            top_features = filtered_features[:k]

            fold_scores = []

            for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):

                X_train = X.iloc[train_idx][top_features]
                X_test = X.iloc[test_idx][top_features]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]

                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                score = rmse(y_test, preds)
                fold_scores.append(score)

                log_event(log_file, {
                    "level": "fold_feature_subset",
                    "dataset": file,
                    "n_features": k,
                    "fold": fold,
                    "rmse": score,
                    "timestamp": time.time()
                })

            mean_rmse = np.mean(fold_scores)

            log_event(log_file, {
                "level": "feature_subset_complete",
                "dataset": file,
                "n_features": k,
                "rmse": mean_rmse,
                "timestamp": time.time()
            })

            all_results.append({
                "dataset": file,
                "n_features": k,
                "rmse": mean_rmse,
                "features": ",".join(top_features)
            })

            print(f"  k={k} RMSE={mean_rmse:.4f}")

        log_event(log_file, {
            "level": "dataset_complete",
            "dataset": file,
            "elapsed_sec": time.time() - dataset_start,
            "timestamp": time.time()
        })

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)

    print(f"\nSaved results to {output_file}")
    print(f"Progress log saved to {log_file}")

    return results_df

folder_path = "data/spearman_reduced_features"
output_file = "data/all_datasets_rmse.csv"

folder = Path(folder_path)

all_files = list(folder.glob("*.csv"))

results_list = []

for file in all_files:
    print(f"\nRunning: {file.name}")

    results = multi_dataset_feature_rmse_analysis(
        data_files=[str(file)],   # pass single file as list
        output_file="temp.csv"    # temporary (will overwrite)
    )

    results["dataset"] = file.name
    results_list.append(results)

final_df = pd.concat(results_list, ignore_index=True)
final_df.to_csv(output_file, index=False)

print(f"\nFinal combined results saved to {output_file}")