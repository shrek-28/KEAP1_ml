import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import time
import os
import argparse

parser = argparse.ArgumentParser(description="Evaluate Random Forest RMSE across feature subset sizes for multiple datasets")
parser.add_argument("-i", "--input", required=True, help="Input folder containing CSV datasets")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
parser.add_argument("-l", "--logs", default="data/rf_progress_log.csv", help="Progress log CSV file")
args = parser.parse_args()

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def log_event(log_file, entry):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    df = pd.DataFrame([entry])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

feature_sizes = (5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 200, 300, 400, 500)
model = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

input_folder = Path(args.input)
output_file = Path(args.output)
output_file.parent.mkdir(parents=True, exist_ok=True)
all_files = list(input_folder.glob("*.csv"))

all_results = []

for i, file in enumerate(all_files, 1):
    dataset_start = time.time()
    print(f"\n[{i}/{len(all_files)}] Processing: {file.name}")

    df = pd.read_csv(file)
    X = df.drop(columns=["Score", "identifier"], errors="ignore")
    y = df["Score"]

    corr_matrix = X.corr(method="spearman").abs()
    score_corr = X.join(y).corr(method="spearman")["Score"].drop("Score")
    ranked_features = score_corr.abs().sort_values(ascending=False).index.tolist()

    filtered_features = []
    for f in ranked_features:
        if all(corr_matrix.loc[f, g] <= 0.9 for g in filtered_features):
            filtered_features.append(f)

    for k in feature_sizes:
        if len(filtered_features) < k:
            continue

        top_features = filtered_features[:k]
        fold_scores = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X), 1):
            X_train, X_test = X.iloc[train_idx][top_features], X.iloc[test_idx][top_features]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            score = rmse(y_test, model.predict(X_test))
            fold_scores.append(score)

            log_event(args.logs, {"level": "fold_feature_subset", "dataset": str(file), "n_features": k, "fold": fold, "rmse": score, "timestamp": time.time()})

        mean_rmse = np.mean(fold_scores)
        log_event(args.logs, {"level": "feature_subset_complete", "dataset": str(file), "n_features": k, "rmse": mean_rmse, "timestamp": time.time()})
        all_results.append({"dataset": file.name, "n_features": k, "rmse": mean_rmse, "features": ",".join(top_features)})
        print(f"  k={k} RMSE={mean_rmse:.4f}")

    log_event(args.logs, {"level": "dataset_complete", "dataset": str(file), "elapsed_sec": time.time() - dataset_start, "timestamp": time.time()})

pd.DataFrame(all_results).to_csv(output_file, index=False)
print(f"\nSaved results to {output_file}")
print(f"Progress log saved to {args.logs}")