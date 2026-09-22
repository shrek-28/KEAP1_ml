#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

parser = argparse.ArgumentParser(description="Nested cross-validation polynomial regression")
parser.add_argument("-i", "--input", required=True, help="Input folder containing CSV files")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
args = parser.parse_args()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

files = [f for f in os.listdir(args.input) if f.endswith(".csv")]
total_files = len(files)
print(f"\nTotal datasets: {total_files}\n")

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "poly__degree": [1, 2],
    "poly__interaction_only": [False, True],
    "model__fit_intercept": [True, False]
}

results = []

for i, file in enumerate(files, 1):
    print(f"\n[{i}/{total_files}] Dataset: {file}")
    df = pd.read_csv(os.path.join(args.input, file))

    if "Score" not in df.columns:
        print("  -> skipped (no Score)")
        continue

    X = df.drop(columns=["Score", "identifier"], errors="ignore")
    y = df["Score"].values
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        print(f"  -> Outer fold {fold_i}/5")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(include_bias=False)),
            ("model", LinearRegression())
        ])

        grid = GridSearchCV(pipeline, param_grid, cv=inner_cv, scoring="neg_root_mean_squared_error", n_jobs=4, error_score="raise")
        print("     - Running GridSearchCV...")
        grid.fit(X_train, y_train)

        preds = grid.best_estimator_.predict(X_test)

        fold_metrics.append({
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "MAPE": mape(y_test, preds)
        })

    fold_df = pd.DataFrame(fold_metrics)
    results.append({
        "dataset": file,
        "MAE_mean": fold_df["MAE"].mean(),
        "MAE_ci": 1.96 * fold_df["MAE"].std() / np.sqrt(len(fold_df)),
        "MSE_mean": fold_df["MSE"].mean(),
        "MSE_ci": 1.96 * fold_df["MSE"].std() / np.sqrt(len(fold_df)),
        "RMSE_mean": fold_df["RMSE"].mean(),
        "RMSE_ci": 1.96 * fold_df["RMSE"].std() / np.sqrt(len(fold_df)),
        "R2_mean": fold_df["R2"].mean(),
        "R2_ci": 1.96 * fold_df["R2"].std() / np.sqrt(len(fold_df)),
        "MAPE_mean": fold_df["MAPE"].mean(),
        "MAPE_ci": 1.96 * fold_df["MAPE"].std() / np.sqrt(len(fold_df))
    })
    print(f"  -> DONE: {file}")

out = pd.DataFrame(results)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
out.to_csv(args.output, index=False)

print("\nAll datasets completed.")
print("\nFINAL SUMMARY")
print(out.sort_values("RMSE_mean"))