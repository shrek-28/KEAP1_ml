#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from xgboost import XGBRegressor

parser = argparse.ArgumentParser(description="Nested cross-validation stacking regression")
parser.add_argument("-i", "--input", required=True, help="Input folder containing CSV files")
parser.add_argument("-o", "--output", required=True, help="Output CSV file")
args = parser.parse_args()

def mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "model__final_estimator__alpha": [0.1, 1.0]
}

n_combinations = len(param_grid["model__final_estimator__alpha"])
fits_per_outer_fold = n_combinations * 3
fits_per_dataset = fits_per_outer_fold * 5

files = [f for f in os.listdir(args.input) if f.endswith(".csv")]
results = []

print(f"\nTotal datasets found: {len(files)}")
print(f"Parameter combinations: {n_combinations}")
print(f"Models trained per outer fold: {fits_per_outer_fold}")
print(f"Models trained per dataset: {fits_per_dataset}\n")

for i, file in enumerate(files, 1):
    print(f"\n[{i}/{len(files)}] Processing dataset: {file}")

    df = pd.read_csv(os.path.join(args.input, file))

    if "Score" not in df.columns:
        print("  -> Skipped (no Score column)")
        continue

    X = df.drop(columns=["Score", "identifier"], errors="ignore")
    y = df["Score"].values
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):
        print(f"\n  -> Outer fold {fold_i}/5")
        print(f"     - GridSearchCV ({n_combinations} combinations × 3 folds = {fits_per_outer_fold} fits)")

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        stack = StackingRegressor(
            estimators=[
                ("svr", SVR(C=10, epsilon=0.1, kernel="rbf")),
                ("rf", RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1
                )),
                ("xgb", XGBRegressor(
                    n_estimators=100,
                    max_depth=3,
                    learning_rate=0.1,
                    objective="reg:squarederror",
                    random_state=42,
                    verbosity=0,
                    n_jobs=1
                ))
            ],
            final_estimator=Ridge(),
            cv=3,
            n_jobs=-1
        )

        pipeline = Pipeline([
            ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
            ("model", stack)
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=inner_cv,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            error_score="raise"
        )

        grid.fit(X_train, y_train)

        print(f"     - Best RMSE: {-grid.best_score_:.4f}")
        print(f"     - Best Params: {grid.best_params_}")

        preds = grid.best_estimator_.predict(X_test)

        fold_metrics.append({
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "MAPE": mape(y_test, preds)
        })

    fold_df = pd.DataFrame(fold_metrics)
    n = len(fold_df)

    results.append({
        "dataset": file,
        "MAE_mean": fold_df["MAE"].mean(),
        "MAE_ci": 1.96 * fold_df["MAE"].std() / np.sqrt(n),
        "MSE_mean": fold_df["MSE"].mean(),
        "MSE_ci": 1.96 * fold_df["MSE"].std() / np.sqrt(n),
        "RMSE_mean": fold_df["RMSE"].mean(),
        "RMSE_ci": 1.96 * fold_df["RMSE"].std() / np.sqrt(n),
        "R2_mean": fold_df["R2"].mean(),
        "R2_ci": 1.96 * fold_df["R2"].std() / np.sqrt(n),
        "MAPE_mean": fold_df["MAPE"].mean(),
        "MAPE_ci": 1.96 * fold_df["MAPE"].std() / np.sqrt(n),
        "total_model_fits": fits_per_dataset
    })

    print(f"\n  -> DONE: {file}")
    print(f"  -> Total inner-CV models trained: {fits_per_dataset}")

out = pd.DataFrame(results)
os.makedirs(os.path.dirname(args.output), exist_ok=True)
out.to_csv(args.output, index=False)

print("\nAll datasets processed. Results saved.")
print("\nFINAL SUMMARY")
print(out.sort_values("RMSE_mean"))