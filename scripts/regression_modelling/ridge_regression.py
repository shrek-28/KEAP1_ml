import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_nested_ridge_knn(folder_path, outer_splits=5, inner_splits=3, output_csv="ridge_nested_results.csv"):

    results = []

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    # ---------------- EXTENDED RIDGE GRID ----------------
    param_grid = {
    # Keep log coverage but reduce density at extremes
    "model__alpha": [
        1e-4, 1e-3, 1e-2,
        1e-1, 1,
        10, 100, 1000
    ],

    # Usually safe to test both, but keep it minimal
    "model__fit_intercept": [True],

    # Reduce solver redundancy (keep representative ones)
    "model__solver": ["auto", "cholesky", "lsqr"],

    # Tolerance: keep only meaningful separation
    "model__tol": [1e-3, 1e-2]
    }

    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    for file in os.listdir(folder_path):
        if not file.endswith(".csv"):
            continue

        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)

        if "Score" not in df.columns:
            continue

        X = df.drop(columns=["Score", "identifier"], errors="ignore")
        y = df["Score"].values

        fold_metrics = []

        for train_idx, test_idx in outer_cv.split(X):

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline([
                ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                ("scaler", StandardScaler()),
                ("model", Ridge())
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=2,
                error_score="raise"
            )

            grid.fit(X_train, y_train)

            best_model = grid.best_estimator_
            preds = best_model.predict(X_test)

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
            "MAPE_ci": 1.96 * fold_df["MAPE"].std() / np.sqrt(len(fold_df)),
        })

        print(f"[DONE] {file}")

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)

    return out

if __name__ == "__main__":

    folder_path = "data/final_features_data"
    output_csv = "data/regression_results/ridge_regression_results.csv"

    results = evaluate_nested_ridge_knn(
        folder_path=folder_path,
        outer_splits=5,
        inner_splits=3,
        output_csv=output_csv
    )

    print("\nFINAL SUMMARY")
    print(results.sort_values("RMSE_mean"))