import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_nested_poly(folder_path, outer_splits=5, inner_splits=3, output_csv="poly_nested_results.csv"):

    results = []

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    total_files = len(files)

    print(f"\nTotal datasets: {total_files}\n")

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    # ---------------- POLYNOMIAL GRID ----------------
    param_grid = {
        "poly__degree": [1, 2],
        "poly__interaction_only": [False, True],
        "model__fit_intercept": [True, False]
    }

    def mape(y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    for i, file in enumerate(files, 1):

        print(f"\n[{i}/{total_files}] Dataset: {file}")

        path = os.path.join(folder_path, file)
        df = pd.read_csv(path)

        if "Score" not in df.columns:
            print("  -> skipped (no Score)")
            continue

        X = df.drop(columns=["Score", "identifier"], errors="ignore")
        y = df["Score"].values

        fold_metrics = []

        for fold_i, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):

            print(f"  -> Outer fold {fold_i}/{outer_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pipeline = Pipeline([
                ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
                ("scaler", StandardScaler()),
                ("poly", PolynomialFeatures(include_bias=False)),
                ("model", LinearRegression())
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=4,
                error_score="raise"
            )

            print("     - Running GridSearchCV...")

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

        print(f"  -> DONE: {file}")

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)

    print("\nAll datasets completed.")
    return out

if __name__ == "__main__":

    folder_path = "data/final_features_data"
    output_csv = "data/regression_results/poly_results.csv"

    results = evaluate_nested_poly(
        folder_path=folder_path,
        outer_splits=5,
        inner_splits=3,
        output_csv=output_csv
    )

    print("\nFINAL SUMMARY")
    print(results.sort_values("RMSE_mean"))