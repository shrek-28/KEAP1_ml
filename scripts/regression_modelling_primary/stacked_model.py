import os
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


def evaluate_nested_stack(folder_path,
                          outer_splits=5,
                          inner_splits=3,
                          output_csv="stack_results.csv"):

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    results = []

    outer_cv = KFold(n_splits=outer_splits, shuffle=True, random_state=42)
    inner_cv = KFold(n_splits=inner_splits, shuffle=True, random_state=42)

    param_grid = {
        "model__final_estimator__alpha": [0.1, 1.0]
    }

    n_combinations = len(param_grid["model__final_estimator__alpha"])

    print(f"Datasets: {len(files)}")
    print(f"Parameter combinations: {n_combinations}")
    print(f"Fits per outer fold: {n_combinations * inner_splits}")
    print(f"Fits per dataset: {n_combinations * inner_splits * outer_splits}")

    def mape(y_true, y_pred):
        return np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8))
        ) * 100

    for file in files:

        print(f"\nProcessing: {file}")

        df = pd.read_csv(os.path.join(folder_path, file))

        if "Score" not in df.columns:
            continue

        X = df.drop(columns=["Score", "identifier"], errors="ignore")
        y = df["Score"].values

        metrics = []

        for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):

            print(f"  Fold {fold}/{outer_splits}")

            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            stack = StackingRegressor(
                estimators=[
                    (
                        "svr",
                        SVR(
                            C=10,
                            epsilon=0.1,
                            kernel="rbf"
                        )
                    ),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                            n_jobs=-1
                        )
                    ),
                    (
                        "xgb",
                        XGBRegressor(
                            n_estimators=100,
                            max_depth=3,
                            learning_rate=0.1,
                            objective="reg:squarederror",
                            random_state=42,
                            verbosity=0,
                            n_jobs=1
                        )
                    )
                ],
                final_estimator=Ridge(),
                cv=3,
                n_jobs=-1
            )

            pipeline = Pipeline([
                (
                    "imputer",
                    KNNImputer(
                        n_neighbors=5,
                        weights="distance"
                    )
                ),
                (
                    "model",
                    stack
                )
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

            preds = grid.best_estimator_.predict(X_test)

            metrics.append({
                "MAE": mean_absolute_error(y_test, preds),
                "MSE": mean_squared_error(y_test, preds),
                "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
                "R2": r2_score(y_test, preds),
                "MAPE": mape(y_test, preds)
            })

        fold_df = pd.DataFrame(metrics)

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

    results = pd.DataFrame(results)

    results.to_csv(output_csv, index=False)

    print("\nFINAL SUMMARY")
    print(results.sort_values("RMSE_mean"))

    return results


if __name__ == "__main__":

    evaluate_nested_stack(
        folder_path="data/final_features_data",
        outer_splits=5,
        inner_splits=3,
        output_csv="data/regression_results/stack_results.csv")