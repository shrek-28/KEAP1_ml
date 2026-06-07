import os
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_nested_decision_tree(
    folder_path,
    outer_splits=5,
    inner_splits=3,
    output_csv="decision_tree_results.csv"
):

    files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    results = []

    print(f"\nFound {len(files)} datasets\n")

    outer_cv = KFold(
        n_splits=outer_splits,
        shuffle=True,
        random_state=42
    )

    inner_cv = KFold(
        n_splits=inner_splits,
        shuffle=True,
        random_state=42
    )

    param_grid = {
        "model__criterion": ["squared_error"],

        "model__max_depth": [5, 10, 20, None],

        "model__min_samples_split": [2, 10],

        "model__min_samples_leaf": [1, 5],

        "model__ccp_alpha": [0, 1e-3]
    }

    def mape(y_true, y_pred):
        return np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8))
        ) * 100

    for file_idx, file in enumerate(files, start=1):

        print(f"\n[{file_idx}/{len(files)}] {file}")

        df = pd.read_csv(os.path.join(folder_path, file))

        if "Score" not in df.columns:
            print("Skipped (no Score column)")
            continue

        X = df.drop(
            columns=["Score", "identifier"],
            errors="ignore"
        )

        y = df["Score"].values

        fold_metrics = []

        for fold_idx, (train_idx, test_idx) in enumerate(
            outer_cv.split(X),
            start=1
        ):

            print(
                f"  Fold {fold_idx}/{outer_splits}",
                flush=True
            )

            X_train, X_test = (
                X.iloc[train_idx],
                X.iloc[test_idx]
            )

            y_train, y_test = (
                y[train_idx],
                y[test_idx]
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
                    DecisionTreeRegressor(
                        random_state=42
                    )
                )
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                error_score="raise"
            )

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

        n = len(fold_df)

        results.append({
            "dataset": file,

            "MAE_mean": fold_df["MAE"].mean(),
            "MAE_ci": 1.96 * fold_df["MAE"].std(ddof=1) / np.sqrt(n),

            "MSE_mean": fold_df["MSE"].mean(),
            "MSE_ci": 1.96 * fold_df["MSE"].std(ddof=1) / np.sqrt(n),

            "RMSE_mean": fold_df["RMSE"].mean(),
            "RMSE_ci": 1.96 * fold_df["RMSE"].std(ddof=1) / np.sqrt(n),

            "R2_mean": fold_df["R2"].mean(),
            "R2_ci": 1.96 * fold_df["R2"].std(ddof=1) / np.sqrt(n),

            "MAPE_mean": fold_df["MAPE"].mean(),
            "MAPE_ci": 1.96 * fold_df["MAPE"].std(ddof=1) / np.sqrt(n)
        })

        print(f"[DONE] {file}")

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"\nSaved results to: {output_csv}"
    )

    return results_df


if __name__ == "__main__":

    folder_path = "data/final_features_data"
    output_csv = "data/regression_results/decision_tree_results.csv"

    results = evaluate_nested_decision_tree(
        folder_path=folder_path,
        outer_splits=5,
        inner_splits=3,
        output_csv=output_csv
    )

    print("\nFINAL SUMMARY")
    print(results.sort_values("RMSE_mean"))