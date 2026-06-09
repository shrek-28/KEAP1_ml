import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_nested_random_forest(
    folder_path,
    outer_splits=5,
    inner_splits=3,
    output_csv="random_forest_results.csv"
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
        "model__n_estimators": [100, 300, 500],
        "model__criterion": ["squared_error"],
        "model__max_depth": [10, 20, None],
        "model__min_samples_split": [2, 10],
        "model__min_samples_leaf": [1, 5],
        "model__max_features": ["sqrt", None]
    }

    def mape(y_true, y_pred):
        return np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8))
        ) * 100

    total_combinations = (
        len(param_grid["model__n_estimators"])
        * len(param_grid["model__criterion"])
        * len(param_grid["model__max_depth"])
        * len(param_grid["model__min_samples_split"])
        * len(param_grid["model__min_samples_leaf"])
        * len(param_grid["model__max_features"])
    )

    fits_per_outer_fold = total_combinations * inner_splits

    print(
        f"Grid contains {total_combinations} parameter combinations "
        f"({fits_per_outer_fold} model fits per outer fold)\n"
    )

    for file_idx, file in enumerate(files, start=1):

        dataset_start = time.time()

        print(f"\n{'='*80}")
        print(f"[{file_idx}/{len(files)}] Processing: {file}")
        print(f"{'='*80}")

        df = pd.read_csv(
            os.path.join(folder_path, file)
        )

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
                f"\nOuter Fold {fold_idx}/{outer_splits}"
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
                    RandomForestRegressor(
                        random_state=42,
                        n_jobs=-1
                    )
                )
            ])

            grid = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                cv=inner_cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
                error_score="raise",
                verbose=3
            )

            fold_start = time.time()

            print(
                f"Starting inner CV "
                f"({fits_per_outer_fold} fits)..."
            )

            grid.fit(X_train, y_train)

            inner_cv_time = time.time() - fold_start

            preds = grid.best_estimator_.predict(
                X_test
            )

            rmse = np.sqrt(
                mean_squared_error(y_test, preds)
            )

            print(
                f"Completed in "
                f"{inner_cv_time:.2f} sec "
                f"({inner_cv_time/60:.2f} min)"
            )

            print(
                f"Best RMSE: "
                f"{-grid.best_score_:.4f}"
            )

            print(
                f"Best Params: "
                f"{grid.best_params_}"
            )

            fold_metrics.append({
                "MAE": mean_absolute_error(
                    y_test,
                    preds
                ),
                "MSE": mean_squared_error(
                    y_test,
                    preds
                ),
                "RMSE": rmse,
                "R2": r2_score(
                    y_test,
                    preds
                ),
                "MAPE": mape(
                    y_test,
                    preds
                ),
                "InnerCV_Time_sec": inner_cv_time
            })

        fold_df = pd.DataFrame(
            fold_metrics
        )

        n = len(fold_df)

        dataset_time = (
            time.time() - dataset_start
        )

        results.append({
            "dataset": file,

            "MAE_mean":
                fold_df["MAE"].mean(),
            "MAE_ci":
                1.96 * fold_df["MAE"].std(ddof=1)
                / np.sqrt(n),

            "MSE_mean":
                fold_df["MSE"].mean(),
            "MSE_ci":
                1.96 * fold_df["MSE"].std(ddof=1)
                / np.sqrt(n),

            "RMSE_mean":
                fold_df["RMSE"].mean(),
            "RMSE_ci":
                1.96 * fold_df["RMSE"].std(ddof=1)
                / np.sqrt(n),

            "R2_mean":
                fold_df["R2"].mean(),
            "R2_ci":
                1.96 * fold_df["R2"].std(ddof=1)
                / np.sqrt(n),

            "MAPE_mean":
                fold_df["MAPE"].mean(),
            "MAPE_ci":
                1.96 * fold_df["MAPE"].std(ddof=1)
                / np.sqrt(n),

            "InnerCV_Time_mean_sec":
                fold_df["InnerCV_Time_sec"].mean(),

            "InnerCV_Time_ci_sec":
                1.96
                * fold_df["InnerCV_Time_sec"].std(ddof=1)
                / np.sqrt(n),

            "Dataset_Total_Time_sec":
                dataset_time
        })

        print(
            f"\n[DONE] {file}"
        )

        print(
            f"Dataset runtime: "
            f"{dataset_time:.2f} sec "
            f"({dataset_time/60:.2f} min)"
        )

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        output_csv,
        index=False
    )

    print(
        f"\nSaved results to: "
        f"{output_csv}"
    )

    return results_df


# Example usage

folder_path = "data/final_features_data"

results = evaluate_nested_random_forest(
    folder_path=folder_path,
    outer_splits=5,
    inner_splits=3,
    output_csv="data/regression_results/random_forest_results.csv"
)

print(results)