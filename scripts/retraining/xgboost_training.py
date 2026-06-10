from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import KFold, ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from xgboost import XGBRegressor


def evaluate_nested_xgboost_with_tracking(
        csv_path,
        outer_splits=5,
        inner_splits=3,
        output_results_csv="data/xgb_results.csv",
        output_hp_csv="data/xgb_hyperparameter_log.csv"
):

    df = pd.read_csv(csv_path)

    if "Score" not in df.columns:
        raise ValueError("Dataset must contain 'Score' column")

    X = df.drop(columns=["Score", "identifier"], errors="ignore")
    y = df["Score"].values

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
        "n_estimators": [100, 300, 500],
        "learning_rate": [0.03, 0.1],
        "max_depth": [3, 6],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
        "reg_lambda": [1, 5],
        "min_child_weight": [1, 5]
    }

    param_list = list(ParameterGrid(param_grid))

    total_fits = (
        len(param_list)
        * inner_splits
        * outer_splits
    )

    global_bar = tqdm(
        total=total_fits,
        desc="Total Fits",
        position=0
    )

    fold_results = []
    hyperparam_log = []

    def mape(y_true, y_pred):
        return np.mean(
            np.abs((y_true - y_pred) / (y_true + 1e-8))
        ) * 100

    outer_bar = tqdm(
        enumerate(outer_cv.split(X), 1),
        total=outer_splits,
        desc="Outer CV",
        position=1
    )

    for fold_i, (train_idx, test_idx) in outer_bar:

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]

        y_train = y[train_idx]
        y_test = y[test_idx]

        best_rmse = np.inf
        best_params_fold = None

        inner_bar = tqdm(
            param_list,
            desc=f"Fold {fold_i} Search",
            leave=False,
            position=2
        )

        for params in inner_bar:

            inner_scores = []

            for tr_idx, val_idx in inner_cv.split(X_train):

                X_tr = X_train.iloc[tr_idx]
                X_val = X_train.iloc[val_idx]

                y_tr = y_train[tr_idx]
                y_val = y_train[val_idx]

                model = Pipeline([
                    (
                        "imputer",
                        KNNImputer(
                            n_neighbors=5,
                            weights="distance"
                        )
                    ),
                    (
                        "model",
                        XGBRegressor(
                            objective="reg:squarederror",
                            random_state=42,
                            n_jobs=1,
                            verbosity=0,
                            **params
                        )
                    )
                ])

                model.fit(X_tr, y_tr)

                preds = model.predict(X_val)

                rmse = np.sqrt(
                    mean_squared_error(y_val, preds)
                )

                inner_scores.append(rmse)

                hyperparam_log.append({
                    "outer_fold": fold_i,
                    **params,
                    "inner_rmse": rmse
                })

                global_bar.update(1)

            mean_rmse = np.mean(inner_scores)

            if mean_rmse < best_rmse:
                best_rmse = mean_rmse
                best_params_fold = params.copy()

            inner_bar.set_postfix(
                best_rmse=f"{best_rmse:.4f}"
            )

        best_model = Pipeline([
            (
                "imputer",
                KNNImputer(
                    n_neighbors=5,
                    weights="distance"
                )
            ),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                    verbosity=0,
                    **best_params_fold
                )
            )
        ])

        best_model.fit(X_train, y_train)

        preds = best_model.predict(X_test)

        fold_results.append({
            "fold": fold_i,
            "MAE": mean_absolute_error(y_test, preds),
            "MSE": mean_squared_error(y_test, preds),
            "RMSE": np.sqrt(mean_squared_error(y_test, preds)),
            "R2": r2_score(y_test, preds),
            "MAPE": mape(y_test, preds)
        })

    global_bar.close()

    results_df = pd.DataFrame(fold_results)
    hp_df = pd.DataFrame(hyperparam_log)

    results_df.to_csv(
        "data/final_xgboost/xgb_fold_results.csv",
        index=False
    )

    summary = {
        "dataset": csv_path,
        "MAE_mean": results_df["MAE"].mean(),
        "MAE_ci": 1.96 * results_df["MAE"].std() / np.sqrt(len(results_df)),
        "MSE_mean": results_df["MSE"].mean(),
        "MSE_ci": 1.96 * results_df["MSE"].std() / np.sqrt(len(results_df)),
        "RMSE_mean": results_df["RMSE"].mean(),
        "RMSE_ci": 1.96 * results_df["RMSE"].std() / np.sqrt(len(results_df)),
        "R2_mean": results_df["R2"].mean(),
        "R2_ci": 1.96 * results_df["R2"].std() / np.sqrt(len(results_df)),
        "MAPE_mean": results_df["MAPE"].mean(),
        "MAPE_ci": 1.96 * results_df["MAPE"].std() / np.sqrt(len(results_df))
    }

    pd.DataFrame([summary]).to_csv(
        output_results_csv,
        index=False
    )

    hp_df.to_csv(
        output_hp_csv,
        index=False
    )

    group_cols = [
        "n_estimators",
        "learning_rate",
        "max_depth",
        "subsample",
        "colsample_bytree",
        "reg_lambda",
        "min_child_weight"
    ]

    hp_ranking = (
        hp_df
        .groupby(group_cols)["inner_rmse"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("mean")
    )

    hp_ranking.to_csv(
        "data/final_xgboost/xgb_hyperparameter_ranking.csv",
        index=False
    )

    best_params = hp_ranking.iloc[0]

    best_params.to_frame().T.to_csv(
        "data/final_xgboost/best_xgb_hyperparameters.csv",
        index=False
    )

    final_model = Pipeline([
        (
            "imputer",
            KNNImputer(
                n_neighbors=5,
                weights="distance"
            )
        ),
        (
            "model",
            XGBRegressor(
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1,
                verbosity=0,
                n_estimators=int(best_params["n_estimators"]),
                learning_rate=float(best_params["learning_rate"]),
                max_depth=int(best_params["max_depth"]),
                subsample=float(best_params["subsample"]),
                colsample_bytree=float(best_params["colsample_bytree"]),
                reg_lambda=float(best_params["reg_lambda"]),
                min_child_weight=float(best_params["min_child_weight"])
            )
        )
    ])

    print("\\nTraining final model on full dataset...")

    final_model.fit(X, y)

    joblib.dump(
        final_model,
        "data/final_xgboost/best_xgboost_model.pkl"
    )

    preds = final_model.predict(X)

    pd.DataFrame({
        "Actual": y,
        "Predicted": preds
    }).to_csv(
        "data/final_xgboost/final_model_predictions.csv",
        index=False
    )

    feature_importance = pd.DataFrame({
        "feature": X.columns,
        "importance":
            final_model.named_steps["model"].feature_importances_
    })

    feature_importance = feature_importance.sort_values(
        "importance",
        ascending=False
    )

    feature_importance.to_csv(
        "data/final_xgboost/feature_importance.csv",
        index=False
    )

    print("\\nDONE")
    print("Saved:")
    print("- xgb_results.csv")
    print("- xgb_fold_results.csv")
    print("- xgb_hyperparameter_log.csv")
    print("- xgb_hyperparameter_ranking.csv")
    print("- best_xgb_hyperparameters.csv")
    print("- best_xgboost_model.pkl")
    print("- final_model_predictions.csv")
    print("- feature_importance.csv")

    return results_df, hp_df


if __name__ == "__main__":

    csv_path = "data/final_features_data/ratios_only.csv"

    results, hp_log = evaluate_nested_xgboost_with_tracking(
        csv_path=csv_path,
        outer_splits=5,
        inner_splits=3,
        output_results_csv="data/final_xgboost/xgb_results.csv",
        output_hp_csv="data/final_xgboost/xgb_hyperparameter_log.csv"
    )

    print("\\nFINAL SUMMARY")
    print(results)
    print("\\nHP LOG SHAPE:", hp_log.shape)
