#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from lightgbm import LGBMRegressor

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", required=True)
parser.add_argument("-o", "--output", required=True)
args = parser.parse_args()

df = pd.read_csv(args.input)

if "Score" not in df.columns:
    raise ValueError("Missing target column: Score")

X = df.drop(columns=["Score", "identifier"], errors="ignore")
y = df["Score"].values

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {
    "model__n_estimators": [200],
    "model__learning_rate": [0.05, 0.1],
    "model__num_leaves": [31]
}

fold_metrics = []
best_rmse = np.inf
best_model = None
best_X_train = None
best_X_test = None
best_y_test = None
best_fold = None
best_parameters = None

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("model", LGBMRegressor(random_state=42, verbosity=-1))
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

    best_estimator = grid.best_estimator_
    preds = best_estimator.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    fold_metrics.append({
        "Fold": fold,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = best_estimator
        best_X_train = X_train.copy()
        best_X_test = X_test.copy()
        best_y_test = y_test.copy()
        best_fold = fold
        best_parameters = grid.best_params_

metrics_df = pd.DataFrame(fold_metrics)

print("\nFINAL METRICS")
print(metrics_df)
print("\nMean RMSE:", metrics_df["RMSE"].mean())

print("\nComputing SHAP values...")

X_train_proc = best_model.named_steps["imputer"].transform(best_X_train)
X_test_proc = best_model.named_steps["imputer"].transform(best_X_test)

lgbm_model = best_model.named_steps["model"]

explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(X_test_proc)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

base_values = explainer.expected_value

if np.ndim(base_values) > 0:
    base_values = np.asarray(base_values).flatten()[0]

os.makedirs(args.output, exist_ok=True)

metadata = pd.DataFrame({
    "InputFile": [args.input],
    "Model": ["LGBMRegressor"],
    "BestFold": [best_fold],
    "TestSamples": [len(best_X_test)],
    "Features": [X.shape[1]],
    "OuterCVSplits": [5],
    "InnerCVSplits": [3],
    "RandomState": [42],
    "BestRMSE": [best_rmse],
    "MeanRMSE": [metrics_df["RMSE"].mean()],
    "SHAPExplainer": ["TreeExplainer"]
})
metadata.to_csv(f"{args.output}/analysis_metadata.csv", index=False)

pd.DataFrame({
    "OriginalIndex": best_X_test.index,
    "BaseValue": np.repeat(base_values, len(shap_values))
}).to_csv(f"{args.output}/base_values.csv", index=False)

pd.DataFrame([
    {"Parameter": key, "Value": value}
    for key, value in best_parameters.items()
]).to_csv(f"{args.output}/best_parameters.csv", index=False)

imputed_df = pd.DataFrame(X_test_proc, columns=X.columns)
imputed_df.insert(0, "OriginalIndex", best_X_test.index)
imputed_df.to_csv(f"{args.output}/feature_values_imputed.csv", index=False)

original_df = best_X_test.copy()
original_df.insert(0, "OriginalIndex", best_X_test.index)
original_df.to_csv(f"{args.output}/feature_values_original.csv", index=False)

if "identifier" in df.columns:
    identifiers = df.loc[best_X_test.index, "identifier"].reset_index()
    identifiers.columns = ["OriginalIndex", "Identifier"]
else:
    identifiers = pd.DataFrame({"OriginalIndex": best_X_test.index})

identifiers.to_csv(f"{args.output}/sample_identifiers.csv", index=False)

metrics_df.to_csv(f"{args.output}/model_performance.csv", index=False)

predictions = best_model.predict(best_X_test)

predictions_df = pd.DataFrame({
    "OriginalIndex": best_X_test.index,
    "ActualScore": best_y_test,
    "PredictedScore": predictions
})
predictions_df["Residual"] = predictions_df["ActualScore"] - predictions_df["PredictedScore"]
predictions_df.to_csv(f"{args.output}/predictions.csv", index=False)

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

importance_df.to_csv(f"{args.output}/shap_importance.csv", index=False)

plot_data = []

for i in range(len(shap_values)):
    for j, feature in enumerate(X.columns):
        plot_data.append({
            "OriginalIndex": best_X_test.index[i],
            "Feature": feature,
            "FeatureValue": X_test_proc[i, j],
            "SHAPValue": shap_values[i, j]
        })

pd.DataFrame(plot_data).to_csv(
    f"{args.output}/shap_plot_data.csv",
    index=False
)

shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df.insert(0, "OriginalIndex", best_X_test.index)
shap_df.to_csv(f"{args.output}/shap_values.csv", index=False)

print("\nTop 20 SHAP Features:")
print(importance_df.head(20))

print("\nSaved outputs:")

for file in [
    "analysis_metadata.csv",
    "base_values.csv",
    "best_parameters.csv",
    "feature_values_imputed.csv",
    "feature_values_original.csv",
    "sample_identifiers.csv",
    "model_performance.csv",
    "predictions.csv",
    "shap_importance.csv",
    "shap_plot_data.csv",
    "shap_values.csv"
]:
    print(f" - {args.output}/{file}")