#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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
    "model__C": [1, 100],
    "model__epsilon": [0.1],
    "model__kernel": ["rbf"]
}

fold_metrics = []

best_rmse = np.inf
best_model = None
best_X_train = None
best_X_test = None
best_y_test = None
best_fold = None
best_parameters = None

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
        ("model", SVR())
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

print("\nComputing SHAP values (KernelExplainer - reduced sampling)...")

X_train_proc = best_model.named_steps["imputer"].transform(best_X_train)
X_train_proc = best_model.named_steps["scaler"].transform(X_train_proc)

X_test_proc = best_model.named_steps["imputer"].transform(best_X_test)
X_test_proc = best_model.named_steps["scaler"].transform(X_test_proc)

background = shap.sample(X_train_proc, 30, random_state=42)

def model_predict(X):
    return best_model.named_steps["model"].predict(X)

explainer = shap.KernelExplainer(model_predict, background)

X_test_small = X_test_proc[:80]
shap_values = explainer.shap_values(X_test_small)

if isinstance(shap_values, list):
    shap_values = shap_values[0]

base_values = explainer.expected_value

if np.ndim(base_values) > 0:
    base_values = np.asarray(base_values).flatten()[0]

test_indices = best_X_test.index[:X_test_small.shape[0]]
y_test_small = best_y_test[:X_test_small.shape[0]]

os.makedirs(args.output, exist_ok=True)

metadata = pd.DataFrame({
    "InputFile": [args.input],
    "Model": ["SVR"],
    "BestFold": [best_fold],
    "TestSamples": [len(X_test_small)],
    "TotalBestFoldTestSamples": [len(best_X_test)],
    "Features": [X.shape[1]],
    "OuterCVSplits": [5],
    "InnerCVSplits": [3],
    "RandomState": [42],
    "BestRMSE": [best_rmse],
    "MeanRMSE": [metrics_df["RMSE"].mean()],
    "SHAPExplainer": ["KernelExplainer"]
})
metadata.to_csv(f"{args.output}/analysis_metadata.csv", index=False)

pd.DataFrame({
    "OriginalIndex": test_indices,
    "BaseValue": np.repeat(base_values, len(shap_values))
}).to_csv(f"{args.output}/base_values.csv", index=False)

pd.DataFrame([
    {"Parameter": key, "Value": value}
    for key, value in best_parameters.items()
]).to_csv(f"{args.output}/best_parameters.csv", index=False)

imputed_df = pd.DataFrame(X_test_small, columns=X.columns)
imputed_df.insert(0, "OriginalIndex", test_indices)
imputed_df.to_csv(f"{args.output}/feature_values_imputed.csv", index=False)

original_df = best_X_test.iloc[:X_test_small.shape[0]].copy()
original_df.insert(0, "OriginalIndex", test_indices)
original_df.to_csv(f"{args.output}/feature_values_original.csv", index=False)

if "identifier" in df.columns:
    identifiers = df.loc[test_indices, "identifier"].reset_index()
    identifiers.columns = ["OriginalIndex", "Identifier"]
else:
    identifiers = pd.DataFrame({"OriginalIndex": test_indices})

identifiers.to_csv(f"{args.output}/sample_identifiers.csv", index=False)

metrics_df.to_csv(f"{args.output}/model_performance.csv", index=False)

predictions = best_model.predict(best_X_test.iloc[:X_test_small.shape[0]])

predictions_df = pd.DataFrame({
    "OriginalIndex": test_indices,
    "ActualScore": y_test_small,
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
            "OriginalIndex": test_indices[i],
            "Feature": feature,
            "FeatureValue": X_test_small[i, j],
            "SHAPValue": shap_values[i, j]
        })

pd.DataFrame(plot_data).to_csv(
    f"{args.output}/shap_plot_data.csv",
    index=False
)

shap_df = pd.DataFrame(shap_values, columns=X.columns)
shap_df.insert(0, "OriginalIndex", test_indices)
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