#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
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
    "poly__degree": [1, 2],
    "poly__interaction_only": [False, True],
    "model__fit_intercept": [True, False]
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

X_train_imputed = best_model.named_steps["imputer"].transform(best_X_train)
X_train_scaled = best_model.named_steps["scaler"].transform(X_train_imputed)
X_train_proc = best_model.named_steps["poly"].transform(X_train_scaled)

X_test_imputed = best_model.named_steps["imputer"].transform(best_X_test)
X_test_scaled = best_model.named_steps["scaler"].transform(X_test_imputed)
X_test_proc = best_model.named_steps["poly"].transform(X_test_scaled)

poly_features = best_model.named_steps["poly"].get_feature_names_out(
    input_features=best_X_train.columns
)

linear_model = best_model.named_steps["model"]

explainer = shap.LinearExplainer(linear_model, X_train_proc)
shap_values = explainer(X_test_proc)

if hasattr(shap_values, "values"):
    shap_array = shap_values.values
    base_values = shap_values.base_values
else:
    shap_array = shap_values
    base_values = explainer.expected_value

if np.ndim(base_values) > 0:
    base_values = np.asarray(base_values).flatten()[0]

os.makedirs(args.output, exist_ok=True)

metadata = pd.DataFrame({
    "InputFile": [args.input],
    "Model": ["PolynomialRegression"],
    "BestFold": [best_fold],
    "TestSamples": [len(best_X_test)],
    "Features": [X.shape[1]],
    "ExpandedFeatures": [len(poly_features)],
    "OuterCVSplits": [5],
    "InnerCVSplits": [3],
    "RandomState": [42],
    "BestRMSE": [best_rmse],
    "MeanRMSE": [metrics_df["RMSE"].mean()],
    "SHAPExplainer": ["LinearExplainer"]
})
metadata.to_csv(f"{args.output}/analysis_metadata.csv", index=False)

pd.DataFrame({
    "OriginalIndex": best_X_test.index,
    "BaseValue": np.repeat(base_values, len(shap_array))
}).to_csv(f"{args.output}/base_values.csv", index=False)

pd.DataFrame([
    {"Parameter": key, "Value": value}
    for key, value in best_parameters.items()
]).to_csv(f"{args.output}/best_parameters.csv", index=False)

imputed_df = pd.DataFrame(X_test_imputed, columns=X.columns)
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
    "Feature": poly_features,
    "MeanAbsSHAP": np.abs(shap_array).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

importance_df.to_csv(f"{args.output}/shap_importance.csv", index=False)

plot_data = []

for i in range(len(shap_array)):
    for j, feature in enumerate(poly_features):
        plot_data.append({
            "OriginalIndex": best_X_test.index[i],
            "Feature": feature,
            "FeatureValue": X_test_proc[i, j],
            "SHAPValue": shap_array[i, j]
        })

pd.DataFrame(plot_data).to_csv(
    f"{args.output}/shap_plot_data.csv",
    index=False
)

shap_df = pd.DataFrame(shap_array, columns=poly_features)
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