#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.ensemble import GradientBoostingRegressor
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
    "model__n_estimators": [200],
    "model__learning_rate": [0.05, 0.1],
    "model__max_depth": [3, 5]
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
        ("model", GradientBoostingRegressor(random_state=42))
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

    print(
        f"Fold {fold} | "
        f"RMSE: {rmse:.4f} | "
        f"R2: {r2:.4f}"
    )

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

# ==========================================================
# PREPARE BEST TEST SET
# ==========================================================

X_train_imputed = best_model.named_steps["imputer"].transform(
    best_X_train
)

X_test_imputed = best_model.named_steps["imputer"].transform(
    best_X_test
)

gbr_model = best_model.named_steps["model"]

predictions = best_model.predict(best_X_test)

# ==========================================================
# SHAP
# ==========================================================

print("\nComputing SHAP values...")

explainer = shap.TreeExplainer(gbr_model)
shap_values = explainer.shap_values(X_test_imputed)

base_values = explainer.expected_value

if np.ndim(base_values) > 0:
    base_values = np.asarray(base_values).flatten()[0]

# ==========================================================
# OUTPUT DIRECTORY
# ==========================================================

os.makedirs(args.output, exist_ok=True)

# ==========================================================
# 1. ANALYSIS METADATA
# ==========================================================

metadata = pd.DataFrame({
    "InputFile": [args.input],
    "Model": ["GradientBoostingRegressor"],
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

metadata.to_csv(
    f"{args.output}/analysis_metadata.csv",
    index=False
)

# ==========================================================
# 2. BASE VALUES
# ==========================================================

pd.DataFrame({
    "OriginalIndex": best_X_test.index,
    "BaseValue": np.repeat(base_values, len(shap_values))
}).to_csv(
    f"{args.output}/base_values.csv",
    index=False
)

# ==========================================================
# 3. BEST PARAMETERS
# ==========================================================

parameters = pd.DataFrame([
    {
        "Parameter": key,
        "Value": value
    }
    for key, value in best_parameters.items()
])

parameters.to_csv(
    f"{args.output}/best_parameters.csv",
    index=False
)

# ==========================================================
# 4. IMPUTED FEATURE VALUES
# ==========================================================

imputed_df = pd.DataFrame(
    X_test_imputed,
    columns=X.columns
)

imputed_df.insert(
    0,
    "OriginalIndex",
    best_X_test.index
)

imputed_df.to_csv(
    f"{args.output}/feature_values_imputed.csv",
    index=False
)

# ==========================================================
# 5. ORIGINAL FEATURE VALUES
# ==========================================================

original_df = best_X_test.copy()

original_df.insert(
    0,
    "OriginalIndex",
    best_X_test.index
)

original_df.to_csv(
    f"{args.output}/feature_values_original.csv",
    index=False
)

# ==========================================================
# 6. SAMPLE IDENTIFIERS
# ==========================================================

if "identifier" in df.columns:

    identifiers = df.loc[
        best_X_test.index,
        "identifier"
    ].reset_index()

    identifiers.columns = [
        "OriginalIndex",
        "Identifier"
    ]

else:

    identifiers = pd.DataFrame({
        "OriginalIndex": best_X_test.index
    })

identifiers.to_csv(
    f"{args.output}/sample_identifiers.csv",
    index=False
)

# ==========================================================
# 7. MODEL PERFORMANCE
# ==========================================================

metrics_df.to_csv(
    f"{args.output}/model_performance.csv",
    index=False
)

# ==========================================================
# 8. PREDICTIONS
# ==========================================================

predictions_df = pd.DataFrame({
    "OriginalIndex": best_X_test.index,
    "ActualScore": best_y_test,
    "PredictedScore": predictions
})

predictions_df["Residual"] = (
    predictions_df["ActualScore"]
    - predictions_df["PredictedScore"]
)

predictions_df.to_csv(
    f"{args.output}/predictions.csv",
    index=False
)

# ==========================================================
# 9. SHAP IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)
}).sort_values(
    "MeanAbsSHAP",
    ascending=False
)

importance_df.to_csv(
    f"{args.output}/shap_importance.csv",
    index=False
)

# ==========================================================
# 10. SHAP PLOT DATA
# ==========================================================

plot_data = []

for i in range(len(shap_values)):
    for j, feature in enumerate(X.columns):
        plot_data.append({
            "OriginalIndex": best_X_test.index[i],
            "Feature": feature,
            "FeatureValue": X_test_imputed[i, j],
            "SHAPValue": shap_values[i, j]
        })

plot_df = pd.DataFrame(plot_data)

plot_df.to_csv(
    f"{args.output}/shap_plot_data.csv",
    index=False
)

# ==========================================================
# 11. SHAP VALUES
# ==========================================================

shap_df = pd.DataFrame(
    shap_values,
    columns=X.columns
)

shap_df.insert(
    0,
    "OriginalIndex",
    best_X_test.index
)

shap_df.to_csv(
    f"{args.output}/shap_values.csv",
    index=False
)

# ==========================================================
# SUMMARY
# ==========================================================

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