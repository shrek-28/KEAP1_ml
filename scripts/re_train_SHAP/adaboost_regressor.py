#!/usr/bin/env python3

import argparse
import os
import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# Argument Parser
# ==========================================

parser = argparse.ArgumentParser(description="AdaBoost SHAP analysis")
parser.add_argument("-i", "--input", required=True, help="Input CSV file")
parser.add_argument("-o", "--output", required=True, help="Output directory")
args = parser.parse_args()

os.makedirs(args.output, exist_ok=True)

# ==========================================
# Load dataset
# ==========================================

print("\nLoading dataset...")
df = pd.read_csv(args.input)

X = df.drop(columns=["Score", "identifier"], errors="ignore")
y = df["Score"].values

print(f"Dataset shape: {df.shape}")
print(f"Number of samples: {len(df)}")
print(f"Number of features: {X.shape[1]}")

# ==========================================
# Nested CV
# ==========================================

outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

pipeline = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("model", AdaBoostRegressor(random_state=42))
])

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1, 1.0]
}

fold_results = []
fold_models = []
fold_data = []

print("\nRunning nested cross-validation...")

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), 1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    grid = GridSearchCV(
        pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    preds = best_model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    fold_results.append({
        "Fold": fold,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    })

    fold_models.append((fold, best_model, grid.best_params_, rmse))
    fold_data.append((fold, train_idx, test_idx, X_train, X_test, y_train, y_test, preds))

    print(f"Fold {fold}: RMSE = {rmse:.4f}")

performance_df = pd.DataFrame(fold_results)

# ==========================================
# Select best outer fold
# ==========================================

best_fold = performance_df.loc[performance_df["RMSE"].idxmin(), "Fold"]
best_fold = int(best_fold)

best_model, best_params, best_rmse = [
    x[1:] for x in fold_models if x[0] == best_fold
][0]

best_fold_data = [x for x in fold_data if x[0] == best_fold][0]
_, train_idx, test_idx, X_train, X_test, y_train, y_test, preds = best_fold_data

print(f"\nBest outer fold: {best_fold}")
print(f"Best fold RMSE: {best_rmse:.4f}")
print(f"Mean outer-fold RMSE: {performance_df['RMSE'].mean():.4f}")

# ==========================================
# Best fold transformed data
# ==========================================

X_train_imputed = best_model.named_steps["imputer"].transform(X_train)
X_test_imputed = best_model.named_steps["imputer"].transform(X_test)

model = best_model.named_steps["model"]

X_test_imputed_df = pd.DataFrame(
    X_test_imputed,
    columns=X.columns,
    index=X_test.index
)

# ==========================================
# SHAP
# ==========================================

print("\nCalculating SHAP values...")

background_size = min(100, X_train_imputed.shape[0])
background = shap.sample(X_train_imputed, background_size, random_state=42)

explainer = shap.Explainer(model.predict, background)
shap_output = explainer(X_test_imputed)

shap_values = np.asarray(shap_output.values)
base_values = np.asarray(shap_output.base_values)

# ==========================================
# 1. analysis_metadata.csv
# ==========================================

pd.DataFrame({
    "Parameter": [
        "Input_File",
        "Model",
        "Best_Fold",
        "Test_Samples",
        "Features",
        "Outer_CV_Splits",
        "Inner_CV_Splits",
        "KNN_Neighbors",
        "SHAP_Background_Size",
        "Random_State",
        "Best_RMSE",
        "Mean_RMSE",
        "SHAP_Explainer"
    ],
    "Value": [
        args.input,
        "AdaBoostRegressor",
        best_fold,
        len(X_test),
        X.shape[1],
        5,
        3,
        5,
        background_size,
        42,
        best_rmse,
        performance_df["RMSE"].mean(),
        "shap.Explainer"
    ]
}).to_csv(
    os.path.join(args.output, "analysis_metadata.csv"),
    index=False
)

# ==========================================
# 2. base_values.csv
# ==========================================

pd.DataFrame({
    "OriginalIndex": X_test.index,
    "BaseValue": base_values
}).to_csv(
    os.path.join(args.output, "base_values.csv"),
    index=False
)

# ==========================================
# 3. best_parameters.csv
# ==========================================

pd.DataFrame([best_params]).to_csv(
    os.path.join(args.output, "best_parameters.csv"),
    index=False
)

# ==========================================
# 4. feature_values_imputed.csv
# ==========================================

X_test_imputed_df.to_csv(
    os.path.join(args.output, "feature_values_imputed.csv"),
    index=False
)

# ==========================================
# 5. feature_values_original.csv
# ==========================================

X_test.to_csv(
    os.path.join(args.output, "feature_values_original.csv"),
    index=False
)

# ==========================================
# 6. sample_identifiers.csv
# ==========================================

if "identifier" in df.columns:
    pd.DataFrame({
        "OriginalIndex": X_test.index,
        "identifier": df.loc[X_test.index, "identifier"].values
    }).to_csv(
        os.path.join(args.output, "sample_identifiers.csv"),
        index=False
    )
else:
    pd.DataFrame({
        "OriginalIndex": X_test.index
    }).to_csv(
        os.path.join(args.output, "sample_identifiers.csv"),
        index=False
    )

# ==========================================
# 7. model_performance.csv
# ==========================================

performance_df.to_csv(
    os.path.join(args.output, "model_performance.csv"),
    index=False
)

# ==========================================
# 8. predictions.csv
# ==========================================

pd.DataFrame({
    "OriginalIndex": X_test.index,
    "ActualScore": y_test,
    "PredictedScore": preds,
    "Residual": y_test - preds
}).to_csv(
    os.path.join(args.output, "predictions.csv"),
    index=False
)

# ==========================================
# 9. shap_importance.csv
# ==========================================

importance_df = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(shap_values).mean(axis=0)
}).sort_values(
    "MeanAbsSHAP",
    ascending=False
)

importance_df.to_csv(
    os.path.join(args.output, "shap_importance.csv"),
    index=False
)

# ==========================================
# 10. shap_plot_data.csv
# ==========================================

plot_data = []

for i, original_index in enumerate(X_test.index):
    for j, feature in enumerate(X.columns):
        plot_data.append({
            "OriginalIndex": original_index,
            "Feature": feature,
            "FeatureValue": X_test_imputed_df.iloc[i, j],
            "SHAPValue": shap_values[i, j]
        })

pd.DataFrame(plot_data).to_csv(
    os.path.join(args.output, "shap_plot_data.csv"),
    index=False
)

# ==========================================
# 11. shap_values.csv
# ==========================================

shap_df = pd.DataFrame(
    shap_values,
    columns=X.columns,
    index=X_test.index
)

shap_df.insert(0, "OriginalIndex", X_test.index)

shap_df.to_csv(
    os.path.join(args.output, "shap_values.csv"),
    index=False
)

# ==========================================
# Summary
# ==========================================

print("\nTop 20 SHAP Features:")
print(importance_df.head(20).to_string(index=False))

print("\nAnalysis complete.")
print(f"Output directory: {args.output}")
print("\nFiles generated:")
print("analysis_metadata.csv")
print("base_values.csv")
print("best_parameters.csv")
print("feature_values_imputed.csv")
print("feature_values_original.csv")
print("sample_identifiers.csv")
print("model_performance.csv")
print("predictions.csv")
print("shap_importance.csv")
print("shap_plot_data.csv")
print("shap_values.csv")