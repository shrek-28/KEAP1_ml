import os
import numpy as np
import pandas as pd
import shap

from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==========================================================
# CONFIG
# ==========================================================

df = pd.read_csv("data/final_features_data/ratios_only.csv")

OUT_SHAP_VALUES = "data/SHAP/lasso_reg/lasso_shap_values.csv"
OUT_SHAP_VALUES_INDEXED = "data/SHAP/lasso_reg/lasso_shap_values_with_index.csv"
OUT_SHAP_IMPORTANCE = "data/SHAP/lasso_reg/lasso_shap_feature_importance.csv"

OUTER_SPLITS = 5
INNER_SPLITS = 3

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv("data/final_features_data/ratios_only.csv")

if "Score" not in df.columns:
    raise ValueError("Missing target column: Score")

X = df.drop(columns=["Score", "identifier"], errors="ignore")
y = df["Score"].values

# ==========================================================
# CV SETUP
# ==========================================================

outer_cv = KFold(n_splits=OUTER_SPLITS, shuffle=True, random_state=42)
inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=42)

param_grid = {
    "model__alpha": [1e-3, 1e-2, 1e-1, 1, 10, 100],
    "model__fit_intercept": [True],
    "model__selection": ["cyclic"],
    "model__max_iter": [30000],
    "model__tol": [1e-3, 1e-2],
}

def mape(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

# ==========================================================
# NESTED CV + BEST MODEL SELECTION
# ==========================================================

fold_metrics = []

best_rmse = np.inf
best_model = None
best_X_train = None
best_X_test = None

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X), start=1):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    pipeline = Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
        ("model", Lasso())
    ])

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=inner_cv,
        scoring="neg_root_mean_squared_error",
        n_jobs=2,
        error_score="raise"
    )

    grid.fit(X_train, y_train)

    best_estimator = grid.best_estimator_
    preds = best_estimator.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)
    mape_val = mape(y_test, preds)

    fold_metrics.append({
        "Fold": fold,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape_val
    })

    print(f"Fold {fold} | RMSE: {rmse:.4f} | R2: {r2:.4f}")

    if rmse < best_rmse:
        best_rmse = rmse
        best_model = best_estimator
        best_X_train = X_train.copy()
        best_X_test = X_test.copy()

metrics_df = pd.DataFrame(fold_metrics)

print("\nFINAL METRICS")
print(metrics_df)
print("\nMean RMSE:", metrics_df["RMSE"].mean())

# ==========================================================
# SHAP (NO PLOTS)
# ==========================================================

print("\nComputing SHAP values...")

# preprocess using fitted pipeline
X_train_proc = best_model.named_steps["imputer"].transform(best_X_train)
X_train_proc = best_model.named_steps["scaler"].transform(X_train_proc)

X_test_proc = best_model.named_steps["imputer"].transform(best_X_test)
X_test_proc = best_model.named_steps["scaler"].transform(X_test_proc)

# SHAP for linear model
explainer = shap.LinearExplainer(
    best_model.named_steps["model"],
    X_train_proc
)

shap_values = explainer(X_test_proc)

# ==========================================================
# 1. SHAP VALUES (PER SAMPLE)
# ==========================================================

shap_df = pd.DataFrame(
    shap_values.values,
    columns=best_X_test.columns
)

shap_df.to_csv(OUT_SHAP_VALUES, index=False)

# ==========================================================
# 2. SHAP VALUES + ORIGINAL INDEX
# ==========================================================

shap_indexed = shap_df.copy()
shap_indexed.insert(0, "OriginalIndex", best_X_test.index)

shap_indexed.to_csv(OUT_SHAP_VALUES_INDEXED, index=False)

# ==========================================================
# 3. GLOBAL FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({
    "Feature": best_X_test.columns,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values("MeanAbsSHAP", ascending=False)

importance_df.to_csv(OUT_SHAP_IMPORTANCE, index=False)

print("\nSaved outputs:")
print(" -", OUT_SHAP_VALUES)
print(" -", OUT_SHAP_VALUES_INDEXED)
print(" -", OUT_SHAP_IMPORTANCE)