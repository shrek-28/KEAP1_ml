import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ==========================================
# Load dataset
# ==========================================

df = pd.read_csv("data/final_features_data/ratios_only.csv")

X = df.drop(columns=["Score", "identifier"], errors="ignore")
y = df["Score"].values

# ==========================================
# Define pipeline
# ==========================================

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
        AdaBoostRegressor(
            random_state=42
        )
    )
])

# ==========================================
# Hyperparameter grid
# ==========================================

param_grid = {
    "model__n_estimators": [50, 100, 200],
    "model__learning_rate": [0.01, 0.05, 0.1, 1.0],
}

# ==========================================
# Grid Search
# ==========================================

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="neg_root_mean_squared_error",
    n_jobs=-1,
    error_score="raise"
)

grid.fit(X, y)

best_pipeline = grid.best_estimator_

print("Best Parameters:")
print(grid.best_params_)

print(f"Best CV RMSE: {-grid.best_score_:.4f}")

# ==========================================
# Evaluate model
# ==========================================

preds = best_pipeline.predict(X)

print(f"MAE  : {mean_absolute_error(y, preds):.4f}")
print(f"MSE  : {mean_squared_error(y, preds):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y, preds)):.4f}")
print(f"R²   : {r2_score(y, preds):.4f}")

# ==========================================
# Get transformed features
# ==========================================

X_imputed = best_pipeline.named_steps["imputer"].transform(X)
model = best_pipeline.named_steps["model"]

# ==========================================
# SHAP (works with AdaBoostRegressor)
# ==========================================

# Use a small background sample for efficiency
background = shap.sample(X_imputed, 100, random_state=42)

explainer = shap.Explainer(
    model.predict,
    background
)

shap_values = explainer(X_imputed)

# ==========================================
# Save SHAP values
# ==========================================

shap_df = pd.DataFrame(
    shap_values.values,
    columns=X.columns
)

shap_df.to_csv(
    "data/SHAP/adaboost_regressor/shap_values.csv",
    index=False
)

# ==========================================
# Save original feature values
# ==========================================

X.to_csv(
    "data/SHAP/adaboost_regressor/feature_values.csv",
    index=False
)

# ==========================================
# Save SHAP importance
# ==========================================

shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
})

shap_importance = shap_importance.sort_values(
    by="MeanAbsSHAP",
    ascending=False
)

shap_importance.to_csv(
    "data/SHAP/adaboost_regressor/shap_importance.csv",
    index=False
)

print("\nTop 20 SHAP Features:")
print(shap_importance.head(20))