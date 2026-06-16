import numpy as np
import pandas as pd
import shap

from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================
# Load data
# ============================

df = pd.read_csv("data/final_features_data/ratios_only.csv")

X = df.drop(columns=["Score", "identifier"], errors="ignore")
y = df["Score"]

# ============================
# Train Linear Regression
# ============================

pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
    ("scaler", StandardScaler()),
    ("model", LinearRegression(fit_intercept=True))
])

pipe.fit(X, y)

# ============================
# Evaluate on full dataset
# ============================

preds = pipe.predict(X)

print(f"MAE  : {mean_absolute_error(y, preds):.4f}")
print(f"MSE  : {mean_squared_error(y, preds):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y, preds)):.4f}")
print(f"R²   : {r2_score(y, preds):.4f}")

# ============================
# Prepare transformed features
# ============================

X_imputed = pipe.named_steps["imputer"].transform(X)
X_scaled = pipe.named_steps["scaler"].transform(X)

model = pipe.named_steps["model"]

# ============================
# Compute SHAP values
# ============================

explainer = shap.LinearExplainer(model, X_scaled)
shap_values = explainer(X_scaled)

# ============================
# Save SHAP values
# ============================

# SHAP value matrix
shap_df = pd.DataFrame(
    shap_values.values,
    columns=X.columns
)
shap_df.to_csv("data/SHAP/lin_reg/shap_values.csv", index=False)

# Original feature values
X.to_csv("data/SHAP/lin_reg/feature_values.csv", index=False)

# Mean absolute SHAP importance
shap_importance = pd.DataFrame({
    "Feature": X.columns,
    "MeanAbsSHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values(
    by="MeanAbsSHAP",
    ascending=False
)

shap_importance.to_csv("data/SHAP/lin_reg/shap_importance.csv", index=False)

print("\nTop 20 features by mean absolute SHAP value:")
print(shap_importance.head(20))