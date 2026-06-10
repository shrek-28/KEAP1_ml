import pandas as pd
import joblib

# Load trained model
model = joblib.load("data/final_xgboost/best_xgboost_model.pkl")

# Load new data
df = pd.read_csv("data/retraining_all_data.csv")

# Keep identifiers
identifiers = df["identifier"]

# Features only
X = df.drop(columns=["identifier"])

# Predict
predictions = model.predict(X)

# Save results
results = pd.DataFrame({
    "identifier": identifiers,
    "predicted_docking_score": predictions
})

results.to_csv("data/new_data_pred/predictions.csv", index=False)

print(results.head())