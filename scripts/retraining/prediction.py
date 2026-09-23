import pandas as pd
import joblib
import argparse

parser = argparse.ArgumentParser(description="Generating final predictions using trained model")
parser.add_input("--model", required=True, help="Model pkl file")
parser.add_input("--data", required=True, help="cleaned data for non-training molecules")
parser.add_input("--output", required=True, help="output file path")

args = parser.parse_args()

# Load trained model
model = joblib.load(args.model)

# Load new data
df = pd.read_csv(args.data)

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

# results.to_csv("data/new_data_pred/predictions.csv", index=False)
results.to_csv(args.output, index=False)

print(results.head())