import os
import time
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------- MODEL ----------------
class ANN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),

            nn.Linear(64, 32),
            nn.ReLU(),

            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)


# ---------------- METRICS ----------------
def mape(y, p):
    return np.mean(np.abs((y - p) / (y + 1e-8))) * 100


# ---------------- TRAIN ----------------
def train(model, train_loader, val_loader, epochs=150):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    best_loss = np.inf
    patience, wait = 15, 0

    for _ in range(epochs):

        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_loss = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_loss.append(loss_fn(model(xb), yb).item())

        val_loss = np.mean(val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = model.state_dict()
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model


# ---------------- MAIN EVALUATION ----------------
def evaluate_ann(csv_path,
                 output_csv="ann_results.csv",
                 n_splits=5):

    df = pd.read_csv(csv_path)

    X = df.drop(columns=["Score", "identifier"], errors="ignore")
    y = df["Score"].values.reshape(-1, 1)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = []
    start_time = time.time()

    fold_times = []

    for fold, (tr, te) in enumerate(cv.split(X), 1):

        fold_start = time.time()
        print(f"\n[Fold {fold}/{n_splits}] Starting...")

        # preprocessing
        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()

        X_train = scaler.fit_transform(imputer.fit_transform(X.iloc[tr]))
        X_test = scaler.transform(imputer.transform(X.iloc[te]))

        y_train, y_test = y[tr], y[te]

        # tensors
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=32,
            shuffle=True
        )

        val_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=32
        )

        model = ANN(X.shape[1])
        model = train(model, train_loader, val_loader)

        model.eval()
        with torch.no_grad():
            preds = model(X_test).numpy().flatten()

        y_true = y_test.flatten()

        results.append({
            "MAE": mean_absolute_error(y_true, preds),
            "MSE": mean_squared_error(y_true, preds),
            "RMSE": np.sqrt(mean_squared_error(y_true, preds)),
            "R2": r2_score(y_true, preds),
            "MAPE": mape(y_true, preds)
        })

        # ETA
        fold_time = time.time() - fold_start
        fold_times.append(fold_time)

        avg_time = np.mean(fold_times)
        remaining = (n_splits - fold) * avg_time

        print(f"Fold {fold} done in {fold_time:.2f}s")
        print(f"ETA remaining: {remaining/60:.2f} min")

    df_out = pd.DataFrame(results)

    def ci(x):
        return 1.96 * np.std(x, ddof=1) / np.sqrt(n_splits)

    summary = {
        "dataset": "ratios_only.csv",

        "MAE_mean": df_out["MAE"].mean(),
        "MAE_CI": ci(df_out["MAE"]),

        "MSE_mean": df_out["MSE"].mean(),
        "MSE_CI": ci(df_out["MSE"]),

        "RMSE_mean": df_out["RMSE"].mean(),
        "RMSE_CI": ci(df_out["RMSE"]),

        "R2_mean": df_out["R2"].mean(),
        "R2_CI": ci(df_out["R2"]),

        "MAPE_mean": df_out["MAPE"].mean(),
        "MAPE_CI": ci(df_out["MAPE"])
    }

    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(output_csv, index=False)

    print("\nFINAL SINGLE ROW RESULT")
    print(df_summary)

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time/60:.2f} min")

    return df_summary


# ---------------- RUN ----------------
if __name__ == "__main__":

    evaluate_ann(
        csv_path="data/final_features_data/ratios_only.csv",
        output_csv="data/regression_results/ann_results.csv"
    )