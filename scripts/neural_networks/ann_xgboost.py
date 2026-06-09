import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor


# ---------------- METRICS ----------------
def mape(y, p):
    return np.mean(np.abs((y - p) / (y + 1e-8))) * 100


def ci(arr):
    arr = np.array(arr)
    return 1.96 * arr.std() / np.sqrt(len(arr))


# ---------------- MODEL ----------------
def get_model():

    base_models = [
        ("ann", MLPRegressor(
            hidden_layer_sizes=(256,128,64,32),
            max_iter=2000,
            alpha=0.001,
            early_stopping=True,
            random_state=42
        )),

        ("xgb", XGBRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            objective="reg:squarederror",
            random_state=42,
            verbosity=0
        ))
    ]

    return Pipeline([
        ("imputer", KNNImputer(n_neighbors=5, weights="distance")),
        ("scaler", StandardScaler()),
        ("model", StackingRegressor(
            estimators=base_models,
            final_estimator=Ridge(alpha=1.0),
            cv=3,
            n_jobs=-1
        ))
    ])


# ---------------- EVALUATION ----------------
def evaluate(csv_path, target="Score", n_splits=5):

    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target, "identifier"], errors="ignore")
    y = df[target].values

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mae_list, mse_list, rmse_list, r2_list, mape_list = [], [], [], [], []

    for fold, (tr, te) in enumerate(cv.split(X), 1):

        print(f"Fold {fold}/{n_splits}")

        model = get_model()

        model.fit(X.iloc[tr], y[tr])
        pred = model.predict(X.iloc[te])

        mae_list.append(mean_absolute_error(y[te], pred))
        mse_list.append(mean_squared_error(y[te], pred))
        rmse_list.append(np.sqrt(mean_squared_error(y[te], pred)))
        r2_list.append(r2_score(y[te], pred))
        mape_list.append(mape(y[te], pred))

    results = {
        "MAE_mean": np.mean(mae_list),
        "MAE_CI": ci(mae_list),

        "MSE_mean": np.mean(mse_list),
        "MSE_CI": ci(mse_list),

        "RMSE_mean": np.mean(rmse_list),
        "RMSE_CI": ci(rmse_list),

        "R2_mean": np.mean(r2_list),
        "R2_CI": ci(r2_list),

        "MAPE_mean": np.mean(mape_list),
        "MAPE_CI": ci(mape_list),
    }

    return results


# ---------------- RUN ----------------
if __name__ == "__main__":

    res = evaluate("data/final_features_data/ratios_only.csv")

    print("\nFINAL RESULTS")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([res]).to_csv(
    "data/regression_results/ann_xgboost_results.csv",
    index=False
)