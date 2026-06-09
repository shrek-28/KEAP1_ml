import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pytorch_tabnet.tab_model import TabNetRegressor


# ---------------- METRICS ----------------
def mape(y, p):
    return np.mean(np.abs((y - p) / (y + 1e-8))) * 100


def ci(x):
    x = np.array(x)
    return 1.96 * x.std() / np.sqrt(len(x))


# ---------------- EVALUATION ----------------
def evaluate_tabnet(csv_path, target="Score", n_splits=5):

    df = pd.read_csv(csv_path)

    X = df.drop(columns=[target, "identifier"], errors="ignore")
    y = df[target].values.reshape(-1, 1)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    mae_list, mse_list, rmse_list, r2_list, mape_list = [], [], [], [], []

    for fold, (tr, te) in enumerate(cv.split(X), 1):

        print(f"Fold {fold}/{n_splits}")

        # preprocessing
        imputer = KNNImputer(n_neighbors=5)
        scaler = StandardScaler()

        X_train = scaler.fit_transform(imputer.fit_transform(X.iloc[tr]))
        X_test = scaler.transform(imputer.transform(X.iloc[te]))

        y_train = y[tr]
        y_test = y[te]

        # TabNet model
        model = TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax"
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            eval_metric=["rmse"],
            max_epochs=200,
            patience=20,
            batch_size=32,
            virtual_batch_size=16,
        )

        pred = model.predict(X_test).flatten()

        mae_list.append(mean_absolute_error(y_test, pred))
        mse_list.append(mean_squared_error(y_test, pred))
        rmse_list.append(np.sqrt(mean_squared_error(y_test, pred)))
        r2_list.append(r2_score(y_test, pred))
        mape_list.append(mape(y_test, pred))

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

    res = evaluate_tabnet("data/final_features_data/ratios_only.csv")

    print("\nFINAL TABNET RESULTS")
    for k, v in res.items():
        print(f"{k}: {v:.4f}")

    pd.DataFrame([res]).to_csv(
        "data/regression_results/tabnet_results.csv",
        index=False
    )