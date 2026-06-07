import pandas as pd
import numpy as np
from pathlib import Path
from scipy.interpolate import UnivariateSpline


# ============================================================
# Kneedle Method
# ============================================================
def kneedle_k(x, y):
    """
    Kneedle for monotonically decreasing RMSE curves.
    """

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    gain = 1 - y_norm
    diff = gain - x_norm

    knee_idx = np.argmax(diff)

    return knee_idx


# ============================================================
# Main Analysis
# ============================================================
def analyze_feature_selection(files, output_file):

    results = []

    for file in files:

        print(f"\nProcessing: {Path(file).name}")

        df = pd.read_csv(file)

        df = df.sort_values("n_features").reset_index(drop=True)

        x = df["n_features"].values
        y = df["rmse"].values

        # -------------------------
        # Kneedle
        # -------------------------
        kneedle_idx = kneedle_k(x, y)

        # -------------------------
        # Curvature
        # -------------------------

        results.append({
            "dataset": Path(file).stem,

            "k":
                int(x[kneedle_idx]),

            "rmse":
                float(y[kneedle_idx]),

        })

        print(
            f"  Kneedle     -> k={x[kneedle_idx]} "
            f"(RMSE={y[kneedle_idx]:.4f})"
        )


    results_df = pd.DataFrame(results)

    results_df.to_csv(output_file, index=False)

    print(f"\nSaved results to: {output_file}")

    return results_df


# ============================================================
# Example Usage
# ============================================================

folder = Path("data/rmse_split_results")

all_files = list(folder.glob("*.csv"))

results = analyze_feature_selection(
    files=all_files,
    output_file="data/knee_detection_results.csv"
)

print("\nFinal Results")
print(results)