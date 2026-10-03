import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Summarize top-20 SHAP ratios across models."
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing model subdirectories with shap_importance.csv"
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <input_dir>/summary_shap_ratios.csv)"
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    output_file = (
        Path(args.output)
        if args.output
        else input_dir / "summary_shap_ratios.csv"
    )

    all_data = {}

    # Find each model's SHAP importance file
    for shap_file in input_dir.glob("*/shap_importance.csv"):
        model_name = shap_file.parent.name

        df = pd.read_csv(shap_file).head(20)

        # Round MeanAbsSHAP to 3 decimals
        df["MeanAbsSHAP"] = df["MeanAbsSHAP"].round(3)

        all_data[model_name] = dict(
            zip(df["Feature"], df["MeanAbsSHAP"])
        )

    if not all_data:
        raise FileNotFoundError(
            f"No */shap_importance.csv files found in {input_dir}"
        )

    # Collect every ratio appearing in at least one top-20 list
    all_features = sorted(
        {
            feature
            for model_data in all_data.values()
            for feature in model_data
        }
    )

    # Build summary
    summary = pd.DataFrame(index=all_features)

    for model_name, model_data in all_data.items():
        summary[model_name] = [
            model_data.get(feature, "NA")
            for feature in all_features
        ]

    summary.index.name = "Feature"

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_file)

    print(f"Models found: {len(all_data)}")
    print(f"Unique ratios: {len(all_features)}")
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    main()