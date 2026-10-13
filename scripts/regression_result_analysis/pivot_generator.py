import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Create a pivot table with two-level column headers, "
                    "dataset averages, and model averages."
    )
    parser.add_argument(
        "-i", "--input", required=True,
        help="Input CSV file"
    )
    parser.add_argument(
        "-o", "--output", default="pivot_results.xlsx",
        help="Output Excel file"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    metrics = {
        "MAE ± CI": ("MAE_mean", "MAE_ci"),
        "RMSE ± CI": ("RMSE_mean", "RMSE_ci"),
        "R² ± CI": ("R2_mean", "R2_ci"),
        "MAPE ± CI": ("MAPE_mean", "MAPE_ci"),
    }

    # Ensure metric columns are numeric
    numeric_cols = [
        col for pair in metrics.values() for col in pair
    ]
    df[numeric_cols] = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    )

    # Preserve original dataset and model order
    dataset_order = df["dataset"].drop_duplicates().tolist()
    model_order = df["model"].drop_duplicates().tolist()
    metric_order = list(metrics.keys())

    # Calculate dataset-level averages across models
    dataset_averages = (
        df.groupby("dataset", sort=False)[numeric_cols]
        .mean()
        .reset_index()
    )
    dataset_averages["model"] = "Average"

    # Format metric values as mean ± CI, rounded to 2 decimals
    def format_metrics(data):
        for label, (mean_col, ci_col) in metrics.items():
            data[label] = (
                data[mean_col].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )
                + " ± "
                + data[ci_col].map(
                    lambda x: f"{x:.2f}" if pd.notna(x) else ""
                )
            )
        return data

    formatted_df = format_metrics(df.copy())
    formatted_averages = format_metrics(dataset_averages.copy())

    # Combine original results with dataset-average rows
    combined = pd.concat(
        [formatted_df, formatted_averages],
        ignore_index=True
    )

    # Build pivot: rows = models, columns = dataset → metric
    pivot = combined.pivot(
        index="model",
        columns="dataset",
        values=metric_order
    )

    # Put dataset at the first column-header level
    pivot = pivot.swaplevel(0, 1, axis=1)

    # Preserve dataset and metric order
    pivot = pivot.reindex(
        columns=pd.MultiIndex.from_product(
            [dataset_order, metric_order]
        )
    )

    # Reorder model rows and place Average at the bottom
    pivot = pivot.reindex(model_order + ["Average"])

    # Calculate model averages across datasets using the original
    # numerical mean and CI columns, rather than formatted strings
    model_averages = (
        df.groupby("model", sort=False)[numeric_cols]
        .mean()
    )

    # Format model averages
    average_columns = {}
    for label, (mean_col, ci_col) in metrics.items():
        average_columns[label] = (
            model_averages[mean_col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
            + " ± "
            + model_averages[ci_col].map(
                lambda x: f"{x:.2f}" if pd.notna(x) else ""
            )
        )

    # Add the model-average column as the final column group
    for label in metric_order:
        pivot[("Average", label)] = average_columns[label].reindex(
            pivot.index
        )

    # Dataset-average row: calculate the overall average across
    # datasets for each metric and populate the final column group
    for label, (mean_col, ci_col) in metrics.items():
        avg_mean = df[mean_col].mean()
        avg_ci = df[ci_col].mean()

        pivot.loc["Average", ("Average", label)] = (
            f"{avg_mean:.2f} ± {avg_ci:.2f}"
            if pd.notna(avg_mean) and pd.notna(avg_ci)
            else ""
        )

    # Write Excel with merged two-level column headers
    pivot.to_excel(args.output, merge_cells=True)

    print(f"Saved pivot table to: {args.output}")


if __name__ == "__main__":
    main()