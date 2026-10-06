#!/usr/bin/env python3

import argparse
import pandas as pd


def parse_args():

    parser = argparse.ArgumentParser(
        description="Filter a Tanimoto similarity matrix to selected CNP IDs."
    )

    parser.add_argument(
        "--matrix",
        default="data/post_dock_analysis/mmr_analysis/tanimoto_similarity_matrix.csv",
        help="Input Tanimoto similarity matrix."
    )

    parser.add_argument(
        "--selected",
        default="data/post_dock_analysis/mmr_analysis/mmr_lambda_results_frequency.csv",
        help="CSV containing selected CNP_IDs."
    )

    parser.add_argument(
        "--output",
        default="data/post_dock_analysis/mmr_analysis/tanimoto_similarity_matrix_final.csv",
        help="Output filtered Tanimoto similarity matrix."
    )

    return parser.parse_args()


def main():

    args = parse_args()

    # ---------------------------------------------------------
    # Read selected CNP IDs
    # ---------------------------------------------------------

    selected_df = pd.read_csv(args.selected)

    if "CNP_ID" not in selected_df.columns:
        raise ValueError(
            "Selected file does not contain a 'CNP_ID' column."
        )

    selected_ids = (
        selected_df["CNP_ID"]
        .astype(str)
        .drop_duplicates()
        .tolist()
    )

    if len(selected_ids) != 5:
        raise ValueError(
            f"Expected 5 unique CNP_IDs, "
            f"found {len(selected_ids)}."
        )

    # ---------------------------------------------------------
    # Read Tanimoto matrix
    # ---------------------------------------------------------

    matrix = pd.read_csv(
        args.matrix,
        index_col=0
    )

    # Convert matrix labels to strings
    matrix.index = matrix.index.astype(str)
    matrix.columns = matrix.columns.astype(str)

    # ---------------------------------------------------------
    # Check that all selected molecules are present
    # ---------------------------------------------------------

    missing_rows = [
        cnp_id
        for cnp_id in selected_ids
        if cnp_id not in matrix.index
    ]

    missing_columns = [
        cnp_id
        for cnp_id in selected_ids
        if cnp_id not in matrix.columns
    ]

    if missing_rows or missing_columns:

        missing = sorted(
            set(missing_rows + missing_columns)
        )

        raise ValueError(
            "The following selected CNP_IDs are missing "
            "from the Tanimoto matrix: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # Filter rows and columns
    # ---------------------------------------------------------

    final_matrix = matrix.loc[
        selected_ids,
        selected_ids
    ]

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    final_matrix.index.name = "CNP_ID"

    final_matrix.to_csv(
        args.output
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\nSelected molecules:")

    for cnp_id in selected_ids:
        print(cnp_id)

    print(
        f"\nFinal matrix shape: "
        f"{final_matrix.shape[0]} × {final_matrix.shape[1]}"
    )

    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()