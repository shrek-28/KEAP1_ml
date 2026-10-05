import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge docking scores with predicted top-scoring molecules."
    )

    parser.add_argument(
        "--docking-scores",
        required=True,
        help="Path to the docking scores CSV."
    )
    parser.add_argument(
        "--top-scorers",
        required=True,
        help="Path to the top scorers CSV."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the merged output CSV."
    )

    args = parser.parse_args()

    # Load files
    df1 = pd.read_csv(args.docking_scores)
    df2 = pd.read_csv(args.top_scorers)

    # Remove 'minimized_' prefix from Ligand
    df1["Ligand"] = df1["Ligand"].str.replace(
        "minimized_", "", regex=False
    )

    # Merge on ligand identifier
    merged = pd.merge(
        df1,
        df2,
        left_on="Ligand",
        right_on="identifier",
        how="inner"
    )

    # Remove duplicate identifier column
    merged.drop(columns=["identifier"], inplace=True)

    # Rename docking score
    merged.rename(
        columns={"DockingScore_kcal_per_mol": "actual_docking_score"},
        inplace=True
    )

    # Save
    merged.to_csv(args.output, index=False)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()