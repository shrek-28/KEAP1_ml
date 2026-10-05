import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge SMILES data with docking intersection data and generate filtered files."
    )

    parser.add_argument(
        "--smiles-data",
        required=True,
        help="Path to the combined SMILES data CSV."
    )
    parser.add_argument(
        "--docking-data",
        required=True,
        help="Path to the merged docking intersection CSV."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the filtered SMILES data CSV."
    )
    parser.add_argument(
        "--smiles-only-output",
        required=True,
        help="Path for the SMILES-only CSV."
    )

    args = parser.parse_args()

    # Load files
    df1 = pd.read_csv(args.smiles_data)
    df2 = pd.read_csv(args.docking_data)

    # Merge on identifier
    merged = pd.merge(
        df1,
        df2,
        left_on="identifier",
        right_on="Ligand",
        how="inner"
    )

    # Remove unnecessary columns
    merged.drop(
        columns=[
            "Ligand",
            "actual_docking_score",
            "predicted_docking_score"
        ],
        inplace=True
    )

    # Rename SMILES column
    merged.rename(
        columns={"canonical_smiles": "SMILES"},
        inplace=True
    )

    # Save filtered dataset
    merged.to_csv(args.output, index=False)

    # Save SMILES-only dataset
    smiles_df = merged[["SMILES"]]
    smiles_df.to_csv(args.smiles_only_output, index=False)

    print(f"Merged rows: {len(merged)}")
    print(f"Saved filtered data to: {args.output}")
    print(f"Saved SMILES-only data to: {args.smiles_only_output}")


if __name__ == "__main__":
    main()