import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Merge SMILES and docking scores into a CNP_ID CSV."
    )

    parser.add_argument("--input", required=True,
                        help="CSV file 1 containing CNP_ID")
    parser.add_argument("--smiles", required=True,
                        help="CSV file 2 containing identifier and canonical_smiles")
    parser.add_argument("--scores", required=True,
                        help="CSV file 3 containing Ligand and BestScore_kcal_per_mol")
    parser.add_argument("--output", required=True,
                        help="Output CSV file")

    args = parser.parse_args()

    # Read input files
    df1 = pd.read_csv(args.input)
    df2 = pd.read_csv(args.smiles)
    df3 = pd.read_csv(args.scores)

    # Rename SMILES columns
    df2 = df2.rename(columns={
        "canonical_smiles": "SMILES"
    })

    # Remove "minimized_" from ligand names
    df3["Ligand"] = (
        df3["Ligand"]
        .astype(str)
        .str.replace("minimized_", "", regex=False)
    )

    # Rename docking score column
    df3 = df3.rename(columns={
        "BestScore_kcal_per_mol": "binding_affinity"
    })

    # Merge SMILES using CNP_ID ↔ identifier
    df = df1.merge(
        df2[["identifier", "SMILES"]],
        left_on="CNP_ID",
        right_on="identifier",
        how="left"
    )

    # Remove duplicate identifier column
    df = df.drop(columns=["identifier"])

    # Merge docking scores using CNP_ID ↔ Ligand
    df = df.merge(
        df3[["Ligand", "binding_affinity"]],
        left_on="CNP_ID",
        right_on="Ligand",
        how="left"
    )

    # Remove duplicate Ligand column
    df = df.drop(columns=["Ligand"])

    # Save
    df.to_csv(args.output, index=False)

    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()