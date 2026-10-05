import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter and rank molecules based on interactions and safety categories."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV file"
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output CSV file"
    )

    args = parser.parse_args()

    # Read input
    df = pd.read_csv(args.input)

    # Safety/toxicity columns
    safety_cols = [
        "Genomic_AMES_Mutagenesis",
        "Organic_hERG_I_Inhibitor",
        "Organic_hERG_II_Inhibitor",
        "Genomic_Micronucleus",
        "Genomic_Carcinogenesis"
    ]

    # Check required columns
    required_cols = [
        "n_conformations_with_bonds_of_interest",
        "total_hydrogen_bond_interactions_of_interest",
        "total_hydrophobic_interactions_of_interest",
        *safety_cols
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"Missing required columns: {', '.join(missing)}"
        )

    # ---------------------------------------------------------
    # 1. Filter: n_conformations_with_bonds_of_interest = 9
    # ---------------------------------------------------------
    df = df[
        df["n_conformations_with_bonds_of_interest"] == 9
    ].copy()

    # ---------------------------------------------------------
    # 2. Normalize safety category values
    # ---------------------------------------------------------
    safety_data = (
        df[safety_cols]
        .astype(str)
        .apply(lambda col: col.str.strip().str.lower())
    )

    # ---------------------------------------------------------
    # 3. Count each safety category across the 5 columns
    # ---------------------------------------------------------
    df["high_safety_count"] = (
        safety_data == "High Safety"
    ).sum(axis=1)

    df["medium_safety_count"] = (
        safety_data == "Medium Safety"
    ).sum(axis=1)

    df["low_safety_count"] = (
        safety_data == "Low Safety"
    ).sum(axis=1)

    df["low_toxicity_count"] = (
        safety_data == "Low Toxicity"
    ).sum(axis=1)

    # ---------------------------------------------------------
    # 4. Sort
    #
    # Priority:
    #   n_conformations = 9
    #   high safety
    #   medium safety
    #   low safety
    #   hydrogen bonds
    #   hydrophobic interactions
    #
    # n_conformations is already filtered to 9.
    # ---------------------------------------------------------
    df = df.sort_values(
        by=[
            "n_conformations_with_bonds_of_interest",
            "high_safety_count",
            "medium_safety_count",
            "low_safety_count",
            "total_hydrogen_bond_interactions_of_interest",
            "total_hydrophobic_interactions_of_interest"
        ],
        ascending=[
            False,
            False,
            False,
            False,
            False,
            False
        ]
    )

    # ---------------------------------------------------------
    # 5. Remove helper columns
    # ---------------------------------------------------------
    df = df.drop(
        columns=[
            "high_safety_count",
            "medium_safety_count",
            "low_safety_count",
            "low_toxicity_count"
        ]
    )

    # Save
    df.to_csv(args.output, index=False)

    print(f"Saved sorted output to: {args.output}")
    print(f"Rows retained: {len(df)}")


if __name__ == "__main__":
    main()