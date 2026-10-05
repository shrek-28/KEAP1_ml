import argparse
import pandas as pd


TOXICITY_COLS = [
    "Genomic_AMES_Mutagenesis",
    "Organic_hERG_I_Inhibitor",
    "Organic_hERG_II_Inhibitor",
    "Genomic_Micronucleus",
    "Genomic_Carcinogenesis",
]


def main():
    parser = argparse.ArgumentParser(
        description="Filter interaction data to CNP IDs in the non-toxic file and add selected toxicity columns."
    )

    parser.add_argument(
        "-i", "--interactions",
        required=True,
        help="Aggregated interaction CSV"
    )

    parser.add_argument(
        "-n", "--non-toxic",
        required=True,
        help="Non-toxic molecules CSV"
    )

    parser.add_argument(
        "-o", "--output",
        required=True,
        help="Output CSV"
    )

    parser.add_argument(
        "--interaction-id",
        required=True,
        help="CNP ID column name in interaction CSV"
    )

    parser.add_argument(
        "--non-toxic-id",
        required=True,
        help="CNP ID column name in non-toxic CSV"
    )

    args = parser.parse_args()

    # Read files
    interactions = pd.read_csv(args.interactions)
    non_toxic = pd.read_csv(args.non_toxic)

    # Validate ID columns
    if args.interaction_id not in interactions.columns:
        raise ValueError(
            f"Column '{args.interaction_id}' not found in interaction file."
        )

    if args.non_toxic_id not in non_toxic.columns:
        raise ValueError(
            f"Column '{args.non_toxic_id}' not found in non-toxic file."
        )

    # Validate toxicity columns
    missing = [
        col for col in TOXICITY_COLS
        if col not in non_toxic.columns
    ]

    if missing:
        raise ValueError(
            f"Missing toxicity columns in non-toxic file: {missing}"
        )

    # Standardize CNP IDs
    interactions[args.interaction_id] = (
        interactions[args.interaction_id]
        .astype(str)
        .str.strip()
    )

    non_toxic[args.non_toxic_id] = (
        non_toxic[args.non_toxic_id]
        .astype(str)
        .str.strip()
    )

    # Keep only CNP ID + requested toxicity columns from non-toxic file
    toxicity = non_toxic[
        [args.non_toxic_id] + TOXICITY_COLS
    ].copy()

    # Rename the non-toxic ID temporarily so the merge key is identical
    toxicity = toxicity.rename(
        columns={args.non_toxic_id: args.interaction_id}
    )

    # Remove duplicate CNP IDs from toxicity data
    toxicity = toxicity.drop_duplicates(
        subset=args.interaction_id,
        keep="first"
    )

    # Filter interaction data to ONLY CNP IDs present in non-toxic file
    valid_cnp_ids = set(toxicity[args.interaction_id])

    interactions = interactions[
        interactions[args.interaction_id].isin(valid_cnp_ids)
    ].copy()

    # Merge all interaction columns with the five toxicity columns
    merged = interactions.merge(
        toxicity,
        on=args.interaction_id,
        how="left"
    )

    # Save
    merged.to_csv(args.output, index=False)

    # Report
    matched_ids = merged[TOXICITY_COLS].notna().any(axis=1).sum()

    print(f"Output written to: {args.output}")
    print(f"Non-toxic CNP IDs: {len(valid_cnp_ids)}")
    print(f"Interaction rows retained: {len(interactions)}")
    print(f"Rows with toxicity data: {matched_ids}")
    print(f"Output rows: {len(merged)}")


if __name__ == "__main__":
    main()
