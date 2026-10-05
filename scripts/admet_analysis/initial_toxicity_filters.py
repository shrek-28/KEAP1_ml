import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Filter out molecules with medium or high toxicity."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the interpretation data CSV."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the non-toxic output CSV."
    )

    args = parser.parse_args()

    df = pd.read_csv(args.input)

    cols = [
        "Genomic_AMES_Mutagenesis",
        "Organic_hERG_I_Inhibitor",
        "Organic_hERG_II_Inhibitor",
        "Genomic_Micronucleus",
        "Genomic_Carcinogenesis",
    ]

    medium_high_mask = (
        df[cols]
        .astype(str)
        .apply(
            lambda col: col.str.contains(
                "High toxicity|Medium toxicity",
                case=False,
                na=False
            )
        )
        .any(axis=1)
    )

    print(
        "At least one MEDIUM or HIGH:",
        medium_high_mask.sum()
    )

    remaining_df = df[~medium_high_mask].copy()

    print("Rows remaining:", len(remaining_df))

    remaining_df.to_csv(args.output, index=False)

    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()