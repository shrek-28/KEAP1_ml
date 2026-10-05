import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract interpretation columns along with identifier and SMILES."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Path to the toxicity results CSV."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the interpretation data CSV."
    )

    args = parser.parse_args()

    # Load data
    df = pd.read_csv(args.input)

    # Select interpretation columns
    interpretation_cols = [
        col for col in df.columns
        if col.startswith("Interpretation_")
    ]

    interpretation_cols += ["identifier", "SMILES"]

    df = df[interpretation_cols]

    # Remove Interpretation_ prefix
    df.columns = df.columns.str.replace(
        "Interpretation_",
        "",
        regex=False
    )

    # Save
    df.to_csv(args.output, index=False)

    print(f"Extracted {len(interpretation_cols) - 2} interpretation columns.")
    print(f"Total rows: {len(df)}")
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()