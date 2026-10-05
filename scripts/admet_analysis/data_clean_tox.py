import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Add identifiers to toxicity results using row order."
    )

    parser.add_argument("--toxicity", required=True, help="Path to the toxicity results CSV.")
    parser.add_argument(
        "--identifiers",
        required=True,
        help="Path to the CSV containing identifiers."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the output CSV."
    )

    args = parser.parse_args()

    # Load files
    tox_df = pd.read_csv(args.toxicity)
    id_df = pd.read_csv(args.identifiers)

    # Ensure same number of rows
    if len(tox_df) != len(id_df):
        raise ValueError(
            f"Row count mismatch: toxicity file has {len(tox_df)} rows, "
            f"identifier file has {len(id_df)} rows."
        )

    # Copy identifier by row order
    tox_df["identifier"] = id_df["identifier"].values

    # Move identifier to first column
    cols = ["identifier"] + [
        c for c in tox_df.columns if c != "identifier"
    ]
    tox_df = tox_df[cols]

    # Save
    tox_df.to_csv(args.output, index=False)

    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()