import argparse
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Combine all CSV files in a folder into a single CSV."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing the input CSV files."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the combined output CSV."
    )

    args = parser.parse_args()

    folder = Path(args.input_dir)

    # Get all CSV files
    csv_files = sorted(folder.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in: {folder}"
        )

    # Read and combine
    combined_df = pd.concat(
        (pd.read_csv(file) for file in csv_files),
        ignore_index=True
    )

    # Save
    combined_df.to_csv(args.output, index=False)

    print(f"Combined {len(csv_files)} files into {args.output}")
    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()