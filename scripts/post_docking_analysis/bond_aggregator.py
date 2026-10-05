#!/usr/bin/env python3

import argparse
import csv
import glob
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple bond-extraction CSV files generated "
            "from post-docking analysis."
        )
    )

    parser.add_argument(
        "--input",
        nargs="+",
        required=True,
        help=(
            "Input CSV files. You can provide five files explicitly "
            "or use a glob pattern."
        )
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output merged CSV file."
    )

    return parser.parse_args()


def resolve_input_files(input_patterns):
    """
    Resolve explicit filenames and glob patterns.
    """

    files = []

    for pattern in input_patterns:

        matches = glob.glob(pattern)

        if matches:
            files.extend(matches)
        elif os.path.isfile(pattern):
            files.append(pattern)
        else:
            print(
                f"Warning: no file found for '{pattern}'",
                file=sys.stderr
            )

    # Remove duplicates while preserving order
    files = list(dict.fromkeys(files))

    return files


def merge_csvs(input_files, output_file):

    if not input_files:
        raise ValueError("No input CSV files found.")

    reference_columns = None
    total_rows = 0

    with open(
        output_file,
        "w",
        encoding="utf-8",
        newline=""
    ) as outfile:

        writer = None

        for input_file in input_files:

            print(f"Reading: {input_file}")

            with open(
                input_file,
                "r",
                encoding="utf-8",
                newline=""
            ) as infile:

                reader = csv.DictReader(infile)

                if reader.fieldnames is None:
                    raise ValueError(
                        f"No header found in: {input_file}"
                    )

                columns = reader.fieldnames

                # First file defines the expected structure
                if reference_columns is None:
                    reference_columns = columns

                    writer = csv.DictWriter(
                        outfile,
                        fieldnames=reference_columns
                    )

                    writer.writeheader()

                # Make sure every file has the same columns
                elif columns != reference_columns:

                    raise ValueError(
                        "\nColumn mismatch detected.\n"
                        f"Expected:\n{reference_columns}\n"
                        f"Found in {input_file}:\n{columns}"
                    )

                file_rows = 0

                for row in reader:
                    writer.writerow(row)

                    file_rows += 1
                    total_rows += 1

                print(
                    f"  Added {file_rows} rows"
                )

    print()
    print(f"Merged {len(input_files)} files.")
    print(f"Total rows: {total_rows}")
    print(f"Output: {output_file}")


def main():

    args = parse_args()

    input_files = resolve_input_files(args.input)

    if not input_files:
        raise FileNotFoundError(
            "No input files were found."
        )

    merge_csvs(
        input_files,
        args.output
    )


if __name__ == "__main__":
    main()
