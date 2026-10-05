#!/usr/bin/env python3

import argparse
import csv
import re
from collections import defaultdict


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Summarize hydrogen-bond and hydrophobic interactions "
            "across 9 docking conformations for each CNP ID."
        )
    )

    parser.add_argument(
        "-i",
        "--input",
        required=True,
        help="Input post-docking analysis CSV file."
    )

    parser.add_argument(
        "-b",
        "--bonds-of-interest",
        required=True,
        help="Text file containing residue/bond-of-interest identifiers."
    )

    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output CSV file."
    )

    return parser.parse_args()


def load_bonds_of_interest(filepath):

    """
    Read residues/bonds of interest.

    Each non-empty line can contain one or multiple identifiers.
    Commas, pipes, and whitespace are accepted as separators.
    """

    bonds = set()

    with open(filepath, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            # Remove comments
            line = line.split("#", 1)[0].strip()

            if not line:
                continue

            # Allow comma, pipe, or whitespace-separated entries
            entries = re.split(r"[,\s|]+", line)

            for entry in entries:

                entry = entry.strip()

                if entry:
                    bonds.add(entry.upper())

    return bonds


def split_interactions(value):

    """
    Split pipe-separated interaction identifiers.

    Example:
        'ASN123 | SER125 | GLY98'

    returns:
        ['ASN123', 'SER125', 'GLY98']
    """

    if value is None:
        return []

    value = value.strip()

    if not value:
        return []

    return [
        x.strip()
        for x in value.split("|")
        if x.strip()
    ]


def split_distances(value):

    """
    Split pipe-separated distances and convert them to floats.

    Invalid distances are stored as None.
    """

    if value is None:
        return []

    value = value.strip()

    if not value:
        return []

    distances = []

    for x in value.split("|"):

        x = x.strip()

        if not x:
            continue

        try:
            distances.append(float(x))

        except ValueError:
            distances.append(None)

    return distances


def get_cnp_id(ligand_name):

    """
    Extract CNP ID from ligand names such as:

        minimized_CNP0017240.4_docked_1
        minimized_CNP0072566.1_docked_9

    Returns:

        CNP0017240.4
        CNP0072566.1
    """

    match = re.search(
        r"(CNP\d+(?:\.\d+)?)_docked_\d+",
        ligand_name
    )

    if match:
        return match.group(1)

    # Fallback if naming convention differs
    match = re.search(
        r"(CNP\d+(?:\.\d+)?)",
        ligand_name
    )

    if match:
        return match.group(1)

    return ligand_name


def analyze_interactions(input_csv, bonds_of_interest):

    """
    Read the docking-interaction CSV.

    Each CNP ID produces exactly one output row.

    Only docking conformations 1-9 are considered.
    """

    # Structure:
    #
    # compounds[CNP_ID][conformation_number] = row
    #
    compounds = defaultdict(dict)

    with open(
        input_csv,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        required_columns = {
            "LIGAND_NAME",
            "Hydrogen_bonds",
            "Hydrogen_bond_distance",
            "Hydrophobic_bonds",
            "Hydrophobic_bond_distance",
        }

        missing = (
            required_columns -
            set(reader.fieldnames or [])
        )

        if missing:

            raise ValueError(
                "Missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:

            ligand_name = row["LIGAND_NAME"].strip()

            if not ligand_name:
                continue

            # Extract conformation number
            match = re.search(
                r"_docked_(\d+)",
                ligand_name
            )

            if not match:
                continue

            conformation = int(match.group(1))

            # Only consider conformations 1-9
            if not 1 <= conformation <= 9:
                continue

            cnp_id = get_cnp_id(ligand_name)

            compounds[cnp_id][conformation] = row

    results = []

    # =============================================================
    # Process each CNP ID
    # =============================================================

    for cnp_id, conformations in sorted(
        compounds.items()
    ):

        # ---------------------------------------------------------
        # Overall statistics
        # ---------------------------------------------------------

        n_conformations = len(conformations)

        n_conformations_with_boi = 0

        # All interactions
        total_hydrogen = 0
        total_hydrophobic = 0

        # Interactions of interest
        total_hydrogen_boi = 0
        total_hydrophobic_boi = 0

        # Store conformation-specific information
        conformation_data = {}

        # ---------------------------------------------------------
        # Process all 9 possible conformations
        # ---------------------------------------------------------

        for conf_num in range(1, 10):

            row = conformations.get(conf_num)

            # -----------------------------------------------------
            # Missing conformation
            # -----------------------------------------------------

            if row is None:

                conformation_data[conf_num] = {

                    "h_bonds": [],

                    "h_distances": [],

                    "h_count": 0,

                    "hydrophobic_bonds": [],

                    "hydrophobic_distances": [],

                    "hydrophobic_count": 0,

                    "has_boi": False,
                }

                continue

            # -----------------------------------------------------
            # Read all interactions
            # -----------------------------------------------------

            hydrogen_bonds = split_interactions(
                row["Hydrogen_bonds"]
            )

            hydrogen_distances = split_distances(
                row["Hydrogen_bond_distance"]
            )

            hydrophobic_bonds = split_interactions(
                row["Hydrophobic_bonds"]
            )

            hydrophobic_distances = split_distances(
                row["Hydrophobic_bond_distance"]
            )

            # -----------------------------------------------------
            # Overall interaction counts
            # -----------------------------------------------------

            total_hydrogen += len(hydrogen_bonds)

            total_hydrophobic += len(hydrophobic_bonds)

            # -----------------------------------------------------
            # Hydrogen bonds of interest
            # -----------------------------------------------------

            h_bonds_of_interest = []

            h_distances_of_interest = []

            for i, residue in enumerate(
                hydrogen_bonds
            ):

                if residue.upper() in bonds_of_interest:

                    h_bonds_of_interest.append(
                        residue
                    )

                    # Keep the corresponding distance
                    if i < len(hydrogen_distances):

                        distance = (
                            hydrogen_distances[i]
                        )

                        if distance is not None:

                            h_distances_of_interest.append(
                                distance
                            )

            # -----------------------------------------------------
            # Hydrophobic interactions of interest
            # -----------------------------------------------------

            hydrophobic_bonds_of_interest = []

            hydrophobic_distances_of_interest = []

            for i, residue in enumerate(
                hydrophobic_bonds
            ):

                if residue.upper() in bonds_of_interest:

                    hydrophobic_bonds_of_interest.append(
                        residue
                    )

                    # Keep the corresponding distance
                    if i < len(hydrophobic_distances):

                        distance = (
                            hydrophobic_distances[i]
                        )

                        if distance is not None:

                            hydrophobic_distances_of_interest.append(
                                distance
                            )

            # -----------------------------------------------------
            # Counts of interactions of interest
            # -----------------------------------------------------

            h_boi_count = len(
                h_bonds_of_interest
            )

            hydrophobic_boi_count = len(
                hydrophobic_bonds_of_interest
            )

            # Add to overall counts
            total_hydrogen_boi += h_boi_count

            total_hydrophobic_boi += (
                hydrophobic_boi_count
            )

            # -----------------------------------------------------
            # Determine whether this conformation contains
            # at least one bond of interest
            # -----------------------------------------------------

            has_boi = (
                h_boi_count > 0
                or
                hydrophobic_boi_count > 0
            )

            if has_boi:

                n_conformations_with_boi += 1

            # -----------------------------------------------------
            # Store conformation-specific data
            # -----------------------------------------------------

            conformation_data[conf_num] = {

                "h_bonds":
                    h_bonds_of_interest,

                "h_distances":
                    h_distances_of_interest,

                "h_count":
                    h_boi_count,

                "hydrophobic_bonds":
                    hydrophobic_bonds_of_interest,

                "hydrophobic_distances":
                    hydrophobic_distances_of_interest,

                "hydrophobic_count":
                    hydrophobic_boi_count,

                "has_boi":
                    has_boi,
            }

        # ---------------------------------------------------------
        # Total interactions
        # ---------------------------------------------------------

        total_interactions = (
            total_hydrogen +
            total_hydrophobic
        )

        # ---------------------------------------------------------
        # Create ONE output row for this CNP ID
        # ---------------------------------------------------------

        result = {

            "CNP_ID":
                cnp_id,

            "n_conformations":
                n_conformations,

            "n_conformations_with_bonds_of_interest":
                n_conformations_with_boi,

            "total_interactions":
                total_interactions,

            "total_hydrogen_bond_interactions":
                total_hydrogen,

            "total_hydrogen_bond_interactions_of_interest":
                total_hydrogen_boi,

            "total_hydrophobic_interactions":
                total_hydrophobic,

            "total_hydrophobic_interactions_of_interest":
                total_hydrophobic_boi,
        }

        # ---------------------------------------------------------
        # Add conformation-specific columns
        # ---------------------------------------------------------

        for conf_num in range(1, 10):

            data = conformation_data[conf_num]

            prefix = f"conf_{conf_num}"

            # H-bonds of interest
            result[
                f"{prefix}_h_bonds_of_interest"
            ] = " | ".join(
                data["h_bonds"]
            )

            # H-bond count of interest
            result[
                f"{prefix}_h_bond_count"
            ] = data["h_count"]

            # H-bond distances
            result[
                f"{prefix}_h_bond_distances"
            ] = " | ".join(
                f"{x:.2f}"
                for x in data["h_distances"]
            )

            # Hydrophobic bonds of interest
            result[
                f"{prefix}_hydrophobic_bonds_of_interest"
            ] = " | ".join(
                data["hydrophobic_bonds"]
            )

            # Hydrophobic count of interest
            result[
                f"{prefix}_hydrophobic_count"
            ] = data["hydrophobic_count"]

            # Hydrophobic distances
            result[
                f"{prefix}_hydrophobic_distances"
            ] = " | ".join(
                f"{x:.2f}"
                for x in data["hydrophobic_distances"]
            )

        results.append(result)

    return results


def write_output(results, output_csv):

    # =============================================================
    # Overall columns
    # =============================================================

    fieldnames = [

        "CNP_ID",

        "n_conformations",

        "n_conformations_with_bonds_of_interest",

        "total_interactions",

        "total_hydrogen_bond_interactions",

        "total_hydrogen_bond_interactions_of_interest",

        "total_hydrophobic_interactions",

        "total_hydrophobic_interactions_of_interest",
    ]

    # =============================================================
    # Conformation-specific columns
    # =============================================================

    for conf_num in range(1, 10):

        prefix = f"conf_{conf_num}"

        fieldnames.extend([

            f"{prefix}_h_bonds_of_interest",

            f"{prefix}_h_bond_count",

            f"{prefix}_h_bond_distances",

            f"{prefix}_hydrophobic_bonds_of_interest",

            f"{prefix}_hydrophobic_count",

            f"{prefix}_hydrophobic_distances",
        ])

    # =============================================================
    # Write CSV
    # =============================================================

    with open(
        output_csv,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(results)


def main():

    args = parse_args()

    # -------------------------------------------------------------
    # Load bonds/residues of interest
    # -------------------------------------------------------------

    bonds_of_interest = load_bonds_of_interest(
        args.bonds_of_interest
    )

    if not bonds_of_interest:

        raise ValueError(
            "No bonds/residues of interest were found in the "
            "provided text file."
        )

    # -------------------------------------------------------------
    # Analyze
    # -------------------------------------------------------------

    results = analyze_interactions(
        args.input,
        bonds_of_interest
    )

    # -------------------------------------------------------------
    # Write output
    # -------------------------------------------------------------

    write_output(
        results,
        args.output
    )

    print(
        f"Processed {len(results)} CNP IDs."
    )

    print(
        f"Output written to: {args.output}"
    )


if __name__ == "__main__":
    main()