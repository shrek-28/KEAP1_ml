#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


SAFETY_COLUMNS = [
    "Genomic_AMES_Mutagenesis",
    "Organic_hERG_I_Inhibitor",
    "Organic_hERG_II_Inhibitor",
    "Genomic_Micronucleus",
    "Genomic_Carcinogenesis",
]


SAFETY_SCORE = {
    "high safety": 1.00,
    "medium safety": 0.67,
    "low safety": 0.33,
    "low toxicity": 0.00,
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="MMR-based diverse candidate selection with secondary safety prioritization."
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input candidate CSV."
    )

    parser.add_argument(
        "--output",
        required=True,
        help="Output lambda sensitivity CSV."
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.60,
        help="Maximum allowed Tanimoto similarity (default: 0.60)."
    )

    parser.add_argument(
        "--n-select",
        type=int,
        default=5,
        help="Number of molecules to select for each lambda."
    )

    return parser.parse_args()


def normalize(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return pd.Series(1.0, index=series.index)

    return (series - minimum) / (maximum - minimum)


def calculate_safety_score(df):

    scores = pd.DataFrame(index=df.index)

    for column in SAFETY_COLUMNS:

        scores[column] = (
            df[column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(SAFETY_SCORE)
        )

        if scores[column].isna().any():

            bad_values = (
                df.loc[
                    scores[column].isna(),
                    column
                ]
                .astype(str)
                .unique()
            )

            raise ValueError(
                f"Unexpected values in {column}: "
                f"{bad_values}"
            )

    return scores.mean(axis=1)


def calculate_fingerprints(smiles):

    fingerprints = []

    for smi in smiles:

        mol = Chem.MolFromSmiles(smi)

        if mol is None:
            raise ValueError(
                f"Invalid SMILES: {smi}"
            )

        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )

        fingerprints.append(fp)

    return fingerprints


def calculate_similarity_matrix(fingerprints):

    n = len(fingerprints)

    matrix = np.zeros((n, n))

    for i in range(n):

        matrix[i, :] = (
            DataStructs.BulkTanimotoSimilarity(
                fingerprints[i],
                fingerprints
            )
        )

    return matrix


def mmr_selection(
    df,
    similarity,
    lambda_value,
    similarity_threshold,
    n_select
):

    quality = df["quality_score"].to_numpy()

    selected = [
        int(np.argmax(quality))
    ]

    while len(selected) < n_select:

        candidates = []

        for i in range(len(df)):

            if i in selected:
                continue

            max_similarity = np.max(
                similarity[i, selected]
            )

            if max_similarity >= similarity_threshold:
                continue

            mmr = (
                lambda_value * quality[i]
                - (1 - lambda_value) * max_similarity
            )

            candidates.append(
                (
                    mmr,
                    i,
                    max_similarity
                )
            )

        if not candidates:
            break

        candidates.sort(
            key=lambda x: x[0],
            reverse=True
        )

        _, selected_idx, _ = candidates[0]

        selected.append(selected_idx)

    return selected


def main():

    args = parse_args()

    df = pd.read_csv(args.input)

    required_columns = [
        "CNP_ID",
        "n_conformations_with_bonds_of_interest",
        "total_hydrogen_bond_interactions_of_interest",
        "total_hydrophobic_interactions_of_interest",
        "binding_affinity",
        "SMILES",
    ] + SAFETY_COLUMNS

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns: "
            + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # 9/9 interaction criterion
    # ---------------------------------------------------------

    df = df[
        df["n_conformations_with_bonds_of_interest"] == 9
    ].copy()

    df.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------
    # Normalize interaction metrics
    # ---------------------------------------------------------

    binding_score = normalize(
        -df["binding_affinity"]
    )

    hbond_score = normalize(
        df[
            "total_hydrogen_bond_interactions_of_interest"
        ]
    )

    hydrophobic_score = normalize(
        df[
            "total_hydrophobic_interactions_of_interest"
        ]
    )

    # ---------------------------------------------------------
    # Safety
    # ---------------------------------------------------------

    df["safety_score"] = calculate_safety_score(df)

    # ---------------------------------------------------------
    # Overall quality
    #
    # Binding       35%
    # H-bonds       30%
    # Hydrophobic   25%
    # Safety        10%
    # ---------------------------------------------------------

    df["quality_score"] = (
        0.25 * binding_score
        + 0.25 * hbond_score
        + 0.25 * hydrophobic_score
        + 0.25 * df["safety_score"]
    )

    # ---------------------------------------------------------
    # Fingerprints
    # ---------------------------------------------------------

    fingerprints = calculate_fingerprints(
        df["SMILES"]
    )

    similarity = calculate_similarity_matrix(
        fingerprints
    )

    # ---------------------------------------------------------
    # Lambda sensitivity
    # ---------------------------------------------------------

    lambda_values = [
        0.5,
        0.6,
        0.7,
        0.8,
        0.9
    ]

    records = []

    for lam in lambda_values:

        selected = mmr_selection(
            df=df,
            similarity=similarity,
            lambda_value=lam,
            similarity_threshold=args.similarity_threshold,
            n_select=args.n_select
        )

        for order, idx in enumerate(
            selected,
            start=1
        ):

            row = df.iloc[idx]

            records.append({
                "lambda": lam,
                "selection_order": order,
                "CNP_ID": row["CNP_ID"],
                "binding_affinity":
                    row["binding_affinity"],
                "Hbond_interactions":
                    row[
                        "total_hydrogen_bond_interactions_of_interest"
                    ],
                "Hydrophobic_interactions":
                    row[
                        "total_hydrophobic_interactions_of_interest"
                    ],
                "safety_score":
                    row["safety_score"],
                "quality_score":
                    row["quality_score"],
            })

    results = pd.DataFrame(records)

    results.to_csv(
        args.output,
        index=False
    )

    # ---------------------------------------------------------
    # Frequency
    # ---------------------------------------------------------

    frequency = (
        results
        .groupby("CNP_ID")
        .size()
        .reset_index(
            name="times_selected"
        )
        .sort_values(
            "times_selected",
            ascending=False
        )
    )

    frequency_output = (
        args.output.rsplit(".", 1)[0]
        + "_frequency.csv"
    )

    frequency.to_csv(
        frequency_output,
        index=False
    )

    # ---------------------------------------------------------
    # Print
    # ---------------------------------------------------------

    print("\nLambda-wise selections:\n")

    for lam in lambda_values:

        subset = results[
            results["lambda"] == lam
        ]

        print(
            f"lambda = {lam}: "
            + ", ".join(
                subset["CNP_ID"].astype(str)
            )
        )

    print("\nSelection frequency:\n")

    print(
        frequency.to_string(
            index=False
        )
    )

    print(
        f"\nDetailed results: {args.output}"
    )

    print(
        f"Frequency results: {frequency_output}"
    )


if __name__ == "__main__":
    main()