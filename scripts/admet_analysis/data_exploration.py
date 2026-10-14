
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Generate grouped and six-category toxicity count tables."
    )
    parser.add_argument("-i", "--input", required=True, help="Input CSV")
    parser.add_argument(
        "--output-grouped",
        default="endpoint_safe_toxic.csv",
        help="Output CSV with Safe and Toxic counts"
    )
    parser.add_argument(
        "--output-six",
        default="endpoint_six_categories.csv",
        help="Output CSV with six individual category counts"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    exclude_cols = {
        "identifier",
        "SMILES",
        "Environmental_Fathead_Minnow_(Regression)",
        "Environmental_T._Pyriformis_(Regression)",
    }
    endpoints = [c for c in df.columns if c not in exclude_cols]

    categories = [
        "Low safety",
        "Medium safety",
        "High safety",
        "Low toxicity",
        "Medium toxicity",
        "High toxicity",
    ]

    # Normalize labels to handle case and surrounding whitespace
    normalized = df[endpoints].apply(
        lambda col: col.astype("string").str.strip().str.lower()
    )

    # Table 1: six individual categories
    six = pd.DataFrame(index=endpoints, columns=categories, dtype=int)

    for endpoint in endpoints:
        counts = normalized[endpoint].value_counts()
        for category in categories:
            six.loc[endpoint, category] = counts.get(category.lower(), 0)

    six.index.name = "Endpoint"
    six = six.astype(int)
    six.to_csv(args.output_six)

    # Table 2: grouped Safe and Toxic counts
    grouped = pd.DataFrame(index=endpoints)
    grouped["Safe"] = six[
        ["Low safety", "Medium safety", "High safety"]
    ].sum(axis=1)
    grouped["Toxic"] = six[
        ["Low toxicity", "Medium toxicity", "High toxicity"]
    ].sum(axis=1)

    grouped.index.name = "Endpoint"
    grouped.to_csv(args.output_grouped)

    print(f"Grouped table saved to: {args.output_grouped}")
    print(f"Six-category table saved to: {args.output_six}")


if __name__ == "__main__":
    main()
