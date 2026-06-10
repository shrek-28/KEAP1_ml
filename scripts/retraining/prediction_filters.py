import pandas as pd 

pred_df = pd.read_csv("data/new_data_pred/predictions.csv")

percentages = [0.1, 0.5, 1, 2, 5, 10]

for pct in percentages:

    n = int(len(pred_df) * pct / 100)

    top_df = pred_df.nsmallest(
        n,
        "predicted_docking_score"
    )

    top_df.to_csv(
        f"data/new_data_pred/top_scorers/top_{pct}_percent.csv",
        index=False
    )

    results = []

    for pct in [0.1, 0.5, 1, 2, 5, 10]:
        n = int(len(pred_df) * pct / 100)

        top_df = pred_df.nsmallest(n, "predicted_docking_score")

        results.append({
            "top_percent": pct,
            "num_molecules": len(top_df),
            "cutoff_score": top_df["predicted_docking_score"].max()
        })

    summary_df = pd.DataFrame(results)

summary_df.to_csv("data/new_data_pred/top_scorers_summary.csv", index=False)
